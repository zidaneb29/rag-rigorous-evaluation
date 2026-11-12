"""Read-only boundaries between Streamlit and prepared QASPER artifacts."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import re
from threading import Lock
from time import perf_counter
import tomllib

import numpy as np
from tokenizers import Tokenizer

from .chunking import build_chunks
from .corpus import load_split, paragraphs, sha256
from .fusion import reciprocal_rank_fusion, rerank
from .phase3 import load_json, load_reranker
from .phase5 import digest
from .retrieval import BGE, Retrieval


METHODS = {"bm25": "BM25", "dense": "BGE-small", "hybrid": "RRF", "reranked": "RRF + MiniLM"}
SCORES = {"bm25": "BM25 score", "dense": "Cosine similarity",
          "hybrid": "RRF score", "reranked": "MiniLM logit"}
SCOPES = {"paper": "Paper scoped", "split-title": "Sampled pool with title cue"}
PREPARATION = "See docs/phase6.md for local assets and the offline preparation command."


def read_toml(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(path):
    config = read_toml(path)
    if (config["schema_version"] != 1 or config["default_method"] not in METHODS
            or not 1 <= config["default_top_k"] <= config["max_top_k"] <= 20):
        raise ValueError("Unsupported app configuration")
    return config


def stamps(paths):
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            result.append((str(path), None, None))
    return tuple(result)


def verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. {PREPARATION}")
    if sha256(path) != expected:
        raise ValueError(f"Checksum mismatch for {path.name}. Restore the verified local artifact.")


def checked_json(path, expected):
    verify(path, expected)
    return load_json(path)


def source_url(paper_id):
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", paper_id):
        return "https://arxiv.org/abs/" + paper_id
    return None


class DevelopmentCatalog:
    def __init__(self, data_dir, report_dir):
        self.report = load_json(report_dir / "report.json")
        if self.report["config"]["split"] != "dev" or self.report["source"]["split"] != "dev":
            raise ValueError("The explorer requires a development-only preparation report")
        for name in ("corpus.py", "chunking.py", "retrieval.py"):
            if sha256(Path(__file__).with_name(name)) != self.report["code"][name]:
                raise ValueError(f"The prepared development index used a different {name}. {PREPARATION}")
        corpus, provenance = load_split(data_dir, "dev")
        if provenance != self.report["source"]:
            raise ValueError("Development archive differs from the prepared index")
        self.papers = {pid: {k: v for k, v in corpus[pid].items() if k != "qas"}
                       for pid in self.report["paper_ids"]}
        self.manifest = checked_json(report_dir / "chunks.json", self.report["files"]["chunks.json"]["sha256"])


class Explorer:
    """Own an index and at most one resident ONNX session per resource cache entry."""

    def __init__(self, catalog, model_dir, vectors_dir, retrieval_config):
        self.catalog = catalog
        self.model_dir = model_dir
        self.vectors_dir = vectors_dir
        self.retrieval_config = retrieval_config
        self.pilot = catalog.report["config"]
        if not 1 <= retrieval_config["rerank_depth"] <= retrieval_config["primary_fusion_depth"]:
            raise ValueError("Invalid fusion or reranking depth")
        tokenizer = model_dir / "tokenizer.json"
        verify(tokenizer, self.pilot["tokenizer_sha256"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.chunks = build_chunks(catalog.papers, self.tokenizer, self.pilot["max_tokens"], self.pilot["overlap"])
        rebuilt = [{k: v for k, v in asdict(c).items() if k != "text"} for c in self.chunks]
        if rebuilt != catalog.manifest:
            raise ValueError("Development chunks do not match the prepared index")
        self.chunk_lookup = {chunk.id: chunk for chunk in self.chunks}
        self.paragraphs = {p.id: p for pid, paper in catalog.papers.items() for p in paragraphs(pid, paper)}
        self.sparse = None
        self.vectors = None
        self.model = None
        self.model_kind = None
        self.lock = Lock()

    def _vectors(self):
        if self.vectors is None:
            meta = self.catalog.report["cache"]
            path = self.vectors_dir / meta["file"]
            verify(path, meta["sha256"])
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            if (matrix.shape != (len(self.chunks), 384) or matrix.dtype != np.float32
                    or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)):
                raise ValueError("Development vector cache has invalid shape or normalization")
            self.vectors = matrix
        return self.vectors

    def _model(self, kind):
        if self.model_kind != kind:
            self.model, self.model_kind = None, None
            if kind == "dense":
                path = self.model_dir / "model.onnx"
                verify(path, self.pilot["model_sha256"])
                self.model = BGE(path, self.tokenizer, self.pilot["batch_size"], self.pilot["threads"])
            else:
                self.model = load_reranker(self.model_dir, self.retrieval_config, self.pilot)
            self.model_kind = kind
        return self.model

    def search(self, question, paper_id, method, k):
        question = question.strip()
        if not question or len(question) > 1000:
            raise ValueError("Enter a question between 1 and 1,000 characters.")
        if paper_id not in self.catalog.papers or method not in METHODS or not 1 <= k <= 20:
            raise ValueError("Invalid paper, method, or Top-k selection")
        with self.lock:
            return self._search(question, paper_id, method, k)

    def _search(self, question, paper_id, method, k):
        search_ms = 0.0
        wall = perf_counter()

        def query(function, *args):
            nonlocal search_ms
            started = perf_counter()
            value = function(*args)
            search_ms += (perf_counter() - started) * 1000
            return value

        fused_method = method in {"hybrid", "reranked"}
        depth = self.retrieval_config["primary_fusion_depth"] if fused_method else k
        sparse, dense = [], []
        if method != "dense":
            if self.sparse is None:
                self.sparse = Retrieval(self.chunks, method="bm25")
            sparse = [(chunk.id, value) for chunk, value in query(self.sparse.search, question, depth, paper_id)]
        if method != "bm25":
            vectors = self._vectors()
            engine = Retrieval(self.chunks, method="dense", vectors=vectors, encoder=self._model("dense"))
            dense = [(chunk.id, value) for chunk, value in query(engine.search, question, depth, paper_id)]
            del engine
        scores = dense if method == "dense" else sparse
        candidate_count = 0
        if fused_method:
            scores = query(reciprocal_rank_fusion, [sparse, dense], self.retrieval_config["rrf_constant"], depth)
        if method == "reranked":
            model = self._model("reranked")
            candidates, scores = query(rerank, question, scores, self.chunk_lookup, model,
                                       self.retrieval_config["rerank_depth"])
            candidate_count = len(candidates)
        passages = []
        for rank, (chunk_id, value) in enumerate(scores[:k], 1):
            chunk = self.chunk_lookup[chunk_id]
            source = self.paragraphs[chunk.paragraph_id]
            if source.paper_id != paper_id:
                raise ValueError("Search returned a passage outside the selected paper")
            passages.append({
                "rank": rank, "score": value,
                "score_label": "RRF tail score" if method == "reranked" and rank > candidate_count else SCORES[method],
                "paper_id": paper_id, "title": self.catalog.papers[paper_id]["title"],
                "section": source.section or "Untitled section", "paragraph_id": source.id,
                "chunk_id": chunk_id, "text": source.text[chunk.start:chunk.end],
                "start": chunk.start, "end": chunk.end, "source": source_url(paper_id),
            })
        return {"passages": passages, "method": method, "paper_id": paper_id, "top_k": k,
                "query_ms": search_ms, "request_ms": (perf_counter() - wall) * 1000}


DASHBOARD_FILES = ("report.json", "protocol.json", "population.json", "rankings.json", "evidence.json", "chunks.json")


def load_dashboard(directory, report_sha256):
    report = checked_json(directory / "report.json", report_sha256)
    assets = {name: checked_json(directory / name, report["files"][name]["sha256"])
              for name in DASHBOARD_FILES[1:]}
    protocol, population = assets["protocol.json"], assets["population.json"]
    for entry, key in ((protocol, "protocol_sha256"), (population, "population_sha256")):
        if entry["sha256"] != report[key] or digest(entry["manifest"]) != entry["sha256"]:
            raise ValueError("Evaluation manifest differs from the frozen report")
    evidence = {r["question_id"]: r for r in assets["evidence.json"]}
    counts = Counter(r["status"] for r in evidence.values())
    if counts != report["eligibility"]:
        raise ValueError("Evaluation eligibility counts do not reconcile")
    ranks = {(r["setting"], r["question_id"]): r for r in assets["rankings.json"]}
    population_manifest = population["manifest"]
    selected = set(population_manifest["selected_ids"])
    for scope, result in report["results"].items():
        rows = [row for (key, _), row in ranks.items() if key == scope]
        if ({row["question_id"] for row in rows} != evidence.keys()
                or any(row["status"] != evidence[row["question_id"]]["status"] or
                       row["paper_id"] != evidence[row["question_id"]]["paper_id"] for row in rows)
                or {row["paper_id"] for row in rows} != selected
                or result["questions"] != len(rows) or result["papers"] != len(selected)
                or result["eligible"] != counts["eligible"]):
            raise ValueError("Evaluation questions, evidence or paper scope do not reconcile")
    return {"report": report, "protocol": protocol["manifest"], "population": population_manifest,
            "evidence": evidence, "rankings": ranks,
            "chunks": {c["id"]: c for c in assets["chunks.json"]}}


def dashboard_snapshot(data, source_sha256):
    """Project frozen dashboard values onto a small, text-free release artifact."""
    report = data["report"]
    examples = sorted({item["question_id"] for item in report["failure_sample"].values()
                       if item["question_id"] is not None})
    rankings, evidence, chunks = [], [], {}
    for qid in examples:
        row = data["rankings"]["paper", qid]
        first_twenty = {m: row["rankings"][m][:20] for m in METHODS}
        rankings.append({"setting": "paper", "question_id": qid,
                         "rankings": first_twenty, "metrics": row["metrics"]})
        evidence.append({"question_id": qid, "sets": data["evidence"][qid]["sets"]})
        for ranks in first_twenty.values():
            for cid, _ in ranks:
                chunks[cid] = {"paragraph_id": data["chunks"][cid]["paragraph_id"]}
    protocol = data["protocol"]
    return {
        "schema_version": 1, "source_report_sha256": source_sha256,
        "report": {key: report[key] for key in (
            "results", "eligibility", "failure_sample", "protocol_sha256", "population_sha256",
            "peak_working_set", "timings", "vector_cache_bytes")},
        "protocol": {key: protocol[key] for key in ("rules", "phase3")},
        "rankings": rankings, "evidence": evidence, "chunks": chunks,
    }


def load_snapshot(path, expected):
    saved = checked_json(path, expected)
    if saved["schema_version"] != 1:
        raise ValueError("Unsupported evaluation snapshot version")
    return saved | {"rankings": {(r["setting"], r["question_id"]): r for r in saved["rankings"]},
                    "evidence": {r["question_id"]: r for r in saved["evidence"]}}
