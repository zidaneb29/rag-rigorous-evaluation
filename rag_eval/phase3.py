"""Run the resumable, development-only Phase 3 experiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sqlite3
import time
import tomllib

import numpy as np
import psutil
from tokenizers import Tokenizer

from .__main__ import stage, write_json
from .chunking import build_chunks
from .corpus import load_split, normalize, paragraphs, sha256
from .evaluation import score
from .evidence_audit import audited_evidence
from .fusion import MiniLM, reciprocal_rank_fusion, rerank
from .phase3_analysis import summarize
from .retrieval import BGE, Retrieval, cached_vectors


SETTINGS = ("paper", "split-question", "split-title")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config, pilot):
    depths = config["fusion_depths"]
    if (config["schema_version"] != 1 or pilot["split"] != "dev"
            or not 1 <= config["expansion_papers"] <= 30
            or not depths or depths != sorted(set(depths))
            or min(depths) < max(pilot["cutoffs"])
            or config["primary_fusion_depth"] not in depths
            or config["rrf_constant"] <= 0
            or not 0 <= config["rerank_depth"] <= config["primary_fusion_depth"]
            or not 1 <= config["rerank_batch"] <= 16
            or not 16 <= config["rerank_max_tokens"] <= 512
            or config["bootstrap_replicates"] < 1):
        raise ValueError("Invalid Phase 3 configuration")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Checkpoints:
    def __init__(self, path, fingerprint):
        self.db = sqlite3.connect(path)
        self.fingerprint = fingerprint
        self.db.execute("""CREATE TABLE IF NOT EXISTS work (
            fingerprint TEXT, stage TEXT, setting TEXT, question_id TEXT,
            payload TEXT NOT NULL, sha256 TEXT NOT NULL,
            PRIMARY KEY (fingerprint, stage, setting, question_id))""")

    def get(self, stage_name, setting, question_id):
        row = self.db.execute("SELECT payload, sha256 FROM work WHERE fingerprint=? AND "
                              "stage=? AND setting=? AND question_id=?",
                              (self.fingerprint, stage_name, setting, question_id)).fetchone()
        if row is None:
            return None
        if digest(row[0]) != row[1]:
            raise ValueError(f"Corrupt checkpoint: {stage_name}/{setting}/{question_id}")
        return json.loads(row[0])

    def put(self, stage_name, setting, question_id, payload):
        text = json.dumps(payload, allow_nan=False)
        with self.db:
            self.db.execute("INSERT INTO work VALUES (?, ?, ?, ?, ?, ?)",
                            (self.fingerprint, stage_name, setting, question_id, text, digest(text)))


def timed(function, *args):
    start = time.perf_counter_ns()
    result = function(*args)
    return result, (time.perf_counter_ns() - start) / 1e6


def phase2_vectors(args, base, chunks, pilot):
    metadata = base["files"]["chunks.json"]
    manifest = args.phase2 / "chunks.json"
    if sha256(manifest) != metadata["sha256"]:
        raise ValueError("Phase 2 manifest checksum mismatch")
    prior_rows = load_json(manifest)
    current_rows = [{key: value for key, value in asdict(c).items() if key != "text"}
                    for c in chunks if c.paper_id in base["paper_ids"]]
    if current_rows != prior_rows or pilot != base["config"]:
        raise ValueError("Phase 2 chunk order or configuration differs")
    for module in ("corpus.py", "chunking.py", "retrieval.py"):
        if sha256(Path(__file__).with_name(module)) != base["code"][module]:
            raise ValueError(f"Phase 2 input implementation differs: {module}")
    cache = base["cache"]
    path = args.phase2_cache / cache["file"]
    if sha256(path) != cache["sha256"]:
        raise ValueError("Phase 2 matrix checksum mismatch")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if (matrix.shape != (len(prior_rows), 384) or matrix.dtype != np.float32
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid Phase 2 vectors")
    return {row["id"]: vector for row, vector in zip(prior_rows, matrix)}


def run(args):
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    pilot = tomllib.loads(args.pilot.read_text(encoding="utf-8"))
    validate_config(config, pilot)
    previous_report = args.output / "report.json"
    if (args.output.resolve() == args.phase2.resolve()
            or (previous_report.exists() and "fingerprint" not in load_json(previous_report))):
        raise ValueError("Phase 3 needs its own output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    available_start = psutil.virtual_memory().available
    timings = {}
    with stage("ingestion_and_chunking", timings):
        corpus, source = load_split(args.data, "dev")
        base = load_json(args.phase2 / "report.json")
        old_ids = set(base["paper_ids"])
        if source != base["source"] or len(old_ids) != pilot["paper_limit"] or not old_ids <= corpus.keys():
            raise ValueError("Phase 2 provenance does not match development release")
        extra = random.Random(pilot["seed"]).sample(sorted(corpus.keys() - old_ids), config["expansion_papers"])
        papers = {pid: corpus[pid] for pid in sorted(old_ids | set(extra))}
        del corpus
        for path, expected in [(args.model / "model.onnx", pilot["model_sha256"]),
                               (args.model / "tokenizer.json", pilot["tokenizer_sha256"])]:
            if sha256(path) != expected:
                raise ValueError(f"Asset checksum mismatch: {path}")
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        chunks = build_chunks(papers, tokenizer, pilot["max_tokens"], pilot["overlap"])
        by_id = {c.id: c for c in chunks}
        by_paper = defaultdict(list)
        for chunk in chunks:
            by_paper[chunk.paper_id].append(chunk)
    identity = {
        "config": config, "pilot": pilot, "source": source, "paper_ids": list(papers),
        "phase2_matrix_sha256": base["cache"]["sha256"],
        "versions": {name: version(name) for name in ("numpy", "rank-bm25", "tokenizers", "onnxruntime", "psutil")},
        "code": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    fingerprint = digest(json.dumps(identity, sort_keys=True))
    checkpoints = Checkpoints(args.output / "checkpoints.sqlite", fingerprint)
    queries = []
    with stage("evidence_audit", timings):
        for pid, paper in papers.items():
            source_paragraphs = list(paragraphs(pid, paper))
            for q in sorted(paper["qas"], key=lambda q: q["question_id"]):
                prior, gold = audited_evidence(pid, paper, q, source_paragraphs, by_paper[pid])
                queries.append((pid, q, prior, gold))
    counts = Counter()
    base_rows = {}
    with stage("base_ranking", timings):
        for pid, q, _, _ in queries:
            for setting in SETTINGS:
                key = (setting, q["question_id"])
                stored = checkpoints.get("base", *key)
                if stored is not None:
                    base_rows[key] = stored
                    counts["base_query_hits"] += 1
        pending = len(base_rows) != len(queries) * len(SETTINGS)
        if pending:
            with stage("base_model_load", timings):
                encoder = BGE(args.model / "model.onnx", tokenizer, pilot["batch_size"], pilot["threads"])
            with stage("base_index_build", timings):
                vectors = phase2_vectors(args, base, chunks, pilot)
                counts["reused_phase2_vectors"] = len(vectors)
                cache_signature = json.dumps({key: pilot[key] for key in
                                              ("model_sha256", "tokenizer_sha256", "batch_size", "threads")},
                                             sort_keys=True) + version("onnxruntime")
                for pid in sorted(extra):
                    group = by_paper[pid]
                    if not group:
                        continue
                    rows, _, hit = cached_vectors(args.output / "embeddings", group, cache_signature, encoder)
                    vectors.update({c.id: row for c, row in zip(group, rows)})
                    counts["cached_extra_vectors" if hit else "computed_extra_vectors"] += len(rows)
                matrix = np.stack([vectors[c.id] for c in chunks])
                dense = Retrieval(chunks, method="dense", vectors=matrix, encoder=encoder)
                sparse = Retrieval(chunks, method="bm25")
                sparse.search(queries[0][1]["question"], 1, queries[0][0])
                dense.search(queries[0][1]["question"], 1, queries[0][0])
            for pid, q, _, _ in queries:
                for setting in SETTINGS:
                    key = (setting, q["question_id"])
                    if key in base_rows:
                        continue
                    text = query_text(q, papers[pid], setting)
                    scope = pid if setting == "paper" else None
                    row = {}
                    for engine in (sparse, dense):
                        hits, elapsed = timed(engine.search, text, max(config["fusion_depths"]), scope)
                        row[engine.method] = [[chunk.id, value] for chunk, value in hits]
                        row[f"{engine.method}_ms"] = elapsed
                    checkpoints.put("base", *key, row)
                    base_rows[key] = row
            del encoder, dense, sparse, vectors, matrix, engine
    rank_rows = []
    before = process.memory_info()
    rss_loaded = None
    cross_encoder = None
    with stage("fusion_reranking_and_scoring", timings):
        for pid, q, prior, gold in queries:
            for setting in SETTINGS:
                key = (setting, q["question_id"])
                base_row = base_rows[key]
                fused, fusion_ms = timed(reciprocal_rank_fusion,
                                         [base_row["bm25"], base_row["dense"]],
                                         config["rrf_constant"], config["primary_fusion_depth"])
                ranks = {"bm25": base_row["bm25"], "dense": base_row["dense"], "hybrid": fused}
                timing = {name: base_row[f"{name}_ms"] for name in ("bm25", "dense")}
                timing["hybrid"] = timing["bm25"] + timing["dense"] + fusion_ms
                candidates = []
                if config["rerank_depth"]:
                    stored = checkpoints.get("rerank", *key)
                    if stored is None:
                        if cross_encoder is None:
                            with stage("reranker_load", timings):
                                cross_encoder = load_reranker(args.model, config, pilot)
                                rss_loaded = process.memory_info().rss
                                cross_encoder.score("Warmup", ["Warmup passage"])
                        (candidates, ranked), rerank_ms = timed(
                            rerank, query_text(q, papers[pid], setting), fused, by_id,
                            cross_encoder, config["rerank_depth"])
                        stored = {"candidates": candidates, "ranking": ranked, "ms": rerank_ms}
                        checkpoints.put("rerank", *key, stored)
                    else:
                        counts["rerank_query_hits"] += 1
                    ranks["reranked"] = stored["ranking"]
                    candidates = stored["candidates"]
                    timing["rerank_only"] = stored["ms"]
                    timing["reranked"] = timing["hybrid"] + stored["ms"]
                metrics = {}
                for method, ranked in ranks.items():
                    metrics[method] = score([by_id[cid] for cid, _ in ranked], gold, pilot["cutoffs"])
                    if setting != "paper":
                        metrics[method].update({
                            f"target_hit@{k}": float(any(by_id[cid].paper_id == pid for cid, _ in ranked[:k]))
                            for k in pilot["cutoffs"]})
                sensitivity = {
                    str(depth): score([by_id[cid] for cid, _ in reciprocal_rank_fusion(
                        [base_row["bm25"], base_row["dense"]], config["rrf_constant"], depth)],
                        gold, pilot["cutoffs"])
                    for depth in config["fusion_depths"] if depth != config["primary_fusion_depth"]
                } if setting == "paper" else {}
                rank_rows.append({
                    "paper_id": pid, "question_id": q["question_id"], "setting": setting,
                    "eligibility": gold.status, "rankings": ranks, "rerank_candidates": candidates,
                    "timing_ms": timing, "metrics": metrics, "fusion_sensitivity": sensitivity,
                    "paragraph_only_metrics": {m: score([by_id[cid] for cid, _ in rank], prior, pilot["cutoffs"])
                                               for m, rank in ranks.items()},
                })
    audits = [{"paper_id": pid, "question_id": q["question_id"], "pilot_paper": pid in old_ids,
               "phase2_status": prior.status, "status": gold.status,
               "sets": [sorted(s) for s in gold.sets], "annotations": gold.annotations,
               "explicit_title_in_question": normalize(papers[pid]["title"]).casefold()
               in normalize(q["question"]).casefold()}
              for pid, q, prior, gold in queries]
    memory = process.memory_info()
    checkpoints.db.close()
    report = {
        **identity, "fingerprint": fingerprint, "chunks": len(chunks), "questions": len(queries),
        "counts": counts, "timings": timings, "memory": {
            "available_start": available_start, "available_end": psutil.virtual_memory().available,
            "peak_working_set": getattr(memory, "peak_wset", None), "rss_before_reranker": before.rss,
            "peak_before_reranker": getattr(before, "peak_wset", None),
            "rss_after_reranker_load": rss_loaded, "rss_after_evaluation": memory.rss},
        **summarize(rank_rows, audits, config),
    }
    for name, rows in [("rankings.json", rank_rows), ("evidence-audit.json", audits),
                       ("chunks.json", [{k: v for k, v in asdict(c).items() if k != "text"} for c in chunks])]:
        write_json(args.output / name, rows)
    report["files"] = {name: {"sha256": sha256(args.output / name),
                              "bytes": (args.output / name).stat().st_size}
                       for name in ("rankings.json", "evidence-audit.json", "chunks.json", "checkpoints.sqlite")}
    report["extra_cache_bytes"] = sum(p.stat().st_size for p in (args.output / "embeddings").glob("*"))
    report["reranker_bytes"] = sum((args.model / "minilm" / file).stat().st_size
                                   for file in ("model.onnx", "tokenizer.json")) if config["rerank_depth"] else 0
    write_json(args.output / "report.json", report)


def query_text(question, paper, setting):
    text = question["question"]
    return f"{paper['title']}\n{text}" if setting == "split-title" else text


def load_reranker(directory, config, pilot):
    model, tokenizer = directory / "minilm/model.onnx", directory / "minilm/tokenizer.json"
    for path, key in [(model, "reranker_sha256"), (tokenizer, "reranker_tokenizer_sha256")]:
        if sha256(path) != config[key]:
            raise ValueError(f"Asset checksum mismatch: {path}")
    return MiniLM(model, Tokenizer.from_file(str(tokenizer)), pilot["threads"],
                  config["rerank_batch"], config["rerank_max_tokens"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.toml"))
    parser.add_argument("--pilot", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument("--phase2", type=Path, default=Path("artifacts/pilot-cold"))
    parser.add_argument("--phase2-cache", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
