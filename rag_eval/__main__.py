"""Run an offline QASPER retrieval comparison and write a structured artifact."""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import sys
import time
import tomllib

import numpy as np
import psutil
from tokenizers import Tokenizer

from .chunking import build_chunks
from .corpus import load_split, paragraphs, sha256
from .evaluation import map_evidence, score
from .retrieval import BGE, Retrieval, cached_vectors


@contextmanager
def stage(name, timings):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    yield
    timings[name] = {"wall_s": time.perf_counter() - wall_start,
                     "cpu_s": time.process_time() - cpu_start}
    print(json.dumps({"stage": name, **timings[name]}), file=sys.stderr, flush=True)


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run(args):
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    cutoffs = sorted(set(config["cutoffs"]))
    settings = config["settings"]
    limit = config["paper_limit"]
    if (config["schema_version"] != 1 or not cutoffs or cutoffs[0] < 1 or limit < 0
            or not settings or set(settings) - {"paper", "split-title", "split-question"}
            or len(settings) != len(set(settings))):
        raise ValueError("Invalid config version, cutoff, paper limit or evaluation setting")
    process = psutil.Process()
    available_start = psutil.virtual_memory().available
    args.output.mkdir(parents=True, exist_ok=True)
    model_path, tokenizer_path = args.model / "model.onnx", args.model / "tokenizer.json"
    timings = {}
    with stage("ingestion", timings):
        corpus, source = load_split(args.data, config["split"])
        paper_ids = sorted(corpus)
        if limit and limit < len(paper_ids):
            paper_ids = sorted(random.Random(config["seed"]).sample(paper_ids, limit))
        papers = {pid: corpus[pid] for pid in paper_ids}
        del corpus
    with stage("model_verify_and_tokenizer", timings):
        for path, field in [(model_path, "model_sha256"), (tokenizer_path, "tokenizer_sha256")]:
            if sha256(path) != config[field]:
                raise ValueError(f"Model asset checksum mismatch: {path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with stage("chunking", timings):
        chunks = build_chunks(papers, tokenizer, config["max_tokens"], config["overlap"])
        by_paper = {pid: [] for pid in paper_ids}
        for chunk in chunks:
            by_paper[chunk.paper_id].append(chunk)
    with stage("evidence_mapping", timings):
        queries = []
        for pid, paper in papers.items():
            source_paragraphs = list(paragraphs(pid, paper))
            for question in sorted(paper["qas"], key=lambda item: item["question_id"]):
                gold = map_evidence(question, source_paragraphs, by_paper[pid])
                queries.append((pid, question, gold))
    with stage("bm25_build", timings):
        sparse = Retrieval(chunks, method="bm25")
    with stage("dense_model_load", timings):
        encoder = BGE(model_path, tokenizer, config["batch_size"], config["threads"])
    identity = {
        "config": config,
        "source": source,
        "versions": {package: version(package) for package in
                     ("numpy", "onnxruntime", "tokenizers", "rank-bm25", "psutil")},
        "code": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    with stage("dense_embeddings_or_cache", timings):
        matrix, cache_path, cache_hit = cached_vectors(
            args.cache, chunks, json.dumps(identity, sort_keys=True), encoder)
        dense = Retrieval(chunks, method="dense", vectors=matrix, encoder=encoder)
    summaries = []
    rankings = []
    with stage("evaluation", timings):
        for engine in (sparse, dense):
            if queries:
                engine.search(queries[0][1]["question"], 1, queries[0][0])
            for setting in settings:
                rows = []
                for pid, question, gold in queries:
                    text = question["question"]
                    if setting == "split-title":
                        text = f"{papers[pid]['title']}\n{text}"
                    started = time.perf_counter_ns()
                    hits = engine.search(text, cutoffs[-1], pid if setting == "paper" else None)
                    latency_ms = (time.perf_counter_ns() - started) / 1e6
                    values = score([chunk for chunk, _ in hits], gold, cutoffs)
                    if setting != "paper":
                        values.update({f"target_paper_hit@{k}": float(
                            any(chunk.paper_id == pid for chunk, _ in hits[:k])) for k in cutoffs})
                    row = {
                        "method": engine.method, "setting": setting, "question_id": question["question_id"],
                        "target_paper_id": pid, "status": gold.status, "latency_ms": latency_ms,
                        "metrics": values,
                        "hits": [{"rank": rank, "chunk_id": chunk.id, "paper_id": chunk.paper_id,
                                  "score": value} for rank, (chunk, value) in enumerate(hits, 1)],
                    }
                    rows.append(row)
                rankings.extend(rows)
                metric_values = {}
                for row in rows:
                    for key, value in row["metrics"].items():
                        metric_values.setdefault(key, []).append(value)
                latencies = [row["latency_ms"] for row in rows]
                summaries.append({
                    "method": engine.method, "setting": setting, "queries": len(rows),
                    "metrics": {key: {"mean": float(np.mean(values)), "n": len(values)}
                                for key, values in metric_values.items()},
                    "latency_ms": dict(zip(("p50", "p95"), map(float, np.percentile(latencies, [50, 95]))))
                    if latencies else {},
                })
    with stage("save_artifacts", timings):
        chunk_file = args.output / "chunks.json"
        mapping_file = args.output / "evidence.json"
        ranking_file = args.output / "rankings.json"
        write_json(chunk_file, [{key: value for key, value in asdict(chunk).items() if key != "text"}
                                for chunk in chunks])
        write_json(mapping_file, [
            {"paper_id": pid, "question_id": question["question_id"], "status": gold.status,
             "gold_sets": [sorted(group) for group in gold.sets], "annotations": gold.annotations}
            for pid, question, gold in queries])
        write_json(ranking_file, rankings)
    memory = process.memory_info()
    report = {
        **identity, "paper_ids": paper_ids, "question_count": len(queries), "chunks": len(chunks),
        "eligibility": dict(Counter(gold.status for _, _, gold in queries)),
        "evidence_labels": dict(Counter(label["status"] for _, _, gold in queries
                                        for annotation in gold.annotations for label in annotation["labels"])),
        "timings": timings, "summaries": summaries,
        "cache": {"hit": cache_hit, "file": cache_path.name, "sha256": sha256(cache_path),
                  "bytes": cache_path.stat().st_size},
        "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for path in (chunk_file, mapping_file, ranking_file)},
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
                     "ram_bytes": psutil.virtual_memory().total,
                     "available_start_bytes": available_start,
                     "available_end_bytes": psutil.virtual_memory().available,
                     "end_rss_bytes": memory.rss, "peak_working_set_bytes": getattr(memory, "peak_wset", None),
                     "onnx_provider": encoder.session.get_providers()},
    }
    write_json(args.output / "report.json", report)
    print(args.output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
