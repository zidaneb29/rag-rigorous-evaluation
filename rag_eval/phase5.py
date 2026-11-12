"""Freeze, run, and replay the held-out retrieval protocol."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time
import tomllib

import numpy as np
import psutil
from tokenizers import Tokenizer

from .__main__ import stage, write_json
from .chunking import build_chunks
from .corpus import ARCHIVES, load_split, normalize, paragraphs, sha256
from .evaluation import score
from .evidence_audit import audited_evidence
from .fusion import reciprocal_rank_fusion, rerank
from .phase3 import Checkpoints, load_json, load_reranker, timed
from .phase3_analysis import means, paper_bootstrap
from .retrieval import BGE, Retrieval, cached_vectors, terms


METHODS = ("bm25", "dense", "hybrid", "reranked")
METRICS = ("recall@1", "recall@5", "recall@10", "mrr@20", "ndcg@5")
PAIRS = (("dense", "bm25"), ("hybrid", "bm25"), ("hybrid", "dense"),
         ("reranked", "bm25"), ("reranked", "dense"), ("reranked", "hybrid"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return load_json(path)


def publish(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        write_json(path, value)


def current_identity():
    return {
        "code": {name: sha256(Path(__file__).with_name(name)) for name in
                 ("phase5.py", "phase3.py", "phase3_analysis.py", "chunking.py", "corpus.py",
                  "retrieval.py", "fusion.py", "evaluation.py", "evidence_audit.py")},
        "versions": {name: version(name) for name in ("numpy", "onnxruntime", "tokenizers", "rank-bm25", "psutil")},
        "python": platform.python_version(),
    }


def development_robustness(rows, audits, report):
    indexed = {a["question_id"]: a for a in audits}
    panels = defaultdict(list)
    for row in rows:
        if row["setting"] != "paper" or row["eligibility"] != "eligible":
            continue
        audit = indexed[row["question_id"]]
        heading = any(label["status"] == "mapped_section" for ann in audit["annotations"]
                      for label in ann["labels"])
        panels["heading" if heading else "other"].append(row)
    return {
        "title_cue": report["title_effect"],
        "heading_panels": {group: {
            "questions": len(members),
            "metrics": {method: means(r["metrics"][method] for r in members) for method in METHODS},
            "paragraph_only": {method: means(r["paragraph_only_metrics"][method] for r in members)
                               for method in METHODS},
        } for group, members in panels.items()},
    }


def freeze(args):
    rules = config(args.config)
    pilot, phase3 = config(Path("configs/pilot.toml")), config(Path("configs/phase3.toml"))
    dev = load_json(args.dev / "report.json")
    cold = load_json(Path("artifacts/phase3-dev/report.json"))
    ranks, audits = [checked_json(args.dev / name, dev["files"][name]["sha256"])
                     for name in ("rankings.json", "evidence-audit.json")]
    robustness = development_robustness(ranks, audits, dev)
    manifest = {
        "rules": rules, "pilot": pilot, "phase3": phase3,
        "archives": {split: {"file": filename, "sha256": checksum}
                     for split, (filename, checksum) in ARCHIVES.items()},
        "development_report_sha256": sha256(args.dev / "report.json"),
        "development_robustness_sha256": digest(robustness),
        "resource_rates": {
            "chunks_per_second": cold["counts"]["computed_extra_vectors"] /
            cold["timings"]["base_index_build"]["wall_s"],
            "rerank_and_search_seconds_per_question_per_scope":
                sum(dev["latency_ms"]["paper"]["reranked"][key] for key in ("p50", "p95")) / 2000,
        },
        **current_identity(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    publish(args.output / "development-robustness.json", robustness)
    publish(args.output / "protocol.json", {"manifest": manifest, "sha256": digest(manifest)})
    print(digest(manifest))


def signatures(paper):
    title = tuple(terms(paper["title"]))
    body = [normalize(p).casefold() for section in paper["full_text"] for p in section["paragraphs"]]
    long = {digest(p) for p in body if len(p) >= 200}
    return title, set(title), digest(body) if any(body) else None, long


def duplicate_audit(split_papers):
    # Compare only document text and IDs. Never use test questions, evidence, or answers.
    previous = []
    findings = []
    for split, papers in split_papers:
        indexed = [(pid, signatures(paper)) for pid, paper in sorted(papers.items())]
        for pid, (title, words, body, long) in indexed:
            for prior_split, prior_id, (t, w, b, other) in previous:
                reasons = []
                if pid == prior_id:
                    reasons.append("id")
                if title and title == t:
                    reasons.append("title")
                if body is not None and body == b:
                    reasons.append("body")
                if min(len(words), len(w)) >= 5 and len(words & w) / len(words | w) >= .85:
                    reasons.append("near_title")
                if min(len(long), len(other)) >= 3 and len(long & other) / min(len(long), len(other)) >= .8:
                    reasons.append("body_overlap")
                if reasons:
                    findings.append({"splits": [prior_split, split], "papers": [prior_id, pid], "reasons": reasons})
        previous.extend((split, pid, sign) for pid, sign in indexed)
    return findings


def choose_papers(papers, audit, count, seed):
    excluded = {item["papers"][1] for item in audit if item["splits"][1] == "test"}
    ordered = sorted(set(papers) - excluded)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:count])


def project(papers, chunks_per_paper, rates, settings):
    chunks = len(papers) * chunks_per_paper
    questions = sum(len(p["qas"]) for p in papers.values())
    return {"papers": len(papers), "questions": questions, "estimated_chunks": chunks,
            "estimated_seconds": chunks / rates["chunks_per_second"] + questions * len(settings) *
            rates["rerank_and_search_seconds_per_question_per_scope"]}


def audit_questions(papers, by_paper):
    rows = []
    for pid, paper in papers.items():
        source = list(paragraphs(pid, paper))
        for q in sorted(paper["qas"], key=lambda q: q["question_id"]):
            prior, gold = audited_evidence(pid, paper, q, source, by_paper[pid])
            rows.append((pid, q, prior, gold))
    return rows


def prepare(args, manifest, tokenizer):
    train, _ = load_split(args.data, "train")
    dev, _ = load_split(args.data, "dev")
    test, provenance = load_split(args.data, "test")
    rules = manifest["rules"]
    findings = duplicate_audit([("train", train), ("dev", dev), ("test", test)])
    eligible_ids = choose_papers(test, findings, len(test), rules["seed"])
    sample_ids = choose_papers(test, findings, min(5, len(eligible_ids)), rules["seed"])
    pilot = manifest["pilot"]
    sample = {pid: test[pid] for pid in sample_ids}
    sample_chunks = build_chunks(sample, tokenizer, pilot["max_tokens"], pilot["overlap"])
    ratio = len(sample_chunks) / len(sample)
    estimated = project({pid: test[pid] for pid in eligible_ids}, ratio, manifest["resource_rates"], rules["settings"])
    limit = len(eligible_ids) if estimated["estimated_seconds"] <= rules["max_projected_seconds"] else rules["subset_papers"]
    selected = choose_papers(test, findings, limit, rules["seed"])
    papers = {pid: test[pid] for pid in selected}
    preflight = {"source": provenance, "audit_sample_ids": sample_ids, "audit_sample_chunks": len(sample_chunks),
                 "full": estimated, "selected": project(papers, ratio, manifest["resource_rates"], rules["settings"]),
                 "selected_ids": selected, "duplicate_findings": findings,
                 "quarantined_test_ids": sorted(set(test) - set(eligible_ids)),
                 "split_paper_counts": {"train": len(train), "dev": len(dev), "test": len(test)}}
    publish(args.output / "population.json", {"manifest": preflight, "sha256": digest(preflight)})
    return papers, preflight


def cached_or_query(cache, stage_name, setting, qid, function):
    value = cache.get(stage_name, setting, qid)
    if value is not None:
        return value, True
    value = function()
    cache.put(stage_name, setting, qid, value)
    return value, False


def query_text(paper, question, setting):
    return paper["title"] + "\n" + question if setting == "split-title" else question


def base_search(sparse, dense, text, depth, scope):
    result = {}
    for engine in (sparse, dense):
        hits, elapsed = timed(engine.search, text, depth, scope)
        result[engine.method] = [[chunk.id, value] for chunk, value in hits]
        result[f"{engine.method}_ms"] = elapsed
    return result


def run_base(args, manifest, cache, papers, chunks, jobs, tokenizer, counts, timings):
    pilot, phase3 = manifest["pilot"], manifest["phase3"]
    depth = phase3["primary_fusion_depth"]
    rows = {(setting, qid): cache.get("base", setting, qid) for setting, qid, _, _ in jobs}
    counts["base_cache_hits"] = sum(row is not None for row in rows.values())
    if counts["base_cache_hits"] == len(jobs):
        return rows
    with stage("dense_load", timings):
        encoder = BGE(args.model / "model.onnx", tokenizer, pilot["batch_size"], pilot["threads"])
    matrices = []
    by_paper = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk.paper_id].append(chunk)
    with stage("embedding_and_vector_index", timings):
        for pid in papers:
            group = by_paper[pid]
            if group:
                matrix, _, hit = cached_vectors(args.output / "embeddings", group, digest(manifest), encoder)
                matrices.append(matrix)
                counts["cached_vectors" if hit else "embedded_vectors"] += len(group)
        dense = Retrieval(chunks, method="dense", vectors=np.concatenate(matrices), encoder=encoder)
    with stage("bm25_index", timings):
        sparse = Retrieval(chunks, method="bm25")
    dense.search("Warmup", 1)
    sparse.search("Warmup", 1)
    with stage("base_query_work", timings):
        for setting, qid, pid, text in jobs:
            if rows[setting, qid] is None:
                result = base_search(sparse, dense, text, depth, pid if setting == "paper" else None)
                cache.put("base", setting, qid, result)
                rows[setting, qid] = result
    return rows


def run_rank(args, manifest, cache, chunks, jobs, base_rows, counts, timings):
    pilot, phase3 = manifest["pilot"], manifest["phase3"]
    model = None
    rows = {}
    with stage("fusion_and_rerank_work", timings):
        for i, (setting, qid, _, text) in enumerate(jobs, 1):
            base = base_rows[setting, qid]

            def finish():
                nonlocal model
                if model is None:
                    with stage("reranker_load", timings):
                        model = load_reranker(args.model, phase3, pilot)
                    model.score("Warmup", ["Warmup passage"])
                fused, fusion_ms = timed(reciprocal_rank_fusion, [base["bm25"], base["dense"]],
                                         phase3["rrf_constant"], phase3["primary_fusion_depth"])
                (candidates, ordered), rerank_ms = timed(rerank, text, fused, chunks, model, phase3["rerank_depth"])
                return {"hybrid": [list(pair) for pair in fused],
                        "reranked": [list(pair) for pair in ordered], "candidates": candidates,
                        "fusion_ms": fusion_ms, "rerank_ms": rerank_ms}

            rows[setting, qid], hit = cached_or_query(cache, "rerank", setting, qid, finish)
            counts["rerank_cache_hits"] += hit
            if i % 50 == 0:
                print(f"Reranking checkpoint {i}/{len(jobs)}", flush=True)
    return rows


def summarize(rows, rules):
    result = {}
    for scope in rules["settings"]:
        group = [r for r in rows if r["setting"] == scope]
        eligible = [r for r in group if r["status"] == "eligible"]
        for row in group:
            if any(bool(row["metrics"][m]) != (row["status"] == "eligible") for m in METHODS):
                raise ValueError("Method eligibility diverged")
        result[scope] = {
            "questions": len(group), "eligible": len(eligible),
            "papers": len({r["paper_id"] for r in group}),
            "eligible_papers": len({r["paper_id"] for r in eligible}),
            "metrics": {method: means(r["metrics"][method] for r in eligible) for method in METHODS},
            "paragraph_only": {method: means(r["paragraph_only"][method] for r in group) for method in METHODS},
            "latency_ms_p50_p95": {method: np.percentile(
                [r["timing_ms"][method] for r in group], [50, 95]).tolist()
                for method in (*METHODS, "rerank_only")},
            "paired": {f"{left}-{right}": {metric: paper_bootstrap([
                (r["paper_id"], r["metrics"][left][metric] - r["metrics"][right][metric]) for r in eligible
            ], rules["bootstrap_seed"], rules["bootstrap_draws"]) for metric in METRICS}
                       for left, right in PAIRS},
        }
    return result


def failure_sample(rows, seed):
    # Up to one reproducible representative per predefined pattern.
    group = [r for r in rows if r["setting"] == "paper" and r["status"] == "eligible"]
    rng = random.Random(seed)
    rng.shuffle(group)
    predicates = {
        "late_fused_evidence": lambda m: m["hybrid"]["recall@5"] == 0 < m["hybrid"]["recall@20"],
        "dense_better": lambda m: m["dense"]["recall@10"] > m["bm25"]["recall@10"],
        "sparse_better": lambda m: m["bm25"]["recall@10"] > m["dense"]["recall@10"],
        "fusion_gain": lambda m: m["hybrid"]["recall@10"] > max(m["dense"]["recall@10"], m["bm25"]["recall@10"]),
        "fusion_loss": lambda m: m["hybrid"]["recall@10"] < max(m["dense"]["recall@10"], m["bm25"]["recall@10"]),
        "rerank_gain": lambda m: m["reranked"]["ndcg@5"] > m["hybrid"]["ndcg@5"],
        "rerank_loss": lambda m: m["reranked"]["ndcg@5"] < m["hybrid"]["ndcg@5"],
    }
    result = {}
    for name, predicate in predicates.items():
        matches = [r for r in group if predicate(r["metrics"])]
        result[name] = {"count": len(matches), "question_id": matches[0]["question_id"] if matches else None,
                        "paper_id": matches[0]["paper_id"] if matches else None}
    missed = [r for r in group if r["fused_pool_recall"] == 0]
    result["missed_fused_pool"] = {"count": len(missed), "question_id": missed[0]["question_id"] if missed else None,
                                   "paper_id": missed[0]["paper_id"] if missed else None}
    return result


def run(args):
    lock = load_json(args.output / "protocol.json")
    manifest = lock["manifest"]
    if lock["sha256"] != digest(manifest) or any(manifest[k] != v for k, v in current_identity().items()):
        raise ValueError("Frozen manifest or implementation mismatch")
    pilot, rules = manifest["pilot"], manifest["rules"]
    for name, key in [("model.onnx", "model_sha256"), ("tokenizer.json", "tokenizer_sha256")]:
        if sha256(args.model / name) != pilot[key]:
            raise ValueError("BGE artifact checksum mismatch")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    timings = {}
    counts = Counter()
    with stage("resource_preflight_and_duplicate_audit", timings):
        papers, population = prepare(args, manifest, tokenizer)
    with stage("chunking_and_evidence_mapping", timings):
        chunks = build_chunks(papers, tokenizer, pilot["max_tokens"], pilot["overlap"])
        by_id = {c.id: c for c in chunks}
        by_paper = defaultdict(list)
        for chunk in chunks:
            by_paper[chunk.paper_id].append(chunk)
        queries = audit_questions(papers, by_paper)
    jobs = [(setting, q["question_id"], pid, query_text(papers[pid], q["question"], setting))
            for setting in rules["settings"] for pid, q, _, _ in queries]
    cache = Checkpoints(args.output / "checkpoints.sqlite", digest([lock["sha256"], population]))
    base = run_base(args, manifest, cache, papers, chunks, jobs, tokenizer, counts, timings)
    ranking = run_rank(args, manifest, cache, by_id, jobs, base, counts, timings)
    cache.db.close()
    records = []
    audits = []
    for pid, q, prior, gold in queries:
        qid = q["question_id"]
        audits.append({"paper_id": pid, "question_id": qid, "status": gold.status,
                       "phase2_status": prior.status, "sets": [sorted(s) for s in gold.sets],
                       "annotations": gold.annotations})
        for setting in rules["settings"]:
            a, b = base[setting, qid], ranking[setting, qid]
            ranks = {"bm25": a["bm25"], "dense": a["dense"], "hybrid": b["hybrid"], "reranked": b["reranked"]}
            timings_ms = {m: a[f"{m}_ms"] for m in ("bm25", "dense")}
            timings_ms["hybrid"] = timings_ms["bm25"] + timings_ms["dense"] + b["fusion_ms"]
            timings_ms["rerank_only"] = b["rerank_ms"]
            timings_ms["reranked"] = timings_ms["hybrid"] + b["rerank_ms"]
            fusion_pool = [by_id[cid] for cid, _ in ranks["hybrid"]]
            pool_depth = max(1, len(fusion_pool))
            records.append({
                "paper_id": pid, "question_id": qid, "setting": setting, "status": gold.status,
                "rankings": ranks, "rerank_candidates": b["candidates"], "timing_ms": timings_ms,
                "fused_pool_recall": score(fusion_pool, gold, [pool_depth]).get(f"recall@{pool_depth}"),
                "metrics": {m: score([by_id[cid] for cid, _ in r], gold, rules["cutoffs"]) for m, r in ranks.items()},
                "paragraph_only": {m: score([by_id[cid] for cid, _ in r], prior, rules["cutoffs"])
                                   for m, r in ranks.items()},
            })
    for name, rows in [("rankings.json", records), ("evidence.json", audits),
                       ("chunks.json", [{k: v for k, v in asdict(c).items() if k != "text"} for c in chunks])]:
        publish(args.output / name, rows)
    memory = psutil.Process().memory_info()
    report = {"protocol_sha256": lock["sha256"], "population_sha256": digest(population), "timings": timings,
              "counts": counts, "chunks": len(chunks), "peak_working_set": getattr(memory, "peak_wset", None),
              "available_memory_end": psutil.virtual_memory().available,
              "vector_cache_bytes": sum(p.stat().st_size for p in (args.output / "embeddings").glob("*")),
              "eligibility": Counter(a["status"] for a in audits),
              "evidence_labels": Counter(l["status"] for a in audits for ann in a["annotations"] for l in ann["labels"]),
              "results": summarize(records, rules), "failure_sample": failure_sample(records, rules["seed"]),
              "files": {name: {"bytes": (args.output / name).stat().st_size, "sha256": sha256(args.output / name)}
                        for name in ("protocol.json", "population.json", "development-robustness.json",
                                     "rankings.json", "evidence.json", "chunks.json", "checkpoints.sqlite")}}
    report_file = args.output / "report.json"
    write_json(args.output / ("replay.json" if report_file.exists() else "report.json"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.toml"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/phase3-final"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("model"))
    args = parser.parse_args()
    (freeze if args.mode == "freeze" else run)(args)


if __name__ == "__main__":
    main()
