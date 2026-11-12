"""Run the CPU feasibility pilot or a bounded paired development experiment."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import tomllib

import numpy as np

from .__main__ import write_json
from .chunking import Chunk
from .corpus import load_split, paragraphs, sha256
from .evaluation import Gold, score
from .generation import PROMPT_VERSION, SYSTEM, SCHEMA, pack_context, parse_answer, server
from .phase3 import Checkpoints, load_json


CONDITIONS = ("bge", "reranked", "reference")
ANSWER_TYPES = ("extractive", "free_form", "yes_no")


def validate(config):
    if (config["schema_version"] != 1 or config["prompt_version"] != PROMPT_VERSION
            or not 128 <= config["context_window"] <= 4096
            or not 0 < config["max_output"] < config["context_window"]
            or not 0 < config["evidence_tokens"] < config["context_window"]
            or not 1 <= config["threads"] <= 4 or not 1 <= config["max_passages"] <= 10):
        raise ValueError("Invalid Phase 4 configuration")


def evidence_found(chunks, gold):
    if gold.status != "eligible" or not chunks:
        return False if gold.status == "eligible" else None
    return bool(score(chunks, gold, [len(chunks)])[f"annotated_set_coverage@{len(chunks)}"])


def load_inputs(data, phase3):
    report = load_json(phase3 / "report.json")
    corpus, source = load_split(data, "dev")
    if report["source"] != source:
        raise ValueError("Phase 3 source mismatch")
    for name in ("rankings.json", "chunks.json", "evidence-audit.json"):
        if sha256(phase3 / name) != report["files"][name]["sha256"]:
            raise ValueError(f"Phase 3 artifact checksum mismatch: {name}")
    papers = {pid: corpus[pid] for pid in report["paper_ids"]}
    source_paragraphs = {p.id: p for pid, paper in papers.items() for p in paragraphs(pid, paper)}
    chunks = {}
    by_paragraph = defaultdict(list)
    for raw in load_json(phase3 / "chunks.json"):
        paragraph = source_paragraphs[raw["paragraph_id"]]
        if paragraph.paper_id != raw["paper_id"] or not 0 <= raw["start"] < raw["end"] <= len(paragraph.text):
            raise ValueError(f"Invalid source span: {raw['id']}")
        chunk = Chunk(**raw, text=paragraph.text[raw["start"]:raw["end"]])
        chunks[chunk.id] = chunk
        by_paragraph[paragraph.id].append(chunk)
    audits = {a["question_id"]: a for a in load_json(phase3 / "evidence-audit.json")}
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    rows = {}
    for row in load_json(phase3 / "rankings.json"):
        if row["setting"] != "paper":
            continue
        qid = row["question_id"]
        q, audit = questions[qid], audits[qid]
        sets = [set(group) for group in audit["sets"]]
        gold = Gold(audit["status"], sets, {
            pid: (len(source_paragraphs[pid].text) - len(source_paragraphs[pid].text.lstrip()),
                  len(source_paragraphs[pid].text.rstrip())) for group in sets for pid in group
        }, audit["annotations"])
        answer_types = Counter(a["answer_type"] for a in audit["annotations"]
                               if a["answer_type"] in ANSWER_TYPES)
        kind = (min(answer_types, key=lambda label: (-answer_types[label], label))
                if answer_types and gold.status == "eligible" else gold.status)
        choices = [[c for pid in sorted(group) for c in by_paragraph[pid]] for group in sets]
        reference = min(choices, key=lambda group: (len(group), sum(len(c.text) for c in group),
                                                     tuple(c.id for c in group)), default=[])
        rows[qid] = {"question": q, "paper_id": row["paper_id"], "kind": kind, "gold": gold,
                     "reference": reference, "retrieval": row}
    return report, rows, chunks, papers, source_paragraphs


def sample(rows, limit, negatives, seed, omit, max_passages):
    rng = random.Random(seed)
    candidate = sorted(set(rows) - set(omit))
    rng.shuffle(candidate)
    strata = defaultdict(list)
    negative_ids = []
    for qid in candidate:
        row = rows[qid]
        if row["kind"] == "unanswerable":
            negative_ids.append(qid)
        if row["kind"] in ANSWER_TYPES and 0 < len(row["reference"]) <= max_passages:
            hit = row["retrieval"]["metrics"]["dense"]["annotated_set_coverage@5"] > 0
            strata[row["kind"], hit].append(qid)
    chosen = []
    while len(chosen) < limit:
        prior = len(chosen)
        for kind in ANSWER_TYPES:
            for hit in (False, True):
                group = strata[kind, hit]
                if group and len(chosen) < limit:
                    chosen.append(group.pop())
        if len(chosen) == prior:
            break
    return chosen, negative_ids[:negatives]


def make_plan(rows, config, mode, pilot):
    if mode == "pilot":
        positive, negative = sample(rows, 6, 2, config["seed"], [], config["max_passages"])
        return [(qid, "reference") for qid in positive] + [(qid, "bge") for qid in negative]
    prior = {row["question_id"] for row in pilot["rows"]}
    positive, negative = sample(rows, 12, 3, config["seed"] + 1, prior, config["max_passages"])
    plan = [(qid, condition) for qid in positive for condition in CONDITIONS]
    plan += [(qid, condition) for qid in negative for condition in CONDITIONS[:2]]
    plan += [(qid, "removed") for qid in positive[:4]]
    return plan


def prepare_candidates(row, condition, chunks, papers, paragraphs_by_id):
    retrieval = row["retrieval"]
    method = {"bge": "dense", "reranked": "reranked"}.get(condition)
    if method:
        ids = [cid for cid, _ in retrieval["rankings"][method]]
    else:
        ids = [c.id for c in row["reference"]] if condition == "reference" else []
    return [{
        "chunk_id": cid, "paper_id": chunks[cid].paper_id,
        "title": papers[chunks[cid].paper_id]["title"], "paragraph_id": chunks[cid].paragraph_id,
        "section": paragraphs_by_id[chunks[cid].paragraph_id].section,
        "start": chunks[cid].start, "end": chunks[cid].end, "text": chunks[cid].text,
    } for cid in ids], retrieval["timing_ms"].get(method) if method else None


def run_one(backend, row, condition, chunks, papers, source_paragraphs, config):
    started = time.perf_counter()
    q = row["question"]
    ranked, retrieval_ms = prepare_candidates(row, condition, chunks, papers, source_paragraphs)
    packed, excluded, tokens, base_tokens = pack_context(backend, q["question"], ranked, config)
    pack_ms = (time.perf_counter() - started) * 1000
    packed_coverage = evidence_found([chunks[p["chunk_id"]] for p in packed], row["gold"])
    record = {
        "question_id": q["question_id"], "question": q["question"], "paper_id": row["paper_id"],
        "answer_type": row["kind"], "condition": condition, "prompt_version": config["prompt_version"],
        "retrieval_method": {"bge": "dense", "reranked": "reranked"}.get(condition),
        "retrieved": [p["chunk_id"] for p in ranked], "context": packed, "excluded": excluded,
        "full_retrieved_annotated_set": evidence_found([chunks[p["chunk_id"]] for p in ranked], row["gold"]),
        "packed_annotated_set": packed_coverage, "retrieval_ms": retrieval_ms, "context_ms": pack_ms,
        "retrieval_ndcg_at_5": row["retrieval"]["metrics"].get(
            {"bge": "dense", "reranked": "reranked"}.get(condition), {}).get("ndcg@5"),
        "prompt_tokens": len(tokens), "evidence_tokens": len(tokens) - base_tokens,
        "prompt_token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "generation_ms": None, "local_latency_ms": None, "composed_end_to_end_ms": None,
        "raw": None, "output_tokens": None, "decode_tokens_per_second": None,
    }
    if condition == "reference" and not packed_coverage:
        record.update(status="reference_overflow", abstained=None)
        return record
    start = time.perf_counter()
    response = backend.generate(tokens, config)
    elapsed = (time.perf_counter() - start) * 1000
    raw = response["content"]
    record.update(parse_answer(raw, {p["id"] for p in packed}))
    record.update(raw=raw, output_tokens=response.get("tokens_predicted"),
                  runtime_timings=response.get("timings", {}),
                  decode_tokens_per_second=response.get("timings", {}).get("predicted_per_second"),
                  stop_type=response.get("stop_type"), runtime_truncated=response.get("truncated"),
                  runtime_prompt_tokens=response.get("tokens_evaluated"), generation_ms=elapsed,
                  local_latency_ms=pack_ms + elapsed,
                  composed_end_to_end_ms=pack_ms + elapsed + retrieval_ms if retrieval_ms is not None else None)
    return record


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["condition"]].append(row)
    result = {}
    for condition, rows in groups.items():
        result[condition] = {"n": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                             "native_unanswerable_n": sum(r["answer_type"] == "unanswerable" for r in rows),
                             "packed_annotated_set_n": sum(r["packed_annotated_set"] is True for r in rows)}
        for field in ("retrieval_ms", "context_ms", "generation_ms", "local_latency_ms",
                      "composed_end_to_end_ms", "decode_tokens_per_second", "prompt_tokens", "output_tokens"):
            values = [r[field] for r in rows if r[field] is not None]
            if values:
                result[condition][field] = dict(zip(("p50", "p95"), np.percentile(values, [50, 95]).tolist()))
    return result


def run(args):
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    validate(config)
    prior = load_json(args.pilot) if args.mode == "study" else None
    if prior is not None and (not prior["pilot_gate"]["passed"] or prior["config"] != config):
        raise ValueError("The same configuration must pass the feasibility pilot before the study")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        existing = load_json(args.output / "report.json")
        if existing.get("mode") != args.mode:
            raise ValueError("Use a dedicated Phase 4 output directory")
    if sha256(args.model) != config["model_sha256"]:
        raise ValueError("GGUF checksum mismatch")
    options = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    runtime = subprocess.check_output([str(args.runtime.resolve()), "--version"], stderr=subprocess.STDOUT,
                                      text=True, **options).strip()
    if f"version: {config['runtime_tag'].removeprefix('b')}" not in runtime:
        raise ValueError("Unexpected llama.cpp runtime version")
    retrieval_report, inputs, chunks, papers, source_paragraphs = load_inputs(args.data, args.phase3)
    plan = make_plan(inputs, config, args.mode, prior)
    identity = {"mode": args.mode, "config": config, "runtime": runtime,
                "runtime_executable_sha256": sha256(args.runtime), "model_bytes": args.model.stat().st_size,
                "retrieval_fingerprint": retrieval_report["fingerprint"],
                "retrieval_inputs": {name: item["sha256"] for name, item in retrieval_report["files"].items()
                                     if name != "checkpoints.sqlite"},
                "prompt": SYSTEM, "response_schema": SCHEMA, "plan": plan,
                "code": {name: sha256(Path(__file__).with_name(name)) for name in
                         ("phase4.py", "generation.py", "phase3.py", "evaluation.py")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (args.output / "report.json").exists() and existing["fingerprint"] != fingerprint:
        raise ValueError("Use a fresh output directory for changed Phase 4 inputs")
    cache = Checkpoints(args.output / "checkpoints.sqlite", fingerprint)
    ready = {(qid, cond): cache.get("generation", cond, qid) for qid, cond in plan}
    cache_hits = sum(row is not None for row in ready.values())
    load_seconds, peak_bytes = None, None
    if cache_hits != len(plan):
        log = args.output / f"runtime-{time.time_ns()}.log"
        with server(args.runtime, args.model, config, log) as (backend, process, load_seconds):
            for qid, cond in plan:
                if ready[qid, cond] is None:
                    row = run_one(backend, inputs[qid], cond, chunks, papers, source_paragraphs, config)
                    cache.put("generation", cond, qid, row)
                    ready[qid, cond] = row
                    print(json.dumps({k: row.get(k) for k in ("question_id", "condition", "status",
                                                             "generation_ms", "decode_tokens_per_second")}),
                          flush=True)
            memory = process.memory_info()
            peak_bytes = getattr(memory, "peak_wset", memory.rss)
    cache.db.close()
    records = [ready[qid, cond] for qid, cond in plan]
    write_json(args.output / "records.json", records)
    report = {**identity, "fingerprint": fingerprint, "cache_hits": cache_hits,
              "load_seconds": load_seconds, "peak_server_working_set_bytes": peak_bytes,
              "summary": summarize(records),
              "rows": [{key: row[key] for key in ("question_id", "condition", "status")} for row in records],
              "files": {name: {"bytes": (args.output / name).stat().st_size,
                                "sha256": sha256(args.output / name)}
                        for name in ("records.json", "checkpoints.sqlite")}}
    if args.mode == "pilot":
        valid = sum(r["status"] in {"answered", "abstained"} for r in records)
        reference_answers = sum(r["condition"] == "reference" and r["status"] == "answered" for r in records)
        speeds = [r["decode_tokens_per_second"] for r in records if r["decode_tokens_per_second"] is not None]
        speed = float(np.median(speeds)) if speeds else 0.0
        report["pilot_gate"] = {"passed": len(records) == 8 and valid >= 7 and reference_answers >= 4 and speed >= 2,
                                "valid": valid, "reference_answers": reference_answers,
                                "median_decode_tokens_per_second": speed}
    if cache_hits == len(plan) and (args.output / "report.json").exists():
        write_json(args.output / "replay.json", report)
    else:
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "study"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--phase3", type=Path, default=Path("artifacts/phase3-final"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase4.toml"))
    parser.add_argument("--pilot", type=Path, default=Path("artifacts/phase4-pilot/report.json"))
    parser.add_argument("--model", type=Path, default=Path("model/qwen/model.gguf"))
    parser.add_argument("--runtime", type=Path, default=Path("model/qwen/runtime/llama-server.exe"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
