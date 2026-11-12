"""Export a frozen development packet and validate results returned from Colab."""

import argparse
from collections import Counter
from pathlib import Path
import tomllib

import numpy as np

from .corpus import sha256
from .generation import SCHEMA, parse_answer
from .phase3 import load_json
from .phase4 import load_inputs, prepare_candidates, sample
from .phase4b import SYNTHETIC, chat_for
from .phase4c_runtime import STAGES, digest, render, save, validate_checkpoint, validate_packet


def validate_config(config, original):
    fixed = ("seed", "context_window", "evidence_tokens", "max_passages", "max_output")
    if (config["protocol"] != "phase4c-v1" or config["prompt"] != "phase4b-C"
            or config["temperature"] != 0 or config["quantization"] != "Q4_K_M"
            or config["model"] != "Qwen/Qwen2.5-7B-Instruct-GGUF"
            or any(config[key] != original[key] for key in fixed)
            or [s["name"] for s in config["shards"]] != [
                f"qwen2.5-7b-instruct-q4_k_m-0000{i}-of-00002.gguf" for i in (1, 2)]
            or any(len(s["sha256"]) != 64 for s in config["shards"])):
        raise ValueError("Phase 4C configuration differs from the frozen protocol")


def case(stage, qid, question, context, answerable, condition):
    return {"stage": stage, "id": qid, "answerable": answerable, "condition": condition,
            "context": context, "chat": chat_for("C", question, context),
            "empty_chat": chat_for("C", question, [])}


def export(args):
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    base = load_json(args.original / "report.json")
    validate_config(config, base["config"])
    original_path = args.original / "records.json"
    if sha256(original_path) != base["files"]["records.json"]["sha256"]:
        raise ValueError("Original Phase 4 records checksum mismatch")
    old = load_json(original_path)
    report, inputs, chunks, papers, paragraphs = load_inputs(args.data, args.phase3)
    cases = [case("sanity", key, question, [{"id": "P1", "text": text}], answerable, "synthetic")
             for key, question, text, answerable in SYNTHETIC]
    cases.extend(case("original", row["question_id"], row["question"], row["context"],
                      row["answer_type"] != "unanswerable", row["condition"]) for row in old)
    # Conservative preselection keeps reference examples short enough for preflight token checks.
    eligible = {qid: row for qid, row in inputs.items()
                if sum(len(c.text) for c in row["reference"]) <= 3800}
    positives, negatives = sample(eligible, 12, 3, config["confirmation_seed"],
                                  [r["question_id"] for r in old], config["max_passages"])
    if len(positives) != 12 or len(negatives) != 3:
        raise ValueError("Frozen confirmation strata did not produce the expected sample")
    for qid in positives + negatives:
        row = inputs[qid]
        answerable = qid in positives
        condition = "reference" if answerable else "bge"
        ranked, _ = prepare_candidates(row, condition, chunks, papers, paragraphs)
        context = [p | {"id": f"P{i}"} for i, p in enumerate(ranked[:config["max_passages"]], 1)]
        cases.append(case("confirmation", qid, row["question"]["question"], context, answerable, condition))
    payload = {"config": config, "schema": SCHEMA, "cases": cases,
               "original_records_sha256": sha256(original_path),
               "retrieval_fingerprint": report["fingerprint"]}
    packet = {"payload": payload, "sha256": digest(payload)}
    validate_packet(packet)
    args.output.mkdir(parents=True, exist_ok=True)
    packet_path = args.output / "phase4c-packet.json"
    if packet_path.exists() and load_json(packet_path) != packet:
        raise ValueError("Use a new output directory for a different packet")
    save(packet_path, packet)
    # Keep QASPER reference answers and review notes on the laptop.
    review = [{"id": c["id"], "stage": c["stage"], "answerable": c["answerable"],
               "references": [a["answer"] for a in inputs[c["id"]]["question"]["answers"]],
               "correct_and_supported": None, "unsupported_factual_answer": None,
               "citation_support": None, "notes": ""}
              for c in cases if c["stage"] != "sanity"]
    review_path = args.output / "review-template.local.json"
    if not review_path.exists():
        save(review_path, review)
    print(packet["sha256"])


def inspect(packet, checkpoint):
    cases = validate_packet(packet)["cases"]
    rows = validate_checkpoint(packet, checkpoint)
    counts = {stage: Counter() for stage in STAGES}
    timing = {stage: [] for stage in STAGES}
    for case, wrapped in zip(cases, rows):
        row = wrapped["record"]
        if row["prompt"] != render(case["chat"]):
            raise ValueError("Returned prompt differs from the frozen manifest")
        parsed = parse_answer(row["raw"], {p["id"] for p in case["context"]})
        counter = counts[case["stage"]]
        counter["completed"] += 1
        counter[parsed["status"]] += 1
        counter["answered_on_answerable"] += int(case["answerable"] and parsed["abstained"] is False)
        counter["abstained_on_negative"] += int(not case["answerable"] and parsed["abstained"] is True)
        counter["invalid_citation_ids"] += len(parsed["invalid_ids"])
        counter["resolving_citation_ids"] += sum(len(c["citations"]) for c in parsed["claims"])
        counter["missing_citations"] += parsed["missing_citations"]
        timing[case["stage"]].append((row["latency_s"], row["output_tokens"] / row["latency_s"]))
    return {"packet_sha256": packet["sha256"], "checkpoint_sha256": digest(checkpoint),
            "automatic_counts": counts, "sessions": checkpoint["sessions"],
            "timing": {stage: {"latency_p50_p95_s": np.percentile(
                [latency for latency, _ in values], [50, 95]).tolist(),
                "median_end_to_end_tokens_per_second": float(np.median([rate for _, rate in values]))}
                for stage, values in timing.items() if values},
            "manual_gate": "pending semantic review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--config", type=Path, default=Path("configs/phase4c.toml"))
    exporter.add_argument("--original", type=Path, default=Path("artifacts/phase4-pilot"))
    exporter.add_argument("--data", type=Path, default=Path("data"))
    exporter.add_argument("--phase3", type=Path, default=Path("artifacts/phase3-final"))
    importer = sub.add_parser("import")
    importer.add_argument("--packet", type=Path, required=True)
    importer.add_argument("--results", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args)
        return
    packet, results = load_json(args.packet), load_json(args.results)
    report = inspect(packet, results)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "imported-results.json"
    if target.exists() and load_json(target) != results:
        raise ValueError("Use a new import directory for different results")
    save(target, results)
    save(args.output / "automatic-review.json", report)
    print(report["checkpoint_sha256"])


if __name__ == "__main__":
    main()
