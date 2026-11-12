"""Export blinded reviews and aggregate deterministic or human-coded outcomes."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

import numpy as np

from .__main__ import write_json
from .corpus import load_split, sha256
from .phase3 import load_json


def proportion(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def machine_metrics(rows):
    positives = [r for r in rows if r["answer_type"] != "unanswerable" and r["condition"] != "removed"]
    negatives = [r for r in rows if r["answer_type"] == "unanswerable"]
    removed = [r for r in rows if r["condition"] == "removed"]
    claims = sum(r.get("claim_count", 0) for r in rows)
    return {
        "statuses": dict(Counter(r["status"] for r in rows)),
        "false_abstentions": proportion(sum(r["abstained"] is True for r in positives), len(positives)),
        "native_abstentions": proportion(sum(r["abstained"] is True for r in negatives), len(negatives)),
        "answers_to_native_unanswerables": proportion(sum(r["abstained"] is False for r in negatives),
                                                       len(negatives)),
        "removed_context_abstentions": proportion(sum(r["abstained"] is True for r in removed), len(removed)),
        "missing_citations": proportion(sum(r.get("missing_citations", 0) for r in rows), claims),
        "invalid_citation_ids": sum(len(r.get("invalid_ids", [])) for r in rows),
        "failure_attribution": dict(Counter(
            "available_in_context" if r["packed_annotated_set"] else
            "retrieved_not_packed" if r["full_retrieved_annotated_set"] else
            "missing_from_recorded_ranking" for r in positives
            if r["condition"] in {"bge", "reranked"})),
    }


def make_packets(records, references, seed):
    rows = list(records)
    random.Random(seed).shuffle(rows)
    blind, key = [], []
    for index, row in enumerate(rows, 1):
        review_id = f"R{index:03}"
        blind.append({
            "review_id": review_id, "question": row["question"],
            "reference_answers": references[row["question_id"]],
            "visible_passages": [{"id": p["id"], "text": p["text"]} for p in row["context"]],
            "raw_output": row["raw"],
        })
        key.append({"review_id": review_id, "question_id": row["question_id"],
                    "condition": row["condition"]})
    return blind, key


def rank_values(values):
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for position in ordered[start:end]:
            ranks[position] = (start + end - 1) / 2
        start = end
    return ranks


def human_metrics(labels, key, rows):
    ids = {item["review_id"]: item for item in key}
    indexed = {(r["question_id"], r["condition"]): r for r in rows}
    quality_by_question = defaultdict(dict)
    results = defaultdict(list)
    seen = set()
    for label in labels:
        rid = label["review_id"]
        if rid in seen or rid not in ids:
            raise ValueError(f"Duplicate or unknown review ID: {rid}")
        seen.add(rid)
        item = ids[rid]
        condition = item["condition"]
        row = indexed[item["question_id"], condition]
        quality, context = label["correctness"], label["context_sufficiency"]
        claim_labels, link_labels = label["claims"], label["citation_links"]
        if (quality not in {"correct", "partial", "incorrect", "abstained", "unjudgeable"}
                or context not in {"sufficient", "insufficient", "uncertain"}
                or any(x not in {"supported", "unsupported", "contradicted", "uncertain"} for x in claim_labels)
                or any(x not in {"supporting", "joint", "unsupported", "contradicted", "invalid", "uncertain"}
                       for x in link_labels)):
            raise ValueError(f"Invalid review label: {rid}")
        results[condition].append((row, quality, context, claim_labels, link_labels))
        quality_by_question[item["question_id"]][condition] = quality
    summary = {}
    for condition, reviewed in results.items():
        quality = Counter(r[1] for r in reviewed)
        context = Counter(r[2] for r in reviewed)
        claims = Counter(x for r in reviewed for x in r[3])
        links = Counter(x for r in reviewed for x in r[4])
        association = [(r["retrieval_ndcg_at_5"], int(q == "correct"))
                       for r, q, _, _, _ in reviewed
                       if r["retrieval_ndcg_at_5"] is not None and q != "unjudgeable"]
        x, y = zip(*association) if association else ([], [])
        rho = float(np.corrcoef(rank_values(x), rank_values(y))[0, 1]) if (
            len(set(x)) > 1 and len(set(y)) > 1) else None
        summary[condition] = {
            "reviewed": len(reviewed), "correctness": dict(quality), "context_sufficiency": dict(context),
            "claim_support": dict(claims), "citation_support": dict(links),
            "citation_precision_among_resolved_links": proportion(links["supporting"] + links["joint"],
                                                                  links.total() - links["uncertain"]),
            "unsupported_or_contradicted_claims": proportion(claims["unsupported"] + claims["contradicted"],
                                                              claims.total() - claims["uncertain"]),
            "sufficient_context_with_unsupported_claim": sum(
                c == "sufficient" and any(x in {"unsupported", "contradicted"} for x in s)
                for _, _, c, s, _ in reviewed),
            "ndcg_correctness_spearman": {"rho": rho, "n": len(association)},
        }
    pairs = [labels for qid, labels in quality_by_question.items()
             if {"bge", "reranked"} <= labels.keys()
             and indexed[qid, "bge"]["answer_type"] != "unanswerable"
             and all(labels[m] != "unjudgeable" for m in ("bge", "reranked"))]
    return {"by_condition": summary, "paired_answerable_correctness": {
        "n": len(pairs),
        "reranked_only_correct": sum(p["reranked"] == "correct" and p["bge"] != "correct" for p in pairs),
        "bge_only_correct": sum(p["bge"] == "correct" and p["reranked"] != "correct" for p in pairs),
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    report = load_json(args.run / "report.json")
    records_path = args.run / "records.json"
    if sha256(records_path) != report["files"]["records.json"]["sha256"]:
        raise ValueError("Generation records checksum mismatch")
    records = load_json(records_path)
    corpus, source = load_split(args.data, "dev")
    references = {}
    for row in records:
        q = next(q for q in corpus[row["paper_id"]]["qas"] if q["question_id"] == row["question_id"])
        references[row["question_id"]] = [{
            key: a["answer"][key] for key in ("unanswerable", "extractive_spans", "yes_no", "free_form_answer")
        } for a in q["answers"]]
    packets, key = make_packets(records, references, report["config"]["seed"])
    by_condition = {condition: machine_metrics([r for r in records if r["condition"] == condition])
                    for condition in sorted({r["condition"] for r in records})}
    analysis = {"records_sha256": sha256(records_path), "source": source, "machine": by_condition,
                "human_review": None}
    if args.labels:
        analysis.update(human_review=human_metrics(load_json(args.labels), key, records),
                        labels_sha256=sha256(args.labels))
    write_json(args.run / "review-packet.json", packets)
    write_json(args.run / "review-key.json", key)
    write_json(args.run / "evaluation.json", analysis)
    print(hashlib.sha256((args.run / "review-packet.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
