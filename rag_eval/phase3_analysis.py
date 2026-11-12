"""Paired paper-level uncertainty and descriptive retrieval summaries."""

from collections import Counter, defaultdict

import numpy as np


def paper_bootstrap(pairs, seed, replicates):
    groups = defaultdict(list)
    for paper_id, difference in pairs:
        groups[paper_id].append(difference)
    if not groups:
        return None
    ordered = [groups[pid] for pid in sorted(groups)]
    totals = np.array([sum(group) for group in ordered])
    sizes = np.array([len(group) for group in ordered])
    draws = np.random.default_rng(seed).integers(0, len(groups), (replicates, len(groups)))
    deltas = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {"delta": float(totals.sum() / sizes.sum()),
            "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
            "papers": len(groups), "questions": int(sizes.sum())}


def means(values):
    grouped = defaultdict(list)
    for item in values:
        for metric, value in item.items():
            grouped[metric].append(value)
    return {name: {"mean": float(np.mean(group)), "n": len(group)} for name, group in grouped.items()}


def summarize(rows, audits, config):
    metrics, paragraph_only, latency, contrasts, complement = {}, {}, {}, {}, {}
    by_setting = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row)
    methods = list(rows[0]["metrics"])
    for setting, group in by_setting.items():
        metrics[setting] = {m: means(row["metrics"][m] for row in group) for m in methods}
        paragraph_only[setting] = {m: means(row["paragraph_only_metrics"][m] for row in group)
                                   for m in methods}
        latency[setting] = {
            name: dict(zip(("p50", "p95"), np.percentile(
                [row["timing_ms"][name] for row in group], [50, 95]).tolist()))
            for name in group[0]["timing_ms"]
        }
        complement[setting] = Counter()
        for row in group:
            left = row["metrics"]["bm25"].get("recall@10")
            right = row["metrics"]["dense"].get("recall@10")
            if left is not None and right is not None:
                complement[setting][("both" if left and right else "bm25_only" if left else
                                     "dense_only" if right else "neither")] += 1
        pairs = [("dense", "bm25"), ("hybrid", "bm25"), ("hybrid", "dense")]
        if "reranked" in methods:
            pairs += [("reranked", "hybrid"), ("reranked", "bm25"), ("reranked", "dense")]
        for left, right in pairs:
            contrasts[f"{setting}/{left}-{right}"] = {
                metric: paper_bootstrap(
                    [(row["paper_id"], row["metrics"][left][metric] - row["metrics"][right][metric])
                     for row in group if metric in row["metrics"][left] and metric in row["metrics"][right]],
                    config["bootstrap_seed"], config["bootstrap_replicates"])
                for metric in ("recall@10", "ndcg@5", "mrr@20")
            }
    uncued = {row["question_id"]: row for row in by_setting["split-question"]}
    title = by_setting["split-title"]
    title_effect = {
        method: {
            metric: paper_bootstrap(
                [(row["paper_id"], row["metrics"][method][metric] -
                  uncued[row["question_id"]]["metrics"][method][metric])
                 for row in title if metric in row["metrics"][method]],
                config["bootstrap_seed"], config["bootstrap_replicates"])
            for metric in ("target_hit@10", "recall@10", "ndcg@5", "mrr@20")
        } for method in methods
    }
    evidence = {}
    for name, group in [("pilot", [a for a in audits if a["pilot_paper"]]), ("expansion", audits)]:
        transitions = Counter(f'{a["phase2_status"]} -> {a["status"]}' for a in group)
        label_transitions = Counter(f'{label["phase2_status"]} -> {label["status"]}'
                                    for a in group for annotation in a["annotations"] for label in annotation["labels"])
        evidence[name] = {"question_transitions": transitions, "label_transitions": label_transitions,
                          "explicit_title_in_question": sum(a["explicit_title_in_question"] for a in group)}
    return {
        "metrics": metrics, "latency_ms": latency, "paragraph_only_sensitivity": paragraph_only,
        "paired_differences": contrasts, "complementarity_at_10": complement, "title_effect": title_effect,
        "evidence": evidence, "fusion_depth_sensitivity": {
            str(depth): means(row["fusion_sensitivity"][str(depth)] for row in by_setting["paper"])
            for depth in config["fusion_depths"] if depth != config["primary_fusion_depth"]},
    }
