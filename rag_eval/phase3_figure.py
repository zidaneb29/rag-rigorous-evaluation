"""Render Phase 3 paired Recall@10 differences from the recorded report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.hashsalt": "phase3-qasper", "font.size": 10})
    report = json.loads(args.report.read_text(encoding="utf-8"))
    pairs = report["paired_differences"]
    methods = {"bm25": "BM25", "dense": "BGE", "hybrid": "RRF", "reranked": "RRF + MiniLM"}
    comparisons = [("RRF − BM25", "hybrid-bm25"), ("RRF − BGE", "hybrid-dense"),
                   ("RRF + MiniLM − RRF", "reranked-hybrid")]
    panels = [
        ("Paper-scoped method comparisons",
         [(label, pairs[f"paper/{key}"]["recall@10"]) for label, key in comparisons
          if f"paper/{key}" in pairs]),
        ("Split-wide title cue effect",
         [(methods[method], values["recall@10"]) for method, values in report["title_effect"].items()]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.15), layout="constrained", sharex=True)
    for axis, (title, entries) in zip(axes, panels):
        axis.axvline(0, color="#697586", linewidth=0.9, linestyle="--")
        for pos, (_, value) in enumerate(entries):
            delta = value["delta"]
            lo, hi = value["ci95"]
            axis.hlines(pos, lo, hi, color="#125577", linewidth=1.6)
            axis.scatter([delta], [pos], color="#125577", zorder=3, s=24)
        axis.set_yticks(range(len(entries)), [label for label, _ in entries])
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Paired Recall@10 difference")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    fig.suptitle(f"QASPER development subset · {len(report['paper_ids'])} papers", fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
