"""Render held-out retrieval comparisons from a saved Phase 5 report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true", help="also save PNG previews")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    matplotlib.rcParams.update({"svg.hashsalt": "phase5", "font.size": 10})
    names = {"bm25": "BM25", "dense": "BGE", "hybrid": "RRF", "reranked": "RRF + MiniLM"}
    titles = {"paper": "Paper scoped", "split-title": "Sampled split, title cued"}
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True, layout="constrained")
    comparisons = ("dense-bm25", "hybrid-dense", "reranked-hybrid")
    for ax, (scope, result) in zip(axes, report["results"].items()):
        ax.axvline(0, linestyle="--", color="#6d7682", linewidth=.8)
        for y, contrast in enumerate(comparisons):
            estimate = result["paired"][contrast]["recall@10"]
            ax.hlines(y, *estimate["ci95"], color="#126581", linewidth=2)
            ax.scatter(estimate["delta"], y, color="#126581", zorder=3)
        ax.set_yticks(range(len(comparisons)), [
            " − ".join(names[m] for m in pair.split("-")) for pair in comparisons])
        ax.invert_yaxis()
        ax.set_title(titles[scope])
        ax.set_xlabel("Paired Recall@10 difference")
    fig.suptitle("QASPER held-out sample · paired 95% paper-cluster intervals")
    fig.savefig(args.output / "paired-recall.svg", metadata={"Date": None})
    if args.preview:
        fig.savefig(args.output / "paired-recall.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True, layout="constrained")
    colors = ("#1a6e8e", "#cb5d2d", "#487438", "#81478f")
    for ax, (scope, result) in zip(axes, report["results"].items()):
        for (method, label), color in zip(names.items(), colors):
            p50, p95 = result["latency_ms_p50_p95"][method]
            recall = result["metrics"][method]["recall@10"]["mean"]
            ax.hlines(recall, p50, p95, linewidth=1.5, color=color)
            ax.scatter(p50, recall, label=label, color=color, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Warm retrieval latency (ms, log scale)")
        ax.set_title(titles[scope])
    axes[0].set_ylabel("Recall@10")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("QASPER held-out sample · median latency with p95 endpoint")
    fig.savefig(args.output / "quality-latency.svg", metadata={"Date": None})
    if args.preview:
        fig.savefig(args.output / "quality-latency.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
