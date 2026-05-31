#!/usr/bin/env python3
"""Plot LongBench official sweep scores."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "qmsum": "QMSum",
    "narrativeqa": "NarrativeQA",
    "repobench-p": "RepoBench-P",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-csv",
        default="/root/autodl-tmp/dataset/eval_results/quality_sweep_300/longbench_official/longbench_official_scores.csv",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    score_csv = Path(args.score_csv)
    output_dir = Path(args.output_dir) if args.output_dir else score_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(score_csv)
    baseline = (
        df[df["config"] == "baseline_fa3"]
        .set_index("dataset")["official_score"]
        .to_dict()
    )
    df["baseline_score"] = df["dataset"].map(baseline)
    df["retention_pct"] = df["official_score"] / df["baseline_score"] * 100
    df["delta_abs"] = df["official_score"] - df["baseline_score"]
    df.to_csv(output_dir / "longbench_official_scores_with_deltas.csv", index=False)

    plt.figure(figsize=(9, 5.2))
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("compression_ratio")
        plt.plot(
            group["compression_ratio"],
            group["official_score"],
            marker="o",
            linewidth=2,
            label=LABELS.get(dataset, dataset),
        )
    plt.xlabel("Compression ratio")
    plt.ylabel("LongBench official score")
    plt.title("LongBench Official Score vs Compression Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "official_score_vs_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.2))
    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("compression_ratio")
        plt.plot(
            group["compression_ratio"],
            group["retention_pct"],
            marker="o",
            linewidth=2,
            label=LABELS.get(dataset, dataset),
        )
    plt.axhline(100, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Compression ratio")
    plt.ylabel("Retention vs baseline (%)")
    plt.title("LongBench Official Score Retention")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "official_retention_vs_ratio.png", dpi=180)
    plt.close()

    pivot = df.pivot_table(
        index="compression_ratio",
        columns="dataset",
        values="official_score",
        aggfunc="first",
    ).sort_index()
    plt.figure(figsize=(8, 5.2))
    plt.imshow(pivot.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Official score")
    plt.xticks(range(len(pivot.index)), [f"{x:.1f}" for x in pivot.index])
    plt.yticks(range(len(pivot.columns)), [LABELS.get(x, x) for x in pivot.columns])
    plt.xlabel("Compression ratio")
    plt.title("LongBench Official Score Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "official_score_heatmap.png", dpi=180)
    plt.close()

    lines = ["# LongBench Official Score Plots", ""]
    lines.append("## Score")
    lines.append("")
    lines.append(
        df.pivot_table(
            index="compression_ratio",
            columns="dataset",
            values="official_score",
            aggfunc="first",
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".2f")
    )
    lines.append("")
    lines.append("## Retention vs Baseline (%)")
    lines.append("")
    lines.append(
        df.pivot_table(
            index="compression_ratio",
            columns="dataset",
            values="retention_pct",
            aggfunc="first",
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".1f")
    )
    lines.append("")
    (output_dir / "longbench_official_plot_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Wrote plots to {output_dir}")
    print("- official_score_vs_ratio.png")
    print("- official_retention_vs_ratio.png")
    print("- official_score_heatmap.png")
    print("- longbench_official_scores_with_deltas.csv")
    print("- longbench_official_plot_summary.md")


if __name__ == "__main__":
    main()
