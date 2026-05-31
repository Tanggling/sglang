#!/usr/bin/env python3
"""Analyze and visualize quality sweep summary JSON files only."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_LABELS = {
    "qmsum": "QMSum (ROUGE-L)",
    "narrativeqa": "NarrativeQA (F1)",
    "repobench-p": "RepoBench-P (F1)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        default="/root/autodl-tmp/dataset/eval_results/quality_sweep_300",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_summaries(summary_dir):
    rows = []
    for path in sorted(Path(summary_dir).glob("*.summary.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No summary rows found in {summary_dir}")
    df["compression_ratio"] = df["compression_ratio"].fillna(0.0).astype(float)
    df["config_order"] = df["compression_ratio"]
    return df.sort_values(["dataset", "length_bin", "compression_ratio"])


def add_baseline_deltas(df):
    all_df = df[df["length_bin"] == "ALL"].copy()
    baseline = all_df[all_df["config"] == "baseline_fa3"]
    baseline_scores = baseline.set_index("dataset")["primary_score"].to_dict()

    all_df["baseline_score"] = all_df["dataset"].map(baseline_scores)
    all_df["delta_abs"] = all_df["primary_score"] - all_df["baseline_score"]
    all_df["retention_pct"] = all_df["primary_score"] / all_df["baseline_score"] * 100
    return all_df


def write_markdown(all_df, output_path):
    lines = []
    lines.append("# Quality Sweep Summary")
    lines.append("")
    lines.append("Scores use each dataset's primary metric: QMSum=ROUGE-L, NarrativeQA=F1, RepoBench-P=F1.")
    lines.append("")

    pivot = all_df.pivot_table(
        index="compression_ratio",
        columns="dataset",
        values="primary_score",
        aggfunc="first",
    ).reset_index()
    lines.append("## Primary Scores")
    lines.append("")
    lines.append(pivot.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    retention = all_df.pivot_table(
        index="compression_ratio",
        columns="dataset",
        values="retention_pct",
        aggfunc="first",
    ).reset_index()
    lines.append("## Retention vs Baseline (%)")
    lines.append("")
    lines.append(retention.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")

    lines.append("## Best Compressed Ratio Per Dataset")
    lines.append("")
    compressed = all_df[all_df["config"] != "baseline_fa3"]
    best_rows = []
    for dataset, group in compressed.groupby("dataset"):
        best = group.sort_values("primary_score", ascending=False).iloc[0]
        best_rows.append(
            {
                "dataset": dataset,
                "best_ratio": best["compression_ratio"],
                "score": best["primary_score"],
                "baseline": best["baseline_score"],
                "delta_abs": best["delta_abs"],
                "retention_pct": best["retention_pct"],
            }
        )
    lines.append(pd.DataFrame(best_rows).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_primary_scores(all_df, output_dir):
    plt.figure(figsize=(9, 5.2))
    for dataset, group in all_df.groupby("dataset"):
        group = group.sort_values("compression_ratio")
        plt.plot(
            group["compression_ratio"],
            group["primary_score"],
            marker="o",
            linewidth=2,
            label=DATASET_LABELS.get(dataset, dataset),
        )
    plt.xlabel("Compression ratio")
    plt.ylabel("Primary score")
    plt.title("Quality vs Compression Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "primary_score_vs_ratio.png", dpi=180)
    plt.close()


def plot_retention(all_df, output_dir):
    plt.figure(figsize=(9, 5.2))
    for dataset, group in all_df.groupby("dataset"):
        group = group.sort_values("compression_ratio")
        plt.plot(
            group["compression_ratio"],
            group["retention_pct"],
            marker="o",
            linewidth=2,
            label=DATASET_LABELS.get(dataset, dataset),
        )
    plt.axhline(100, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Compression ratio")
    plt.ylabel("Retention vs baseline (%)")
    plt.title("Quality Retention vs Baseline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "retention_vs_ratio.png", dpi=180)
    plt.close()


def plot_length_bins(df, output_dir):
    non_all = df[df["length_bin"] != "ALL"].copy()
    for dataset, group in non_all.groupby("dataset"):
        pivot = group.pivot_table(
            index="compression_ratio",
            columns="length_bin",
            values="primary_score",
            aggfunc="first",
        ).sort_index()
        plt.figure(figsize=(9, 5.2))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
        plt.xlabel("Compression ratio")
        plt.ylabel("Primary score")
        plt.title(f"{DATASET_LABELS.get(dataset, dataset)} by Length Bin")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_by_length_bin.png", dpi=180)
        plt.close()


def main():
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir) if args.output_dir else summary_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(summary_dir)
    all_df = add_baseline_deltas(df)

    df.to_csv(output_dir / "summary_rows.csv", index=False)
    all_df.to_csv(output_dir / "overall_with_deltas.csv", index=False)
    with (output_dir / "all_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    write_markdown(all_df, output_dir / "quality_sweep_summary.md")
    plot_primary_scores(all_df, output_dir)
    plot_retention(all_df, output_dir)
    plot_length_bins(df, output_dir)

    print(f"Wrote analysis to {output_dir}")
    print((output_dir / "quality_sweep_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
