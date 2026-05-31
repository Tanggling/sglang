#!/usr/bin/env python3
"""Convert existing predictions to LongBench official format and evaluate.

Input:  quality_sweep_300/*.jsonl from run_quality_eval.py
Output: LongBench/pred/<config>/*.jsonl plus official result.json files and a
        merged CSV/JSON summary.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DATASETS = ("qmsum", "narrativeqa", "repobench-p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        default="/root/autodl-tmp/dataset/eval_results/quality_sweep_300",
    )
    parser.add_argument(
        "--longbench-dir",
        default="/root/autodl-tmp/dataset/eval_script/LongBench/LongBench",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def iter_prediction_files(pred_dir, configs):
    pred_dir = Path(pred_dir)
    if configs:
        names = [x.strip() for x in configs.split(",") if x.strip()]
        for name in names:
            path = pred_dir / f"{name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            yield path
    else:
        for path in sorted(pred_dir.glob("*.jsonl")):
            yield path


def convert_file(pred_file, official_pred_root):
    config = pred_file.stem
    rows_by_dataset = defaultdict(list)
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = row["dataset"]
            if dataset not in DATASETS:
                continue
            rows_by_dataset[dataset].append(
                {
                    "pred": row.get("output", ""),
                    "answers": row.get("references", []),
                    "all_classes": row.get("all_classes"),
                    "length": row.get("length", row.get("input_tokens", 0)),
                }
            )

    config_dir = official_pred_root / config
    config_dir.mkdir(parents=True, exist_ok=True)
    for dataset, rows in rows_by_dataset.items():
        out_path = config_dir / f"{dataset}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return config, {dataset: len(rows) for dataset, rows in rows_by_dataset.items()}


def run_official_eval(longbench_dir, config):
    cmd = [sys.executable, "eval.py", "--model", config]
    subprocess.run(cmd, cwd=longbench_dir, check=True)
    result_path = longbench_dir / "pred" / config / "result.json"
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio_from_config(config):
    if config == "baseline_fa3":
        return 0.0
    if config.startswith("compressed_"):
        return float(config.removeprefix("compressed_").replace("p", "."))
    return None


def write_summary(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "longbench_official_scores.json"
    csv_path = output_dir / "longbench_official_scores.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = []
    for config, scores in sorted(results.items(), key=lambda kv: ratio_from_config(kv[0]) or 0):
        ratio = ratio_from_config(config)
        for dataset, score in scores.items():
            rows.append(
                {
                    "config": config,
                    "compression_ratio": ratio,
                    "dataset": dataset,
                    "official_score": score,
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config", "compression_ratio", "dataset", "official_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    longbench_dir = Path(args.longbench_dir)
    official_pred_root = longbench_dir / "pred"
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir / "longbench_official"

    converted = {}
    for pred_file in iter_prediction_files(pred_dir, args.configs):
        config, counts = convert_file(pred_file, official_pred_root)
        converted[config] = counts

    print("Converted predictions:")
    print(json.dumps(converted, ensure_ascii=False, indent=2))

    results = {}
    if not args.skip_eval:
        for config in sorted(converted, key=lambda x: ratio_from_config(x) or 0):
            print(f"Running official LongBench eval for {config}...")
            results[config] = run_official_eval(longbench_dir, config)
    else:
        for config in converted:
            result_path = official_pred_root / config / "result.json"
            if result_path.exists():
                with result_path.open("r", encoding="utf-8") as f:
                    results[config] = json.load(f)

    json_path, csv_path = write_summary(results, output_dir)
    print(f"Wrote official score JSON: {json_path}")
    print(f"Wrote official score CSV:  {csv_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
