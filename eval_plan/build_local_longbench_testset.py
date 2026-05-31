#!/usr/bin/env python3
"""Build a type- and length-balanced local LongBench test set.

The script reads /root/autodl-tmp/dataset/THUDM_LongBench/data.zip and writes a
JSONL file with a unified schema for quality evaluation.
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path


TASKS = [
    ("qmsum", "summarization"),
    ("narrativeqa", "long_qa"),
    ("repobench-p", "code_repo"),
]

LENGTH_BINS = [
    (0, 8192, "000k_008k"),
    (8192, 16384, "008k_016k"),
    (16384, 32768, "016k_032k"),
    (32768, 65536, "032k_064k"),
    (65536, 131072, "064k_128k"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--longbench-zip",
        default="/root/autodl-tmp/dataset/THUDM_LongBench/data.zip",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/dataset/eval_samples/longbench_quality_300.jsonl",
    )
    parser.add_argument("--total-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=131072)
    return parser.parse_args()


def length_bin(length):
    for lo, hi, name in LENGTH_BINS:
        if lo <= length < hi:
            return name
    return None


def build_prompt(dataset, context, question):
    context = context.strip()
    question = question.strip()
    if dataset == "qmsum":
        return (
            "You are given a meeting transcript and a query. Write a concise, factual answer "
            "based only on the transcript.\n\n"
            f"Transcript:\n{context}\n\nQuery:\n{question}\n\nAnswer:"
        )
    if dataset == "narrativeqa":
        return (
            "Read the following story or document, then answer the question with a short answer.\n\n"
            f"Document:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    if dataset == "repobench-p":
        return (
            "Complete the following code-repository task using the provided context. "
            "Return only the missing code or final answer.\n\n"
            f"Repository context:\n{context}\n\nTask:\n{question}\n\nAnswer:"
        )
    return f"{context}\n\n{question}\n\nAnswer:"


def load_task(zip_path, dataset, task_type, max_length):
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"data/{dataset}.jsonl") as f:
            for line in f:
                row = json.loads(line)
                length = int(row.get("length", 0))
                bin_name = length_bin(length)
                if bin_name is None or length > max_length:
                    continue
                samples.append(
                    {
                        "sample_id": str(row.get("_id")),
                        "dataset": dataset,
                        "task_type": task_type,
                        "length": length,
                        "length_bin": bin_name,
                        "language": row.get("language"),
                        "prompt": build_prompt(
                            dataset,
                            str(row.get("context", "")),
                            str(row.get("input", "")),
                        ),
                        "references": row.get("answers", []),
                    }
                )
    return samples


def pick_balanced(samples, target, rng):
    by_bin = defaultdict(list)
    for sample in samples:
        by_bin[sample["length_bin"]].append(sample)

    non_empty_bins = [name for _, _, name in LENGTH_BINS if by_bin[name]]
    base = target // len(non_empty_bins)
    rem = target % len(non_empty_bins)
    selected = []
    leftovers = []

    for idx, bin_name in enumerate(non_empty_bins):
        bucket = list(by_bin[bin_name])
        rng.shuffle(bucket)
        want = base + (1 if idx < rem else 0)
        take = min(want, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])

    rng.shuffle(selected)
    return selected[:target], by_bin


def main():
    args = parse_args()
    if args.total_samples < len(TASKS):
        raise ValueError("total-samples must be at least the number of tasks")

    rng = random.Random(args.seed)
    per_task = args.total_samples // len(TASKS)
    remainder = args.total_samples % len(TASKS)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_selected = []
    summary = []
    for idx, (dataset, task_type) in enumerate(TASKS):
        target = per_task + (1 if idx < remainder else 0)
        samples = load_task(args.longbench_zip, dataset, task_type, args.max_length)
        selected, by_bin = pick_balanced(samples, target, rng)
        all_selected.extend(selected)

        for _, _, bin_name in LENGTH_BINS:
            selected_count = sum(1 for x in selected if x["length_bin"] == bin_name)
            summary.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "length_bin": bin_name,
                    "available": len(by_bin[bin_name]),
                    "selected": selected_count,
                }
            )

    rng.shuffle(all_selected)
    with output.open("w", encoding="utf-8") as f:
        for sample in all_selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_selected": len(all_selected),
                "target": args.total_samples,
                "tasks": [dataset for dataset, _ in TASKS],
                "length_bins": [name for _, _, name in LENGTH_BINS],
                "distribution": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(all_selected)} samples to {output}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
