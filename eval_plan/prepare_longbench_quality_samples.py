#!/usr/bin/env python3
"""Prepare length-balanced LongBench quality-evaluation samples.

This uses THUDM/LongBench subsets with a unified schema:
  - qmsum
  - narrativeqa
  - repobench-p

The output JSONL is consumed by run_quality_eval.py.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_DATASETS = ["qmsum", "narrativeqa", "repobench-p"]
LENGTH_BINS = [
    (0, 8192, "000k_008k"),
    (8192, 16384, "008k_016k"),
    (16384, 32768, "016k_032k"),
    (32768, 65536, "032k_064k"),
    (65536, 131072, "064k_128k"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="eval_plan/data/quality_samples.jsonl")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--samples-per-dataset", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def normalize_answers(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def build_prompt(dataset_name, row):
    context = str(row.get("context", "")).strip()
    question = str(row.get("input", row.get("question", ""))).strip()

    if dataset_name == "qmsum":
        return (
            "You are given a meeting transcript and a query. Write a concise, factual answer "
            "based only on the transcript.\n\n"
            f"Transcript:\n{context}\n\nQuery:\n{question}\n\nAnswer:"
        )
    if dataset_name == "narrativeqa":
        return (
            "Read the following story or document, then answer the question with a short answer.\n\n"
            f"Document:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    if dataset_name == "repobench-p":
        return (
            "Complete or answer the following code-repository task using the provided context. "
            "Return only the final answer.\n\n"
            f"Repository context:\n{context}\n\nTask:\n{question}\n\nAnswer:"
        )
    return f"{context}\n\n{question}\n\nAnswer:"


def length_bin(num_tokens):
    for lo, hi, name in LENGTH_BINS:
        if lo <= num_tokens < hi:
            return name
    return None


def allocate_from_bins(items_by_bin, target_count, rng):
    non_empty_bins = [name for _, _, name in LENGTH_BINS if items_by_bin.get(name)]
    if not non_empty_bins:
        return []


    base = target_count // len(non_empty_bins)
    rem = target_count % len(non_empty_bins)
    selected = []
    leftovers = []

    for idx, bin_name in enumerate(non_empty_bins):
        items = list(items_by_bin[bin_name])
        rng.shuffle(items)
        want = base + (1 if idx < rem else 0)
        take = min(want, len(items))
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    if len(selected) < target_count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])

    rng.shuffle(selected)
    return selected[:target_count]


def main():
    args = parse_args()
    if int(datasets.__version__.split(".", maxsplit=1)[0]) >= 4:
        raise RuntimeError(
            "THUDM/LongBench uses a HuggingFace dataset script, but datasets>=4 "
            "no longer supports dataset scripts. Please install datasets<4, for "
            "example: pip install 'datasets==3.6.0'"
        )

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_selected = []
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    for dataset_name in dataset_names:
        print(f"Loading THUDM/LongBench:{dataset_name} split={args.split}")
        ds = load_dataset(
            "THUDM/LongBench",
            dataset_name,
            split=args.split,
            trust_remote_code=True,
        )
        items_by_bin = defaultdict(list)

        for idx, row in enumerate(tqdm(ds, desc=f"tokenizing {dataset_name}")):
            prompt = build_prompt(dataset_name, row)
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            bin_name = length_bin(input_tokens)
            if bin_name is None or input_tokens > args.max_input_tokens:
                continue

            sample = {
                "sample_id": str(row.get("_id", f"{dataset_name}_{idx}")),
                "dataset": dataset_name,
                "length_bin": bin_name,
                "input_tokens": input_tokens,
                "prompt": prompt,
                "references": normalize_answers(row.get("answers", row.get("answer"))),
            }
            items_by_bin[bin_name].append(sample)

        selected = allocate_from_bins(items_by_bin, args.samples_per_dataset, rng)
        print(f"Selected {len(selected)} samples for {dataset_name}")
        for _, _, bin_name in LENGTH_BINS:
            print(f"  {bin_name}: available={len(items_by_bin[bin_name])}")
        all_selected.extend(selected)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in all_selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_selected)} samples to {output_path}")


if __name__ == "__main__":
    main()
