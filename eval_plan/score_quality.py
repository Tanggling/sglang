#!/usr/bin/env python3
"""Score quality predictions for the LongBench subset sweep."""

import argparse
import json
import re
import string
from collections import defaultdict


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for tok in ref_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return float(normalize_text(prediction) == normalize_text(reference))


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction, reference):
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()
    if not pred or not ref:
        return 0.0
    lcs = lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_over_refs(metric_fn, prediction, references):
    if not references:
        return 0.0
    return max(metric_fn(prediction, ref) for ref in references)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            refs = row.get("references", [])
            pred = row.get("output", "")
            row["f1"] = best_over_refs(token_f1, pred, refs)
            row["em"] = best_over_refs(exact_match, pred, refs)
            row["rouge_l"] = best_over_refs(rouge_l, pred, refs)
            rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["dataset"], "ALL")].append(row)
        groups[(row["config"], row["dataset"], row["length_bin"])].append(row)

    summary = []
    for (config, dataset, length_bin), group in sorted(groups.items()):
        metric = "rouge_l" if dataset == "qmsum" else "f1"
        summary.append(
            {
                "config": config,
                "compression_ratio": group[0].get("compression_ratio"),
                "dataset": dataset,
                "length_bin": length_bin,
                "n": len(group),
                "primary_metric": metric,
                "primary_score": sum(x[metric] for x in group) / len(group),
                "rouge_l": sum(x["rouge_l"] for x in group) / len(group),
                "f1": sum(x["f1"] for x in group) / len(group),
                "em": sum(x["em"] for x in group) / len(group),
                "error_count": sum(1 for x in group if x.get("error")),
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
