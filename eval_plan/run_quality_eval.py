#!/usr/bin/env python3
"""Run deterministic generation against an OpenAI-compatible SGLang server."""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


MAX_TOKENS_BY_DATASET = {
    "qmsum": 1024,
    "narrativeqa": 128,
    "repobench-p": 256,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--compression-ratio", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_done_ids(path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["config"], row["dataset"], row["sample_id"]))
    return done


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = args.model or client.models.list().data[0].id
    samples = list(load_jsonl(args.samples))
    if args.limit is not None:
        samples = samples[: args.limit]

    done = load_done_ids(output_path)
    with output_path.open("a", encoding="utf-8") as out:
        for sample in tqdm(samples, desc=args.config_name):
            key = (args.config_name, sample["dataset"], sample["sample_id"])
            if key in done:
                continue
            max_tokens = MAX_TOKENS_BY_DATASET.get(
                sample["dataset"], args.default_max_tokens
            )
            started = time.time()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": sample["prompt"]}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content or ""
                error = None
            except Exception as exc:
                output = ""
                error = repr(exc)

            row = dict(sample)
            row.update(
                {
                    "config": args.config_name,
                    "compression_ratio": args.compression_ratio,
                    "model": model,
                    "max_new_tokens": max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "output": output,
                    "error": error,
                    "wall_time_seconds": time.time() - started,
                }
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
