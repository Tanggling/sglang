#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
SAMPLES=${SAMPLES:-/root/autodl-tmp/dataset/eval_samples/longbench_quality_300.jsonl}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/dataset/eval_results/quality_sweep_300}
PYTHON=${PYTHON:-python}
RUN_BASELINE=${RUN_BASELINE:-1}

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SAMPLES}" ]]; then
  echo "Samples file not found: ${SAMPLES}" >&2
  echo "Build it first with:" >&2
  echo "  python eval_plan/build_local_longbench_testset.py --total-samples 300 --output ${SAMPLES}" >&2
  exit 1
fi

wait_server() {
  local url="http://127.0.0.1:${PORT}/v1/models"
  for _ in $(seq 1 300); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Server did not become ready: ${url}" >&2
  return 1
}

kill_existing_servers() {
  local pids
  pids=$(pgrep -f "sglang.launch_server" || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "Killing existing SGLang server processes: ${pids}"
  kill ${pids} >/dev/null 2>&1 || true
  sleep 10

  pids=$(pgrep -f "sglang.launch_server" || true)
  if [[ -n "${pids}" ]]; then
    echo "Force killing lingering SGLang server processes: ${pids}"
    kill -9 ${pids} >/dev/null 2>&1 || true
    sleep 2
  fi
}

stop_server() {
  local pid=$1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    kill -9 "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup_all() {
  kill_existing_servers
}

trap cleanup_all EXIT INT TERM

run_one() {
  local name=$1
  local backend=$2
  local ratio=$3
  local mem_fraction=$4
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local summary_file="${OUT_DIR}/${name}.summary.json"
  local log_file="${OUT_DIR}/${name}.server.log"

  echo ""
  echo "==== ${name} ===="
  echo "backend=${backend}, ratio=${ratio}, samples=${SAMPLES}"
  echo "predictions=${pred_file}"

  kill_existing_servers

  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 USE_PINNED_MEMORY=0 \
  ${PYTHON} -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --port "${PORT}" \
    --tp 1 \
    --disable-cuda-graph \
    --mem-fraction-static "${mem_fraction}" \
    --disable-radix-cache \
    --disable-context-len-check \
    --page-size 1 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 131072 \
    --context-length 131072 \
    --attention-backend "${backend}" \
    --kv-compression-ratio "${ratio}" \
    >"${log_file}" 2>&1 &
  local server_pid=$!

  trap 'stop_server "${server_pid}"; cleanup_all' RETURN
  wait_server

  ${PYTHON} eval_plan/run_quality_eval.py \
    --samples "${SAMPLES}" \
    --output "${pred_file}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --config-name "${name}" \
    --compression-ratio "${ratio}" \
    --temperature 0 \
    --top-p 1

  ${PYTHON} eval_plan/score_quality.py \
    --predictions "${pred_file}" \
    --output "${summary_file}"

  stop_server "${server_pid}"
  kill_existing_servers
  trap cleanup_all EXIT INT TERM
}

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_one baseline_fa3 fa3 0.0 0.55
else
  echo "Skipping baseline_fa3 because RUN_BASELINE=${RUN_BASELINE}"
fi

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  safe_ratio=${ratio/./p}
  run_one "compressed_${safe_ratio}" compressed_fa3 "${ratio}" 0.55
done

${PYTHON} - <<'PY' "${OUT_DIR}"
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.summary.json")):
    with path.open("r", encoding="utf-8") as f:
        rows.extend(json.load(f))

overall = [r for r in rows if r["length_bin"] == "ALL"]
with (out_dir / "all_summaries.json").open("w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("\nOverall summary:")
print(f"{'config':<18} {'dataset':<14} {'metric':<10} {'score':>10} {'n':>5} {'errors':>7}")
print("-" * 72)
for r in overall:
    print(
        f"{r['config']:<18} {r['dataset']:<14} {r['primary_metric']:<10} "
        f"{r['primary_score']:>10.4f} {r['n']:>5} {r['error_count']:>7}"
    )
PY

echo ""
echo "Sweep complete. Results: ${OUT_DIR}"
