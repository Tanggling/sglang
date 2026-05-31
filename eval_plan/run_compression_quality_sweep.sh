#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-../../../autodl-tmp/Qwen2.5-7B-Instruct/}
PORT=${PORT:-30000}
SAMPLES=${SAMPLES:-./data/quality_samples.jsonl}
OUT_DIR=${OUT_DIR:-./results/quality_sweep}
PYTHON=${PYTHON:-python}

mkdir -p "${OUT_DIR}"

wait_server() {
  local url="http://127.0.0.1:${PORT}/v1/models"
  for _ in $(seq 1 240); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_one() {
  local name=$1
  local backend=$2
  local ratio=$3
  local mem_fraction=$4
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local log_file="${OUT_DIR}/${name}.server.log"

  echo "Starting ${name}: backend=${backend}, ratio=${ratio}"
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

  cleanup() {
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  }
  trap cleanup RETURN

  wait_server
  ${PYTHON} ./run_quality_eval.py \
    --samples "${SAMPLES}" \
    --output "${pred_file}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --config-name "${name}" \
    --compression-ratio "${ratio}" \
    --temperature 0 \
    --top-p 1

  cleanup
  trap - RETURN
  ${PYTHON} ./score_quality.py \
    --predictions "${pred_file}" \
    --output "${OUT_DIR}/${name}.summary.json"
}

if [[ ! -f "${SAMPLES}" ]]; then
  ${PYTHON} ./prepare_longbench_quality_samples.py \
    --model-path "${MODEL_PATH}" \
    --output "${SAMPLES}" \
    --samples-per-dataset 50
fi

run_one baseline_fa3 fa3 0.0 0.55

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  safe_ratio=${ratio/./p}
  run_one "compressed_${safe_ratio}" compressed_fa3 "${ratio}" 0.55
done

echo "All results are under ${OUT_DIR}"
