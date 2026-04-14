#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_INDEX="${1:-1}"
FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-100000}"
UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-20}"
POLL_INTERVAL_SEC="${GPU_POLL_INTERVAL_SEC:-60}"

PYTHON_BIN="${ROOT_DIR}/external_repos/DreamDojo/.venv/bin/python"
EVAL_SCRIPT="${ROOT_DIR}/scripts/eval/run_robotwin_lingbot_smoke.py"
OUTPUT_JSON="${ROOT_DIR}/evaluation_results_dualsystem/robotwin_lingbot_full_3way_gpu${GPU_INDEX}.json"
SAVE_ROOT="${ROOT_DIR}/evaluation_results_dualsystem/robotwin_lingbot_full_3way_gpu${GPU_INDEX}_artifacts"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "Waiting for GPU${GPU_INDEX} to become idle..."
while true; do
  free_mib="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  util_pct="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${timestamp}] GPU${GPU_INDEX} free=${free_mib}MiB util=${util_pct}%"

  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && [[ "${util_pct}" =~ ^[0-9]+$ ]] && (( free_mib >= FREE_THRESHOLD_MIB )) && (( util_pct <= UTIL_THRESHOLD )); then
    break
  fi

  sleep "${POLL_INTERVAL_SEC}"
done

echo "Starting full LingBot evaluation on GPU${GPU_INDEX}..."
exec env TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --modes task_token_only,dual_llm,llm_val \
  --episodes-per-level 100 \
  --max-step-count 10000 \
  --judge-model gpt-5.4 \
  --planner-model gpt-5.4 \
  --output-json "${OUTPUT_JSON}" \
  --save-root "${SAVE_ROOT}" \
  --resume \
  --no-save-video
