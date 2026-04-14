#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/external_repos/DreamDojo/.venv/bin/python}"
DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${REPO_ROOT}/external_repos/DreamDojo}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DREAMDOJO_ROOT}/checkpoints/2B_AgiBot_post-train/2B_AgiBot_post-train}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/iter_000050000/model_ema_bf16.pt}"
SHARED_META="${SHARED_META:-${DREAMDOJO_ROOT}/shared_meta}"
DREAMDOJO_TIMEOUT="${DREAMDOJO_TIMEOUT:-1800}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/evaluation_results_dualsystem/dreamdojo_full300}"

MANIFEST_L1="${MANIFEST_L1:-${REPO_ROOT}/data/final_data1/Agi_L1_150/meta/manifest.json}"
MANIFEST_L3="${MANIFEST_L3:-${REPO_ROOT}/data/final_data1/Agi_L3_150/meta/manifest.json}"

mkdir -p "${OUT_DIR}"
EXTRA_ARGS=("$@")

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval/run_agibot_dreamdojo_3way_compare.py" \
  --dataset-name "AgiBot300" \
  --manifest-path "${MANIFEST_L1}" \
  --manifest-path "${MANIFEST_L3}" \
  --num-episodes 300 \
  --modes task_token_only,dual_llm,val_llm \
  --eval-steps 49 \
  --num-frames 49 \
  --dreamdojo-root "${DREAMDOJO_ROOT}" \
  --dreamdojo-python "${PYTHON_BIN}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --shared-meta "${SHARED_META}" \
  --dreamdojo-timeout "${DREAMDOJO_TIMEOUT}" \
  --output-json "${OUT_DIR}/agibot300_dreamdojo_3way.json" \
  --log "${OUT_DIR}/agibot300_dreamdojo_3way.log" \
  --save-root "${OUT_DIR}/agibot300_videos" \
  --tmp-root "${REPO_ROOT}/tmp/agibot300_dreamdojo_3way" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval/summarize_agibot_dreamdojo_results.py" \
  --results-path "${OUT_DIR}/agibot300_dreamdojo_3way.json" \
  --dataset-name "AgiBot300" \
  --output-log "${OUT_DIR}/agibot300_dreamdojo_summary.log"
