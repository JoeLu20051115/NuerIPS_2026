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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/evaluation_results_dualsystem/dreamdojo_selected130}"

mkdir -p "${OUT_DIR}"
EXTRA_ARGS=("$@")

COMMON_ARGS=(
  --modes task_token_only,dual_llm,val_llm
  --eval-steps 49
  --num-frames 49
  --dreamdojo-root "${DREAMDOJO_ROOT}"
  --dreamdojo-python "${PYTHON_BIN}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --shared-meta "${SHARED_META}"
  --dreamdojo-timeout "${DREAMDOJO_TIMEOUT}"
)

run_split() {
  local dataset_name="$1"
  local manifest_path="$2"
  shift 2
  local output_json="${OUT_DIR}/${dataset_name}_dreamdojo_3way.json"
  local output_log="${OUT_DIR}/${dataset_name}_dreamdojo_3way.log"
  local save_root="${OUT_DIR}/${dataset_name}_videos"
  local tmp_root="${REPO_ROOT}/tmp/${dataset_name}_dreamdojo_3way"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval/run_agibot_dreamdojo_3way_compare.py" \
    --dataset-name "${dataset_name}" \
    --manifest-path "${manifest_path}" \
    --output-json "${output_json}" \
    --log "${output_log}" \
    --save-root "${save_root}" \
    --tmp-root "${tmp_root}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval/summarize_agibot_dreamdojo_results.py" \
    --results-path "${output_json}" \
    --dataset-name "${dataset_name}" \
    --output-log "${OUT_DIR}/${dataset_name}_dreamdojo_summary.log"
}

run_split "Agi_L1_130" "${REPO_ROOT}/data/Agi_L1_130/meta/manifest.json"
run_split "Agi_L3_130" "${REPO_ROOT}/data/Agi_L3_130/meta/manifest.json"
