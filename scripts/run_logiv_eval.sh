#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU PORT OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
  exit 64
fi

gpu=$1
port=$2
output_dir=$3
shift 3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
openpi_dir="$repo_root/external_repos/openpi"
val_dir="$repo_root/artifacts/tools/val-ubuntu22"
manifest="$repo_root/src/pi05_libero_repro/logiv/config/val-build.json"
image=pi05-libero-eval:650c5b0

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is required" >&2; exit 1; }
docker image inspect "$image" >/dev/null
[[ -x "$val_dir/Validate" && -f "$val_dir/libVAL.so" ]] || {
  echo "missing compatible VAL build; run scripts/build_val_for_libero.sh" >&2
  exit 1
}
[[ "$(sha256sum "$val_dir/Validate" | cut -d' ' -f1)" == "$(jq -r .validate_sha256 "$manifest")" ]] || {
  echo "Validate hash mismatch" >&2
  exit 1
}
mkdir -p "$output_dir"
output_dir=$(realpath --canonicalize-existing "$output_dir")

exec docker run --rm --network host --gpus "device=$gpu" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/cache \
  -e MUJOCO_GL=egl \
  -e LD_LIBRARY_PATH=/val \
  -e PYTHONPATH=/repro/src:/app:/app/packages/openpi-client/src:/app/third_party/libero \
  -e OPENAI_API_KEY \
  -v "$repo_root:/repro:ro" \
  -v "$openpi_dir:/app:ro" \
  -v "$val_dir:/val:ro" \
  -v "$output_dir:/outputs" \
  "$image" bash -lc 'source /.venv/bin/activate && exec python /repro/scripts/eval_logiv_libero.py "$@"' bash \
  --run-id origin \
  --method-arm FULL_LOGIV \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode ORIGIN \
  --perception-backend gpt4o \
  --prompt-locked \
  --host 127.0.0.1 \
  --port "$port" \
  --output-dir /outputs \
  "$@"
