#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 METHOD_ARM GPU PORT OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
  exit 64
fi

method_arm=$1
gpu=$2
port=$3
output_dir=$4
shift 4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
openpi_dir="$repo_root/external_repos/openpi"
val_dir="$repo_root/artifacts/tools/val-ubuntu22"
manifest="$repo_root/configs/logiv/origin/val-build.json"
checkpoint_manifest="$repo_root/artifacts/manifests/full-checkpoint.json"
image=pi05-libero-eval:650c5b0

case "$method_arm" in
  BASE|SHADOW_LOGIV|LOGIV_ONLINE|LOGIV_REPAIR_OVERLAY|STAGE_ONLY|GRAPH_WITHOUT_VAL|VAL_WITHOUT_LOCALIZED_REPAIR|FULL_LOGIV) ;;
  *) echo "invalid method arm: $method_arm" >&2; exit 64 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
docker image inspect "$image" >/dev/null
[[ -f "$checkpoint_manifest" ]] || {
  echo "missing full checkpoint manifest" >&2
  exit 1
}
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

openai_env=()
for evaluator_arg in "$@"; do
  if [[ "$evaluator_arg" == "gpt4o" ]]; then
    [[ -n "${OPENAI_API_KEY:-}" ]] || {
      echo "OPENAI_API_KEY is required for the gpt4o backend" >&2
      exit 1
    }
    openai_env=(-e OPENAI_API_KEY)
    break
  fi
done

exec docker run --rm --network host --gpus "device=$gpu" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/cache \
  -e MUJOCO_GL=egl \
  -e LD_LIBRARY_PATH=/val \
  -e PYTHONPATH=/repro/src:/app:/app/packages/openpi-client/src:/app/third_party/libero \
  "${openai_env[@]}" \
  -v "$repo_root:/repro:ro" \
  -v "$openpi_dir:/app:ro" \
  -v "$val_dir:/val:ro" \
  -v "$output_dir:/outputs" \
  "$image" bash -lc 'source /.venv/bin/activate && exec python /repro/scripts/eval_logiv_libero.py "$@"' bash \
  --method-arm "$method_arm" --host 127.0.0.1 --port "$port" --output-dir /outputs "$@"
