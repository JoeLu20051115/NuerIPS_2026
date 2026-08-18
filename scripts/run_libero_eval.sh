#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 CHECKPOINT_NAME GPU PORT OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
  exit 64
fi

checkpoint_name=$1
gpu=$2
port=$3
output_dir=$4
shift 4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
openpi_dir="$repo_root/external_repos/openpi"
image=pi05-libero-eval:650c5b0

[[ "$checkpoint_name" == "full" || "$checkpoint_name" == "early" ]] || { echo "invalid checkpoint name" >&2; exit 64; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
docker image inspect "$image" >/dev/null
[[ -f "$repo_root/artifacts/manifests/$checkpoint_name-checkpoint.json" ]] || { echo "missing checkpoint manifest" >&2; exit 1; }
mkdir -p "$output_dir"
output_dir=$(realpath --canonicalize-existing "$output_dir")
if [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be empty: $output_dir" >&2
  exit 1
fi

exec docker run --rm --network host --gpus "device=$gpu" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/cache \
  -e MUJOCO_GL=egl \
  -e PYTHONPATH=/repro/src:/app:/app/packages/openpi-client/src:/app/third_party/libero \
  -v "$repo_root:/repro:ro" \
  -v "$openpi_dir:/app:ro" \
  -v "$output_dir:/outputs" \
  "$image" bash -lc 'source /.venv/bin/activate && exec python /repro/scripts/eval_libero.py "$@"' bash \
  --checkpoint-name "$checkpoint_name" --host 127.0.0.1 --port "$port" --output-dir /outputs "$@"

