#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CHECKPOINT_NAME GPU PORT CHECKPOINT_DIR LOG_DIR" >&2
  exit 64
fi

checkpoint_name=$1
gpu=$2
port=$3
checkpoint_dir=$4
log_dir=$5
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
openpi_dir="$repo_root/external_repos/openpi"

[[ "$checkpoint_name" == "full" || "$checkpoint_name" == "early" ]] || { echo "invalid checkpoint name" >&2; exit 64; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
checkpoint_dir=$(realpath --canonicalize-existing "$checkpoint_dir")
[[ -d "$checkpoint_dir/params" ]] || { echo "missing params: $checkpoint_dir" >&2; exit 1; }
norm_stats="$checkpoint_dir/assets/physical-intelligence/libero/norm_stats.json"
[[ -f "$norm_stats" ]] || { echo "missing norm stats: $norm_stats" >&2; exit 1; }
[[ "$(git -C "$openpi_dir" rev-parse HEAD)" == "650c5b0283a49c42784fb5055a0507da2c6d347d" ]] || { echo "OpenPI revision mismatch" >&2; exit 1; }
expected_norm=$(jq -r ".checkpoints.$checkpoint_name.norm_stats_sha256" "$repo_root/configs/artifacts.json")
actual_norm=$(sha256sum "$norm_stats" | cut -d' ' -f1)
[[ "$actual_norm" == "$expected_norm" ]] || { echo "norm stats hash mismatch" >&2; exit 1; }
mkdir -p "$log_dir"
log_dir=$(realpath --canonicalize-existing "$log_dir")
if [[ -n "$(find "$log_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "log directory must be empty: $log_dir" >&2
  exit 1
fi
exec > >(tee "$log_dir/server.log") 2>&1

printf 'checkpoint=%s gpu=%s port=%s dir=%s norm_stats_sha256=%s\n' \
  "$checkpoint_name" "$gpu" "$port" "$checkpoint_dir" "$actual_norm"
cd "$openpi_dir"
exec env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  uv run scripts/serve_policy.py --port "$port" \
  policy:checkpoint --policy.config pi05_libero --policy.dir "$checkpoint_dir"
