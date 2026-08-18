#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU PORT CHECKPOINT_DIR" >&2
  exit 64
fi

gpu=$1
port=$2
checkpoint_dir=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
openpi_dir="$repo_root/external_repos/openpi"

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
checkpoint_dir=$(realpath --canonicalize-existing "$checkpoint_dir")
[[ -d "$checkpoint_dir/params" ]] || { echo "missing params: $checkpoint_dir" >&2; exit 1; }
norm_stats="$checkpoint_dir/assets/physical-intelligence/libero/norm_stats.json"
[[ -f "$norm_stats" ]] || { echo "missing norm stats: $norm_stats" >&2; exit 1; }
[[ "$(git -C "$openpi_dir" rev-parse HEAD)" == "650c5b0283a49c42784fb5055a0507da2c6d347d" ]] || { echo "OpenPI revision mismatch" >&2; exit 1; }
actual_norm=$(sha256sum "$norm_stats" | cut -d' ' -f1)

printf 'gpu=%s port=%s checkpoint=%s norm_stats_sha256=%s\n' \
  "$gpu" "$port" "$checkpoint_dir" "$actual_norm"
cd "$openpi_dir"
exec env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  uv run "$repo_root/scripts/serve_policy.py" --port "$port" \
  --policy-config pi05_libero --policy-dir "$checkpoint_dir"
