#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

early_dir="$repo_root/artifacts/checkpoints/pi05_libero_2000"
dataset_dir="$repo_root/artifacts/datasets"
mkdir -p "$early_dir" "$dataset_dir"

uvx --from 'huggingface-hub==0.34.4' hf download brandonyang/openpi-libero-2000 \
  --revision aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e \
  --include _CHECKPOINT_METADATA 'assets/**' 'params/**' \
  --local-dir "$early_dir"

uvx --from 'huggingface-hub==0.34.4' hf download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --revision f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --include 'libero_10/*.hdf5' \
  --local-dir "$dataset_dir"

for cache_dir in "$early_dir/.cache/huggingface" "$dataset_dir/.cache/huggingface"; do
  if [[ -d "$cache_dir" ]]; then
    case "$cache_dir" in
      "$repo_root"/artifacts/*/.cache/huggingface) find "$cache_dir" -depth -delete ;;
      *) echo "refusing unsafe cache path: $cache_dir" >&2; exit 2 ;;
    esac
  fi
done

if [[ -e "$early_dir/train_state" ]]; then
  echo "unexpected train_state payload" >&2
  exit 1
fi

early_files=$(find "$early_dir" -type f | wc -l)
early_bytes=$(find "$early_dir" -type f -printf '%s\n' | awk '{total += $1} END {print total + 0}')
dataset_files=$(find "$dataset_dir/libero_10" -maxdepth 1 -type f -name '*.hdf5' | wc -l)
dataset_bytes=$(find "$dataset_dir/libero_10" -maxdepth 1 -type f -name '*.hdf5' -printf '%s\n' | awk '{total += $1} END {print total + 0}')

[[ "$early_files" -eq 17 ]] || { echo "early file count: $early_files != 17" >&2; exit 1; }
[[ "$early_bytes" -eq 12440616902 ]] || { echo "early bytes: $early_bytes != 12440616902" >&2; exit 1; }
[[ "$dataset_files" -eq 10 ]] || { echo "dataset file count: $dataset_files != 10" >&2; exit 1; }
[[ "$dataset_bytes" -eq 13730608904 ]] || { echo "dataset bytes: $dataset_bytes != 13730608904" >&2; exit 1; }

printf 'early checkpoint: %s files, %s bytes\n' "$early_files" "$early_bytes"
printf 'LIBERO-10 dataset: %s files, %s bytes\n' "$dataset_files" "$dataset_bytes"
