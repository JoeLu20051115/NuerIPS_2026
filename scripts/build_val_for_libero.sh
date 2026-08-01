#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
manifest="$repo_root/configs/logiv/val-libero-build.json"
image=$(jq -r .builder_image "$manifest")
revision=$(jq -r .source_revision "$manifest")
source_repository=$(jq -r .source_repository "$manifest")
output="$repo_root/artifacts/tools/val-ubuntu22"

docker image inspect "$image" >/dev/null
if [[ -e "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite nonempty VAL output: $output" >&2
  exit 1
fi
mkdir -p "$output"
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
git -C "$temporary" init -q VAL
git -C "$temporary/VAL" remote add origin "$source_repository"
git -C "$temporary/VAL" fetch -q --depth 1 origin "$revision"
git -C "$temporary/VAL" checkout -q --detach FETCH_HEAD

docker run --rm \
  -v "$temporary/VAL:/src:ro" \
  -v "$output:/out" \
  "$image" bash -lc '
    set -euo pipefail
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake flex bison >/out/apt.log 2>&1
    cp -a /src /tmp/VAL
    cd /tmp/VAL
    bash scripts/linux/build_linux64.sh all Release >/out/build.log 2>&1
    cp build/linux64/Release/bin/Validate /out/Validate
    cp build/linux64/Release/bin/libVAL.so /out/libVAL.so
  '

expected_validate=$(jq -r .validate_sha256 "$manifest")
expected_libval=$(jq -r .libval_sha256 "$manifest")
actual_validate=$(sha256sum "$output/Validate" | cut -d' ' -f1)
actual_libval=$(sha256sum "$output/libVAL.so" | cut -d' ' -f1)
[[ "$actual_validate" == "$expected_validate" ]] || { echo "Validate hash mismatch: $actual_validate" >&2; exit 1; }
[[ "$actual_libval" == "$expected_libval" ]] || { echo "libVAL hash mismatch: $actual_libval" >&2; exit 1; }
docker run --rm -e LD_LIBRARY_PATH=/val -v "$output:/val:ro" "$image" \
  bash -lc '/val/Validate -h 2>&1 | grep -F "Version 4:"'
