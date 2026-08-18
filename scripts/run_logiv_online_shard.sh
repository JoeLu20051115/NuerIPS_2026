#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU PORT SHARD OUTPUT_ROOT" >&2
  exit 64
fi

gpu=$1
port=$2
shard=$3
output_root=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

case "$shard" in
  0) task_ids="0,3,6,9" ;;
  1) task_ids="1,4,7" ;;
  2) task_ids="2,5,8" ;;
  *) echo "invalid shard: $shard" >&2; exit 64 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $gpu" >&2; exit 64; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 64; }
mkdir -p "$output_root"
output_root=$(realpath --canonicalize-existing "$output_root")

seeds=(7 17)
for seed in "${seeds[@]}"; do
  base_run="logiv-online-base-seed${seed}-shard${shard}"
  "$repo_root/scripts/run_logiv_eval.sh" \
    BASE "$gpu" "$port" "$output_root/$base_run" \
    --run-id "$base_run" \
    --goal-mode METADATA_ASSISTED \
    --deviation-mode ONLINE_TUNING_FINAL \
    --development-only \
    --task-ids "$task_ids" \
    --episode-indices 0:50 \
    --seed "$seed" \
    --no-video

  online_run="logiv-online-seed${seed}-shard${shard}"
  "$repo_root/scripts/run_logiv_eval.sh" \
    LOGIV_ONLINE "$gpu" "$port" "$output_root/$online_run" \
    --run-id "$online_run" \
    --goal-mode METADATA_ASSISTED \
    --deviation-mode ONLINE_TUNING_FINAL \
    --oracle-grounding \
    --development-only \
    --task-ids "$task_ids" \
    --episode-indices 0:50 \
    --seed "$seed" \
    --no-video \
    --prompt-config /repro/configs/logiv/prompts/pi05-subtasks-online-v5.json \
    --prompt-version pi05-subtasks-online-v5 \
    --proposal-config /repro/configs/logiv/libero10-scripted-proposals-online-v1.json \
    --coverage-manifest /repro/configs/logiv/libero10-coverage-online-v1.json \
    --max-action-steps 520 \
    --frontier-followup-steps 5 \
    --frontier-completion-followup-steps 120 \
    --frontier-completion-recovery-only \
    --frontier-recovery-max-consumed-steps 180 \
    --frontier-fallback-after-steps 200 \
    --frontier-fallback-followup-steps 220 \
    --post-stop-grounding-reobservation-steps 20 \
    --post-stop-grounding-reobservation-task-ids 2,3,7,9 \
    --place-effect-confirmation-steps 1 \
    --place-effect-confirmation-task-ids 5 \
    --place-effect-stabilization-steps 10 \
    --place-effect-stabilization-task-ids 5 \
    --access-effect-stabilization-steps 10 \
    --held-target-divergence-confirmation-steps 5 \
    --target-divergence-confirmation-steps 5 \
    --online-confirmations 3 \
    --online-monitor-interval-steps 5 \
    --online-min-intervention-step 120 \
    --online-stall-steps 180 \
    --online-recovery-requires-goal-task-ids 6 \
    --online-stall-requires-handempty-task-ids 8
done
