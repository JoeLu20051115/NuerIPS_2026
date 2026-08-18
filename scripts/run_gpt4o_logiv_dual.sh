#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU_A PORT_A GPU_B PORT_B OUTPUT_ROOT" >&2
  exit 64
fi

gpu_a=$1
port_a=$2
gpu_b=$3
port_b=$4
output_root=$5
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

[[ -n "${OPENAI_API_KEY:-}" ]] || {
  echo "OPENAI_API_KEY is required" >&2
  exit 1
}
for value in "$gpu_a" "$port_a" "$gpu_b" "$port_b"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "GPU and port values must be integers" >&2; exit 64; }
done
mkdir -p "$output_root"
output_root=$(realpath --canonicalize-existing "$output_root")

run_shard() {
  local gpu=$1
  local port=$2
  local shard=$3
  local task_ids=$4
  local seed base_run online_run
  for seed in 7 17; do
    base_run="gpt4o-logiv-base-seed${seed}-shard${shard}"
    "$repo_root/scripts/run_logiv_eval.sh" \
      BASE "$gpu" "$port" "$output_root/$base_run" \
      --run-id "$base_run" \
      --goal-mode METADATA_ASSISTED \
      --deviation-mode GPT4O_LOGIV_TUNING \
      --development-only \
      --task-ids "$task_ids" \
      --episode-indices 0:50 \
      --seed "$seed" \
      --no-video

    online_run="gpt4o-logiv-seed${seed}-shard${shard}"
    "$repo_root/scripts/run_logiv_eval.sh" \
      LOGIV_ONLINE "$gpu" "$port" "$output_root/$online_run" \
      --run-id "$online_run" \
      --goal-mode METADATA_ASSISTED \
      --deviation-mode GPT4O_LOGIV_TUNING \
      --perception-backend gpt4o \
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
}

run_shard "$gpu_a" "$port_a" 0 "0,2,4,6,8" \
  >"$output_root/shard0.log" 2>&1 &
pid_a=$!
run_shard "$gpu_b" "$port_b" 1 "1,3,5,7,9" \
  >"$output_root/shard1.log" 2>&1 &
pid_b=$!

status=0
wait "$pid_a" || status=$?
wait "$pid_b" || status=$?
exit "$status"
