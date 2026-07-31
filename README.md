# π₀.₅ LIBERO-Long Reproduction

This repository evaluates the full and 2,000-step π₀.₅ checkpoints on the ten long-horizon tasks in the official `libero_10` suite. Each primary result contains 50 valid trials per task (500 per checkpoint) using OpenPI's camera, state, control, reset, and native success semantics.

## Locked inputs

- OpenPI: `650c5b0283a49c42784fb5055a0507da2c6d347d`
- LIBERO: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- robosuite 1.4.1 and MuJoCo 3.2.3 in the official evaluator image
- Full checkpoint: `gs://openpi-assets/checkpoints/pi05_libero/`
- Early checkpoint: `brandonyang/openpi-libero-2000@aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e`
- Dataset: `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`, `libero_10` only

The raw HDF5 demonstrations are downloaded for provenance; evaluation uses LIBERO's pinned BDDL files and `.init` states.

## Setup and public assets

```bash
git submodule update --init external_repos/openpi
git -C external_repos/openpi submodule update --init third_party/libero
uv sync --dev
bash scripts/download_public_assets.sh
```

Verify all payloads before a run:

```bash
uv run python scripts/verify_artifacts.py verify \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  artifacts/manifests/full-checkpoint.json
uv run python scripts/verify_artifacts.py verify \
  artifacts/checkpoints/pi05_libero_2000 \
  artifacts/manifests/early-checkpoint.json
uv run python scripts/verify_artifacts.py verify \
  artifacts/datasets artifacts/manifests/libero10-dataset.json
```

## Model architecture

Both checkpoints load the pinned `pi05_libero` architecture: π₀.₅ mode, Gemma 2B PaliGemma backbone, Gemma 300M action expert, bfloat16, continuous state input, ten-step horizon, and 32 padded internal action dimensions. The LIBERO output transform returns the first seven dimensions.

Regenerate both parameter-tree snapshots:

```bash
CUDA_VISIBLE_DEVICES=1 external_repos/openpi/.venv/bin/python scripts/snapshot_architecture.py \
  --checkpoint-name full \
  --checkpoint-dir /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --checkpoint-manifest artifacts/manifests/full-checkpoint.json \
  --output artifacts/manifests/pi05-libero-architecture-full.json

CUDA_VISIBLE_DEVICES=1 external_repos/openpi/.venv/bin/python scripts/snapshot_architecture.py \
  --checkpoint-name early \
  --checkpoint-dir artifacts/checkpoints/pi05_libero_2000 \
  --checkpoint-manifest artifacts/manifests/early-checkpoint.json \
  --output artifacts/manifests/pi05-libero-architecture-early.json
```

## Evaluator image

```bash
docker build -t pi05-libero-eval:650c5b0 \
  -f docker/Dockerfile.libero external_repos/openpi
docker run --rm --gpus 'device=1' pi05-libero-eval:650c5b0 \
  nvidia-smi --query-gpu=name --format=csv,noheader
```

The local Dockerfile is identical in runtime dependencies to the pinned OpenPI
file, except that it prebuilds `bddl==1.0.1` with setuptools 75.3.0. This avoids
an upstream Python 3.8 build-isolation regression and does not change the
installed evaluation environment.

## Smoke and pilot gates

Start a fresh model server. Do not send any unrelated request to its port:

```bash
scripts/run_policy_server.sh full 1 8001 \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  runs/server-full
```

From a second terminal, run two task-0 episodes:

```bash
scripts/run_libero_eval.sh full 1 8001 runs/two-episode-full --task-ids 0 --trials 2
```

Stop the server and start a fresh process before the 10×5 pilot:

```bash
scripts/run_libero_eval.sh full 1 8001 runs/pilot-full --trials 5
```

Repeat with checkpoint name `early`, GPU 2, port 8002, and `artifacts/checkpoints/pi05_libero_2000`. A gate passes only with no `invalid.json`, no predicate disagreement, finite 10×7 actions, unique JSONL keys, and playable videos.

## Primary 10×50 runs

Use one uninterrupted server/evaluator chain per checkpoint. The two chains may run concurrently:

```bash
scripts/run_policy_server.sh full 1 8001 \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  runs/server-full-primary
scripts/run_libero_eval.sh full 1 8001 runs/primary-full
```

```bash
scripts/run_policy_server.sh early 2 8002 \
  artifacts/checkpoints/pi05_libero_2000 runs/server-early-primary
scripts/run_libero_eval.sh early 2 8002 runs/primary-early
```

OpenPI starts each JAX policy at `key(0)` and advances it per inference request. If a server restarts, another client queries it, or an episode becomes invalid, preserve the run as diagnostic and restart that checkpoint from task 0 episode 0 in a new empty directory. Never splice RNG streams into a primary result.

## Report and acceptance audit

```bash
uv run python scripts/report_results.py \
  --full runs/primary-full/episodes.jsonl \
  --early runs/primary-early/episodes.jsonl \
  --json results/pi05-libero-long-summary.json \
  --markdown results/pi05-libero-long-summary.md
```

The command exits zero only when both logs contain exactly 500 valid unique episodes, every task has indices 0–49 with seed 7, predicates agree, and success lies in these approved intervals:

- Full π₀.₅: 89.4%–95.4% around the public 92.4%.
- Early π₀.₅: 38%–48% around the COAST 43% reference.

If a score misses, preserve evidence and debug one variable at a time in this order: source/artifact revisions, camera rendering and 180° rotation, state/quaternion, action unnormalization and gripper sign, reset/init/RNG sequence, replanning/control horizon, then BDDL/native success semantics. COAST's 15/30 protocol is an additional early-checkpoint diagnostic, not a substitute for the requested 500-trial result.

Large checkpoints, datasets, videos, and logs are excluded from Git. Their committed manifests and the final machine-readable report provide the audit trail.

## Design documents

- [Reproduction design](docs/superpowers/specs/2026-08-01-pi05-libero-long-reproduction-design.md)
- [Implementation plan](docs/superpowers/plans/2026-08-01-pi05-libero-long-reproduction.md)
