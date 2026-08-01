# π₀.₅ LIBERO-Long Reproduction

This repository evaluates the full and 2,000-step π₀.₅ checkpoints on the ten long-horizon tasks in the official `libero_10` suite. Each primary result contains 50 valid trials per task (500 per checkpoint) using OpenPI's camera, state, control, reset, and native success semantics.

## Reproduced result

Both requested 10×50 evaluations passed the predeclared acceptance audit:

| Checkpoint | Successes | Reproduced | Reference |
| --- | ---: | ---: | ---: |
| Full π₀.₅ | 460/500 | **92.0%** | 92.4% |
| Early π₀.₅ (2k steps) | 218/500 | **43.6%** | 43% |

All 1,000 episodes are valid. Each checkpoint has exactly 50 trials for each
task, all native success predicates agree, and every H.264 video is 224×224
with its decoded frame count equal to the recorded control-step count. The 500
cross-checkpoint pairs have identical task/episode keys, initial-state hashes,
and first-frame hashes.

- [Human-readable per-task report](results/pi05-libero-long-summary.md)
- [Machine-readable report](results/pi05-libero-long-summary.json)
- [Final protocol and integrity audit](results/pi05-libero-long-audit.md)
- [中文任务信息与逐任务结果](markdown/pi05-libero-long-task-results.md)
- [Full-run SHA-256 manifest](artifacts/manifests/primary-full.json)
- [Early-run SHA-256 manifest](artifacts/manifests/primary-early.json)

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

## LOGIV closed-loop development

LOGIV is implemented under `src/pi05_libero_repro/logiv/` as a task-level
closed-loop layer around the frozen full π₀.₅ checkpoint. The current no-API
configuration is deliberately labeled `scripted-vlm/oracle-grounding`: the
initial proposals are frozen in
`configs/logiv/libero10-scripted-proposals.json`, while runtime facts come from
the current LIBERO simulator predicates and grasp state. These oracle facts are
development instrumentation, not a claim about GPT-4o perception quality.

The fixed typed STRIPS domain is
`configs/logiv/logiv-libero-domain.pddl`; the frozen ten-task coverage contract
is `configs/logiv/libero10-coverage.json`. Plans are checked by the real VAL
binary after a signed three-valued trace. Build the Ubuntu 22.04-compatible,
hash-pinned VAL executable used inside the official evaluator container once:

```bash
scripts/build_val_for_libero.sh
uv run pytest -q
```

Start a fresh full-checkpoint server, then run the task-8 development smoke in a
second terminal:

```bash
scripts/run_policy_server.sh full 1 8001 \
  /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  runs/logiv-server-task8-v1

scripts/run_logiv_eval.sh FULL_LOGIV 0 8001 runs/logiv-task8-v1 \
  --run-id logiv-task8-v1 \
  --goal-mode METADATA_ASSISTED \
  --deviation-mode NOMINAL \
  --oracle-grounding \
  --development-only \
  --prompt-config /repro/configs/logiv/prompts/pi05-subtasks-v5.json \
  --prompt-version pi05-subtasks-v5 \
  --task-ids 8 \
  --episode-indices 0:3
```

The evaluator and policy server implement an episode-local RNG reset protocol.
Namespace-separated seeds for the policy and simulator are derived from
`(master seed, task ID, episode index)`. Before every reset the evaluator seeds
Python, NumPy, and LIBERO; every policy request carries the independent policy
seed and the server returns a matching inference-index receipt. This makes a
resumed or isolated episode independent of the order in which earlier episodes
were evaluated. Run paired method arms serially against the **same server
process**. A diagnostic run found that two separately loaded server instances
could disagree despite matching reset state, first-frame hash, and episode
seed; cross-instance arm comparisons are therefore rejected rather than
treated as paired evidence.

Task 8 is an explicit regression gate for the graph representation: its two
moka-pot actions must have no edge between them and the recorded initial action
layer width must be at least two. The canonical agenda is deterministic, but it
does not turn the causal DAG into an adjacency chain.

The evaluator supports five isolated arms: `BASE`, `STAGE_ONLY`,
`GRAPH_WITHOUT_VAL`, `VAL_WITHOUT_LOCALIZED_REPAIR`, and `FULL_LOGIV`.
`GRAPH_WITHOUT_VAL` uses a separately marked schema-only graph and cannot reuse
a Full LOGIV certificate. A holdout run requires `--prompt-locked`; unlocked
prompts are accepted only with `--development-only`. `goal_mode` and
`deviation_mode` are mandatory and never silently default to mixed settings.

Each allocated episode writes an append-only `episodes.jsonl`, a hash-chained
event journal, the initial/final graph hashes, certificate, exact per-attempt
prompt and actions, STOPPED evidence, safety context, receipts, and video under
its artifact directory. Terminal, grounding, validation, timeout, budget, and
evaluator failures remain in the success-rate denominator. Generate task-wise
Wilson intervals and the predeclared equal-task paired bootstrap with:

```bash
uv run python scripts/report_logiv_results.py \
  --episodes runs/logiv-locked/episodes.jsonl \
  --baseline runs/primary-full/episodes.jsonl \
  --json results/logiv-libero10-summary.json \
  --markdown results/logiv-libero10-summary.md
```

Passing unit tests and symbolic/VAL smoke checks demonstrates conformance to the
runtime contract only. It does not establish simulator improvement, perception
accuracy, physical safety, or real-robot performance; those claims require the
allocated interactive rollouts and their external LIBERO evaluator receipts.

The current task-8 development gate (five preselected episode IDs, not the
preregistered 50-episode result) is recorded in
`results/logiv-task8-seeded-dev5-v5c.json`: Full LOGIV succeeds on 4/5 versus
Base on 3/5 with the same episode seeds. The paired difference is +0.20 with a
wide 10,000-draw bootstrap interval `[0.00, 0.60]`. This is directional debugging
evidence only. The successful recovered episode contains two certified repair
rounds and an external LIBERO success receipt; all initial task-8 graphs retain
action-layer width two.

## Design documents

- [Reproduction design](docs/superpowers/specs/2026-08-01-pi05-libero-long-reproduction-design.md)
- [Implementation plan](docs/superpowers/plans/2026-08-01-pi05-libero-long-reproduction.md)
- [LOGIV closed-loop design](docs/superpowers/specs/2026-08-02-logiv-libero-closed-loop-design.md)
- [LOGIV implementation plan](docs/superpowers/plans/2026-08-02-logiv-libero-closed-loop.md)
