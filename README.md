# LOGIV Origin

LOGIV Origin is the deployable reference implementation of a closed-loop
robot-control method that combines visual fact grounding, symbolic planning,
formal plan validation, causal DAG execution, and bounded online repair.

This branch is intentionally a source release. It contains the method,
canonical configuration, integration code, tests, and deployment instructions.
Historical experiments, result archives, run logs, tuning sweeps, and episode
selection utilities are not part of the current tree.

## Method at a glance

LOGIV runs the following control loop:

1. Ground the current observation into typed three-valued facts.
2. Plan over the remaining symbolic task state.
3. Validate the proposed plan with VAL.
4. Compile the certified plan into a causal execution DAG.
5. Dispatch a bounded policy action chunk.
6. Re-observe and gate progress on fresh evidence.
7. Retry uncertainty within a finite budget or repair a confirmed local
   failure without reopening confirmed persistent milestones.
8. Accept completion only when the environment's native success signal agrees
   with the terminal LOGIV state.

The method is described in [docs/METHOD.md](docs/METHOD.md).

## Repository layout

```text
docker/                     LIBERO/OpenPI runtime image
external_repos/openpi/      Pinned OpenPI submodule
patches/robotwin/           Clean TACO/RoboTwin integration patch
scripts/                    LIBERO and RoboTwin deployment entry points
src/pi05_libero_repro/logiv LOGIV runtime and packaged Origin configuration
tests/                      Method, integration, and release-contract tests
```

## Install

Requirements:

- Linux with Python 3.11 or newer
- `uv`
- Docker and an NVIDIA runtime for LIBERO deployment
- a compatible VAL build
- an OpenAI API key for GPT-4o state grounding
- benchmark-specific policy checkpoints supplied outside the repository

Clone with submodules and create the Python environment:

```bash
git clone --recurse-submodules https://github.com/JoeLu20051115/NuerIPS_2026.git LOGIV_Origin
cd LOGIV_Origin
uv sync
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Build the compatible VAL binary:

```bash
scripts/build_val_for_libero.sh
```

The generated validator is placed under `artifacts/tools/`, which is ignored by
Git.

## LIBERO deployment

Build the pinned OpenPI runtime image:

```bash
docker build \
  -t pi05-libero-eval:650c5b0 \
  -f docker/Dockerfile.libero \
  external_repos/openpi
```

Start the π0.5 policy server. The checkpoint directory must contain `params/`
and `assets/physical-intelligence/libero/norm_stats.json`:

```bash
scripts/run_policy_server.sh 0 8000 /path/to/pi05_libero
```

In another terminal, run LOGIV. The output directory is caller-owned and should
normally be outside the repository:

```bash
export OPENAI_API_KEY='...'
scripts/run_logiv_eval.sh \
  0 8000 /tmp/logiv-origin-run \
  --task-ids 0 \
  --episode-indices 0:1 \
  --no-video
```

The launcher fixes the deployed method to `FULL_LOGIV`, GPT-4o perception,
metadata-assisted goals, the canonical Origin configuration, and locked
prompts. Extra arguments are forwarded to the evaluator for runtime limits and
task selection.

## RoboTwin deployment

Apply [the Origin patch](patches/robotwin/README.md) to a clean TACO checkout at
the documented base commit. Then run an ordinary sequence of simulator
episodes:

```bash
export OPENAI_API_KEY='...'
uv run python scripts/run_robotwin_logiv.py \
  --task turn_switch \
  --episodes 1 \
  --seed 0 \
  --taco /path/to/TACO \
  --checkpoint /path/to/pi05_TACO_robotwin2_finetuned \
  --tokenizer /path/to/tokenizer.model \
  --val-binary /path/to/Validate
```

Here `--seed` is only the simulator RNG start value for the requested run.
Episodes proceed sequentially from that value. The Origin interface does not
scan, rank, reject, replace, or select episodes.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for environment details and
failure checks.

## Verification

Run the complete retained suite:

```bash
uv run pytest -q
```

Useful smoke checks:

```bash
uv run python scripts/eval_logiv_libero.py --help
uv run python scripts/run_robotwin_logiv.py --help
bash -n scripts/*.sh
uv build
```

Generated checkpoints, datasets, validators, outputs, videos, screenshots,
JSONL records, and logs are ignored by Git. Keep credentials and machine-local
paths outside committed configuration.
