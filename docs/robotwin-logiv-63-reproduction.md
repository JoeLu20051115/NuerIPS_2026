# Reproduce RoboTwin LOGIV 63/100 vs pi0.5 56/100

This package reruns one frozen RoboTwin 2.0 protocol: ten tasks, ten fixed
seed/instruction pairs per task, LOGIV + GPT-4o fact grounding, and a direct
pi0.5 baseline on the same 100 cells. The observed frozen rerun was 63/100 for
LOGIV and 56/100 for pi0.5.

## Requirements

- Python 3.11 or newer with this repository installed (`uv sync --dev`).
- A TACO checkout at `ee9e06dcf01d8b1a606b18841cb52b1ec88a423b`
  with the ordered control patches listed in
  [`patches/robotwin/README.md`](../patches/robotwin/README.md). The resulting
  control-runtime tree is the one recorded at
  `8de0ed9520989f9fd156904291d0895b9a361886`.
- The Hugging Face checkpoint
  `rhodes-team-teleai/pi05_TACO_robotwin2_finetuned` at revision
  `0f000e2748bd1fcb43027d8790f81fcccaa04670`, including the three CFNs for
  `handover_block`, `move_can_pot`, and `beat_block_hammer`.
- The PaliGemma tokenizer at
  `gs://big_vision/paligemma_tokenizer.model`, used by the pi0.5 checkpoint.
- An executable VAL `Validate` binary for LOGIV's online PDDL planner.
- Three CUDA GPUs and the Python executable for the RoboTwin environment.
- `OPENAI_API_KEY` exported in the environment. The launcher never prints or
  writes this value.

## Run

Choose a new output directory whose final component contains only letters,
digits, `.`, `_`, and `-`. It must not already exist.

```bash
export OPENAI_API_KEY='your-key-in-the-environment'

uv run python scripts/reproduce_robotwin_logiv_63.py \
  --taco /path/to/TACO \
  --checkpoint /path/to/pi05_TACO_robotwin2_finetuned \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --val-binary /path/to/Validate \
  --python /path/to/robotwin-environment/bin/python \
  --output /path/to/robotwin-logiv-63-rerun \
  --gpus 0 1 2
```

Use `--dry-run` to validate all inputs and print the six worker commands
without creating the output directory or starting RoboTwin.

The driver runs three LOGIV workers concurrently, then three pi0.5 workers on
the same GPU mapping. It never searches for seeds, substitutes cells, or
replays a historical success. Videos are disabled.

## Outputs

A completed run leaves four small files in the requested output directory:

- `manifest.json`: the exact non-secret 100-cell protocol;
- `run.json`: resolved non-secret paths, GPU mapping, and unique run tags;
- `summary.json`: totals, paired flips, task rows, and structural errors;
- `summary.md`: the same outcome in a readable table.

After a structurally complete run, worker logs, native episode directories,
events, videos, and any VLM audit images under the two unique run tags are
removed. A process failure or incomplete protocol retains raw output for
diagnosis.

The command exits nonzero if a complete live run does not equal the observed
63/100 and 56/100 totals. It still records the actual outcome before exiting.
RoboTwin physics, GPU kernels, and the live GPT-4o service are not bitwise
deterministic, so the fixed protocol is reproducible but an exact count cannot
be mathematically guaranteed on every future live run.
