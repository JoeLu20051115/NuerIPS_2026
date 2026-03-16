# LIBERO Subset Plan

## Goal

Stop treating LIBERO as a zero-shot smoke test and move to a realistic
fine-tuning target:

- Use a small, fixed, runnable LIBERO subset.
- Keep DreamZero's core training/inference recipe.
- Target `> 30%` average success on the fixed subset before expanding scope.

## What Changed

This repo now includes:

- `scripts/data/build_libero_subset_manifest.py`
  - Filters out tasks with missing init states.
  - Optionally filters out tasks that fail environment initialization.
  - Expands a task subset into concrete `(task_id, init_state_id)` episodes.

- `groot/vla/configs/data/dreamzero/libero_relative.yaml`
  - A DreamZero data config for a converted LIBERO subset in LeRobot layout.
  - Uses the same single-arm modality schema as the DROID config.
  - Sets LIBERO video FPS to `5`.

- `scripts/train/libero_subset_training.sh`
  - Uses:
    - full training (`train_architecture=full`)
    - `action_horizon=24`
    - `num_frame_per_block=2`
    - `num_action_per_block=24`
    - `max_chunk_size=4`
  - Keeps the existing DreamZero freezing behavior:
    - train DiT / projector / action-side modules
    - freeze text encoder / image encoder / VAE

## Recommended First Subset

Start with a fixed standard `libero_10` subset rather than the broken local
PRO variants.

Suggested build command:

```bash
python scripts/data/build_libero_subset_manifest.py \
  --suite libero_10 \
  --num-tasks 10 \
  --episodes-per-task 10 \
  --verify-env \
  --output evaluation_results_dualsystem/libero_10_good_100_manifest.json
```

This yields `100` clean episodes from the first `10` tasks and first `10`
init states per task.

## Important Limitation

The current repo still needs a proper LIBERO-to-LeRobot conversion step before
DreamZero can be trained on the subset. The new training config and launcher
assume the subset has already been converted into the same camera/state/action
schema used by the existing single-arm DreamZero data pipeline.

That means:

- zero-shot LIBERO rollout is still only a smoke test
- `> 30%` success is **not** guaranteed by code cleanup alone
- reaching that target requires actual subset training / fine-tuning

## Evaluation Guidance

For online LIBERO experiments, use:

- `success_rate`
- `average_steps_to_success` (for successful episodes)
- rollout videos for failure analysis

Do **not** treat `mean_joint_action_l2` as the main metric. It is only a
proxy for action magnitude, not task completion.
