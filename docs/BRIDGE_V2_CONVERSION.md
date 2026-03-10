# BridgeData V2 Preprocessing (Aligned to DROID Eval Layout)

This workflow converts BridgeData V2 (Hugging Face OXE export) into the same local structure used by current DROID evaluations:

- one parquet per episode (`timestamp`, `observation.state`, `action`, `task_index`)
- one mp4 per camera per episode
- `meta/tasks.jsonl`, `meta/episodes.jsonl`
- `L1_test_set.json`, `L3_test_set.json` with `{episode_id, path}`

It also writes a canonical intermediate set:

- `canonical/episodes/*.npz`
- `canonical/videos/<cam>/episode_xxxxxx.mp4`
- `canonical/meta.jsonl`

## 1) Install deps

Use your existing `dreamzero` env (already includes most eval deps). If needed:

```bash
conda run -n dreamzero pip install datasets huggingface_hub pyarrow polars av
```

## 2) Run conversion from Hugging Face (long-chain + 20-step protocol)

```bash
conda run -n dreamzero python scripts/data/convert_bridge_v2.py \
  --hf-dataset-id mbodiai/oxe_bridge_v2 \
  --hf-config default \
  --hf-splits shard_0 \
  --output-dir data/bridge_v2_lerobot \
  --canonical-dir data/bridge_v2_canonical \
  --test-sets-dir test_sets_bridge \
  --fps 10 \
  --max-episodes 200 \
  --min-episode-len 20 \
  --min-atomic-actions 3 \
  --test-set-size 20 \
  --mapping-mode bridge_eef \
  --camera-fill-mode duplicate
```

### Important mapping note

- `bridge_eef` (recommended): uses Bridge native EE pose + gripper, avoids fake joint labels.
- `droid_compat`: pads data into DROID-like index slots for script compatibility.

If you use `bridge_eef`, report metrics as cross-dataset trend comparisons (not absolute value parity with DROID joint-L2).

## 3) Evaluate with 20 steps

Use generated test sets and set `--eval-steps 20`.

Example (dual-system):

```bash
conda run -n dreamzero python scripts/eval/run_dualsystem_evaluation.py \
  --host localhost --port 8000 \
  --dataset-root data/bridge_v2_lerobot \
  --test-sets-dir test_sets_bridge \
  --tier L3 \
  --method bridge_dual_llm_l3_th010_s20 \
  --planner-mode llm \
  --prompt-mode subtask \
  --error-threshold 0.1 \
  --eval-steps 20
```

## 4) Output checklist

Check these files exist:

- `data/bridge_v2_lerobot/meta/tasks.jsonl`
- `data/bridge_v2_lerobot/meta/episodes.jsonl`
- `data/bridge_v2_lerobot/data/chunk-000/episode_000000.parquet`
- `data/bridge_v2_lerobot/videos/chunk-000/observation.images.exterior_image_1_left/episode_000000.mp4`
- `test_sets_bridge/L1_test_set.json`
- `test_sets_bridge/L3_test_set.json`

## 5) Data quality rules implemented

- drop empty/invalid frames
- drop episodes with missing key state/action fields
- drop episodes with long all-zero action segments (default >=20)
- keep long-chain candidates by atomic action heuristic (`>=3`) for test set selection
