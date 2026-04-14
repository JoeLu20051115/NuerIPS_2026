# Seed-42 Per-Dataset 150 Splits

- Seed: `42`
- Target per dataset: `150` episodes.
- Split per dataset: `L1=50`, `L2=50`, `L3=50`.
- Level definition: per-dataset internal step-count tertiles.

## DROID

- Cleaning: `{'raw_pool': 57774, 'eligible_after_cleaning': 896, 'dropped': {'missing_sensor': 1336, 'missing_parquet': 55541, 'too_short': 1}, 'source_isolation': 'Unavailable: local DROID metadata exposes only the public pool, not a public non-train split.', 'success_rule': 'Keep success == True from data/droid_lerobot/meta/episodes.jsonl.', 'validity_rule': 'Require step_count >= 10 plus parquet and all declared videos on disk.'}`
- Cutoffs: `{'q33_max': 162, 'q66_max': 253, 'candidate_counts': {'L2': 298, 'L1': 300, 'L3': 298}}`
- Selected counts: `{'L1': 50, 'L2': 50, 'L3': 50}`
- Unique task groups in selected split: `{'L1': 50, 'L2': 50, 'L3': 50}`

## AgiBot

- Cleaning: `{'raw_pool': 189, 'eligible_after_cleaning': 189, 'dropped': {}, 'source_isolation': 'Satisfied: use only the official AgiBot 2026 val split; train is never touched.', 'success_rule': 'No explicit success/reward field is provided in the official val files, so only non-train, sensor-complete episodes are retained.', 'validity_rule': 'Require step_count >= 10 and the full five-file episode bundle.'}`
- Cutoffs: `{'q33_max': 29, 'q66_max': 38, 'candidate_counts': {'L2': 64, 'L1': 66, 'L3': 59}}`
- Selected counts: `{'L1': 50, 'L2': 50, 'L3': 50}`
- Unique task groups in selected split: `{'L1': 10, 'L2': 10, 'L3': 10}`

## EgoDex

- Cleaning: `{'raw_pool': 3243, 'eligible_after_cleaning': 3243, 'dropped': {}, 'source_isolation': 'Satisfied: use only the official EgoDex test.zip pool.', 'success_rule': 'The official test zip does not provide explicit success/reward labels, so all sensor-complete official test episodes are retained.', 'validity_rule': 'Require paired .hdf5/.mp4 entries and step_count >= 10.'}`
- Cutoffs: `{'q33_max': 107, 'q66_max': 300, 'candidate_counts': {'L1': 1083, 'L2': 1546, 'L3': 614}}`
- Selected counts: `{'L1': 50, 'L2': 50, 'L3': 50}`
- Unique task groups in selected split: `{'L1': 50, 'L2': 50, 'L3': 50}`

## Updated Complexity Table

The table below recomputes the old complexity summary on the new `seed42_per_dataset150_l123` splits.

| Dataset | Mean Time (s) L1 / L2 / L3 | Mean Step Proxy L1 / L2 / L3 | Stateful-op Rate L1 / L2 / L3 |
| --- | --- | --- | --- |
| Official-DROID | 7.71 / 13.63 / 25.12 | 1.26 / 0.96 / 1.24 | 38% / 20% / 30% |
| Official-AgiBot | 4.57 / 6.55 / 11.00 | N/A | N/A |
| Official-EgoDex | 3.76 / 12.27 / 37.52 | 0.86 / 0.58 / 0.78 | 34% / 24% / 32% |

## Complexity Notes

- `Official-DROID` time follows the old table convention: `step_count / 15`.
- `Official-EgoDex` time follows the old table convention: `step_count / 20`.
- `Official-AgiBot` time is recomputed from `proprio_stats.h5["timestamp"]` as `(last - first) / 1e9`.
- `Mean Step Proxy` and `Stateful-op Rate` are recomputed with the same lexical heuristic used in `scripts/analysis/summarize_semantic_complexity_proxies.py`.
- `Official-AgiBot` does not expose the old curated manifest fields such as `action_plan` or task-name text, so its semantic columns are not directly reproducible under the original AgiBot metric definition.
