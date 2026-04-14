# Validation Planning-Time Overhead Summary

This summary compares archived per-episode `plan_time` before and after adding VAL to the planner.

Important scope note:
- The numbers below measure planner latency only (`plan_time`).
- They do **not** include the shared policy inference / video generation part of the rollout.
- So `VAL - dual` is the direct extra planning cost of adding VAL, while the true end-to-end task-time overhead should be smaller as a fraction of the full rollout.

## Main Comparison

| Dataset | Split | Paired N | Mean episode duration (s) | Original mean plan (s) | Dual mean plan (s) | VAL mean plan (s) | VAL - Original (s) | VAL - Dual (s) | VAL / Dual | (VAL - Original) / duration | (VAL - Dual) / duration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DreamZero-DROID | ALL | 150 | 10.70 | 0.000 | 1.407 | 4.378 | 4.378 | 2.971 | 3.11 | 40.9% | 27.8% |
| DreamZero-DROID | L1 | 50 | 6.60 | 0.000 | 1.457 | 4.311 | 4.311 | 2.854 | 2.96 | 65.3% | 43.3% |
| DreamZero-DROID | L2 | 50 | 9.99 | 0.000 | 1.394 | 4.426 | 4.426 | 3.032 | 3.17 | 44.3% | 30.4% |
| DreamZero-DROID | L3 | 50 | 15.52 | 0.000 | 1.370 | 4.398 | 4.398 | 3.028 | 3.21 | 28.3% | 19.5% |
| DreamZero-AgiBot | ALL | 147 | 64.34 | 0.000 | 1.475 | 6.622 | 6.622 | 5.148 | 4.49 | 10.3% | 8.0% |
| DreamZero-AgiBot | L1 | 48 | 21.42 | 0.000 | 1.348 | 6.789 | 6.789 | 5.441 | 5.03 | 31.7% | 25.4% |
| DreamZero-AgiBot | L2 | 49 | 44.22 | 0.000 | 1.483 | 6.577 | 6.577 | 5.094 | 4.43 | 14.9% | 11.5% |
| DreamZero-AgiBot | L3 | 50 | 125.25 | 0.000 | 1.588 | 6.507 | 6.507 | 4.919 | 4.10 | 5.2% | 3.9% |

## Medians

| Dataset | Split | Original median (s) | Dual median (s) | VAL median (s) |
| --- | --- | ---: | ---: | ---: |
| DreamZero-DROID | ALL | 0.000 | 1.324 | 4.481 |
| DreamZero-DROID | L1 | 0.000 | 1.332 | 4.595 |
| DreamZero-DROID | L2 | 0.000 | 1.392 | 4.473 |
| DreamZero-DROID | L3 | 0.000 | 1.301 | 4.382 |
| DreamZero-AgiBot | ALL | 0.000 | 1.324 | 6.407 |
| DreamZero-AgiBot | L1 | 0.000 | 1.269 | 6.732 |
| DreamZero-AgiBot | L2 | 0.000 | 1.329 | 6.332 |
| DreamZero-AgiBot | L3 | 0.000 | 1.368 | 6.266 |

## Supplemental Small-Sample Check

- `agibot_dreamdojo_3way_compare.json` has a paired `n=9` pilot overlap:
- `original = 0.000s`, `dual = 1.389s`, `VAL = 5.696s`, so `VAL - dual = 4.306s` and `VAL / dual = 4.10x`.
- This is directionally consistent with the two main archived datasets above.

## Interpretation

- Across the two main datasets, adding VAL increases planner time by about `+4.38s` on DreamZero-DROID and `+6.62s` on DreamZero-AgiBot versus `original/task_token_only`.
- If the fairer baseline is already using an LLM planner, then the marginal VAL cost is about `+2.97s` on DreamZero-DROID and `+5.15s` on DreamZero-AgiBot versus `dual_llm`.
- So the extra time is not literally "almost no difference" at the planner stage; it is a real few-second overhead.
- But for longer-horizon tasks the relative cost becomes much smaller. The clearest example is AgiBot `L3`, where `VAL - dual` is only about `3.9%` of the mean task-duration proxy.
- The overhead is most noticeable on short tasks, especially DROID `L1` and AgiBot `L1`.

## Data Sources

- `evaluation_results_dualsystem/droid_3way_selected150.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L1_50_results.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L2_50_results.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L3_50_results.json`
- `evaluation_results_dualsystem/agibot_3way_compare.json`
- `evaluation_results_dualsystem/agibot_dreamdojo_3way_compare.json`
- `data/Agi_DualBetter_L1_50/meta/manifest.json`
- `data/Agi_DualBetter_L2_50/meta/manifest.json`
- `data/Agi_DualBetter_L3_50/meta/manifest.json`
