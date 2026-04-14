# Unified Amortized Time Accounting Summary

This summary follows the user's Section 3.3 definition:

- Total episode time: `T_total_ep = K_valid * T_policy + T_plan`
- Unified equivalent per-step time: `T_equiv = T_policy + T_plan / K`

Key practical consequence for the archived 3-way results:

- The archived `DROID` and `AgiBot` 3-way files store `plan_time` and `num_steps`.
- In all checked rows, `K_valid = num_steps` is exactly equal to the configured `K`.
- Therefore, for these archived comparisons:
  - `Delta T_total_ep = Delta T_plan`
  - `Delta T_equiv = Delta T_plan / K`
- So even though the 3-way files do not store `T_policy`, the **increment introduced by adding VAL** is still recoverable exactly under the stated formula.

## Exact Formula-Based Comparison

| Dataset | Split | Paired N | Configured `K` | Mean `K_valid` (orig / dual / val) | `VAL - Original` total overhead (s/episode) | `VAL - Dual` total overhead (s/episode) | `VAL - Original` amortized overhead (s/step) | `VAL - Dual` amortized overhead (s/step) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| DreamZero-DROID | ALL | 150 | 3 | 3 / 3 / 3 | 4.378 | 2.971 | 1.459 | 0.990 |
| DreamZero-DROID | L1 | 50 | 3 | 3 / 3 / 3 | 4.311 | 2.854 | 1.437 | 0.951 |
| DreamZero-DROID | L2 | 50 | 3 | 3 / 3 / 3 | 4.426 | 3.032 | 1.475 | 1.011 |
| DreamZero-DROID | L3 | 50 | 3 | 3 / 3 / 3 | 4.398 | 3.028 | 1.466 | 1.009 |
| DreamZero-AgiBot | ALL | 147 | 3 | 3 / 3 / 3 | 6.622 | 5.148 | 2.207 | 1.716 |
| DreamZero-AgiBot | L1 | 48 | 3 | 3 / 3 / 3 | 6.789 | 5.441 | 2.263 | 1.814 |
| DreamZero-AgiBot | L2 | 49 | 3 | 3 / 3 / 3 | 6.577 | 5.094 | 2.192 | 1.698 |
| DreamZero-AgiBot | L3 | 50 | 3 | 3 / 3 / 3 | 6.507 | 4.919 | 2.169 | 1.640 |
| DreamDojo-AgiBot pilot | pilot | 9 | 49 | 49 / 49 / 49 | 5.696 | 4.306 | 0.116 | 0.088 |

## Direct Reading

- Under this accounting, the **per-episode** overhead of adding VAL is still a real few seconds:
  - DreamZero-DROID: about `+4.38s` vs `original`, or `+2.97s` vs `dual_llm`
  - DreamZero-AgiBot: about `+6.62s` vs `original`, or `+5.15s` vs `dual_llm`
- But the **amortized per-step** overhead depends entirely on `K`.
- In the DreamZero 3-way archives, `K = 3`, so the amortized overhead is still noticeable:
  - DreamZero-DROID: about `+0.99s/step` vs `dual_llm`
  - DreamZero-AgiBot: about `+1.72s/step` vs `dual_llm`
- In the DreamDojo pilot, `K = 49`, so the same front-loaded planner cost is heavily diluted:
  - DreamDojo-AgiBot pilot: only about `+0.088s/step` vs `dual_llm`

## Interpretation For The “差的不多吗？” Question

- If the comparison target is **total episode time**, then for the archived DreamZero 3-way experiments the difference is **not small**: it is still about `3-5s` more per episode versus `dual_llm`.
- If the comparison target is the **amortized equivalent per-step time** in Eq. (12), then the answer depends on `K`:
  - For the archived DreamZero 3-way runs with `K=3`, the difference is still clearly visible.
  - For longer-horizon settings with much larger `K` (the checked DreamDojo pilot uses `K=49`), the difference becomes small.
- So the strongest empirically supported phrasing is:
  - “The planner overhead is front-loaded. It is not negligible in short `K=3` evaluations, but it becomes small after amortization when the evaluation horizon is long enough.”

## Important Limitation

- The archived 3-way result files do **not** store `T_policy`, so absolute `T_equiv` values cannot be reconstructed exactly from those files alone.
- What **can** be reconstructed exactly is the method delta:
  - `Delta T_total_ep = Delta T_plan`
  - `Delta T_equiv = Delta T_plan / K`
- That is sufficient for the before/after comparison requested here.

## Data Sources

- `evaluation_results_dualsystem/droid_3way_selected150.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L1_50_results.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L2_50_results.json`
- `evaluation_results_dualsystem/selected_splits/Agi_DualBetter_L3_50_results.json`
- `evaluation_results_dualsystem/agibot_3way_compare.json`
- `evaluation_results_dualsystem/agibot_dreamdojo_3way_compare.json`
- `scripts/eval/run_droid_3way_compare.py`
- `scripts/eval/run_agibot_3way_compare.py`
- `scripts/eval/run_agibot_dreamdojo_3way_compare.py`
