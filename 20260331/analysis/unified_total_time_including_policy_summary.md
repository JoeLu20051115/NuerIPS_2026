# Unified Total Time (Plan + Policy)

Using your Eq. (11) and Eq. (12):

- `T_total_ep = K * T_policy + T_plan`
- `T_equiv = T_policy + T_plan / K`

This table reports **original (`task_token_only`)** and **`llm_val`** time together, i.e. not only the planner time.

## Timing baselines used

- **DreamZero-DROID**:
  - `T_policy(original)` is taken from the matched archived `system1` timing summaries in:
    - `evaluation_results/overall_system1_full_strict_h200_th014_evaluation.json`
  - The paired `dualsystem` summary shows policy time is nearly unchanged (`|delta| < 0.005 s/step`), so using the original executor time for `llm_val` is consistent with the amortized-time formulation.
  - Only `L1` and `L3` have archived per-tier timing; `L2` is set to the mean of `L1` and `L3`. The difference between `L1` and `L3` is only `0.0016 s/step`, so this interpolation is negligible.

- **DreamZero-AgiBot**:
  - `T_policy(original)` is taken from the archived original-style AgiBot executor timing summary:
    - `evaluation_results_dualsystem/L3_case6_system1_token_th010_summary.json`
  - Its matched dual-planner real-API summary
    - `evaluation_results_dualsystem/L3_case6_dual_llm_subtask_t0_th010_realapi_summary.json`
    shows policy time differs by only about `0.0005 s/step`, again supporting the “executor time unchanged” assumption.
  - This AgiBot executor baseline is only archived at the executor level rather than per split, so the same `T_policy` is used for `L1/L2/L3/ALL`.

- **`T_plan(llm_val)`** is taken directly from the 3-way comparison archives:
  - `evaluation_results_dualsystem/droid_3way_selected150.json`
  - `evaluation_results_dualsystem/agibot_3way_compare.json`

- **AgiBot `ALL`** below follows the same selected subset used in the earlier comparison:
  - `L1=48`, `L2=49`, `L3=50`, so `ALL=147`.

## Table

| Dataset | Split | `K` | `n` | original `T_policy` (s/step) | original `T_plan` (s) | original `T_total_ep` (s) | `llm_val` `T_plan` (s) | `llm_val` `T_total_ep` (s) | original `T_equiv` (s/step) | `llm_val` `T_equiv` (s/step) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DreamZero-DROID | ALL | 3 | 150 | 2.133 | 0.000 | 6.398 | 4.378 | 10.776 | 2.133 | 3.592 |
| DreamZero-DROID | L1 | 3 | 50 | 2.132 | 0.000 | 6.396 | 4.308 | 10.704 | 2.132 | 3.568 |
| DreamZero-DROID | L2 | 3 | 50 | 2.133 | 0.000 | 6.398 | 4.360 | 10.758 | 2.133 | 3.586 |
| DreamZero-DROID | L3 | 3 | 50 | 2.133 | 0.000 | 6.400 | 4.466 | 10.867 | 2.133 | 3.622 |
| DreamZero-AgiBot | ALL | 3 | 147 | 2.117 | 0.000 | 6.351 | 6.622 | 12.973 | 2.117 | 4.324 |
| DreamZero-AgiBot | L1 | 3 | 48 | 2.117 | 0.000 | 6.351 | 6.789 | 13.140 | 2.117 | 4.380 |
| DreamZero-AgiBot | L2 | 3 | 49 | 2.117 | 0.000 | 6.351 | 6.577 | 12.927 | 2.117 | 4.309 |
| DreamZero-AgiBot | L3 | 3 | 50 | 2.117 | 0.000 | 6.351 | 6.507 | 12.858 | 2.117 | 4.286 |

## Direct takeaway

- After including both **planning + executor inference**, `llm_val` is **not** “almost the same” as `original` on the current `K=3` DreamZero evaluations.
- In total episode time:
  - **DROID** increases from about `6.4s` to `10.7-10.9s`
  - **AgiBot** increases from about `6.35s` to `12.86-13.14s`
- So under the current short-horizon setting, the extra planner call is still clearly visible in total time.

The machine-readable version is saved at `20260331/analysis/unified_total_time_including_policy_summary.tsv`.
