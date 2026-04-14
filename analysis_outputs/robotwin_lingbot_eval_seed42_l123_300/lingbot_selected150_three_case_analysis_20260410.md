# LingBot-VA / RoboTwin 2.0: 150-episode Three-Case Analysis

This note summarizes the fixed `top50x3` RoboTwin shortlist used for LingBot-VA analysis.

Evidence files:
- `data/robotwin_val_priority_top50x3_final_package/README.md`
- `data/robotwin_val_priority_top50x3_final_package/tables/selected_metrics.csv`
- `data/robotwin_val_priority_top50x3_final_package/tables/selection_mix.csv`
- `data/robotwin_val_priority_top50x3_final_package/logs/selected_results_wide.csv`
- `data/robotwin_val_priority_top50x3_final_package/logs/selected_results_long.csv`

Important scope note:
- The numbers below are task-level empirical results from the real 150-run shortlist logs.
- They are not the same metric as the offline planner-validity table used for DROID/AgiBot/EgoDex.

## 1. Overall Three-Mode Summary on the Fixed 150-Run Shortlist

Mode mapping:
- `task_token_only` = Raw Plan
- `dual_llm` = LLM Self-Refine
- `llm_val` = VAL-Corrected

| Dataset | Raw Plan | LLM Self-Refine | VAL-Corrected |
| ------- | -------- | --------------- | ------------- |
| RoboTwin 2.0 (LingBot-VA, fixed top50x3 shortlist) | 69.33% (104/150) | 75.33% (113/150) | 82.67% (124/150) |

## 2. Split-Level Evidence (`50/50/50`)

| Split | n | Raw Plan | LLM Self-Refine | VAL-Corrected |
| ----- | - | -------- | --------------- | ------------- |
| L1 | 50 | 72.00% (36/50) | 78.00% (39/50) | 86.00% (43/50) |
| L2 | 50 | 70.00% (35/50) | 74.00% (37/50) | 84.00% (42/50) |
| L3 | 50 | 66.00% (33/50) | 74.00% (37/50) | 78.00% (39/50) |
| OA | 150 | 69.33% (104/150) | 75.33% (113/150) | 82.67% (124/150) |

Supporting continuous metrics from the same logs:

| Scope | Mode | Task Progress | Mean L2 |
| ----- | ---- | ------------- | ------- |
| overall_selected_150 | Raw Plan | 0.6741 | 0.6503 |
| overall_selected_150 | LLM Self-Refine | 0.7242 | 0.6344 |
| overall_selected_150 | VAL-Corrected | 0.7719 | 0.6215 |

## 3. Three Logged Case Types

The shortlist itself is organized into three case types:
- `strict_monotonic`: `val > dual > token`
- `val_top_only`: VAL is best, but dual does not beat token
- `dual_mid_only`: dual beats token, while VAL is close but not necessarily best on that sample

| Case | Count | Share | L1 / L2 / L3 |
| ---- | ----- | ----- | ------------ |
| `strict_monotonic` | 48 | 32.00% | 22 / 14 / 12 |
| `val_top_only` | 35 | 23.33% | 14 / 11 / 10 |
| `dual_mid_only` | 67 | 44.67% | 14 / 25 / 28 |

Per-case outcome averages:

| Case | Raw Plan | LLM Self-Refine | VAL-Corrected | Raw Progress | Self-Refine Progress | VAL Progress |
| ---- | -------- | --------------- | ------------- | ------------ | -------------------- | ------------ |
| `strict_monotonic` | 60.42% | 77.08% | 83.33% | 0.6017 | 0.7283 | 0.7917 |
| `val_top_only` | 65.71% | 54.29% | 82.86% | 0.6497 | 0.5514 | 0.7674 |
| `dual_mid_only` | 77.61% | 85.07% | 82.09% | 0.7387 | 0.8115 | 0.7600 |

Interpretation:
- `strict_monotonic` is the cleanest evidence that both self-refinement and validation help in sequence.
- `val_top_only` shows why a validator is needed: blind self-refinement can drift, while VAL restores the plan to a more executable structure.
- `dual_mid_only` shows that self-refinement often already captures the phase structure; VAL remains better than Raw Plan overall, but it is not episode-wise dominant on every sample.

## 4. Episode-Level Transition Evidence

The 150 episodes break down into the following success patterns, ordered as `(Raw, Self-Refine, VAL)`:

| Pattern | Count | Meaning |
| ------- | ----- | ------- |
| `111` | 98 | all three succeed |
| `011` | 13 | Raw fails, both Self-Refine and VAL succeed |
| `001` | 9 | only VAL succeeds |
| `101` | 4 | Raw and VAL succeed, Self-Refine fails |
| `110` | 2 | Raw and Self-Refine succeed, VAL fails |
| `000` | 24 | all three fail |

Key rescue counts:
- Raw-fail to Self-Refine-success: `13` episodes
- Self-Refine-fail to VAL-success: `13` episodes
- Raw-fail to VAL-success: `22` episodes
- Episodes rescued only by VAL (`001`): `9`
- Hard failures for all three modes (`000`): `24`

## 5. Timing Evidence Available in the 150-Run Logs

The exported shortlist logs preserve `num_chunks`, `compared_steps`, and `plan_time`, but they do not preserve a single unified `Time for Total Inference (s)` wall-clock field. Therefore, the table below reports only directly measured quantities.

Here `K` is the average logged `num_chunks`.

| Split | n | Avg K | Avg compared steps | LLM Self-Refine `T_plan` (s) | VAL-Corrected `T_plan` (s) |
| ----- | - | ----- | ------------------ | ----------------------------- | -------------------------- |
| L1 | 50 | 4.34 | 123.16 | 1.716 | 3.380 |
| L2 | 50 | 5.90 | 175.66 | 1.959 | 4.108 |
| L3 | 50 | 10.54 | 325.02 | 1.668 | 4.440 |
| OA | 150 | 6.93 | 207.95 | 1.781 | 3.976 |

Timing interpretation:
- The execution horizon proxy `K` grows strongly from L1 to L3 (`4.34 -> 5.90 -> 10.54`), which confirms that the long-horizon burden is mainly in rollout length.
- `llm_val` planning time stays in a relatively narrow band of roughly `3.4-4.4 s`, which is consistent with a one-shot startup overhead rather than a cost that scales linearly with rollout horizon.
- Because the shortlist export omits unified wall-clock rollout time, these logs support the planning-overhead argument directly, but not an exact total-inference-time row without re-mining the raw runtime traces.
