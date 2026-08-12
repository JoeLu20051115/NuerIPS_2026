# LOGIV Online vs BASE

> This is **development/tuning evidence**, not independent holdout evidence.
>
> This is a fixed-seed, per-episode parameter portfolio rather than one frozen
> task-level configuration. Task 4 / seed 7 / episodes 34 and 42 are excluded
> because same-card no-trigger reruns also succeeded, so their gains were not
> attributable to LOGIV repair.

Paired episodes: **1000**. BASE: **924**; LOGIV Online: **950**.

Positive flips: **26**; negative flips: **0**; net flips: **+26**.

Intervention-attributable positive flips: **26**; no-trigger positive flips: **0**.

Failure-first feasibility: **26/76** prior BASE failures recovered.

No-trigger parity: outcome 899/899, steps 899/899, policy requests 899/899.

| Task | Paired | BASE | LOGIV | Positive | Attributable | Negative | Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 93 | 95 | 2 | 2 | 0 | +2 |
| 1 | 100 | 99 | 99 | 0 | 0 | 0 | +0 |
| 2 | 100 | 98 | 99 | 1 | 1 | 0 | +1 |
| 3 | 100 | 97 | 98 | 1 | 1 | 0 | +1 |
| 4 | 100 | 94 | 96 | 2 | 2 | 0 | +2 |
| 5 | 100 | 100 | 100 | 0 | 0 | 0 | +0 |
| 6 | 100 | 94 | 95 | 1 | 1 | 0 | +1 |
| 7 | 100 | 96 | 99 | 3 | 3 | 0 | +3 |
| 8 | 100 | 61 | 74 | 13 | 13 | 0 | +13 |
| 9 | 100 | 92 | 95 | 3 | 3 | 0 | +3 |

Task-stratified paired delta: **+0.026** (bootstrap 95% [+0.017, +0.036]).
