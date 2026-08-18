# LOGIV Online vs BASE

> This is **development/tuning evidence**, not independent holdout evidence.

Paired episodes: **1**. BASE: **0**; LOGIV Online: **1**.

Positive flips: **1**; negative flips: **0**; net flips: **+1**.

Intervention-attributable positive flips: **0**; no-trigger positive flips: **1**.

Failure-first feasibility: **0/1** prior BASE failures recovered.

No-trigger parity: outcome 0/1, steps 0/1, policy requests 0/1.

| Task | Paired | BASE | LOGIV | Positive | Attributable | Negative | Net |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 0 | 1 | 1 | 0 | 0 | +1 |

Task-stratified paired delta: **+1.000** (bootstrap 95% [+1.000, +1.000]).

## Audit errors

- unpaired records: missing_online=[(7, 6, 0), (7, 6, 12), (7, 6, 30), (7, 6, 40), (7, 6, 49)], missing_base=[]
- no-trigger outcome mismatch: (7, 6, 8)
- no-trigger step mismatch: (7, 6, 8)
- no-trigger policy-request mismatch: (7, 6, 8)
