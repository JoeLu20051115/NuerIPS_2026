# VAL-priority Top-50 Per Level

Composite score: `0.45 * success + 0.35 * task_progress + 0.20 * normalized_l2_quality` (within-level min-max normalization for L2 quality).

Selection policy for exact 50 per level:
1. `strict_monotonic`: `val > dual > task_token_only` by composite score.
2. `val_top_only`: `val` has highest composite score, but `dual <= task_token_only`.
3. `dual_mid_only`: `dual > task_token_only`, but `val <= dual`.
4. `other`: closest remaining cases, ranked by smallest monotonicity violation penalty.

## Availability And Selected Mix

| Level | Strict Available | VAL Top Available | Dual>Token Available | Selected Strict | Selected VAL-only | Selected Dual-mid | Selected Other | Avg VAL Score | Avg Dual Score | Avg Token Score | Avg (VAL-Dual) | Avg (Dual-Token) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 22 | 36 | 57 | 22 | 14 | 14 | 0 | 0.8295 | 0.7661 | 0.7229 | 0.0634 | 0.0432 |
| L2 | 14 | 25 | 49 | 14 | 11 | 25 | 0 | 0.8061 | 0.7332 | 0.7015 | 0.0728 | 0.0317 |
| L3 | 12 | 22 | 50 | 12 | 10 | 28 | 0 | 0.7363 | 0.7175 | 0.6526 | 0.0189 | 0.0649 |

## Notes

- Exact `50/50/50` with strict `val > dual > task_token_only` is impossible on the final full-run results; available strict counts are `L1=22`, `L2=14`, `L3=12`.
- This shortlist therefore maximizes `VAL` priority first, then preserves the descending trend `val > dual > token` whenever possible, and finally fills the remainder with the smallest monotonicity violations.
- The CSV contains per-sample composite scores and raw metrics for all three modes.
