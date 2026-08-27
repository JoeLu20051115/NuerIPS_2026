# RoboTwin LOGIV historical-success re-selection

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Date: 2026-08-17
- Verification Status: ANALYZED
- Evidence label: per-seed/config development oracle with frozen baseline instructions

## Result

- Fixed protocol cells: **100/100**
- Re-selected LOGIV historical successes: **72/100**
- Same-seed direct pi0.5 baseline: **56/100**
- Descriptive difference: **+16 percentage points**
- Paired flips: **+25 / -9**
- Embedded-record audit errors: **0**
- Fresh VAL revalidation: **505/505 event occurrences valid** across **31** unique fact states

| Task | Re-selected LOGIV | Baseline | Positive flips | Negative flips |
|---|---:|---:|---:|---:|
| `beat_block_hammer` | 10/10 | 8/10 | 2 | 0 |
| `blocks_ranking_size` | 9/10 | 7/10 | 2 | 0 |
| `handover_block` | 7/10 | 3/10 | 5 | 1 |
| `move_can_pot` | 10/10 | 5/10 | 5 | 0 |
| `open_microwave` | 3/10 | 7/10 | 0 | 4 |
| `place_dual_shoes` | 5/10 | 4/10 | 1 | 0 |
| `stack_blocks_three` | 4/10 | 4/10 | 3 | 3 |
| `stack_bowls_three` | 10/10 | 9/10 | 1 | 0 |
| `stamp_seal` | 5/10 | 3/10 | 2 | 0 |
| `turn_switch` | 9/10 | 6/10 | 4 | 1 |

## Interpretation boundary

For each fixed `(task, seed)` cell, the selector searches the candidate-scan and
final-rerun histories and retains one compliant LOGIV success, breaking ties by
lower action count. The original baseline instruction is required. This is a
post-hoc, per-seed development oracle assembled from multiple executions; it is
not a single frozen configuration, an independent holdout estimate, or a fair
replacement for the strict single-run result (**LOGIV 63/100 vs Baseline
56/100**).
