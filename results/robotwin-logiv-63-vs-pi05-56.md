# RoboTwin 2.0 frozen 10x10 rerun

- Evidence label: **development/frozen-rerun**
- Observed LOGIV result: **63/100**
- Same-cell direct pi0.5 result: **56/100**
- Paired flips: **+17 / -10**
- Completed cells: **100/100 per method**

| Task | LOGIV | pi0.5 |
|---|---:|---:|
| `beat_block_hammer` | 10/10 | 8/10 |
| `blocks_ranking_size` | 7/10 | 7/10 |
| `handover_block` | 7/10 | 3/10 |
| `move_can_pot` | 10/10 | 5/10 |
| `open_microwave` | 3/10 | 7/10 |
| `place_dual_shoes` | 4/10 | 4/10 |
| `stack_blocks_three` | 3/10 | 4/10 |
| `stack_bowls_three` | 9/10 | 9/10 |
| `stamp_seal` | 3/10 | 3/10 |
| `turn_switch` | 7/10 | 6/10 |

This is the fresh frozen rerun of the fixed 100 cells. Nine successes in an
earlier post-hoc 72/100 historical collection did not reproduce, so the live
rerun result is 63/100. The historical selected records are intentionally not
retained here. A new live run can vary because GPU execution, simulation, and
the GPT-4o service are not bitwise deterministic.
