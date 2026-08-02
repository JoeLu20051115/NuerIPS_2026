# LOGIV v28 all-task development smoke

This is a one-seed-per-task development smoke test on the frozen LIBERO-10
task IDs. It is an engineering regression gate, not a success-rate estimate,
not a paired Base comparison, and not holdout evidence.

## Configuration

- Backbone: full `pi05_libero` checkpoint
- Goal mode: `METADATA_ASSISTED`
- Run: `runs/logiv-v28-all10-smoke-e0/`
- Prompt: `configs/logiv/prompts/pi05-subtasks-v28.json`
- Episode index: `0` for each task
- Low-level action budget: 520 per episode
- Grounding: simulator/oracle development adapter with three-valued facts

## Result

| Task | Success | Steps | Initial DAG width | Action-to-action edges |
| ---: | :---: | ---: | ---: | ---: |
| 0 | yes | 274 | 2 | 0 |
| 1 | yes | 257 | 2 | 0 |
| 2 | yes | 227 | 2 | 0 |
| 3 | yes | 216 | 1 | 2 |
| 4 | yes | 237 | 2 | 0 |
| 5 | no | 520 | 1 | 1 |
| 6 | yes | 221 | 2 | 0 |
| 7 | yes | 239 | 2 | 0 |
| 8 | yes | 376 | 2 | 0 |
| 9 | yes | 368 | 1 | 1 |

The smoke result is `9/10`, up from v27's `8/10` on the same episode index.
Task 6 changed from failure to success and no previously successful smoke task
regressed. Task 6 was also reproduced in the targeted run
`runs/logiv-v28-task5-6-e0/` with the same 221-step success.

Task 6 keeps the two nominal actions causally independent (`width=2`, no
action-to-action edge). Its improvement comes from registering a real living
room table recovery state and using action-grounded language that protects the
already placed mug while moving the pudding. It does not come from inserting a
synthetic precedence edge. Tasks 3, 5, and 9 have genuine support chains in
their fixed action semantics; the compiler does not add edges merely to make a
linear agenda.

Task 5 remains a development failure. Splitting acquisition from precise
placement made the failing stage observable, but neither a longer effect
confirmation window nor the current stable-placement prompt produced success.
This task must remain in the denominator in subsequent paired evaluation.

## Claim boundary

This smoke test establishes cross-task execution feasibility and catches one
regression. It does not establish that v28 improves the ten-task macro-average.
That claim requires matched Base/Full episodes for every task, task-wise
Wilson intervals, and the pre-registered paired within-task bootstrap.
