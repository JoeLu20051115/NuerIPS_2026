# RoboTwin 2.0 pi0.5 baseline — 10 tasks × 10 episodes

## Outcome

- Overall success: **41/100 = 41.0%**
- Completed tasks: **10/10**
- Completed evaluated episodes: **100/100**
- Worker exit status: **GPU 0 = 0, GPU 1 = 0, GPU 2 = 0**
- Verification: **0 consistency errors**, 10 result files, 0 videos, 0 active evaluator processes

| Task | Success | Accuracy |
|---|---:|---:|
| `handover_block` | 0/10 | 0% |
| `open_microwave` | 2/10 | 20% |
| `place_dual_shoes` | 2/10 | 20% |
| `stamp_seal` | 4/10 | 40% |
| `blocks_ranking_size` | 4/10 | 40% |
| `move_can_pot` | 7/10 | 70% |
| `turn_switch` | 4/10 | 40% |
| `stack_blocks_three` | 4/10 | 40% |
| `stack_bowls_three` | 6/10 | 60% |
| `beat_block_hammer` | 8/10 | 80% |

## Evaluation configuration

- Benchmark: RoboTwin 2.0
- Policy: pure pi0.5 inference baseline
- Checkpoint: `rhodes-team-teleai/pi05_TACO_robotwin2_finetuned`
- Local checkpoint: `/mnt/data3/data_xingrui/lueq/NuerIPS_2026/artifacts/checkpoints/pi05_TACO_robotwin2_finetuned`
- Checkpoint SHA-256: `5af5866f0e5f2ca446ee28d935b0dfc07c72031b455a2318e79249b3278ab87a`
- Checkpoint setting: `unified_50tasks`
- Scene config: `demo_clean`
- Instruction type: `unseen`
- Initial seed group: `0` (base simulator seed `100000`)
- Evaluated episodes per task: `10`
- Action chunk horizon: `50`
- Online additions: no LOGIV, no PDDL, no VLM/GPT-4o, no TACO/CFN reranking
- Videos: disabled
- Evaluator implementation commit: `39e84f4`

The checkpoint name contains `TACO` because it is the supplied fine-tuned weight. “Baseline” here means direct pi0.5 rollout using that checkpoint without any online planning, verification, reranking, or correction layer.

## Accepted simulator seeds

RoboTwin's official expert pre-check skips seeds whose scene/expert rollout is invalid. Therefore each task has 10 distinct accepted seeds, but the accepted lists are not always exactly `100000`–`100009`.

| Task | Accepted seeds |
|---|---|
| `handover_block` | 100000, 100002, 100003, 100004, 100006, 100007, 100008, 100009, 100010, 100011 |
| `open_microwave` | 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100010, 100011, 100013 |
| `place_dual_shoes` | 100001, 100002, 100007, 100008, 100009, 100012, 100013, 100019, 100020, 100021 |
| `stamp_seal` | 100000, 100001, 100004, 100005, 100006, 100007, 100008, 100010, 100011, 100013 |
| `blocks_ranking_size` | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009 |
| `move_can_pot` | 100000, 100001, 100002, 100003, 100005, 100006, 100007, 100008, 100009, 100010 |
| `turn_switch` | 100000, 100001, 100002, 100004, 100006, 100007, 100008, 100009, 100010, 100011 |
| `stack_blocks_three` | 100000, 100001, 100002, 100003, 100004, 100005, 100006, 100008, 100009, 100010 |
| `stack_bowls_three` | 100000, 100001, 100002, 100004, 100006, 100008, 100009, 100015, 100016, 100017 |
| `beat_block_hammer` | 100002, 100003, 100005, 100006, 100007, 100008, 100009, 100010, 100011, 100012 |

Machine-readable values are in `per_task.csv`. Raw task logs and RoboTwin-native
`_result.txt` files remain local because they contain bulky simulator output.
The evaluator change is reproducible from
`patches/robotwin/0001-fix-robotwin-parameterize-pi05-baseline-evaluation.patch`.
