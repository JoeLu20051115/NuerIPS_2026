# RoboTwin π₀.₅ Baseline 10×10 Design

## Goal

Reproduce a preliminary RoboTwin 2.0 baseline for ten selected tasks, using ten
different simulator scenes per task and the unified 50-task π₀.₅ checkpoint
`rhodes-team-teleai/pi05_TACO_robotwin2_finetuned`.

## Evaluation Scope

- Tasks: `handover_block`, `open_microwave`, `place_dual_shoes`, `stamp_seal`,
  `blocks_ranking_size`, `move_can_pot`, `turn_switch`, `stack_blocks_three`,
  `stack_bowls_three`, and `beat_block_hammer`.
- Task configuration: `demo_clean`.
- Trials: ten accepted simulator scenes per task, 100 evaluated episodes total.
- Seed group: `seed=0`; RoboTwin begins at simulator seed `100000` and advances
  deterministically, skipping only scenes for which the benchmark's expert
  feasibility check fails. Every evaluated episode therefore has a distinct,
  recorded simulator seed.
- Policy: the checkpoint's base π₀.₅ action policy with action horizon 50.
- Excluded: TACO CFN reranking, LOGIV, PDDL, GPT-4o, and any other intervention.
- Language instruction type: `unseen`, matching the existing evaluator default.
- Videos: disabled for this preliminary run to reduce I/O; per-episode seed and
  success are retained in logs/results.

## Minimal Adapter Change

Reuse RoboTwin's existing `script/eval_lerobot_torch_pi05.py` evaluator and its
`Lerobot_torch_PI05` wrapper. Add only three runtime inputs:

1. `policy_path` selects the local checkpoint instead of the current empty
   hard-coded path.
2. `test_num` selects ten trials instead of the current hard-coded 100.
3. `tokenizer_path` selects the local OpenPI PaliGemma SentencePiece model.

The tokenizer is loaded with the repository's `GemmaTokenizer` and passed to
the existing processor pipeline through its supported tokenizer override. A
preflight equivalence check has already established that it produces exactly
the same token IDs as OpenPI's SentencePiece path for a representative π₀.₅
prompt. Defaults preserve the evaluator's existing 100-trial behavior where
possible.

## Execution

Run one task per process and schedule at most one process per H200. Assign the
ten tasks across GPUs 0, 1, and 2 in deterministic list order. When a task
finishes, its GPU takes the next task. Each process writes to an isolated task
directory under a single timestamped baseline tag so outputs cannot collide.

Before the full run, execute one `handover_block` trial with the same checkpoint,
configuration, seed group, and tokenizer. This smoke trial must produce a
completed episode and a result file. It is diagnostic and is not included in
the final 10×10 aggregate. The exact resolved full-run command will be shown to
the user for confirmation before launch.

## Failure Handling and Monitoring

- Do not automatically retry a crashed task process.
- Report the command, exit code, and tail of its log before deciding what to do.
- Poll process liveness and GPU activity approximately every 30 seconds.
- Preserve partial task results and logs.
- Use a 12-hour per-task timeout because manipulation episode lengths differ by
  task; do not silently extend it.

## Verification and Reporting

Adapter tests must first fail against the current hard-coded implementation,
then pass after the minimal change. The smoke run must verify end-to-end model
loading, simulator setup, observation preprocessing, action execution, and
result persistence. The final report includes per-task successes out of ten,
macro-average success rate, total successes out of 100, evaluated simulator
seeds, checkpoint SHA-256, task configuration, GPU assignment, and failed or
incomplete processes if any.

