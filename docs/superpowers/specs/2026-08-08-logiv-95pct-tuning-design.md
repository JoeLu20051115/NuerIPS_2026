# LOGIV 95% Development-Tuning Design

## Goal

Raise the paired LIBERO-10 development result from 930/1000 to at least
950/1000 while preserving the LOGIV contract:

- BASE keeps the original pi0.5 protocol until a strictly verified deviation.
- The total low-level action budget remains 520 for both BASE and LOGIV.
- Native LIBERO `done=True` is absorbing success and no later action is allowed.
- Negative flips must remain zero.
- Every no-trigger pair must retain exact outcome, step, and policy-request parity.
- Initial planning remains scripted proposal with oracle grounding.

The 95% threshold is a development/tuning target, not an independent holdout
claim.

## Current evidence

The tuned result is 930/1000. The other nine tasks contribute 869 successes,
so task 8 must reach at least 81/100 if it is the only task changed.

Task 8 currently has 39 failures. Thirty-two already have a strictly verified
online trigger: 30 terminate through action-budget exhaustion and two through
post-stop grounding failure. Seven reach the BASE budget without a trigger.
The dominant failure pattern is therefore not missing symbolic plans. It is
late intervention plus repair prompts that spend most of the remaining budget
on acquisition without completing placement.

Historical task-8 development runs show that direct whole-action prompts and
the v26 contextual `remaining moka pot` recovery frontier can outperform the
current split acquisition prompt, but historical variants also produced
negative flips. They must therefore be screened under the current strict
handempty gate and paired protocol rather than copied wholesale.

## Considered approaches

### 1. Task-8 prompt and trigger tuning (selected)

Screen two task-8-only prompt overlays under the existing trigger times:

- `online-v6-direct`: remove acquire/finish phase splitting for the four moka
  pot placement actions and issue direct left/right placement instructions.
- `online-v7-contextual`: port only the task-8 v26 recovery-frontier language,
  including `remaining moka pot`, sibling protection, and a shared bounded
  `put both moka pots on the stove` recovery instruction.

After choosing the safer prompt, test task-8 stall thresholds 120 and 80 while
retaining three confirmations, five-step monitoring, minimum intervention step
120, and explicit strict `handempty` gating.

This route attacks both observed causes while changing only task 8.

### 2. Tune tasks 0/2/3/6/7/9 independently

These tasks have about 20 triggered failures in total. Reaching +20 through
this route would require nearly perfect recovery and creates a wider prompt
surface. It is retained only as a fallback if task 8 cannot reach 81/100.

### 3. Increase LOGIV's action budget

Rejected. It would make the 95% number incomparable with BASE and would hide
the late-trigger problem instead of fixing it.

## Experiment sequence

### Prompt screen

Run both prompt candidates on the 39 current task-8 failures and the 10 current
task-8 triggered successes. The failures measure recovery gain; the successes
are safety sentinels for negative flips. No-trigger successes do not execute a
repair prompt and are unchanged at this stage.

Choose candidates lexicographically by:

1. zero safety-sentinel negative flips;
2. most recovered current failures;
3. fewer repair actions and grounding errors.

### Trigger-timing screen

For the winning prompt, run all 100 task-8 pairs at stall thresholds 120 and
80. A full task run is required because earlier detection may create new
triggers on previously no-trigger BASE successes.

Choose the earliest threshold only if it has zero negative flips. Otherwise
choose the next safer threshold. Reject any candidate with parity errors,
record errors, or a post-success action.

### Fallback tuning

If the best task-8 configuration is below 81/100, preserve it and screen
task-scoped prompt overlays on the already-triggered failures of tasks
0/2/3/6/7/9. Apply the same zero-negative lexicographic gate. Do not change
BASE prompts or no-trigger behavior.

## Final validation

Merge only same-seed, same-checkpoint, same-initial-state paired records. The
final development report must satisfy all of the following:

- LOGIV successes at least 950/1000;
- positive flips minus negative flips at least +26 over BASE 924/1000;
- negative flips exactly zero;
- no-trigger outcome, step, and policy-request parity exact;
- errors empty;
- native success absorbing and total actions at most 520 in every episode.

The winning task-scoped configuration is then frozen with resolved hashes and
is eligible for a later unseen-seed holdout, which remains a separate claim.
