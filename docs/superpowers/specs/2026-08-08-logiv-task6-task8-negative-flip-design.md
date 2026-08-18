# LOGIV Task 6/8 Negative-Flip Design

## Goal

Eliminate the seven observed BASE-success to LOGIV-failure flips on LIBERO-10
tasks 6 and 8 without changing no-trigger BASE behavior, while retaining the
task-6 intervention-attributable recovery.

## Evidence and root cause

- Task 6 has three negative flips. Each is an `UNPLANNED_RECOVERY_SURFACE`
  trigger on the porcelain mug before either official goal has been achieved.
  The one positive flip occurs after the mug goal is achieved and the chocolate
  pudding remains on the recovery surface.
- Task 8 has four negative flips and no positive flips in the 1,000-pair run.
  Same-GPU prompt trials preserve the exact BASE prefix and repair trigger. The
  v5 phase prompts recover episodes 19, 1, and 32. Episode 14 remains a failure
  under both the old and simplified held-object prompts because the detector
  interrupts BASE while it is already carrying a moka pot.

## Considered approaches

1. Disable repair for tasks 6 and 8. This guarantees BASE parity but discards a
   verified task-6 recovery and does not improve task-8 repair capability.
2. Increase global confirmation or stall thresholds. This suppresses some late
   task-8 triggers but does not address task-6 recovery triggers and can remove
   valid recoveries on other tasks.
3. Add one task-6 eligibility condition and repair task-8 prompts by physical
   phase. This is selected because it matches the observed failure mechanisms
   and keeps the rest of LOGIV unchanged.

## Design

For task 6, the online detector receives a configurable
`recovery_requires_achieved_goal` flag. When enabled, recovery-surface evidence
cannot become a candidate until at least one positive or negative official goal
literal has previously been observed. The intervention is accepted only after
the strict oracle snapshot has also verified a goal milestone; advisory-only
progress cannot permanently unlock recovery. The evaluator enables the flag
only for task 6 and records the task-id list in `run.json`.

For task 8, retain prompt v5 and configure frontier-stall detection to require
explicitly true `handempty` with no true `holding` fact in both the advisory and
strict confirmation snapshots. Unknown hand state is not eligible. Holding an
object is verified manipulation progress, not sufficient evidence of a stall.
This task-scoped condition leaves the task-7 holding-state positive recovery
unchanged.

## Validation

1. Unit-test the task-6 eligibility condition with red-green TDD.
2. Re-run task-8 episode 14 on GPU 2/port 8030 with the handempty stall gate and
   require exact BASE continuation plus native success.
3. Re-run all four original task-8 negative flips with the winning prompt on
   GPU 2 and all four task-6 positive/negative flip cases with the new gate on
   GPU 0.
4. Require zero negative flips, retain the task-6 positive flip, exact parity
   for every no-trigger case, and no audit errors before expanding to all 200
   task-6/task-8 pairs.
