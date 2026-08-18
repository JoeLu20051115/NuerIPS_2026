# LOGIV Online Graph Repair Design

## Material Passport

- Origin Skill: brainstorming + academic-research-suite / experiment-agent
- Origin Date: 2026-08-08
- Verification Status: DESIGN_FIXED_IMPLEMENTATION_PENDING
- Version Label: logiv-online-graph-repair-v1

## 1. Objective

Build the requested LOGIV controller around the frozen pi0.5 LIBERO policy:

1. At episode start, use the existing `ScriptedProposalProvider`, oracle grounding,
   VAL, and DAG compiler to construct and certify the initial task graph.
2. Run pi0.5 with the unchanged BASE full-task prompt and action protocol.
3. Observe execution through the existing oracle-grounded Shadow path and project
   real progress onto the certified graph.
4. If no certified failure is present, do not alter BASE actions, requests, prompt,
   or termination.
5. If a failure is certified, discard only the unexecuted suffix of the current
   BASE action chunk, re-ground the current state, locally repair/reorder/add graph
   nodes through the existing bounded VAL repair path, and continue with the same
   pi0.5 client using prompts for the repaired frontier nodes.
6. A LIBERO `done=True` signal is an absorbing success and immediately ends the
   episode before any monitor-triggered intervention or settling action.
7. No action may execute after success, and later symbolic or evaluator output may
   not turn that success into failure.

The implementation remains explicitly a scripted-proposal/oracle-grounding LIBERO
benchmark implementation. It does not claim an API VLM or real-robot perception.

## 2. Approaches considered

### Selected: reuse Shadow detection and the certified controller

Connect the existing accepted initial Shadow proposal and graph tracker to the
existing bounded controller. BASE remains the nominal prefix. A confirmed Shadow
trigger creates one handoff snapshot; the current state is rebased into the same
task problem, repaired by the existing `RepairOperator`, compiled into a new DAG,
and executed by the same pi0.5 service.

This is the smallest path because proposal validation, grounding, graph projection,
VAL repair, prompt rendering, and pi0.5 macro execution already exist.

### Rejected: run FULL_LOGIV from step zero

This replaces BASE with per-node policy calls even on successful episodes and has
already produced negative flips. It violates the requirement that LOGIV leave BASE
unchanged when no failure is detected.

### Rejected: a separate recovery policy or extra action budget

A second checkpoint changes the method being tested. Giving LOGIV 520 BASE actions
plus extra recovery actions makes paired improvement uninterpretable. Recovery must
use the same pi0.5 client and the remainder of the shared 520-action budget.

## 3. Runtime state machine

```text
PREPARE
  scripted proposal -> oracle snapshot -> VAL certificate -> initial DAG
      |
      v
BASE_NOMINAL
  pi0.5 full-task prompt -> action chunk -> execute one action -> observe graph
      | done=True
      +------------------------------> SUCCESS (absorbing)
      | no confirmed deviation
      +------------------------------> BASE_NOMINAL
      | confirmed deviation
      v
REPAIR_HANDOFF
  discard pending chunk -> strict current snapshot -> native success check
      | already successful
      +------------------------------> SUCCESS (absorbing)
      | not successful
      v
LOCAL_REPAIR
  rebase problem -> bounded repair/VAL -> compile replacement DAG
      | uncertified / budget exhausted
      +------------------------------> FAIL
      | certified
      v
REPAIRED_EXECUTION
  same pi0.5 client -> repaired frontier prompt -> effect/goal gates
      | done=True or native success
      +------------------------------> SUCCESS (absorbing)
      | budget remains and another certified failure
      +------------------------------> LOCAL_REPAIR
      | otherwise
      +------------------------------> FAIL
```

## 4. Detection contract for all ten tasks

Every task uses its certified graph as the generic monitor contract. Task-specific
evidence rules may strengthen a decision but may not weaken generic safety checks.

A control handoff is allowed only for stable, oracle-grounded strong evidence:

- a previously achieved task goal regresses;
- an attributable manipulation attempt reaches its effect deadline without the
  declared graph effect;
- a registered task object is stably observed at an unplanned recovery surface
  after attributable manipulation; or
- the current active graph frontier is unchanged through the frozen progress
  deadline and a strict re-observation confirms the expected effect is false.

`UNKNOWN`, a stale certificate alone, a single anomalous observation, or weak
progress evidence alone never authorizes intervention. Grounding or monitor errors
fail open to BASE. Repair preparation and execution fail closed: no uncertified node
is dispatched.

## 5. Native success and action ordering

The episode loop owns terminal ordering:

1. execute one environment action;
2. read `done`;
3. if `done=True`, record success and return immediately;
4. otherwise update the graph monitor;
5. only then consider a handoff.

No settling action runs after `done=True`. A read-only `check_success()` audit may be
recorded, but disagreement cannot revoke native success. Before any repair dispatch,
the controller performs a read-only native success check so a late monitor callback
cannot cause post-success motion.

## 6. Budget and comparison contract

- Both BASE and LOGIV use the same frozen full pi0.5 checkpoint.
- Both arms use the same official initial state, policy seed, simulator seed,
  520-action global cap, and native success definition for each paired case.
- Initial planning, oracle grounding, VAL, and monitor compute are reported separately
  but do not consume physical-action budget.
- BASE actions and inference requests must be identical until the first certified
  handoff. The discarded pending chunk is recorded and never executed.
- LOGIV repair actions consume the remaining portion of the same 520-action cap.
- Per-case records include positive flip, negative flip, intervention reason,
  graph versions, repair edits, action/request counts, and success provenance.

## 7. Development and experiment sequence

### Feasibility gate

First run unit/integration tests plus a small simulator set containing known BASE
failures and protected BASE successes from every task. A candidate proceeds only if:

- no-trigger cases preserve exact BASE action-prefix parity;
- at least one known BASE failure becomes a LOGIV success;
- no protected success executes an action after native success;
- every intervention has strong evidence and a certified replacement graph; and
- combined BASE-prefix and repair actions never exceed 520.

### Failure-driven tuning

Replay previously observed BASE failures first and tune only monitor deadlines,
bounded repair choices, and repaired-node prompts. Re-run protected BASE successes
after every change. Keep a candidate only when cumulative net flips are positive;
never hide negative flips behind aggregate accuracy.

### Ten-task comparison

The repository exposes 50 official initial states per task. The requested 100 cases
per task are therefore defined as the Cartesian product of those 50 states and two
fixed policy master seeds. BASE and LOGIV are paired on `(task_id, episode_idx,
master_seed)`, yielding 1,000 pairs and 2,000 arm episodes.

The user requested tuning on this result set before checking the effect. Therefore
all 10x100 results are labeled development/tuning evidence, not independent final
confirmation. Report per task and overall:

- BASE and LOGIV successes out of 100;
- positive flips, negative flips, and net flips;
- Wilson intervals;
- paired success difference with a task-stratified bootstrap interval;
- intervention precision, repair certification rate, repair execution success,
  mean action cost, and graph edit counts.

Three GPUs run independent task shards. Episode-seeded request envelopes make each
case reproducible across processes; both arms for one paired case remain on the same
GPU shard and use the same derived seeds.

## 8. Acceptance boundary

The implementation is feasible when the simulator gate demonstrates a real positive
flip with no protocol violation. The 10x100 development comparison succeeds when:

- overall net flips are strictly positive;
- at least one prior BASE failure is repaired successfully;
- no task has an unexplained action after `done=True`;
- every intervention is backed by stored strong evidence and a VAL-certified graph;
- every paired case respects the shared 520-action cap; and
- all failures, negative flips, and tuning decisions remain present in artifacts.

An independent paper claim requires a later frozen run on new master seeds because
this first 10x100 matrix is used for tuning.
