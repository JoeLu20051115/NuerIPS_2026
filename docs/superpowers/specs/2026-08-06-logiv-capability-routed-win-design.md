# LOGIV Capability-Routed Win Design

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-06
- Verification Status: DESIGN_REVIEWED_IMPLEMENTATION_UNVERIFIED
- Version Label: logiv-capability-routed-v1

## 1. Objective and acceptance boundary

Build one runnable LOGIV arm that exceeds the frozen pi0.5 BASE without relabeling diagnostic
overlays or granting extra policy-action budget. The first accepted result must satisfy all of the
following:

- identical LIBERO-10 task definitions, checkpoint, 520-action cap, episode states, simulator seed,
  policy seed, and final native evaluator;
- a frozen task-to-controller capability contract selected before confirmation;
- strictly more paired native successes than BASE;
- zero BASE-success to LOGIV-failure flips in the confirmation sample;
- complete provenance for controller choice, prompts, configs, seeds, actions, and artifacts; and
- a fresh independent confirmation policy master seed, fixed here as `20260806`.

Development evidence is not a final claim. The existing seed-7 Task 4 run is used only to choose the
candidate. Confirmation begins only after code, configs, and the routing contract are frozen.

## 2. Evidence and corrected diagnosis

The current comparable development matrix is `FULL_LOGIV=427/500` and `BASE=437/500`. Applying
LOGIV everywhere is therefore rejected.

One bounded candidate is already supported by paired development evidence: Task 4 v68 produced
`50/50` versus BASE `49/50`, with 50/50 matching initial-state hashes, 50/50 matching first-frame
hashes, one positive flip (episode 45), and zero negative flips. V68 changes only the first Task 4
prompt to `Put the solid white mug on the left plate.` and enables ten-step place-effect
stabilization. The causal graph remains width two.

The failed Task 5 terminal-held design is not continued. In all three frozen failures, the target
became true during early settling and regressed around settling step six; the final strict snapshot
then had no confirmed registered location. These roots are unstable-placement/goal-regression
states, not terminal held-book states.

## 3. Approaches considered

### 3.1 Selected: capability-routed LOGIV with BASE fallback

A versioned router enables FULL_LOGIV v68 only for Task 4. Tasks 0--3 and 5--9 execute the existing
BASE path directly, without proposal, grounding, VAL, graph construction, subtask dispatch, or
recovery. This produces a single deployable method arm while limiting experimental risk to the one
task class with positive paired development evidence.

### 3.2 Deferred: settling goal-regression recovery for Task 5

Extend settling observations through the typed `GOAL_REGRESSION` evidence path, capture the first
reliable regressed state, and train or validate a stable re-insertion macro. This is the next tranche
only if Task 4 fails independent confirmation or if a larger margin is required.

### 3.3 Rejected: terminal held-book recovery

The mandatory preflight yielded `0/3` eligible held roots and `go=false`. Training the registered
held-state policy would not cover the observed failures and is prohibited.

## 4. Runtime architecture

Add `LOGIV_CAPABILITY_ROUTED` as a distinct method arm and load one self-hashed JSON contract. The
contract contains exactly one enabled entry:

```text
task 4
  coverage: libero10-coverage-v38-task0-recovery.json
  proposal: libero10-scripted-proposals-v53-task6-order.json
  prompt: pi05-subtasks-v68-task4-left.json
  effect confirmation: 5
  place stabilization: 10
  frontier follow-up: 5
  recovery-only completion follow-up: 120
  fallback follow-up: 120
  per-attempt cap: 260
  shared cap: 520
```

For an enabled task, the evaluator executes the existing certified FULL_LOGIV path with the exact
contract values. For every other task, it calls the existing BASE episode runner with the official
task instruction, 520-action cap, and ten settling steps. The record keeps the outer method arm and
adds `selected_controller=FULL_LOGIV_V68` or `selected_controller=BASE_FALLBACK` plus the contract
hash. BASE fallback must emit zero LOGIV proposal requests, physical LOGIV attempts, repair rounds,
VAL calls, or recovery requests.

The router is a static capability registry, not an outcome oracle. It may use only `task_id`; it may
not inspect episode index, seed, observation, Base outcome, or evaluation result when selecting a
controller.

## 5. Data flow and failure containment

```text
frozen task + episode + seeds
        -> self-hashed capability router
        -> task 4: certified FULL_LOGIV v68
        -> other tasks: unchanged BASE runner
        -> native ten-step settling and LIBERO success predicate
        -> paired record + provenance audit
```

Unknown task IDs, hash mismatches, undeclared runtime overrides, or invalid contract values fail
before simulator reset. A LOGIV preparation failure on the enabled task remains an invalid LOGIV
episode; it must not silently fall back after observing state. Disabled tasks select BASE before any
observation and cannot enter LOGIV later.

## 6. Verification and experiment gates

1. Unit tests first prove contract hashing, task-only routing, undeclared override rejection, and
   zero LOGIV accounting on BASE fallback.
2. Existing adapter tests prove v68 resolves from the tracked parent config and place stabilization
   resumes execution when an initially observed placement rebounds.
3. Reproduce the seed-7 Task 4 hard roots and paired episode 45 before confirmation.
4. Freeze code/config/contract and run Task 4 BASE versus routed LOGIV for all 50 episode indices at
   policy master seed `20260806`. Require matching initial-state/first-frame hashes, zero negative
   flips, and at least one positive flip.
5. If Task 4 passes, run the routed arm and BASE over the full 10x50 confirmation matrix. Disabled
   tasks must additionally show exact action-prefix parity. Report per-task counts, paired flips,
   Wilson intervals, and a 10,000-repeat task-stratified paired bootstrap interval.
6. If Task 4 does not pass strictly, do not change the confirmation seed or tune v68 on those
   outcomes. Archive the result and begin the separate Task 5 goal-regression recovery design.

## 7. Scope limits

This tranche does not train a new policy, alter BASE, add an episode-index exception, reuse the
failed held-book permit, or claim real-robot perception. Oracle grounding remains explicitly
development/benchmark infrastructure for the one enabled FULL_LOGIV task. A paper claim must name
the controller as capability-routed LOGIV and report the routing contract and confirmation protocol.
