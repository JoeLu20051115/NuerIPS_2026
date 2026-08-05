# Shadow Temporal Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the read-only Shadow graph track normal macro execution and post-settling official goals without native-result leakage.

**Architecture:** Preserve audited raw predicate evidence beside the exclusive planning projection, add a stateful fixed-graph tracker, admit monitor-only unlocated transport snapshots, and observe every settling step through an isolated callback. Strict controller grounding and Base execution remain unchanged.

**Tech Stack:** Python 3.11, frozen dataclasses, NumPy, pytest, LIBERO simulator, existing LOGIV graph and grounding types.

## Global Constraints

- The graph structure, Base prompt, Base policy requests, Base actions, and settling actions must remain unchanged.
- Shadow must not read `env.check_success()` or the native episode result.
- Strict controller grounding must continue enforcing exactly-one state.
- R2M takeover and recovery-policy execution remain out of scope.
- Use the existing random-100 seed manifest as the only simulator regression set.

---

### Task 1: Preserve raw evidence and monitor-only transport snapshots

**Files:**
- Modify: `src/pi05_libero_repro/logiv/model.py`
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Test: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Produces: `FactSnapshot.raw_truth(fact: Fact) -> TruthValue`.
- Produces: `LiberoOracleGrounder.peek_advisory_partial_snapshot()` that admits zero confirmed movable states while retaining explicit FALSE evidence.
- Preserves: `ground(...)` and `peek_snapshot()` strict exactly-one behavior.

- [x] **Step 1: Write failing tests for raw dominance evidence and all-false advisory transport**

Add assertions that an audited target changed to normalized FALSE by
`reliable-holding-over-at` has `raw_truth(target) is TruthValue.TRUE`, and that
an all-false movable group succeeds only through
`peek_advisory_partial_snapshot()`.

- [x] **Step 2: Run the focused tests and verify the old behavior fails**

Run: `pytest -q tests/logiv/test_libero_adapter.py -k 'raw_truth or advisory_partial_snapshot_rejects_exactly_one'`

Expected: the raw-truth test fails because the method is absent and the
all-false advisory test fails because it raises `GroundingError`.

- [x] **Step 3: Implement the minimal evidence and advisory behavior**

Parse the already validated `dominance_overrides` list in `FactSnapshot` and
return TRUE for an overridden target; otherwise return `truth(fact)`.  In the
grounder, bypass zero-confirmed movable exactly-one failure only when
`advisory_partial=True`; keep conflicting multi-TRUE groups and strict calls
unchanged.

- [x] **Step 4: Run focused model/grounder tests**

Run: `pytest -q tests/logiv/test_libero_adapter.py tests/logiv/test_recovery_records.py`

Expected: all selected tests pass.

- [x] **Step 5: Commit Task 1**

Run: `git add src/pi05_libero_repro/logiv/model.py src/pi05_libero_repro/logiv/libero_adapter.py tests/logiv/test_libero_adapter.py && git commit -m "fix(logiv): preserve shadow transport evidence"`

### Task 2: Add the temporal fixed-graph tracker and nominal reconciliation

**Files:**
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_monitor.py`
- Test: `tests/logiv/test_shadow_graph_trace.py`
- Test: `tests/logiv/test_shadow_monitor.py`
- Test: `tests/logiv/test_shadow_runtime.py`

**Interfaces:**
- Produces: `ShadowGraphTracker(graph: CausalGraph, problem: TaskProblem)` and `project(snapshot, *, policy_step, observation_generation, certificate_state, phase, settling_step=None)`.
- Preserves: `project_graph_state(...)` as a stateless compatibility wrapper.
- Produces: schema-aware in-flight reconciliation for `place-on`, `place-in`, and `place-relative`.

- [x] **Step 1: Write failing temporal graph tests**

Use a placement graph and four snapshots: source/handempty, holding, raw-target
plus holding dominance, and released target.  Assert node states READY, ACTIVE,
EFFECT_OBSERVED, and COMPLETED, and assert raw goal completion during the third
snapshot.

- [x] **Step 2: Write failing certificate-envelope tests**

Assert that source-to-holding and holding-to-target changes for one ready macro
keep `CertificateState.CURRENT`, while target-to-absent after completion becomes
`CertificateState.STALE`.

- [x] **Step 3: Run focused tests and verify they fail**

Run: `pytest -q tests/logiv/test_shadow_graph_trace.py tests/logiv/test_shadow_monitor.py -k 'temporal or macro_transport'`

Expected: failures show missing `ShadowGraphTracker` and current stateless
reconciliation.

- [x] **Step 4: Implement the minimal temporal tracker**

Track only statuses and the previous snapshot.  Recognize active internal
transport by schema and the action object.  Use normalized facts for READY and
COMPLETED, `raw_truth` for EFFECT_OBSERVED and GOAL, and preserve graph node
ordering in every emitted state.

- [x] **Step 5: Implement schema-aware certificate reconciliation**

Maintain an in-flight node set.  Admit changes to the manipulated object's
registered locations, holding, and handempty while a matching ready macro is
in flight.  Remove the node after normalized declared effects complete; do not
cover later regression.

- [x] **Step 6: Wire one tracker instance into each Shadow runtime**

Create the tracker after initial certification and use it from `record_state`
instead of calling the one-shot projector for every sample.

- [x] **Step 7: Run the focused Shadow test modules**

Run: `pytest -q tests/logiv/test_shadow_graph_trace.py tests/logiv/test_shadow_monitor.py tests/logiv/test_shadow_runtime.py`

Expected: all selected tests pass.

- [x] **Step 8: Commit Task 2**

Run: `git add src/pi05_libero_repro/logiv/shadow_runtime.py src/pi05_libero_repro/logiv/shadow_monitor.py tests/logiv/test_shadow_graph_trace.py tests/logiv/test_shadow_monitor.py tests/logiv/test_shadow_runtime.py && git commit -m "feat(logiv): track temporal shadow topology"`

### Task 3: Observe the post-action settling barrier

**Files:**
- Modify: `src/pi05_libero_repro/protocol.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `scripts/eval_logiv_libero.py`
- Test: `tests/test_protocol.py`
- Test: `tests/logiv/test_shadow_runtime.py`

**Interfaces:**
- Produces: frozen `ShadowSettlingContext` with observation, policy step,
  settling step, and total settling steps.
- Adds: optional `shadow_settling_observer` argument to `run_episode`.
- Adds: `ShadowRuntime.settling_observer` for topology-only tracing.

- [x] **Step 1: Write a failing isolated-settling callback test**

Run a transient-success fake environment with two settling steps.  Assert two
ordered settling contexts, deep-copy isolation, preserved Python/NumPy RNG,
unchanged Base actions, and unchanged post-settling failure.

- [x] **Step 2: Write a failing runtime terminal-regression test**

Feed a policy snapshot with raw goal truth followed by settling snapshots that
lose the target.  Assert the final trace phase is SETTLING and final GOAL is
BLOCKED.

- [x] **Step 3: Run focused tests and verify they fail**

Run: `pytest -q tests/test_protocol.py tests/logiv/test_shadow_runtime.py -k 'settling_observer or settling_regression'`

Expected: failures show the missing callback interface.

- [x] **Step 4: Implement contained settling callbacks**

After each existing dummy settling action, invoke the optional observer with a
deep-copied observation.  Restore Python and NumPy RNG states, contain callback
exceptions in Shadow accounting, and never expose the native success value.

- [x] **Step 5: Connect settling samples to the same tracker**

Add a topology-only settling observer that reads a snapshot, reconciles it,
and appends `phase="SETTLING"` with a one-based settling index.  Pass it from
`scripts/eval_logiv_libero.py` to `run_episode`.

- [x] **Step 6: Run protocol and Shadow tests**

Run: `pytest -q tests/test_protocol.py tests/logiv/test_shadow_runtime.py tests/logiv/test_shadow_graph_trace.py`

Expected: all selected tests pass.

- [x] **Step 7: Commit Task 3**

Run: `git add src/pi05_libero_repro/protocol.py src/pi05_libero_repro/logiv/shadow_runtime.py scripts/eval_logiv_libero.py tests/test_protocol.py tests/logiv/test_shadow_runtime.py && git commit -m "feat(logiv): trace shadow through settling"`

### Task 4: Verify and rerun the fixed random-100 regression

**Files:**
- Create: `results/logiv-shadow-temporal-random100-20260805.json`
- Create: `results/logiv-shadow-temporal-random100-20260805.md`
- Create: `scripts/report_shadow_temporal_random100.py`
- Create: `runs/shadow-logiv-random100-temporal-final-v4-20260805/` runtime artifacts (ignored)
- Create: `runs/base-random100-paired-v4-20260805/` runtime artifacts (ignored)

**Interfaces:**
- Consumes: unchanged `runs/shadow-logiv-random100-20260804/seed_manifest.json`.
- Produces: paired Base/new-Shadow audit with action hashes, outcomes, trace coverage, trace errors, certificate state, and terminal confusion matrix.

- [x] **Step 1: Run the complete local test suite**

Run: `pytest -q`

Expected: zero failures.

- [x] **Step 2: Launch the two existing policy-server shards**

Use the repository's existing evaluation launcher and the same checkpoint,
ports 8010/8020, and device allocation recorded by the original random-100 run.
Verify each server metadata endpoint before simulation.

- [x] **Step 3: Run new SHADOW_LOGIV and paired Base cases from the frozen manifest**

For each of the 100 manifest entries, use its stored task ID, episode index,
master seed, policy seed derivation, simulator seed derivation, topology-only
mode, 10 settling steps, and original policy-server shard.  Write Shadow into
the new run directory, then run paired Base on the same live servers so action
hashes can be compared without cross-restart policy nondeterminism.  Do not run
legacy full intervention.

- [x] **Step 4: Audit every paired case**

Compare each new Shadow `base_execution.json` with the paired Base record.  The
required checks are exact steps, done signal, post-settling success, policy
request count, action SHA-256, 100 valid records, zero aggregate Shadow errors,
and one trace sample per initial/policy/settling callback.

- [x] **Step 5: Compute terminal topology results**

Use the last SETTLING graph sample for every case.  Report the confusion matrix
against post-settling native success, per-task accuracy, node transition counts,
and any remaining `PRECONDITION_UNKNOWN`, `STALE`, trace error, or gap.

- [x] **Step 6: Update the result report and commit**

Run: `git add results/logiv-shadow-temporal-random100-20260805.json results/logiv-shadow-temporal-random100-20260805.md scripts/report_shadow_temporal_random100.py docs/superpowers/specs/2026-08-05-shadow-temporal-topology-design.md docs/superpowers/plans/2026-08-05-shadow-temporal-topology.md && git commit -m "results(logiv): validate temporal shadow topology"`

Expected: the report states measured results only; if any gate fails it records
the exact residual cases rather than claiming readiness for R2M takeover.
