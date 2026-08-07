# LOGIV Online Graph Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a ten-task LOGIV arm that preserves the original pi0.5 BASE trajectory until a certified online graph deviation, then uses the same pi0.5 client and the remaining shared 520-action budget to execute a VAL-certified local graph repair.

**Architecture:** Extend the existing Shadow proposal/grounding/DAG path with one generic ten-task deviation detector and expose its first confirmed request to the nominal episode loop. The loop treats native `done=True` as absorbing success, otherwise hands the current observation to the existing rebased certified controller, which renders repaired-node prompts and uses the same episode-seeded pi0.5 client. Keep the older `FULL_LOGIV`, `SHADOW_LOGIV`, and diagnostic overlay arms intact.

**Tech Stack:** Python 3.11+, pytest, NumPy, LIBERO/robosuite, OpenPI WebSocket policy service, VAL, JSON/SHA-256 artifacts, Docker, three NVIDIA H200 GPUs.

## Global Constraints

- Initial planning uses `ScriptedProposalProvider`; online state uses `LiberoOracleGrounder`.
- The implementation must be labeled scripted/oracle development infrastructure, not API-VLM evidence.
- `done=True` is an absorbing success checked before monitoring, settling, repair, or any other physical action.
- No-trigger LOGIV must preserve BASE actions and policy requests exactly.
- A stale certificate, `UNKNOWN`, or one anomalous observation never authorizes repair.
- The unexecuted suffix of the active BASE chunk is discarded at handoff.
- BASE prefix plus repair execution must not exceed 520 physical actions.
- Repair uses the same episode-seeded pi0.5 client and repaired-node prompts; no second policy or extra action budget.
- Every replacement graph is VAL-certified before dispatch.
- The 10x100 matrix is development/tuning evidence because the user elected to tune on it.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Make native LIBERO success absorbing in both execution paths

**Files:**

- Modify: `src/pi05_libero_repro/protocol.py` (`EpisodeOutcome`, `run_episode`)
- Modify: `src/pi05_libero_repro/logiv/controller.py` (`ExecutorStatus`, controller outcome branch)
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py` (`Pi05MacroExecutor.await_outcome`)
- Test: `tests/test_protocol.py`
- Test: `tests/logiv/test_libero_adapter.py`
- Test: `tests/logiv/test_controller.py`

**Interfaces:**

- Consumes: the boolean `done` returned by each `env.step`.
- Produces: `ExecutorStatus.EPISODE_SUCCESS`, an absorbing `EpisodeOutcome.success`, and zero post-success physical actions.

- [ ] **Step 1: Write failing BASE-loop tests**

Add a fake environment whose `done` becomes true while `check_success()` deliberately returns false. Assert success remains true, settling is skipped, the intervention callback is not called for the terminal step, and no action follows it.

```python
def test_done_is_absorbing_before_monitor_or_settling() -> None:
    env = FakeEnv(succeed_on_policy_step=2)
    env.success_override = False
    intervention_steps: list[int] = []
    outcome = run_episode(
        env,
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        max_steps=12,
        settling_steps=4,
        intervention_monitor=lambda _obs, _action, step: (
            intervention_steps.append(step) or True
        ),
    )
    assert outcome.success is True
    assert outcome.done is True
    assert outcome.steps == 2
    assert intervention_steps == [1]
    assert len(env.actions) == env.wait_action_count + 2
```

- [ ] **Step 2: Run the BASE test and verify RED**

Run: `uv run pytest -q tests/test_protocol.py::test_done_is_absorbing_before_monitor_or_settling`

Expected: failure because current code performs settling and derives `success` from `check_success`.

- [ ] **Step 3: Write failing certified-executor tests**

Add an executor fixture where the first repaired-node action makes LIBERO return `done=True` before the node-local effect gate confirms. Assert the executor returns `EPISODE_SUCCESS`, performs no settling/re-observation, and the controller immediately returns `ControllerStatus.EPISODE_SUCCESS` without another evaluator or graph dispatch.

```python
assert attempt.executor_status is ExecutorStatus.EPISODE_SUCCESS
assert attempt.reason == "simulator task predicate became true"
assert controller_result.status is ControllerStatus.EPISODE_SUCCESS
assert controller_result.terminal_cause == "EPISODE_SUCCESS"
assert len(env.actions_after_native_done) == 0
```

- [ ] **Step 4: Run the executor/controller tests and verify RED**

Run: `uv run pytest -q tests/logiv/test_libero_adapter.py -k native_done_absorbing tests/logiv/test_controller.py -k native_done_absorbing`

Expected: failure because `EPISODE_SUCCESS` is not an executor status and effect-gated execution ignores `done`.

- [ ] **Step 5: Implement the minimum absorbing-success path**

In `run_episode`, check `done` immediately after `env.step` and before the Shadow or intervention callbacks. Skip settling when `done`, retain a read-only `check_success` audit, and set `success = done or check_success`.

Add `ExecutorStatus.EPISODE_SUCCESS`. In `Pi05MacroExecutor`, latch it immediately on any `done=True`, flush the local deque, skip settling/re-observation, and return it as a stopped outcome. In `ClosedLoopController`, return episode success before any post-stop grounding/effect gate when this status is received.

- [ ] **Step 6: Verify GREEN and regressions**

Run:

```bash
uv run pytest -q tests/test_protocol.py tests/logiv/test_libero_adapter.py tests/logiv/test_controller.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/pi05_libero_repro/protocol.py \
  src/pi05_libero_repro/logiv/controller.py \
  src/pi05_libero_repro/logiv/libero_adapter.py \
  tests/test_protocol.py tests/logiv/test_libero_adapter.py \
  tests/logiv/test_controller.py
git commit -m "fix(logiv): make native success absorbing"
```

### Task 2: Add a generic ten-task graph-deviation detector

**Files:**

- Create: `src/pi05_libero_repro/logiv/online_repair.py`
- Create: `tests/logiv/test_online_repair.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `tests/logiv/test_shadow_runtime.py`

**Interfaces:**

- Consumes: `TaskProblem`, accepted `CausalGraph`, advisory `FactSnapshot`, projected graph state, and strict snapshot confirmation.
- Produces: `OnlineRepairRequest` exactly once for stable `GOAL_REGRESSION`, `UNPLANNED_RECOVERY_SURFACE`, or `FRONTIER_STALL`; `ShadowRuntime.online_repair_request` exposes the latched request read-only.

- [ ] **Step 1: Write detector RED tests**

Test the three strong trigger classes and the fail-open cases:

```python
def test_recovery_surface_requires_repeated_true_and_strict_confirmation(): ...
def test_goal_regression_requires_prior_confirmed_goal(): ...
def test_frontier_stall_requires_min_step_and_unchanged_graph_window(): ...
def test_unknown_or_stale_certificate_alone_never_triggers(): ...
def test_detector_latches_only_the_first_request(): ...
```

Use a frozen return record:

```python
@dataclass(frozen=True)
class OnlineRepairRequest:
    kind: OnlineDeviationKind
    policy_step: int
    first_observed_step: int
    signature: tuple[str, ...]
    snapshot: FactSnapshot
    observation: Mapping[str, Any]
    source_graph_hash: str
    request_sha256: str
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/logiv/test_online_repair.py`

Expected: collection fails because `online_repair.py` does not exist.

- [ ] **Step 3: Implement the minimal detector**

Create `OnlineGraphDeviationDetector(problem, graph, *, confirmation_count, min_intervention_step, stall_steps)`. Reuse `FactSnapshot.truth`, the graph node statuses, and canonical JSON/SHA-256 helpers. Candidate semantics:

```python
class OnlineDeviationKind(str, Enum):
    GOAL_REGRESSION = "GOAL_REGRESSION"
    UNPLANNED_RECOVERY_SURFACE = "UNPLANNED_RECOVERY_SURFACE"
    FRONTIER_STALL = "FRONTIER_STALL"
```

- recovery: a true `(at object *_recovery_surface)` fact not present in the initial state or goal;
- regression: the exact signed goal was previously satisfied and is now reliably opposite, never `UNKNOWN`;
- stall: after `min_intervention_step`, the ordered action-node status signature and relevant-fact hash remain unchanged for `stall_steps`, at least one action is `READY`/`ACTIVE`, the goal is not satisfied, and strict confirmation shows the frontier effects remain false.

Require `confirmation_count` consecutive equal candidate signatures. Strict snapshot failure returns no request and clears the provisional streak.

- [ ] **Step 4: Integrate with accepted Shadow topology**

Add optional detector settings to `build_shadow_runtime`. After `ShadowGraphTracker.project`, feed the state and snapshot to the detector. For a candidate, call the proposal validation's strict snapshot reader and latch the first confirmed request in `ShadowRuntime.online_repair_request`. Append every decision to `deviation_trace`; do not alter the read-only `SHADOW_LOGIV` path when no detector is supplied.

- [ ] **Step 5: Verify GREEN and Shadow regressions**

Run:

```bash
uv run pytest -q tests/logiv/test_online_repair.py tests/logiv/test_shadow_runtime.py tests/logiv/test_shadow_graph_trace.py tests/logiv/test_shadow_monitor.py
```

Expected: all tests pass, including exact Shadow/Base parity tests.

- [ ] **Step 6: Commit**

```bash
git add src/pi05_libero_repro/logiv/online_repair.py \
  src/pi05_libero_repro/logiv/shadow_runtime.py \
  tests/logiv/test_online_repair.py tests/logiv/test_shadow_runtime.py
git commit -m "feat(logiv): detect online graph deviations"
```

### Task 3: Connect BASE handoff to certified local graph repair

**Files:**

- Modify: `src/pi05_libero_repro/logiv/evaluation.py`
- Modify: `src/pi05_libero_repro/protocol.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `tests/test_protocol.py`
- Modify: `tests/logiv/test_evaluator.py`
- Modify: `tests/logiv/test_records.py`

**Interfaces:**

- Consumes: `ShadowRuntime.online_repair_request`, `EpisodeOutcome.final_observation`, the same `EpisodeSeededClient`, and `remaining_actions = 520 - outcome.steps`.
- Produces: `MethodArm.LOGIV_ONLINE`, one immediate handoff, a certified replacement graph, and combined accounting bounded by 520.

- [ ] **Step 1: Write handoff RED tests**

Extend `EpisodeOutcome` with `discarded_pending_actions`. Assert a trigger at action 3 returns at action 3, skips settling, records the two remaining actions from a 5-action chunk as discarded, and never executes them.

Add evaluator tests that assert:

```python
assert MethodArm("LOGIV_ONLINE") is MethodArm.LOGIV_ONLINE
assert repair_max_total_action_steps == 520 - base_prefix_steps
assert combined_actions == base_prefix_steps + repair_steps <= 520
assert same_episode_client is repair_episode_client
assert initial_graph_hash != final_graph_hash  # when a repair is installed
```

Also test that no request yields a BASE-equivalent result and that a native-success prefix never enters `_execute_symbolic_arm`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/test_protocol.py -k discarded_pending tests/logiv/test_evaluator.py -k logiv_online`

Expected: failure because the arm and accounting do not exist.

- [ ] **Step 3: Add the method arm and CLI contract**

Add `LOGIV_ONLINE` plus:

```text
--online-monitor-interval-steps 5
--online-confirmations 3
--online-min-intervention-step 120
--online-stall-steps 120
```

Require `--oracle-grounding --development-only`. Do not require the Task 5/8 frozen monitor-contract registry for this arm. Record all values in `run.json`.

- [ ] **Step 4: Build the graph before the BASE prefix**

For `LOGIV_ONLINE`, call `_build_evaluator_shadow_runtime(..., monitor_contract=None, topology_only=True, online_detector_settings=...)` before `run_episode`. Its step-zero callback must accept and certify the initial proposal before the first BASE policy request. A rejected proposal fails open to an unchanged BASE episode and records `INITIAL_GRAPH_REJECTED`; it cannot intervene later.

- [ ] **Step 5: Perform the same-client bounded handoff**

Pass the runtime observer to `run_episode` and use an intervention callback that returns true only when `runtime.online_repair_request` is latched. After return:

1. call the native evaluator read-only;
2. if `outcome.done` or the native evaluator is successful, return absorbing success;
3. if no request, return the normal BASE result;
4. otherwise call `_execute_symbolic_arm(..., recovery_state=True)` with the same `episode_client`, the handoff observation, and `max_total_action_steps=520-outcome.steps`;
5. combine actions/requests/budgets and persist initial/final graph plus trigger evidence.

Never use `overlay_repair_max_steps` for this arm.

- [ ] **Step 6: Persist audit fields**

Add artifacts and record fields for trigger kind/step/hash, discarded actions, BASE-prefix action hash, initial/final graph hashes, repair actions, combined actions, and positive/negative flip derivation inputs. Preserve existing schema compatibility through defaults.

- [ ] **Step 7: Verify GREEN and integration regressions**

Run:

```bash
uv run pytest -q tests/test_protocol.py tests/logiv/test_evaluator.py \
  tests/logiv/test_records.py tests/logiv/test_shadow_runtime.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/pi05_libero_repro/logiv/evaluation.py \
  src/pi05_libero_repro/protocol.py scripts/eval_logiv_libero.py \
  src/pi05_libero_repro/logiv/records.py tests/test_protocol.py \
  tests/logiv/test_evaluator.py tests/logiv/test_records.py
git commit -m "feat(logiv): add online certified repair arm"
```

### Task 4: Add paired reporting and three-GPU launch support

**Files:**

- Modify: `scripts/run_logiv_eval.sh`
- Create: `scripts/report_logiv_online_pairs.py`
- Create: `scripts/run_logiv_online_3gpu.sh`
- Create: `tests/logiv/test_online_pair_report.py`

**Interfaces:**

- Consumes: BASE and LOGIV `episodes.jsonl` files from two master seeds.
- Produces: a validated 1,000-pair JSON/Markdown report and three deterministic task shards.

- [ ] **Step 1: Write report RED tests**

Create fixtures with positive, negative, both-success, and both-failure pairs. Require exact equality of task, episode index, master seed, initial-state hash, first-frame hash, checkpoint hash, and action cap before including a pair.

```python
assert report["positive_flips"] == 2
assert report["negative_flips"] == 1
assert report["net_flips"] == 1
assert report["paired_cases"] == 4
```

Reject duplicates, missing arms, mismatched hashes, action-budget violations, and any `LOGIV_ONLINE` record without a certified trigger artifact when intervention is claimed.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/logiv/test_online_pair_report.py`

Expected: collection fails because the report module does not exist.

- [ ] **Step 3: Implement the report**

Use the standard library plus existing Wilson/bootstrap helpers. Report per task and macro totals: successes, rates, Wilson intervals, positive/negative/net flips, paired difference, 10,000-repeat task-stratified bootstrap interval, intervention count, certified repair rate, repair success, and mean combined actions.

- [ ] **Step 4: Add deterministic three-GPU launcher**

Allow `LOGIV_ONLINE` in `run_logiv_eval.sh`. The launcher assigns task shards `(0,1,2,3)`, `(4,5,6)`, `(7,8,9)` to GPUs 0, 1, 2 and ports 8010, 8020, 8030. For each master seed, run BASE then LOGIV on the same shard/server so the checkpoint remains resident and each pair uses the same episode-seeded envelopes. Never overwrite an existing run directory.

- [ ] **Step 5: Verify GREEN and shell syntax**

Run:

```bash
uv run pytest -q tests/logiv/test_online_pair_report.py
bash -n scripts/run_logiv_eval.sh scripts/run_logiv_online_3gpu.sh
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_logiv_eval.sh scripts/run_logiv_online_3gpu.sh \
  scripts/report_logiv_online_pairs.py tests/logiv/test_online_pair_report.py
git commit -m "feat(logiv): add paired online evaluation"
```

### Task 5: Verify, run feasibility gates, tune, and execute 10x100 development comparison

**Files:**

- Create: `docs/experiments/2026-08-08-logiv-online-feasibility.md`
- Create: `results/logiv-online-development-10x100.json`
- Create: `results/logiv-online-development-10x100.md`
- Output: new run directories under `runs/logiv-online-*`

**Interfaces:**

- Consumes: the completed implementation, prior BASE failure records, three full pi0.5 servers, and two fixed master seeds.
- Produces: tested implementation, failure-driven tuning evidence, and a paired development report.

- [ ] **Step 1: Run repository verification**

Run:

```bash
uv run pytest -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 2: Build the feasibility manifest**

Read existing BASE records and select, for each task, prior BASE failures plus at least one protected success. Freeze `(task_id, episode_idx, master_seed, initial_state_sha256, first_frame_sha256)` before running LOGIV. Record selection as development evidence.

- [ ] **Step 3: Start three policy servers**

Use GPUs 0/1/2 and ports 8010/8020/8030 with the verified full checkpoint. Confirm each port reports episode RNG protocol v1 and each server log records the same checkpoint/norm-statistics hashes.

- [ ] **Step 4: Run failure-first feasibility cases**

Run BASE and LOGIV pairs for the frozen feasibility manifest. Proceed only when at least one BASE failure flips positive, no protected success violates absorbing-success or action-budget contracts, and every intervention artifact is certified.

- [ ] **Step 5: Tune conservatively on failures plus guards**

Adjust only online monitor timing, bounded repair selection, or repaired-node prompt overlays. After each candidate, rerun all accumulated positive guards. Record rejected variants and all negative flips. Stop tuning when cumulative net flips are positive and a complete guard pass succeeds.

- [ ] **Step 6: Freeze two master seeds and run 10x100**

Use 50 official initial states under each of two recorded master seeds. Run BASE and frozen LOGIV for all ten task shards. Monitor process liveness, output growth, GPU use, and server logs; do not silently retry invalid episodes.

- [ ] **Step 7: Validate and report**

Generate the paired JSON/Markdown report. Require exactly 1,000 valid pairs, exact pair hashes, all combined action counts <=520, and no post-success actions. Label the result `DEVELOPMENT_TUNING_EVIDENCE`.

- [ ] **Step 8: Final verification and commit code/docs**

Run the full suite and `git diff --check` again. Commit only source, tests, configs, report summaries, and experiment documentation; do not commit large videos or raw simulator states.

## Self-Review

- Spec coverage: initial certified DAG, unchanged BASE prefix, ten-task online tracking, stable strong triggers, local graph repair, same pi0.5 continuation, absorbing native success, shared 520-action cap, three GPUs, failure-first tuning, and 10x100 paired reporting are covered.
- Placeholder scan: no implementation step relies on TBD/TODO behavior.
- Type consistency: `OnlineRepairRequest`, `ExecutorStatus.EPISODE_SUCCESS`, `MethodArm.LOGIV_ONLINE`, and `EpisodeOutcome.discarded_pending_actions` have one definition and consistent consumers.
- Scope: no API VLM, separate recovery checkpoint, extra action budget, or unrelated refactor is included.
