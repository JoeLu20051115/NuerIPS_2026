# LOGIV R2M Phase 0 Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-open initial LOGIV proposal, a read-only local shadow monitor, reproducible recovery-root artifacts, and mechanically enforced train/dev/held-out isolation while keeping every Base action and Base policy request unchanged.

**Architecture:** `run_episode` remains the only nominal rollout loop and gains an observation-only callback whose return value is ignored and whose failures are contained. `SHADOW_LOGIV` runs that exact loop after one separately accounted initial proposal; accepted proposals may enable local grounding and recovery-state capture, but Phase 0 has no action handoff or recovery policy. Recovery snapshots use exact `root_id` values for state reproduction and broader `recovery_group_id` values for leakage-safe splitting.

**Tech Stack:** Python 3.11, dataclasses, NumPy, pytest, existing LIBERO/MuJoCo adapter, existing VAL wrapper, Bash/Docker evaluation launcher.

## Global Constraints

- Execute from a clean isolated worktree created from commit `8254a26b` with `superpowers:using-git-worktrees`; do not modify or copy the dirty experimental files in the original worktree.
- Add no Python runtime dependency beyond the repository's existing dependencies.
- `BASE` must not construct a proposal provider, grounder, monitor, or recovery artifact writer.
- `SHADOW_LOGIV` must send the same prompt, action history, policy RNG envelope, number of Base policy requests, and action count as `BASE` for the same task, episode, and seed.
- The shadow callback receives copies of observations/actions, returns no control decision, and cannot terminate or alter the nominal rollout.
- Phase 0 uses only the local synchronous oracle grounder. It records `shadow_vlm_requests=0`; external VLM monitoring remains a future asynchronous, non-blocking extension.
- Exactly one initial proposal request is permitted in `SHADOW_LOGIV`. Proposal or certification rejection disables monitoring and recovery-root capture, records the reason, and still runs Base.
- Phase 0 never invokes a recovery policy and records `recovery_policy_requests=0`.
- All recovery roots collected from the already inspected 50 LIBERO initial states are development data; no held-out performance claim is permitted from them.
- This plan implements only the data foundation. Recovery-policy training/evaluation and capability-gated atomic runtime handoff require separate Phase 1 and Phase 2 plans.

## File Map

- Create `src/pi05_libero_repro/logiv/initial_proposal.py`: fail-open proposal/certification boundary and request/time accounting.
- Modify `src/pi05_libero_repro/protocol.py`: observation-only shadow callback in the single Base rollout loop.
- Create `src/pi05_libero_repro/logiv/recovery_records.py`: recovery-root IDs, manifests, atomic NPZ/JSON persistence, and verified loading.
- Create `src/pi05_libero_repro/logiv/recovery_splits.py`: cross-split leakage checks and dataset summary.
- Create `scripts/validate_recovery_dataset.py`: command-line split validation.
- Create `src/pi05_libero_repro/logiv/shadow_monitor.py`: stable local recovery-surface/goal-regression detector.
- Create `src/pi05_libero_repro/logiv/shadow_runtime.py`: accepted-proposal-to-grounder wiring and root collection.
- Modify `src/pi05_libero_repro/logiv/evaluation.py`: register `SHADOW_LOGIV` as a development arm.
- Modify `src/pi05_libero_repro/logiv/records.py`: explicit request/compute buckets for Base, proposal, shadow, and recovery.
- Modify `scripts/eval_logiv_libero.py`: run `BASE` and `SHADOW_LOGIV` through the same nominal path and persist audit artifacts.
- Modify `scripts/report_logiv_results.py`: report the new compute buckets without folding them into Base requests.
- Modify `scripts/run_logiv_eval.sh`: admit `SHADOW_LOGIV`.
- Create focused tests under `tests/logiv/` and extend the existing protocol/evaluator/record tests.

---

### Task 1: Fail-open initial proposal boundary

**Files:**
- Create: `src/pi05_libero_repro/logiv/initial_proposal.py`
- Create: `tests/logiv/test_initial_proposal.py`

**Interfaces:**
- Consumes: a provider exposing `propose(task_id: int, epoch_id: int, goal_mode: GoalMode) -> ProposalPackage` and a `validator(package: ProposalPackage) -> T` callable.
- Produces: `InitialProposalStatus`, generic `InitialProposalResult[T]`, and `run_initial_proposal -> InitialProposalResult[T]`.

- [ ] **Step 1: Write rejection and acceptance tests**

```python
from pi05_libero_repro.logiv.initial_proposal import (
    InitialProposalStatus,
    run_initial_proposal,
)
from pi05_libero_repro.logiv.model import GoalMode


class Provider:
    provider = "scripted-vlm-v1"

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = 0

    def propose(self, task_id, epoch_id, goal_mode):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.value


def test_provider_failure_is_a_fail_open_rejection():
    provider = Provider(error=RuntimeError("offline"))
    result = run_initial_proposal(
        provider,
        provider_name=provider.provider,
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=lambda package: package,
        clock=iter((10.0, 10.25)).__next__,
    )
    assert result.status is InitialProposalStatus.REJECTED
    assert result.package is None and result.validation is None
    assert result.request_count == 1
    assert result.elapsed_seconds == 0.25
    assert result.reason == "RuntimeError: offline"
    assert not result.intervention_enabled


def test_validator_failure_discards_the_uncertified_package():
    provider = Provider(value=object())

    def reject(package):
        raise ValueError("VAL rejected")

    result = run_initial_proposal(
        provider,
        provider_name=provider.provider,
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=reject,
    )
    assert result.status is InitialProposalStatus.REJECTED
    assert result.package is None and result.validation is None
    assert result.reason == "ValueError: VAL rejected"


def test_accepted_result_keeps_package_and_certification():
    package = object()
    certificate = object()
    result = run_initial_proposal(
        Provider(value=package),
        provider_name="scripted-vlm-v1",
        task_id=8,
        epoch_id=0,
        goal_mode=GoalMode.METADATA_ASSISTED,
        validator=lambda value: certificate,
    )
    assert result.status is InitialProposalStatus.ACCEPTED
    assert result.package is package
    assert result.validation is certificate
    assert result.request_count == 1
    assert result.reason is None
    assert result.intervention_enabled
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_initial_proposal.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'pi05_libero_repro.logiv.initial_proposal'`.

- [ ] **Step 3: Implement the typed fail-open boundary**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Callable, Generic, TypeVar

from pi05_libero_repro.logiv.model import GoalMode, ProposalPackage


T = TypeVar("T")


class InitialProposalStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class InitialProposalResult(Generic[T]):
    status: InitialProposalStatus
    provider: str
    request_count: int
    elapsed_seconds: float
    package: ProposalPackage | None
    validation: T | None
    reason: str | None

    @property
    def intervention_enabled(self) -> bool:
        return (
            self.status is InitialProposalStatus.ACCEPTED
            and self.package is not None
            and self.validation is not None
        )


def run_initial_proposal(
    provider: Any,
    *,
    provider_name: str,
    task_id: int,
    epoch_id: int,
    goal_mode: GoalMode,
    validator: Callable[[ProposalPackage], T],
    clock: Callable[[], float] = time.perf_counter,
) -> InitialProposalResult[T]:
    started = clock()
    try:
        package = provider.propose(task_id, epoch_id, goal_mode)
        validation = validator(package)
    except Exception as error:
        return InitialProposalResult(
            status=InitialProposalStatus.REJECTED,
            provider=provider_name,
            request_count=1,
            elapsed_seconds=clock() - started,
            package=None,
            validation=None,
            reason=f"{type(error).__name__}: {error}",
        )
    return InitialProposalResult(
        status=InitialProposalStatus.ACCEPTED,
        provider=provider_name,
        request_count=1,
        elapsed_seconds=clock() - started,
        package=package,
        validation=validation,
        reason=None,
    )
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest -q tests/logiv/test_initial_proposal.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/pi05_libero_repro/logiv/initial_proposal.py tests/logiv/test_initial_proposal.py
git commit -m "feat(logiv): add fail-open initial proposal boundary"
```

---

### Task 2: Non-destructive shadow callback in the Base rollout

**Files:**
- Modify: `src/pi05_libero_repro/protocol.py:3-8,73-82,125-203`
- Modify: `tests/test_protocol.py:1-236`

**Interfaces:**
- Consumes: `shadow_observer(observation: Mapping[str, Any], action: np.ndarray, policy_step: int) -> None`.
- Produces: `EpisodeOutcome.shadow_calls`, `shadow_errors`, and `shadow_wall_seconds`; the new keyword-only `run_episode` parameters `shadow_observer=None` and `clock=time.perf_counter`.

- [ ] **Step 1: Add a parity test with a mutating, failing observer**

```python
def test_shadow_observer_cannot_change_or_abort_base_actions():
    baseline_env = FakeEnv(succeed_on_policy_step=7)
    shadow_env = FakeEnv(succeed_on_policy_step=7)
    baseline = run_episode(
        baseline_env, FakeClient(), np.array([9.0]), "prompt", FakeImageTools()
    )
    seen_steps = []

    def hostile_observer(obs, action, policy_step):
        seen_steps.append(policy_step)
        obs["robot0_eef_pos"][0] = -999
        action[:] = -999
        if policy_step == 3:
            raise RuntimeError("shadow failed")

    ticks = iter(value * 0.01 for value in range(100))
    shadow = run_episode(
        shadow_env,
        FakeClient(),
        np.array([9.0]),
        "prompt",
        FakeImageTools(),
        shadow_observer=hostile_observer,
        clock=ticks.__next__,
    )
    np.testing.assert_array_equal(np.asarray(shadow.actions), np.asarray(baseline.actions))
    assert shadow.steps == baseline.steps
    assert shadow.inference_requests == baseline.inference_requests
    assert (shadow.done, shadow.check_success) == (baseline.done, baseline.check_success)
    assert seen_steps == list(range(1, baseline.steps + 1))
    assert shadow.shadow_calls == baseline.steps
    assert shadow.shadow_errors == 1
    assert shadow.shadow_wall_seconds == pytest.approx(baseline.steps * 0.01)
```

- [ ] **Step 2: Run the test and verify the signature failure**

Run: `uv run pytest -q tests/test_protocol.py::test_shadow_observer_cannot_change_or_abort_base_actions`

Expected: FAIL with `TypeError: run_episode() got an unexpected keyword argument 'shadow_observer'`.

- [ ] **Step 3: Add copied callback inputs and isolated accounting**

Add `Callable` and `Mapping` imports plus `import time`. Append these fields to `EpisodeOutcome`:

```python
    shadow_calls: int = 0
    shadow_errors: int = 0
    shadow_wall_seconds: float = 0.0
```

Append keyword-only parameters to `run_episode`:

```python
    *,
    shadow_observer: Callable[[Mapping[str, Any], np.ndarray, int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
```

Initialize the three counters next to `inference_requests`. Immediately after `env.step(action.tolist())`, add:

```python
            if shadow_observer is not None:
                shadow_started = clock()
                shadow_calls += 1
                copied_observation = {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in obs.items()
                }
                try:
                    shadow_observer(
                        copied_observation,
                        action.copy(),
                        len(executed_actions),
                    )
                except Exception:
                    shadow_errors += 1
                finally:
                    shadow_wall_seconds += clock() - shadow_started
```

Return all three counters in `EpisodeOutcome`. Do not use the observer's return value and do not move the existing `done` check ahead of the callback.

- [ ] **Step 4: Run protocol tests**

Run: `uv run pytest -q tests/test_protocol.py`

Expected: all protocol tests pass, including the parity test.

- [ ] **Step 5: Commit**

```bash
git add src/pi05_libero_repro/protocol.py tests/test_protocol.py
git commit -m "feat(logiv): add isolated shadow observation hook"
```

---

### Task 3: Reproducible recovery-root records

**Files:**
- Create: `src/pi05_libero_repro/logiv/recovery_records.py`
- Create: `tests/logiv/test_recovery_records.py`

**Interfaces:**
- Consumes: `FactSnapshot`, flattened MuJoCo state, raw LIBERO observation, immutable task/seed metadata, and a Base action-prefix hash.
- Produces: `RecoverySplit`, `RecoveryRootManifest`, `RecoveryRootArtifacts`, `make_recovery_root_manifest`, `write_recovery_root`, and `load_recovery_root`.

- [ ] **Step 1: Write ID semantics and round-trip tests**

```python
def test_group_id_ignores_branch_seeds_but_root_id_does_not():
    first = _manifest(perturbation_seed=10, branch_seed=1)
    second = _manifest(perturbation_seed=11, branch_seed=2)
    assert first.recovery_group_id == second.recovery_group_id
    assert first.root_id != second.root_id


def test_recovery_root_round_trip_preserves_exact_physics_and_observation(tmp_path):
    simulator_state = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    observation = _observation()
    manifest = _manifest(simulator_state=simulator_state)
    artifacts = write_recovery_root(tmp_path, manifest, simulator_state, observation)
    loaded_manifest, loaded_state = load_recovery_root(artifacts.directory)
    assert loaded_manifest == manifest
    np.testing.assert_array_equal(loaded_state["simulator_state"], simulator_state)
    for key, value in observation.items():
        np.testing.assert_array_equal(loaded_state[key], value)
```

The `_manifest` helper must pass identical task/scene/object/initial/parent/family fields while varying only the explicit seeds in the first test. `_observation` must contain `agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos` NumPy arrays.

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_recovery_records.py`

Expected: FAIL during collection because `recovery_records` does not exist.

- [ ] **Step 3: Implement canonical IDs and manifest construction**

Define:

```python
class RecoverySplit(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    HELDOUT = "HELDOUT"


@dataclass(frozen=True)
class RecoveryRootManifest:
    schema_version: int
    root_id: str
    recovery_group_id: str
    split: RecoverySplit
    task_id: int
    episode_idx: int
    scene_sha256: str
    object_instance_ids: tuple[str, ...]
    initial_state_sha256: str
    parent_snapshot_sha256: str
    perturbation_family: str
    perturbation_seed: int | None
    branch_seed: int | None
    master_seed: int
    policy_seed: int
    simulator_seed: int
    trigger_class: str
    policy_step: int
    policy_request_generation: int
    simulator_state_sha256: str
    state_fingerprint: str
    base_action_prefix_sha256: str
    fact_evidence_hash: str
    true_facts: tuple[str, ...]
    false_facts: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryRootArtifacts:
    directory: Path
    manifest_json: Path
    state_npz: Path
```

`make_recovery_root_manifest` is keyword-only and accepts, in dataclass field order,
`split`, `task_id`, `episode_idx`, `scene_sha256`, `object_instance_ids`,
`initial_state_sha256`, `parent_snapshot_sha256`, `perturbation_family`,
`perturbation_seed`, `branch_seed`, `master_seed`, `policy_seed`, `simulator_seed`,
`trigger_class`, `policy_step`, `policy_request_generation`, followed by
`simulator_state: np.ndarray`, `base_action_prefix_sha256`, and `snapshot: FactSnapshot`.
It returns `RecoveryRootManifest`. `write_recovery_root(output_dir, manifest,
simulator_state, observation)` returns `RecoveryRootArtifacts`.
`load_recovery_root(directory)` returns
`tuple[RecoveryRootManifest, Mapping[str, np.ndarray]]`. These names and orders are
the final interfaces used by Tasks 4 and 6.

Use canonical sorted JSON for ID payloads. `recovery_group_id` hashes exactly `task_id`, `scene_sha256`, sorted `object_instance_ids`, `initial_state_sha256`, `parent_snapshot_sha256`, and `perturbation_family`. `root_id` additionally hashes `perturbation_seed`, `branch_seed`, and the exact simulator-state SHA-256. Compute `state_fingerprint` from the simulator state rounded to four decimal places, including dtype and shape in the digest. Reject negative IDs/steps/seeds and non-64-character lowercase SHA-256 fields.

- [ ] **Step 4: Implement atomic persistence and verified loading**

`write_recovery_root` must create `<output>/<root_id>/`, write `.state.tmp.npz` with `np.savez_compressed`, replace it with `state.npz`, then write and replace `.recovery_root.json.tmp` as `recovery_root.json`. Refuse to overwrite a non-identical existing root. `load_recovery_root` must reconstruct the enum/dataclass, load with `allow_pickle=False`, and verify `simulator_state_sha256` before returning.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/logiv/test_recovery_records.py`

Expected: both ID and exact round-trip tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pi05_libero_repro/logiv/recovery_records.py tests/logiv/test_recovery_records.py
git commit -m "feat(logiv): persist reproducible recovery roots"
```

---

### Task 4: Mechanical recovery split isolation

**Files:**
- Create: `src/pi05_libero_repro/logiv/recovery_splits.py`
- Create: `scripts/validate_recovery_dataset.py`
- Create: `tests/logiv/test_recovery_splits.py`

**Interfaces:**
- Consumes: `Sequence[RecoveryRootManifest]` or directories containing `recovery_root.json`.
- Produces: `RecoverySplitError`, `validate_recovery_splits(manifests) -> Mapping[str, int]`, and a zero/nonzero validation CLI.

- [ ] **Step 1: Write leakage tests for every forbidden relation**

```python
@pytest.mark.parametrize(
    "field,value",
    [
        ("recovery_group_id", "same-group"),
        ("initial_state_sha256", "1" * 64),
        ("parent_snapshot_sha256", "2" * 64),
        ("state_fingerprint", "3" * 64),
    ],
)
def test_validator_rejects_cross_split_leakage(field, value):
    train = replace(_manifest(RecoverySplit.TRAIN), **{field: value})
    heldout = replace(_manifest(RecoverySplit.HELDOUT), **{field: value})
    with pytest.raises(RecoverySplitError, match=field):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_perturbation_seed_reuse_within_task():
    train = replace(_manifest(RecoverySplit.TRAIN), perturbation_seed=44)
    dev = replace(_manifest(RecoverySplit.DEV), perturbation_seed=44)
    with pytest.raises(RecoverySplitError, match="perturbation_seed"):
        validate_recovery_splits([train, dev])


def test_validator_reports_disjoint_dataset_counts():
    summary = validate_recovery_splits(
        [_manifest(RecoverySplit.TRAIN), _manifest(RecoverySplit.DEV, salt="dev")]
    )
    assert summary == {"TRAIN": 1, "DEV": 1, "HELDOUT": 0, "total": 2}
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_recovery_splits.py`

Expected: FAIL during collection because `recovery_splits` does not exist.

- [ ] **Step 3: Implement the validator**

For each key below, build `key -> set(split)` and raise `RecoverySplitError` when the set has more than one member:

```python
checks = {
    "recovery_group_id": lambda item: item.recovery_group_id,
    "initial_state_sha256": lambda item: item.initial_state_sha256,
    "parent_snapshot_sha256": lambda item: item.parent_snapshot_sha256,
    "state_fingerprint": lambda item: item.state_fingerprint,
    "perturbation_seed": lambda item: (
        (item.task_id, item.perturbation_seed)
        if item.perturbation_seed is not None
        else None
    ),
}
```

Also reject duplicate `root_id` values even within one split. Return deterministic counts for all three enum values and `total`.

- [ ] **Step 4: Add the dataset CLI**

The script accepts one or more directory arguments, recursively loads files named `recovery_root.json` through `load_recovery_root`, validates them, prints the sorted JSON summary, and exits nonzero with the `RecoverySplitError` message on leakage. It must reject an empty dataset.

- [ ] **Step 5: Run unit and CLI tests**

Run: `uv run pytest -q tests/logiv/test_recovery_splits.py`

Expected: all split tests pass.

Run: `uv run python scripts/validate_recovery_dataset.py --help`

Expected: exit 0 and usage text containing `recovery_root.json`.

- [ ] **Step 6: Commit**

```bash
git add src/pi05_libero_repro/logiv/recovery_splits.py scripts/validate_recovery_dataset.py tests/logiv/test_recovery_splits.py
git commit -m "feat(logiv): enforce recovery split isolation"
```

---

### Task 5: Stable local recovery-state monitor

**Files:**
- Create: `src/pi05_libero_repro/logiv/shadow_monitor.py`
- Create: `tests/logiv/test_shadow_monitor.py`

**Interfaces:**
- Consumes: a certified `TaskProblem`, support-surface object IDs, and `snapshot_reader(observation) -> FactSnapshot`.
- Produces: `ShadowTrigger`, `ShadowMonitorMetrics`, and callable `StableRecoveryObserver`.

- [ ] **Step 1: Write sampling, confirmation, and one-shot tests**

```python
def test_unplanned_support_requires_three_sampled_confirmations_and_fires_once():
    snapshots = iter([_dropped_snapshot(5), _dropped_snapshot(10), _dropped_snapshot(15), _dropped_snapshot(20)])
    triggers = []
    observer = StableRecoveryObserver(
        problem=_problem(),
        support_surfaces=frozenset({"kitchen_table_recovery_surface"}),
        snapshot_reader=lambda observation: next(snapshots),
        on_confirmed=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_observation(), np.zeros(7), step)
    assert len(triggers) == 1
    assert triggers[0].policy_step == 15
    assert triggers[0].trigger_class == "UNPLANNED_SUPPORT_STABLE"
    assert observer.metrics.snapshot_calls == 4
    assert observer.metrics.confirmed_triggers == 1


def test_goal_regression_is_detected_only_after_goal_was_observed_true():
    snapshots = iter([_goal_snapshot(True, 5), _goal_snapshot(False, 10), _goal_snapshot(False, 15), _goal_snapshot(False, 20)])
    triggers = []
    observer = StableRecoveryObserver(
        problem=_problem(),
        support_surfaces=frozenset(),
        snapshot_reader=lambda observation: next(snapshots),
        on_confirmed=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_observation(), np.zeros(7), step)
    assert [trigger.trigger_class for trigger in triggers] == ["COMPLETED_GOAL_REGRESSION_STABLE"]


def test_snapshot_error_resets_confirmation_without_escaping():
    calls = 0

    def reader(observation):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("grounding")
        return _dropped_snapshot(calls * 5)

    triggers = []
    observer = StableRecoveryObserver(
        problem=_problem(),
        support_surfaces=frozenset({"kitchen_table_recovery_surface"}),
        snapshot_reader=reader,
        on_confirmed=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_observation(), np.zeros(7), step)
    assert observer.metrics.snapshot_errors == 1
    assert len(triggers) == 1
    assert triggers[0].policy_step == 20
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_shadow_monitor.py`

Expected: FAIL during collection because `shadow_monitor` does not exist.

- [ ] **Step 3: Implement deterministic signatures**

`StableRecoveryObserver` samples only when `policy_step % interval_steps == 0`. On each successful snapshot it:

1. Adds currently true positive goal facts and currently false negative goal facts to an internal achieved-goal set.
2. Emits `UNPLANNED_SUPPORT_STABLE` candidates for a true `at(object, support_surface)` fact that is in neither the problem initial state nor the signed goal.
3. Emits `COMPLETED_GOAL_REGRESSION_STABLE` candidates when a previously achieved signed-goal literal is now reliably opposite.
4. Sorts the candidate PDDL strings into an immutable signature.

Define the public records exactly as:

```python
@dataclass(frozen=True)
class ShadowTrigger:
    trigger_class: str
    signature: tuple[str, ...]
    policy_step: int
    snapshot: FactSnapshot
    observation: Mapping[str, Any]


@dataclass
class ShadowMonitorMetrics:
    snapshot_calls: int = 0
    snapshot_errors: int = 0
    confirmed_triggers: int = 0
```

`StableRecoveryObserver.__init__` is keyword-only and accepts `problem: TaskProblem`,
`support_surfaces: FrozenSet[str]`, `snapshot_reader:
Callable[[Mapping[str, Any]], FactSnapshot]`, `on_confirmed:
Callable[[ShadowTrigger], None]`, `interval_steps: int`, and
`confirmation_count: int`. Its call signature is `(observation: Mapping[str, Any],
action: np.ndarray, policy_step: int) -> None`.

The observer resets its streak on an empty/different signature or reader error, confirms after exactly `confirmation_count` identical sampled signatures, and stores fired signatures so it never repeats the same root event. `on_confirmed` exceptions must increment `snapshot_errors` and remain inside the observer. Copy NumPy values before placing the observation into `ShadowTrigger`.

- [ ] **Step 4: Run monitor tests**

Run: `uv run pytest -q tests/logiv/test_shadow_monitor.py`

Expected: all three monitor tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pi05_libero_repro/logiv/shadow_monitor.py tests/logiv/test_shadow_monitor.py
git commit -m "feat(logiv): detect stable recovery states in shadow"
```

---

### Task 6: Shadow runtime, evaluator arm, and separate compute accounting

**Files:**
- Create: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Create: `tests/logiv/test_shadow_runtime.py`
- Modify: `src/pi05_libero_repro/logiv/evaluation.py:39-45`
- Modify: `src/pi05_libero_repro/logiv/records.py:21-29,165-216`
- Modify: `scripts/eval_logiv_libero.py:4-68,542-597,600-971,974-1035`
- Modify: `scripts/report_logiv_results.py`
- Modify: `scripts/run_logiv_eval.sh:20-23`
- Modify: `tests/logiv/test_evaluator.py`
- Modify: `tests/logiv/test_records.py`

**Interfaces:**
- Consumes: Tasks 1-5, existing `ScriptedProposalProvider`, `TaskBinding`, `certify_initial_package`, `LiberoObservationStore`, `LiberoOracleGrounder`, `ValWrapper`, and `run_episode`.
- Produces: `MethodArm.SHADOW_LOGIV`, `ShadowRuntime`, `build_shadow_runtime`, schema-3 compute buckets, `initial_proposal.json`, `shadow_monitor.json`, `compute_accounting.json`, `base_execution.json`, and optional `recovery_roots/`.

- [ ] **Step 1: Add arm and parser/accounting contract tests**

Extend evaluator tests to assert:

```python
args = _parser().parse_args([
    "--run-id", "shadow",
    "--method-arm", "SHADOW_LOGIV",
    "--goal-mode", "METADATA_ASSISTED",
    "--deviation-mode", "NOMINAL",
    "--oracle-grounding",
    "--development-only",
    "--task-ids", "8",
    "--episode-indices", "0",
    "--port", "8010",
    "--output-dir", "/tmp/shadow",
    "--collect-recovery-roots",
    "--recovery-root-split", "DEV",
])
assert args.collect_recovery_roots
assert args.recovery_root_split == "DEV"
assert MethodArm(args.method_arm) is MethodArm.SHADOW_LOGIV
```

Extend record tests with a schema-3 `SHADOW_LOGIV` record and assert:

```python
assert record.inference_requests == record.base_policy_requests
assert record.initial_proposal_requests == 1
assert record.shadow_vlm_requests == 0
assert record.recovery_policy_requests == 0
```

- [ ] **Step 2: Run the tests and verify the unknown-arm failures**

Run: `uv run pytest -q tests/logiv/test_evaluator.py tests/logiv/test_records.py`

Expected: FAIL because `SHADOW_LOGIV` is not a valid `MethodArm` or record arm.

- [ ] **Step 3: Register the arm and schema-3 buckets**

Add `SHADOW_LOGIV = "SHADOW_LOGIV"` after `BASE` and add it to `METHOD_ARMS` and the shell launcher case. Append defaulted fields to `LogivEpisodeRecord`:

```python
    base_policy_requests: int = 0
    initial_proposal_requests: int = 0
    shadow_vlm_requests: int = 0
    recovery_policy_requests: int = 0
    shadow_monitor_calls: int = 0
    shadow_monitor_errors: int = 0
    shadow_monitor_seconds: float = 0.0
```

For schema version 3, `validate_episode_records` must require `inference_requests == base_policy_requests`, require `initial_proposal_requests == 0` for `BASE` and `== 1` for `SHADOW_LOGIV`, and reject nonzero `shadow_vlm_requests` or `recovery_policy_requests` in Phase 0 records. Add mean values for all seven fields to the report's operational metrics.

- [ ] **Step 4: Implement `ShadowRuntime` with lazy grounding**

Define the cross-file records before the function implementation:

```python
T = TypeVar("T")


@dataclass
class ShadowRuntimeCounters:
    root_count: int = 0
    root_write_errors: int = 0


@dataclass(frozen=True)
class ShadowRuntime(Generic[T]):
    initial_proposal: InitialProposalResult[T]
    observer: Callable[[Mapping[str, Any], np.ndarray, int], None] | None
    monitor: StableRecoveryObserver | None
    counters: ShadowRuntimeCounters
```

`build_shadow_runtime` is keyword-only and accepts `provider: Any`,
`provider_name: str`, `task_id: int`, `goal_mode: GoalMode`, `validator:
Callable[[ProposalPackage], T]`, `snapshot_reader_factory:
Callable[[ProposalPackage, T], Callable[[Mapping[str, Any]], FactSnapshot]]`,
`support_surfaces: FrozenSet[str]`, `root_collector:
Callable[[ShadowTrigger, str], RecoveryRootArtifacts | None]`,
`interval_steps: int`, and `confirmation_count: int`; it returns
`ShadowRuntime[T]`. This is the final interface used by evaluator code and tests.

`build_shadow_runtime` performs `run_initial_proposal` immediately. Its validator must verify exact registered-object equality, call `certify_initial_package` against `package.proposal.initial_snapshot`, and use `binding.supported_action_schemas | binding.recovery_schemas`, configured repair bounds, retry policy, macro decomposition set, and VAL timeout.

If rejected, return `ShadowRuntime(initial_proposal=result, observer=None,
monitor=None, counters=ShadowRuntimeCounters())`. If accepted, construct a callback
that lazily initializes `LiberoObservationStore` and `LiberoOracleGrounder` from the
first copied post-action observation, updates the store on later actions, and feeds
`peek_snapshot()` into `StableRecoveryObserver`.

The wrapper must update a SHA-256 digest with each copied Base action before the monitor samples. On confirmation it calls `env.get_sim_state()`, builds a manifest with:

- immutable task/episode/scene/object/initial-state IDs;
- master, policy, and simulator seeds;
- `perturbation_family="observed_nominal_failure"`, with no branch seed;
- trigger class, policy step, `ceil(policy_step / replan_steps)` request generation;
- current fact snapshot and current Base action-prefix hash.

It writes only when `collect_recovery_roots` is true. A root write failure increments monitor errors and cannot escape into the nominal rollout. Derive support surfaces only from binding entries whose `kind` is `support_surface_alias`.

- [ ] **Step 5: Test proposal rejection and root-write failure containment**

Use injected fake provider/validator/snapshot reader/root writer in `tests/logiv/test_shadow_runtime.py`. Assert proposal rejection returns no observer, and assert a root writer raising `OSError("disk")` leaves the callback callable while incrementing the error counter. No test may import LIBERO.

Run: `uv run pytest -q tests/logiv/test_shadow_runtime.py`

Expected: all shadow runtime tests pass.

- [ ] **Step 6: Route Base and Shadow through one evaluator branch**

Change the branch condition to:

```python
arm = MethodArm(args.method_arm)
if arm in {MethodArm.BASE, MethodArm.SHADOW_LOGIV}:
```

Construct `ShadowRuntime` only for `SHADOW_LOGIV`; pass `runtime.observer` to `run_episode`. Do not change `str(task.language)`, `initial_state`, `episode_client`, `max_steps`, `wait_steps`, `replan_steps`, or `settling_steps` between the arms. Both arms must build the same Base `ControllerResult` and native terminal evaluation.

Write `base_execution.json` for both arms with `steps`, `base_policy_requests`, `done_signal`, `post_settling_success`, and a SHA-256 over the dtype, shape, and bytes of every executed action. For Shadow also write:

```json
{
  "initial_proposal.json": ["status", "provider", "request_count", "elapsed_seconds", "reason", "certificate_hash", "graph_hash"],
  "shadow_monitor.json": ["callback_calls", "callback_errors", "callback_seconds", "snapshot_calls", "snapshot_errors", "confirmed_triggers", "root_count"],
  "compute_accounting.json": ["base_policy_requests", "initial_proposal_requests", "shadow_vlm_requests", "recovery_policy_requests", "initial_proposal_seconds", "shadow_monitor_seconds"]
}
```

The notation above names required JSON fields; each filename is a separate object. Set `initial_proposal_requests` from the proposal result, `shadow_vlm_requests=0`, `recovery_policy_requests=0`, and `base_policy_requests=outcome.inference_requests`. Proposal rejection must keep `valid=True` unless the Base rollout itself fails.

- [ ] **Step 7: Add CLI/config fields**

Add:

```python
parser.add_argument("--collect-recovery-roots", action="store_true")
parser.add_argument("--recovery-root-split", choices=("TRAIN", "DEV", "HELDOUT"), default="DEV")
parser.add_argument("--shadow-monitor-interval-steps", default=5, type=int)
parser.add_argument("--shadow-confirmations", default=3, type=int)
```

Persist all four values in `run.json`. In Phase 0, reject collection unless
`--development-only` is set, `--recovery-root-split DEV` is selected, and the method
arm is `SHADOW_LOGIV`; the other enum values exist so imported/system-generated
datasets can be validated without relabeling their manifests, not so the evaluator
can create a held-out claim from the inspected 50 states.

- [ ] **Step 8: Run evaluator, record, runtime, and protocol tests**

Run: `uv run pytest -q tests/logiv/test_shadow_runtime.py tests/logiv/test_evaluator.py tests/logiv/test_records.py tests/test_protocol.py`

Expected: all selected tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/pi05_libero_repro/logiv/shadow_runtime.py src/pi05_libero_repro/logiv/evaluation.py src/pi05_libero_repro/logiv/records.py scripts/eval_logiv_libero.py scripts/report_logiv_results.py scripts/run_logiv_eval.sh tests/logiv/test_shadow_runtime.py tests/logiv/test_evaluator.py tests/logiv/test_records.py
git commit -m "feat(logiv): add auditable shadow data collection arm"
```

---

### Task 7: Full verification and paired real smoke test

**Files:**
- Create: `docs/experiments/2026-08-03-logiv-r2m-phase0-smoke.md`

**Interfaces:**
- Consumes: all Phase 0 implementation tasks and an already running episode-seeded policy server on port `8010`.
- Produces: test evidence, Base/Shadow parity evidence, and a validated development recovery dataset.

- [ ] **Step 1: Run static diff checks and the complete unit suite**

Run: `git diff --check`

Expected: no output, exit 0.

Run: `uv run pytest -q`

Expected: all tests pass; the clean starting point had `235 passed`, so the final count must be greater than 235.

- [ ] **Step 2: Run one real paired Task 8 episode**

```bash
scripts/run_logiv_eval.sh BASE 0 8010 runs/r2m-phase0-base-task8-seed0 \
  --run-id r2m-phase0-base-task8-seed0 \
  --goal-mode METADATA_ASSISTED --deviation-mode NOMINAL \
  --development-only --task-ids 8 --episode-indices 0 --seed 7

scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 runs/r2m-phase0-shadow-task8-seed0 \
  --run-id r2m-phase0-shadow-task8-seed0 \
  --goal-mode METADATA_ASSISTED --deviation-mode NOMINAL \
  --oracle-grounding --development-only --task-ids 8 --episode-indices 0 --seed 7 \
  --collect-recovery-roots --recovery-root-split DEV
```

Expected: both commands exit 0 and each writes exactly one episode record.

- [ ] **Step 3: Prove Base/Shadow nominal parity**

```bash
jq -S '{steps,base_policy_requests,done_signal,post_settling_success,actions_sha256}' \
  runs/r2m-phase0-base-task8-seed0/artifacts/task_08/episode_000/base_execution.json \
  > /tmp/r2m-base-execution.json
jq -S '{steps,base_policy_requests,done_signal,post_settling_success,actions_sha256}' \
  runs/r2m-phase0-shadow-task8-seed0/artifacts/task_08/episode_000/base_execution.json \
  > /tmp/r2m-shadow-execution.json
diff -u /tmp/r2m-base-execution.json /tmp/r2m-shadow-execution.json
```

Expected: `diff` prints nothing and exits 0. `initial_proposal.json` must report
one request; `compute_accounting.json` must report zero VLM and recovery-policy
requests.

- [ ] **Step 4: Validate any collected roots**

Run: `find runs/r2m-phase0-shadow-task8-seed0 -name recovery_root.json -print`.

If at least one root exists, run:

```bash
uv run python scripts/validate_recovery_dataset.py \
  runs/r2m-phase0-shadow-task8-seed0/artifacts
```

Expected: exit 0 with JSON containing only `DEV` roots. If the episode produces no stable trigger, record `root_count=0`; the deterministic root round-trip and leakage tests remain the Phase 0 artifact acceptance evidence.

- [ ] **Step 5: Write the smoke report**

Record the implementation commit, complete pytest count, both episode terminal results, the five parity fields, proposal/monitor request buckets, root count, split-validator output, and the explicit statement: `Phase 0 collected development evidence only; recovery capability and takeover are not enabled.`

- [ ] **Step 6: Commit the verified report**

```bash
git add docs/experiments/2026-08-03-logiv-r2m-phase0-smoke.md
git commit -m "docs: record LOGIV R2M phase0 verification"
```

## Phase 0 Acceptance Gate

Phase 0 is complete only when all of the following are true:

- the full unit suite passes from the isolated worktree;
- Base and Shadow real smoke artifacts have identical action hash, action count, Base policy request count, done signal, and native terminal result;
- initial proposal failure is proven fail-open by tests;
- shadow callback and root-writer failures are proven non-fatal by tests;
- exact simulator state and observation arrays round-trip through NPZ;
- split validation rejects group, initial-state, parent, perturbation-seed, and state-fingerprint leakage;
- all added requests and compute are reported outside `base_policy_requests`;
- no recovery action, recovery policy, capability claim, or online takeover exists in the Phase 0 code path.
