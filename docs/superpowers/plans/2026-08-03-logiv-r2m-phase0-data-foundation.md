# LOGIV R2M Phase 0 Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-open initial LOGIV proposal, an event-aware read-only shadow monitor with dynamically invalidated certificates, reproducible labeled recovery-root artifacts, and mechanically enforced train/dev/held-out isolation while keeping every Base action and Base policy request unchanged.

**Architecture:** `run_episode` remains the only nominal rollout loop and gains an observation-only callback whose return value is ignored and whose failures are contained. `SHADOW_LOGIV` runs that exact loop with at most one separately accounted initial proposal from the shared step-0 observation. The monitor separates stable anomaly candidates from historically evidenced confirmed deviations, treats the Initial DAG as an advisory certificate that becomes stale on uncovered task-relevant transitions, and records both labels without ever producing a control decision. Recovery snapshots use exact `root_id` values for state reproduction, event-origin-bound `recovery_group_id` values for leakage-safe splitting, and coarser parent-trajectory `independence_unit_id` values for statistical denominators and role isolation.

**Tech Stack:** Python 3.11, dataclasses, NumPy, pytest, existing LIBERO/MuJoCo adapter, existing VAL wrapper, Bash/Docker evaluation launcher.

## Global Constraints

- Execute from a clean isolated worktree created from commit `8254a26b` with `superpowers:using-git-worktrees`; do not modify or copy the dirty experimental files in the original worktree.
- Add no Python runtime dependency beyond the repository's existing dependencies.
- `BASE` must not construct a proposal provider, grounder, monitor, or recovery artifact writer.
- `SHADOW_LOGIV` must send the same prompt, action history, policy RNG envelope, number of Base policy requests, and action count as `BASE` for the same task, episode, and seed.
- The shadow callback receives copies of observations/actions, returns no control decision, and cannot terminate or alter the nominal rollout.
- Stable location is never a failure authorization. Phase 0 records `ANOMALY_CANDIDATE` separately from `CONFIRMED_DEVIATION`; capability and budget are not part of detection and no permit type exists in this phase.
- The Initial DAG is advisory. A stale certificate is recorded and may be reconciled by a later
  phase, but can never authorize any physical action; Phase 0 contains no handoff path.
- Phase 0 uses only the local synchronous oracle grounder. It records `shadow_vlm_requests=0`; external VLM monitoring remains a future asynchronous, non-blocking extension.
- The Phase 0 proposal provider is the deterministic local `ScriptedProposalProvider`; no network or
  unbounded external call is permitted on the step-0 callback. A provider-raised `TimeoutError` is a
  fail-open rejection. Any future remote provider must run asynchronously with a client-enforced
  deadline and may publish only at a later observation-generation boundary; this synchronous helper
  does not claim it can preempt an arbitrary blocking Python call.
- At most one initial proposal request is permitted in `SHADOW_LOGIV`. On the normal step-0 path it
  is exactly one. Proposal/certification rejection disables monitoring and root capture; if callback
  input preparation itself fails, the artifact records `NOT_ATTEMPTED` and zero requests. Both paths
  record the reason and still run Base.
- Phase 0 never invokes a recovery policy and records `recovery_policy_requests=0`.
- All recovery roots collected from the already inspected 50 LIBERO initial states are development data; no held-out performance claim is permitted from them.
- This plan implements only the data foundation. Recovery-policy training/evaluation and capability-gated atomic runtime handoff require separate Phase 1 and Phase 2 plans.

## Implementation Checkpoint (not acceptance)

- The isolated branch contains the earlier Task 1 commit `6955c43c`; it remains provisional until
  Task 6 proves certification against the copied live step-0 observation.
- The earlier Task 2 commit `333b749c` is **reopened**. It predates the deep-copy containment,
  RNG-restoration, step-0 callback, pending-chunk/offset, request-index, and action-prefix contracts in
  this revision. Passing its old tests is not evidence that revised Task 2 is complete.
- Tasks 3-7 and all Phase 1/2 capability claims remain unimplemented/unverified. Document review is
  not empirical success evidence.

## File Map

- Create `src/pi05_libero_repro/logiv/initial_proposal.py`: fail-open proposal/certification boundary and request/time accounting.
- Modify `src/pi05_libero_repro/protocol.py`: observation-only shadow callback in the single Base rollout loop.
- Create `src/pi05_libero_repro/logiv/recovery_records.py`: recovery-root IDs, manifests, atomic NPZ/JSON persistence, and verified loading.
- Modify `src/pi05_libero_repro/logiv/model.py` and `src/pi05_libero_repro/logiv/libero_adapter.py`: add the
  versioned, auditable fact universe, retain the exact canonical grounding-evidence payload, and
  expose the version-bound read-only transition-feature reader.
- Create `src/pi05_libero_repro/logiv/recovery_splits.py`: cross-split leakage checks and dataset summary.
- Create `scripts/validate_recovery_dataset.py`: command-line split validation.
- Create `scripts/build_recovery_training_manifest.py`: mechanically filter eligible roots and sign
  an immutable role-specific training manifest.
- Create `scripts/allocate_recovery_roles.py`: pre-outcome complete role allocation plus
  append-only canonical registry CAS update.
- Create `src/pi05_libero_repro/logiv/shadow_monitor.py`: event-aware anomaly/deviation detector plus relevant-fact certificate reconciliation.
- Create `configs/logiv/r2m-monitor-evidence-v1.json`: versioned Task 5/8 normal/abnormal regions,
  action-event rules, deadlines, grace windows, and progress windows.
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
- Consumes: `shadow_observer(context: ShadowStepContext) -> None` and an optional protocol-owned
  `request_envelope_reader(index: int, require_issued: bool) -> str | None`; `policy_step=0,
  last_action=None` is the settled live initial observation before the first Base request.
- Produces: `BaseActionPrefixHasher`, `ShadowFailureRecord`, an immutable per-request envelope log,
  `EpisodeOutcome.shadow_calls`, `shadow_errors`, `shadow_failure_records`, and
  `shadow_wall_seconds`/`shadow_parity_valid`; the new keyword-only `run_episode` parameters `shadow_observer=None`,
  `request_envelope_reader=None`, and `clock=time.perf_counter`.

- [ ] **Step 1: Add a parity test with a mutating, failing observer**

Extend the fake observation fixture with `nested={"items": [{"value": 1}]}`; the Base observation
preparation ignores it, making it a direct aliasing probe.

```python
def test_shadow_observer_cannot_change_or_abort_base_actions():
    baseline_env = FakeEnv(succeed_on_policy_step=7)
    shadow_env = FakeEnv(succeed_on_policy_step=7)
    baseline = run_episode(
        baseline_env, FakeClient(), np.array([9.0]), "prompt", FakeImageTools()
    )
    seen_steps = []

    def hostile_observer(context):
        seen_steps.append(context.policy_step)
        context.observation["robot0_eef_pos"][0] = -999
        context.observation["nested"]["items"][0]["value"] = -999
        random.random()
        np.random.random()
        if context.last_action is not None:
            context.last_action[:] = -999
        context.pending_base_actions[:] = -999
        if context.policy_step == 3:
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
    assert seen_steps == list(range(0, baseline.steps + 1))
    assert shadow.shadow_calls == baseline.steps + 1
    assert shadow.shadow_errors == 1
    assert shadow.shadow_wall_seconds == pytest.approx((baseline.steps + 1) * 0.01)
```

- [ ] **Step 2: Run the test and verify the signature failure**

Run: `uv run pytest -q tests/test_protocol.py::test_shadow_observer_cannot_change_or_abort_base_actions`

Expected: FAIL with `TypeError: run_episode() got an unexpected keyword argument 'shadow_observer'`.

- [ ] **Step 3: Add step-0 observation, deep-copied callback inputs, RNG restoration, and isolated accounting**

Add `Callable` and `Mapping` imports plus `import copy`, `import random`, and `import time`. Append these fields to `EpisodeOutcome`:

```python
@dataclass(frozen=True)
class ShadowStepContext:
    observation: Mapping[str, Any]
    last_action: np.ndarray | None
    policy_step: int
    base_policy_request_count: int
    active_base_request_index: int | None
    next_base_request_index: int
    active_base_request_envelope_json: str | None
    next_base_replay_envelope_json: str | None
    base_action_response_size: int | None
    base_action_chunk_size: int
    pending_base_action_offset: int
    pending_base_actions: np.ndarray
    base_action_prefix_sha256: str


@dataclass(frozen=True)
class ShadowFailureRecord:
    policy_step: int
    stage: str
    reason: str
```

Append these fields to `EpisodeOutcome`:

```python
    shadow_calls: int = 0
    shadow_errors: int = 0
    shadow_failure_records: tuple[ShadowFailureRecord, ...] = ()
    shadow_wall_seconds: float = 0.0
    shadow_parity_valid: bool = True
```

Append keyword-only parameters to `run_episode`:

```python
    *,
    shadow_observer: Callable[[ShadowStepContext], None] | None = None,
    request_envelope_reader: Callable[[int, bool], str | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
```

Initialize the three counters, mutable failure-record list, chunk response/logical size/offset, and a
canonical shadow-side Base-action-prefix
SHA-256 after `action_plan` is created and before the step-0 callback. The digest covers each executed
action's dtype, shape, and bytes in order and is updated inside the contained helper after `env.step`
and before the corresponding callback. Use a domain-separated
`LOGIV_BASE_ACTION_PREFIX_V1` SHA-256 stream with length-prefixed dtype, shape and contiguous bytes;
`update_and_hexdigest(None)` performs no update, so the step-0 digest is the hash of the domain prefix
alone. Each update is transactional: hash into a copied digest and replace the internal digest only
after dtype/shape/byte encoding succeeds, so a caught encoding exception cannot leave a partially
advanced prefix. Implement that algorithm once as `BaseActionPrefixHasher`; Task 6 reuses it when writing
`base_execution.json`, so parity and root provenance cannot silently use different hash formats.
Define one nested helper
used by both the initial and post-action paths:

```python
    def call_shadow(observation, action, policy_step):
        nonlocal shadow_calls, shadow_errors, shadow_wall_seconds, shadow_parity_valid
        if shadow_observer is None:
            return
        shadow_calls += 1
        shadow_started = None
        python_rng_state = None
        numpy_rng_state = None
        stage = "RNG_CAPTURE"

        def record_failure(failure_stage, error):
            nonlocal shadow_errors
            shadow_errors += 1
            shadow_failure_records.append(
                ShadowFailureRecord(
                    policy_step=policy_step,
                    stage=failure_stage,
                    reason=type(error).__name__[:128],
                )
            )

        try:
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            stage = "CLOCK_START"
            shadow_started = clock()
            stage = "ACTION_HASH"
            copied_action = None if action is None else np.array(action, copy=True)
            action_prefix_sha256 = shadow_action_prefix.update_and_hexdigest(copied_action)
            stage = "INPUT_COPY"
            copied_observation = copy.deepcopy(observation)
            pending_actions = (
                np.stack(tuple(action_plan), axis=0)
                if action_plan
                else np.empty((0, 7), dtype=current_chunk_dtype)
            )
            stage = "ENVELOPE_READ"
            active_envelope = (
                None
                if request_envelope_reader is None or current_chunk_request_index is None
                else request_envelope_reader(current_chunk_request_index, True)
            )
            next_envelope = (
                None
                if request_envelope_reader is None
                else request_envelope_reader(inference_requests, False)
            )
            stage = "OBSERVER_ESCAPE"
            shadow_observer(
                ShadowStepContext(
                    observation=copied_observation,
                    last_action=copied_action,
                    policy_step=policy_step,
                    base_policy_request_count=inference_requests,
                    active_base_request_index=current_chunk_request_index,
                    next_base_request_index=inference_requests,
                    active_base_request_envelope_json=active_envelope,
                    next_base_replay_envelope_json=next_envelope,
                    base_action_response_size=current_response_size,
                    base_action_chunk_size=current_chunk_size,
                    pending_base_action_offset=current_chunk_offset,
                    pending_base_actions=np.array(pending_actions, copy=True),
                    base_action_prefix_sha256=action_prefix_sha256,
                )
            )
        except Exception as error:
            record_failure(stage, error)
        finally:
            try:
                if shadow_started is not None:
                    shadow_wall_seconds += clock() - shadow_started
            except Exception as error:
                record_failure("CLOCK_END", error)
            try:
                if python_rng_state is not None:
                    random.setstate(python_rng_state)
            except Exception as error:
                record_failure("PYTHON_RNG_RESTORE", error)
                shadow_parity_valid = False
            try:
                if numpy_rng_state is not None:
                    np.random.set_state(numpy_rng_state)
            except Exception as error:
                record_failure("NUMPY_RNG_RESTORE", error)
                shadow_parity_valid = False
```

Before each Base inference, set `current_chunk_request_index` to the zero-based request envelope index;
after a successful response increment `inference_requests`. Thus step 0 has count 0, active index
`None`, next index 0; the first executed chunk has count 1, active index 0, next index 1. Store
`current_response_size=action_chunk.shape[0]` for audit, but define `current_chunk_size` strictly as
the number of actions actually enqueued after the `action_chunk[:replan_steps, :7]` slice. Reset
`current_chunk_dtype` from that sliced chunk without coercion (use a fixed documented float dtype
only before any chunk exists at step 0). Build a nonempty pending suffix with `np.stack`, never
`np.asarray(..., dtype=float64)`, so dtype/shape/bytes stay exact. Reset
`current_chunk_offset=0` whenever Base returns that logical chunk;
increment the offset after each
executed action from that chunk. The callback receives the offset and exact unexecuted suffix; neither
field is reconstructed later from `ceil(policy_step / replan_steps)`.

Extend `EpisodeSeededClient` with a canonical `BaseRequestEnvelopeV1` containing episode seed,
zero-based inference index, and frozen policy-client/checkpoint-config hash. Before each infer, append
that exact canonical JSON to an immutable per-episode issued-request log; after the server echoes the
matching RNG envelope, mark that index acknowledged without changing the envelope bytes.
`request_envelope_reader(index, require_issued=True)` returns canonical JSON only for an issued and
acknowledged index. With `require_issued=False`, it deterministically renders the future replay
envelope for `next_base_request_index` from the same frozen episode seed/config without reading the
client's mutable current counter. Missing support returns `None` and later makes the root ineligible
for the live-continuation estimand. Add tests for the first chunk `(active=0,next=1)`, a second chunk,
step 0 `(active=None,next=0)`, server-envelope rejection, and a generic client with no replay support.
Add optional constructor keyword `policy_client_config_sha256: str | None = None` for backward
compatibility. If absent, inference behavior is unchanged but replay-envelope rendering returns
`None`; Task 6 must supply the frozen hash for Shadow collection and write the same hash for Base.

`ShadowFailureRecord.stage` is restricted to `RNG_CAPTURE`, `CLOCK_START`, `ACTION_HASH`,
`INPUT_COPY`, `ENVELOPE_READ`, `OBSERVER_ESCAPE`, `CLOCK_END`, `PYTHON_RNG_RESTORE`, and
`NUMPY_RNG_RESTORE`. `reason` is only the exception type name capped at 128 characters; stage + type
is the stable reason code. The evaluator can therefore write a step-0 `NOT_ATTEMPTED` reason even
when the observer was never entered; exception messages, tracebacks and environment secrets are not
persisted, and failure reporting does not invoke a user-defined exception `__str__`.

Call `call_shadow(obs, None, 0)` once after reset/set-init/wait and before the first observation is sent
to the Base policy. Call `call_shadow(obs, action, len(executed_actions))` immediately after every
policy `env.step`, including the terminal step, before the existing `done` break. All shadow input
preparation/accounting—including clock, RNG state and prefix hashing—is inside the containment
helper; a failing `deepcopy`, callback, RNG-consuming clock, or provider cannot abort or perturb the
Base loop. Return the counters and immutable tuple of failure records in `EpisodeOutcome`. Never
use the observer's return value.

Extend the parity test with a nested dict/list mutation and compare the post-episode Python/NumPy RNG
draws against a separately seeded Base run. Make the injected clock consume both Python and NumPy
RNG to prove capture happens before the start clock and restore happens after the end clock. Add a
second test whose nested value raises from `__deepcopy__`; it must increment `shadow_errors`, preserve
Base actions/outcome/requests, continue, and expose an `INPUT_COPY` failure record at policy step 0
which Task 6 maps to `NOT_ATTEMPTED`.

Assert step 0 has pending shape `(0, 7)`, response size `None`, logical chunk size/offset 0,
`(request_count, active_index, next_index) == (0, None, 0)`, a domain-only prefix hash, and a rendered
next envelope when replay support exists. At every action callback, independently replay
`BaseActionPrefixHasher` over `outcome.actions[:policy_step]` and compare the exact digest and logical
pending-suffix invariant.

Python/NumPy `setstate` with a state just returned by the same standard-library implementation is a
trusted platform primitive. If restoration nevertheless raises, Base still runs, but
`shadow_parity_valid=False`; the paired parity row is invalid and cannot support the nominal
non-destruction claim. Phase 0 acceptance requires zero such records.

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
- Modify: `src/pi05_libero_repro/logiv/model.py`
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: an audited `FactSnapshot`, flattened MuJoCo state, raw LIBERO observation, immutable episode
  lineage/seed metadata, the pending Base action chunk, policy replay provenance, and a Base
  action-prefix hash.
- Produces: `RecoverySplit`, `RecoveryRootManifest`, `RecoveryRootArtifacts`,
  `make_recovery_root_manifest`, `write_recovery_root`, `load_recovery_root`, and
  `find_orphan_root_temps`.

- [ ] **Step 1: Write ID semantics and round-trip tests**

```python
def test_group_id_ignores_branch_seeds_but_root_id_does_not():
    first = _manifest(perturbation_seed=10, branch_seed=1)
    second = _manifest(perturbation_seed=11, branch_seed=2)
    assert first.recovery_group_id == second.recovery_group_id
    assert first.root_id != second.root_id


def test_candidate_upgrade_gets_a_new_artifact_id_but_not_a_new_independent_group():
    candidate = _manifest(
        policy_step=15,
        deviation_status="ANOMALY_CANDIDATE",
    )
    confirmed = _manifest(
        policy_step=20,
        deviation_status="CONFIRMED_DEVIATION",
    )
    assert candidate.root_id != confirmed.root_id
    assert candidate.recovery_group_id == confirmed.recovery_group_id
    assert candidate.independence_unit_id == confirmed.independence_unit_id
    assert candidate.event_origin_parent_sha256 == confirmed.event_origin_parent_sha256


def test_parent_snapshots_from_one_trajectory_are_one_independence_unit():
    early = _manifest(event_origin_parent_sha256="1" * 64, policy_step=15)
    late = _manifest(event_origin_parent_sha256="2" * 64, policy_step=25)
    assert early.recovery_group_id != late.recovery_group_id
    assert early.independence_unit_id == late.independence_unit_id


def test_recovery_root_round_trip_preserves_exact_physics_and_observation(tmp_path):
    simulator_state = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    observation = _observation()
    pending_actions = manifest_pending_actions()
    manifest = _manifest(
        simulator_state=simulator_state, pending_base_actions=pending_actions
    )
    artifacts = write_recovery_root(
        tmp_path, manifest, simulator_state, observation, pending_actions
    )
    loaded_manifest, loaded_state = load_recovery_root(artifacts.directory)
    assert loaded_manifest == manifest
    np.testing.assert_array_equal(loaded_state["simulator_state"], simulator_state)
    for key, value in observation.items():
        np.testing.assert_array_equal(loaded_state[key], value)
    np.testing.assert_array_equal(
        loaded_state["pending_base_actions"], pending_actions
    )


def test_audited_fact_universe_round_trips_unknown_and_rejects_conflicts(tmp_path):
    snapshot = _audited_snapshot(
        true=("at(book_1,table)",),
        false=("in(book_1,caddy)",),
        unknown=("holding(book_1)",),
    )
    manifest = _manifest(snapshot=snapshot)
    loaded, _ = _round_trip(tmp_path, manifest)
    assert loaded.unknown_facts == ("holding(book_1)",)
    with pytest.raises(ValueError, match="partition|universe"):
        _audited_snapshot(
            true=("at(book_1,table)",),
            false=("at(book_1,table)",),
            unknown=(),
        )


def test_standalone_root_rejects_contract_or_evidence_timing_tampering(tmp_path):
    artifacts = _write_manifest_root(tmp_path, _manifest())
    _tamper_root_json(
        artifacts.manifest_json, "monitor_contract_json", _contract_with_changed_ttl()
    )
    with pytest.raises(ValueError, match="contract|hash"):
        load_recovery_root(artifacts.directory)
    _restore_root_json(artifacts.manifest_json)
    _tamper_evidence(
        artifacts.manifest_json, field="effect_due_policy_step", delta=1
    )
    with pytest.raises(ValueError, match="due|rule"):
        load_recovery_root(artifacts.directory)


def test_crash_before_directory_publish_never_exposes_half_root(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery_records, "_publish_directory", _raise_oserror)
    with pytest.raises(OSError):
        _write_manifest_root(tmp_path, _manifest())
    assert not (tmp_path / _manifest().root_id).exists()
    assert find_orphan_root_temps(tmp_path)
```

The `_manifest` helper must pass identical task/scene/object/initial/parent/family fields while varying only the explicit seeds in the first test. `_observation` must contain `agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos` NumPy arrays.

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py`

Expected: FAIL during collection because `recovery_records` does not exist.

- [ ] **Step 3: Implement canonical IDs and manifest construction**

Extend `FactSnapshot` compatibly by appending optional audit fields after the existing four fields:

```python
    fact_universe: FrozenSet[Fact] | None = None
    fact_universe_version: str | None = None
    fact_universe_sha256: str | None = None
    evidence_payload_json: str | None = None
```

Legacy unit snapshots may leave all four fields null, but recovery-root creation and live shadow
certification require all four. For an audited snapshot, enforce
`true_facts ∪ false_facts ⊆ fact_universe`; compute explicit unknown as
`fact_universe - true_facts - false_facts`; reject known conflicts, missing universe members,
non-canonical/duplicate facts, partial audit fields, or a universe hash mismatch.
`fact_universe_sha256` is a domain-separated hash of universe version plus sorted PDDL facts and
excludes epoch, pixels and evidence. `evidence_payload_json` is the canonical grounder payload that
already contains epoch, observation hash, sorted `(fact, TruthValue)` pairs and dominance overrides;
re-hash it and require equality with the existing `evidence_hash`.

Change `LiberoOracleGrounder` to pass `frozenset(values)`, the frozen grounding-rule/coverage version,
its universe hash, and the exact canonical evidence JSON into `FactSnapshot`. Add adapter tests for
TRUE/FALSE/UNKNOWN partition, epoch/noise changing evidence but not universe hash, evidence-payload
tampering, missing/duplicate/conflicting facts, and a legacy snapshot being rejected only at the
recovery-root boundary rather than globally.

Define:

```python
class RecoverySplit(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    HELDOUT = "HELDOUT"


class CollectionLabel(str, Enum):
    DEV_COLLECTION = "DEV_COLLECTION"
    IMPORTED_FROZEN = "IMPORTED_FROZEN"


@dataclass(frozen=True)
class RecoveryRootManifest:
    schema_version: int
    root_id: str
    recovery_group_id: str
    independence_unit_id: str
    split: RecoverySplit
    collection_label: CollectionLabel
    task_id: int
    episode_idx: int
    scene_sha256: str
    object_instance_ids: tuple[str, ...]
    initial_state_sha256: str
    source_parent_snapshot_sha256: str | None
    event_origin_parent_sha256: str
    parent_trajectory_lineage_sha256: str
    base_prompt_sha256: str
    base_checkpoint_sha256: str
    policy_client_config_sha256: str
    perturbation_family: str
    perturbation_seed: int | None
    branch_seed: int | None
    master_seed: int
    policy_seed: int
    simulator_seed: int
    trigger_class: str
    deviation_status: str
    deviation_event_id: str
    historical_failure_evidence_json: tuple[str, ...]
    relevant_fact_sha256: str
    source_graph_version: str
    source_observation_generation: int
    certificate_state: str
    grounding_rule_sha256: str
    event_detector_sha256: str
    monitor_contract_sha256: str
    monitor_contract_json: str
    policy_step: int
    policy_request_generation: int
    base_policy_request_count: int
    active_base_request_index: int | None
    next_base_request_index: int
    active_base_request_envelope_json: str | None
    active_base_request_envelope_sha256: str | None
    next_base_replay_envelope_json: str | None
    next_base_replay_envelope_sha256: str | None
    policy_replay_contract_sha256: str | None
    base_action_response_size: int | None
    base_action_chunk_size: int
    pending_base_action_offset: int
    pending_base_action_count: int
    pending_base_actions_sha256: str
    live_base_continuation_eligible: bool
    live_base_continuation_ineligibility_reason: str | None
    simulator_state_sha256: str
    observation_sha256: str
    state_fingerprint: str
    base_action_prefix_sha256: str
    fact_epoch_id: int
    fact_universe_version: str
    fact_universe_sha256: str
    fact_evidence_hash: str
    fact_evidence_payload_json: str
    true_facts: tuple[str, ...]
    false_facts: tuple[str, ...]
    unknown_facts: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryRootArtifacts:
    directory: Path
    manifest_json: Path
    state_npz: Path
```

`make_recovery_root_manifest` is keyword-only and accepts, in dataclass field order,
`split`, `collection_label`, `task_id`, `episode_idx`, `scene_sha256`,
`object_instance_ids`, `initial_state_sha256`, `source_parent_snapshot_sha256`,
`event_origin_parent_sha256`, `parent_trajectory_lineage_sha256`, `base_prompt_sha256`,
`base_checkpoint_sha256`, `policy_client_config_sha256`, `perturbation_family`,
`perturbation_seed`, `branch_seed`, `master_seed`, `policy_seed`, `simulator_seed`,
`trigger_class`, `deviation_status`, `deviation_event_id`,
`historical_failure_evidence`, `relevant_fact_sha256`, `source_graph_version`,
`source_observation_generation`, `certificate_state`, `grounding_rule_sha256`,
`event_detector_sha256`, `monitor_contract_json`, `policy_step`, `policy_request_generation`,
`base_policy_request_count`, `active_base_request_index`, `next_base_request_index`,
`active_base_request_envelope_json`, `next_base_replay_envelope_json`,
`policy_replay_contract_sha256`, `base_action_response_size`, `base_action_chunk_size`,
`pending_base_action_offset`, followed by
`simulator_state: np.ndarray`,
`observation: Mapping[str, np.ndarray]`, `pending_base_actions: np.ndarray`,
`base_action_prefix_sha256`, and
`snapshot: FactSnapshot`.
It returns `RecoveryRootManifest`. `write_recovery_root(output_dir, manifest,
simulator_state, observation, pending_base_actions)` returns `RecoveryRootArtifacts`.
`load_recovery_root(directory)` returns
`tuple[RecoveryRootManifest, Mapping[str, np.ndarray]]`. These names and orders are
the final interfaces used by Tasks 4 and 6.

`monitor_contract_json` is the exact canonical contract with its self-hash field present; parse it,
re-canonicalize it, recompute `monitor_contract_sha256`, and reject an unknown contract schema or a
hash mismatch. Its task, grounding-rule and event-detector fields must equal the manifest fields.
Each `ActionEventEvidence` is serialized as one canonical sorted JSON object with the exact typed
fields from Task 5; the manifest stores those objects in deterministic order. Reject a bare evidence
kind/string, an unknown/missing field, a detector-hash mismatch,
`start_policy_step > effect_due_policy_step`, `effect_due_policy_step > emitted_policy_step`,
`emitted_policy_step > evidence_expires_policy_step`, or duplicate `evidence_id`. Permit the same
`attempt_id` on different evidence kinds, but reject duplicate `(attempt_id, evidence_kind)`.
The loader parses and re-canonicalizes each record, resolves `rule_id` in the embedded contract, and
verifies detector hash plus the exact due/expiry equations: action timeout uses
`due=start+rule.effect_due_after_policy_steps` and `expiry=emitted+rule.evidence_ttl_policy_steps`;
abnormal transfer uses `due=emitted` and the same rule evidence TTL; Goal regression/progress use
their reserved rule IDs and corresponding contract
windows. It also verifies the attempt is within the rule's manipulation-attribution window. A root
is independently verifiable without a mutable external contract registry.

Use canonical sorted JSON for ID payloads. `recovery_group_id` hashes exactly `task_id`,
`scene_sha256`, sorted `object_instance_ids`, `initial_state_sha256`,
`event_origin_parent_sha256`, and `perturbation_family`. `event_origin_parent_sha256` is latched on
the first anomaly observation and reused for every immutable candidate→confirmed artifact of the
same `deviation_event_id`; it is not recomputed from the later capture frame.
`source_parent_snapshot_sha256` means the exact clean simulator snapshot from which a synthetic
perturbation was generated and is required for synthetic roots. Naturally observed Base failures
store it as null rather than inventing a clean parent; their episode ancestry comes from
`parent_trajectory_lineage_sha256` and their event origin from the signed-fact hash above.
`independence_unit_id` hashes only `task_id`, `scene_sha256`, `initial_state_sha256`, and
`parent_trajectory_lineage_sha256`, intentionally ignoring capture snapshot, perturbation family,
seeds, and deviation status. `root_id` additionally hashes
`perturbation_seed`, `branch_seed`, `policy_step`, `deviation_status`, `deviation_event_id`, and the
exact simulator-state SHA-256. Thus a later candidate→confirmed upgrade cannot collide with an
immutable earlier artifact even if physics is bit-identical; both remain in the same recovery group
and one independence unit. Compute `state_fingerprint` from the simulator state rounded to
four decimal places, including dtype and shape in the digest. Reject negative IDs/steps/seeds and
non-64-character lowercase SHA-256 fields.

`observation_sha256` uses sorted observation keys and, for each required NumPy value, length-prefixed
key, dtype, shape, and contiguous exact bytes. Object arrays and undeclared/non-NumPy observation
values are rejected rather than pickled. Recompute `fact_universe_sha256` from the saved version and
complete partition, then recompute `fact_evidence_hash` from canonical
`fact_evidence_payload_json`. Verify that payload values yield exactly the saved
true/false/unknown partition, that its epoch equals `fact_epoch_id`, and that its observation hash
matches the loaded observation. Reject a negative fact epoch. UNKNOWN
means “registered in this versioned universe but not reliably TRUE/FALSE”; unregistered predicates
are absent, not silently labeled UNKNOWN.

`deviation_status` is exactly `ANOMALY_CANDIDATE` or `CONFIRMED_DEVIATION`.
Candidate roots may be retained for detector development, but downstream recovery training and
capability validation must filter to confirmed deviations or separately generated perturbation
roots; they may not silently treat static candidates as failures. `certificate_state` is exactly
`CURRENT`, `STALE`, or `RECONCILED`. A stale source certificate is valid diagnostic provenance but
can never be interpreted as a permit. A root with `certificate_state=STALE` is never training
eligible from this observation alone: it requires a separately signed fresh label whose state hash,
graph version, relevant-fact hash, event ID, and evidence records match the root.
`deviation_event_id` participates in the exact artifact
`root_id` as stated above but is excluded from `recovery_group_id`; relevant-fact, graph and rule
version hashes remain audit fields only. Detector or graph changes cannot manufacture a new
independent group or split correlated physics states across datasets.

The saved `pending_base_actions` contains the exact unexecuted suffix of the current Base action
chunk. Verify its dtype/shape/bytes against `pending_base_actions_sha256`, and verify
`pending_base_action_offset + pending_base_action_count == base_action_chunk_size`. The
request count/indices and both replay-envelope records are required provenance. A missing envelope
stores null JSON/hash plus an explicit ineligibility reason and marks the root ineligible for the
live-Base-continuation estimand; it may only enter
the separately registered both-arms-flush diagnostic.
Parse and re-canonicalize non-null envelope JSON; its seed/config/index must match the episode
metadata and corresponding active/next fields before hashing. A mismatch is corruption, not merely
an ineligible live-continuation root.
Envelope availability is necessary but not sufficient: `live_base_continuation_eligible` also
requires a frozen `policy_replay_contract_sha256` backed by an exact held-out request-replay test for
the same client/checkpoint config. Without it, the root stays in the both-arms-flush diagnostic.
`policy_request_generation` is the overlay response-invalidation generation (fixed at 0 in Phase 0),
not a Base inference count or index.

- [ ] **Step 4: Implement atomic persistence and verified loading**

`write_recovery_root` must acquire the output registry lock, create a unique sibling temporary
directory `<output>/.<root_id>.tmp.<nonce>/`, and write final-named `state.npz` plus
`recovery_root.json` inside it. The NPZ includes simulator state, raw observation, and pending Base
actions. Flush/fsync both files and the temporary directory, then publish with one same-filesystem
directory rename to `<output>/<root_id>/` and fsync the parent. `load_recovery_root` ignores all temp
directories and only accepts the final directory containing both verified files. On retry, an
existing identical final root is success and a non-identical one is corruption; orphan temp
directories are reported by a validation/cleanup command and never treated as roots or overwritten
implicitly. This avoids exposing a state-only half root after a crash. `load_recovery_root`
must reconstruct the enums/dataclass, load with `allow_pickle=False`, and verify simulator-state,
pending-action, evidence, and observation hashes before returning.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py`

Expected: both ID and exact round-trip tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pi05_libero_repro/logiv/recovery_records.py src/pi05_libero_repro/logiv/model.py src/pi05_libero_repro/logiv/libero_adapter.py tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py
git commit -m "feat(logiv): persist reproducible recovery roots"
```

---

### Task 4: Mechanical recovery split isolation

**Files:**
- Create: `src/pi05_libero_repro/logiv/recovery_splits.py`
- Create: `scripts/validate_recovery_dataset.py`
- Create: `scripts/build_recovery_training_manifest.py`
- Create: `scripts/allocate_recovery_roles.py`
- Create: `tests/logiv/test_recovery_splits.py`

**Interfaces:**
- Consumes: `Sequence[RecoveryRootManifest]`, optional signed fresh labels, or directories containing
  `recovery_root.json`.
- Produces: `RecoverySplitError`, `DatasetRole`, `FreshRecoveryLabel`,
  `validate_recovery_splits(manifests) -> Mapping[str, int]`,
  `build_training_manifest(...) -> TrainingManifest`,
  `validate_role_allocation`, `append_role_allocation`,
  `validate_dataset_roles(manifests, *, role_allocations, role_registry_head)`, and zero/nonzero
  validation CLIs.

- [ ] **Step 1: Write leakage tests for every forbidden relation**

```python
def test_validator_rejects_same_recovery_group_across_splits():
    train = _manifest(RecoverySplit.TRAIN, perturbation_seed=1, branch_seed=1)
    heldout = _manifest(RecoverySplit.HELDOUT, perturbation_seed=2, branch_seed=2)
    assert train.recovery_group_id == heldout.recovery_group_id
    with pytest.raises(RecoverySplitError, match="recovery_group_id"):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_same_independence_unit_with_different_events():
    train = _manifest(RecoverySplit.TRAIN, event_origin_parent_sha256="1" * 64)
    heldout = _manifest(
        RecoverySplit.HELDOUT,
        event_origin_parent_sha256="2" * 64,
        perturbation_family="partial_place",
    )
    assert train.recovery_group_id != heldout.recovery_group_id
    assert train.independence_unit_id == heldout.independence_unit_id
    with pytest.raises(RecoverySplitError, match="independence_unit_id"):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_same_initial_state_across_parent_seed_lineages():
    train = _manifest(RecoverySplit.TRAIN, parent_trajectory_lineage_sha256="3" * 64)
    heldout = _manifest(
        RecoverySplit.HELDOUT,
        parent_trajectory_lineage_sha256="4" * 64,
        event_origin_parent_sha256="5" * 64,
    )
    with pytest.raises(RecoverySplitError, match="initial_state_sha256"):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_near_duplicate_state_fingerprint_across_splits():
    train = _manifest(RecoverySplit.TRAIN, simulator_state=np.array([1.00001]))
    heldout = _manifest(
        RecoverySplit.HELDOUT,
        simulator_state=np.array([1.00002]),
        initial_state_sha256="6" * 64,
        parent_trajectory_lineage_sha256="7" * 64,
        event_origin_parent_sha256="8" * 64,
    )
    assert train.simulator_state_sha256 != heldout.simulator_state_sha256
    assert train.state_fingerprint == heldout.state_fingerprint
    with pytest.raises(RecoverySplitError, match="state_fingerprint"):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_same_synthetic_source_parent_across_splits():
    train = _manifest(
        RecoverySplit.TRAIN, source_parent_snapshot_sha256="c" * 64
    )
    heldout = _manifest(
        RecoverySplit.HELDOUT,
        source_parent_snapshot_sha256="c" * 64,
        initial_state_sha256="d" * 64,
        parent_trajectory_lineage_sha256="e" * 64,
        event_origin_parent_sha256="f" * 64,
        simulator_state=np.array([3.0]),
    )
    with pytest.raises(RecoverySplitError, match="source_parent_snapshot_sha256"):
        validate_recovery_splits([train, heldout])


def test_validator_rejects_perturbation_seed_reuse_within_task():
    train = _manifest(RecoverySplit.TRAIN, perturbation_seed=44)
    dev = _manifest(
        RecoverySplit.DEV,
        perturbation_seed=44,
        initial_state_sha256="9" * 64,
        parent_trajectory_lineage_sha256="a" * 64,
        event_origin_parent_sha256="b" * 64,
        simulator_state=np.array([2.0]),
    )
    with pytest.raises(RecoverySplitError, match="perturbation_seed"):
        validate_recovery_splits([train, dev])


def test_validator_reports_disjoint_dataset_counts():
    summary = validate_recovery_splits(
        [
            _manifest(RecoverySplit.TRAIN, deviation_status="CONFIRMED_DEVIATION"),
            _manifest(
                RecoverySplit.DEV,
                initial_state_sha256="4" * 64,
                parent_trajectory_lineage_sha256="5" * 64,
                deviation_status="ANOMALY_CANDIDATE",
            ),
        ]
    )
    assert summary == {
        "TRAIN": 1,
        "DEV": 1,
        "HELDOUT": 0,
        "ANOMALY_CANDIDATE": 1,
        "CONFIRMED_DEVIATION": 1,
        "unique_independence_units": 2,
        "total": 2,
    }


def test_raw_candidate_and_stale_root_are_not_training_eligible():
    candidate = _manifest(RecoverySplit.TRAIN, deviation_status="ANOMALY_CANDIDATE")
    stale = _manifest(
        RecoverySplit.TRAIN,
        deviation_status="CONFIRMED_DEVIATION",
        certificate_state="STALE",
    )
    with pytest.raises(RecoverySplitError, match="candidate|fresh label"):
        build_training_manifest(
            [candidate, stale],
            fresh_labels=[],
            **_training_build_kwargs(role=DatasetRole.TRAIN),
        )


def test_signed_fresh_label_can_admit_matching_stale_root_only():
    stale = _manifest(
        RecoverySplit.TRAIN,
        deviation_status="CONFIRMED_DEVIATION",
        certificate_state="STALE",
    )
    label = _fresh_label_for(stale)
    training = build_training_manifest(
        [stale],
        fresh_labels=[label],
        **_training_build_kwargs(role=DatasetRole.TRAIN),
    )
    assert tuple(item.root_id for item in training.entries) == (stale.root_id,)


def test_role_validator_rejects_one_independence_unit_in_two_roles():
    train = _training_manifest(role=DatasetRole.TRAIN, independence_unit_id="u")
    calibration = _training_manifest(
        role=DatasetRole.PERMIT_CALIBRATION, independence_unit_id="u"
    )
    allocations, head = _conflicting_role_registry_fixture(train, calibration)
    with pytest.raises(RecoverySplitError, match="independence_unit_id"):
        validate_dataset_roles(
            [train, calibration],
            role_allocations=allocations,
            role_registry_head=head,
        )


def test_role_allocation_must_cover_source_units_exactly_once():
    source = _source_dataset(independence_units=("u1", "u2"))
    incomplete = _role_allocation(source, entries=(("u1", DatasetRole.TRAIN),))
    with pytest.raises(RecoverySplitError, match="u2|complete"):
        validate_role_allocation(source, incomplete)


def test_role_registry_compare_and_swap_rejects_stale_or_alternate_head(tmp_path):
    registry = _registry(tmp_path, head="1" * 64)
    allocation = _role_allocation_for_units("u1")
    with pytest.raises(RecoverySplitError, match="expected head"):
        append_role_allocation(registry, allocation, expected_head="2" * 64)
    assert registry.read_head_sha256() == "1" * 64


def test_repackaging_a_historical_unit_under_a_new_source_hash_cannot_reassign_it(tmp_path):
    registry = _registry_with_assignment(
        tmp_path, source_dataset_sha256="1" * 64,
        independence_unit_id="u1", role=DatasetRole.TRAIN,
    )
    repackaged = _role_allocation_for_units(
        "u1", source_dataset_sha256="2" * 64,
        role=DatasetRole.PAPER_CONFIRMATION,
    )
    with pytest.raises(RecoverySplitError, match="historical|independence_unit_id"):
        append_role_allocation(
            registry, repackaged, expected_head=registry.read_head_sha256()
        )


def test_role_allocation_rejects_initial_state_or_near_duplicate_across_roles():
    source = _source_dataset_with_related_units(
        relation="initial_state_sha256", roles=(DatasetRole.TRAIN, DatasetRole.PAPER_CONFIRMATION)
    )
    with pytest.raises(RecoverySplitError, match="initial_state_sha256"):
        validate_role_allocation(source.manifests, source.allocation)


def test_role_must_match_frozen_recovery_split_and_collection_label():
    heldout = _manifest(RecoverySplit.HELDOUT, collection_label=CollectionLabel.IMPORTED_FROZEN)
    invalid = _role_allocation_for_manifest(heldout, role=DatasetRole.TRAIN)
    with pytest.raises(RecoverySplitError, match="role|split"):
        validate_role_allocation([heldout], invalid)
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_recovery_splits.py`

Expected: FAIL during collection because `recovery_splits` does not exist.

- [ ] **Step 3: Implement the validator**

For each key below, build `key -> set(split)` and raise `RecoverySplitError` when the set has more than one member:

```python
checks = {
    "recovery_group_id": lambda item: item.recovery_group_id,
    "independence_unit_id": lambda item: item.independence_unit_id,
    "initial_state_sha256": lambda item: item.initial_state_sha256,
    "source_parent_snapshot_sha256": lambda item: item.source_parent_snapshot_sha256,
    "parent_trajectory_lineage_sha256": lambda item: item.parent_trajectory_lineage_sha256,
    "state_fingerprint": lambda item: item.state_fingerprint,
    "perturbation_seed": lambda item: (
        (item.task_id, item.perturbation_seed)
        if item.perturbation_seed is not None
        else None
    ),
}
```

Skip `None` only for optional `source_parent_snapshot_sha256` and perturbation seed; a null value is
not itself a shared lineage relation.

Define the derived-record boundary exactly:

```python
class DatasetRole(str, Enum):
    TRAIN = "TRAIN"
    DEV_SELECTION = "DEV_SELECTION"
    PERMIT_CALIBRATION = "PERMIT_CALIBRATION"
    PAPER_CONFIRMATION = "PAPER_CONFIRMATION"


@dataclass(frozen=True)
class RoleAllocationEntry:
    source_dataset_sha256: str
    independence_unit_id: str
    role_isolation_keys: tuple[str, ...]
    role: DatasetRole


@dataclass(frozen=True)
class RoleAllocationManifest:
    schema_version: int
    source_dataset_sha256: str
    source_independence_unit_ids: tuple[str, ...]
    entries: tuple[RoleAllocationEntry, ...]
    allocation_version: str
    registry_parent_head_sha256: str
    allocation_sha256: str


@dataclass(frozen=True)
class RoleRegistryHead:
    schema_version: int
    registry_id: str
    revision: int
    parent_head_sha256: str | None
    allocations: tuple[RoleAllocationEntry, ...]
    head_sha256: str


@dataclass(frozen=True)
class FreshRecoveryLabel:
    schema_version: int
    root_id: str
    simulator_state_sha256: str
    fresh_observation_sha256: str
    deviation_event_id: str
    historical_evidence_sha256: str
    relevant_fact_sha256: str
    fresh_fact_epoch_id: int
    fact_universe_sha256: str
    fresh_fact_evidence_hash: str
    graph_version: str
    certificate_sha256: str
    grounding_rule_sha256: str
    event_detector_sha256: str
    monitor_contract_sha256: str
    labeler_version: str
    label_sha256: str


@dataclass(frozen=True)
class TrainingManifestEntry:
    root_id: str
    recovery_group_id: str
    independence_unit_id: str
    fresh_label_sha256: str | None


@dataclass(frozen=True)
class TrainingManifest:
    schema_version: int
    role: DatasetRole
    entries: tuple[TrainingManifestEntry, ...]
    source_dataset_sha256: str
    role_allocation_sha256: str
    role_registry_head_sha256: str
    builder_version: str
    manifest_sha256: str
```

`build_training_manifest(manifests, fresh_labels, *, role, source_dataset_sha256,
role_allocation, role_registry_head, builder_version) -> TrainingManifest` is keyword-only after the
first two arguments. Label, allocation, registry-head and
manifest hashes cover canonical JSON with their own hash field omitted. Entries are sorted by
`root_id`; duplicate root/group/unit associations are rejected rather than silently deduplicated.
For a stale root, the signed fresh label must be produced after exact simulator-state restoration and
must bind a newly observed image/state hash, nonnegative fresh fact epoch, the root's frozen fact
universe, recomputable fresh grounding-evidence hash, unchanged relevant facts/event/history, and
the exact grounding/detector/monitor contract hashes. A fresh epoch/evidence hash need not equal the
old capture epoch/hash, but both payloads must yield the same task-relevant signed facts; otherwise
the label cannot admit the root.
Every `RoleAllocationEntry.source_dataset_sha256` must equal its containing allocation's source
hash. Its sorted `role_isolation_keys` are domain-prefixed canonical hashes recomputed from every
raw relation carried by that unit: unit/group/initial-state/source-parent/parent-lineage/state-
fingerprint/task-perturbation-seed. `RoleAllocationManifest` must list every distinct independence unit in the immutable source dataset
exactly once before any branch outcome is observed; missing/extra/duplicate units or a source hash
mismatch are fatal. The canonical project `RoleRegistryHead` is append-only and maps every previously
seen `independence_unit_id` and every role-isolation key globally to exactly one role; repackaging the
same unit or a related root under a different source-dataset hash cannot reallocate it. The allocation CLI takes the expected current head hash,
locks the canonical registry path, rejects a CAS mismatch or any prior unit, writes a new head with
parent linkage atomically, and records its hash in `run.json` before outcomes. Callers cannot provide
an ad-hoc “complete ledger” list.

`validate_dataset_roles(manifests, *, role_allocations, role_registry_head)` checks role manifests
against their immutable source dataset, corresponding complete allocations, and the exact canonical registry
head, and rejects any root/recovery-group/independence-unit assigned inconsistently. The builder CLI
reads the registry from the frozen project path/ID and requires the allocation/head hashes; an absent,
stale or alternative registry is a hard error for every role. This makes omission of an older
manifest detectable rather than trusting the caller to enumerate history.

Role isolation is at least as strict as raw split isolation. Across `DatasetRole` values, reject a
shared `recovery_group_id`, `independence_unit_id`, `initial_state_sha256`, non-null synthetic
source-parent, parent trajectory lineage, rounded state fingerprint, or task/perturbation seed. The
allocation validator derives those relations from the immutable root manifests—it cannot verify only
the caller's unit-ID list. For `IMPORTED_FROZEN`, freeze the compatibility map:

```text
TRAIN               -> RecoverySplit.TRAIN
DEV_SELECTION       -> RecoverySplit.DEV
PERMIT_CALIBRATION  -> RecoverySplit.HELDOUT
PAPER_CONFIRMATION  -> RecoverySplit.HELDOUT
```

`PERMIT_CALIBRATION` and `PAPER_CONFIRMATION` remain distinct roles and therefore also undergo the
cross-role relation checks. Phase 0 `DEV_COLLECTION` roots all carry `RecoverySplit.DEV` and may map
only to TRAIN/DEV_SELECTION; that explicit development exception cannot admit them to permit/paper.
Permit/paper roots require `IMPORTED_FROZEN`, a fresh pre-outcome allocation, and frozen
code/config hashes.

Also reject duplicate `root_id` values even within one split. Parse every canonical evidence record;
a `CONFIRMED_DEVIATION` must contain at least one active, exact object/effect/region-matched strong
record (`GOAL_REGRESSION`, `ATTEMPTED_EFFECT_TIMEOUT`, or
`ABNORMAL_TRANSFER_AFTER_MANIPULATION`); `PROGRESS_TIMEOUT` alone is invalid. Candidate and
confirmed artifacts for one semantic event must share the latched event origin, recovery group, and
independence unit. Return deterministic counts for all three splits, both deviation statuses,
`unique_independence_units`, and `total`. Raw validation reports provenance only and does not label
any observed root training eligible.

Define immutable data roles `TRAIN`, `DEV_SELECTION`, `PERMIT_CALIBRATION`, and
`PAPER_CONFIRMATION`. `build_training_manifest` accepts only confirmed `CURRENT`/`RECONCILED` roots,
or `STALE` roots with a matching signed `FreshRecoveryLabel`; it always rejects candidates. It writes
canonical root IDs, recovery-group IDs, independence-unit IDs, source dataset hash, role, builder
version, and a manifest SHA-256. A recovery trainer must accept this signed manifest as its only data
entry point and refuse a raw root directory. One independence unit and all descendant recovery
groups may appear in exactly one role; Phase 0 `DEV_COLLECTION` roots can later be assigned only to
`TRAIN` or `DEV_SELECTION`, never to permit calibration or paper confirmation.

`FreshRecoveryLabel` binds `root_id`, exact simulator-state hash, deviation event ID, canonical
evidence hash, fresh relevant-fact hash, new graph version/certificate hash, labeler version, and
`label_sha256`. Its content hash is computed over canonical JSON with `label_sha256` omitted. A label
for a different state, event, detector, or graph cannot rehabilitate a stale root.

- [ ] **Step 4: Add the dataset CLI**

The validation script accepts one or more directory arguments, recursively loads files named
`recovery_root.json` through `load_recovery_root`, validates them, prints the sorted JSON summary, and
exits nonzero with the `RecoverySplitError` message on leakage. It must reject an empty dataset. Add
a separate `scripts/build_recovery_training_manifest.py` command which requires an explicit role and
fresh-label file plus allocation/registry hashes, writes the signed manifest atomically, and never
mutates raw root manifests. `scripts/allocate_recovery_roles.py` creates the complete pre-outcome
allocation and performs the locked expected-head CAS update on the canonical registry.

- [ ] **Step 5: Run unit and CLI tests**

Run: `uv run pytest -q tests/logiv/test_recovery_splits.py`

Expected: all split tests pass.

Run: `uv run python scripts/validate_recovery_dataset.py --help`

Expected: exit 0 and usage text containing `recovery_root.json`.

Run: `uv run python scripts/build_recovery_training_manifest.py --help`

Expected: exit 0 and usage text requiring a dataset role and output manifest.

Run: `uv run python scripts/allocate_recovery_roles.py --help`

Expected: exit 0 and usage text requiring source dataset, complete allocation, canonical registry,
and expected head hash.

- [ ] **Step 6: Commit**

```bash
git add src/pi05_libero_repro/logiv/recovery_splits.py scripts/validate_recovery_dataset.py scripts/build_recovery_training_manifest.py scripts/allocate_recovery_roles.py tests/logiv/test_recovery_splits.py
git commit -m "feat(logiv): enforce recovery split isolation"
```

---

### Task 5: Event-aware deviation monitor and stale-certificate reconciliation

**Files:**
- Create: `src/pi05_libero_repro/logiv/shadow_monitor.py`
- Create: `tests/logiv/test_shadow_monitor.py`
- Create: `configs/logiv/r2m-monitor-evidence-v1.json`
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: a cycle-free `ShadowPlanContext` projected from `CertifiedEpisode`, a versioned
  `MonitorEvidenceContract`,
  `snapshot_reader(observation) -> FactSnapshot`, and an `ActionEventTracker` updated from every
  `ShadowStepContext`.
- Produces: `ShadowPlanContext`, `MonitorEvidenceContract`, `ActionEventEvidence`,
  `ActionEventRule`, `ReservedFactEventRuleId`, `ActionTransitionFeatures`, `VersionedActionEventTracker`,
  `ActionEventTracker`, `TransitionFeatureReader`, `build_libero_transition_feature_reader`,
  `DeviationStatus`, `CertificateState`, `ShadowTrigger`,
  `ShadowMonitorMetrics`, `ShadowCertificateReconciler`, and callable
  `StableRecoveryObserver`.

- [ ] **Step 1: Write candidate-versus-confirmed and normal-state exclusion tests**

```python
def test_static_unplanned_support_is_only_an_anomaly_candidate():
    snapshots = iter([_dropped_snapshot(5), _dropped_snapshot(10), _dropped_snapshot(15), _dropped_snapshot(20)])
    triggers = []
    observer = StableRecoveryObserver(
        plan_context=_shadow_plan_context(),
        monitor_contract=_monitor_contract(
            abnormal_support_surfaces=("kitchen_table_recovery_surface",)
        ),
        snapshot_reader=lambda observation: next(snapshots),
        action_event_tracker=_event_tracker(),
        on_trigger=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_step_context(step))
    assert len(triggers) == 1
    assert triggers[0].policy_step == 15
    assert triggers[0].trigger_class == "UNPLANNED_SUPPORT_STABLE"
    assert triggers[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE
    assert triggers[0].historical_failure_evidence == ()
    assert observer.metrics.snapshot_calls == 4
    assert observer.metrics.anomaly_candidates == 1
    assert observer.metrics.confirmed_deviations == 0


def test_goal_regression_becomes_a_confirmed_deviation_only_after_goal_was_true():
    snapshots = iter([_goal_snapshot(True, 5), _goal_snapshot(False, 10), _goal_snapshot(False, 15), _goal_snapshot(False, 20)])
    triggers = []
    observer = StableRecoveryObserver(
        plan_context=_shadow_plan_context(),
        monitor_contract=_monitor_contract(abnormal_support_surfaces=()),
        snapshot_reader=lambda observation: next(snapshots),
        action_event_tracker=_event_tracker(),
        on_trigger=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_step_context(step))
    assert [trigger.trigger_class for trigger in triggers] == ["COMPLETED_GOAL_REGRESSION_STABLE"]
    assert triggers[0].deviation_status is DeviationStatus.CONFIRMED_DEVIATION
    assert tuple(item.evidence_kind for item in triggers[0].historical_failure_evidence) == (
        "GOAL_REGRESSION",
    )


def test_initial_or_normal_intermediate_surface_never_becomes_a_candidate():
    # Cover both an object still at its declared initial source and a registered
    # settling/holding transition. Neither is a failure, regardless of duration.
    initial = _run_observer(_initial_surface_snapshots(), evidence=())
    settling = _run_observer(
        _dropped_snapshots(), evidence=(_evidence(evidence_kind="TRANSIENT_RELEASE"),)
    )
    assert initial == []
    assert settling == []


def test_progress_timeout_alone_does_not_confirm_but_attempted_effect_timeout_does():
    weak = _run_observer(
        _dropped_snapshots(), evidence=(_evidence(evidence_kind="PROGRESS_TIMEOUT"),)
    )
    strong = _run_observer(
        _dropped_snapshots(),
        evidence=(_evidence(evidence_kind="ATTEMPTED_EFFECT_TIMEOUT"),),
    )
    assert weak[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE
    assert strong[0].deviation_status is DeviationStatus.CONFIRMED_DEVIATION


def test_abnormal_transfer_requires_task_relevant_manipulation_evidence():
    without_event = _run_observer(_transferred_snapshots(), evidence=())
    with_event = _run_observer(
        _transferred_snapshots(),
        evidence=(_evidence(evidence_kind="ABNORMAL_TRANSFER_AFTER_MANIPULATION"),),
    )
    assert without_event[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE
    assert with_event[0].deviation_status is DeviationStatus.CONFIRMED_DEVIATION


def test_late_strong_evidence_upgrades_candidate_once_without_changing_event_id():
    triggers = _run_observer(
        _persistent_dropped_snapshots(),
        evidence_by_step={
            15: (),
            20: (_evidence(evidence_kind="ATTEMPTED_EFFECT_TIMEOUT"),),
            25: (),
        },
    )
    assert [item.deviation_status for item in triggers] == [
        DeviationStatus.ANOMALY_CANDIDATE,
        DeviationStatus.CONFIRMED_DEVIATION,
    ]
    assert len({item.deviation_event_id for item in triggers}) == 1


def test_uncovered_relevant_transition_marks_certificate_stale_but_noise_does_not():
    reconciler = ShadowCertificateReconciler(_shadow_plan_context())
    noise = reconciler.reconcile(_snapshot(5), _same_facts_new_pixels_snapshot(10))
    uncovered = reconciler.reconcile(_snapshot(10), _alternative_producer_snapshot(15))
    assert noise.certificate_state is CertificateState.CURRENT
    assert uncovered.certificate_state is CertificateState.STALE
    still_stale = reconciler.reconcile(
        _alternative_producer_snapshot(15), _no_relevant_change_snapshot(20)
    )
    assert still_stale.certificate_state is CertificateState.STALE


def test_evidence_is_joined_by_object_effect_region_and_active_expiry():
    tracker = _event_tracker(
        records=(
            _evidence(object_id="book_1", attempted_effect="in(book_1,caddy)", due=10, expires=15),
            _evidence(object_id="book_2", attempted_effect="holding(book_2)", due=15, expires=30),
            _evidence(object_id="book_2", attempted_effect="in(book_2,caddy)", due=15, expires=30),
        )
    )
    triggers = _run_observer(
        _book_2_dropped_snapshots(), tracker=tracker, first_confirmation_step=20
    )
    assert len(triggers) == 1
    assert {item.object_id for item in triggers[0].historical_failure_evidence} == {"book_2"}
    assert {item.attempted_effect for item in triggers[0].historical_failure_evidence} == {
        "in(book_2,caddy)"
    }


def test_expired_or_cross_object_evidence_cannot_confirm():
    tracker = _event_tracker(
        records=(
            _evidence(object_id="book_1", attempted_effect="in(book_1,caddy)", due=15, expires=30),
            _evidence(object_id="book_2", attempted_effect="in(book_2,caddy)", due=10, expires=15),
        )
    )
    triggers = _run_observer(
        _book_2_dropped_snapshots(), tracker=tracker, first_confirmation_step=20
    )
    assert [item.deviation_status for item in triggers] == [
        DeviationStatus.ANOMALY_CANDIDATE
    ]


def test_event_tracker_observes_intervening_actions_between_fact_samples():
    tracker = _recording_event_tracker()
    _run_observer(_dropped_snapshots(), tracker=tracker, interval_steps=5)
    assert tracker.observed_policy_steps == list(range(0, 21))


def test_concrete_tracker_emits_timeout_only_at_due_and_cancels_on_success():
    tracker = _concrete_tracker(due_after=3, ttl=5)
    tracker.observe(_raw_grasp_start(step=4, object_id="book_2"))
    tracker.observe(_raw_effect(step=6, object_id="book_2", truth=TruthValue.FALSE))
    assert _active_for_book_2(tracker, policy_step=6) == ()
    tracker.observe(_raw_effect(step=7, object_id="book_2", truth=TruthValue.FALSE))
    evidence = _active_for_book_2(tracker, policy_step=7)
    assert [item.evidence_kind for item in evidence] == ["ATTEMPTED_EFFECT_TIMEOUT"]
    assert evidence[0].effect_due_policy_step == 7
    assert evidence[0].evidence_expires_policy_step == 12

    successful = _concrete_tracker(due_after=3, ttl=5)
    successful.observe(_raw_grasp_start(step=4, object_id="book_2"))
    successful.observe(_raw_effect(step=6, object_id="book_2", truth=TruthValue.TRUE))
    successful.observe(_raw_effect(step=7, object_id="book_2", truth=TruthValue.FALSE))
    assert _active_for_book_2(successful, policy_step=7) == ()


def test_concrete_tracker_rejects_cross_region_unknown_and_expired_evidence():
    unknown = _concrete_tracker(due_after=2, ttl=2)
    unknown.observe(_raw_place_start(step=3, object_id="book_1", destination="wrong_region"))
    unknown.observe(_raw_place_start(step=4, object_id="book_2", destination="caddy"))
    unknown.observe(_raw_effect(step=6, object_id="book_2", truth=TruthValue.UNKNOWN))
    unknown.observe(_raw_effect(step=7, object_id="book_2", truth=TruthValue.FALSE))
    assert _active_for_book_2(unknown, policy_step=7) == ()

    expired = _concrete_tracker(due_after=2, ttl=2)
    expired.observe(_raw_place_start(step=4, object_id="book_2", destination="caddy"))
    expired.observe(_raw_effect(step=6, object_id="book_2", truth=TruthValue.FALSE))
    assert _active_for_book_2(expired, policy_step=8)
    assert _active_for_book_2(expired, policy_step=9) == ()


def test_abnormal_transfer_requires_registered_manipulation_for_same_object():
    tracker = _concrete_tracker(due_after=3, ttl=4)
    tracker.observe(_raw_abnormal_region(step=2, object_id="book_2"))
    assert _active_for_book_2(tracker, policy_step=2) == ()
    tracker.observe(_raw_grasp_start(step=3, object_id="book_1"))
    tracker.observe(_raw_abnormal_region(step=4, object_id="book_2"))
    assert _active_for_book_2(tracker, policy_step=4) == ()
    tracker.observe(_raw_grasp_start(step=5, object_id="book_2"))
    tracker.observe(_raw_abnormal_region(step=6, object_id="book_2"))
    assert [
        item.evidence_kind for item in _active_for_book_2(tracker, policy_step=6)
    ] == ["ABNORMAL_TRANSFER_AFTER_MANIPULATION"]


def test_successful_grasp_keeps_bounded_same_object_manipulation_attribution():
    tracker = _concrete_tracker(attribution_ttl=4, evidence_ttl=3)
    tracker.observe(_raw_grasp_start(step=2, object_id="book_2"))
    tracker.observe(_raw_effect(step=3, object_id="book_2", truth=TruthValue.TRUE))
    tracker.observe(_raw_abnormal_region(step=4, object_id="book_1"))
    assert _active_for_book_1(tracker, policy_step=4) == ()
    tracker.observe(_raw_abnormal_region(step=5, object_id="book_2"))
    evidence = _active_for_book_2(tracker, policy_step=5)
    assert [item.evidence_kind for item in evidence] == [
        "ABNORMAL_TRANSFER_AFTER_MANIPULATION"
    ]
    assert evidence[0].attempt_id == tracker.attempts[0].attempt_id
    assert evidence[0].evidence_id != evidence[0].attempt_id


def test_abnormal_transfer_after_attribution_ttl_is_not_emitted():
    tracker = _concrete_tracker(attribution_ttl=2)
    tracker.observe(_raw_grasp_start(step=2, object_id="book_2"))
    tracker.observe(_raw_effect(step=3, object_id="book_2", truth=TruthValue.TRUE))
    tracker.observe(_raw_abnormal_region(step=5, object_id="book_2"))
    assert _active_for_book_2(tracker, policy_step=5) == ()


def test_tracker_applies_raw_thresholds_and_rejects_rule_identity_mismatch_or_gap():
    tracker = _concrete_tracker(close_threshold=-0.5)
    tracker.observe(_raw_feature(step=0, rule_id="grasp_book_2", gripper_qpos=0.2))
    with pytest.raises(ValueError, match="rule"):
        tracker.observe(_raw_feature(step=1, rule_id="other_rule", gripper_qpos=-0.8))
    tracker.observe(_raw_null_feature_context(step=2))
    tracker.observe(_raw_feature(step=3, rule_id="grasp_book_2", gripper_qpos=-0.8))
    assert tracker.attempt_record_count == 0  # no threshold crossing may bridge the missing step
    tracker.observe(_raw_feature(step=4, rule_id="grasp_book_2", gripper_qpos=0.2))
    tracker.observe(_raw_feature(step=5, rule_id="grasp_book_2", gripper_qpos=-0.8))
    assert tracker.attempt_record_count == 1

    with pytest.raises(ValueError, match="missing rule"):
        tracker.observe(_raw_context_omitting_registered_rule(step=6))


def test_event_ledger_overflow_is_diagnostic_and_cannot_confirm():
    tracker = _event_tracker(max_records=1)
    tracker.observe(_attempt_context(object_id="book_1", attempt_id="a"))
    tracker.observe(_attempt_context(object_id="book_2", attempt_id="b"))
    assert tracker.overflow_count == 1
    triggers = _run_observer(_book_2_dropped_snapshots(), tracker=tracker)
    assert triggers[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE


def test_expiry_does_not_refund_cumulative_evidence_capacity():
    tracker = _concrete_tracker(max_evidence_records=1, due_after=1, ttl=1)
    _emit_timeout(tracker, attempt_id="a", start=1)
    assert _active_for_book_2(tracker, policy_step=4) == ()
    _emit_timeout(tracker, attempt_id="b", start=5)
    assert tracker.overflow_count == 1


def test_monitor_contract_rejects_hash_drift_unregistered_ids_and_nonpositive_windows():
    with pytest.raises(ValueError):
        _load_monitor_contract(hash_override="0" * 64)
    with pytest.raises(ValueError):
        _load_monitor_contract(abnormal_support_surfaces=("unknown_surface",))
    with pytest.raises(ValueError):
        _load_monitor_contract(rule_override={"evidence_ttl_policy_steps": 0})
    with pytest.raises(ValueError):
        _load_monitor_contract(rule_override={"contact_min_count": 2}, keep_old_hash=True)
    with pytest.raises(ValueError, match="interval"):
        _observer(interval_steps=4, monitor_contract=_monitor_contract(monitor_interval_steps=5))


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
        plan_context=_shadow_plan_context(),
        monitor_contract=_monitor_contract(
            abnormal_support_surfaces=("kitchen_table_recovery_surface",)
        ),
        snapshot_reader=reader,
        action_event_tracker=_event_tracker(
            evidence=(_evidence(evidence_kind="ABNORMAL_TRANSFER_AFTER_MANIPULATION"),)
        ),
        on_trigger=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_step_context(step))
    assert observer.metrics.snapshot_errors == 1
    assert len(triggers) == 1
    assert triggers[0].policy_step == 20
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `uv run pytest -q tests/logiv/test_shadow_monitor.py tests/logiv/test_libero_adapter.py`

Expected: FAIL during collection because `shadow_monitor` does not exist.

- [ ] **Step 3: Implement relevant-fact certificates and deterministic deviation records**

`ShadowCertificateReconciler` derives the relevant-fact projection from signed Goal literals,
all registered plan preconditions/effects, holding/handempty, access/articulation, and protected
facts. It hashes only this projection plus the grounding-rule version; raw simulator-state hashes
remain recovery artifacts and are not certificate freshness keys.

For each pair of sampled snapshots it classifies the transition as:

1. `CURRENT`: changed relevant literals are covered by a DAG action whose signed preconditions hold
   in the previous snapshot and whose predecessor obligations are already satisfied by observed
   facts, or no relevant literal changed; emit a new observation generation/relevant-fact hash with
   the same graph version, and never fabricate Base action occurrence IDs;
2. `STALE`: a task-relevant literal changed outside the current DAG coverage, including a legal
   alternative producer not represented by the Initial DAG;
3. `RECONCILED`: reserved for a later callback that rebuilds a shadow graph from fresh facts.

Phase 0 implements and tests `CURRENT`/`STALE`; it never attempts physical authorization.
Continuous pose/image changes and facts outside the relevant projection do not stale the
certificate. A stale certificate remains provenance and may accompany a physically confirmed
Goal regression or external action-event failure record, but no future handoff may reuse it.
`STALE` is sticky for the rest of Phase 0: ordinary covered/no-change transitions cannot return it to
`CURRENT`. Only a later-phase explicit fresh rebuild with a new graph version may create
`RECONCILED/CURRENT`.

Implement `VersionedActionEventTracker`, not only the Protocol. A registered
`TransitionFeatureReader(ShadowStepContext) -> tuple[ActionTransitionFeatures, ...]` returns
exactly one record for every registered rule on every callback, including exact
rule/object/source/destination identity, gripper
position, contact count, holding truth, registered-region truth/distance, EEF/object
motion correlation and a canonical raw-transition hash on every callback. The LIBERO implementation
uses versioned registered observation/object-state keys; a missing key produces UNKNOWN/no strong
evidence, never a guessed event.
The transition hash covers only those registered feature values plus policy step/rule version and
excludes pixels/unregistered observation keys.
The reader exposes `monitor_contract_sha256`, `tracker_version`, and the exact ordered rule IDs; the
tracker verifies all three at construction and verifies the repeated contract/version fields on
every feature record. The reader reports raw `gripper_qpos`; `VersionedActionEventTracker` retains the previous registered
sample and applies the contract's open/close thresholds itself to derive transitions. It similarly
checks finite registered-region distances against `region_distance_max`; a pre-labeled transition or
region name cannot bypass the frozen thresholds. Step 0 only initializes the previous feature state
and cannot start an attempt because `last_action=None`; later steps require consecutive feature
samples, so a missing/error sample breaks transition continuity rather than bridging across a gap.
Missing numeric keys are canonical JSON null (`None`), never 0, NaN or a negative sentinel. A null,
non-finite numeric value, UNKNOWN required truth, missing/duplicate rule record, or reader exception
clears continuity for the affected rule and cannot start/advance an attempt or emit strong evidence.
The transition hash domain-separately encodes null and covers the contract hash/tracker version.
Implement `build_libero_transition_feature_reader(env, binding, monitor_contract)` in the LIBERO
adapter. It freezes the registered object/state accessors and rule IDs at construction, exposes the
three version properties above, and uses observation values plus named read-only simulator getters;
the returned reader retains only a narrowed `ReadOnlyLiberoStateView` of those bound getters, not the
environment object, and has no `env.step`, action-dispatch, reset or state-mutation capability. An adapter integration fake
counts all calls and proves step/reset/set_state are never invoked, step 0 only seeds previous state,
and a missing object/observation key returns the required null/UNKNOWN record for that rule.

For each `ActionEventRule`, use this deterministic state machine:

```text
IDLE
  -- grasp: CLOSE transition ∧ contact>=min ∧ corr>=min ∧ source TRUE --> ATTEMPT_ACTIVE
  -- place: prior holding TRUE ∧ OPEN transition ∧ destination proximity TRUE --> ATTEMPT_ACTIVE

ATTEMPT_ACTIVE creates two bounded ledgers:

TIMEOUT_STATUS
  -- attempted effect TRUE before/on due --> EFFECT_SATISFIED (cancel timeout only)
  -- policy_step < effect_due_policy_step --> no timeout evidence
  -- policy_step >= effect_due_policy_step ∧ attempted effect FALSE
       --> emit ATTEMPTED_EFFECT_TIMEOUT once
  -- attempted effect UNKNOWN at due --> INCONCLUSIVE (never strong timeout)

RECENT_MANIPULATION (independent of TIMEOUT_STATUS)
  -- retain through start_policy_step + manipulation_attribution_ttl_policy_steps
  -- same-object registered abnormal region within window --> emit ABNORMAL_TRANSFER once
       with effect_due_policy_step := emitted_policy_step
  -- other object, unregistered region, missing continuity, or expired window --> no evidence

EMITTED evidence is joinable only when
  effect_due_policy_step <= current_step <= evidence_expires_policy_step
then expires deterministically and can never be revived by a later graph version.
```

Effect success terminates only timeout monitoring; it does not erase recent manipulation provenance,
so “successful holding, then same-object drop” remains attributable. A registered holding/release
transition is added to supporting provenance but does not refresh the absolute attribution expiry.
The tracker rejects a feature record whose rule/object/source/destination tuple differs from its
frozen `ActionEventRule`, or duplicate records for one rule and step. `attempt_id` hashes rule ID,
object, start step and start transition hash. Due equals
`start_policy_step + effect_due_after_policy_steps`; expiry equals
`emitted_policy_step + evidence_ttl_policy_steps` for timeout evidence. For immediately observable
`ABNORMAL_TRANSFER_AFTER_MANIPULATION` and `GOAL_REGRESSION`, set the due step equal to the emission
step; the common active-window predicate therefore cannot delay or backdate them. A due-time UNKNOWN
closes that timeout attempt as inconclusive; a later FALSE cannot retroactively manufacture timeout
evidence. Supporting transition hashes include attempt start,
effect checks and emission transition. The tracker verifies monotonic policy steps and exact
object/effect/source/destination rule identity. `max_active_attempts_per_object` limits current active
attempts; `max_attempt_records_per_episode` and `max_evidence_records_per_episode` are cumulative
accepted-record hard caps. Expiry never refunds cumulative capacity. Overflow is diagnostic only and
cannot create evidence.
Each emitted record has `evidence_id = hash(attempt_id, evidence_kind, emitted_policy_step,
supporting_transition_hashes)`. One attempt may therefore produce distinct timeout and later
abnormal-transfer evidence, but at most one record per `(attempt_id, evidence_kind)`; duplicate
`evidence_id` is always corruption. Goal/progress fact events use their reserved rule IDs.

`GOAL_REGRESSION` is produced by the fact observer, not synthesized as a manipulation attempt. When
a signed Goal literal is first reliably satisfied, store its fact-evidence hash and policy step. On
the first later reliably opposite snapshot, emit one typed record with
`rule_id=ReservedFactEventRuleId.GOAL_REGRESSION.value` whose object/effect come from that
literal, whose `attempt_id` domain-separately hashes the Goal literal plus achieved step/hash, whose
start step is the achieved step, whose due and emission step are the regression step, whose expiry is
emission plus `goal_regression_evidence_ttl_policy_steps`, and whose supporting hashes are the
achieved/regressed fact-evidence hashes. It uses the same detector/contract hash, cumulative evidence
cap, active-window check and monotonic event-record state as tracker evidence. UNKNOWN never emits or
extends it. `PROGRESS_TIMEOUT` uses `ReservedFactEventRuleId.PROGRESS_TIMEOUT.value`, due equal to start plus
`progress_window_observations * monitor_interval_steps`, and its own frozen evidence TTL; it remains weak.
The observer inserts both kinds through `ActionEventTracker.record_fact_event`; that method accepts
only these two fact-event kinds, revalidates their fields/hashes, and returns false on duplicate or
capacity rejection. Thus manipulation and fact events share one cumulative evidence counter and one
overflow metric rather than each silently receiving the full cap.

`StableRecoveryObserver` samples only when `policy_step % interval_steps == 0`. On each successful
snapshot it:

1. Requires all four fact-universe audit fields, the same frozen universe version/hash as the live
   initial certificate, and registration of every relevant/contract fact; drift/missing coverage is
   a snapshot error that resets the streak and produces no evidence.
2. Adds currently true positive goal facts and currently false negative goal facts to an internal achieved-goal set.
3. Builds `UNPLANNED_SUPPORT_STABLE` only for a true `at(object, abnormal_surface)` fact that is in
   neither the initial state nor the signed Goal and is outside holding/release/settling grace.
4. Emits `COMPLETED_GOAL_REGRESSION_STABLE` only when a previously achieved signed-goal literal is
   now reliably opposite.
5. Joins versioned `ActionEventEvidence` by exact object/effect/source/destination only after
   `effect_due_policy_step` and through `evidence_expires_policy_step`. Strong kinds are `GOAL_REGRESSION`,
   `ATTEMPTED_EFFECT_TIMEOUT`, and `ABNORMAL_TRANSFER_AFTER_MANIPULATION`; `PROGRESS_TIMEOUT` is
   weak and cannot confirm by itself. `TRANSIENT_HOLDING`, `TRANSIENT_RELEASE`, and
   `NORMAL_PHASE` suppress the candidate and reset the streak.
6. Emits a stable anomaly after three identical samples. It upgrades that record to
   `CONFIRMED_DEVIATION` only when strong historical evidence exists; capability and budget are
   intentionally absent from this Phase 0 detector.
7. Sorts candidate facts and evidence tokens into immutable deterministic signatures.

Define the public records exactly as:

```python
@dataclass(frozen=True)
class ShadowPlanContext:
    problem: TaskProblem
    plan: tuple[GroundAction, ...]
    graph: CausalGraph
    certificate_hash: str


@dataclass(frozen=True)
class MonitorEvidenceContract:
    contract_id: str
    task_id: int
    object_ids: tuple[str, ...]
    nominal_source_facts: tuple[str, ...]
    abnormal_support_surfaces: tuple[str, ...]
    task_relevant_effects: tuple[str, ...]
    tracker_version: str
    action_event_rules: tuple[ActionEventRule, ...]
    monitor_interval_steps: int
    confirmation_count: int
    settling_grace_observations: int
    progress_window_observations: int
    progress_evidence_ttl_policy_steps: int
    goal_regression_evidence_ttl_policy_steps: int
    max_active_attempts_per_object: int
    max_attempt_records_per_episode: int
    max_evidence_records_per_episode: int
    grounding_rule_sha256: str
    event_detector_sha256: str
    contract_sha256: str


class ReservedFactEventRuleId(str, Enum):
    GOAL_REGRESSION = "__goal_regression__"
    PROGRESS_TIMEOUT = "__progress_timeout__"


@dataclass(frozen=True)
class ActionEventRule:
    rule_id: str
    object_id: str
    attempt_kind: str
    attempted_effect: str
    source_region: str | None
    destination_region: str | None
    gripper_close_threshold: float
    gripper_open_threshold: float
    contact_min_count: int
    motion_correlation_min: float
    region_distance_max: float
    effect_due_after_policy_steps: int
    evidence_ttl_policy_steps: int
    manipulation_attribution_ttl_policy_steps: int


@dataclass(frozen=True)
class ActionTransitionFeatures:
    policy_step: int
    monitor_contract_sha256: str
    tracker_version: str
    rule_id: str
    object_id: str
    source_region: str | None
    destination_region: str | None
    gripper_qpos: float | None
    contact_count: int | None
    holding: TruthValue
    source_region_truth: TruthValue
    destination_region_truth: TruthValue
    abnormal_region_truth: TruthValue
    source_region_distance: float | None
    destination_region_distance: float | None
    abnormal_region_id: str | None
    abnormal_region_distance: float | None
    object_eef_motion_correlation: float | None
    transition_sha256: str


@dataclass(frozen=True)
class ActionEventEvidence:
    evidence_id: str
    evidence_kind: str
    rule_id: str
    object_id: str
    attempted_effect: str
    source_region: str | None
    destination_region: str | None
    attempt_id: str
    start_policy_step: int
    effect_due_policy_step: int
    emitted_policy_step: int
    evidence_expires_policy_step: int
    supporting_transition_hashes: tuple[str, ...]
    detector_sha256: str


class ActionEventTracker(Protocol):
    @property
    def overflow_count(self) -> int: ...

    def observe(self, context: ShadowStepContext) -> None: ...
    def record_fact_event(self, evidence: ActionEventEvidence) -> bool: ...
    def active_for(
        self,
        *,
        object_id: str,
        attempted_effect: str,
        source_region: str | None,
        destination_region: str | None,
        policy_step: int,
    ) -> tuple[ActionEventEvidence, ...]: ...


class VersionedActionEventTracker:
    @property
    def attempt_record_count(self) -> int: ...

    @property
    def evidence_record_count(self) -> int: ...

    def __init__(
        self,
        monitor_contract: MonitorEvidenceContract,
        transition_feature_reader: TransitionFeatureReader,
    ) -> None: ...


class TransitionFeatureReader(Protocol):
    monitor_contract_sha256: str
    tracker_version: str
    rule_ids: tuple[str, ...]

    def __call__(
        self, context: ShadowStepContext
    ) -> tuple[ActionTransitionFeatures, ...]: ...


class DeviationStatus(str, Enum):
    ANOMALY_CANDIDATE = "ANOMALY_CANDIDATE"
    CONFIRMED_DEVIATION = "CONFIRMED_DEVIATION"


class CertificateState(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    RECONCILED = "RECONCILED"


@dataclass(frozen=True)
class ShadowTrigger:
    trigger_class: str
    deviation_status: DeviationStatus
    signature: tuple[str, ...]
    historical_failure_evidence: tuple[ActionEventEvidence, ...]
    deviation_event_id: str
    event_origin_parent_sha256: str
    policy_step: int
    observation_generation: int
    relevant_fact_sha256: str
    source_graph_version: str
    certificate_state: CertificateState
    snapshot: FactSnapshot
    observation: Mapping[str, Any]


@dataclass
class ShadowMonitorMetrics:
    snapshot_calls: int = 0
    snapshot_errors: int = 0
    event_tracker_errors: int = 0
    evidence_overflows: int = 0
    trigger_callback_errors: int = 0
    anomaly_candidates: int = 0
    confirmed_deviations: int = 0
    stale_certificates: int = 0
```

`ShadowPlanContext.__post_init__` requires `graph.certificate_hash == certificate_hash`,
`len(plan) == len(graph.canonical_agenda)`, and exact action/agenda correspondence; reject mismatched
projections rather than monitoring against mixed plan artifacts.

`shadow_monitor.py` must not import `evaluation.py`; doing so would create
`evaluation → shadow_runtime → shadow_monitor → evaluation`. `ShadowRuntime` constructs the immutable
`ShadowPlanContext` from its `CertifiedEpisode`, and the monitor depends only on model/dag types.

`MonitorEvidenceContract` is loaded from `configs/logiv/r2m-monitor-evidence-v1.json`; canonical JSON
with the `contract_sha256` field omitted must match `contract_sha256`. Phase 0 defaults are monitor
interval 5 Base steps, per-rule effect due after 15 Base steps, manipulation-attribution TTL 30 Base
steps, per-rule evidence TTL 20 Base steps, settling grace 2 monitor observations, confirmation count
3, progress window 4 observations, progress/Goal-regression evidence TTL 20 Base steps, at most 8 active
attempts per object, and at most 128 evidence records per episode.
The cumulative attempt-record cap is also 128. Attempt/evidence caps count all accepted records over
the episode and are not refunded after resolution/expiry. When a capacity is reached, reject the new record and increment an overflow metric; never
evict an unexpired record in a way that can manufacture a confirmation. IDs/facts/surfaces/effects
must be subsets of the accepted proposal/coverage
manifest, all windows must be positive, and progress timeout remains weak evidence.
Rule IDs and object/effect/source/destination tuples must be unique; thresholds must be finite and
ordered, counts/due/TTL positive, and every rule ID/object/region/effect covered by the accepted
proposal and coverage manifest. The self-hash covers tracker version and the complete canonical rule
records; changing a threshold or TTL necessarily changes `monitor_contract_sha256`.
`ActionEventRule.rule_id` may not use the reserved `__...__` namespace. Loader, observer and tests
import `ReservedFactEventRuleId`; no component may repeat the literal strings independently.

`StableRecoveryObserver.__init__` is keyword-only and accepts
`plan_context: ShadowPlanContext`, `monitor_contract: MonitorEvidenceContract`,
`snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot]`,
`action_event_tracker: ActionEventTracker`, `on_trigger: Callable[[ShadowTrigger], None]`,
`interval_steps: int`, and `confirmation_count: int`. Its call signature is
`(context: ShadowStepContext) -> None`.
Require `interval_steps == monitor_contract.monitor_interval_steps`; the runtime cannot override the
frozen sampling cadence independently of per-rule due/TTL calibration. Likewise require
`confirmation_count == monitor_contract.confirmation_count`; the detector threshold is part of the
self-hashed contract, not an untracked CLI degree of freedom.

`ActionEventTracker.observe(context)` runs on every callback, including the four actions between fact
samples; the stable-fact snapshot reader runs only on step 0 and interval boundaries. The tracker
maintains a bounded per-object/attempt event ledger so strong evidence
that precedes the third stable sample remains available through its pre-registered expiry; evidence
does not have to occur on the confirmation frame itself. The observer resets its streak on an
empty/different signature, normal/transient evidence, UNKNOWN, or reader error. It emits after exactly
`confirmation_count` identical sampled signatures and stores a monotonic per-event record state:
`UNSEEN → CANDIDATE_RECORDED → CONFIRMED_RECORDED`. The same status is never emitted twice after a
graph/certificate hash change, but later strong evidence may upgrade one previously recorded candidate
to one confirmed-deviation record with the same semantic event ID; downgrades are forbidden.
If strong evidence is already present on the first stable emission, emit only the confirmed record,
not a candidate and confirmed duplicate from the same snapshot.
`deviation_event_id` hashes episode-local trigger class, object/region facts, first-observed step, and
protected-goal signature; it excludes graph/certificate version. `on_trigger` exceptions increment
`trigger_callback_errors` and remain inside the observer; tracker failures increment
`event_tracker_errors`, clear evidence from that step, and remain fail-open. Copy NumPy values before placing the observation
into `ShadowTrigger`.

At the first sampled observation of a new candidate signature, hash a domain-separated
`LOGIV_EVENT_ORIGIN_V1` payload containing task ID, trigger class, object/effect/registered-region
signature, protected-goal signature, and sorted signed task-relevant facts, and latch it as
`event_origin_parent_sha256`. Explicitly exclude epoch/generation, `evidence_hash`, observation/pixel
hash, camera fields and continuous/micro-pose values. Candidate and later confirmed triggers
for the same semantic event must carry that identical hash even though their capture physics states
and `policy_step` differ. Clearing a streak without an emitted event may discard the provisional
origin; once a candidate/confirmed record is emitted, its origin is immutable.
Add invariance tests showing that epoch, camera noise and within-bin micro-pose changes preserve the
event-origin/group ID, while an object/effect/semantic-region/protected-goal change does not.

- [ ] **Step 4: Run monitor tests**

Run: `uv run pytest -q tests/logiv/test_shadow_monitor.py`

Expected: all monitor tests pass, including static-candidate, normal-intermediate, weak-evidence,
stale-certificate, noise-projection, and one-shot cases.

- [ ] **Step 5: Commit**

```bash
git add src/pi05_libero_repro/logiv/shadow_monitor.py src/pi05_libero_repro/logiv/libero_adapter.py tests/logiv/test_shadow_monitor.py tests/logiv/test_libero_adapter.py configs/logiv/r2m-monitor-evidence-v1.json
git commit -m "feat(logiv): detect evidenced deviations in shadow"
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
assert record.initial_proposal_status == "ACCEPTED"
assert record.initial_proposal_reason_code is None
assert record.shadow_vlm_requests == 0
assert record.recovery_policy_requests == 0
```

Add table-driven record tests for `BASE/NOT_APPLICABLE/0`, `SHADOW/ACCEPTED/1`,
`SHADOW/REJECTED/1`, and `SHADOW/NOT_ATTEMPTED/0`, including every invalid status/request/reason
cross-product. `validate_episode_records` must not need to open sidecar artifacts to enforce this.

- [ ] **Step 2: Run the tests and verify the unknown-arm failures**

Run: `uv run pytest -q tests/logiv/test_evaluator.py tests/logiv/test_records.py`

Expected: FAIL because `SHADOW_LOGIV` is not a valid `MethodArm` or record arm.

- [ ] **Step 3: Register the arm and schema-3 buckets**

Add `SHADOW_LOGIV = "SHADOW_LOGIV"` after `BASE` and add it to `METHOD_ARMS` and the shell launcher case. Append defaulted fields to `LogivEpisodeRecord`:

```python
    base_policy_requests: int = 0
    initial_proposal_requests: int = 0
    initial_proposal_status: str = "NOT_APPLICABLE"
    initial_proposal_reason_code: str | None = None
    shadow_vlm_requests: int = 0
    recovery_policy_requests: int = 0
    shadow_monitor_calls: int = 0
    shadow_monitor_errors: int = 0
    shadow_monitor_seconds: float = 0.0
    shadow_parity_valid: bool = True
```

For schema version 3, `validate_episode_records` must require
`inference_requests == base_policy_requests`, require `initial_proposal_requests == 0` for `BASE`,
`initial_proposal_status=NOT_APPLICABLE`, and no reason. Require `SHADOW_LOGIV` to have exactly one
request for `ACCEPTED/REJECTED` or zero only for an explicit `NOT_ATTEMPTED` step-0 containment
failure; `REJECTED/NOT_ATTEMPTED` require a stable reason code and `ACCEPTED` forbids one. Reject
unknown statuses, values above one, or nonzero
`shadow_vlm_requests` or `recovery_policy_requests` in Phase 0 records. Add mean values for all seven
numeric fields and the parity-valid rate to the report's operational metrics.
Require `shadow_parity_valid=True` for `BASE`; a false Shadow value preserves the native rollout row
but excludes it from parity/nominal non-destruction evidence and fails the Phase 0 acceptance gate.

- [ ] **Step 4: Implement `ShadowRuntime` with lazy grounding**

Define the cross-file records before the function implementation:

```python
class CertifiedEpisodeLike(Protocol):
    problem: TaskProblem
    plan: tuple[GroundAction, ...]
    graph: CausalGraph
    certificate: PlanCertificate


@dataclass(frozen=True)
class ShadowValidatedProposal:
    certified_episode: CertifiedEpisodeLike
    snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot]


@dataclass(frozen=True)
class ShadowEpisodeContext:
    task_id: int
    episode_idx: int
    initial_epoch_id: int
    scene_sha256: str
    object_instance_ids: tuple[str, ...]
    initial_state_sha256: str
    parent_trajectory_lineage_sha256: str
    base_prompt_sha256: str
    base_checkpoint_sha256: str
    policy_client_config_sha256: str
    policy_replay_contract_sha256: str | None
    master_seed: int
    policy_seed: int
    simulator_seed: int
    replan_steps: int
    collect_recovery_roots: bool
    collection_label: CollectionLabel
    root_output_dir: Path
    simulator_state_reader: Callable[[], np.ndarray]
    transition_feature_reader: TransitionFeatureReader


@dataclass
class ShadowRuntimeCounters:
    root_count: int = 0
    root_write_errors: int = 0
    proposal_callback_errors: int = 0
    provenance_errors: int = 0


@dataclass
class ShadowRuntime:
    initial_proposal: InitialProposalResult[ShadowValidatedProposal] | None
    observer: Callable[[ShadowStepContext], None] | None
    monitor: StableRecoveryObserver | None
    counters: ShadowRuntimeCounters
```

`build_shadow_runtime` is keyword-only and accepts `provider: Any`,
`provider_name: str`, `episode_context: ShadowEpisodeContext`, `goal_mode: GoalMode`, `live_validator:
Callable[[ProposalPackage, Mapping[str, Any]], ShadowValidatedProposal]`,
`monitor_contract: MonitorEvidenceContract`, `root_collector:
Callable[[ShadowTrigger, ShadowStepContext, ShadowEpisodeContext], RecoveryRootArtifacts | None]`,
`interval_steps: int`, and `confirmation_count: int`; it returns
`ShadowRuntime`. This is the final interface used by evaluator code and tests.

`build_shadow_runtime` does not call the provider during construction. Its returned observer requires
the first invocation to have `policy_step=0`, `last_action=None`, no pending actions, request count 0,
active request `None`, next request index 0, and the settled live initial
observation, then calls
`run_initial_proposal` exactly once with `epoch_id=episode_context.initial_epoch_id`. Phase 0 requires
`initial_epoch_id=0`; the live step-0 `FactSnapshot.epoch_id`, proposal package epoch and resulting
certificate source epoch must all equal it or certification is rejected.
`live_validator` must verify exact registered-object equality, construct a local observation
store/grounder from the live step-0 observation, obtain fresh Ground Facts, and call
`certify_initial_package` against that live `FactSnapshot`—never against
`package.proposal.initial_snapshot` as self-evidence. It uses
`binding.supported_action_schemas | binding.recovery_schemas`, configured repair bounds, retry
policy, macro decomposition set, and VAL timeout, and returns the certified episode plus an audited
snapshot reader bound to the same grounding session.

`CertifiedEpisodeLike` is a structural typing boundary. `shadow_runtime.py` must not import
`evaluation.CertifiedEpisode`, because `evaluation.py` imports the runtime integration; the concrete
result of `certify_initial_package` satisfies the protocol without a runtime dependency cycle.

Before step 0, `initial_proposal=None` and the callback is installed but powerless. Proposal or live
certification rejection stores the rejected result and makes later action-step callbacks no-ops.
Acceptance stores the result, projects the full certified validation into `ShadowPlanContext`, and
constructs `StableRecoveryObserver` with the returned snapshot reader and versioned
`VersionedActionEventTracker(monitor_contract,
episode_context.transition_feature_reader)`. Before returning from that same contained callback, the
runtime invokes the newly constructed monitor once with the step-0 context. This seeds achieved Goal
facts and the tracker's previous raw feature state; because `last_action=None`, step 0 cannot create
an action attempt. Later callbacks update/read that same grounding session.
The feature reader is evaluator-owned, read-only and version-bound; it extracts registered
gripper/contact/holding/object-region features from copied observation plus read-only simulator
queries and never executes an action. This binds the initial certificate
to the actual settled episode observation without a module import cycle.

The protocol, not the runtime, computes the Base-action-prefix digest and pending action suffix; the
runtime independently extends its previous digest with the copied `last_action`, checks exact equality
with the supplied digest whenever `policy_step` increases, and passes the values
unchanged to the evaluator-owned collector. On either stable candidate or confirmed deviation the
collector uses only `ShadowEpisodeContext.simulator_state_reader` to capture physics and builds a
manifest with:

- immutable task/episode/scene/object/initial-state and parent-trajectory-lineage IDs;
- master, policy, and simulator seeds;
- `perturbation_family="observed_nominal_failure"`, with no branch seed;
- trigger class, deviation status/event ID, historical evidence, relevant-fact hash, source graph
  version, observation generation, certificate state, grounding/event-detector hashes and policy step;
- exact canonical monitor-contract JSON/self-hash whose concrete action-event rules produced the evidence;
- exact Base request count, active/next zero-based request indices and their protocol-supplied
  canonical envelope JSON/hashes, plus pending chunk bytes/response-size/logical-size/count/offset,
  frozen replay-contract hash, current fact snapshot, and current Base action-prefix hash.

A step gap, duplicate step with different content, digest mismatch, or request-index regression
disables monitoring/root capture for the rest of the episode and records a provenance error; it never
changes or aborts Base.

The evaluator computes `parent_trajectory_lineage_sha256` before rollout from task/scene/initial-state
hashes, frozen Base checkpoint/prompt/policy-client-config hashes, and the episode master-seed
envelope. It is
constant across the episode and all derived perturbation/branch roots; capture step, graph version,
perturbation family and branch seed are excluded.

The collector copies the monitor-latched `event_origin_parent_sha256` and rejects any later trigger
that changes it for the same `deviation_event_id`. It computes
`independence_unit_id` from the parent trajectory lineage, never from the capture frame. If either
active/next protocol-supplied replay envelope or the held-out replay-contract hash is unavailable,
collection may continue but the manifest is explicitly
ineligible for live-Base continuation; the code must not substitute
`ceil(policy_step / replan_steps)` for any actual request count/index.

It writes only when `collect_recovery_roots` is true. Candidate and confirmed roots remain separately
labeled; downstream training cannot silently consume candidates. A root collector/write failure
increments only `root_write_errors` and cannot escape into the nominal rollout; it does not increment
`snapshot_errors` or `trigger_callback_errors`. Derive abnormal support surfaces only
from the validated monitor contract; reject contract object/surface/effect IDs not covered by the
accepted proposal/binding rather than deriving recovery surfaces from every nominal
`support_surface_alias`.

- [ ] **Step 5: Test proposal rejection and root-write failure containment**

Use an injected fake provider plus a minimal object satisfying `CertifiedEpisodeLike`, live validator,
snapshot/event tracker, complete `ShadowEpisodeContext`, and root writer in
`tests/logiv/test_shadow_runtime.py`. Assert the provider
has zero calls before the step-0 callback and exactly one afterward; assert live-validator failure
stores a rejected result and leaves later callbacks as no-ops; and
assert a root writer raising `OSError("disk")` leaves the callback callable while incrementing the
root-write error counter but not snapshot/event/trigger counters. Assert the collector receives the
actual request count, active/next indices, pending chunk/offset, replay envelope, action-prefix digest,
lineage and seeds;
candidate→confirmed records must keep one event-origin parent and independence unit. No test may
import LIBERO.

Test both the accepted `initial_epoch_id=0` path and a proposal/live-snapshot epoch mismatch; the
latter must yield one fail-open rejection, disable monitoring/root capture, and preserve Base parity.
On acceptance, assert that the monitor receives the same step-0 context exactly once, initializes
the achieved-goal/previous-feature state, and emits neither an action attempt nor a trigger at step 0.

Add an integration-level fake `EpisodeOutcome` with a step-0 `INPUT_COPY` failure and no proposal
result; artifact rendering must produce `NOT_ATTEMPTED`, zero proposal requests, and the exact stable
reason. Add one-failure-per-bucket plus combined-counter tests for the ownership table.

Run: `uv run pytest -q tests/logiv/test_shadow_runtime.py`

Expected: all shadow runtime tests pass.

- [ ] **Step 6: Route Base and Shadow through one evaluator branch**

Change the branch condition to:

```python
arm = MethodArm(args.method_arm)
if arm in {MethodArm.BASE, MethodArm.SHADOW_LOGIV}:
```

Construct `ShadowRuntime` only for `SHADOW_LOGIV`; pass `runtime.observer` to `run_episode`. Do not change `str(task.language)`, `initial_state`, `episode_client`, `max_steps`, `wait_steps`, `replan_steps`, or `settling_steps` between the arms. Both arms must build the same Base `ControllerResult` and native terminal evaluation.

The initial proposal and live VAL certification run only in the step-0 shadow callback after the
shared reset/set-init/wait sequence and before the first Base policy request. Task 2's callback helper
restores Python/NumPy RNG state even when provider/validator code consumes randomness or throws;
shadow code never reseeds or resets the environment. Add a unit test whose proposal provider consumes
`random` and `np.random`, then assert Base and Shadow policy inputs, post-callback draws, executed
actions, and request counts are identical.

Write `base_execution.json` for both arms with `steps`, `base_policy_requests`, `done_signal`,
`post_settling_success`, `initial_state_sha256`, `base_prompt_sha256`, `base_checkpoint_sha256`,
`policy_client_config_sha256`, `request_envelope_log_sha256`, and a SHA-256 from the shared
`BaseActionPrefixHasher` over every executed action. For Shadow also write:

```json
{
  "initial_proposal.json": ["status", "provider", "request_count", "elapsed_seconds", "reason", "certificate_hash", "graph_hash"],
  "shadow_monitor.json": ["callback_calls", "callback_errors", "callback_seconds", "shadow_parity_valid", "failure_records", "proposal_callback_errors", "provenance_errors", "snapshot_calls", "snapshot_errors", "event_tracker_errors", "evidence_overflows", "trigger_callback_errors", "root_write_errors", "anomaly_candidates", "confirmed_deviations", "stale_certificates", "root_count"],
  "compute_accounting.json": ["base_policy_requests", "initial_proposal_requests", "shadow_vlm_requests", "recovery_policy_requests", "initial_proposal_seconds", "shadow_monitor_seconds"]
}
```

The notation above names required JSON fields; each filename is a separate object. Set
`initial_proposal_requests` from the proposal result, or zero with `status=NOT_ATTEMPTED` and the
step-0 containment reason when no result exists; set `shadow_vlm_requests=0`,
`recovery_policy_requests=0`, and `base_policy_requests=outcome.inference_requests`. Proposal
rejection/non-attempt must keep `valid=True` unless the Base rollout itself fails.
When no proposal result exists, select the earliest step-0 `ShadowFailureRecord` by append order as
the stable `NOT_ATTEMPTED` reason and persist all records in `shadow_monitor.json`. If a proposal
result exists, a later `CLOCK_END`/restore failure does not rewrite its accepted/rejected status.
Write the same audit status into `LogivEpisodeRecord.initial_proposal_status`; use exception type for
`REJECTED` reason code and `stage:exception_type` for `NOT_ATTEMPTED`. The full diagnostic reason may
remain only in `initial_proposal.json`; record validation is self-contained.
`shadow_monitor_errors` in the episode record is the documented aggregate of protocol callback-input,
proposal-callback, provenance, snapshot, event-tracker, evidence-overflow, trigger-callback, and
root-write errors; the
component counters above remain authoritative and must sum to the aggregate without double counting.

Freeze the following ownership table in code/tests:

| Failure source | Sole owner/bucket |
|---|---|
| Task 2 clock/RNG/hash/copy/envelope preparation | one `ShadowFailureRecord` per caught exception; protocol-input bucket |
| An exception escaping the entire runtime observer | Task 2 `OBSERVER_ESCAPE` only; runtime must not also count it |
| Provider/validator returns `REJECTED` | proposal status/reason only, not an error bucket |
| Unexpected runtime setup exception contained before escape | `proposal_callback_errors` only |
| Step/order/digest/request-envelope mismatch | `provenance_errors` only |
| Grounding/snapshot reader | `snapshot_errors` only |
| Action event tracker / bounded ledger rejection | `event_tracker_errors` / `evidence_overflows` only |
| Non-root trigger consumer | `trigger_callback_errors` only |
| Evaluator root collector/state read/persistence | `root_write_errors` only |

The runtime wraps root collection before passing it as `on_trigger`, so a root error never reaches the
monitor's trigger-callback counter. Unit tests inject one failure at a time and one combined episode,
then assert the component sum equals `shadow_monitor_errors` exactly.

- [ ] **Step 7: Add CLI/config fields**

Add:

```python
parser.add_argument("--collect-recovery-roots", action="store_true")
parser.add_argument("--recovery-root-split", choices=("TRAIN", "DEV", "HELDOUT"), default="DEV")
parser.add_argument("--shadow-monitor-interval-steps", default=5, type=int)
parser.add_argument("--shadow-confirmations", default=3, type=int)
parser.add_argument(
    "--shadow-monitor-contract",
    default="configs/logiv/r2m-monitor-evidence-v1.json",
)
```

Persist all five values plus the loaded contract SHA-256 in `run.json`. In Phase 0, reject collection unless
`--development-only` is set, `--recovery-root-split DEV` is selected, and the method
arm is `SHADOW_LOGIV`; the other enum values exist so imported/system-generated
datasets can be validated without relabeling their manifests, not so the evaluator
can create a held-out claim from the inspected 50 states. Reject startup if the interval or
confirmation CLI values differ from the corresponding self-hashed monitor-contract fields; the CLI
cannot silently override detector semantics.

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
jq -S '{steps,base_policy_requests,done_signal,post_settling_success,initial_state_sha256,base_prompt_sha256,base_checkpoint_sha256,policy_client_config_sha256,request_envelope_log_sha256,actions_sha256}' \
  runs/r2m-phase0-base-task8-seed0/artifacts/task_08/episode_000/base_execution.json \
  > /tmp/r2m-base-execution.json
jq -S '{steps,base_policy_requests,done_signal,post_settling_success,initial_state_sha256,base_prompt_sha256,base_checkpoint_sha256,policy_client_config_sha256,request_envelope_log_sha256,actions_sha256}' \
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

Expected: exit 0 with JSON containing only `DEV` roots and separate counts for
`ANOMALY_CANDIDATE` and `CONFIRMED_DEVIATION`, plus unique recovery-group and independence-unit
counts. Inspect one loaded artifact to verify the canonical evidence records, pending Base suffix,
actual request count/active/next indices, replay-envelope/replay-contract status, event-origin parent and trajectory
lineage. If the
episode produces no stable record, record
`root_count=0`; the deterministic root round-trip and leakage tests remain the Phase 0 artifact
acceptance evidence.

- [ ] **Step 5: Write the smoke report**

Record the implementation commit, complete pytest count, both episode terminal results, all ten
parity fields, proposal/monitor request buckets and distinct error buckets, root count,
split-validator output, and the explicit statement: `Phase 0 collected development evidence only;
recovery capability and takeover are not enabled.` Also record that raw roots were not passed to a
trainer and that Task 4's manifest-builder tests reject candidate/stale-without-fresh-label inputs.

- [ ] **Step 6: Commit the verified report**

```bash
git add docs/experiments/2026-08-03-logiv-r2m-phase0-smoke.md
git commit -m "docs: record LOGIV R2M phase0 verification"
```

## Phase 0 Acceptance Gate

Phase 0 is complete only when all of the following are true:

- the full unit suite passes from the isolated worktree;
- Base and Shadow real smoke artifacts have identical initial/prompt/checkpoint/client/envelope-log
  hashes, action hash/count, Base policy request count, done signal, and native terminal result;
- initial proposal is certified from the shared settled live step-0 observation; rejection,
  `NOT_ATTEMPTED`, and local-provider timeout are proven fail-open by tests;
- hostile nested mutation, input-copy failure, RNG consumption, snapshot/event/trigger failures and
  root-writer failures are proven non-fatal without changing Base inputs/actions/requests;
- the paired smoke has `shadow_parity_valid=true`, zero RNG-restore failures, and exact protocol-owned
  active/next request envelopes; otherwise it cannot support the non-destruction claim;
- a static abnormal-surface fact without historical evidence is labeled only as an anomaly candidate;
  initial/normal/transient states do not emit a candidate, and progress timeout alone cannot become a
  confirmed deviation;
- Goal regression, attempted-effect timeout, and manipulation-backed abnormal transfer have explicit
  versioned structured evidence joined by exact object/effect/region/due/expiry fields; cross-object, expired,
  and unsampled-intervening-action cases are tested; capability and budget are absent from detection;
- feature records cover every registered rule/callback, encode missing numerics as canonical null,
  bind the monitor-contract/tracker version, and cannot bridge a missing/error sample; successful
  effect cancels timeout but retains bounded same-object manipulation attribution;
- task-relevant uncovered transitions stale the Initial DAG certificate while irrelevant observation
  changes do not; stale is sticky, remains diagnostic-only and Phase 0 has no permit/handoff type;
- exact simulator state, observation arrays and pending Base action suffix round-trip through NPZ;
  request count/active/next indices, offset, replay-envelope/replay-contract status, action-prefix,
  event origin and
  trajectory lineage
  survive loading;
- a final root is published by one directory rename and embeds the canonical monitor contract; its
  standalone loader rechecks rule IDs, evidence IDs, due/expiry/attribution equations, and ignores
  orphan temporary directories;
- every root/live certificate carries a complete versioned fact universe and exact canonical
  evidence payload; TRUE/FALSE/UNKNOWN are a disjoint exhaustive partition, and universe, payload,
  observation, epoch, or partition tampering is rejected;
- split validation rejects recovery-group, independence-unit, initial-state, parent-lineage,
  perturbation-seed, and state-fingerprint leakage;
- dataset summaries keep anomaly candidates and confirmed deviations separate; the trainer refuses raw
  root directories, and the manifest builder rejects candidates and stale roots without an exactly
  matching fresh label;
- every immutable source unit is allocated pre-outcome exactly once, and role builders reject an
  incomplete allocation, a repackaged historical unit, or a non-canonical/stale registry head;
- episode records self-validate Base/Shadow proposal status, request count, and reason-code
  cross-products without relying on sidecar JSON;
- all added requests and compute are reported outside `base_policy_requests`;
- no recovery action, recovery policy, capability claim, or online takeover exists in the Phase 0 code path.
