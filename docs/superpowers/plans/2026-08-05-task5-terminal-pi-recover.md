# Task 5 Terminal `pi_recover` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one independent Task 5 held-book recovery policy so `LOGIV_R2M` preserves every Base success and converts at least two of the three frozen Task 5 failures.

**Architecture:** Base and Shadow run unchanged through native terminal settling. A fail-closed terminal assessor obtains a fresh strict snapshot, certifies exactly one `place-held-in` option, and hands control once to a separately trained and separately seeded LoRA recovery service; a joint symbolic/native commit check decides success. Training data comes only from stable held-book suffixes of the 50 official Task 5 demonstrations, and checkpoint choice is made by loss validation before simulator capability testing.

**Tech Stack:** Python 3.11+, pytest, NumPy, HDF5/h5py, LIBERO/robosuite, LeRobot, OpenPI/JAX/Flax NNX/Optax, VAL, WebSocket policy service, Docker, JSON/SHA-256 manifests.

## Global Constraints

- Implement only LIBERO-10 Task 5 terminal recovery for `holding(black_book_1)` and not `at(black_book_1, desk_caddy_1_back_contain_region)`.
- Base runs normally and is never intercepted at step 0; Base-success episodes execute zero recovery actions and make zero recovery requests.
- Shadow uses the existing fixed graph and must be `CURRENT`; only native terminal failure may create `TERMINAL_GOAL_UNSATISFIED`.
- The only permitted physical option is `(place-held-in black_book_1 desk_caddy_1_back_contain_region desk_caddy_1_access)`.
- Permit only on a fresh strict, exactly-one-consistent snapshot with holding TRUE, target FALSE, active Task 5 place node, current certificate/hash provenance, signed-state success, VAL success, enabled capability, and sufficient budget.
- `option_action_cap = min(180, 520 - base_policy_steps)`; Base plus recovery model actions must never exceed 520.
- Use one handoff, one recovery macro, no retry, no fallback prompt, no concurrent Base/recovery execution, and no return to Base in this tranche.
- The recovery instruction is exactly `place the held book in the back compartment of the caddy`.
- Recovery seed derivation is exactly `uint32(sha256("LOGIV-recovery-policy-seed-v1:master:task:episode:event")[:4])`.
- Build splits before model evaluation using demonstration identity: 35 train, 5 loss validation, 10 simulator capability validation.
- Block training unless at least 46/50 suffixes are valid, including at least 32 train, 4 loss-validation, and all 10 capability demonstrations.
- Train one LoRA run from the frozen full `pi05_libero` expert: seed 42, batch 64, AdamW clip 1.0, peak LR `5e-5`, 4,000 optimizer steps, checkpoints 1,000/2,000/3,000/4,000, EMA disabled, external logging disabled.
- Both PaliGemma language/vision and action-expert LoRA adapters are trainable; all non-LoRA parameters remain frozen.
- Select the checkpoint by minimum loss-validation loss; ties select the earlier checkpoint. Simulator capability results cannot change that selection.
- Capability gate: at least 8/10 native successes and zero protected-invariant violations on the disjoint fixed roots.
- Frozen ten-case gate: at least 2/3 positive flips on `t05-r02`, `t05-r03`, `t05-r04`, zero negative flips, exact Base-prefix parity, and no combined-budget violation.
- Do not implement Task 8, dropped-book recovery, early takeover, Base re-entry, multiple recovery attempts, external memory/HiMe, or prompt search.
- Do not alter or report `LOGIV_REPAIR_OVERLAY` as `LOGIV_R2M`.
- Preserve all unrelated dirty-worktree files; stage only the exact files named by each task.

## File Map

**New source files**

- `src/pi05_libero_repro/logiv/task5_terminal_recovery.py` — versioned capability contract, terminal eligibility, recovery-plan certification, seed derivation, and commit verification.
- `src/pi05_libero_repro/logiv/task5_recovery_dataset.py` — stable suffix detector, deterministic demo split, dataset manifest types, and content hashes; no simulator or LeRobot imports at module import time.
- `src/pi05_libero_repro/logiv/task5_recovery_training.py` — frozen training specification and lazy OpenPI config construction.
- `configs/logiv/task5-terminal-pi-recover-v1.json` — the only capability/prompt/budget/invariant contract used by training, service, and evaluator.

**New executable files**

- `scripts/report_task5_terminal_preflight.py` — evaluate the three fresh strict terminal snapshots and emit a machine-readable go/no-go report.
- `scripts/build_task5_recovery_dataset.py` — replay the official HDF5 demonstrations and materialize train/loss-validation LeRobot repos plus ten simulator roots.
- `scripts/train_task5_pi_recover.py` — run exactly 4,000 optimizer steps and save the four registered checkpoints.
- `scripts/score_task5_pi_recover.py` — compute deterministic loss-validation means and freeze the selected checkpoint manifest.
- `scripts/eval_task5_recovery_capability.py` — execute only the preselected checkpoint from the ten held-state simulator roots.
- `scripts/report_task5_pi_recover_gate.py` — audit capability or frozen paired results without changing policy selection.

**Modified runtime files**

- `src/pi05_libero_repro/logiv/evaluation.py` — add the distinct `LOGIV_R2M` enum value.
- `src/pi05_libero_repro/logiv/records.py` — accept and validate R2M records and separate Base/recovery accounting.
- `src/pi05_libero_repro/protocol.py` — expose discarded Base suffix accounting and add a current-state recovery macro runner that never resets the environment.
- `src/pi05_libero_repro/logiv/shadow_runtime.py` — expose a fail-closed terminal strict-snapshot callback and project recovery observations into the same fixed graph.
- `scripts/eval_logiv_libero.py` — preflight capture, independent recovery client, terminal permit, one recovery macro, commit verification, and required artifacts.
- `scripts/serve_episode_seeded_policy.py` — lazily construct the recovery LoRA config and advertise recovery checkpoint/capability hashes without changing the Base-server default.
- `scripts/run_logiv_eval.sh` — allow `LOGIV_R2M` and pass a second service endpoint/manifests into the read-only evaluation container.
- `scripts/verify_artifacts.py` — validate R2M artifact hashes, action accounting, and zero-action denial paths.

**New tests**

- `tests/logiv/test_task5_terminal_recovery.py`
- `tests/logiv/test_task5_recovery_dataset.py`
- `tests/logiv/test_task5_recovery_training.py`
- `tests/test_recovery_protocol.py`
- `tests/logiv/test_task5_r2m_evaluator.py`
- `tests/logiv/test_task5_pi_recover_reports.py`

---

### Task 1: Freeze the capability contract and pure terminal decision boundary

**Files:**

- Create: `configs/logiv/task5-terminal-pi-recover-v1.json`
- Create: `src/pi05_libero_repro/logiv/task5_terminal_recovery.py`
- Create: `tests/logiv/test_task5_terminal_recovery.py`

**Interfaces:**

- Consumes: `Fact`, `FactSnapshot`, `GroundAction`, `TaskProblem`, `CausalGraph`, `PlanCertificate`, `ValWrapper`.
- Produces: `load_task5_recovery_capability(path: Path) -> Task5RecoveryCapability`, `assess_task5_terminal(capability: Task5RecoveryCapability, **terminal_inputs) -> TerminalAssessment`, `certify_task5_recovery(capability: Task5RecoveryCapability, **certification_inputs) -> RecoveryPermit`, `derive_recovery_policy_seed(master_seed: int, task_id: int, episode_idx: int, event_id: str) -> int`, and `verify_task5_recovery_commit(capability: Task5RecoveryCapability, **commit_inputs) -> RecoveryCommit`.

- [ ] **Step 1: Write the contract fixture and failing contract/seed tests**

Create the JSON with these exact semantic values. The recorded `capability_sha256` is the SHA-256 of canonical JSON after removing that field; Step 3 independently recomputes and verifies it.

```json
{
  "schema_version": 1,
  "capability_id": "task5-held-book-place-v1",
  "task_id": 5,
  "event_type": "TERMINAL_GOAL_UNSATISFIED",
  "prompt_version": "task5-held-book-v1",
  "prompt": "place the held book in the back compartment of the caddy",
  "holding_fact": "(holding black_book_1)",
  "target_fact": "(at black_book_1 desk_caddy_1_back_contain_region)",
  "action": "(place-held-in black_book_1 desk_caddy_1_back_contain_region desk_caddy_1_access)",
  "active_node_statuses": ["ACTIVE"],
  "protected_invariants": [
    "(accessible desk_caddy_1_back_contain_region desk_caddy_1_access)",
    "(open desk_caddy_1_access)"
  ],
  "effect_confirmation_observations": 3,
  "settling_steps": 10,
  "max_recovery_actions": 180,
  "max_combined_actions": 520,
  "capability_required_successes": 8,
  "capability_total": 10,
  "capability_sha256": "fe307bb9fd88f1d807464e41a0e17cb127d56a990b4a5179eea259e86ba2e651"
}
```

Add concrete tests:

```python
def test_contract_is_self_hashed_and_locked():
    capability = load_task5_recovery_capability(CONTRACT)
    assert capability.task_id == 5
    assert capability.prompt == "place the held book in the back compartment of the caddy"
    assert capability.max_recovery_actions == 180
    assert capability.max_combined_actions == 520
    assert capability.capability_sha256 == capability.recompute_sha256()


def test_recovery_seed_uses_independent_domain():
    first = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-1")
    second = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-2")
    assert 0 <= first < 2**32
    assert first != second
    payload = "LOGIV-recovery-policy-seed-v1:54804909:5:36:terminal-event-1"
    assert first == int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
```

- [ ] **Step 2: Run the tests and verify the new module is absent**

Run: `uv run pytest tests/logiv/test_task5_terminal_recovery.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: pi05_libero_repro.logiv.task5_terminal_recovery`.

- [ ] **Step 3: Implement strict parsing, canonical hashing, and seed derivation**

Use frozen dataclasses and reject unknown/missing JSON keys. The seed implementation is exactly:

```python
RECOVERY_SEED_DOMAIN = "LOGIV-recovery-policy-seed-v1"


def derive_recovery_policy_seed(
    master_seed: int, task_id: int, episode_idx: int, event_id: str
) -> int:
    if not (0 <= master_seed < 2**32 and 0 <= task_id <= 9 and 0 <= episode_idx < 50):
        raise ValueError("recovery seed inputs are outside the frozen evaluation domain")
    if not event_id:
        raise ValueError("event_id must be nonempty")
    payload = f"{RECOVERY_SEED_DOMAIN}:{master_seed}:{task_id}:{episode_idx}:{event_id}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
```

Compute and insert the contract self-hash with:

```bash
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path("configs/logiv/task5-terminal-pi-recover-v1.json")
d = json.loads(p.read_text())
d.pop("capability_sha256")
raw = json.dumps(d, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
print(hashlib.sha256(raw.encode()).hexdigest())
PY
```

Apply the printed digest through `apply_patch`; do not rewrite the JSON with a script.

- [ ] **Step 4: Add failing terminal eligibility tests for every denial boundary**

Use one eligible fixture and parameterize mutations:

```python
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"base_success": True}, "BASE_SUCCEEDED"),
        ({"task_id": 4}, "UNSUPPORTED_TASK"),
        ({"certificate_state": "STALE"}, "CERTIFICATE_NOT_CURRENT"),
        ({"place_node_status": "READY"}, "PLACE_NODE_NOT_ACTIVE"),
        ({"base_policy_steps": 520}, "ACTION_BUDGET_EXHAUSTED"),
    ],
)
def test_terminal_assessment_denies_without_side_effects(eligible_inputs, change, reason):
    result = assess_task5_terminal(**(eligible_inputs | change))
    assert result.eligible is False
    assert result.reason == reason
    assert result.option_action_cap == 0


def test_terminal_assessment_requires_explicit_signed_facts(eligible_inputs):
    snapshot = replace(eligible_inputs["snapshot"], false_facts=frozenset())
    result = assess_task5_terminal(**(eligible_inputs | {"snapshot": snapshot}))
    assert result.reason == "TARGET_NOT_EXPLICITLY_FALSE"
```

- [ ] **Step 5: Implement the pure terminal assessor**

The assessor must check in this order so denial reasons are stable: Base success, task, native failure, graph/certificate currentness and hashes, strict audited snapshot, exactly-one location, holding TRUE, target FALSE, active matching node, protected invariants, then budget. Return `option_action_cap=min(180, 520-base_policy_steps)` only when eligible. Never accept UNKNOWN as FALSE.

```python
@dataclass(frozen=True)
class TerminalAssessment:
    eligible: bool
    reason: str
    event_id: str | None
    event_type: str | None
    option_action_cap: int
    snapshot_sha256: str | None
    graph_hash: str | None
    certificate_hash: str | None
    monitor_contract_sha256: str | None
    protected_true_facts: frozenset[Fact]


def assess_task5_terminal(
    capability: Task5RecoveryCapability,
    *,
    task_id: int,
    episode_id: str,
    base_success: bool,
    native_terminal_status: str,
    base_policy_steps: int,
    snapshot: FactSnapshot | None,
    problem: TaskProblem | None,
    graph: CausalGraph | None,
    certificate: PlanCertificate | None,
    monitor_contract_sha256: str | None,
    certificate_state: str | None,
    place_node_action: GroundAction | None,
    place_node_status: str | None,
) -> TerminalAssessment:
    def denied(reason: str) -> TerminalAssessment:
        return TerminalAssessment(
            eligible=False,
            reason=reason,
            event_id=None,
            event_type=None,
            option_action_cap=0,
            snapshot_sha256=None if snapshot is None else snapshot.evidence_hash,
            graph_hash=None if graph is None else graph.graph_hash,
            certificate_hash=None if certificate is None else certificate.certificate_hash,
            monitor_contract_sha256=monitor_contract_sha256,
            protected_true_facts=frozenset(),
        )

    if base_success:
        return denied("BASE_SUCCEEDED")
    if task_id != capability.task_id:
        return denied("UNSUPPORTED_TASK")
    if native_terminal_status != "EPISODE_FAIL":
        return denied("NATIVE_TERMINAL_NOT_FAILED")
    if problem is None or graph is None or certificate is None:
        return denied("MISSING_GRAPH_OR_CERTIFICATE")
    if certificate_state != "CURRENT":
        return denied("CERTIFICATE_NOT_CURRENT")
    if graph.certificate_hash != certificate.certificate_hash:
        return denied("GRAPH_CERTIFICATE_HASH_MISMATCH")
    if (
        monitor_contract_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", monitor_contract_sha256) is None
    ):
        return denied("MONITOR_CONTRACT_HASH_INVALID")
    if snapshot is None or snapshot.fact_universe is None:
        return denied("STRICT_AUDITED_SNAPSHOT_REQUIRED")
    location_facts = frozenset(
        fact for fact in snapshot.fact_universe
        if (fact.predicate == "holding" and fact.arguments == ("black_book_1",))
        or (fact.predicate == "at" and fact.arguments[:1] == ("black_book_1",))
    )
    if snapshot.unknown(location_facts) or len(snapshot.true_facts & location_facts) != 1:
        return denied("EXACTLY_ONE_LOCATION_NOT_PROVEN")
    if capability.holding_fact not in snapshot.true_facts:
        return denied("HOLDING_NOT_EXPLICITLY_TRUE")
    if capability.target_fact not in snapshot.false_facts:
        return denied("TARGET_NOT_EXPLICITLY_FALSE")
    expected_action = FixedDomain().ground(
        problem,
        "place-held-in",
        (
            "black_book_1",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_access",
        ),
    )
    if place_node_action != expected_action:
        return denied("PLACE_NODE_ACTION_MISMATCH")
    if place_node_status not in capability.active_node_statuses:
        return denied("PLACE_NODE_NOT_ACTIVE")
    if not capability.protected_invariants <= snapshot.true_facts:
        return denied("PROTECTED_INVARIANT_NOT_TRUE")
    remaining = capability.max_combined_actions - base_policy_steps
    if remaining <= 0:
        return denied("ACTION_BUDGET_EXHAUSTED")
    event_payload = (
        f"{episode_id}:{snapshot.evidence_hash}:{graph.graph_hash}:"
        f"{certificate.certificate_hash}:{monitor_contract_sha256}:"
        f"{capability.event_type}"
    )
    event_id = hashlib.sha256(
        b"LOGIV_TERMINAL_DEVIATION_V1\0" + event_payload.encode("utf-8")
    ).hexdigest()
    return TerminalAssessment(
        eligible=True,
        reason="ELIGIBLE",
        event_id=event_id,
        event_type=capability.event_type,
        option_action_cap=min(capability.max_recovery_actions, remaining),
        snapshot_sha256=snapshot.evidence_hash,
        graph_hash=graph.graph_hash,
        certificate_hash=certificate.certificate_hash,
        monitor_contract_sha256=monitor_contract_sha256,
        protected_true_facts=frozenset(
            capability.protected_invariants | (problem.goal & snapshot.true_facts)
        ),
    )
```

Parse `holding_fact`, `target_fact`, and `protected_invariants` into `Fact` values when loading the contract so the implementation above compares typed facts only.

- [ ] **Step 6: Add and implement one-option signed-state plus VAL certification tests**

Tests must prove that a valid plan contains exactly one grounded action and that zero/multiple actions, signed-state failure, VAL error, graph hash change, and capability hash change return a denied `RecoveryPermit`. Build the action with `FixedDomain().ground(problem, "place-held-in", ("black_book_1", "desk_caddy_1_back_contain_region", "desk_caddy_1_access"))`, rebase `TaskProblem.initial_state/initial_false` on the strict snapshot, call `run_signed_trace`, and then call `ValWrapper.validate` with a one-item canonical occurrence sidecar and `ContextPhase.RECOVERY_VAL`.

```python
@dataclass(frozen=True)
class RecoveryPermit:
    granted: bool
    reason: str
    permit_sha256: str
    event_id: str
    action_cap: int
    plan: Sequence[GroundAction]
    plan_sha256: str | None
    certificate: PlanCertificate | None
    capability_sha256: str
    recovery_checkpoint_sha256: str
```

- [ ] **Step 7: Add and implement joint commit-verification tests**

The commit function must require the target TRUE, every goal fact TRUE at handoff still TRUE, every configured protected invariant TRUE, unchanged event/permit/plan/checkpoint/capability hashes, no UNKNOWN required literal, no combined action overflow, and native evaluator success. Test each failure independently and assert `committed is False` with a stable reason.

```python
@dataclass(frozen=True)
class RecoveryCommit:
    committed: bool
    reason: str
    commit_sha256: str
    base_actions: int
    recovery_actions: int
    combined_actions: int
```

- [ ] **Step 8: Run focused and existing symbolic tests**

Run: `uv run pytest tests/logiv/test_task5_terminal_recovery.py tests/logiv/test_val.py tests/logiv/test_domain.py -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit the pure contract boundary**

```bash
git add configs/logiv/task5-terminal-pi-recover-v1.json \
  src/pi05_libero_repro/logiv/task5_terminal_recovery.py \
  tests/logiv/test_task5_terminal_recovery.py
git commit -m "feat(logiv): define task5 terminal recovery permit"
```

### Task 2: Capture and gate the three strict terminal preflight roots

**Files:**

- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py` (`ShadowValidatedProposal`, `ShadowRuntime`)
- Modify: `scripts/eval_logiv_libero.py` (`_validate_shadow_options`, Base/Shadow terminal branch, parser)
- Create: `scripts/report_task5_terminal_preflight.py`
- Modify: `tests/logiv/test_shadow_runtime.py`
- Modify: `tests/logiv/test_evaluator.py`
- Create: `tests/logiv/test_task5_pi_recover_reports.py`

**Interfaces:**

- Consumes: Task 1 `assess_task5_terminal`, existing frozen ten-case seed manifest, `LiberoOracleGrounder.peek_snapshot()`.
- Produces: `--capture-task5-terminal-preflight`, `strict_terminal_snapshot_reader() -> FactSnapshot`, per-episode `current_snapshot.json`/`terminal_deviation.json`, and `task5-terminal-preflight.json`.

- [ ] **Step 1: Write failing tests for a distinct strict terminal reader**

Test that advisory monitoring still calls `peek_advisory_partial_snapshot`, while terminal preflight calls `peek_snapshot` once on the final observation and propagates ambiguity as a denial rather than falling back to the advisory snapshot.

```python
def test_terminal_reader_is_strict_and_separate_from_advisory_reader(fake_grounder):
    validated = build_validated_proposal(fake_grounder)
    validated.snapshot_reader({"frame": 1})
    terminal = validated.strict_terminal_snapshot_reader({"frame": 2})
    assert terminal is fake_grounder.strict_snapshot
    assert fake_grounder.advisory_calls == 1
    assert fake_grounder.strict_calls == 1
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/logiv/test_shadow_runtime.py::test_terminal_reader_is_strict_and_separate_from_advisory_reader tests/logiv/test_evaluator.py -q`

Expected: FAIL because `strict_terminal_snapshot_reader` and the CLI flag do not exist.

- [ ] **Step 3: Expose the strict reader without changing policy-time Shadow behavior**

Extend `ShadowValidatedProposal` with:

```python
strict_terminal_snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot]
```

In `_build_evaluator_shadow_runtime.live_validator`, update the same observation store and call `grounder.peek_snapshot()` in this callback. Keep the existing `snapshot_reader` on `peek_advisory_partial_snapshot()`.

- [ ] **Step 4: Add the development-only preflight evaluator path**

Add `--capture-task5-terminal-preflight`. Validation must require `SHADOW_LOGIV`, `--development-only`, task IDs exactly `(5,)`, and no recovery service. After Base native settling, write the strict snapshot and Task 1 assessment even though `SHADOW_LOGIV` still returns the original Base result. Serialize audited facts as sorted PDDL strings and include all graph/certificate/capability hashes.

```python
if args.capture_task5_terminal_preflight:
    terminal_snapshot = validation.strict_terminal_snapshot_reader(outcome.final_observation)
    terminal_assessment = assess_task5_terminal(
        capability,
        task_id=task_id,
        episode_id=episode_id,
        base_success=outcome.check_success,
        native_terminal_status=evaluated.value,
        base_policy_steps=outcome.steps,
        snapshot=terminal_snapshot,
        problem=certified.problem,
        graph=certified.graph,
        certificate=certified.certificate,
        monitor_contract_sha256=monitor_contract.contract_sha256,
        certificate_state=terminal_certificate_state,
        place_node_action=terminal_place_action,
        place_node_status=terminal_place_status,
    )
```

Catch strict grounding errors and emit `GROUNDING_ERROR` with zero physical side effects. Do not change `result`, `steps`, or Base request accounting.

- [ ] **Step 5: Implement the preflight report gate**

The report accepts exactly the three case artifact directories, verifies IDs `t05-r02/r03/r04`, fresh run timestamps, native Base failure, strict audited snapshots, and counts eligible held-state roots.

```python
def summarize_preflight(case_dirs: Sequence[Path]) -> dict[str, object]:
    rows = tuple(load_case(path) for path in case_dirs)
    if {row.case_id for row in rows} != {"t05-r02", "t05-r03", "t05-r04"}:
        raise ValueError("preflight requires the three frozen Task 5 failures")
    eligible = sum(row.eligible for row in rows)
    return {
        "schema_version": 1,
        "case_ids": [row.case_id for row in rows],
        "eligible_held_roots": eligible,
        "required_held_roots": 2,
        "go": eligible >= 2,
    }
```

- [ ] **Step 6: Run unit tests**

Run: `uv run pytest tests/logiv/test_shadow_runtime.py tests/logiv/test_evaluator.py tests/logiv/test_task5_pi_recover_reports.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Start the frozen Base policy service and rerun the three cases in simulation**

In terminal A:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=external_repos/openpi/src \
  external_repos/openpi/.venv/bin/python scripts/serve_episode_seeded_policy.py \
  --port 8010 \
  --policy-config pi05_libero \
  --policy-dir /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

In terminal B, run the exact frozen roots:

```bash
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 runs/task5-terminal-preflight-20260805/t05-r02 \
  --run-id task5-terminal-preflight-t05-r02 --goal-mode METADATA_ASSISTED --deviation-mode NOMINAL \
  --oracle-grounding --development-only --capture-task5-terminal-preflight \
  --task-ids 5 --episode-indices 36 --seed 54804909 --no-video
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 runs/task5-terminal-preflight-20260805/t05-r03 \
  --run-id task5-terminal-preflight-t05-r03 --goal-mode METADATA_ASSISTED --deviation-mode NOMINAL \
  --oracle-grounding --development-only --capture-task5-terminal-preflight \
  --task-ids 5 --episode-indices 18 --seed 3546047300 --no-video
scripts/run_logiv_eval.sh SHADOW_LOGIV 0 8010 runs/task5-terminal-preflight-20260805/t05-r04 \
  --run-id task5-terminal-preflight-t05-r04 --goal-mode METADATA_ASSISTED --deviation-mode NOMINAL \
  --oracle-grounding --development-only --capture-task5-terminal-preflight \
  --task-ids 5 --episode-indices 42 --seed 1564298395 --no-video
```

Expected: each command exits 0, retains native Base failure, and writes strict terminal artifacts without recovery requests/actions.

- [ ] **Step 8: Evaluate the mandatory pre-training gate**

```bash
uv run python scripts/report_task5_terminal_preflight.py \
  runs/task5-terminal-preflight-20260805/t05-r02 \
  runs/task5-terminal-preflight-20260805/t05-r03 \
  runs/task5-terminal-preflight-20260805/t05-r04 \
  --output results/task5-terminal-preflight-20260805.json
```

Expected: exit 0 and JSON contains `"eligible_held_roots": 2` or `3` and `"go": true`. If it reports fewer than two, stop this plan before Task 3 and return to design; do not train.

- [ ] **Step 9: Commit code and the small preflight report, not simulator arrays/video**

```bash
git add src/pi05_libero_repro/logiv/shadow_runtime.py scripts/eval_logiv_libero.py \
  scripts/report_task5_terminal_preflight.py tests/logiv/test_shadow_runtime.py \
  tests/logiv/test_evaluator.py tests/logiv/test_task5_pi_recover_reports.py \
  results/task5-terminal-preflight-20260805.json
git commit -m "feat(logiv): gate task5 recovery on strict terminal roots"
```

### Task 3: Build the deterministic held-suffix recovery dataset

**Files:**

- Create: `src/pi05_libero_repro/logiv/task5_recovery_dataset.py`
- Create: `scripts/build_task5_recovery_dataset.py`
- Create: `tests/logiv/test_task5_recovery_dataset.py`
- Create at execution time: `artifacts/manifests/task5-pi-recover-dataset-v1.json`

**Interfaces:**

- Consumes: official Task 5 HDF5, pinned BDDL, `TaskBinding`, `LiberoOracleGrounder`, Task 1 contract.
- Produces: `assign_demo_splits(demo_ids: Sequence[str]) -> Mapping[str, DemoSplit]`, `HeldSuffixDetector.observe(frame_index: int, holding: bool, target: bool | None) -> int | None`, `Task5RecoveryDatasetManifest`, LeRobot repos `logiv/task5-held-book-v1-train` and `logiv/task5-held-book-v1-loss-validation`, and ten `capability_roots/demo_XX.npz` files.

- [ ] **Step 1: Write failing pure tests for split, suffix, rejection, and hashes**

```python
def test_demo_split_is_identity_based_and_fixed_size():
    demo_ids = tuple(f"demo_{i}" for i in range(50))
    split = assign_demo_splits(demo_ids)
    assert sum(v is DemoSplit.TRAIN for v in split.values()) == 35
    assert sum(v is DemoSplit.LOSS_VALIDATION for v in split.values()) == 5
    assert sum(v is DemoSplit.CAPABILITY_VALIDATION for v in split.values()) == 10
    assert split == assign_demo_splits(tuple(reversed(demo_ids)))


def test_suffix_begins_at_first_of_three_stable_held_frames():
    detector = HeldSuffixDetector(required_stability=3)
    values = [(False, False), (True, False), (True, False), (True, False)]
    assert [detector.observe(i, h, t) for i, (h, t) in enumerate(values)] == [None, None, None, 1]


def test_target_true_or_unknown_breaks_held_streak():
    detector = HeldSuffixDetector(required_stability=3)
    detector.observe(4, True, False)
    detector.observe(5, True, None)
    assert detector.observe(6, True, False) is None
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/logiv/test_task5_recovery_dataset.py -q`

Expected: FAIL during collection because the dataset module is absent.

- [ ] **Step 3: Implement the pure dataset boundary**

The split ranking must be independent of filesystem/HDF5 ordering:

```python
def assign_demo_splits(demo_ids: Sequence[str]) -> dict[str, DemoSplit]:
    ids = tuple(demo_ids)
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RecoveryDatasetError("expected exactly 50 unique demonstration IDs")
    ranked = sorted(
        ids,
        key=lambda item: hashlib.sha256(
            b"LOGIV_TASK5_DEMO_SPLIT_V1\0" + item.encode("utf-8")
        ).digest(),
    )
    return {
        item: DemoSplit.TRAIN if index < 35 else
        DemoSplit.LOSS_VALIDATION if index < 40 else
        DemoSplit.CAPABILITY_VALIDATION
        for index, item in enumerate(ranked)
    }
```

`HeldSuffixDetector` accepts only explicit booleans; `None` resets the streak. Manifest validation requires 50 reports, at least 46 valid, and valid counts `TRAIN>=32`, `LOSS_VALIDATION>=4`, `CAPABILITY_VALIDATION==10`. Hash source HDF5, BDDL, every output episode frame range, every simulator root, schemas, prompt, and the canonical manifest body.

- [ ] **Step 4: Add failing CLI tests using a six-frame fake replay adapter**

Factor the executable around an injected `replay_demo(demo_id)` iterator so unit tests do not import LIBERO/h5py/LeRobot. Verify RGB arrays remain `uint8`, state is finite shape `(8,)`, action is finite shape `(7,)`, all frames carry the locked prompt, and an ambiguous location snapshot rejects the demo.

- [ ] **Step 5: Implement actual HDF5/LIBERO/LeRobot conversion**

In the real adapter:

1. Open only `artifacts/datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5`.
2. Assert keys are exactly the expected Task 5 demo set and source hash matches the emitted manifest.
3. Create a fresh pinned `OffScreenRenderEnv` for each demo, set the stored initial simulator state, replay 7-D actions, and update one `LiberoObservationStore` per observation.
4. Read strict facts with `LiberoOracleGrounder.peek_snapshot()`; accept the first three-observation interval with holding TRUE and target FALSE.
5. Copy from the first interval observation through native Task 5 success. A demo without native success is invalid.
6. For train/loss-validation, call `LeRobotDataset.add_frame` with `image`, `wrist_image`, `state`, `actions`, and `task`, then `save_episode()` once per demo.
7. For capability validation, save the flat simulator state and first observation at the suffix boundary in a compressed NPZ; do not write those demos into either training repo.
8. Write the manifest atomically after hashing all outputs.

The state mapping must match `prepare_observation`: EEF position (3), quaternion converted to axis-angle (3), gripper qpos (2). Do not use joint state in place of this 8-D vector.

- [ ] **Step 6: Run pure tests and a one-demo loader smoke test**

```bash
uv run pytest tests/logiv/test_task5_recovery_dataset.py -q
docker run --rm --network host --gpus device=0 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/repro/src:/app:/app/third_party/libero \
  -v "$PWD:/repro" -v "$PWD/external_repos/openpi:/app:ro" pi05-libero-eval:650c5b0 \
  bash -lc 'source /.venv/bin/activate && python -m pip install -q h5py && \
    python /repro/scripts/build_task5_recovery_dataset.py --demo-ids demo_0 \
      --smoke-only --output-root /repro/artifacts/datasets/task5-pi-recover-smoke'
```

Expected: tests PASS; smoke output reports one valid or explicitly rejected demo and verifies actual image/state/action shapes without creating the full manifest.

- [ ] **Step 7: Build all 50 demos after the Task 2 go gate**

```bash
docker run --rm --network host --gpus device=0 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/repro/src:/app:/app/third_party/libero \
  -e HF_LEROBOT_HOME=/repro/artifacts/datasets/lerobot \
  -v "$PWD:/repro" -v "$PWD/external_repos/openpi:/app:ro" pi05-libero-eval:650c5b0 \
  bash -lc 'source /.venv/bin/activate && python -m pip install -q h5py && \
    python /repro/scripts/build_task5_recovery_dataset.py \
      --source-hdf5 /repro/artifacts/datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5 \
      --capability-contract /repro/configs/logiv/task5-terminal-pi-recover-v1.json \
      --output-root /repro/artifacts/datasets/task5-pi-recover-v1 \
      --manifest /repro/artifacts/manifests/task5-pi-recover-dataset-v1.json'
```

Expected: exit 0; manifest has `valid_total>=46`, `valid_train>=32`, `valid_loss_validation>=4`, `valid_capability_validation=10`, no overlap, and all hashes populated.

- [ ] **Step 8: Commit builder, tests, and small manifest only**

```bash
git add src/pi05_libero_repro/logiv/task5_recovery_dataset.py \
  scripts/build_task5_recovery_dataset.py tests/logiv/test_task5_recovery_dataset.py \
  artifacts/manifests/task5-pi-recover-dataset-v1.json
git commit -m "feat(logiv): build task5 held-state recovery dataset"
```

### Task 4: Register and verify the exact independent LoRA training run

**Files:**

- Create: `src/pi05_libero_repro/logiv/task5_recovery_training.py`
- Create: `scripts/train_task5_pi_recover.py`
- Create: `tests/logiv/test_task5_recovery_training.py`

**Interfaces:**

- Consumes: Task 3 train repo/manifest and full expert checkpoint `/mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero`.
- Produces: `Task5RecoveryTrainingSpec`, `make_task5_recovery_train_config(spec, repo_id)`, `registered_checkpoint_steps() -> (1000, 2000, 3000, 4000)`, and checkpoint directories under ignored `artifacts/checkpoints/task5_pi_recover_v1/seed42/`.

- [ ] **Step 1: Write failing frozen-spec tests**

```python
def test_training_spec_is_pre_registered():
    spec = load_training_spec(DATASET_MANIFEST, SOURCE_CHECKPOINT)
    assert spec.seed == 42
    assert spec.batch_size == 64
    assert spec.num_train_steps == 4000
    assert spec.checkpoint_steps == (1000, 2000, 3000, 4000)
    assert spec.peak_lr == 5e-5
    assert spec.clip_gradient_norm == 1.0
    assert spec.ema_decay is None
    assert spec.wandb_enabled is False


def test_resume_requires_exact_manifest_and_config_hashes(tmp_path):
    state = make_resume_state(dataset_sha256="a" * 64, config_sha256="b" * 64)
    with pytest.raises(TrainingContractError, match="dataset manifest hash mismatch"):
        validate_resume(state, dataset_sha256="c" * 64, config_sha256="b" * 64)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/logiv/test_task5_recovery_training.py -q`

Expected: FAIL because the training module is absent.

- [ ] **Step 3: Implement the frozen spec and lazy OpenPI config**

Use this exact model/config construction inside the lazy function:

```python
model = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
return config.TrainConfig(
    name="task5_pi_recover_v1",
    exp_name="seed42",
    model=model,
    data=config.LeRobotLiberoDataConfig(
        repo_id=repo_id,
        assets=config.AssetsConfig(
            assets_dir=str(spec.source_checkpoint / "assets"),
            asset_id="physical-intelligence/libero",
        ),
        base_config=config.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=64,
    seed=42,
    lr_schedule=optimizer.CosineDecaySchedule(
        warmup_steps=100,
        peak_lr=5e-5,
        decay_steps=4000,
        decay_lr=5e-6,
    ),
    optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    freeze_filter=model.get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        str(spec.source_checkpoint / "params")
    ),
    num_train_steps=4000,
    save_interval=1000,
    keep_period=1000,
    wandb_enabled=False,
    checkpoint_base_dir=str(spec.checkpoint_base_dir),
)
```

The manifest hash, source checkpoint manifest hash, config hash, git revision, and dirty-diff hash must be saved before the first optimizer step. Resume is allowed only when all five match.

- [ ] **Step 4: Write a failing exact-checkpoint-loop test**

Use a fake train state whose `step` increments after every update; assert saves occur after completed optimizer steps 1000, 2000, 3000, 4000, not at loop indices 1000/2000/3000/3999.

```python
def test_checkpoint_schedule_uses_completed_optimizer_steps():
    saved = []
    run_registered_steps(
        start_step=0,
        final_step=4000,
        update=lambda step: step + 1,
        save=lambda step: saved.append(step),
    )
    assert saved == [1000, 2000, 3000, 4000]
```

- [ ] **Step 5: Implement the root-local training loop using OpenPI primitives**

Reuse OpenPI `init_train_state`, `train_step`, sharding, loader, and checkpoint manager, but own the loop so the registered completed-step schedule is exact:

```python
while int(train_state.step) < spec.num_train_steps:
    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(train_rng, train_state, batch)
    completed_step = int(train_state.step)
    if completed_step in spec.checkpoint_steps:
        checkpoints.save_state(manager, train_state, data_loader, completed_step)
    batch = next(data_iter)
manager.wait_until_finished()
```

Reject `--overwrite` once any checkpoint exists. `--resume` must call the hash check before restoration. Do not initialize W&B in any mode.

- [ ] **Step 6: Add and run a one-batch initialization/gradient integration test**

In the OpenPI environment, initialize from the full expert and take one batch. Flatten parameter paths before/after the update and assert:

```python
assert changed_paths
assert all("lora" in path.lower() for path in changed_paths)
assert any("PaliGemma" in path for path in changed_paths)
assert any("action_expert" in path.lower() for path in changed_paths)
assert frozen_non_lora_paths_before == frozen_non_lora_paths_after
```

Run:

```bash
PYTHONPATH=src:external_repos/openpi/src \
  external_repos/openpi/.venv/bin/python -m pytest \
  tests/logiv/test_task5_recovery_training.py -q
```

Expected: all tests PASS, including the marked one-batch integration test on a GPU host.

- [ ] **Step 7: Commit training code before launching the long run**

```bash
git add src/pi05_libero_repro/logiv/task5_recovery_training.py \
  scripts/train_task5_pi_recover.py tests/logiv/test_task5_recovery_training.py
git commit -m "feat(logiv): register task5 pi-recover LoRA run"
```

### Task 5: Train once, score validation loss, and freeze checkpoint identity

**Files:**

- Create: `scripts/score_task5_pi_recover.py`
- Modify: `tests/logiv/test_task5_recovery_training.py`
- Create at execution time: `artifacts/manifests/task5-pi-recover-training-v1.json`
- Create at execution time: `artifacts/manifests/task5-pi-recover-selected-v1.json`

**Interfaces:**

- Consumes: Task 4 exact training runner and Task 3 loss-validation repo.
- Produces: four immutable checkpoint hashes, a validation-loss table, and one `selected_checkpoint_step` chosen before simulation.

- [ ] **Step 1: Write failing deterministic checkpoint-selection tests**

```python
def test_minimum_validation_loss_selects_checkpoint():
    rows = [(1000, 0.40), (2000, 0.31), (3000, 0.33), (4000, 0.36)]
    assert select_checkpoint(rows) == 2000


def test_validation_loss_tie_selects_earlier_checkpoint():
    rows = [(1000, 0.30), (2000, 0.30), (3000, 0.32), (4000, 0.35)]
    assert select_checkpoint(rows) == 1000
```

- [ ] **Step 2: Implement deterministic loss scoring**

For each registered checkpoint, load the same LoRA model/config and the loss-validation LeRobot repo with shuffle disabled. Fold JAX RNG by `(checkpoint_step, batch_index)`, call `model.compute_loss(batch_rng, observation, actions, train=False)`, and aggregate `sum(loss*valid_examples)/sum(valid_examples)` in float64. Verify every validation demo contributes before writing the table atomically. Selection is `min(rows, key=lambda row: (row.mean_loss, row.step))`.

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/logiv/test_task5_recovery_training.py -q`

Expected: all pure tests PASS.

- [ ] **Step 4: Launch the single registered training run**

```bash
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME="$PWD/artifacts/datasets/lerobot" \
PYTHONPATH=src:external_repos/openpi/src \
external_repos/openpi/.venv/bin/python scripts/train_task5_pi_recover.py \
  --dataset-manifest artifacts/manifests/task5-pi-recover-dataset-v1.json \
  --source-checkpoint /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --checkpoint-base-dir artifacts/checkpoints \
  --training-manifest artifacts/manifests/task5-pi-recover-training-v1.json
```

Expected: exit 0; checkpoint directories `1000`, `2000`, `3000`, `4000` exist; no W&B run exists; training manifest hashes all four.

- [ ] **Step 5: Score all four checkpoints and freeze the winner**

```bash
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME="$PWD/artifacts/datasets/lerobot" \
PYTHONPATH=src:external_repos/openpi/src \
external_repos/openpi/.venv/bin/python scripts/score_task5_pi_recover.py \
  --dataset-manifest artifacts/manifests/task5-pi-recover-dataset-v1.json \
  --training-manifest artifacts/manifests/task5-pi-recover-training-v1.json \
  --checkpoint-root artifacts/checkpoints/task5_pi_recover_v1/seed42 \
  --output artifacts/manifests/task5-pi-recover-selected-v1.json
```

Expected: exactly four finite mean losses and one selected step; the manifest states `selection_basis=LOSS_VALIDATION_ONLY` and contains no simulator metrics.

- [ ] **Step 6: Verify separate restoration and commit the small immutable manifests**

Run a fresh process that loads only the selected checkpoint, performs one inference on a held-root observation, and exits. Expected: finite action array with shape `(10, 7)` and metadata hashes matching the selected manifest.

```bash
git add scripts/score_task5_pi_recover.py tests/logiv/test_task5_recovery_training.py \
  artifacts/manifests/task5-pi-recover-training-v1.json \
  artifacts/manifests/task5-pi-recover-selected-v1.json
git commit -m "feat(logiv): freeze selected task5 recovery checkpoint"
```

### Task 6: Serve the selected recovery checkpoint as an independent seeded policy

**Files:**

- Modify: `scripts/serve_episode_seeded_policy.py`
- Modify: `tests/test_protocol.py`
- Modify: `tests/logiv/test_task5_recovery_training.py`

**Interfaces:**

- Consumes: Task 5 selected manifest and Task 4 config builder.
- Produces: optional `--recovery-training-manifest`, `--recovery-selected-manifest`, and server metadata keys `logiv_recovery_service_protocol`, `logiv_recovery_capability_sha256`, `logiv_recovery_checkpoint_sha256`.

- [ ] **Step 1: Write failing server-configuration tests**

Test the Base default still calls `config.get_config("pi05_libero")`. Test recovery mode rejects missing/hash-mismatched manifests and constructs its config from the selected checkpoint. Test metadata includes RNG protocol v1 plus all recovery hashes.

- [ ] **Step 2: Implement lazy recovery mode without changing Base defaults**

Add optional dataclass fields:

```python
recovery_dataset_manifest: str = ""
recovery_training_manifest: str = ""
recovery_selected_manifest: str = ""
recovery_capability_contract: str = ""
```

If all are empty, preserve the existing Base path byte-for-byte. If any is set, require all, validate cross-hashes, build the LoRA config, require `policy_dir` equals the selected checkpoint directory, and advertise:

```python
metadata.update({
    "logiv_recovery_service_protocol": 1,
    "logiv_recovery_capability_sha256": capability.capability_sha256,
    "logiv_recovery_checkpoint_sha256": selected.checkpoint_sha256,
    "logiv_recovery_prompt_version": capability.prompt_version,
})
```

- [ ] **Step 3: Run Base and recovery service tests**

Run: `uv run pytest tests/test_protocol.py tests/logiv/test_task5_recovery_training.py -q`

Expected: all tests PASS; existing Base seed envelope tests remain unchanged.

- [ ] **Step 4: Start the selected service and run a metadata health check**

```bash
CUDA_VISIBLE_DEVICES=1 HF_LEROBOT_HOME="$PWD/artifacts/datasets/lerobot" \
PYTHONPATH=src:external_repos/openpi/src \
external_repos/openpi/.venv/bin/python scripts/serve_episode_seeded_policy.py \
  --port 8011 \
  --policy-dir "$(jq -r .selected_checkpoint_dir artifacts/manifests/task5-pi-recover-selected-v1.json)" \
  --recovery-dataset-manifest artifacts/manifests/task5-pi-recover-dataset-v1.json \
  --recovery-training-manifest artifacts/manifests/task5-pi-recover-training-v1.json \
  --recovery-selected-manifest artifacts/manifests/task5-pi-recover-selected-v1.json \
  --recovery-capability-contract configs/logiv/task5-terminal-pi-recover-v1.json
```

Expected: port 8011 is distinct from Base port 8010 and metadata hashes match all local manifests.

- [ ] **Step 5: Commit service support**

```bash
git add scripts/serve_episode_seeded_policy.py tests/test_protocol.py \
  tests/logiv/test_task5_recovery_training.py
git commit -m "feat(logiv): serve independent task5 recovery policy"
```

### Task 7: Add a bounded current-state recovery macro protocol

**Files:**

- Modify: `src/pi05_libero_repro/protocol.py`
- Create: `tests/test_recovery_protocol.py`

**Interfaces:**

- Consumes: a terminal live environment, independent `EpisodeSeededClient`, locked prompt, strict effect observer.
- Produces: `RecoveryStepContext`, `RecoveryOutcome`, and `run_recovery_macro(env, client, prompt, image_tools, *, action_cap, replan_steps, effect_reader, effect_confirmation_observations, settling_steps, recovery_observer) -> RecoveryOutcome`; extends `EpisodeOutcome` with `discarded_pending_actions`.

- [ ] **Step 1: Write failing protocol tests**

Test that recovery does not call `reset` or `set_init_state`, sends the locked prompt, discards no Base actions itself, stops after three stable effect observations, stops at the cap otherwise, counts requests/actions separately, handles policy errors as failure, and never exceeds the supplied action cap.

```python
def test_recovery_runs_from_current_state_and_stops_on_stable_effect():
    effects = iter([False, True, True, True])
    outcome = run_recovery_macro(
        env,
        recovery_client,
        LOCKED_PROMPT,
        image_tools,
        action_cap=12,
        replan_steps=5,
        effect_reader=lambda _observation: next(effects),
        effect_confirmation_observations=3,
    )
    assert env.reset_calls == 0
    assert outcome.effect_confirmed is True
    assert outcome.actions_executed == 4
    assert outcome.policy_requests == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/test_recovery_protocol.py -q`

Expected: FAIL because the recovery protocol does not exist.

- [ ] **Step 3: Expose discarded Base suffix accounting**

At every `run_episode` return, record `discarded_pending_actions=len(action_plan)`. The list is already local and is destroyed at return; this field makes the flush explicit. Existing Base hashes must remain computed only from executed actions.

- [ ] **Step 4: Implement the current-state macro loop**

The loop mirrors observation preparation and chunk validation from `run_episode` but omits reset/wait and owns a fresh empty action deque. It calls a recovery observer after every step so strict snapshots can update the same fixed graph. On exit it clears its own unused chunk, performs exactly the configured dummy-action settling, and returns all executed actions for hashing.

```python
@dataclass(frozen=True)
class RecoveryOutcome:
    effect_confirmed: bool
    terminal_reason: str
    actions_executed: int
    policy_requests: int
    actions: Sequence[np.ndarray]
    discarded_recovery_actions: int
    final_observation: Mapping[str, Any]
    inference_error: str | None
```

Catch client/shape/nonfinite errors inside recovery and return `inference_error`; do not fall back to Base or issue another prompt.

- [ ] **Step 5: Run protocol regression tests**

Run: `uv run pytest tests/test_recovery_protocol.py tests/test_protocol.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit the bounded executor**

```bash
git add src/pi05_libero_repro/protocol.py tests/test_recovery_protocol.py
git commit -m "feat(logiv): execute one bounded recovery macro"
```

### Task 8: Run the disjoint 10-root capability gate

**Files:**

- Create: `scripts/eval_task5_recovery_capability.py`
- Modify: `scripts/report_task5_pi_recover_gate.py`
- Modify: `tests/logiv/test_task5_pi_recover_reports.py`
- Create at execution time: `artifacts/manifests/task5-pi-recover-capability-v1.json`

**Interfaces:**

- Consumes: Task 3 ten capability roots, Task 5 selected checkpoint only, Task 6 service, Task 7 executor.
- Produces: ten native simulator outcomes and a pass/fail capability manifest that cannot alter checkpoint selection.

- [ ] **Step 1: Write failing report and provenance tests**

Test that the gate rejects fewer/more than ten roots, any root outside `CAPABILITY_VALIDATION`, a checkpoint different from the selected manifest, any protected-invariant regression, duplicate demo IDs, and success below 8/10.

```python
def test_capability_gate_requires_eight_of_ten_and_zero_regressions():
    rows = [capability_row(success=i < 8) for i in range(10)]
    report = summarize_capability(rows, selected_manifest=SELECTED)
    assert report["successes"] == 8
    assert report["invariant_violations"] == 0
    assert report["go"] is True
```

- [ ] **Step 2: Implement simulator-root evaluation**

For each root, create a fresh Task 5 environment, seed the simulator with the root's fixed branch seed derived as `uint32(sha256("LOGIV-task5-capability-seed-v1:demo_id")[:4])`, set the saved simulator state, strictly verify holding TRUE/target FALSE/protected facts TRUE, derive an independent recovery policy seed, run one recovery macro capped at 180, settle, strictly reground, and require both joint commit verification and native success. Emit one JSON record per root.

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/logiv/test_task5_pi_recover_reports.py tests/test_recovery_protocol.py -q`

Expected: all tests PASS.

- [ ] **Step 4: Run capability simulation against the already selected checkpoint**

```bash
docker run --rm --network host --gpus device=0 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/repro/src:/app:/app/packages/openpi-client/src:/app/third_party/libero \
  -v "$PWD:/repro" -v "$PWD/external_repos/openpi:/app:ro" pi05-libero-eval:650c5b0 \
  bash -lc 'source /.venv/bin/activate && python /repro/scripts/eval_task5_recovery_capability.py \
    --recovery-host 127.0.0.1 --recovery-port 8011 \
    --dataset-manifest /repro/artifacts/manifests/task5-pi-recover-dataset-v1.json \
    --selected-manifest /repro/artifacts/manifests/task5-pi-recover-selected-v1.json \
    --capability-contract /repro/configs/logiv/task5-terminal-pi-recover-v1.json \
    --output /repro/runs/task5-pi-recover-capability-20260805'
```

Then:

```bash
uv run python scripts/report_task5_pi_recover_gate.py capability \
  runs/task5-pi-recover-capability-20260805 \
  --selected-manifest artifacts/manifests/task5-pi-recover-selected-v1.json \
  --output artifacts/manifests/task5-pi-recover-capability-v1.json
```

Expected: `successes>=8`, `invariant_violations=0`, `go=true`. If false, stop before R2M integration; diagnose data/execution only and do not select another checkpoint.

- [ ] **Step 5: Commit capability code and small gate manifest**

```bash
git add scripts/eval_task5_recovery_capability.py scripts/report_task5_pi_recover_gate.py \
  tests/logiv/test_task5_pi_recover_reports.py \
  artifacts/manifests/task5-pi-recover-capability-v1.json
git commit -m "test(logiv): pass task5 recovery capability gate"
```

### Task 9: Integrate the real `LOGIV_R2M` terminal handoff and verification

**Files:**

- Modify: `src/pi05_libero_repro/logiv/evaluation.py`
- Modify: `src/pi05_libero_repro/logiv/records.py`
- Modify: `src/pi05_libero_repro/logiv/shadow_runtime.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `scripts/run_logiv_eval.sh`
- Modify: `scripts/verify_artifacts.py`
- Create: `tests/logiv/test_task5_r2m_evaluator.py`
- Modify: `tests/logiv/test_records.py`
- Modify: `tests/logiv/test_shadow_runtime.py`

**Interfaces:**

- Consumes: Tasks 1, 6, 7, and passed Task 8 capability manifest.
- Produces: `MethodArm.LOGIV_R2M`, one fail-closed terminal handoff path, recovery graph trace, required artifacts, and separate Base/recovery action/request accounting.

- [ ] **Step 1: Write failing arm/record tests**

Add exact enum value `LOGIV_R2M`. Record tests must require `base_policy_requests`, `recovery_policy_requests`, `base_actions`, `recovery_actions`, `combined_actions`, `base_action_prefix_sha256`, and optional `recovery_action_sha256`; assert `combined_actions == base_actions + recovery_actions <= 520`.

- [ ] **Step 2: Write failing evaluator scenario tests**

Use fakes to cover:

- Base success: recovery client is never constructed.
- Task 5 failure but not held, target UNKNOWN, stale certificate, inactive place node, VAL failure, missing service, mismatched server metadata, failed capability gate, or insufficient budget: zero recovery requests/actions and original Base failure.
- Eligible failure: exactly one connection, one handoff, one macro, generation increment, independent seed, fixed prompt, and no Base re-entry.
- Effect failure, invariant regression, hash mismatch, native failure, or recovery inference error: terminal failure.
- Verified effect/invariants/native success: one positive flip.
- Every path: Base prefix/hash through terminal is unchanged and combined actions are at most 520.

```python
def test_base_success_never_contacts_recovery_service(r2m_harness):
    result = r2m_harness.run(base_success=True)
    assert result.success is True
    assert result.recovery_factory_calls == 0
    assert result.recovery_actions == 0


def test_eligible_terminal_failure_commits_one_handoff(r2m_harness):
    result = r2m_harness.run(base_success=False, held=True, target=False)
    assert result.success is True
    assert result.handoffs == 1
    assert result.recovery_macros == 1
    assert result.base_reentries == 0
```

- [ ] **Step 3: Add `LOGIV_R2M` to the Base-plus-Shadow prefix branch**

`LOGIV_R2M` must build the same Shadow runtime and call the same `run_episode` as `SHADOW_LOGIV`, with identical Base arguments. Do not pass an intervention monitor. Run native settling/evaluation before any recovery decision. Preserve `_base_execution_payload` before recovery.

- [ ] **Step 4: Implement fail-closed terminal assessment and certification**

Only on native Base failure:

1. Obtain the strict snapshot through the Task 2 callback.
2. Read the final fixed-graph state and require `CURRENT` plus Task 5 place node `ACTIVE`.
3. Call Task 1 terminal assessment.
4. Create one read-only recovery-service client, query metadata without inference, and verify the passed capability manifest plus service hashes.
5. Rebase, signed-validate, and VAL-certify exactly one option.
6. Persist `terminal_deviation.json`, `current_snapshot.json`, `recovery_plan.json`, and `handoff_permit.json` atomically before dispatch.

Every exception maps to a stable denial record and zero recovery actions; the episode remains the Base failure.

- [ ] **Step 5: Dispatch once and project live recovery topology**

After the durable permit, wrap the metadata-validated recovery client from Step 4 in a fresh `EpisodeSeededClient` using Task 1 recovery seed and a recovery-specific client-config hash. Increment request generation to 1, record `recovery_rng.json`, flush/audit `outcome.discarded_pending_actions`, and call `run_recovery_macro` once. The metadata query performs no inference and Base-success or preliminarily ineligible episodes never connect to the recovery service.

For every recovery observation, strict-ground and call the same `ShadowGraphTracker.project` with `phase="RECOVERY"`; append to `runtime.state_trace`. Effect observation is TRUE only after the target has been strict TRUE for three consecutive recovery observations.

- [ ] **Step 6: Verify commit and emit all artifacts/accounting**

After the macro's ten dummy settling steps, take another fresh strict snapshot, run joint commit verification, then independently call native `env.check_success()`. Write:

- `recovery_rng.json`
- `recovery_execution.json`
- `post_recovery_snapshot.json`
- `recovery_commit.json`
- updated `shadow_graph.json`

Set success only from `RecoveryCommit.committed`. Record Base and recovery actions/requests separately; total `steps` is combined actions, while Base parity claims read only `base_execution.json`.

- [ ] **Step 7: Validate server options and shell runner**

Add evaluator arguments `--recovery-host`, `--recovery-port`, `--recovery-capability-contract`, `--recovery-dataset-manifest`, `--recovery-training-manifest`, `--recovery-selected-manifest`, and `--recovery-capability-manifest`. Require them only for `LOGIV_R2M`; `BASE` and `SHADOW_LOGIV` configs remain unchanged. Allow `LOGIV_R2M` in `run_logiv_eval.sh` and mount manifests read-only.

- [ ] **Step 8: Extend artifact verification**

`verify_artifacts.py` must recompute every self-hash and cross-hash, prove denial paths have no recovery RNG/execution actions, prove granted paths have one plan/permit/macro, prove Base prefix provenance is independent of recovery actions, and enforce combined budget. It must reject missing files rather than infer success from `episodes.jsonl`.

- [ ] **Step 9: Run focused and full repository tests**

```bash
uv run pytest tests/logiv/test_task5_r2m_evaluator.py tests/logiv/test_records.py \
  tests/logiv/test_shadow_runtime.py tests/test_recovery_protocol.py tests/test_protocol.py -q
uv run pytest -q
git diff --check
```

Expected: focused tests PASS; full suite PASS; diff check has no output.

- [ ] **Step 10: Commit real R2M integration**

```bash
git add src/pi05_libero_repro/logiv/evaluation.py src/pi05_libero_repro/logiv/records.py \
  src/pi05_libero_repro/logiv/shadow_runtime.py src/pi05_libero_repro/protocol.py \
  scripts/eval_logiv_libero.py scripts/run_logiv_eval.sh scripts/verify_artifacts.py \
  tests/logiv/test_task5_r2m_evaluator.py tests/logiv/test_records.py \
  tests/logiv/test_shadow_runtime.py tests/test_recovery_protocol.py tests/test_protocol.py
git commit -m "feat(logiv): hand off terminal task5 failures to pi-recover"
```

### Task 10: Run and audit the frozen ten-case Base versus `LOGIV_R2M` gate

**Files:**

- Modify: `scripts/report_task5_pi_recover_gate.py`
- Modify: `tests/logiv/test_task5_pi_recover_reports.py`
- Create at execution time: `results/task5-pi-recover-frozen10-20260805.json`
- Create at execution time: `results/task5-pi-recover-frozen10-20260805.md`

**Interfaces:**

- Consumes: frozen `runs/shadow-confirmed-deviation-dryrun-20260805/seed_manifest.json`, existing Base results, Task 9 R2M evaluator.
- Produces: paired parity/flip/accounting audit and go/no-go for the later fresh random100 experiment.

- [ ] **Step 1: Write failing paired-audit tests**

Test exact comparison of task/episode/master seed, initial state, simulator seed, Base policy seed, first frame, Base prompt, Base checkpoint, Base client config, and `actions_sha256`. Test positive/negative flips, eligibility/permit/macro counts, invariant violations, and combined budget. Any Base prefix mismatch invalidates the pair and the gate.

```python
def test_frozen_gate_requires_two_positive_and_zero_negative_flips():
    report = summarize_frozen_pairs(make_pairs(base_successes=7, r2m_successes=9))
    assert report["positive_flips"] == 2
    assert report["negative_flips"] == 0
    assert report["base_prefix_parity"] == 10
    assert report["go"] is True
```

- [ ] **Step 2: Implement exact frozen paired reporting**

The report must list all ten case IDs and, per case, Base result, R2M result, terminal eligibility/denial reason, recovery actions/requests, effect/invariant/native checks, combined actions, and parity fields. The gate is:

```python
go = (
    valid_pairs == 10
    and base_prefix_parity == 10
    and positive_task5_flips >= 2
    and negative_flips == 0
    and invariant_violations == 0
    and budget_violations == 0
)
```

- [ ] **Step 3: Run report tests**

Run: `uv run pytest tests/logiv/test_task5_pi_recover_reports.py -q`

Expected: all tests PASS.

- [ ] **Step 4: Ensure both independent services are healthy**

Keep Base on port 8010 from Task 2 and recovery on port 8011 from Task 6. Query metadata and require Base has only RNG protocol v1, while recovery also has the selected capability/checkpoint hashes. Record both metadata payload hashes in the run root.

- [ ] **Step 5: Run `LOGIV_R2M` on exactly the frozen ten-case manifest**

For each manifest row, invoke `scripts/run_logiv_eval.sh LOGIV_R2M` using that row's task ID, episode index, and master seed, Base port 8010, recovery port 8011, and all frozen manifests. Use a separate output directory per case under `runs/task5-pi-recover-frozen10-20260805/`; do not overwrite the earlier Base/Shadow runs.

The per-case evaluator tail is exactly:

```bash
--goal-mode METADATA_ASSISTED --deviation-mode NOMINAL --oracle-grounding --development-only \
--no-video --recovery-host 127.0.0.1 --recovery-port 8011 \
--recovery-capability-contract /repro/configs/logiv/task5-terminal-pi-recover-v1.json \
--recovery-dataset-manifest /repro/artifacts/manifests/task5-pi-recover-dataset-v1.json \
--recovery-training-manifest /repro/artifacts/manifests/task5-pi-recover-training-v1.json \
--recovery-selected-manifest /repro/artifacts/manifests/task5-pi-recover-selected-v1.json \
--recovery-capability-manifest /repro/artifacts/manifests/task5-pi-recover-capability-v1.json
```

Expected: all ten evaluator processes exit 0. A denied recovery is a valid episode result, not a process error.

- [ ] **Step 6: Verify every episode artifact before summarizing**

```bash
uv run python scripts/verify_artifacts.py runs/task5-pi-recover-frozen10-20260805
```

Expected: exit 0, 10 valid records, no hash/accounting/budget violation.

- [ ] **Step 7: Generate and inspect the frozen gate report**

```bash
uv run python scripts/report_task5_pi_recover_gate.py frozen-pairs \
  --seed-manifest runs/shadow-confirmed-deviation-dryrun-20260805/seed_manifest.json \
  --base-root runs/shadow-confirmed-deviation-dryrun-20260805/cases \
  --r2m-root runs/task5-pi-recover-frozen10-20260805 \
  --json-output results/task5-pi-recover-frozen10-20260805.json \
  --markdown-output results/task5-pi-recover-frozen10-20260805.md
```

Expected go condition: `BASE=7/10`, `LOGIV_R2M>=9/10`, at least two of the three Task 5 failures flip positive, all seven Base successes remain successes, Base-prefix parity is 10/10, and no invariant/budget violation.

- [ ] **Step 8: Run final regression and self-review**

```bash
uv run pytest -q
git diff --check
rg -n "TBD|TODO|implement later|fill in details|NotImplementedError|pass$" \
  src/pi05_libero_repro/logiv/task5_* scripts/*task5* tests/logiv/test_task5* \
  configs/logiv/task5-terminal-pi-recover-v1.json
```

Expected: full suite PASS; diff check empty; placeholder scan empty. Cross-check every approved design requirement against one emitted artifact or test.

- [ ] **Step 9: Commit the report code and frozen gate evidence**

```bash
git add scripts/report_task5_pi_recover_gate.py tests/logiv/test_task5_pi_recover_reports.py \
  results/task5-pi-recover-frozen10-20260805.json \
  results/task5-pi-recover-frozen10-20260805.md
git commit -m "results(logiv): validate task5 pi-recover frozen gate"
```

- [ ] **Step 10: Stop at the approved implementation boundary**

If the gate passes, freeze the code/dataset/checkpoint/prompt/contracts/seeds and prepare a separate fresh-random100 execution plan. Do not generate or inspect the random100 seeds in this implementation tranche. If the gate fails, classify the failure as exactly one of `DATASET`, `RECOVERY_EXECUTION`, `PERMIT`, or `COMMIT_VERIFICATION`, and return to the corresponding task without prompt/model sweep.
