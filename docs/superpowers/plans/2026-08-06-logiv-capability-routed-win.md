# LOGIV Capability-Routed Win Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship and independently validate one capability-routed LOGIV arm that uses verified Task 4 v68 execution, exact BASE fallback elsewhere, and achieves strictly more paired native successes than BASE.

**Architecture:** A self-hashed static contract selects a controller from `task_id` only. Task 4 runs the existing certified FULL_LOGIV path with frozen v68 settings; every other task enters the unchanged BASE episode runner before any observation-dependent logic. Episode records and a report auditor prove routing, pairing, and BASE-fallback action parity.

**Tech Stack:** Python 3.11+, pytest, NumPy, LIBERO/robosuite, OpenPI WebSocket policy service, VAL, JSON/SHA-256 manifests, Docker.

## Global Constraints

- Confirmation policy master seed is frozen at `20260806` before any confirmation outcome is observed.
- The router may use only `task_id`; never episode index, seed, observation, policy output, or outcome.
- Task 4 uses exactly the v68 paths, hashes, prompts, and numeric bounds in the design.
- Tasks 0--3 and 5--9 execute BASE with zero proposal, grounding, VAL, graph, repair, or recovery work.
- Every arm uses the same full `pi05_libero` checkpoint, paired episode state, paired derived policy/simulator seeds, 520-action cap, ten settling steps, and native evaluator.
- Confirmation requires strictly positive paired successes and zero negative flips; otherwise archive and move to the Task 5 goal-regression design without retuning on confirmation.
- Preserve unrelated dirty-worktree files and implement from clean `91c22cd4` in an isolated worktree.

---

### Task 1: Freeze and parse the static capability contract

**Files:**

- Create: `configs/logiv/capability-routed-v1.json`
- Create: `src/pi05_libero_repro/logiv/capability_router.py`
- Create: `tests/logiv/test_capability_router.py`

**Interfaces:**

- Consumes: task ID and evaluator runtime settings.
- Produces: `load_capability_router(path: Path) -> CapabilityRouterContract`, `CapabilityRouterContract.route_for_task(task_id: int) -> CapabilityRoute`, and `CapabilityRouterContract.validate_runtime(args: Namespace) -> None`.

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import replace
from pathlib import Path
import json

import pytest

from pi05_libero_repro.logiv.capability_router import (
    SelectedController,
    load_capability_router,
)

CONTRACT = Path(__file__).parents[2] / "configs/logiv/capability-routed-v1.json"


def test_router_enables_only_task4_and_is_self_hashed() -> None:
    router = load_capability_router(CONTRACT)
    assert router.recompute_sha256() == router.contract_sha256
    assert router.confirmation_policy_master_seed == 20260806
    assert router.route_for_task(4).controller is SelectedController.FULL_LOGIV_V68
    for task_id in (*range(4), *range(5, 10)):
        assert router.route_for_task(task_id).controller is SelectedController.BASE


def test_router_rejects_unknown_tasks_and_tampering(tmp_path: Path) -> None:
    router = load_capability_router(CONTRACT)
    with pytest.raises(ValueError, match="task_id"):
        router.route_for_task(10)
    payload = json.loads(CONTRACT.read_text())
    payload["enabled_tasks"]["4"]["max_total_action_steps"] = 521
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-hash"):
        load_capability_router(path)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/logiv/test_capability_router.py`

Expected: collection fails with `ModuleNotFoundError` because the router module does not exist.

- [ ] **Step 3: Add the exact contract**

```json
{
  "schema_version": 1,
  "capability_id": "logiv-capability-routed-v1",
  "enabled_tasks": {
    "4": {
      "controller": "FULL_LOGIV_V68",
      "coverage_manifest": "configs/logiv/libero10-coverage-v38-task0-recovery.json",
      "coverage_manifest_sha256": "8ccc34d932ac32704cddffaf4433e146f0d0e418363088ad6fbb47dd709f6920",
      "proposal_config": "configs/logiv/libero10-scripted-proposals-v53-task6-order.json",
      "proposal_config_sha256": "4b0171c513a74845eb87957d381594e244566414c3b2720d0f408ec601ec7cb1",
      "prompt_config": "configs/logiv/prompts/pi05-subtasks-v68-task4-left.json",
      "prompt_config_sha256": "858d63e1639d4b391f698f4158a9365e29f24328336229ee1ec0ad58751bbad4",
      "prompt_version": "pi05-subtasks-v68-task4-left",
      "effect_confirmation_steps": 5,
      "place_effect_stabilization_steps": 10,
      "frontier_followup_steps": 5,
      "frontier_completion_followup_steps": 120,
      "frontier_completion_recovery_only": true,
      "frontier_fallback_followup_steps": 120,
      "max_action_steps": 260,
      "max_total_action_steps": 520
    }
  },
  "fallback_controller": "BASE",
  "base_max_steps": 520,
  "settling_steps": 10,
  "confirmation_policy_master_seed": 20260806,
  "contract_sha256": "ec1103e6125739144dc70190dad41c52b4f918d9b13f846312770fa3e6c857e1"
}
```

- [ ] **Step 4: Implement the minimum typed loader**

Use frozen dataclasses, reject missing/unknown fields, remove `contract_sha256` before canonical
hashing, require enabled task IDs to be a strict subset of `0..9`, and return BASE for every omitted
task. `validate_runtime` compares all contract paths/hashes/bounds against the parsed CLI namespace;
it must not read observations or episode metadata.

```python
class SelectedController(str, Enum):
    BASE = "BASE"
    FULL_LOGIV_V68 = "FULL_LOGIV_V68"


@dataclass(frozen=True)
class CapabilityRoute:
    task_id: int
    controller: SelectedController
    settings: Mapping[str, object] | None


@dataclass(frozen=True)
class CapabilityRouterContract:
    capability_id: str
    enabled_tasks: Mapping[int, Mapping[str, object]]
    base_max_steps: int
    settling_steps: int
    confirmation_policy_master_seed: int
    contract_sha256: str

    def route_for_task(self, task_id: int) -> CapabilityRoute:
        if not 0 <= task_id <= 9:
            raise ValueError("task_id is outside LIBERO-10")
        settings = self.enabled_tasks.get(task_id)
        return CapabilityRoute(
            task_id,
            SelectedController.BASE if settings is None else SelectedController.FULL_LOGIV_V68,
            settings,
        )
```

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest -q tests/logiv/test_capability_router.py`

Expected: all tests pass.

Commit:

```bash
git add configs/logiv/capability-routed-v1.json \
  src/pi05_libero_repro/logiv/capability_router.py \
  tests/logiv/test_capability_router.py
git commit -m "feat(logiv): freeze capability routing contract"
```

### Task 2: Rebuild the minimal v68 place-stabilization capability

**Files:**

- Create: `configs/logiv/prompts/pi05-subtasks-v68-task4-left.json`
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `tests/logiv/test_libero_adapter.py`
- Modify: `tests/logiv/test_evaluator.py`

**Interfaces:**

- Consumes: sourced place schemas and existing effect observations.
- Produces: `Pi05MacroExecutor(..., place_effect_stabilization_steps: int = 0)` and CLI flag `--place-effect-stabilization-steps`.

- [ ] **Step 1: Write failing executor tests**

Add one fake trace where a sourced place effect becomes true for five samples, rebounds during the
ten-step stabilization window, then succeeds after policy execution resumes. Assert a held-place
schema is unchanged because the v68 contract covers only sourced place macros.

```python
def test_place_effect_stabilization_resumes_after_rebound() -> None:
    executor, grounder, client = build_place_executor(
        effect_trace=[False] * 2 + [True] * 5 + [True] * 3 + [False] + [True] * 15,
        place_effect_stabilization_steps=10,
    )
    outcome = execute_one_place(executor)
    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert client.inference_requests >= 2
    assert executor.results[0].reason == "observed stabilized declared effects"


def test_place_stabilization_does_not_change_held_place() -> None:
    executor, _, _ = build_held_place_executor(place_effect_stabilization_steps=10)
    outcome = execute_one_place(executor)
    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert executor.results[0].reason == "observed declared effects"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/logiv/test_libero_adapter.py -k 'place_effect_stabilization'`

Expected: failures because the constructor has no place-stabilization parameter.

- [ ] **Step 3: Implement the smallest executor change**

Validate and store the nonnegative bound. In `await_outcome`, use the existing close-access
stabilization state machine with a local bound:

```python
sourced_place_schemas = {"place-on", "place-in", "place-relative"}
effect_stabilization_steps = (
    self.place_effect_stabilization_steps
    if queued.action.schema in sourced_place_schemas
    else self.access_effect_stabilization_steps
    if queued.action.schema == "close-access"
    else 0
)
```

Replace the two close-only bound checks with `effect_stabilization_steps`; do not alter prompts,
effect predicates, retries, held-place behavior, or total budget accounting.

- [ ] **Step 4: Add the one-file v68 overlay and CLI plumbing**

```json
{
  "extends": "pi05-subtasks-v55-routed-context.json",
  "prompt_version": "pi05-subtasks-v68-task4-left",
  "action_overrides": {
    "(place-on porcelain_mug_1 living_room_table_porcelain_mug_init_region plate_1)": "Put the solid white mug on the left plate."
  },
  "action_phase_overrides": {
    "(place-on porcelain_mug_1 living_room_table_porcelain_mug_init_region plate_1)": {}
  }
}
```

Pass `args.place_effect_stabilization_steps` into `Pi05MacroExecutor`, record it in `run.json`, and
add the integer CLI flag with default zero.

- [ ] **Step 5: Verify focused and full tests, then commit**

Run:

```bash
uv run pytest -q tests/logiv/test_libero_adapter.py -k 'place_effect_stabilization or v68'
uv run pytest -q tests/logiv/test_libero_adapter.py tests/logiv/test_evaluator.py
```

Expected: all selected tests pass.

Commit:

```bash
git add configs/logiv/prompts/pi05-subtasks-v68-task4-left.json \
  src/pi05_libero_repro/logiv/libero_adapter.py scripts/eval_logiv_libero.py \
  tests/logiv/test_libero_adapter.py tests/logiv/test_evaluator.py
git commit -m "feat(logiv): stabilize certified task4 placement"
```

### Task 3: Add the capability-routed method arm and auditable records

**Files:**

- Modify: `src/pi05_libero_repro/logiv/evaluation.py`
- Modify: `src/pi05_libero_repro/logiv/records.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `scripts/run_logiv_eval.sh`
- Modify: `tests/logiv/test_evaluator.py`
- Modify: `tests/logiv/test_records.py`

**Interfaces:**

- Consumes: Task 1 router and Task 2 FULL_LOGIV executor.
- Produces: `MethodArm.LOGIV_CAPABILITY_ROUTED`, `--capability-router-contract`, and per-episode `selected_controller` / `capability_router_sha256` fields.

- [ ] **Step 1: Write failing routing and record tests**

```python
def test_routed_arm_selects_full_only_for_task4(router) -> None:
    assert effective_arm(MethodArm.LOGIV_CAPABILITY_ROUTED, router, 4) is MethodArm.FULL_LOGIV
    for task_id in (*range(4), *range(5, 10)):
        assert effective_arm(MethodArm.LOGIV_CAPABILITY_ROUTED, router, task_id) is MethodArm.BASE


def test_routed_base_record_requires_zero_logiv_accounting() -> None:
    record = routed_record(task_id=3, selected_controller="BASE")
    assert validate_episode_records([record]) == []
    changed = replace(record, total_val_calls=1)
    assert any("BASE fallback" in item for item in validate_episode_records([changed]))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/logiv/test_evaluator.py tests/logiv/test_records.py -k routed`

Expected: failures because the enum, selector, fields, and validation do not exist.

- [ ] **Step 3: Add the method and pure selector**

```python
class MethodArm(str, Enum):
    BASE = "BASE"
    SHADOW_LOGIV = "SHADOW_LOGIV"
    LOGIV_CAPABILITY_ROUTED = "LOGIV_CAPABILITY_ROUTED"
    # existing arms remain unchanged


def effective_arm(
    requested: MethodArm, router: CapabilityRouterContract | None, task_id: int
) -> MethodArm:
    if requested is not MethodArm.LOGIV_CAPABILITY_ROUTED:
        return requested
    if router is None:
        raise ValueError("capability-routed arm requires a router contract")
    return (
        MethodArm.FULL_LOGIV
        if router.route_for_task(task_id).controller is SelectedController.FULL_LOGIV_V68
        else MethodArm.BASE
    )
```

- [ ] **Step 4: Route before reset and bind the symbolic executor to the effective arm**

Load and validate the router before creating the suite or environment. Select the effective arm from
`task_id` at the top of each task loop. Extend `_execute_symbolic_arm` with an explicit
`arm: MethodArm | None = None` parameter so it never re-reads the outer method by accident. Write
`routing.json` before reset with task, selected controller, and contract hash.

- [ ] **Step 5: Add backward-compatible record fields and invariants**

```python
selected_controller: str = "NOT_APPLICABLE"
capability_router_sha256: str | None = None
```

For routed records require controller in `{BASE, FULL_LOGIV_V68}`, a valid router hash, and:

```python
if record.selected_controller == "BASE" and any((
    record.initial_proposal_requests,
    record.recovery_policy_requests,
    record.total_val_calls,
    record.repair_rounds,
    record.graph_installs,
)):
    errors.append(f"BASE fallback has nonzero LOGIV accounting: {label}")
```

Preserve the outer method name in `record.method_arm`; set `oracle_grounding` from the selected
controller, not the outer method.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
uv run pytest -q tests/logiv/test_evaluator.py tests/logiv/test_records.py
uv run pytest -q
git diff --check
```

Expected: the full suite passes and diff check is clean.

Commit:

```bash
git add src/pi05_libero_repro/logiv/evaluation.py \
  src/pi05_libero_repro/logiv/records.py scripts/eval_logiv_libero.py \
  scripts/run_logiv_eval.sh tests/logiv/test_evaluator.py tests/logiv/test_records.py
git commit -m "feat(logiv): route only validated task capabilities"
```

### Task 4: Build the paired routing and parity auditor

**Files:**

- Create: `scripts/report_capability_routed_results.py`
- Create: `tests/logiv/test_capability_routed_report.py`

**Interfaces:**

- Consumes: BASE and routed `episodes.jsonl`, their artifact roots, and the frozen router contract.
- Produces: JSON/Markdown reports with per-task counts, flips, Wilson intervals, BASE-fallback parity, and 10,000-repeat task-stratified paired bootstrap.

- [ ] **Step 1: Write failing report tests**

Fixtures must include one enabled-task positive flip and two fallback tasks with equal
`base_execution.json/actions_sha256`. Assert a mismatched fallback action hash is rejected.

```python
def test_report_requires_base_fallback_action_parity(tmp_path: Path) -> None:
    base_root, routed_root = write_paired_fixture(tmp_path, mismatch_task=3)
    with pytest.raises(ValueError, match="BASE fallback action hash mismatch"):
        summarize(base_root, routed_root, CONTRACT, bootstrap_samples=100, bootstrap_seed=2026)


def test_report_counts_one_positive_and_zero_negative_flips(tmp_path: Path) -> None:
    base_root, routed_root = write_paired_fixture(tmp_path)
    report = summarize(base_root, routed_root, CONTRACT, bootstrap_samples=100, bootstrap_seed=2026)
    assert report["paired_positive_flips"] == 1
    assert report["paired_negative_flips"] == 0
    assert report["paired_net_successes"] == 1
    assert report["base_fallback_action_parity"] is True
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/logiv/test_capability_routed_report.py`

Expected: collection fails because the report module does not exist.

- [ ] **Step 3: Implement deterministic validation and statistics**

Pair on `(task_id, episode_idx)`. Require equal task sets, init hash, first-frame hash, seed,
checkpoint, and router contract hash. For every BASE-fallback pair require equal success, steps,
inference requests, and `actions_sha256`. Use `pi05_libero_repro.records.wilson_interval` and a local
NumPy RNG seeded at 2026 for the stratified bootstrap. The acceptance field is exactly:

```python
accepted = positive_flips > negative_flips and negative_flips == 0 and fallback_parity
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest -q tests/logiv/test_capability_routed_report.py`

Expected: all tests pass.

Commit:

```bash
git add scripts/report_capability_routed_results.py \
  tests/logiv/test_capability_routed_report.py
git commit -m "feat(logiv): audit capability-routed paired results"
```

### Task 5: Reproduce development evidence and run locked confirmation

**Files:**

- Create at execution time: `results/logiv-capability-routed-seed7-replay.json`
- Create at execution time: `results/logiv-capability-routed-confirm-20260806.json`
- Create at execution time: `results/logiv-capability-routed-confirm-20260806.md`
- Modify: `docs/experiments/2026-08-03-logiv-v53-failure-seed-retrospective.md`

**Interfaces:**

- Consumes: frozen commit from Tasks 1--4 and full pi0.5 service on port 8010.
- Produces: a seed-7 replay gate followed by the independent seed-20260806 paired result.

- [ ] **Step 1: Start the frozen episode-seeded policy service**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=external_repos/openpi/src \
  external_repos/openpi/.venv/bin/python scripts/serve_episode_seeded_policy.py \
  --port 8010 --policy-config pi05_libero \
  --policy-dir /mnt/data3/data_xingrui/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

Require healthy metadata with LOGIV episode RNG protocol v1 before evaluation.

- [ ] **Step 2: Replay the development hard roots and pair episode 45**

Run BASE and routed LOGIV for Task 4 episodes `14,41,44,45` at seed 7, no video. Require all four
routed successes, BASE failure only at 45, matching pair hashes, and zero negative flips. A failed
replay stops before confirmation.

- [ ] **Step 3: Freeze the candidate commit and run independent Task 4 confirmation**

Run BASE then `LOGIV_CAPABILITY_ROUTED` for Task 4 episodes `0-49` with seed `20260806`, exact
contract paths, no video, and separate output directories. Do not change code/configs between arms.

- [ ] **Step 4: Audit the Task 4 confirmation gate**

```bash
uv run python scripts/report_capability_routed_results.py \
  --base-root runs/logiv-capability-base-task4-confirm-20260806 \
  --routed-root runs/logiv-capability-routed-task4-confirm-20260806 \
  --contract configs/logiv/capability-routed-v1.json \
  --json-output results/logiv-capability-routed-confirm-20260806.json \
  --markdown-output results/logiv-capability-routed-confirm-20260806.md
```

Expected: `accepted=true`, at least one positive flip, zero negative flips, and 50 matching initial
state / first-frame pairs. If not, archive and begin the Task 5 goal-regression design; never retune
v68 on confirmation.

- [ ] **Step 5: Run the full 10x50 paired matrix after the Task 4 gate passes**

Run BASE and routed LOGIV at seed `20260806` for all ten tasks and 50 episode indices. Audit exact
fallback action parity on 450 pairs and strict positive net success on all 500 pairs.

- [ ] **Step 6: Verify artifacts, tests, and commit the accepted evidence**

Run:

```bash
uv run pytest -q
git diff --check
uv run python scripts/verify_artifacts.py runs/logiv-capability-routed-confirm-20260806
```

Update the experiment retrospective with separate development and confirmation tables, exact commit
and contract hashes, paired flips, Wilson intervals, bootstrap interval, and oracle-grounding scope.
Commit only small manifests/reports/docs; do not commit videos or simulator arrays.

## Self-Review

- Spec coverage: static routing, v68 capability, BASE fallback, provenance, action parity, paired
  statistics, locked seed, rejection path, and full-matrix confirmation all map to explicit tasks.
- Placeholder scan: no TBD/TODO/implicit implementation steps remain.
- Type consistency: the router returns `CapabilityRoute`; evaluator selection uses
  `SelectedController`; records use stable string values `BASE` and `FULL_LOGIV_V68`.
