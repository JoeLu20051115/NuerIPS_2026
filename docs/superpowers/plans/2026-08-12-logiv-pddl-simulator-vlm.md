# LOGIV PDDL Planner and Simulator-VLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fixed PDDL planner own LOGIV initial planning and local replanning, constrain GPT-4o to closed visual fact confirmation, and complete a simulator-backed zero-API smoke run.

**Architecture:** Rename the existing bounded `FixedDomain` state-space search boundary to `PddlPlanner` while retaining a compatibility alias for existing callers. Use that one planner instance for both pre-install certification and controller repair; remove all active evaluator construction of GPT-4o proposal/repair adapters. Restrict the LOGIV GPT-4o transport to `state_gate`, then smoke-test the unchanged controller with `LiberoOracleGrounder` as an explicitly labeled simulator-VLM test double.

**Tech Stack:** Python 3.11, pytest, fixed STRIPS PDDL domain, VAL, deterministic causal-DAG compiler, LIBERO simulator, frozen pi0.5 policy.

## Global Constraints

- Follow `LOGIV_method.pdf`: PDDL owns initial planning and local repair; GPT-4o owns visual fact confirmation only.
- Do not call the OpenAI API during unit, integration, or simulator smoke tests.
- Preserve the fixed PDDL Domain, registered objects, official goal, VAL, deterministic DAG compiler, controller Gates, retry budgets, and frozen low-level policy.
- Treat `UNKNOWN` as unresolved evidence: it cannot authorize dispatch, completion, or repair.
- Keep the existing scripted proposal fixture only as metadata and search-ranking input.
- Preserve all unrelated and pre-existing dirty-worktree changes.
- Do not run the later RoboTwin 10-task by 10-seed baseline as part of this plan.

---

### Task 1: Establish one PDDL planner boundary

**Files:**
- Modify: `src/pi05_libero_repro/logiv/repair.py`
- Modify: `src/pi05_libero_repro/logiv/evaluation.py`
- Modify: `src/pi05_libero_repro/logiv/controller.py`
- Modify: `tests/logiv/test_repair.py`
- Modify: `tests/logiv/test_evaluator.py`

**Interfaces:**
- Consumes: `TaskProblem`, current `FactSnapshot`, fixed allowed schemas, optional rough-plan ranking, `CausalSlice`, retry exclusions, and VAL budgets.
- Produces: `PddlPlanner.repair(...) -> RepairResult`; `CertifiedEpisode.planner`; compatibility aliases `RepairOperator` and `CertifiedEpisode.repair_operator`.

- [ ] **Step 1: Write the failing planner identity tests**

Add a focused test in `tests/logiv/test_repair.py`:

```python
def test_repair_operator_is_only_a_compatibility_name_for_pddl_planner() -> None:
    assert RepairOperator is PddlPlanner
```

Add a real-VAL certification test in `tests/logiv/test_evaluator.py` that constructs one `PddlPlanner`, passes it as `planner=planner`, and asserts:

```python
assert certified.planner is planner
assert certified.repair_operator is planner
assert certified.certificate is not None
assert certified.graph.certificate_hash == certified.certificate.certificate_hash
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/logiv/test_repair.py::test_repair_operator_is_only_a_compatibility_name_for_pddl_planner \
  tests/logiv/test_evaluator.py::test_initial_and_repair_share_one_pddl_planner
```

Expected: collection fails because `PddlPlanner`, the `planner` keyword, and `CertifiedEpisode.planner` do not exist.

- [ ] **Step 3: Implement the minimum planner rename and shared ownership**

Change the search class declaration in `repair.py` to:

```python
class PddlPlanner:
    """Bounded symbolic search over the fixed PDDL domain, followed by VAL."""

    # Keep the current constructor, catalog, sidecar, priority search, retry,
    # causal-slice ranking, and repair(...) implementation unchanged.


RepairOperator = PddlPlanner
```

In `evaluation.py`, make `CertifiedEpisode` store `planner: PddlPlanner`, expose:

```python
@property
def repair_operator(self) -> PddlPlanner:
    return self.planner
```

Rename `certify_initial_package(..., repair_operator=None)` to
`certify_initial_package(..., planner=None)`, instantiate `PddlPlanner` when it
is absent, use it for initial search/VAL, and return the same object for runtime
repair. Update the controller type annotation to `PddlPlanner` without changing
controller behavior.

- [ ] **Step 4: Run GREEN tests and the planner regression set**

Run:

```bash
.venv/bin/pytest -q tests/logiv/test_repair.py tests/logiv/test_evaluator.py
```

Expected: all focused tests pass with no failure.

---

### Task 2: Remove GPT-4o from planning and enforce VLM-only transport

**Files:**
- Modify: `src/pi05_libero_repro/logiv/gpt4o.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `tests/logiv/test_gpt4o.py`
- Modify: `tests/logiv/test_gpt4o_planning.py`
- Modify: `tests/logiv/test_evaluator.py`

**Interfaces:**
- Consumes: `--perception-backend scripted-oracle|gpt4o`.
- Produces: GPT-4o calls whose only accepted purpose is `state_gate`; evaluator paths that always use `ScriptedProposalProvider` as metadata scaffold and `PddlPlanner` for planning.

- [ ] **Step 1: Write failing responsibility-boundary tests**

Add to `tests/logiv/test_gpt4o.py`:

```python
@pytest.mark.parametrize("purpose", ["initial_plan", "local_repair"])
def test_logiv_gpt4o_transport_rejects_non_vlm_purposes(purpose: str) -> None:
    opened = []
    client = Gpt4oClient("secret", urlopen=lambda *args, **kwargs: opened.append(args))
    with pytest.raises(Gpt4oRequestError, match="state_gate"):
        client.complete_json(
            purpose=purpose,
            system="system",
            text="facts",
            images=(),
            schema_name="facts",
            schema={"type": "object"},
        )
    assert opened == []
    assert client.request_counts == {}
```

Replace the old GPT-planning behavior tests in
`tests/logiv/test_gpt4o_planning.py` with an architecture test:

```python
def test_evaluator_has_no_gpt4o_planning_or_repair_runtime_path() -> None:
    source = Path(evaluator_script.__file__).read_text()
    assert "gpt4o_planning" not in source
    assert "Gpt4oProposalProvider" not in source
    assert "Gpt4oRepairOperator" not in source
```

Add a metric test in `tests/logiv/test_evaluator.py` requiring a helper to
return `(state_gate_count, 0)` for an absent simulator client and to reject any
request-count key other than `state_gate`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest -q \
  tests/logiv/test_gpt4o.py::test_logiv_gpt4o_transport_rejects_non_vlm_purposes \
  tests/logiv/test_gpt4o_planning.py \
  tests/logiv/test_evaluator.py -k 'gpt4o_request_accounting'
```

Expected: the transport accepts non-VLM purposes and the evaluator source still imports and constructs both GPT-4o planning adapters.

- [ ] **Step 3: Implement the VLM-only boundary**

At the beginning of `Gpt4oClient.complete_json`, before counters or HTTP work,
add:

```python
if purpose != "state_gate":
    raise Gpt4oRequestError(
        "LOGIV GPT-4o is restricted to state_gate visual fact confirmation"
    )
```

In `scripts/eval_logiv_libero.py`:

- remove the `gpt4o_planning` import;
- always obtain task metadata from `ScriptedProposalProvider`;
- construct `Gpt4oClient` only when the perception backend is `gpt4o`;
- keep `Gpt4oGrounder` as the sole consumer of that client;
- let `certify_initial_package` create one `PddlPlanner` for initial planning
  and controller repair;
- remove all `Gpt4oRepairOperator` branches from direct and shadow validation;
- add `_gpt4o_request_accounting(client)` which accepts only `state_gate` and
  returns `{"shadow_vlm_requests": count, "recovery_policy_requests": 0}`;
- use that helper in both normal and exception accounting paths.

Update transport retry/timeout tests that used `initial_plan` or `local_repair`
as arbitrary labels to use `state_gate`; retain their original retry assertions.

- [ ] **Step 4: Run GREEN tests and evaluator regression tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/logiv/test_gpt4o.py \
  tests/logiv/test_gpt4o_grounding.py \
  tests/logiv/test_gpt4o_planning.py \
  tests/logiv/test_evaluator.py \
  tests/logiv/test_controller.py \
  tests/logiv/test_closed_loop_scenarios.py
```

Expected: all selected tests pass and no test issues an HTTP request.

---

### Task 3: Verify the method boundary and run one zero-API simulator smoke

**Files:**
- Modify only if a verified smoke defect requires it: files already listed in Tasks 1-2 and their focused tests.
- Create: `docs/experiments/2026-08-12-logiv-pddl-simulator-vlm-smoke.md`
- Runtime artifact: `artifacts/logiv-pddl-simulator-vlm-smoke/`

**Interfaces:**
- Consumes: existing frozen LIBERO checkpoint/server, task 0, episode index 0, master seed 7, `--perception-backend scripted-oracle`.
- Produces: one simulator episode record with a VAL-certified initial graph, Gate events, explicit simulator/oracle labeling, and zero GPT-4o requests.

- [ ] **Step 1: Run static and full unit verification**

Run:

```bash
.venv/bin/pytest -q
bash -n scripts/run_logiv_eval.sh
git diff --check
rg -n "Gpt4oProposalProvider|Gpt4oRepairOperator|gpt4o_planning" \
  scripts src/pi05_libero_repro/logiv \
  -g '!src/pi05_libero_repro/logiv/gpt4o_planning.py'
```

Expected: pytest exits 0, shell syntax exits 0, diff check is silent, and the
search returns no active runtime reference.

- [ ] **Step 2: Confirm and execute the exact simulator command**

Before execution, present the fully resolved command to the user as required by
the experiment workflow. It must select exactly task 0, episode 0, master seed
7, `FULL_LOGIV`, `scripted-oracle`, the existing checkpoint, no GPT-4o key
forwarding, and a new output directory. After confirmation, execute that exact
command on the currently free GPU without changing seeds or budgets.

- [ ] **Step 3: Validate the smoke artifact**

Parse the generated episode JSON and assert:

```python
assert record["task_id"] == 0
assert record["episode_idx"] == 0
assert record["seed"] == 7
assert record["perception_backend"] == "scripted-oracle"
assert record["oracle_grounding"] is True
assert record["record_accounting"]["shadow_vlm_requests"] == 0
assert record["record_accounting"]["recovery_policy_requests"] == 0
assert record["initial_certificate_hash"]
assert record["initial_graph_hash"]
```

Also require no uncaught exception and inspect the event journal for graph
activation plus pre-dispatch fact authorization. Record terminal success/fail
without altering the plumbing acceptance decision.

- [ ] **Step 4: Write the smoke report and rerun final checks**

Document exact command, GPU, checkpoint, task, seed, simulator/oracle label,
terminal status, action count, VAL/graph hashes, Gate evidence, and zero API
calls in `docs/experiments/2026-08-12-logiv-pddl-simulator-vlm-smoke.md`.

Run:

```bash
.venv/bin/pytest -q
git diff --check
```

Expected: full suite passes, diff check is silent, and the report links the
immutable smoke artifact.
