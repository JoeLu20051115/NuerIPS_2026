# LOGIV DAG-Frontier Execution Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task, with `superpowers:test-driven-development` for every behavior change and `superpowers:systematic-debugging` for every failed experiment.

**Goal:** Remove the artificial per-occurrence interruption that makes Full LOGIV regress below Base on LIBERO-Long task 8, while preserving symbolically certified plans, fresh-fact gates, one-receipt-per-occurrence semantics, equal global action budgets, and a genuine non-chain causal DAG.

**Architecture:** The graph compiler remains unchanged in meaning and exposes the currently ready action antichain. The controller derives an immutable completion hint from that certified graph when dispatching the canonical primary occurrence. The pi0.5 executor may use the union of ready actions' effects as its stopping condition only when every action renders to the exact same policy prompt; ownership, effect-gating, commit, retry lineage, and recovery remain attached solely to the primary occurrence. Task 8's nominal prompt is restored byte-for-byte to the official task language.

**Tech Stack:** Python 3.11, typed STRIPS/PDDL, pytest, NetworkX-free immutable DAG model, OpenPI pi0.5 policy server, LIBERO interactive simulator, JSON experiment manifests.

## Global Constraints

- Keep the task-8 action layer width at two; never add a synthetic `place-both` schema or an edge between independent `place-on` occurrences.
- Keep `max_total_policy_steps=520`, `replan_steps=5`, task IDs, seeds, simulator initialization, and policy server process paired with Base.
- Do not implicitly commit a sibling occurrence. A fresh snapshot and the ordinary precondition/repair path must account for incidental sibling completion.
- Do not let prompts, VLM output, or runtime heuristics introduce graph edges or execution authorization.
- Preserve fail-closed STOPPED, settling, post-action grounding, VAL, RetryPolicy, and budget behavior.
- Record the actual completion mode, actions, and literals in every attempt artifact.

### Task 1: Expose the certified ready action frontier

**Files:**

- Modify: `src/pi05_libero_repro/logiv/dag.py`
- Test: `tests/logiv/test_dag.py`

**Step 1: Write failing tests**

Add tests showing that the task-8 graph initially returns both independent `place-on` occurrence IDs in canonical-rank order, while a drawer graph returns only the first action because its later actions have action predecessors. Add a test showing that committing one task-8 occurrence removes it but leaves its sibling ready.

**Step 2: Verify RED**

Run `pytest -q tests/logiv/test_dag.py` and confirm failure because `ready_action_ids` does not exist.

**Step 3: Implement minimally**

Add `CausalGraph.ready_action_ids(committed: AbstractSet[str]) -> tuple[str, ...]`. An action is ready when it is uncommitted and every incoming predecessor whose node kind is `ACTION` is committed. `INIT` is already satisfied; `GOAL` is never returned. Sort by canonical agenda rank, not dictionary order.

**Step 4: Verify GREEN and commit**

Run `pytest -q tests/logiv/test_dag.py`, then commit the focused change.

### Task 2: Carry a graph-derived completion hint through dispatch

**Files:**

- Modify: `src/pi05_libero_repro/logiv/controller.py`
- Modify: `tests/logiv/test_controller.py`

**Step 1: Write failing controller tests**

Introduce an immutable `ExecutionCompletionHint` containing ordered occurrence IDs and grounded actions. Extend the fake dispatcher to record the hint. Assert that task 8's first dispatch receives two ready actions, a serial drawer plan receives one, and the controller still advances/commits only the primary occurrence.

**Step 2: Verify RED**

Run the targeted controller tests and confirm the missing keyword/type failure.

**Step 3: Implement minimally**

At the atomic dispatch gate, derive committed IDs from the canonical prefix, call `ready_action_ids`, resolve the action payloads, and pass `completion_hint=` to the dispatcher. The current occurrence remains the canonical cursor occurrence and is first in the hint. The hint is advisory and certificate-derived.

**Step 4: Verify GREEN and commit**

Run `pytest -q tests/logiv/test_controller.py`, then commit.

### Task 3: Implement prompt-safe frontier completion in the pi0.5 executor

**Files:**

- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `scripts/eval_logiv_libero.py`
- Modify: `tests/logiv/test_libero_adapter.py`

**Step 1: Write failing executor tests**

Create a two-action hint whose primary effect becomes true before the sibling effect. With identical rendered prompts, assert that execution does not stop on the primary effect and later stops with `observed frontier effects` after the union is stable. Assert that differing prompts or a one-action hint retain `observed declared effects`. Assert that audit output includes completion mode, occurrence/action list, and union positive/negative literals.

**Step 2: Verify RED**

Run the new tests and confirm that the executor currently stops on the primary effect and lacks audit fields.

**Step 3: Implement minimally**

Extend queued attempt metadata and `AttemptResult`. In `consume_permit_and_enqueue`, accept an optional hint and render each action. Select `DAG_FRONTIER` only for at least two actions with byte-identical prompts; otherwise select `OCCURRENCE`. For frontier mode, compute union Add/Del literals and use the fresh online symbolic snapshot for completion confirmation. Preserve the existing primary target-divergence, safety, global budget, STOPPED, and settling logic. Update `_attempt_json` to serialize the audit fields.

**Step 4: Verify GREEN and commit**

Run `pytest -q tests/logiv/test_libero_adapter.py tests/logiv/test_controller.py`, then commit.

### Task 4: Restore the official nominal task-8 prompt exactly

**Files:**

- Create: `configs/logiv/prompts/pi05-subtasks-v13.json`
- Modify: `tests/logiv/test_libero_adapter.py`

**Step 1: Write the failing prompt contract**

Assert that both nominal task-8 `place-on` actions render exactly `put both moka pots on the stove`, including lowercase and absence of trailing punctuation. Assert that recovery overrides remain action/fact-specific.

**Step 2: Verify RED**

Run the prompt test against the new version path and confirm it is absent.

**Step 3: Add the minimal configuration**

Create v13 from the validated v10 structure, change only the version and both nominal task-8 overrides to the exact official string, and retain targeted recovery prompts.

**Step 4: Verify GREEN and commit**

Run the prompt tests, validate JSON parsing, then commit.

### Task 5: Run regression verification

**Files:**

- Test: `tests/logiv/`

**Step 1: Run focused suites**

Run `pytest -q tests/logiv/test_dag.py tests/logiv/test_controller.py tests/logiv/test_libero_adapter.py`.

**Step 2: Run the full LOGIV suite**

Run `pytest -q tests/logiv` and record the exact pass/fail count.

**Step 3: Run static/config checks**

Run the repository's LOGIV lint/type/config commands discovered from project metadata. Check `git diff --check` and inspect the full diff for unintended changes.

### Task 6: Development-seed paired experiment and root-cause loop

**Files:**

- Runtime artifacts: `runs/logiv-task8-base-sameserver-diag5/`
- Runtime artifacts: `runs/logiv-task8-frontier-v13-dev5/`
- Modify only after evidence: implementation/config/tests above

**Step 1: Run the locked candidate on exposed development seeds**

Using the already-running policy server, run Full LOGIV v13 on task 8 seeds `2,5,7,8,14`, `max_action_steps=520`, global policy steps `520`, replan interval `5`, and existing effect confirmation. Preserve initial-state and first-frame hashes.

**Step 2: Audit pairing and graph structure**

Programmatically verify exact seed/hash pairing with Base and assert every candidate graph still has action-layer width two and no edge between the two nominal actions.

**Step 3: Apply the development gate**

Require candidate success strictly greater than the same-server Base result `3/5`. If it fails, inspect receipts, step budgets, stop reasons, grounded facts, and prompt audit before changing code. The next allowed hypothesis is a training-demonstration-derived milestone/recovery-reserve policy, implemented with new failing tests and equal total budget; do not expand to holdout.

**Step 4: Expand development only after the gate**

Run paired seeds `0--19` on the same server process and require Full to exceed Base with exact hashes. Freeze code/config immediately after the best predeclared candidate passes.

### Task 7: Unseen holdout, task-8 full run, and all-task expansion

**Files:**

- Runtime artifacts: new immutable run directories
- Modify: experiment manifests/reporting documentation only after measurements

**Step 1: Run untouched paired holdout**

Run task 8 episodes `20--49` for frozen Full and Base on the same policy server and equal budgets. Do not inspect or tune on these seeds before candidate freeze.

**Step 2: Apply the holdout gate**

Require Full holdout success strictly above Base. Compute paired differences and Wilson intervals; retain every allocated episode and failure in the denominator.

**Step 3: Complete task 8 and apply the acceptance gate**

Merge the immutable `0--19` development and `20--49` holdout logs only when protocol/config hashes match. Require Full at least `32/50` and greater than Base. If the gate fails, return to a new development split/hypothesis without contaminating the reported holdout.

**Step 4: Run the ten-task protocol only after task-8 success**

Run the frozen Full arm across all ten manifest task IDs and matched Base comparison using identical server/checkpoint, seeds, budgets, and evaluator contract. Report task-wise `x/N`, Wilson intervals, paired macro-average bootstrap, recovery metrics, failure taxonomy, latency, VAL calls, and graph width. Do not label planned or partial episodes as completed results.

### Task 8: Final independent review and evidence handoff

**Files:**

- Review: all changed source/tests/config/docs
- Create or modify: reproducible experiment summary only from immutable artifacts

Run `superpowers:requesting-code-review` over the completed diff, resolve technically valid findings with TDD, then run `superpowers:verification-before-completion`. The handoff must distinguish code correctness, development evidence, unseen holdout evidence, and full ten-task evidence; include exact commands, run paths, commit hashes, sample counts, and any remaining limitations.
