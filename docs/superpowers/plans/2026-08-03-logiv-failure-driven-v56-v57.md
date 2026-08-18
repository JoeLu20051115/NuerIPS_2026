# LOGIV Failure-Driven v56/v57 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the identified Task 1--5 negative flips without regressing protected positive seeds, while preserving fail-closed execution and non-chain causal DAGs wherever independent actions exist.

**Architecture:** Route each failure class to the smallest matching intervention. Prompt routing remains task/phase-specific, Task 3/4 observation instability is first isolated with a longer settling diagnostic and only then promoted to bounded same-attempt re-observation if supported, and Task 5 initial execution is represented as the existing `place-in` backbone macro while fine-grained actions remain available for recovery.

**Tech Stack:** Python 3, pytest, typed STRIPS/PDDL configs, LIBERO interactive simulator, frozen full pi0.5 policy servers.

## Global Constraints

- Keep `METADATA_ASSISTED`, `NOMINAL`, `development-only oracle grounding`, paired episode states, and the full pi0.5 checkpoint fixed during development comparisons.
- A candidate is rejected if any protected success seed becomes a failure.
- `UNKNOWN` facts never satisfy gates; persistent post-stop uncertainty remains fail-closed.
- A same-attempt re-observation must not create a policy dispatch, occurrence, attempt, receipt, retry context, or cursor advance.
- Do not add precedence edges to inflate DAG width. Tasks with independent certified actions must retain graph width at least two; a genuine single macro occurrence may have width one.
- Development hard seeds and 50-seed tuning results are not independent final evidence. Freeze the selected version before confirmation with a new policy master seed.

---

### Task 1: Isolate Task 3/4 Post-Stop Observation Timing

**Files:**
- Read: `docs/experiments/2026-08-03-logiv-v53-failure-seed-retrospective.md`
- Read: `scripts/run_logiv_eval.sh`
- Output: existing experiment JSONL directory selected by the evaluation script

**Interfaces:**
- Consumes: v55 prompt config and v53 proposal/coverage configs.
- Produces: paired hard-seed outcomes at `settling_steps=30` for Task 3 seeds `2,3,7,12,21,24,43` and Task 4 seeds `14,41,44,45`.

- [ ] **Step 1: Run the isolated timing diagnostic**

Run Full LOGIV with the standard v53 flags, `configs/logiv/prompts/pi05-subtasks-v55-routed-context.json`, and only change `--settling-steps 30`.

- [ ] **Step 2: Verify the hypothesis from receipts**

For each seed, record success, terminal reason, attempt count, post-stop grounding status, initial-state hash, and first-frame hash. The timing hypothesis is supported only if at least one previous post-stop grounding negative flip is fixed and all protected seeds remain successful.

- [ ] **Step 3: Reject or promote**

If unsupported, keep production settling unchanged and investigate the exact grounded variable transition. If supported, implement Task 2 using TDD instead of globally setting every attempt to 30 settling steps.

### Task 2: Add Bounded Same-Attempt Re-observation Only If Task 1 Supports It

**Files:**
- Modify: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Modify: `scripts/eval_logiv_libero.py`
- Test: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: a `GroundingError` from the first post-stop snapshot while the executor still owns the same attempt.
- Produces: either one reliable replacement snapshot for the same attempt or the existing post-stop grounding failure.

- [ ] **Step 1: Write a failing UNKNOWN-to-reliable test**

Add a fake environment trace whose first post-stop snapshot violates an exactly-one state and whose bounded hold-step snapshot is reliable. Assert that the same attempt returns the reliable snapshot, exactly one policy dispatch occurred, and no new attempt ID was created.

- [ ] **Step 2: Run the test and verify RED**

Run `uv run pytest -q tests/logiv/test_libero_adapter.py -k post_stop_reobservation`. It must fail because bounded re-observation is not implemented.

- [ ] **Step 3: Write a failing persistent-UNKNOWN test**

Add a trace whose first and bounded replacement snapshots remain unreliable. Assert that the result contains the original fail-closed post-stop error and never commits effects.

- [ ] **Step 4: Run both tests and verify RED**

Run `uv run pytest -q tests/logiv/test_libero_adapter.py -k post_stop_reobservation`. Both tests must fail for the missing behavior, not from fixture errors.

- [ ] **Step 5: Implement the minimum same-attempt retry**

Add one optional bounded hold-step count to `Pi05MacroExecutor`; on the first post-stop `GroundingError`, advance only zero/hold actions inside the existing attempt, call `peek_snapshot` once more, and preserve the existing error if the second snapshot is unreliable. Expose a CLI integer defaulting to zero.

- [ ] **Step 6: Verify GREEN and regression suite**

Run the targeted test, then `uv run pytest -q tests/logiv/test_libero_adapter.py`.

### Task 3: Align Task 5 Initial Plan With the Existing pi0.5 Macro Skill

**Files:**
- Create: `configs/logiv/libero10-coverage-v56-task5-place-in.json`
- Create: `configs/logiv/libero10-scripted-proposals-v56-task5-place-in.json`
- Create: `configs/logiv/prompts/pi05-subtasks-v56-task5-place-in.json`
- Modify: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: existing fixed-domain `place-in` schema, Task 5 registered objects, and v55 configs.
- Produces: one certified Task 5 initial occurrence `place-in(black_book_1, study_table_black_book_init_region, desk_caddy_1_back_contain_region, desk_caddy_1_access)`; recovery schemas remain available.

- [ ] **Step 1: Write the failing configuration integration test**

Load the three not-yet-created v56 configs through `ScriptedProposalProvider`, build the Task 5 initial problem, run schema/type/object checks plus certified repair/validation, and assert the resulting grounded sequence is exactly the single `place-in` macro.

- [ ] **Step 2: Write graph and isolation assertions**

Compile the certified Task 5 plan and assert it contains one executable occurrence plus `INIT`/`GOAL`, with support metadata derived from effects. Also assert Task 0--4 and Task 6--9 proposals are byte-for-byte equivalent to the parent v53 proposal output.

- [ ] **Step 3: Run the new tests and verify RED**

Run `uv run pytest -q tests/logiv/test_libero_adapter.py -k 'v56 or task5_place_in'`. It must fail because the v56 config files do not exist.

- [ ] **Step 4: Add the minimum config overlays**

Extend v53/v55 configs, add `place-in` to Task 5 supported initial schemas without removing recovery schemas, replace only Task 5's candidate sequence, and bind the occurrence prompt to the official full-task instruction.

- [ ] **Step 5: Verify GREEN and domain lint**

Run the targeted tests, the complete adapter test file, and the repository's PDDL/domain lint tests.

### Task 4: Hard-Seed Simulation Gates

**Files:**
- Output: experiment JSONL and summary artifacts only

**Interfaces:**
- Consumes: Tasks 1--3 candidates.
- Produces: hard-seed guard tables for Task 3/4/5.

- [ ] **Step 1: Run Task 5 representative screen**

Evaluate negatives `5,8,12,14,15,41,49` and guards `0,2,3,7,28,38,45` with v56 configs. Reject on any guard regression.

- [ ] **Step 2: Expand Task 5 to all known flips**

If the screen passes, evaluate all known positive and negative flip seeds `0,2,3,5,7,8,12,13,14,15,16,19,20,27,28,29,31,38,41,45,46,49`.

- [ ] **Step 3: Run Task 3/4 hard gates for a promoted timing fix**

Evaluate Task 3 `2,3,7,12,21,24,43` and Task 4 `14,41,44,45`; reject on any protected seed regression.

### Task 5: Development 50-Pair and Independent Confirmation

**Files:**
- Update: `docs/experiments/2026-08-03-logiv-v53-failure-seed-retrospective.md`
- Output: frozen run manifest, per-episode receipts, task summaries, paired bootstrap table

**Interfaces:**
- Consumes: only candidates that pass Task 4.
- Produces: development 50-pair results followed by a separately labeled independent confirmation run.

- [ ] **Step 1: Run affected-task 50-pair development evaluation**

Use identical initial-state and first-frame hashes for Full/Base. Require `Full >= Base` for every evaluated task before scheduling the ten-task matrix.

- [ ] **Step 2: Verify causal graph structure**

Report executable-node count, edge reasons, maximum ready frontier, and width for every episode. Reject any compiler change that adds unsupported precedence solely to create a chain or width.

- [ ] **Step 3: Freeze the best development checkpoint**

Commit only passing code/config/tests/docs and create an immutable git tag. Record rejected candidates separately; never overwrite the prior v53 tag.

- [ ] **Step 4: Run independent confirmation**

Change the policy master seed while retaining paired environment states. Report per-task `x/50`, Wilson intervals, paired flips, macro paired difference, and 10,000-repeat within-task paired bootstrap interval. Keep development and confirmation tables separate.

## Self-Review

- Spec coverage: prompt routing, bounded grounding recovery, Task 5 macro alignment, graph non-degeneration, hard-seed gates, development/confirmation separation, and rollback are covered.
- Placeholder scan: no deferred implementation fields or unspecified tests remain.
- Type consistency: Task 2 stays inside one `Pi05MacroExecutor` attempt; Task 3 uses the existing `place-in` grounded action shape and existing config inheritance.
