# LOGIV LIBERO Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a VAL-certified, fact-gated LOGIV controller around the frozen π₀.₅ policy on all ten LIBERO-10 tasks, with real causal DAG branches, bounded recovery, and auditable Scripted-VLM proposal artifacts.

**Architecture:** Add a focused `pi05_libero_repro.logiv` package. A single typed STRIPS model renders the public PDDL Domain and drives local search/DAG compilation; external VAL alone grants certificates. A synchronous controller consumes versioned proposals and fresh fact snapshots, while a LIBERO adapter supplies oracle grounding and maps high-level occurrences to frozen π₀.₅ subtask prompts.

**Tech Stack:** Python 3.11 repository tooling, Python 3.8 LIBERO Docker runtime, standard library dataclasses/JSON/subprocess/hashlib, NumPy, pytest, VAL 4, pinned OpenPI/LIBERO, MuJoCo, and the existing websocket policy client.

## Global Constraints

- Preserve the pinned OpenPI commit `650c5b0283a49c42784fb5055a0507da2c6d347d` and LIBERO commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Do not modify checkpoint parameters, official BDDL files, native success predicates, initial states, or the existing open-loop evaluator semantics.
- Use the real `/home/xingrui/.local/bin/Validate`; local STRIPS execution may prune candidates but may not issue authorization.
- Keep `VALIDATION_ERROR` distinct from `INVALID` and fail closed on grounding, fence, compiler, certificate, or budget errors.
- Persist first-pass subtasks as `candidate.plan` plus `proposal.json`, `initial_problem.pddl`, and an occurrence sidecar.
- Never add linear-plan adjacency edges. Task 8 must contain two unordered `place-on` occurrences and a DAG layer width of at least two.
- In the no-API phase, label results `scripted-vlm/oracle-grounding`; do not claim measured VLM perception improvement.
- Use the first 20 initial states per task for prompt development and lock states 20–49 until the prompt is fixed.
- Every feature and bug fix follows observed RED → GREEN; all existing tests remain green.

---

## File Map

- `src/pi05_libero_repro/logiv/model.py`: immutable facts, actions, proposals, snapshots, occurrences, certificates, receipts, graphs, and terminal enums.
- `src/pi05_libero_repro/logiv/domain.py`: fixed typed STRIPS schemas, grounding, local transition, lint, and deterministic PDDL rendering.
- `src/pi05_libero_repro/logiv/proposal.py`: ten Scripted-VLM task proposals and proposal/Problem/plan/sidecar artifact writer.
- `src/pi05_libero_repro/logiv/val.py`: fail-closed VAL subprocess wrapper and certificate hashing.
- `src/pi05_libero_repro/logiv/dag.py`: causal support/conflict compiler, stable topological agenda, and causal backtracking.
- `src/pi05_libero_repro/logiv/repair.py`: bounded deterministic edit/search, trace obligations, and retry-key filtering.
- `src/pi05_libero_repro/logiv/controller.py`: attempt lifecycle, gates, atomic state transitions, budgets, retries, repair, and graph installation.
- `src/pi05_libero_repro/logiv/libero_adapter.py`: BDDL task mapping, oracle facts, synchronous stopped fence, and π₀.₅ macro-attempt loop.
- `src/pi05_libero_repro/logiv/prompts.py`: versioned proposal and π₀.₅ subtask prompt templates.
- `src/pi05_libero_repro/logiv/records.py`: durable LOGIV episode/event records and comparison summaries.
- `configs/logiv/logiv-libero-domain.pddl`: generated public Domain.
- `configs/logiv/libero10-scripted-proposals.json`: generated ten-task Scripted-VLM fixture.
- `scripts/eval_logiv_libero.py`: real simulator evaluator using the existing policy server.
- `scripts/report_logiv_results.py`: baseline/LOGIV report with task rates and controller metrics.
- `tests/logiv/`: behavior-first unit, integration, and real-VAL tests.

### Task 1: Typed STRIPS Model and Fixed Domain

**Files:**
- Create: `src/pi05_libero_repro/logiv/__init__.py`
- Create: `src/pi05_libero_repro/logiv/model.py`
- Create: `src/pi05_libero_repro/logiv/domain.py`
- Create: `configs/logiv/logiv-libero-domain.pddl`
- Test: `tests/logiv/test_domain.py`

**Interfaces:**
- Produces: `Fact(predicate: str, arguments: tuple[str, ...])`, `GroundAction(schema, arguments, preconditions, add_effects, del_effects, repeatable)`, `ObjectDecl`, `TaskProblem`, `FixedDomain.ground(...)`, `apply_action(state, action)`, `render_domain_pddl()` and `render_problem_pddl(problem)`.
- Consumers: Proposal, VAL, DAG, Repair, and Controller tasks.

- [ ] **Step 1: Write failing domain behavior tests**

Create literal tests proving: task-8 `place-on` actions move only their named object; `place-in` requires an open access; `place-held-in` requires `holding(object)` and establishes `handempty`; wrong types and exactly-one violations fail; rendered PDDL contains all fixed schemas and no task-specific object IDs.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_domain.py -v`. Expect collection failure because `pi05_libero_repro.logiv.domain` does not exist.

- [ ] **Step 3: Implement immutable model and minimal fixed schemas**

Use frozen dataclasses and `frozenset[Fact]`. Define schemas `place-on`, `place-in`, `place-relative`, `open-access`, `close-access`, `turn-on`, `turn-off`, `put-down`, `place-held-on`, `place-held-in`, and `place-held-relative`. Make parameter binding type-checked against `TaskProblem.objects`. Keep `handempty` unchanged across successful nominal macro placements; held recovery actions delete `holding(object)` and add `handempty`.

- [ ] **Step 4: Generate public PDDL and verify GREEN**

Render `configs/logiv/logiv-libero-domain.pddl` deterministically from `FixedDomain`; run the test file and require all tests pass.

- [ ] **Step 5: Commit**

Commit with `feat: add fixed LOGIV STRIPS domain`.

### Task 2: Ten Scripted-VLM Proposals and PDDL-like Artifacts

**Files:**
- Create: `src/pi05_libero_repro/logiv/proposal.py`
- Create: `configs/logiv/libero10-scripted-proposals.json`
- Test: `tests/logiv/test_proposal.py`

**Interfaces:**
- Consumes: Task 1 model/domain rendering.
- Produces: `ScriptedProposalProvider.propose(task_id, epoch_id) -> Proposal` and `write_proposal_artifacts(directory, proposal, domain) -> ProposalArtifacts`.

- [ ] **Step 1: Write failing artifact tests**

Use task 3 and task 8 literal fixtures. Assert task 3 candidate plan is `place-in` then `close-access`; task 8 contains two `place-on` actions with distinct moka-pot IDs. Run the writer and assert `proposal.json`, `initial_problem.pddl`, `candidate.plan`, and `candidate.occurrences.json` parse and agree on action order and occurrence IDs. Reject unknown task IDs and schema/object/type drift.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_proposal.py -v`; expect import failure.

- [ ] **Step 3: Implement all ten proposals**

Translate official BDDL registered objects, relevant initial relations, access state, device state, original goals, and candidate macro actions into the fixture. Include evidence strings that identify `task_instruction`, `initial_image_review`, or `official_registered_object_metadata`; do not label BDDL Goal as visual evidence.

- [ ] **Step 4: Render and verify all proposals**

Generate the committed JSON fixture, round-trip all ten proposals, ground every candidate action, and require `uv run pytest tests/logiv/test_proposal.py -v` to pass.

- [ ] **Step 5: Commit**

Commit with `feat: add LIBERO scripted VLM proposals`.

### Task 3: Real VAL Wrapper and Bound Certificates

**Files:**
- Create: `src/pi05_libero_repro/logiv/val.py`
- Test: `tests/logiv/test_val.py`

**Interfaces:**
- Consumes: rendered Domain/Problem/Plan and occurrence sidecar bytes.
- Produces: `ValidationStatus`, `ValidationResult`, `PlanCertificate`, `ValWrapper.validate(...)`, and `verify_certificate(...)`.

- [ ] **Step 1: Write failing real-process tests**

Build tiny valid and invalid plans from the task-3 proposal. Assert real VAL returns `VALID` for the nominal plan and `INVALID` when `close-access` precedes `place-in`. Use a temporary fake executable for timeout, nonzero unrecognized output, and malformed output; all must return `VALIDATION_ERROR`. Mutating any bound byte or retry-ledger version must fail certificate verification.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_val.py -v`; expect import failure.

- [ ] **Step 3: Implement fail-closed wrapper**

Invoke `subprocess.run` with explicit argv, captured UTF-8 output, no shell, and timeout. Recognize VAL success only from exit zero plus its success marker; recognize ordinary invalid plans only from known plan-failure markers. Hash length-prefixed named payloads to prevent concatenation ambiguity.

- [ ] **Step 4: Verify GREEN against installed VAL**

Run the test file and `/home/xingrui/.local/bin/Validate -h`; record version text in the certificate configuration.

- [ ] **Step 5: Commit**

Commit with `feat: add fail-closed VAL certification`.

### Task 4: Causal DAG Compiler Without Adjacency Edges

**Files:**
- Create: `src/pi05_libero_repro/logiv/dag.py`
- Test: `tests/logiv/test_dag.py`

**Interfaces:**
- Consumes: certified plan, Initial State, Goal, fixed action semantics, and verified certificate.
- Produces: `CausalGraph`, `CausalEdge`, `compile_graph(...)`, `canonical_agenda(...)`, `topological_width(...)`, and `causal_slice(...)`.

- [ ] **Step 1: Write failing graph tests**

Assert task 8 compiles as `INIT → place(moka1) → GOAL` and `INIT → place(moka2) → GOAL` with no edge between placements and width two. Assert task 3 has necessary `place-in → close-access` conflict protection because closing deletes the open fact needed by placement. Assert multiple facts between one node pair merge into one edge, canonical order follows certified indices only as a tie break, and cycles/self-loops/reverse-rank edges are rejected.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_dag.py -v`; expect import failure.

- [ ] **Step 3: Implement support provenance and threat protection**

Track the latest unthreatened producer for each action precondition and Goal fact. Add protection precedence only when a deleter would otherwise threaten a causal link. Store all reasons on one edge. Never iterate adjacent plan pairs to create edges.

- [ ] **Step 4: Verify GREEN and mutation property**

Run graph tests, then temporarily reason-check that adding an adjacency edge would fail task-8 width/no-edge assertions.

- [ ] **Step 5: Commit**

Commit with `feat: compile partial-order causal DAGs`.

### Task 5: Bounded Repair, Trace, and Retry Policy

**Files:**
- Create: `src/pi05_libero_repro/logiv/repair.py`
- Test: `tests/logiv/test_repair.py`

**Interfaces:**
- Consumes: Current Problem, original Goal, remaining plan, causal slice, grounded action catalog, budgets, retry ledger, and ValWrapper.
- Produces: `FailureObligation`, `TraceKind`, `trace_invalid_plan(...)`, `RetryPolicy`, and `RepairOperator.repair(...) -> RepairResult`.

- [ ] **Step 1: Write failing recovery tests**

Cover literal bowl states: closed drawer before placement inserts `open-access`; dropped bowl rebuilds a placement from its current location; held bowl selects `place-held-in`; final open drawer yields only `close-access`; forbidden retry keys reject renamed equivalent occurrences. Assert trace emits exactly the earliest action failure or final goal failure and bounded exhaustion says `NO_CERTIFIED_REPAIR_WITHIN_BUDGET`, not `NO_SOLUTION`.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_repair.py -v`; expect import failure.

- [ ] **Step 3: Implement deterministic bounded search**

Enumerate applicable grounded actions from the fixed task catalog, prune with local STRIPS transition, deduplicate `(state, remaining-goal, forbidden-keys)` keys, and order candidates by preserved goals, edit distance, plan length, old rank, then lexical grounded action. Call real VAL for every installable full plan.

- [ ] **Step 4: Verify GREEN and budgets**

Run repair tests and assert VAL call counters match literal expectations for zero-edit vs edited paths.

- [ ] **Step 5: Commit**

Commit with `feat: add bounded certified repair`.

### Task 6: Fact-Gated Controller and Attempt Lifecycle

**Files:**
- Create: `src/pi05_libero_repro/logiv/controller.py`
- Test: `tests/logiv/test_controller.py`

**Interfaces:**
- Consumes: proposal/certificate/graph, `FactGrounder.ground`, `AttemptExecutor.execute`, RepairOperator, RetryPolicy, and budgets.
- Produces: `LogivController.run() -> ControllerResult` plus immutable event/receipt history.

- [ ] **Step 1: Write failing state-machine tests**

Use synchronous real fakes, not assertion-only mocks. Test normal commit; stale receipt cannot authorize; precondition failure skips suffix recertification; effect failure remaining plan includes the failed occurrence and recertifies once; goal failure with empty agenda skips empty VAL; no replacement before STOPPED; retries create new attempt IDs but preserve occurrence/lineage; atomic install rejects stale parent graph/epoch; each budget prevents further dispatch.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_controller.py -v`; expect import failure.

- [ ] **Step 3: Implement minimal synchronous controller**

Keep all mutable runtime state in one `ControllerState` owned by the controller. Implement explicit transitions `EXECUTING`, `RUNNING`, `STOPPED_UNCOMMITTED`, `RECOVERING`, `SUCCESS`, and `TERMINAL`. Create attempt IDs only at the dispatch gate; perform receipt/cursor changes in one method after stopped effect verification.

- [ ] **Step 4: Verify GREEN and full symbolic suite**

Run `uv run pytest tests/logiv -q`; require no warnings or failures.

- [ ] **Step 5: Commit**

Commit with `feat: add LOGIV fact-gated controller`.

### Task 7: LIBERO Oracle Grounder, Synchronous Fence, and Prompts

**Files:**
- Create: `src/pi05_libero_repro/logiv/prompts.py`
- Create: `src/pi05_libero_repro/logiv/libero_adapter.py`
- Test: `tests/logiv/test_libero_adapter.py`

**Interfaces:**
- Consumes: existing `prepare_observation`, fixed task proposal, inner LIBERO predicate API, and websocket policy client.
- Produces: `LiberoOracleGrounder`, `Pi05MacroExecutor`, `SubtaskPromptRenderer`, and a complete `AttemptResult` with stopped fence evidence, frames, actions, and post-action snapshot.

- [ ] **Step 1: Write failing adapter tests**

Create a behavior fake matching the real nested `OffScreenRenderEnv.env` shape. Assert BDDL relations map to exactly one location or holding; handempty/holding are mutually exclusive; completion flushes unused policy actions; settling preserves the last gripper command; prompt names exactly one current subtask and explicitly forbids advancing to the next goal; malformed/non-finite actions fail closed.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_libero_adapter.py -v`; expect import failure.

- [ ] **Step 3: Implement adapter with explicit simulator disclosure**

Use synchronous `env.step` return plus local deque flush as the simulator fence. Use an oracle completion detector only to choose the macro boundary and then take one full fresh snapshot for the controller gates. Record detector calls separately so they cannot be mistaken for VLM GroundFacts calls. Settle with zero Cartesian deltas and the last finite gripper command.

- [ ] **Step 4: Verify GREEN and import compatibility**

Run adapter and existing protocol tests. Import the module in the LIBERO Docker/OpenPI environment without contacting the policy server.

- [ ] **Step 5: Commit**

Commit with `feat: integrate LOGIV with LIBERO and pi05`.

### Task 8: Durable Evaluation Records and LOGIV Evaluator

**Files:**
- Create: `src/pi05_libero_repro/logiv/records.py`
- Create: `scripts/eval_logiv_libero.py`
- Create: `scripts/report_logiv_results.py`
- Test: `tests/logiv/test_records.py`
- Test: `tests/logiv/test_evaluator.py`

**Interfaces:**
- Consumes: existing suite/task/reset setup, Controller, proposal provider, policy client, and baseline records.
- Produces: append-only episode JSONL, per-episode artifact directories/videos, invalid terminal record, and Markdown/JSON comparisons.

- [ ] **Step 1: Write failing record/evaluator tests**

Assert unique run/task/episode keys, atomic JSONL append, event hash chain, task-8 graph width metrics, valid terminal-cause enum, prompt/config hashes, and exact initial-state/first-frame pairing with baseline. Run a fake two-subtask episode through the evaluator boundary and assert each occurrence receives its own prompt.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/logiv/test_records.py tests/logiv/test_evaluator.py -v`; expect import/script failure.

- [ ] **Step 3: Implement evaluator and report**

Reuse existing task ordering, initial states, rendering, video naming conventions, and image preparation. Add `--task-ids`, `--episode-indices`, `--prompt-version`, `--max-action-steps`, budgets, `--development-only`, and explicit `--oracle-grounding` acknowledgement. Refuse nonempty unrecognized output directories and any holdout request when the prompt version is unlocked.

- [ ] **Step 4: Verify GREEN and CLI help**

Run all evaluator tests plus both scripts with `--help`.

- [ ] **Step 5: Commit**

Commit with `feat: add auditable LOGIV evaluator`.

### Task 9: Complete Closed-Loop Fault Matrix and Regression Gate

**Files:**
- Create: `tests/logiv/test_closed_loop_scenarios.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior components.
- Produces: executable evidence for the paper's recovery matrix and documented commands.

- [ ] **Step 1: Add end-to-end scenario tests**

Exercise nominal bowl, drawer closed before placement, repeatable effect failure, dropped bowl, held bowl, multi-producer recovery, final-goal reopening, forbidden equivalent retry, grounding failure, validation error, fence failure, stale install, and all budget exhaustion branches. Assert every physical dispatch is preceded by a fresh-fact authorization event and every changed plan by a valid certificate event.

- [ ] **Step 2: Verify tests catch missing authorization**

Run only this file before any integration adjustment; expect failures for any missing controller event wiring, fix the smallest root cause, and repeat until green.

- [ ] **Step 3: Document reproducible commands and claims**

Update README with the scripted/oracle label, VAL prerequisite, artifact layout, smoke commands, development/holdout rule, and statement that successful unit tests do not prove simulator improvement.

- [ ] **Step 4: Run full local verification**

Run `uv run pytest -q`, `git diff --check`, render all ten candidate/Problem pairs, and validate all ten certified nominal plans with real VAL.

- [ ] **Step 5: Commit**

Commit with `test: verify LOGIV closed-loop recovery matrix`.

### Task 10: π₀.₅ Hard-Task Prompt Experiment and Final 10-Task Gate

**Files:**
- Create: `configs/logiv/prompts/pi05-subtask-v1.json`
- Create as runs occur: `runs/logiv-*/` ignored large runtime artifacts
- Create: `results/logiv-development-summary.json`
- Create: `results/logiv-development-summary.md`
- Create after locked evaluation: `results/logiv-libero10-summary.json`
- Create after locked evaluation: `results/logiv-libero10-summary.md`

**Interfaces:**
- Consumes: real full π₀.₅ server, pinned LIBERO evaluator image, task 8/9 development states, then locked states.
- Produces: measured prompt-version comparison and final baseline-vs-LOGIV report.

- [ ] **Step 1: Start fresh full-checkpoint server and run smoke**

Use GPU 1 and a free port. Run task 8 episodes 0–2 with prompt v1. Preserve server/evaluator logs and verify valid actions, stopped fences, fact epochs, VAL certificates, graph width, videos, and native success agreement.

- [ ] **Step 2: Run task 8/9 development states**

Run episodes 0–19 for tasks 8 and 9. Compare against existing baseline by identical initial-state hash. Classify failures from facts/events/video before changing one prompt factor.

- [ ] **Step 3: Iterate prompt only on development states**

Version every change. A prompt candidate advances only if task-8 development success improves without a new terminal-safety failure and task-9 does not regress by more than one episode. After two failures of the same prompt strategy, change the prompt structure rather than wording.

- [ ] **Step 4: Lock prompt and evaluate task 8 all 50 states**

Run a fresh policy server and empty output directory. Require exactly 50 valid records and compare to 27/50. Report improvement only at 32/50 or higher; otherwise continue root-cause-driven development without looking at other task holdouts.

- [ ] **Step 5: Run complete 10×50 and report**

After task-8 acceptance, run all ten tasks from task 0 episode 0 with a fresh uninterrupted server. Require 500 valid records, all native success agreements, and auditable artifacts. Compare against 460/500; state the measured result even if the overall target is not met.

- [ ] **Step 6: Final verification and commit small reports/config only**

Run the full test suite, verify manifests and report cardinality, ensure no videos/logs/keys are staged, and commit prompt configs plus small result summaries with `results: evaluate LOGIV on pi05 LIBERO-10`.

