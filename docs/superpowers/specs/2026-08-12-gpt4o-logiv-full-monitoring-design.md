# GPT-4o LOGIV Full-Monitoring Design

**Date:** 2026-08-12  
**Method source:** `LOGIV_method.pdf`  
**Evaluation label:** development/tuning evidence, not independent holdout evidence

## Goal

Replace the current `scripted-vlm/oracle-grounding` implementation used by
`LOGIV_ONLINE` with GPT-4o at the three model-defined boundaries while keeping
the rest of the LOGIV method unchanged:

1. GPT-4o proposes the initial high-level plan from the initial images, the
   fixed PDDL Domain, the current Problem, registered objects, and legal actions.
2. GPT-4o performs three-valued `TRUE/FALSE/UNKNOWN` visual fact observation
   throughout execution, beginning before the first BASE policy request and
   continuing at the existing monitor cadence.
3. After a strictly confirmed failure, GPT-4o proposes a local modification to
   the affected plan slice. The program merges the proposal with the protected
   plan remainder, VAL validates the complete plan, and the deterministic
   compiler installs a new DAG.

LOGIV monitors for the entire episode. A failure changes control authority; it
does not start monitoring. Before a confirmed failure, BASE retains high-level
execution. After the handoff, LOGIV retains high-level scheduling authority and
continues State, Node, Graph, and Repair Gate monitoring. The frozen pi0.5 policy
continues to generate low-level robot actions in both phases.

## Fixed invariants

The following behavior is outside the tuning surface and must remain unchanged:

- the frozen full pi0.5 checkpoint and policy server;
- the PDDL Domain, registered objects, action schemas, preconditions, effects,
  goals, VAL validation, and deterministic DAG construction;
- the existing State, Node, Graph, and Repair Gate ordering;
- `UNKNOWN` never authorizes repair or node completion;
- BASE actions and policy requests remain identical to BASE before a certified
  handoff, apart from the read-only monitoring latency;
- a handoff discards only the unexecuted suffix of the current action chunk;
- handoff direction is `BASE -> LOGIV`; there is no return to BASE high-level
  scheduling;
- native LIBERO `done=True` remains absorbing success;
- the total 520-low-level-action budget, monitor cadence, confirmation counts,
  intervention thresholds, retry ledger, repair budgets, VAL budgets, prompt
  configuration for pi0.5, simulator reset protocol, and reporting rules;
- no unregistered object, new action schema, goal rewrite, DAG edge proposed by
  a model, or continuous control value may be accepted.

The existing scripted proposal and oracle grounder remain available only as
explicit offline/test fixtures. They are not silently used as a fallback in a
GPT-4o LOGIV run.

## Selected architecture

The implementation adds narrow GPT-4o adapters at existing provider boundaries
instead of rewriting the evaluator or duplicating LOGIV as a second method arm.
This keeps the control loop, evidence contracts, and reporting code shared.

### OpenAI transport

A small transport module calls the OpenAI API using `OPENAI_API_KEY` from the
process environment. The key is never accepted as a command-line argument,
written to an artifact, included in an exception, or logged. The launcher
forwards the existing environment variable into Docker by name.

The configured model is `gpt-4o`. Every request uses strict JSON-schema output,
a bounded network timeout, and bounded retries for transient transport or rate
limit errors. A refusal, malformed response, missing key, unavailable GPT-4o
model, exhausted retry budget, or non-transient API error is explicit evidence;
the implementation does not silently substitute another model.

Each call records only non-secret provenance: request purpose, model returned by
the service, response ID, usage counters, latency, retry count, request-content
SHA-256, response-content SHA-256, episode context, and validation outcome.

### Initial proposal adapter

The initial adapter receives both correctly oriented 224x224 LIBERO camera
views, task instruction, rendered fixed PDDL Domain, rendered current Problem,
registered objects, and the legal grounded action catalog. GPT-4o returns an
ordered list containing only action schema names, registered arguments, and a
short node instruction.

Program code parses and grounds every action through `FixedDomain`; it rejects
unknown schemas, wrong arity or types, unregistered objects, duplicate or empty
plans, and any field outside the schema. The existing certification path then
validates and, within its unchanged bounds, processes the candidate before VAL
and deterministic DAG compilation. GPT-4o never supplies graph edges.

Initial State Gate visual facts are obtained before the initial plan request so
the rendered current Problem reflects the physical initial observation. The
official goal remains frozen from BDDL metadata and is not predicted or edited
by GPT-4o.

### Visual State Gate adapter

At initialization and each existing monitoring point, the adapter asks only
about the finite fact set requested by the controller for the current graph and
context. It supplies the two current camera images, canonical PDDL fact strings,
registered object identifiers, task instruction, epoch, and graph version.

GPT-4o returns exactly one of `TRUE`, `FALSE`, or `UNKNOWN` for every requested
fact plus a short evidence label. Program code rejects extra facts, missing
facts, stale context, conflicting exactly-one facts, or an invalid object or
graph version. Rejected or visually unresolved output becomes `UNKNOWN` and
causes re-observation under the existing Gate policy; it never becomes a
failure by assumption.

The adapter implements the existing grounder interface so Graph projection,
Node checks, strict confirmation, and controller logic remain unchanged. The
strict confirmation read is a fresh GPT-4o request bound to the new image hash
and current graph version, not reuse of the candidate observation.

### Local repair proposal adapter

After a confirmed Node- or Graph-level failure, existing code constructs the
current Problem and causal slice. GPT-4o receives the current images and facts,
the validated old plan, affected occurrence IDs, protected completed results,
unchanged final goal, legal grounded action catalog, retry exclusions, and the
VAL error from the prior rejected candidate when applicable.

GPT-4o returns only a replacement action sequence for the affected slice. The
program preserves unaffected actions and still-valid completed results, merges
the replacement, enforces the existing edit/retry/budget limits, and submits
the complete merged plan to VAL. An invalid candidate is not installed; its VAL
error may be returned to GPT-4o until the unchanged attempt budget is exhausted.
Only a complete valid certificate permits atomic DAG replacement.

## Runtime flow

1. Reset the episode with the existing simulator and policy seeds.
2. Capture the two initial images and run GPT-4o State Gate grounding.
3. Render the current PDDL Problem with those facts.
4. Ask GPT-4o for the initial candidate plan.
5. Ground the candidate, run existing certification and VAL, and compile the
   initial DAG programmatically.
6. Begin BASE execution while LOGIV observes from the first step at the existing
   cadence and projects GPT-4o facts onto the DAG.
7. Continue BASE unchanged while State, Node, and Graph Gates find no strictly
   confirmed failure.
8. On confirmed failure and absent native success, flush the pending old action
   suffix and transfer high-level scheduling to LOGIV.
9. Ask GPT-4o for a local affected-slice change, merge it, validate the complete
   plan with VAL, and atomically install the new DAG.
10. Execute new DAG nodes with the same pi0.5 low-level policy while continuing
    all four Gates until success or an existing safe terminal condition.

## Failure behavior

- Before handoff, monitoring/API failure is recorded and follows the current
  fail-open BASE behavior, but the episode is marked as lacking valid GPT-4o
  monitoring evidence and cannot support a LOGIV performance claim.
- After handoff, a missing or unresolved required fact pauses new dispatch and
  retries observation within the existing limits; exhaustion produces the
  existing safe stop rather than guessed state.
- Initial proposal failure rejects LOGIV initialization for that episode. It
  does not reuse a scripted plan while labeling the run GPT-4o.
- Repair proposal or VAL exhaustion terminates through the existing bounded
  repair status. The old graph is never partially overwritten.
- API responses and images are bound to episode, epoch, phase, graph version,
  and content hashes so a late response cannot authorize a new graph.

## Test strategy

All new behavior is developed test-first with a fake HTTP transport; unit tests
make no paid API calls. Tests cover:

- strict initial-plan JSON parsing and illegal PDDL action rejection;
- two-camera image encoding and orientation;
- complete three-valued fact partitions, `UNKNOWN`, stale context, and
  exactly-one conflicts;
- affected-slice-only repair, protected results, full-plan VAL validation, and
  rejection without atomic installation;
- monitoring from initialization and the first BASE step;
- pre-handoff action/request parity and one-way handoff;
- missing key, timeout, retry, refusal, malformed response, and secret
  redaction;
- Docker environment forwarding without exposing the key value;
- accounting of initial, visual, and repair GPT-4o requests.

The full existing test suite, shell syntax checks, and `git diff --check` must
pass before interactive evaluation.

## Prompt tuning and evaluation

The current reported development baseline contains exactly 1000 paired
episodes: for every task 0 through 9, episode indices 0 through 49 are run once
with master seed 7 and once with master seed 17. BASE has 924 successes and the
current scripted/oracle LOGIV portfolio has 952 successes. The new GPT-4o run
uses the same tasks, episode indices, master seeds, checkpoint, policy seed
derivation, simulator seed derivation, initial states, action budget, and
reporter.

Tuning is restricted to the three GPT-4o prompts and their strict output
schemas. LOGIV thresholds, budgets, pi0.5 prompts, task-specific execution
parameters, and per-episode routing are frozen. The process is:

1. Run an API capability smoke that verifies `gpt-4o`, image input, and strict
   structured output without printing the key.
2. Run the same-seed initial-plan and State Gate contract checks on a small,
   predeclared mix of current BASE successes and failures from all ten tasks.
3. Compare at most three prompt variants using schema-valid fact accuracy
   against the existing oracle only as a development label, initial VAL pass
   rate, false intervention count, and repair VAL pass rate. Select by zero
   negative flips first, then success, then fewer API calls/latency.
4. Run the selected prompt on the complete 1000-pair seed-7/seed-17 protocol.
5. Generate a fresh report without combining GPT-4o episodes with old
   scripted/oracle LOGIV episodes.

The target is approximately 95%, operationalized as at least 950/1000 with zero
negative flips, exact no-trigger action/request parity, no post-success action,
no validation errors, and every episode at or below 520 low-level actions.
Because remote model execution is not guaranteed bitwise deterministic, using
the same simulator and policy seeds controls the embodied evaluation but does
not guarantee exactly 952 successes. A result below 950 is reported honestly
with its failure breakdown; it is not repaired by selecting episodes or mixing
old oracle records.

## Acceptance criteria

- `LOGIV_ONLINE` uses GPT-4o for initial proposal, full-episode visual facts,
  and local affected-slice repair.
- LOGIV monitoring is active before the first BASE policy request and throughout
  both BASE-supervised and LOGIV-controlled execution.
- All fixed invariants above remain enforced by tests and artifacts.
- `OPENAI_API_KEY` is accepted only from the environment and is never persisted.
- The final same-seed report contains 1000 fresh GPT-4o LOGIV pairs and clearly
  labels the evidence as development/tuning.
- The target report reaches at least 950 successes with zero negative flips; if
  it does not, the actual measured result and blockers are preserved.
