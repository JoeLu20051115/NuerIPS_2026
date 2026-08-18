# LOGIV PDDL Planner and Simulator-VLM Design

**Date:** 2026-08-12

**Method source:** `LOGIV_method.pdf`

**Status:** approved for implementation

**Evaluation label:** simulator-backed development smoke test, not VLM evidence

## Goal

Bring the implementation back to the responsibility split specified by the
method PDF:

1. a symbolic PDDL planner produces the initial plan;
2. the same planner performs bounded local replanning after an explicit Gate
   failure;
3. VAL validates every complete plan before installation;
4. deterministic program code builds and atomically installs the causal DAG;
5. GPT-4o is only a three-valued visual fact observer.

The first end-to-end run uses the simulator-state adapter in place of GPT-4o so
that it makes zero OpenAI API calls. After this smoke test, changing only the
perception backend to `gpt4o` activates the real VLM path.

## Fixed method boundaries

### PDDL planner

The planner receives only the fixed Domain, registered objects, current
TRUE/FALSE facts, immutable final goal, legal action schemas, retry exclusions,
and (for repair) the affected causal slice. It returns an ordered sequence of
ground PDDL actions.

The planner may not create an object or action schema, alter the final goal,
construct DAG edges, emit low-level robot controls, or consume camera images.
Initial planning and repair use the same planner implementation. Existing
programmatic search over `FixedDomain` is retained as the local PDDL search
backend; no external planner dependency is added for the smoke test.

### VAL and DAG compiler

Every candidate plan is rendered and passed through the existing independent
VAL wrapper. Only a valid certificate permits deterministic causal-DAG
construction. During repair, the completed and unaffected portion of the old
execution is preserved, the new suffix is merged, the complete resulting plan
is validated, and the new graph is installed atomically.

### Visual fact observer

The observer receives current camera evidence and a closed finite list of
registered PDDL facts. It may return only `TRUE`, `FALSE`, or `UNKNOWN` for each
fact. It does not decide whether an episode or node failed, propose an action,
repair a plan, edit a goal, or construct a graph.

The production `gpt4o` backend keeps this closed-question contract and uses the
request purpose `state_gate` only. The development `scripted-oracle` backend is
the simulator-state test double for the same grounding interface. It is
explicitly reported as simulator/oracle evidence and cannot be presented as a
GPT-4o or real-VLM result.

## Runtime flow

1. Reset the simulator and capture the initial observation.
2. Run the State Gate through the selected fact-observer interface.
3. Construct the current PDDL Problem from registered objects, confirmed facts,
   and the frozen task goal.
4. Let the PDDL planner search the initial action sequence.
5. Validate the complete plan with VAL and compile the DAG deterministically.
6. Before dispatch, confirm the first node's preconditions through the State and
   Node Gates.
7. Execute only the READY node using the frozen low-level policy.
8. After the node, confirm its effects; `UNKNOWN` requests fresh evidence and
   never becomes `FAIL` by assumption.
9. On the first explicit failure, identify the affected downstream slice and
   locally replan from the current reliable symbolic state.
10. VAL-check the complete merged plan, rebuild the DAG, and install it
    atomically before further execution.
11. Continue State, Node, and Graph monitoring at every node boundary until the
    native simulator evaluator reports success or an existing bounded safe-stop
    condition is reached.

## Initial-plan inputs

The existing scripted proposal fixture remains only a task scaffold for the
official goal, registered objects, task instruction, and optional search
ranking. Its candidate action list is not accepted as a certified plan and is
not attributed to a VLM. The PDDL planner must derive and validate the installed
plan from the grounded current state. A later refactor may replace the fixture
metadata with a RoboTwin task registry without changing planner or observer
interfaces.

## Failure and UNKNOWN behavior

- A required `UNKNOWN` fact blocks dispatch or completion and causes bounded
  re-observation with fresh evidence.
- Only an explicit `FALSE` required precondition/effect or a graph inconsistency
  enters repair.
- If facts remain unresolved, planning exceeds its bounds, VAL fails, or repair
  budgets are exhausted, the controller follows its existing safe terminal
  path; it does not guess.
- A failed validation never partially replaces the current graph.
- Native simulator success remains absorbing.

## Test strategy

Implementation is test-first. Focused tests must prove:

- the initial installed plan is returned by the PDDL planner rather than
  GPT-4o or a scripted proposal being accepted directly;
- initial planning and local repair use the same symbolic planner boundary;
- the GPT-4o planning and repair adapters are unreachable from evaluator and
  shadow-runtime paths;
- simulator-backed smoke execution makes exactly zero OpenAI calls;
- `gpt4o` accounting contains `state_gate` calls only;
- unknown facts block execution and cannot trigger repair;
- every installed initial or repaired graph has a VAL certificate;
- the deterministic compiler, current goal, registered objects, retry limits,
  and frozen low-level policy remain unchanged.

## Smoke-test protocol

First run unit and integration tests without a simulator. Then run one existing
LOGIV simulator task with one fixed seed and `scripted-oracle` perception. This
is only a plumbing test: acceptance is a clean episode terminal record, valid
initial VAL certificate and graph, State/Node/Graph Gate activity from the
start, no uncaught exception, and zero GPT-4o API requests. Task success is
recorded but is not required to establish that the control path runs.

The later RoboTwin baseline is a separate experiment: ten declared RoboTwin 2.0
tasks, ten distinct simulator seeds per task, and the selected frozen pi0.5
checkpoint. Its exact command is confirmed immediately before execution and its
results are not mixed with this LOGIV smoke test.

## Acceptance criteria

- No active runtime imports or constructs `Gpt4oProposalProvider` or
  `Gpt4oRepairOperator`.
- The planner owns initial planning and local replanning.
- GPT-4o is callable only from the fact-grounding boundary and only under the
  explicit `gpt4o` perception backend.
- The simulator-backed end-to-end smoke run reports zero OpenAI calls.
- Initial and repaired plans must pass VAL before deterministic DAG install.
- Existing unrelated experiment files and user worktree changes are preserved.
