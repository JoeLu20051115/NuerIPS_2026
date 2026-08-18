# LOGIV method

LOGIV is a closed-loop execution method for long-horizon robot tasks. It joins
visual state grounding, symbolic planning, formal validation, causal execution,
and bounded repair in one runtime.

## 1. Typed state

The runtime represents the world with typed PDDL objects and three-valued
facts: true, false, and unknown. Every grounded state carries a monotonic epoch
and evidence identity. Contradictory or stale facts are rejected before they
can authorize an action.

## 2. Proposal

An initial proposal contains the task objects, grounded facts, goal facts, and
candidate symbolic actions. The proposal layer validates schema names, object
types, fact coverage, and provenance before constructing a `TaskProblem`.

## 3. Planning and certification

The fixed domain defines manipulation, placement, access, and device actions.
LOGIV renders the current problem and proposed plan, invokes a pinned VAL
binary, and installs only a valid `PlanCertificate`. The certificate binds the
domain, problem, plan, validator, and invocation identity.

## 4. Causal DAG

Certified actions are compiled into a graph containing occurrence nodes,
causal links, ordering edges, and a goal node. The compiler checks acyclicity,
producer compatibility, complete dependencies, unique occurrences, and
agreement with the certificate.

## 5. Grounding gate

Before dispatch, an independent visual gate reads fresh observations and
returns structured facts. A permit is issued only when the required symbolic
preconditions are confirmed for the current state epoch.

## 6. Bounded execution

Each DAG node dispatches a bounded chunk of policy actions. The controller then
stops continuation, obtains a fresh observation, and checks the declared
effects. Success, uncertainty, stale evidence, confirmed failure, and terminal
states follow separate controller paths.

## 7. Monitoring

The transition monitor evaluates action evidence at a fixed interval and
requires configured confirmations. Persistent milestones remain latched after
confirmation, while transient facts such as holding an object remain
reversible.

## 8. Local repair

A confirmed failure creates a causal slice from the affected node and its
dependencies. Repair planning preserves confirmed milestones, excludes
exhausted retries, validates the replacement plan with VAL, and installs a new
certificate and DAG. Retry and action budgets keep the procedure finite.

## 9. Completion

Visual evidence alone cannot declare completion. LOGIV finishes only when the
symbolic terminal state and the environment's native completion signal agree.

## 10. Main modules

- `model.py`: typed facts, actions, snapshots, and context.
- `domain.py`: action schemas and PDDL rendering.
- `val.py`: validation and certificate verification.
- `dag.py`: causal graph construction and invariants.
- `controller.py`: permits, execution, budgets, and terminal states.
- `repair.py`: causal slicing and bounded repair search.
- `gpt4o.py`, `gpt4o_grounding.py`: structured visual grounding.
- `gpt4o_planning.py`, `initial_proposal.py`: proposal construction.
- `shadow_monitor.py`: transition monitoring.
- `libero_adapter.py`, `robotwin.py`: environment integration.

All canonical assets are packaged under
`src/pi05_libero_repro/logiv/config/`.
