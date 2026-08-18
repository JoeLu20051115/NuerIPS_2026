# LOGIV method

LOGIV is a closed-loop execution method for long-horizon robot tasks. It uses a
learned policy for physical actions, but places a typed symbolic controller
around that policy so that every dispatched subtask is justified by the latest
observation and by a formally valid remaining plan.

## 1. Typed task model

A task is represented by:

- typed objects and locations;
- positive, negative, and unknown facts;
- a fixed action domain with preconditions and effects;
- a frozen goal derived from benchmark metadata;
- a finite set of supported and recovery action schemas.

`TruthValue` distinguishes `TRUE`, `FALSE`, and `UNKNOWN`. Unknown evidence is
not silently treated as false. Facts are normalized and hashed so that a plan
certificate, observation epoch, and controller context can be bound together.

The canonical LIBERO domain is packaged at
`src/pi05_libero_repro/logiv/config/domain.pddl`. RoboTwin task stages and
prompts are defined in `src/pi05_libero_repro/logiv/robotwin.py`.

## 2. Observation and state grounding

The grounder receives a fresh observation plus the fact universe required by
the active plan frontier. It returns a typed `FactSnapshot` associated with a
specific observation epoch and request context.

The deployment backend uses GPT-4o only for structured state-gate queries. The
client requires a strict JSON-schema response, uses temperature zero, validates
the returned model and payload, and applies a finite retry bound. It does not
ask the vision model to emit low-level robot actions.

The LIBERO adapter also audits simulator-derived facts used by the integration
layer. The RoboTwin adapter uses independent camera views for state gates, so a
policy observation and a gate observation are not implicitly the same signal.

## 3. Proposal and frozen goal

The proposal layer binds benchmark objects to grounded actions. The official
benchmark goal remains frozen for the episode; local repair changes only the
remaining route to that goal.

Each candidate action carries an occurrence identity and lineage. These values
prevent a failed action from being renamed and retried indefinitely under a new
graph. The proposal, grounded problem, occurrence sidecar, and context are
hashed before validation.

## 4. VAL certification

Before installing a plan, LOGIV renders the grounded problem and candidate plan
and calls a pinned VAL binary. A valid result becomes a `PlanCertificate` that
binds:

- domain and problem content;
- ordered grounded actions;
- occurrence metadata;
- controller context and fact epoch;
- VAL binary identity and wrapper version;
- retry-ledger state.

The controller refuses to dispatch from a missing, stale, mismatched, timed-out,
or invalid certificate. A validation error is not converted into a repairable
planning failure.

## 5. Causal DAG compilation

The certified linear plan is compiled into a causal DAG. Edges record support,
threat, and goal dependencies. Independent actions may become a ready frontier,
while dependent actions remain blocked until their predecessors are confirmed.

The graph keeps a canonical agenda for deterministic control decisions. A
schema-only compiler exists for explicit diagnostic use, but the deployed
`FULL_LOGIV` path installs only VAL-backed graphs.

## 6. Bounded policy execution

An execution permit identifies the authorized occurrence, graph version,
certificate, state epoch, and safety epoch. The underlying policy receives a
single stage or frontier prompt and may execute only a bounded action chunk.

After the chunk, LOGIV stops policy continuation, collects fresh evidence, and
checks declared effects. A successful effect commits the occurrence. A failed,
unknown, stale, unsafe, or terminal outcome follows a distinct controller path;
these cases are not collapsed into one generic retry.

For RoboTwin, a receding-horizon macro executor can continue within a certified
DAG node, resample a local repair action from fresh evidence, and optionally use
a bundled CFN to select repair samples. Each node boundary still returns to the
state gate.

## 7. Monitoring and uncertainty

The monitor observes action transitions at a fixed interval and requires the
configured number of confirmations. Persistent milestones are latched after
confirmation so that noisy later observations do not reopen completed geometry.
Transient facts, such as holding an object, are not latched as permanent.

If the active frontier is unknown, the controller may collect new evidence
within a finite budget. Unknown evidence never authorizes an action whose
precondition must be true. Exhausting the evidence or physical-action budget
produces a safe terminal outcome.

## 8. Local repair

Confirmed deviations are converted into signed obligations. The repair planner
extracts the relevant causal slice, searches only allowed recovery schemas, and
validates each candidate with VAL. A repair is installed only when its new
certificate matches the current facts and retry ledger.

Repair obeys four boundaries:

1. the episode goal remains frozen;
2. confirmed persistent milestones remain closed;
3. retry counts are keyed by action lineage;
4. edit, candidate, VAL-call, repair-round, and physical-attempt budgets are
   finite.

Terminal recovery for the LIBERO book-placement task is represented as a normal
capability contract under
`src/pi05_libero_repro/logiv/config/terminal-recovery.json`, not as an external
result-specific exception.

## 9. Completion rule

Visual goal evidence alone cannot declare success. LOGIV reaches successful
termination only when the symbolic terminal state and the benchmark's native
success signal agree. A disagreement triggers further observation, repair, or a
bounded stop depending on the remaining budget.

## 10. Main implementation modules

- `model.py`: typed facts, problems, actions, snapshots, and context.
- `domain.py`: fixed domain and PDDL rendering.
- `val.py`: validation wrapper and certificate verification.
- `dag.py`: causal graph construction and graph invariants.
- `controller.py`: permits, execution loop, budgets, and terminal states.
- `repair.py`: causal slicing, retry ledger, and PDDL repair search.
- `gpt4o.py`, `gpt4o_grounding.py`: structured visual state gates.
- `libero_adapter.py`: LIBERO state and policy integration.
- `shadow_monitor.py`, `shadow_runtime.py`: transition monitoring and online
  control support.
- `robotwin.py`: RoboTwin tasks, grounding, planning, and episode controller.

All deployable configuration is packaged under
`src/pi05_libero_repro/logiv/config/`; no runtime default depends on an
inherited tuning chain.
