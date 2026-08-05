# Shadow Temporal Topology Design

## Objective

Make `SHADOW_LOGIV` judge the fixed certified topology correctly before any
LOGIV-R2M takeover is enabled.  Shadow remains read-only: Base executes the
same actions, Shadow emits only the fixed graph plus its time-varying node
states, and simulator-native success is used only as an external evaluation
label.

## Evidence and Root Cause

The original random-100 run produced 59 true positives, 29 false negatives,
10 true negatives, and one false positive at the final exact snapshot.  The
failure is structural rather than a task-goal transcription error:

1. LIBERO evaluates each official goal from raw BDDL predicates.  A target
   relation may therefore be true while the robot still grasps the object.
2. `LiberoOracleGrounder` normalizes the planning state by making reliable
   `holding(object)` dominate every simultaneous `at(object, location)` fact.
   The normalized fact is correct for exclusive PDDL state, but it erases raw
   target evidence when `project_graph_state` evaluates the official goal.
3. Graph projection is stateless.  It rechecks an atomic macro's starting
   preconditions on every physical control step.  A normal grasp makes
   `handempty` and the source relation false, so a running placement is shown
   as `BLOCKED`; the certificate reconciler likewise treats the macro's
   internal state as an uncovered change and becomes `STALE`.
4. During transport, all registered `at` and `holding` facts can be false for
   a short interval.  The topology-only snapshot currently raises an
   exactly-one error, accounting for trace gaps instead of representing an
   observed out-of-model transport state.
5. Shadow receives no observation during the ten settling actions.  Its final
   state can therefore preserve a transient goal that the post-settling native
   evaluator rejects.

## Decision

Use a dual-layer, temporal topology tracker.

The raw evidence layer retains the simulator predicate result before planning
normalization.  The normalized layer remains the only layer used for strict
PDDL grounding, repair preconditions, and effect/invariant gates.  Shadow may
read raw evidence from the audited dominance record, but it may not read
`env.check_success()` or the episode result.

The fixed graph structure never changes.  Only node state changes over time.
Action nodes use these states:

- `BLOCKED`: known starting preconditions are false and no matching macro is
  in flight.
- `PRECONDITION_UNKNOWN`: required evidence is unknown, or the involved object
  is temporarily outside the registered exclusive abstraction.
- `READY`: predecessors are complete and starting preconditions are true.
- `ACTIVE`: a ready placement macro has entered a schema-declared internal
  transport state such as holding or unlocated transport.
- `EFFECT_OBSERVED`: the raw destination relation is true while stricter
  operational completion, such as release, is not yet established.
- `COMPLETED`: normalized declared effects are satisfied.

`GOAL` is evaluated directly from raw official signed goal predicates.  This
allows benchmark goal completion while an action remains `EFFECT_OBSERVED`.
It does not authorize recovery completion: recovery still requires normalized
effects and protected invariants.

## Temporal Rules

For `place-on`, `place-in`, and `place-relative`, a transition from a ready
state into `holding(object)` or an all-false registered location group marks
the matching node `ACTIVE`.  The all-false case is transport only when
`holding(object)` is explicitly FALSE; UNKNOWN exclusive evidence remains
`PRECONDITION_UNKNOWN`.  Raw destination truth while holding marks it
`EFFECT_OBSERVED` only after matching temporal progress.  A normalized
destination effect marks it `COMPLETED`.

The certificate reconciler maintains the matching macro as in flight.  For
that object it accepts the schema-declared transport envelope—changes to
`holding`, `handempty`, and registered locations—until the declared effect is
complete.  A grounded release at a non-target location exits the envelope and
makes the certificate stale; returning to the declared starting state is the
only non-completing release that returns the node to ready.  A completed effect
that later regresses is not covered and may make the certificate stale.

Topology snapshots may preserve an all-false movable location group.  They
must still reject two simultaneous normalized locations.  Strict controller
grounding continues to require exactly one location-or-holding fact.

Each settling action produces a separate read-only Shadow callback with no
Base action provenance.  Those samples append to the same graph state trace.
Consequently a transient goal can regress before the final graph snapshot,
and the final Shadow state observes the same simulator barrier as the native
terminal evaluator.

Topology-only mode records every policy callback regardless of the recovery
monitor's sampling interval.  Interval sampling remains a recovery-monitor
optimization and cannot be used by the temporal graph tracker, whose state
transitions depend on consecutive observations.

## Interfaces and Artifacts

- `FactSnapshot.raw_truth(fact)` reconstructs pre-normalization target truth
  only from validated dominance evidence; unaudited snapshots fall back to
  normalized truth.
- `ShadowGraphTracker.project(...)` owns temporal node state.  The existing
  `project_graph_state(...)` remains a one-shot compatibility wrapper.
- `ShadowSettlingContext` carries copied observation, policy-step count,
  settling index, and total settling count.  It contains no success label.
- `ShadowRuntime.settling_observer` extends topology-only observation through
  the post-action barrier.
- `graph.json` remains the sole topology output.  Each trace sample contains
  graph node statuses and its `POLICY` or `SETTLING` phase.

## Future HiMe-Inspired Agent Memory Layer (Out of Scope)

HiMe (arXiv:2607.03449) separates long-horizon control into a high-frequency
Executor, a Sentry that detects subtask completion or invalidation, and a
lower-frequency Planner that manages contextual and procedural memory through
explicit Add, Update, and Delete operations.  Its architecture is a useful
upper-layer reference for the later R2M design, after the current Shadow gate
passes.

The intended mapping is:

- Base is the high-frequency Executor and continues normally from step zero.
- The fixed topology plus `ShadowGraphTracker` is the Sentry and procedural
  subtask state.  Its completion/deviation decision remains simulator-grounded
  rather than being accepted from an LLM assertion.
- A future global episode memory stores validated completed nodes, the active
  node, protected invariants, confirmed deviations, attempted recovery macros,
  and effect receipts.  Only grounded evidence may commit Add/Update/Delete
  operations, preventing a Planner hallucination from poisoning memory.
- The Planner is invoked only at a verified subtask boundary or confirmed
  deviation.  It retrieves this memory and emits an explicit language
  instruction for the next Base subtask or an independently executable
  recovery macro.
- On deviation, independent `pi_recover` executes the recovery macro.  Effect
  and protected-invariant gates—not the Planner—decide whether execution can
  be handed back to Base.

This borrows HiMe's frequency separation and active memory management without
replacing R2M's certificate, invariant, and hand-back safety boundaries.

## Safety Boundaries

- Do not call or copy `env.check_success()` inside Shadow judgment.
- Do not change Base prompts, policy requests, action chunks, actions, early
  termination, or settling actions.
- Do not weaken strict grounding used to authorize control.
- Do not convert missing registered location evidence into a fabricated
  location fact.
- Do not implement R2M takeover or `pi_recover` in this change.

## Verification

Focused unit tests must demonstrate:

1. raw target truth survives a holding-dominance override;
2. normal macro transport is `ACTIVE`, raw target overlap is
   `EFFECT_OBSERVED`, and release is `COMPLETED`;
3. raw official goal truth can complete `GOAL` while the action is not yet
   operationally complete;
4. topology snapshots retain all-false movable groups but strict grounding
   rejects them;
5. nominal macro transitions do not stale the certificate, while a later
   effect regression does;
6. settling observations are isolated, ordered, and included without changing
   Base actions or the native terminal result.

The fixed random-100 manifest is then rerun for `SHADOW_LOGIV` only.  The gate
is 100 valid paired executions, exact action/outcome parity with the saved Base
records, zero Shadow/trace errors, full topology trace coverage, and final
`GOAL` agreement with post-settling native success for all 100 cases.
