# Task 5 Terminal `pi_recover` Design

**Status:** Approved direction; implementation specification

**Date:** 2026-08-05

**Target branch:** `pi05-libero-reproduction`

**Primary objective:** Produce the first real `LOGIV_R2M > BASE` result without changing any Base-success trajectory.

## 1. Scope and success definition

This tranche implements one narrowly defined recovery capability for LIBERO-10 Task 5:

```text
holding(black_book_1)
and not at(black_book_1, desk_caddy_1_back_contain_region)
    -> place-held-in(black_book_1, desk_caddy_1_back_contain_region,
                     desk_caddy_1_access)
```

The capability is terminal-only. Base runs normally through its native terminal and settling checks.
`LOGIV_R2M` may act only when Base has failed, Shadow has a fresh current topology, the book is
explicitly held, the destination is explicitly false, a one-option recovery plan is certified, and
the remaining shared action budget is sufficient. Base-success episodes never enter the recovery
path.

The first development go condition is:

- before training, confirm by fresh terminal grounding that at least two of the three frozen Task 5
  failures belong to the held-book state class;
- recover at least two of the three frozen Task 5 failures `t05-r02`, `t05-r03`, and `t05-r04`;
- preserve all seven Base successes in the existing ten-case random manifest;
- preserve the exact Base prefix up to terminal on every pair;
- perform at most one recovery handoff and one physical recovery macro per episode;
- introduce no protected-invariant violation; and
- keep total Base plus recovery model actions at or below 520.

Passing this gate permits a new frozen random 100-case paired development run. The 100-case result
must have zero Base-success to R2M-failure flips and strictly more native successes than Base. It is
development evidence, not the final paper confirmation.

## 2. Non-goals

This tranche does not implement Task 8 recovery, early takeover, multiple recovery attempts, return
to Base after an incomplete recovery, a general memory agent, or a prompt sweep. The HiMe-style
global memory/planner decomposition remains a later option if one recovery macro is insufficient;
it is not needed to test the current execution bottleneck.

The existing `LOGIV_REPAIR_OVERLAY` and frozen pi0.5 prompt variants remain diagnostics. They must
not be renamed or reported as `LOGIV_R2M`, because they do not use an independent recovery
checkpoint.

## 3. Evidence motivating the scope

The frozen ten-case simulator sample produced `BASE=7/10` and read-only `SHADOW_LOGIV=7/10` with
exact execution parity on all ten pairs. All three failures are Task 5. Their final fixed-graph traces
show the pick node completed while the `place-held-in` node remained active. This isolates the next
bottleneck to the held-book placement skill rather than initial planning or Shadow topology.

The detector tranche already provides:

- advisory partial monitoring with strict initial grounding;
- per-action event tracking;
- settling projection into the same fixed graph;
- explicit terminal topology status; and
- separation of confirmed subattempt failure from Base self-recovery.

The repository has 50 successful official Task 5 demonstrations, a full `pi05_libero` checkpoint,
and JAX LoRA training support. A held-state suffix policy is therefore the smallest independent
recovery model that can test a positive flip.

## 4. Architecture

The implementation has five bounded components.

### 4.1 Recovery dataset builder

The builder consumes only the official Task 5 HDF5 demonstration file. It replays each demonstration
state/action sequence through the pinned LIBERO environment and existing Task 5 grounder. A suffix
is accepted only after `holding(black_book_1)` is TRUE for three consecutive observations while the
target fact remains FALSE. The first observation of that stable interval begins a new recovery
episode; subsequent images, robot state, and actions are copied through native Task 5 success.

Every output frame uses the single versioned instruction:

```text
place the held book in the back compartment of the caddy
```

The output is a local LeRobot dataset with 7-D actions and the existing LIBERO camera/state mapping.
Demonstration identity is the split unit. A deterministic hash split assigns 35 demonstrations to
training, 5 to loss validation, and 10 to simulator capability validation before any checkpoint is
evaluated. Invalid or ambiguous suffixes are reported and excluded. Training is blocked unless at
least 46 of 50 demonstrations yield valid suffixes and the surviving split contains at least 32
training, 4 loss-validation, and all 10 capability-validation demonstrations.

The dataset manifest records source HDF5 hash, BDDL hash, selected demo IDs, split hash, frame ranges,
prompt version, observation/action schema, and output dataset hash. Large dataset files are not
committed to git; the manifest and builder are.

### 4.2 Independent recovery checkpoint

`pi_recover` is initialized from the frozen full `pi05_libero` expert parameters, not from the
generic `pi05_base` model. It uses the matching pi0.5 model shape, action horizon, LIBERO transforms,
and normalization assets. Both language/vision and action-expert LoRA adapters are trainable; base
weights remain frozen and EMA is disabled.

There is one pre-registered training run, not a hyperparameter sweep:

- seed: 42;
- global batch size: 64;
- optimizer: AdamW with gradient-norm clipping at 1.0;
- peak learning rate: `5e-5`;
- training horizon: 4,000 steps;
- saved checkpoints: 1,000, 2,000, 3,000, and 4,000 steps;
- external experiment logging disabled.

Checkpoint selection first minimizes validation loss. A tie is broken by the earlier checkpoint.
Simulator capability validation is then a pass/fail gate on that one selected checkpoint; it cannot
select a different checkpoint after observing evaluation outcomes.

The selected parameters, model config, normalization assets, source checkpoint, training config,
dataset manifest, and code revision receive immutable SHA-256 manifests. Inference runs in a second
episode-seeded policy service and never shares hidden policy state with Base.

### 4.3 Terminal deviation and handoff permit

The new `LOGIV_R2M` arm follows the exact `SHADOW_LOGIV` path until Base returns from its native
settling evaluation. It creates a terminal confirmed-deviation record only when all of the following
are true:

1. native Base evaluation is failure;
2. the final Shadow graph and certificate are `CURRENT`;
3. fresh strict grounding succeeds with exactly one location state;
4. `holding(black_book_1)` is TRUE;
5. `at(black_book_1, desk_caddy_1_back_contain_region)` is FALSE; and
6. the failed obligation maps to the active Task 5 place node.

Native terminal failure is the strong evidence for the
`TERMINAL_GOAL_UNSATISFIED` event; weak progress timeout alone still cannot authorize control. The
terminal evidence type and monitor-contract self-hash are versioned.

The planner rebases the current problem on that fresh snapshot and may return exactly one physical
option: the registered `place-held-in` action. The option must pass signed-state validation and VAL,
match the enabled capability contract, preserve all already completed Goal facts, and fit within:

```text
option_action_cap = min(180, 520 - base_policy_steps)
```

Any UNKNOWN fact, stale hash, missing capability, budget failure, plan with zero or multiple physical
options, or VAL error denies the permit and preserves the Base failure without physical side effects.

### 4.4 Recovery execution

After a durable permit, the controller flushes any recorded Base suffix, increments policy request
generation, and sends the versioned held-book instruction to the independent recovery service.
Recovery RNG is derived from a separate domain:

```text
uint32(sha256("LOGIV-recovery-policy-seed-v1:master:task:episode:event")[:4])
```

Only one recovery macro is dispatched. Execution stops when the declared effect is stable or when the
option cap is exhausted. No second option, retry, or fallback prompt is allowed in this tranche.

### 4.5 Commit verification

After recovery stops, the controller performs bounded dummy-action settling and obtains a fresh
strict snapshot. Recovery commits only if all predicates below hold together:

- the declared destination effect is TRUE;
- every Goal fact that was TRUE at handoff remains TRUE;
- no registered protected invariant regressed;
- the snapshot, permit, plan, checkpoint, and event hashes still match; and
- the native LIBERO evaluator reports task success.

Success terminates the episode; terminal-local recovery has no need to call Base again. If any check
is FALSE or UNKNOWN, the single recovery attempt fails closed and the episode remains a failure.

## 5. Provenance and artifacts

Each `LOGIV_R2M` episode retains the existing Base execution, Shadow monitor, and graph trace and adds:

- `terminal_deviation.json`;
- `current_snapshot.json`;
- `recovery_plan.json` and VAL certificate;
- `handoff_permit.json` or a stable denial reason;
- `recovery_rng.json`;
- `recovery_execution.json`;
- `post_recovery_snapshot.json`;
- `recovery_commit.json`;
- capability-contract and recovery-checkpoint hashes; and
- separate Base and recovery request/action accounting.

Paired audits require identical initial state, first frame, Base prompt, Base checkpoint, Base policy
seed, simulator seed, and Base action-prefix hash through terminal. Recovery actions and requests are
reported separately and never included in a claim of Base trajectory equality.

## 6. Failure containment

- Dataset replay or grounding ambiguity blocks the affected demonstration; insufficient valid data
  blocks training.
- Training interruption may resume only from a hash-matching checkpoint and manifest.
- Missing or unhealthy recovery service denies handoff and returns the original Base failure.
- Snapshot UNKNOWN, exactly-one conflict, stale certificate, or provenance mismatch denies handoff.
- Recovery timeout, inference error, failed effect, invariant regression, or native evaluator failure
  consumes the only attempt and terminates failure.
- No error path executes Base and recovery actions concurrently or exceeds the shared 520-action cap.

## 7. Verification strategy

### 7.1 Code and data tests

- Unit tests for stable held-suffix boundaries, split determinism, manifest hashes, and ambiguous
  demonstration rejection.
- Loader smoke test for image/state/action shapes and the versioned instruction.
- One-batch model initialization and gradient test proving only LoRA parameters update.
- Checkpoint restoration test proving a separate inference service loads the frozen recovery assets.

### 7.2 Controller tests

- Base success, non-held terminal failure, UNKNOWN grounding, stale graph, failed VAL, missing service,
  and insufficient budget all produce zero recovery actions.
- An eligible held-state terminal failure produces exactly one committed handoff.
- Effect or invariant verification failure cannot be reported as success.
- Base-prefix hashes remain identical to paired Base through the handoff boundary.
- Combined action accounting cannot exceed 520.

### 7.3 Simulator gates

1. Before dataset construction or training, rerun the three frozen Task 5 failures and persist fresh
   strict terminal snapshots. Require at least two roots with `holding(book)=TRUE` and destination
   FALSE; otherwise reject this state class and return to design.
2. Validate the selected checkpoint on the disjoint capability-validation demonstration roots with
   fixed branch seeds; require at least 8/10 native successes and zero protected-invariant violations.
3. Run the frozen ten-case paired sample; require at least 2/3 positive flips among the Task 5
   failures, zero negative flips, and exact Base-prefix parity.
4. Freeze code, dataset, checkpoint, prompt, contracts, and seeds; generate a fresh random 100-case
   manifest across all ten tasks and run paired `BASE` versus `LOGIV_R2M` in simulation.
5. Proceed only if `LOGIV_R2M` is strictly above Base with zero negative flips. Otherwise diagnose
   the single failing layer—data, recovery execution, permit, or verification—without broad prompt
   search.

## 8. Implementation boundary

The first implementation ends after the ten-case paired gate and its audit report. Task 8, dropped
book recovery, non-terminal handoff, Base re-entry, and external memory require separate approved
designs. This keeps the first performance claim attributable to one independent held-state recovery
skill and one auditable terminal permit.
