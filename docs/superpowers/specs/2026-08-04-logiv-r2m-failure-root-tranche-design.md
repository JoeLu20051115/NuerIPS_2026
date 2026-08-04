# LOGIV-R2M Phase 0 Failure-Root Tranche Design

## Material Passport

- Artifact type: development-only experiment and orchestration design
- Design date: 2026-08-04 (Asia/Singapore)
- Implementation branch: `logiv-r2m-phase0`
- Starting commit: `c7940275`
- Backbone: frozen full `pi0.5` LIBERO checkpoint
- Tasks: zero-based LIBERO-10 Task 5 and Task 8
- Status: approved design; implementation and real tranche unverified

## 1. Decision Summary

Run a bounded, failure-seeking Phase 0 tranche on Task 5 and Task 8 to prove that
the existing read-only `SHADOW_LOGIV` arm can detect a real, contract-supported
deviation and persist a valid recovery root without changing any Base action or
Base policy request.

The tranche alternates between Task 5 and Task 8. Every candidate uses a paired
`BASE` then `SHADOW_LOGIV` execution under the same frozen policy server process,
master seed, task, episode index, checkpoint, prompt, budget, and evaluator
protocol. Each task stops after its first valid confirmed root or after six
candidates, whichever comes first.

This tranche does not train or invoke a recovery policy and cannot improve task
success by construction. Its purpose is to establish the live data boundary
needed before Phase 1 recovery training. The later paired-development success
criterion remains a net improvement of at least `+10/100` over Base across
Task 5 and Task 8, with at most two negative flips per task and no intervention
without an enabled capability contract.

## 2. Goals and Non-Goals

### Goals

1. Obtain at least one `CONFIRMED_DEVIATION` and one mechanically valid recovery
   root for each of Task 5 and Task 8 within the bounded candidate set.
2. Prove per candidate that Shadow preserves the exact Base rollout envelope.
3. Validate live recovery-root persistence, verified loading, evidence freshness,
   and DEV split isolation on real simulator states.
4. Produce an auditable candidate manifest and a per-pair report that includes
   failures and zero-yield candidates rather than only successful roots.
5. Permit at most one evidence-backed detector fix if a contract-supported real
   deviation is observed but missed.

### Non-Goals

- No physical recovery, handoff, recovery policy request, or takeover permit.
- No recovery-policy training and no raw root passed to a trainer.
- No prompt, checkpoint, action-budget, or policy-server sweep.
- No held-out claim, detector-recall claim, or task-success improvement claim.
- No root eligibility from terminal failure alone, an anomaly candidate alone,
  or a stale certificate without a matching fresh confirmed label.
- No changes to the dirty `pi05-libero-reproduction` worktree.

## 3. Fixed Experimental Envelope

The tranche runs from the existing clean linked worktree on
`logiv-r2m-phase0`. It uses one frozen policy server process for all attempted
pairs so process-to-process policy variation cannot be confused with a method
effect. A server restart invalidates the in-progress pair; both arms of that
pair must be rerun after the restart.

The fixed evaluator envelope is:

- master seed: `7`;
- policy and simulator RNG protocol: `episode-seeded-v1`;
- checkpoint: `full`;
- goal mode: `METADATA_ASSISTED`;
- deviation mode: `NOMINAL`;
- action budget: `520` total Base actions;
- Shadow monitor interval: `5` Base steps;
- Shadow confirmations: `3`;
- recovery-root split: `DEV`;
- collection requires `--development-only`, `--oracle-grounding`, and
  `--collect-recovery-roots`;
- Base never constructs a proposal provider, monitor, or root writer;
- Shadow records `shadow_vlm_requests=0` and `recovery_policy_requests=0`.

Within this design, the informal word “seed” means `episode_idx` under master
seed `7`. The derived policy seed is recorded from the evaluator artifact and
is not manually overridden.

## 4. Candidate Manifest and Scheduling

Candidate order is frozen before observing new tranche outcomes:

| Attempt | Task 5 `episode_idx` | Task 8 `episode_idx` |
| ---: | ---: | ---: |
| 1 | 0 | 4 |
| 2 | 3 | 0 |
| 3 | 38 | 2 |
| 4 | 45 | 3 |
| 5 | 2 | 6 |
| 6 | 6 | 8 |

The order prioritizes previously observed Task 5 failures with varied rollout
lengths and Task 8 diagnostics spanning a dropped-object branch, a late drop,
and a transient-done/settling failure. Prior outcomes are development routing
evidence only; current episode-seeded behavior must be measured afresh.

Scheduling alternates tasks:

1. run the next Task 5 `BASE`/`SHADOW_LOGIV` pair;
2. validate and classify it immediately;
3. run the next Task 8 pair;
4. validate and classify it immediately;
5. repeat only for tasks that have not yet obtained a valid confirmed root.

A task stops independently at its first valid confirmed root. The other task
continues up to its own six-candidate cap. No successful or failed candidate is
removed from the report.

## 5. Pair Execution and Parity Gate

For each candidate, run `BASE` first and `SHADOW_LOGIV` second into new,
candidate-specific output directories. Reusing an existing output directory or
using diagnostic resume is forbidden.

Before any trigger or root is interpreted, compare the paired
`base_execution.json` records. All of these fields must match exactly:

1. `steps`;
2. `base_policy_requests`;
3. `done_signal`;
4. `post_settling_success`;
5. `initial_state_sha256`;
6. `base_prompt_sha256`;
7. `base_checkpoint_sha256`;
8. `policy_client_config_sha256`;
9. `request_envelope_log_sha256`;
10. `actions_sha256`.

The Shadow episode must also report `shadow_parity_valid=true`. Any mismatch is
a tranche-level stop: the pair is invalid, its roots are ineligible, later
candidates are not run, and the issue is investigated as nominal interference
rather than detector performance.

## 6. Deviation and Root Eligibility

Base terminal failure is not sufficient evidence of a deviation. A detector
miss may be claimed only when development oracle facts, event provenance, and
saved visual/state evidence establish one of the frozen Task 5/8 monitor
contracts, such as a stable target object on a registered abnormal recovery
surface or a fresh manipulation-backed abnormal transfer.

A successful collection requires all of the following:

- the monitor emits `CONFIRMED_DEVIATION` from fresh evidence;
- the root label matches the confirmed event, certificate universe, evidence
  contract, and event-origin lineage;
- the root can be loaded with `load_recovery_root` and reproduces its exact
  simulator state, raw observation, pending Base context, and hashes;
- the DEV dataset passes `validate_recovery_dataset.py`;
- callback, proposal callback, provenance, snapshot, event tracker, evidence
  overflow, trigger callback, and root-writer error buckets are zero.

An anomaly candidate, a stale certificate, or an unmatched/stale label may be
retained for detector development but is not a successful root and is not
training eligible.

If `CONFIRMED_DEVIATION` is emitted but no corresponding root is persisted, the
pair stops as a data-pipeline failure. It does not consume the one detector-fix
allowance because detection already succeeded.

## 7. One-Fix Detector Budget

The detector remains frozen unless a real, contract-supported deviation is
established from the development evidence and the monitor fails to emit the
matching confirmed record. Outcome failure alone cannot authorize a fix.

At most one detector change is permitted for the entire tranche:

1. preserve the complete failed pair and record the expected evidence path;
2. add a focused regression test that fails for the observed root cause;
3. verify the test fails for that cause rather than a fixture or syntax error;
4. make the smallest change to one parser, frozen contract, or monitor state
   transition needed to admit the already-supported evidence;
5. do not change checkpoint, prompts, action budgets, confirmation thresholds,
   or unrelated state classes;
6. run the focused test, the full unit suite, compile checks, shell syntax check,
   and `git diff --check`;
7. commit the fix separately;
8. rerun both arms of the affected pair from new output directories before
   resuming the frozen candidate order.

If a second distinct detector defect appears, the tranche stops and reports
`NO_GO`; it is not patched inside this experiment.

## 8. Failure Containment

- If either arm crashes, times out, loses the policy server, or produces an
  invalid row, mark the pair ineligible, preserve its artifacts, and rerun both
  arms from new directories after the infrastructure fault is resolved.
- If the server restarts, do not reuse the Base result from the prior process.
- If parity fails, stop the tranche before validating or allocating roots.
- If root loading, checksums, or split validation fail, stop and classify the
  issue as a recovery-data failure.
- Raw root manifests are immutable. Role allocation or training-manifest tools
  must not mutate them.
- Partial output directories and rejected candidates remain auditable but are
  excluded from the eligible-root count.

## 9. Verification and Reporting

Before launching the tranche, the branch must pass:

- `git diff --check`;
- `bash -n scripts/run_logiv_eval.sh`;
- `uv run python -m compileall -q src scripts`;
- `uv run pytest -q`, with the current baseline of 379 passing tests.

Every completed pair receives one report row containing task, episode index,
derived policy seed, both terminal outcomes, steps, Base requests, parity result,
proposal status, certificate state, anomaly count, confirmed count, root count,
all error buckets, validator result, and artifact paths. The report includes the
frozen candidate manifest, policy-server provenance, implementation commit,
monitor-contract hash, checkpoint hash, and any detector-fix commit.

Tranche outcomes are:

- `GO`: both Task 5 and Task 8 obtain at least one valid confirmed root within
  their six-candidate caps;
- `PARTIAL`: exactly one task obtains a valid confirmed root; only that task may
  advance to a larger development collection, while the other records its
  detector or data-coverage gap;
- `NO_GO`: parity cannot be restored, an evidence/data invariant fails, the
  one-fix budget is exceeded, or neither task obtains a valid confirmed root.

`GO` establishes that the real collection path works. It does not establish
detector recall, recovery competence, or sufficient sample size for training.

## 10. Downstream Success Boundary

After a `GO`, separate Phase 1 work may expand development roots, generate or
collect macro bridges, train REC-A/REC-B, and calibrate capability contracts.
Only an `ENABLED` contract may later authorize physical recovery.

The eventual Task 5/8 paired-development gate is fixed at:

- 50 paired episodes per task under the same 520-step total budget;
- combined net improvement over Base of at least `+10/100`;
- no more than two negative flips per task;
- every intervention matched to an enabled capability contract.

Until that later gate is passed, no result from this failure-root tranche may be
described as an empirical success-rate improvement over Base.
