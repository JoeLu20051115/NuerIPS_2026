# LOGIV-R2M Phase 0 Failure-Root Tranche Report

Date: 2026-08-04

Branch: `logiv-r2m-phase0`

Implementation commit: `95580ae2f4fe7b986889c612339cdc19b760f086`

Outcome: **NO_GO — evidence/data invariant failure**

## 1. Frozen Task and Stop Rule

The approved task was to alternate paired `BASE` then `SHADOW_LOGIV` runs on
Task 5 and Task 8, with no more than six candidates per task, until each task
produced at least one mechanically valid `CONFIRMED_DEVIATION` DEV recovery
root. Every pair had to preserve the exact Base execution envelope. At most one
evidence-backed local detector fix was permitted; a second distinct detector or
data defect required a tranche-level `NO_GO` stop.

This Phase 0 tranche was read-only by construction. It did not train or invoke a
recovery policy and therefore could not itself improve task success over Base.
The later success gate remains a paired 50-episode evaluation per task with at
least `+10/100` net successes over Base, at most two negative flips per task,
and an enabled capability contract for every intervention.

## 2. Frozen Runtime Provenance

- Policy server: one uninterrupted `episode-seeded-v1` process on GPU 0, port
  8010, for all formal pairs.
- Checkpoint: `full`; Base checkpoint SHA-256
  `7fcf7ee6e020cbcb67ccad4c7d378b892e4303b0c3f09be59df74dbc1154210d`.
- Checkpoint normalization statistics SHA-256:
  `b3a44a096a62a058f85e1932d75d05edaf371957cfb2ef084e2efef3747abd84`.
- Master seed: `7`; goal mode: `METADATA_ASSISTED`; deviation mode:
  `NOMINAL`; Base budget: 520 actions.
- Coverage manifest SHA-256:
  `0c6cf2d1f5266c48522f2c4f4b0f2c1d23c26c089d606bc48027ef05b32e6e35`.
- Task 8 monitor contract SHA-256:
  `05e5da5e028134c9df93e8ec5333f9268fdb0a2617ce375750df4bf23da9c140`.
- Policy-server log SHA-256:
  `3d99d26c3a59cdb40070a246641ed987e6412fe7a7233f631127d528031ac13b`.

Before launch, `git diff --check`, shell syntax checks, Python compile checks,
and the full test suite passed; pytest reported `379 passed`.

## 3. Formal Pair Results

| Task | Episode | Policy seed | Base / Shadow outcome | Steps | Base requests | Exact Base parity | Proposal | Snapshot errors | Candidates | Confirmed | Roots | Classification |
|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 5 | 0 | 1239936538 | fail / fail | 162 | 33 | pass | accepted | 0 | 0 | 0 | 0 | `NO_ROOT` |
| 8 | 4 | 15428879 | fail / fail | 520 | 104 | pass | accepted | 5 | 1 | 0 | 1 | hard-gate failure |

For both pairs, the ten frozen `base_execution.json` fields matched exactly,
the policy RNG records matched, and Shadow reported
`shadow_parity_valid=true`. Shadow issued zero Shadow VLM requests and zero
recovery-policy requests.

### Task 5, episode 0

The pair was a valid zero-yield candidate. All monitor error buckets were zero:
163 callbacks, 33 snapshots, no anomaly candidate, no confirmed deviation, and
no root. Video review showed terminal failure but did not establish a frozen
contract-supported deviation, so terminal failure was not relabeled as a
detector miss.

Artifacts:

- `runs/r2m-phase0-root-tranche/task-05-episode-000/base`
- `runs/r2m-phase0-root-tranche/task-05-episode-000/shadow`

### Task 8, episode 4

The Base/Shadow rollout was exactly paired, but the formal Shadow audit failed:

- 521 callbacks and 105 snapshot attempts;
- `snapshot_errors=5`, hence `aggregate_errors=5`;
- one `ANOMALY_CANDIDATE`, zero `CONFIRMED_DEVIATION`;
- one stale DEV root with no historical failure evidence.

The root at policy step 240 is mechanically loadable and split-clean, but is
not training eligible. Dataset validation returned:

```json
{"ANOMALY_CANDIDATE":1,"CONFIRMED_DEVIATION":0,"DEV":1,"HELDOUT":0,"TRAIN":0,"total":1,"unique_independence_units":1}
```

The root records `moka_pot_2` on `kitchen_table_recovery_surface`, off the
stove, `handempty`, a stale certificate, and an empty
`historical_failure_evidence_json`.

Artifacts:

- `runs/r2m-phase0-root-tranche/task-08-episode-004/base`
- `runs/r2m-phase0-root-tranche/task-08-episode-004/shadow`
- `runs/r2m-phase0-root-tranche/task-08-episode-004/shadow/artifacts/task_08/episode_004/recovery_roots/db899df1c06e0ec46f5f65ef9c56a5c16e8ee2190e3e8b02888280d9d8b0bf64/recovery_root.json`

## 4. Root-Cause Evidence

A deterministic read-only replay reproduced exactly five failures at policy
steps 155, 160, 165, 220, and 225. Every failure was:

```text
GroundingError: exactly-one violation for moka_pot_2: confirmed=[]
```

A bounded prefix replay logged the full fact partition at all five steps. The
values were identical:

```text
(at moka_pot_2 flat_stove_1_cook_region)                 FALSE
(at moka_pot_2 kitchen_table_moka_pot_left_init_region)  FALSE
(at moka_pot_2 kitchen_table_moka_pot_right_init_region) FALSE
(at moka_pot_2 kitchen_table_recovery_surface)           UNKNOWN
(holding moka_pot_2)                                     FALSE
```

This proves the registered abstract location state is non-exhaustive during two
live manipulation intervals. Treating the recovery surface's `UNKNOWN` value as
`TRUE` would silence the invariant by assertion, not by physical evidence, and
could create false recovery roots.

A second, independent coverage boundary is visible in the event attribution
path. A place attempt is created only when the gripper opens while the previous
state is holding and the current state is already true and near the nominal
destination. A release or transfer outside the destination therefore creates
no place attempt. At step 240 the abnormal support state was stable, but the
root had no matching historical action-event evidence and remained an anomaly
candidate.

Resolving the first boundary soundly requires an explicit abstract state for an
unsupported/unplaced movable (or equivalent physically evidenced grounding).
Resolving the second requires a separate release/abnormal-transfer attribution
rule. Those changes cross the state model, grounder/coverage contract, and event
detector boundary; they are not one local detector repair.

## 5. Decision and Remaining Work

The Pair Audit Contract defines any nonzero snapshot error as a hard gate, not a
zero-yield seed. The approved one-fix budget cannot soundly cover both distinct
boundaries. No detector code or frozen contract was changed, and the remaining
Task 5/8 candidates were not launched after the hard gate.

Current result:

- Task 5: 1 of 6 candidates run, 0 valid confirmed roots.
- Task 8: 1 of 6 candidates run, 0 valid confirmed roots; 1 diagnostic-only
  anomaly root.
- Tranche: `NO_GO`; no recovery training and no Base-improvement claim.

The next iteration must be approved as a new design tranche with two explicit
capabilities: (1) an auditable unsupported/unplaced location state that keeps
exactly-one semantics sound during manipulation, and (2) release-based abnormal
transfer attribution that does not require the nominal destination to already
be true. It must add focused failing tests first, update all affected provenance
hashes/contracts, rerun both arms of Task 8 episode 4 into new directories, and
reapply the same parity and zero-error gates before resuming the candidate order.
