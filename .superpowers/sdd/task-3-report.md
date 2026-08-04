# Task 3 Report: Reproducible recovery-root records

## Status

DONE. Implemented from starting HEAD `43220985359a52e1b2463fb136114aac4bbdbb3f` with strict test-first development and no new dependency.

## RED evidence

### Required focused collection RED

Command:

```text
uv run pytest -q tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py
```

Relevant output:

```text
ERROR collecting tests/logiv/test_recovery_records.py
E   ImportError: cannot import name 'recovery_records' from 'pi05_libero_repro.logiv'
1 error in 0.18s
```

The failure was the brief's expected missing-module collection failure.

### Audited grounder RED

Command:

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_oracle_grounder_emits_a_complete_audited_true_false_unknown_partition tests/logiv/test_libero_adapter.py::test_epoch_and_observation_noise_change_evidence_but_not_fact_universe tests/logiv/test_libero_adapter.py::test_audited_snapshot_rejects_tampered_missing_duplicate_or_conflicting_evidence
```

Relevant output:

```text
E   AttributeError: 'FactSnapshot' object has no attribute 'fact_universe'
E   AttributeError: 'FactSnapshot' object has no attribute 'evidence_payload_json'
3 failed in 0.11s
```

### Phase-0 generation and byte-identical retry RED

Command:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_invalid_arrays_and_synthetic_parent_provenance_are_rejected tests/logiv/test_recovery_records.py::test_identical_retry_compares_exact_array_bytes
```

Relevant output:

```text
E   Failed: DID NOT RAISE <class 'ValueError'>
E   ValueError: existing recovery root is non-identical corruption
2 failed in 0.07s
```

This exposed a missing Phase-0 generation check and NumPy value equality incorrectly treating byte-identical NaN arrays as different.

### Future Task 5 contract-shape RED

Command:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_group_id_ignores_branch_seeds_but_root_id_does_not
```

Relevant output after changing the fixture to the exact future `MonitorEvidenceContract` dataclass shape (which has no `schema_version` field):

```text
E   ValueError: monitor contract schema fields mismatch
1 failed in 0.07s
```

The validator was corrected to use the exact Task 5 field set; unknown schemas are rejected by missing/extra fields while `contract_id` remains an opaque nonempty identity.

### Flattened state/chunk provenance RED

Command:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_flattened_state_and_base_chunk_shape_and_response_size_are_verified
```

Relevant output:

```text
E   Failed: DID NOT RAISE <class 'ValueError'>
1 failed in 0.06s
```

### Canonical fact-value order RED

Command:

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_audited_snapshot_rejects_tampered_missing_duplicate_or_conflicting_evidence
```

Relevant output after reversing and re-hashing the otherwise valid `values` list:

```text
E   Failed: DID NOT RAISE <class 'ValueError'>
1 failed in 0.16s
```

## GREEN evidence

Audited grounder slice:

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_oracle_grounder_emits_a_complete_audited_true_false_unknown_partition tests/logiv/test_libero_adapter.py::test_epoch_and_observation_noise_change_evidence_but_not_fact_universe tests/logiv/test_libero_adapter.py::test_audited_snapshot_rejects_tampered_missing_duplicate_or_conflicting_evidence
...                                                                      [100%]
3 passed in 0.07s
```

Recovery-root initial GREEN:

```text
uv run pytest -q tests/logiv/test_recovery_records.py
..............                                                           [100%]
14 passed in 0.08s
```

Phase-0 generation and exact-byte retry GREEN:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_invalid_arrays_and_synthetic_parent_provenance_are_rejected tests/logiv/test_recovery_records.py::test_identical_retry_compares_exact_array_bytes
..                                                                       [100%]
2 passed in 0.05s
```

Flattened-state/chunk GREEN:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_flattened_state_and_base_chunk_shape_and_response_size_are_verified
.                                                                        [100%]
1 passed in 0.05s
```

Canonical fact-value order GREEN:

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_audited_snapshot_rejects_tampered_missing_duplicate_or_conflicting_evidence
.                                                                        [100%]
1 passed in 0.07s
```

Final required focused command:

```text
uv run pytest -q tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py
........................................................................ [ 85%]
............                                                             [100%]
84 passed in 0.17s
```

## Full-suite verification

Command:

```text
uv run pytest -q
```

Output:

```text
........................................................................ [ 32%]
........................................................................ [ 65%]
........................................................................ [ 97%]
.....                                                                    [100%]
221 passed in 1.73s
```

Additional checks:

```text
git diff --check
uv run python -m compileall -q src/pi05_libero_repro/logiv tests/logiv/test_recovery_records.py
```

Both exited 0 with no output.

## Implementation

- Appended the four optional audit fields to `FactSnapshot`, preserving all-null legacy snapshots while validating complete audited universes, canonical fact partitions, hashes, epochs, evidence JSON, and deterministic value/override ordering.
- Changed `LiberoOracleGrounder` to emit a complete TRUE/FALSE/UNKNOWN universe, frozen task-scoped grounding version/hash, and exact canonical evidence payload using the same observation hash as recovery loading.
- Added exact `RecoverySplit`, `CollectionLabel`, `RecoveryRootManifest`, and `RecoveryRootArtifacts` interfaces.
- Added keyword-only canonical manifest construction with the exact Task 3 parameter order, canonical recovery/group/independence IDs, typed evidence/contract timing verification, request-envelope identity checks, live-continuation eligibility derivation, and exact state/observation/pending-action hashes.
- Added locked, fsynced temporary-directory publication by one same-filesystem rename, idempotent exact-byte retry, corrupt-existing-root rejection, orphan-temp reporting, `allow_pickle=False` loading, and full manifest/NPZ re-verification.

## Files

- `src/pi05_libero_repro/logiv/recovery_records.py` (new)
- `src/pi05_libero_repro/logiv/model.py`
- `src/pi05_libero_repro/logiv/libero_adapter.py`
- `tests/logiv/test_recovery_records.py` (new)
- `tests/logiv/test_libero_adapter.py`
- `.superpowers/sdd/task-3-report.md` (new)

## Self-review

- Checked every brief field against the public dataclass and `make_recovery_root_manifest` signature/order.
- Checked ID inclusion/exclusion semantics: seeds/status change only artifact identity; event origin determines recovery group; trajectory lineage determines independence unit.
- Checked future Task 5 compatibility against the exact planned `MonitorEvidenceContract`, `ActionEventRule`, and `ActionEventEvidence` records.
- Checked atomicity ordering: both final-named files fsynced, temp directory fsynced, one sibling rename, then parent fsync; publication failure leaves only a discoverable orphan temp.
- Checked loader mutations for contract/evidence timing, fact conflicts/duplicates, simulator bytes, observation/evidence binding, request envelopes, and unknown manifest fields.
- Checked legacy `FactSnapshot` behavior remains accepted outside recovery-root construction.
- Checked no unrelated experimental files or dependencies changed.

## Concerns

No Task 3 correctness concern. Non-blocking integration note: Task 6 must provide the frozen replay-contract hash and acknowledged/future request envelopes for live-Base-continuation eligibility; otherwise roots are intentionally labeled for the both-arms-flush diagnostic.

## Independent-review fixes

An independent review found five correctness gaps and four hardening opportunities. All Critical and Important findings were addressed test-first:

- Observation arrays are now owned immutable copies. Fact capture holds the observation-store lock for its full grounding transaction, while synchronous simulator advance holds the same lock across `env.step`, observation publication, and epoch increment. A grounding context is checked against the captured epoch, so a snapshot cannot mix simulator state and observation epochs.
- The grounder now treats its registered monitored facts as a frozen universe and rejects required facts outside it.
- Historical evidence now recomputes versioned action-attempt, Goal-regression-attempt, and evidence IDs when the embedded provenance is sufficient. Evidence must have been emitted by the root step and be active at the root step (`due <= root <= expiry`).
- Every action-event rule object must belong to the embedded contract and root object-instance sets, and every evidence object must belong to the root object-instance set.
- Manifest, contract, evidence, and envelope inputs now enforce exact scalar/container types, including rejecting Python `bool` as an integer and JSON floats as integer aliases.
- Canonical fact ordering uses one PDDL sort key. Dominance overrides now validate their kind, uniqueness, universe membership, fact predicates, object identity, and canonical order.
- Schema v1 has an explicit frozen field tuple independent of dataclass introspection.
- Atomic publication has an explicit order-spy characterization test covering lock, both file fsyncs, temporary-directory fsync, rename, parent-directory fsync, and unlock.

### Review-fix RED evidence

Atomic observation ownership, simulator transaction, and frozen-universe regressions:

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_observation_store_owns_immutable_array_copies tests/logiv/test_libero_adapter.py::test_snapshot_and_synchronous_simulator_advance_cannot_mix_epochs tests/logiv/test_libero_adapter.py::test_required_facts_cannot_expand_the_frozen_registered_universe
FFF                                                                      [100%]
3 failed, 1 warning in 0.17s
```

The failures respectively showed caller-owned array aliasing, the absent coordinated `advance` transaction, and silent required-fact universe expansion.

Evidence identity, root-time validity, and attribution closure regressions:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_historical_evidence_recomputes_attempt_and_evidence_ids tests/logiv/test_recovery_records.py::test_historical_evidence_must_be_active_at_the_root_policy_step tests/logiv/test_recovery_records.py::test_contract_and_evidence_objects_are_closed_over_root_instances
FFF                                                                      [100%]
3 failed in 0.07s
```

All three failures were `DID NOT RAISE`, confirming that arbitrary IDs, future/expired evidence, and cross-object attribution were previously accepted.

Exact-type, frozen-schema, and dominance-override regressions:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_v1_manifest_field_contract_is_explicit_and_frozen tests/logiv/test_recovery_records.py::test_loader_rejects_bool_integer_manifest_scalar_aliases tests/logiv/test_recovery_records.py::test_contract_and_envelopes_reject_float_integer_aliases tests/logiv/test_recovery_records.py::test_audited_snapshot_validates_dominance_override_schema
FFFFFFF                                                                  [100%]
7 failed in 0.10s
```

The failures covered the missing explicit v1 schema constant plus accepted Boolean/integer aliases, float/integer aliases, invented override kinds, duplicate overrides, and out-of-universe overrides.

Direct constructor container regression:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_manifest_constructor_rejects_list_aliases_for_tuple_fields
F                                                                        [100%]
1 failed in 0.06s
```

The failure was `DID NOT RAISE` for list aliases passed to tuple-only public fields.

### Review-fix GREEN evidence

```text
uv run pytest -q tests/logiv/test_libero_adapter.py::test_observation_store_owns_immutable_array_copies tests/logiv/test_libero_adapter.py::test_snapshot_and_synchronous_simulator_advance_cannot_mix_epochs tests/logiv/test_libero_adapter.py::test_required_facts_cannot_expand_the_frozen_registered_universe
...                                                                      [100%]
3 passed in 0.19s

uv run pytest -q tests/logiv/test_recovery_records.py::test_historical_evidence_recomputes_attempt_and_evidence_ids tests/logiv/test_recovery_records.py::test_historical_evidence_must_be_active_at_the_root_policy_step tests/logiv/test_recovery_records.py::test_contract_and_evidence_objects_are_closed_over_root_instances
...                                                                      [100%]
3 passed in 0.05s

uv run pytest -q tests/logiv/test_recovery_records.py::test_v1_manifest_field_contract_is_explicit_and_frozen tests/logiv/test_recovery_records.py::test_loader_rejects_bool_integer_manifest_scalar_aliases tests/logiv/test_recovery_records.py::test_contract_and_envelopes_reject_float_integer_aliases tests/logiv/test_recovery_records.py::test_audited_snapshot_validates_dominance_override_schema
.......                                                                  [100%]
7 passed in 0.06s

uv run pytest -q tests/logiv/test_recovery_records.py::test_manifest_constructor_rejects_list_aliases_for_tuple_fields
.                                                                        [100%]
1 passed in 0.05s
```

The atomic publication order test was a characterization test rather than a RED regression because the original implementation already had the required order:

```text
uv run pytest -q tests/logiv/test_recovery_records.py::test_atomic_publish_orders_lock_fsyncs_rename_and_parent_fsync
.                                                                        [100%]
1 passed in 0.05s
```

### Final verification after review fixes

Focused Task 3 suites:

```text
uv run pytest -q tests/logiv/test_recovery_records.py tests/logiv/test_libero_adapter.py
........................................................................ [ 72%]
...........................                                              [100%]
99 passed in 0.24s
```

Full repository suite:

```text
uv run pytest -q
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 91%]
....................                                                     [100%]
236 passed in 1.83s
```

Static sanity checks:

```text
git diff --check
uv run python -m compileall -q src/pi05_libero_repro/logiv
```

Both exited 0 with no output.

### Review-fix self-review

- Rechecked each confirmed Critical and Important finding against its production path and regression test.
- Confirmed both production synchronous `env.step` paths publish through the coordinated store transaction and no direct production `store.update` remains.
- Confirmed historical evidence is revalidated both during construction and loading, including root-time validity, object attribution, versioned IDs, canonical JSON, and contract timing.
- Confirmed direct construction and JSON loading share exact type rules, while JSON arrays are converted to tuples only after validation.
- Confirmed one PDDL ordering key is used for universe hashes, emitted fact partitions, grounding values, and holding-fact processing.
- Confirmed the final diff is limited to Task 3 production code, tests, and this report.

No additional correctness concern found during review-fix self-review.
