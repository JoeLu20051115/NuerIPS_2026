from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pi05_libero_repro.logiv.recovery_records import CollectionLabel, RecoverySplit
from pi05_libero_repro.logiv.recovery_splits import (
    CANONICAL_ROLE_REGISTRY_ID,
    CANONICAL_ROLE_REGISTRY_PATH,
    DatasetRole,
    FreshRecoveryLabel,
    RecoverySplitError,
    RoleAllocationManifest,
    RoleRegistry,
    TrainingManifest,
    append_role_allocation,
    build_training_manifest,
    load_training_manifest,
    make_role_allocation,
    preflight_role_allocation_outputs,
    preview_role_allocation_append,
    recovery_source_dataset_sha256,
    require_canonical_role_registry,
    validate_dataset_roles,
    validate_recovery_splits,
    validate_role_allocation,
    write_training_manifest,
)
from tests.logiv.test_recovery_records import (
    _canonical_json,
    _manifest,
    _observation,
    manifest_pending_actions,
)


ZERO_HASH = "0" * 64


def _json_hash(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _domain_hash(domain: bytes, value) -> str:
    return hashlib.sha256(
        domain + b"\0" + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _fresh_label_for(manifest, **overrides) -> FreshRecoveryLabel:
    values = {
        "schema_version": 1,
        "root_id": manifest.root_id,
        "simulator_state_sha256": manifest.simulator_state_sha256,
        "fresh_observation_sha256": "e" * 64,
        "deviation_event_id": manifest.deviation_event_id,
        "historical_evidence_sha256": _json_hash(
            list(manifest.historical_failure_evidence_json)
        ),
        "relevant_fact_sha256": manifest.relevant_fact_sha256,
        "fresh_fact_epoch_id": manifest.fact_epoch_id + 1,
        "fact_universe_sha256": manifest.fact_universe_sha256,
        "fresh_fact_evidence_hash": "",
        "graph_version": manifest.source_graph_version,
        "certificate_sha256": "f" * 64,
        "grounding_rule_sha256": manifest.grounding_rule_sha256,
        "event_detector_sha256": manifest.event_detector_sha256,
        "monitor_contract_sha256": manifest.monitor_contract_sha256,
        "labeler_version": "fresh-labeler-v1",
        "label_sha256": "",
    }
    values.update(overrides)
    if not values["fresh_fact_evidence_hash"]:
        values["fresh_fact_evidence_hash"] = _domain_hash(
            b"LOGIV_FRESH_FACT_EVIDENCE_V1",
            {
                "fact_epoch_id": values["fresh_fact_epoch_id"],
                "fact_universe_sha256": values["fact_universe_sha256"],
                "fresh_observation_sha256": values["fresh_observation_sha256"],
                "relevant_fact_sha256": values["relevant_fact_sha256"],
            },
        )
    payload = dict(values)
    payload.pop("label_sha256")
    values["label_sha256"] = _json_hash(payload)
    return FreshRecoveryLabel(**values)


def _allocation(manifests, roles, *, parent=ZERO_HASH):
    return make_role_allocation(
        manifests,
        roles=roles,
        allocation_version="allocation-v1",
        registry_parent_head_sha256=parent,
    )


def _registry(tmp_path: Path) -> RoleRegistry:
    registry = RoleRegistry(tmp_path / "role-registry.json", "phase0-registry")
    registry.initialize()
    return registry


def _source_datasets(*datasets):
    return {
        recovery_source_dataset_sha256(manifests): manifests
        for manifests in datasets
    }


def _build(manifests, allocation, registry, *, role, labels=()):
    return build_training_manifest(
        manifests,
        labels,
        role=role,
        source_dataset_sha256=recovery_source_dataset_sha256(manifests),
        role_allocation=allocation,
        role_registry_head=registry.read_head(),
        builder_version="builder-v1",
    )


@pytest.mark.parametrize(
    ("field", "overrides", "match"),
    [
        (
            "recovery_group_id",
            {"perturbation_seed": 11, "branch_seed": 2},
            "recovery_group_id",
        ),
        (
            "independence_unit_id",
            {
                "event_origin_parent_sha256": "1" * 64,
                "perturbation_family": "partial_place",
            },
            "independence_unit_id",
        ),
        (
            "initial_state_sha256",
            {
                "parent_trajectory_lineage_sha256": "2" * 64,
                "event_origin_parent_sha256": "3" * 64,
            },
            "initial_state_sha256",
        ),
        (
            "source_parent_snapshot_sha256",
            {
                "initial_state_sha256": "d" * 64,
                "parent_trajectory_lineage_sha256": "e" * 64,
                "event_origin_parent_sha256": "f" * 64,
                "simulator_state": np.array([3.0]),
            },
            "source_parent_snapshot_sha256",
        ),
        (
            "parent_trajectory_lineage_sha256",
            {
                "initial_state_sha256": "7" * 64,
                "source_parent_snapshot_sha256": "6" * 64,
                "event_origin_parent_sha256": "8" * 64,
                "simulator_state": np.array([4.0]),
            },
            "parent_trajectory_lineage_sha256",
        ),
        (
            "state_fingerprint",
            {
                "initial_state_sha256": "9" * 64,
                "source_parent_snapshot_sha256": "8" * 64,
                "parent_trajectory_lineage_sha256": "a" * 64,
                "event_origin_parent_sha256": "b" * 64,
                "simulator_state": np.array([1.00002]),
            },
            "state_fingerprint",
        ),
        (
            "perturbation_seed",
            {
                "initial_state_sha256": "c" * 64,
                "source_parent_snapshot_sha256": "b" * 64,
                "parent_trajectory_lineage_sha256": "d" * 64,
                "event_origin_parent_sha256": "e" * 64,
                "simulator_state": np.array([5.0]),
            },
            "perturbation_seed",
        ),
    ],
)
def test_validator_rejects_every_related_root_across_splits(field, overrides, match):
    train_overrides = {"split": RecoverySplit.TRAIN}
    if field == "state_fingerprint":
        train_overrides["simulator_state"] = np.array([1.00001])
    train = _manifest(**train_overrides)
    heldout = _manifest(split=RecoverySplit.HELDOUT, **overrides)
    assert getattr(train, field) == getattr(heldout, field)

    with pytest.raises(RecoverySplitError, match=match):
        validate_recovery_splits([train, heldout])


def test_validator_skips_null_optional_lineage_relations():
    train = _manifest(
        split=RecoverySplit.TRAIN,
        source_parent_snapshot_sha256=None,
        perturbation_seed=None,
    )
    heldout = _manifest(
        split=RecoverySplit.HELDOUT,
        initial_state_sha256="1" * 64,
        source_parent_snapshot_sha256=None,
        event_origin_parent_sha256="2" * 64,
        parent_trajectory_lineage_sha256="3" * 64,
        perturbation_seed=None,
        simulator_state=np.array([8.0]),
    )

    summary = validate_recovery_splits([train, heldout])

    assert summary["total"] == 2


def test_validator_reports_disjoint_dataset_counts():
    summary = validate_recovery_splits(
        [
            _manifest(
                split=RecoverySplit.TRAIN,
                collection_label=CollectionLabel.IMPORTED_FROZEN,
            ),
            _manifest(
                split=RecoverySplit.DEV,
                initial_state_sha256="1" * 64,
                source_parent_snapshot_sha256="f" * 64,
                parent_trajectory_lineage_sha256="2" * 64,
                event_origin_parent_sha256="3" * 64,
                perturbation_seed=12,
                simulator_state=np.array([8.0]),
                deviation_status="ANOMALY_CANDIDATE",
                deviation_event_id="event-2",
                historical_failure_evidence=(),
            ),
        ]
    )
    assert summary == {
        "TRAIN": 1,
        "DEV": 1,
        "HELDOUT": 0,
        "ANOMALY_CANDIDATE": 1,
        "CONFIRMED_DEVIATION": 1,
        "unique_independence_units": 2,
        "total": 2,
    }


def test_validator_rejects_duplicate_root_and_progress_only_confirmation():
    confirmed = _manifest()
    with pytest.raises(RecoverySplitError, match="duplicate root_id"):
        validate_recovery_splits([confirmed, confirmed])

    progress_only = _manifest(
        historical_failure_evidence=(),
        deviation_status="ANOMALY_CANDIDATE",
    )
    progress_only = replace(
        progress_only,
        deviation_status="CONFIRMED_DEVIATION",
        historical_failure_evidence_json=(
            _canonical_json(
                {
                    "evidence_kind": "PROGRESS_TIMEOUT",
                    "rule_id": "__progress_timeout__",
                }
            ),
        ),
    )
    with pytest.raises(RecoverySplitError, match="strong|PROGRESS_TIMEOUT"):
        validate_recovery_splits([progress_only])


def test_candidate_and_confirmation_for_event_keep_latched_lineage():
    candidate = _manifest(
        deviation_status="ANOMALY_CANDIDATE",
        historical_failure_evidence=(),
    )
    confirmation = _manifest(policy_step=20)
    invalid = replace(
        confirmation,
        deviation_event_id=candidate.deviation_event_id,
        event_origin_parent_sha256="f" * 64,
    )
    with pytest.raises(RecoverySplitError, match="semantic event|event_origin"):
        validate_recovery_splits([candidate, invalid])


def test_raw_candidate_and_stale_without_fresh_label_are_not_training_eligible(
    tmp_path: Path,
):
    candidate = _manifest(deviation_status="ANOMALY_CANDIDATE", historical_failure_evidence=())
    stale = _manifest(
        policy_step=16,
        deviation_event_id="event-2",
        event_origin_parent_sha256="1" * 64,
        certificate_state="STALE",
    )
    manifests = [candidate, stale]
    registry = _registry(tmp_path)
    allocation = _allocation(
        manifests,
        {candidate.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )

    with pytest.raises(RecoverySplitError, match="candidate|fresh label"):
        _build(manifests, allocation, registry, role=DatasetRole.TRAIN)


def test_matching_signed_fresh_label_admits_only_its_stale_root(tmp_path: Path):
    stale = _manifest(certificate_state="STALE")
    manifests = [stale]
    registry = _registry(tmp_path)
    allocation = _allocation(
        manifests,
        {stale.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )

    training = _build(
        manifests,
        allocation,
        registry,
        role=DatasetRole.TRAIN,
        labels=[_fresh_label_for(stale)],
    )

    assert tuple(item.root_id for item in training.entries) == (stale.root_id,)
    assert training.entries[0].fresh_label_sha256 is not None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"simulator_state_sha256": "1" * 64}, "state"),
        ({"deviation_event_id": "other-event"}, "event"),
        ({"event_detector_sha256": "2" * 64}, "detector"),
        ({"graph_version": "other-graph"}, "graph"),
        ({"fresh_fact_evidence_hash": "3" * 64}, "evidence"),
    ],
)
def test_stale_root_rejects_mismatched_fresh_label(tmp_path: Path, override, match):
    stale = _manifest(certificate_state="STALE")
    manifests = [stale]
    registry = _registry(tmp_path)
    allocation = _allocation(
        manifests,
        {stale.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )

    with pytest.raises(RecoverySplitError, match=match):
        _build(
            manifests,
            allocation,
            registry,
            role=DatasetRole.TRAIN,
            labels=[_fresh_label_for(stale, **override)],
        )


def test_role_allocation_must_cover_source_units_exactly_once():
    first = _manifest()
    second = _manifest(
        initial_state_sha256="1" * 64,
        source_parent_snapshot_sha256="f" * 64,
        parent_trajectory_lineage_sha256="2" * 64,
        event_origin_parent_sha256="3" * 64,
        perturbation_seed=12,
        simulator_state=np.array([8.0]),
    )
    complete = _allocation(
        [first, second],
        {
            first.independence_unit_id: DatasetRole.TRAIN,
            second.independence_unit_id: DatasetRole.DEV_SELECTION,
        },
    )
    incomplete_entries = complete.entries[:-1]
    payload = asdict(complete)
    payload["entries"] = [asdict(item) for item in incomplete_entries]
    payload.pop("allocation_sha256")
    incomplete = replace(
        complete,
        entries=incomplete_entries,
        allocation_sha256=_json_hash(payload),
    )

    with pytest.raises(RecoverySplitError, match="complete|missing"):
        validate_role_allocation([first, second], incomplete)


def test_role_allocation_rejects_bool_schema_alias():
    source = _manifest()
    allocation = _allocation(
        [source], {source.independence_unit_id: DatasetRole.TRAIN}
    )
    invalid = replace(allocation, schema_version=True, allocation_sha256="")
    payload = asdict(invalid)
    payload.pop("allocation_sha256")
    invalid = replace(invalid, allocation_sha256=_json_hash(payload))

    with pytest.raises(RecoverySplitError, match="schema"):
        validate_role_allocation([source], invalid)


def test_role_allocation_rejects_related_units_across_roles():
    train = _manifest()
    paper = _manifest(
        split=RecoverySplit.HELDOUT,
        collection_label=CollectionLabel.IMPORTED_FROZEN,
        event_origin_parent_sha256="1" * 64,
        parent_trajectory_lineage_sha256="2" * 64,
    )
    assert train.initial_state_sha256 == paper.initial_state_sha256

    with pytest.raises(RecoverySplitError, match="initial_state_sha256"):
        _allocation(
            [train, paper],
            {
                train.independence_unit_id: DatasetRole.TRAIN,
                paper.independence_unit_id: DatasetRole.PAPER_CONFIRMATION,
            },
        )


def test_role_must_match_frozen_split_and_development_collection_label():
    heldout = _manifest(
        split=RecoverySplit.HELDOUT,
        collection_label=CollectionLabel.IMPORTED_FROZEN,
    )
    with pytest.raises(RecoverySplitError, match="role|split"):
        _allocation(
            [heldout], {heldout.independence_unit_id: DatasetRole.TRAIN}
        )

    development = _manifest()
    with pytest.raises(RecoverySplitError, match="DEV_COLLECTION|permit|paper"):
        _allocation(
            [development],
            {development.independence_unit_id: DatasetRole.PERMIT_CALIBRATION},
        )


def test_permit_and_paper_roles_require_frozen_code_hashes():
    heldout = _manifest(
        split=RecoverySplit.HELDOUT,
        collection_label=CollectionLabel.IMPORTED_FROZEN,
    )
    mutable = replace(heldout, base_checkpoint_sha256="mutable")

    with pytest.raises(RecoverySplitError, match="frozen|checkpoint|hash"):
        _allocation(
            [mutable],
            {mutable.independence_unit_id: DatasetRole.PERMIT_CALIBRATION},
        )


def test_role_registry_cas_and_historical_repackaging_are_rejected(tmp_path: Path):
    source = _manifest()
    registry = _registry(tmp_path)
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    with pytest.raises(RecoverySplitError, match="expected head"):
        append_role_allocation(registry, allocation, expected_head="2" * 64)
    assert registry.read_head().revision == 0

    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )
    repackaged = replace(
        allocation,
        source_dataset_sha256="3" * 64,
        entries=tuple(
            replace(item, source_dataset_sha256="3" * 64)
            for item in allocation.entries
        ),
        registry_parent_head_sha256=registry.read_head_sha256(),
        allocation_sha256="4" * 64,
    )
    payload = asdict(repackaged)
    payload.pop("allocation_sha256")
    repackaged = replace(repackaged, allocation_sha256=_json_hash(payload))
    with pytest.raises(RecoverySplitError, match="historical|independence_unit_id"):
        append_role_allocation(
            registry, repackaged, expected_head=registry.read_head_sha256()
        )


def test_role_registry_genesis_can_be_previewed_without_mutation(tmp_path: Path):
    registry = RoleRegistry(tmp_path / "role-registry.json", "phase0-registry")

    preview = registry.genesis_head()

    assert not registry.path.exists()
    assert registry.initialize() == preview


def test_alternate_project_registry_path_or_id_is_rejected(tmp_path: Path):
    with pytest.raises(RecoverySplitError, match="canonical.*path"):
        require_canonical_role_registry(
            tmp_path / "alternate.json", CANONICAL_ROLE_REGISTRY_ID
        )
    with pytest.raises(RecoverySplitError, match="canonical.*ID"):
        require_canonical_role_registry(
            CANONICAL_ROLE_REGISTRY_PATH, "alternate-registry"
        )


def test_registry_rejects_allocation_with_an_extra_declared_source_unit(
    tmp_path: Path,
):
    source = _manifest()
    registry = _registry(tmp_path)
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    invalid = replace(
        allocation,
        source_independence_unit_ids=tuple(
            sorted(allocation.source_independence_unit_ids + ("z",))
        ),
        allocation_sha256="",
    )
    payload = asdict(invalid)
    payload.pop("allocation_sha256")
    invalid = replace(invalid, allocation_sha256=_json_hash(payload))

    with pytest.raises(RecoverySplitError, match="complete|source unit"):
        append_role_allocation(
            registry, invalid, expected_head=registry.read_head_sha256()
        )


def test_registry_rejects_conflicting_relation_roles_within_one_append(
    tmp_path: Path,
):
    train = _manifest()
    dev = _manifest(
        initial_state_sha256="1" * 64,
        source_parent_snapshot_sha256="f" * 64,
        parent_trajectory_lineage_sha256="2" * 64,
        event_origin_parent_sha256="3" * 64,
        perturbation_seed=12,
        simulator_state=np.array([8.0]),
        deviation_event_id="event-2",
    )
    registry = _registry(tmp_path)
    allocation = _allocation(
        [train, dev],
        {
            train.independence_unit_id: DatasetRole.TRAIN,
            dev.independence_unit_id: DatasetRole.DEV_SELECTION,
        },
        parent=registry.read_head_sha256(),
    )
    shared = next(
        key
        for key in allocation.entries[0].role_isolation_keys
        if key.startswith("initial_state_sha256:")
    )
    forged_entries = (
        allocation.entries[0],
        replace(allocation.entries[1], role_isolation_keys=(shared,)),
    )
    forged = replace(
        allocation, entries=forged_entries, allocation_sha256=""
    )
    payload = asdict(forged)
    payload.pop("allocation_sha256")
    forged = replace(forged, allocation_sha256=_json_hash(payload))

    with pytest.raises(RecoverySplitError, match="initial_state_sha256|role"):
        append_role_allocation(
            registry, forged, expected_head=registry.read_head_sha256()
        )


def test_conflicting_derived_output_is_rejected_before_registry_cas(
    tmp_path: Path,
):
    source = _manifest()
    registry = _registry(tmp_path)
    current = registry.read_head()
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=current.head_sha256,
    )
    predicted = preview_role_allocation_append(
        current, allocation, expected_head=current.head_sha256
    )
    output = tmp_path / "allocation.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(RecoverySplitError, match="immutable|exists"):
        preflight_role_allocation_outputs(
            output,
            allocation,
            tmp_path / "run.json",
            role_registry_head_sha256=predicted.head_sha256,
        )

    assert registry.read_head() == current


def test_dataset_role_validator_requires_exact_allocation_and_registry_head(
    tmp_path: Path,
):
    source = _manifest()
    registry = _registry(tmp_path)
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )
    training = _build(
        [source], allocation, registry, role=DatasetRole.TRAIN
    )
    assert validate_dataset_roles(
        [training],
        role_allocations=[allocation],
        role_registry_head=registry.read_head(),
        source_datasets=_source_datasets([source]),
    ) == {"TRAIN": 1, "total": 1}

    stale_head = replace(registry.read_head(), head_sha256="5" * 64)
    with pytest.raises(RecoverySplitError, match="registry|head"):
        validate_dataset_roles(
            [training],
            role_allocations=[allocation],
            role_registry_head=stale_head,
            source_datasets=_source_datasets([source]),
        )


def test_dataset_role_validator_detects_an_omitted_allocated_unit(tmp_path: Path):
    train = _manifest()
    dev = _manifest(
        initial_state_sha256="1" * 64,
        source_parent_snapshot_sha256="f" * 64,
        parent_trajectory_lineage_sha256="2" * 64,
        event_origin_parent_sha256="3" * 64,
        perturbation_seed=12,
        simulator_state=np.array([8.0]),
        deviation_event_id="event-2",
    )
    manifests = [train, dev]
    registry = _registry(tmp_path)
    allocation = _allocation(
        manifests,
        {
            train.independence_unit_id: DatasetRole.TRAIN,
            dev.independence_unit_id: DatasetRole.DEV_SELECTION,
        },
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )
    train_manifest = _build(
        manifests, allocation, registry, role=DatasetRole.TRAIN
    )

    with pytest.raises(RecoverySplitError, match="complete|omitted|missing"):
        validate_dataset_roles(
            [train_manifest],
            role_allocations=[allocation],
            role_registry_head=registry.read_head(),
            source_datasets=_source_datasets(manifests),
        )


def test_dataset_role_validator_rejects_omitted_historical_allocation(
    tmp_path: Path,
):
    first = _manifest()
    second = _manifest(
        initial_state_sha256="1" * 64,
        source_parent_snapshot_sha256="f" * 64,
        parent_trajectory_lineage_sha256="2" * 64,
        event_origin_parent_sha256="3" * 64,
        perturbation_seed=12,
        simulator_state=np.array([8.0]),
        deviation_event_id="event-2",
    )
    registry = _registry(tmp_path)
    first_allocation = _allocation(
        [first],
        {first.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, first_allocation, expected_head=registry.read_head_sha256()
    )
    second_allocation = _allocation(
        [second],
        {second.independence_unit_id: DatasetRole.DEV_SELECTION},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, second_allocation, expected_head=registry.read_head_sha256()
    )
    second_manifest = _build(
        [second],
        second_allocation,
        registry,
        role=DatasetRole.DEV_SELECTION,
    )

    with pytest.raises(RecoverySplitError, match="historical|complete|registry"):
        validate_dataset_roles(
            [second_manifest],
            role_allocations=[second_allocation],
            role_registry_head=registry.read_head(),
            source_datasets=_source_datasets([second]),
        )


def test_dataset_role_validator_rejects_root_outside_immutable_source(
    tmp_path: Path,
):
    source = _manifest()
    registry = _registry(tmp_path)
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )
    training = _build(
        [source], allocation, registry, role=DatasetRole.TRAIN
    )
    forged_entry = replace(training.entries[0], root_id="not-a-source-root")
    forged = replace(
        training, entries=(forged_entry,), manifest_sha256=""
    )
    payload = asdict(forged)
    payload.pop("manifest_sha256")
    forged = replace(forged, manifest_sha256=_json_hash(payload))

    with pytest.raises(RecoverySplitError, match="root_id|source"):
        validate_dataset_roles(
            [forged],
            role_allocations=[allocation],
            role_registry_head=registry.read_head(),
            source_datasets=_source_datasets([source]),
        )


def test_training_manifest_writer_refuses_conflicting_overwrite(
    tmp_path: Path,
):
    source = _manifest()
    registry = _registry(tmp_path)
    allocation = _allocation(
        [source],
        {source.independence_unit_id: DatasetRole.TRAIN},
        parent=registry.read_head_sha256(),
    )
    append_role_allocation(
        registry, allocation, expected_head=registry.read_head_sha256()
    )
    manifest = _build(
        [source], allocation, registry, role=DatasetRole.TRAIN
    )
    output = tmp_path / "training.json"
    write_training_manifest(output, manifest)
    changed = replace(manifest, builder_version="builder-v2", manifest_sha256="")
    payload = asdict(changed)
    payload.pop("manifest_sha256")
    changed = replace(changed, manifest_sha256=_json_hash(payload))

    with pytest.raises(RecoverySplitError, match="immutable|exists"):
        write_training_manifest(output, changed)


def test_training_loader_refuses_a_raw_root_directory(tmp_path: Path):
    raw = tmp_path / "raw-root"
    raw.mkdir()
    (raw / "recovery_root.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RecoverySplitError, match="training manifest|raw root|directory"):
        load_training_manifest(raw)


def test_validation_cli_rejects_an_empty_dataset(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_recovery_dataset.py",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "empty" in result.stderr.lower()
