from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pi05_libero_repro.logiv import recovery_records
from pi05_libero_repro.logiv.model import Fact, FactSnapshot, TruthValue
from pi05_libero_repro.protocol import BaseRequestEnvelopeV1
from pi05_libero_repro.logiv.recovery_records import (
    CollectionLabel,
    RecoveryRootManifest,
    RecoverySplit,
    find_orphan_root_temps,
    load_recovery_root,
    make_recovery_root_manifest,
    write_recovery_root,
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fact_universe_sha256(version: str, facts: frozenset[Fact]) -> str:
    payload = {
        "facts": [fact.pddl() for fact in sorted(facts)],
        "version": version,
    }
    return hashlib.sha256(
        b"LOGIV_FACT_UNIVERSE_V1\0" + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _update_bytes(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _update_array(digest, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    _update_bytes(digest, array.dtype.str.encode("ascii"))
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.nbytes.to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))


def _observation_sha256(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"LOGIV_OBSERVATION_V1\0")
    for key in sorted(observation):
        _update_bytes(digest, key.encode("utf-8"))
        _update_array(digest, observation[key])
    return digest.hexdigest()


def _observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eye_in_hand_image": np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.3], dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float64),
    }


def manifest_pending_actions() -> np.ndarray:
    return np.arange(14, dtype=np.float32).reshape(2, 7) / 10.0


def _audited_snapshot(
    *,
    true: tuple[Fact, ...] = (Fact("at", ("book_1", "table")),),
    false: tuple[Fact, ...] = (Fact("in", ("book_1", "caddy")),),
    unknown: tuple[Fact, ...] = (Fact("holding", ("book_1",)),),
    epoch_id: int = 7,
    observation: dict[str, np.ndarray] | None = None,
) -> FactSnapshot:
    observation = _observation() if observation is None else observation
    universe = frozenset(true + false + unknown)
    version = "libero-grounding-v1"
    values = [(fact.pddl(), TruthValue.TRUE.value) for fact in true]
    values.extend((fact.pddl(), TruthValue.FALSE.value) for fact in false)
    values.extend((fact.pddl(), TruthValue.UNKNOWN.value) for fact in unknown)
    payload_json = _canonical_json(
        {
            "dominance_overrides": [],
            "epoch_id": epoch_id,
            "observation_hash": _observation_sha256(observation),
            "values": sorted(values),
        }
    )
    return FactSnapshot(
        epoch_id=epoch_id,
        true_facts=frozenset(true),
        false_facts=frozenset(false),
        evidence_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        fact_universe=universe,
        fact_universe_version=version,
        fact_universe_sha256=_fact_universe_sha256(version, universe),
        evidence_payload_json=payload_json,
    )


def _monitor_contract_json(
    *, ttl: int = 5, contract_id: str = "r2m-monitor-evidence-v1-task-5"
) -> str:
    contract = {
        "contract_id": contract_id,
        "task_id": 5,
        "object_ids": ["book_1"],
        "nominal_source_facts": ["(at book_1 table)"],
        "abnormal_support_surfaces": ["floor"],
        "task_relevant_effects": ["(in book_1 caddy)"],
        "tracker_version": "tracker-v1",
        "action_event_rules": [
            {
                "rule_id": "place_book_1",
                "object_id": "book_1",
                "attempt_kind": "PLACE",
                "attempted_effect": "(in book_1 caddy)",
                "source_region": "table",
                "destination_region": "caddy",
                "gripper_close_threshold": -0.5,
                "gripper_open_threshold": 0.5,
                "contact_min_count": 1,
                "motion_correlation_min": 0.8,
                "region_distance_max": 0.1,
                "effect_due_after_policy_steps": 3,
                "evidence_ttl_policy_steps": ttl,
                "manipulation_attribution_ttl_policy_steps": 7,
            }
        ],
        "monitor_interval_steps": 5,
        "confirmation_count": 3,
        "settling_grace_observations": 2,
        "progress_window_observations": 4,
        "progress_evidence_ttl_policy_steps": 20,
        "goal_regression_evidence_ttl_policy_steps": 20,
        "max_active_attempts_per_object": 8,
        "max_attempt_records_per_episode": 128,
        "max_evidence_records_per_episode": 128,
        "grounding_rule_sha256": "a" * 64,
        "event_detector_sha256": "d" * 64,
    }
    contract["contract_sha256"] = hashlib.sha256(
        _canonical_json(contract).encode("utf-8")
    ).hexdigest()
    return _canonical_json(contract)


def _evidence(**overrides) -> dict:
    record = {
        "evidence_id": "e" * 64,
        "evidence_kind": "ATTEMPTED_EFFECT_TIMEOUT",
        "rule_id": "place_book_1",
        "object_id": "book_1",
        "attempted_effect": "(in book_1 caddy)",
        "source_region": "table",
        "destination_region": "caddy",
        "attempt_id": "f" * 64,
        "start_policy_step": 2,
        "effect_due_policy_step": 5,
        "emitted_policy_step": 5,
        "evidence_expires_policy_step": 10,
        "supporting_transition_hashes": ("1" * 64, "2" * 64),
        "detector_sha256": "d" * 64,
    }
    record.update(overrides)
    return record


def _envelope(*, inference_index: int) -> str:
    return BaseRequestEnvelopeV1(
        episode_seed=17,
        inference_index=inference_index,
        policy_client_config_sha256="c" * 64,
    ).canonical_json()


def _manifest(**overrides) -> RecoveryRootManifest:
    observation = overrides.pop("observation", _observation())
    simulator_state = overrides.pop(
        "simulator_state", np.array([1.0, 2.5, -3.0], dtype=np.float64)
    )
    pending_base_actions = overrides.pop(
        "pending_base_actions", manifest_pending_actions()
    )
    snapshot = overrides.pop(
        "snapshot", _audited_snapshot(observation=observation)
    )
    values = {
        "split": RecoverySplit.DEV,
        "collection_label": CollectionLabel.DEV_COLLECTION,
        "task_id": 5,
        "episode_idx": 3,
        "scene_sha256": "3" * 64,
        "object_instance_ids": ("book_1",),
        "initial_state_sha256": "4" * 64,
        "source_parent_snapshot_sha256": "5" * 64,
        "event_origin_parent_sha256": "6" * 64,
        "parent_trajectory_lineage_sha256": "7" * 64,
        "base_prompt_sha256": "8" * 64,
        "base_checkpoint_sha256": "9" * 64,
        "policy_client_config_sha256": "c" * 64,
        "perturbation_family": "pose-jitter",
        "perturbation_seed": 10,
        "branch_seed": 1,
        "master_seed": 11,
        "policy_seed": 17,
        "simulator_seed": 13,
        "trigger_class": "UNPLANNED_SUPPORT_STABLE",
        "deviation_status": "CONFIRMED_DEVIATION",
        "deviation_event_id": "event-1",
        "historical_failure_evidence": (_evidence(),),
        "relevant_fact_sha256": "b" * 64,
        "source_graph_version": "graph-v1",
        "source_observation_generation": 4,
        "certificate_state": "CURRENT",
        "grounding_rule_sha256": "a" * 64,
        "event_detector_sha256": "d" * 64,
        "monitor_contract_json": _monitor_contract_json(),
        "policy_step": 15,
        "policy_request_generation": 0,
        "base_policy_request_count": 1,
        "active_base_request_index": 0,
        "next_base_request_index": 1,
        "active_base_request_envelope_json": _envelope(inference_index=0),
        "next_base_replay_envelope_json": _envelope(inference_index=1),
        "policy_replay_contract_sha256": "0" * 64,
        "base_action_response_size": 5,
        "base_action_chunk_size": 5,
        "pending_base_action_offset": 3,
        "simulator_state": simulator_state,
        "observation": observation,
        "pending_base_actions": pending_base_actions,
        "base_action_prefix_sha256": "2" * 64,
        "snapshot": snapshot,
    }
    values.update(overrides)
    return make_recovery_root_manifest(**values)


def _round_trip(tmp_path: Path, manifest: RecoveryRootManifest):
    observation = _observation()
    state = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    pending = manifest_pending_actions()
    artifacts = write_recovery_root(tmp_path, manifest, state, observation, pending)
    return load_recovery_root(artifacts.directory)


def _write_manifest_root(tmp_path: Path, manifest: RecoveryRootManifest):
    return write_recovery_root(
        tmp_path,
        manifest,
        np.array([1.0, 2.5, -3.0], dtype=np.float64),
        _observation(),
        manifest_pending_actions(),
    )


def _rewrite_manifest(path: Path, payload: dict) -> None:
    path.write_text(_canonical_json(payload), encoding="utf-8")


def test_group_id_ignores_branch_seeds_but_root_id_does_not() -> None:
    first = _manifest(perturbation_seed=10, branch_seed=1)
    second = _manifest(perturbation_seed=11, branch_seed=2)
    assert first.recovery_group_id == second.recovery_group_id
    assert first.root_id != second.root_id


def test_candidate_upgrade_gets_a_new_artifact_id_but_not_a_new_independent_group() -> None:
    candidate = _manifest(policy_step=15, deviation_status="ANOMALY_CANDIDATE")
    confirmed = _manifest(policy_step=20, deviation_status="CONFIRMED_DEVIATION")
    assert candidate.root_id != confirmed.root_id
    assert candidate.recovery_group_id == confirmed.recovery_group_id
    assert candidate.independence_unit_id == confirmed.independence_unit_id
    assert candidate.event_origin_parent_sha256 == confirmed.event_origin_parent_sha256


def test_parent_snapshots_from_one_trajectory_are_one_independence_unit() -> None:
    early = _manifest(event_origin_parent_sha256="1" * 64, policy_step=15)
    late = _manifest(event_origin_parent_sha256="2" * 64, policy_step=25)
    assert early.recovery_group_id != late.recovery_group_id
    assert early.independence_unit_id == late.independence_unit_id


def test_recovery_root_round_trip_preserves_exact_physics_and_observation(tmp_path: Path) -> None:
    simulator_state = np.array([1.0, 2.5, -3.0], dtype=np.float64)
    observation = _observation()
    pending_actions = manifest_pending_actions()
    manifest = _manifest(
        simulator_state=simulator_state,
        observation=observation,
        pending_base_actions=pending_actions,
    )
    artifacts = write_recovery_root(
        tmp_path, manifest, simulator_state, observation, pending_actions
    )
    loaded_manifest, loaded_state = load_recovery_root(artifacts.directory)
    assert loaded_manifest == manifest
    np.testing.assert_array_equal(loaded_state["simulator_state"], simulator_state)
    for key, value in observation.items():
        np.testing.assert_array_equal(loaded_state[key], value)
    np.testing.assert_array_equal(
        loaded_state["pending_base_actions"], pending_actions
    )


def test_audited_fact_universe_round_trips_unknown_and_rejects_conflicts(tmp_path: Path) -> None:
    snapshot = _audited_snapshot()
    manifest = _manifest(snapshot=snapshot)
    loaded, _ = _round_trip(tmp_path, manifest)
    assert loaded.unknown_facts == ("(holding book_1)",)
    with pytest.raises(ValueError, match="TRUE and FALSE|partition|universe"):
        _audited_snapshot(
            true=(Fact("at", ("book_1", "table")),),
            false=(Fact("at", ("book_1", "table")),),
            unknown=(),
        )


def test_legacy_snapshot_is_rejected_only_at_recovery_root_boundary() -> None:
    legacy = FactSnapshot(
        epoch_id=0,
        true_facts=frozenset({Fact("handempty")}),
        false_facts=frozenset(),
        evidence_hash="sha256:legacy",
    )
    assert legacy.truth(Fact("handempty")) is TruthValue.TRUE
    with pytest.raises(ValueError, match="audit|universe|evidence"):
        _manifest(snapshot=legacy)


def test_standalone_root_rejects_contract_or_evidence_timing_tampering(tmp_path: Path) -> None:
    first = _write_manifest_root(tmp_path / "contract", _manifest())
    payload = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    payload["monitor_contract_json"] = _monitor_contract_json(ttl=6)
    _rewrite_manifest(first.manifest_json, payload)
    with pytest.raises(ValueError, match="contract|hash"):
        load_recovery_root(first.directory)

    second = _write_manifest_root(tmp_path / "evidence", _manifest())
    payload = json.loads(second.manifest_json.read_text(encoding="utf-8"))
    evidence = json.loads(payload["historical_failure_evidence_json"][0])
    evidence["effect_due_policy_step"] += 1
    payload["historical_failure_evidence_json"][0] = _canonical_json(evidence)
    _rewrite_manifest(second.manifest_json, payload)
    with pytest.raises(ValueError, match="due|rule"):
        load_recovery_root(second.directory)


def test_monitor_contract_rejects_an_unknown_versioned_schema() -> None:
    contract = json.loads(_monitor_contract_json(contract_id="opaque-contract-id"))
    contract.pop("contract_sha256")
    contract["schema_version"] = 2
    contract["contract_sha256"] = hashlib.sha256(
        _canonical_json(contract).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="contract schema|field"):
        _manifest(monitor_contract_json=_canonical_json(contract))


def test_loader_rejects_npz_bytes_and_observation_partition_tampering(tmp_path: Path) -> None:
    artifacts = _write_manifest_root(tmp_path / "state", _manifest())
    with np.load(artifacts.state_npz, allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    state["simulator_state"][0] += 0.25
    np.savez(artifacts.state_npz, **state)
    with pytest.raises(ValueError, match="simulator|hash"):
        load_recovery_root(artifacts.directory)

    artifacts = _write_manifest_root(tmp_path / "facts", _manifest())
    payload = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    payload["unknown_facts"].append(payload["true_facts"][0])
    _rewrite_manifest(artifacts.manifest_json, payload)
    with pytest.raises(ValueError, match="partition|duplicate|universe"):
        load_recovery_root(artifacts.directory)


def test_duplicate_evidence_or_envelope_identity_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate|attempt"):
        _manifest(historical_failure_evidence=(_evidence(), _evidence()))
    with pytest.raises(ValueError, match="envelope|index"):
        _manifest(active_base_request_envelope_json=_envelope(inference_index=1))


def test_missing_replay_contract_or_envelopes_are_explicitly_ineligible() -> None:
    missing_contract = _manifest(policy_replay_contract_sha256=None)
    assert missing_contract.live_base_continuation_eligible is False
    assert "replay contract" in missing_contract.live_base_continuation_ineligibility_reason
    missing_envelope = _manifest(next_base_replay_envelope_json=None)
    assert missing_envelope.next_base_replay_envelope_sha256 is None
    assert missing_envelope.live_base_continuation_eligible is False
    assert "envelope" in missing_envelope.live_base_continuation_ineligibility_reason


def test_invalid_arrays_and_synthetic_parent_provenance_are_rejected() -> None:
    observation = _observation()
    observation["metadata"] = "not an array"
    with pytest.raises((TypeError, ValueError), match="NumPy|array|observation"):
        _manifest(observation=observation)
    with pytest.raises(ValueError, match="source parent|synthetic"):
        _manifest(source_parent_snapshot_sha256=None, perturbation_seed=10)
    with pytest.raises(ValueError, match="request generation|Phase 0|zero"):
        _manifest(policy_request_generation=1)


def test_flattened_state_and_base_chunk_shape_and_response_size_are_verified() -> None:
    with pytest.raises(ValueError, match="flat|simulator state"):
        _manifest(simulator_state=np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="pending Base actions|shape|7"):
        _manifest(pending_base_actions=np.zeros((2, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="response|chunk"):
        _manifest(base_action_response_size=4, base_action_chunk_size=5)
    with pytest.raises(ValueError, match="response|request"):
        _manifest(base_action_response_size=None)


def test_identical_retry_succeeds_but_corrupt_existing_root_does_not(tmp_path: Path) -> None:
    manifest = _manifest()
    first = _write_manifest_root(tmp_path, manifest)
    second = _write_manifest_root(tmp_path, manifest)
    assert second == first
    payload = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    payload["policy_step"] += 1
    _rewrite_manifest(first.manifest_json, payload)
    with pytest.raises(ValueError, match="existing|corrupt|root"):
        _write_manifest_root(tmp_path, manifest)


def test_identical_retry_compares_exact_array_bytes(tmp_path: Path) -> None:
    state = np.array([np.nan, -0.0, 3.0], dtype=np.float64)
    manifest = _manifest(simulator_state=state)
    first = write_recovery_root(
        tmp_path, manifest, state, _observation(), manifest_pending_actions()
    )
    second = write_recovery_root(
        tmp_path, manifest, state.copy(), _observation(), manifest_pending_actions()
    )
    assert second == first


def test_crash_before_directory_publish_never_exposes_half_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_oserror(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated crash")

    manifest = _manifest()
    monkeypatch.setattr(recovery_records, "_publish_directory", raise_oserror)
    with pytest.raises(OSError):
        _write_manifest_root(tmp_path, manifest)
    assert not (tmp_path / manifest.root_id).exists()
    assert find_orphan_root_temps(tmp_path)


def test_manifest_json_has_exact_public_fields_after_round_trip(tmp_path: Path) -> None:
    artifacts = _write_manifest_root(tmp_path, _manifest())
    payload = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert set(payload) == set(asdict(_manifest()))
    payload["unexpected"] = True
    _rewrite_manifest(artifacts.manifest_json, payload)
    with pytest.raises(ValueError, match="field|schema"):
        load_recovery_root(artifacts.directory)
