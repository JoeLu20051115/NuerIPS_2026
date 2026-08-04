from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from pi05_libero_repro.logiv.model import (
    FactSnapshot,
    fact_pddl_sort_key,
    fact_universe_sha256,
    parse_pddl_fact,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_REQUIRED_OBSERVATION_KEYS = frozenset(
    {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
)
_STATE_KEYS = frozenset({"simulator_state", "pending_base_actions"})
_DEVIATION_STATUSES = frozenset({"ANOMALY_CANDIDATE", "CONFIRMED_DEVIATION"})
_CERTIFICATE_STATES = frozenset({"CURRENT", "STALE", "RECONCILED"})
_ACTION_EVIDENCE_KINDS = frozenset(
    {"ATTEMPTED_EFFECT_TIMEOUT", "ABNORMAL_TRANSFER_AFTER_MANIPULATION"}
)
_FACT_EVIDENCE_RULES = {
    "GOAL_REGRESSION": "__goal_regression__",
    "PROGRESS_TIMEOUT": "__progress_timeout__",
}
_RULE_FIELDS = frozenset(
    {
        "rule_id",
        "object_id",
        "attempt_kind",
        "attempted_effect",
        "source_region",
        "destination_region",
        "gripper_close_threshold",
        "gripper_open_threshold",
        "contact_min_count",
        "motion_correlation_min",
        "region_distance_max",
        "effect_due_after_policy_steps",
        "evidence_ttl_policy_steps",
        "manipulation_attribution_ttl_policy_steps",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "contract_id",
        "task_id",
        "object_ids",
        "nominal_source_facts",
        "abnormal_support_surfaces",
        "task_relevant_effects",
        "tracker_version",
        "action_event_rules",
        "monitor_interval_steps",
        "confirmation_count",
        "settling_grace_observations",
        "progress_window_observations",
        "progress_evidence_ttl_policy_steps",
        "goal_regression_evidence_ttl_policy_steps",
        "max_active_attempts_per_object",
        "max_attempt_records_per_episode",
        "max_evidence_records_per_episode",
        "grounding_rule_sha256",
        "event_detector_sha256",
        "contract_sha256",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "evidence_kind",
        "rule_id",
        "object_id",
        "attempted_effect",
        "source_region",
        "destination_region",
        "attempt_id",
        "start_policy_step",
        "effect_due_policy_step",
        "emitted_policy_step",
        "evidence_expires_policy_step",
        "supporting_transition_hashes",
        "detector_sha256",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _domain_json_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(
        domain + b"\0" + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _require_sha256(name: str, value: str | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")


def _require_nonnegative(name: str, value: int | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _update_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _validate_array(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype.hasobject:
        raise ValueError(f"{name} object arrays are forbidden")
    return np.ascontiguousarray(value)


def _update_array(digest: Any, value: np.ndarray) -> None:
    array = _validate_array("array", value)
    _update_bytes(digest, array.dtype.str.encode("ascii"))
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.nbytes.to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))


def _array_sha256(domain: bytes, value: np.ndarray) -> str:
    digest = hashlib.sha256(domain + b"\0")
    _update_array(digest, value)
    return digest.hexdigest()


def _validated_observation(
    observation: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping of NumPy arrays")
    keys = set(observation)
    missing = _REQUIRED_OBSERVATION_KEYS - keys
    if missing:
        raise ValueError(f"observation is missing required arrays: {sorted(missing)}")
    if keys & _STATE_KEYS:
        raise ValueError("observation keys conflict with recovery state arrays")
    result: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if not isinstance(key, str) or not _OBSERVATION_KEY.fullmatch(key):
            raise ValueError(f"undeclared observation key: {key!r}")
        result[key] = _validate_array(f"observation[{key!r}]", value)
    return result


def observation_sha256(observation: Mapping[str, np.ndarray]) -> str:
    values = _validated_observation(observation)
    digest = hashlib.sha256(b"LOGIV_OBSERVATION_V1\0")
    for key in sorted(values):
        _update_bytes(digest, key.encode("utf-8"))
        _update_array(digest, values[key])
    return digest.hexdigest()


class RecoverySplit(str, Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    HELDOUT = "HELDOUT"


class CollectionLabel(str, Enum):
    DEV_COLLECTION = "DEV_COLLECTION"
    IMPORTED_FROZEN = "IMPORTED_FROZEN"


@dataclass(frozen=True)
class RecoveryRootManifest:
    schema_version: int
    root_id: str
    recovery_group_id: str
    independence_unit_id: str
    split: RecoverySplit
    collection_label: CollectionLabel
    task_id: int
    episode_idx: int
    scene_sha256: str
    object_instance_ids: tuple[str, ...]
    initial_state_sha256: str
    source_parent_snapshot_sha256: str | None
    event_origin_parent_sha256: str
    parent_trajectory_lineage_sha256: str
    base_prompt_sha256: str
    base_checkpoint_sha256: str
    policy_client_config_sha256: str
    perturbation_family: str
    perturbation_seed: int | None
    branch_seed: int | None
    master_seed: int
    policy_seed: int
    simulator_seed: int
    trigger_class: str
    deviation_status: str
    deviation_event_id: str
    historical_failure_evidence_json: tuple[str, ...]
    relevant_fact_sha256: str
    source_graph_version: str
    source_observation_generation: int
    certificate_state: str
    grounding_rule_sha256: str
    event_detector_sha256: str
    monitor_contract_sha256: str
    monitor_contract_json: str
    policy_step: int
    policy_request_generation: int
    base_policy_request_count: int
    active_base_request_index: int | None
    next_base_request_index: int
    active_base_request_envelope_json: str | None
    active_base_request_envelope_sha256: str | None
    next_base_replay_envelope_json: str | None
    next_base_replay_envelope_sha256: str | None
    policy_replay_contract_sha256: str | None
    base_action_response_size: int | None
    base_action_chunk_size: int
    pending_base_action_offset: int
    pending_base_action_count: int
    pending_base_actions_sha256: str
    live_base_continuation_eligible: bool
    live_base_continuation_ineligibility_reason: str | None
    simulator_state_sha256: str
    observation_sha256: str
    state_fingerprint: str
    base_action_prefix_sha256: str
    fact_epoch_id: int
    fact_universe_version: str
    fact_universe_sha256: str
    fact_evidence_hash: str
    fact_evidence_payload_json: str
    true_facts: tuple[str, ...]
    false_facts: tuple[str, ...]
    unknown_facts: tuple[str, ...]


RECOVERY_ROOT_MANIFEST_V1_FIELDS = (
    "schema_version",
    "root_id",
    "recovery_group_id",
    "independence_unit_id",
    "split",
    "collection_label",
    "task_id",
    "episode_idx",
    "scene_sha256",
    "object_instance_ids",
    "initial_state_sha256",
    "source_parent_snapshot_sha256",
    "event_origin_parent_sha256",
    "parent_trajectory_lineage_sha256",
    "base_prompt_sha256",
    "base_checkpoint_sha256",
    "policy_client_config_sha256",
    "perturbation_family",
    "perturbation_seed",
    "branch_seed",
    "master_seed",
    "policy_seed",
    "simulator_seed",
    "trigger_class",
    "deviation_status",
    "deviation_event_id",
    "historical_failure_evidence_json",
    "relevant_fact_sha256",
    "source_graph_version",
    "source_observation_generation",
    "certificate_state",
    "grounding_rule_sha256",
    "event_detector_sha256",
    "monitor_contract_sha256",
    "monitor_contract_json",
    "policy_step",
    "policy_request_generation",
    "base_policy_request_count",
    "active_base_request_index",
    "next_base_request_index",
    "active_base_request_envelope_json",
    "active_base_request_envelope_sha256",
    "next_base_replay_envelope_json",
    "next_base_replay_envelope_sha256",
    "policy_replay_contract_sha256",
    "base_action_response_size",
    "base_action_chunk_size",
    "pending_base_action_offset",
    "pending_base_action_count",
    "pending_base_actions_sha256",
    "live_base_continuation_eligible",
    "live_base_continuation_ineligibility_reason",
    "simulator_state_sha256",
    "observation_sha256",
    "state_fingerprint",
    "base_action_prefix_sha256",
    "fact_epoch_id",
    "fact_universe_version",
    "fact_universe_sha256",
    "fact_evidence_hash",
    "fact_evidence_payload_json",
    "true_facts",
    "false_facts",
    "unknown_facts",
)

if tuple(field.name for field in fields(RecoveryRootManifest)) != RECOVERY_ROOT_MANIFEST_V1_FIELDS:
    raise RuntimeError("RecoveryRootManifest no longer matches schema v1")


@dataclass(frozen=True)
class RecoveryRootArtifacts:
    directory: Path
    manifest_json: Path
    state_npz: Path


def _parse_contract(contract_json: str) -> dict[str, Any]:
    if not isinstance(contract_json, str):
        raise ValueError("monitor contract must be canonical JSON")
    try:
        contract = json.loads(contract_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("monitor contract is not valid JSON") from error
    if _canonical_json(contract) != contract_json:
        raise ValueError("monitor contract JSON is not canonical")
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS:
        raise ValueError("monitor contract schema fields mismatch")
    contract_id = contract["contract_id"]
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("monitor contract ID must be nonempty")
    if type(contract["task_id"]) is not int or contract["task_id"] < 0:
        raise ValueError("monitor contract task ID must be a nonnegative integer")
    for key in (
        "object_ids",
        "nominal_source_facts",
        "abnormal_support_surfaces",
        "task_relevant_effects",
    ):
        values = contract[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"monitor contract {key} must be a unique string list")
    if not contract["object_ids"]:
        raise ValueError("monitor contract object IDs must be nonempty")
    if not isinstance(contract["tracker_version"], str) or not contract["tracker_version"]:
        raise ValueError("monitor contract tracker version must be nonempty")
    signed = dict(contract)
    recorded_hash = signed.pop("contract_sha256")
    _require_sha256("monitor contract hash", recorded_hash)
    if _json_sha256(signed) != recorded_hash:
        raise ValueError("monitor contract self-hash mismatch")
    for key in ("grounding_rule_sha256", "event_detector_sha256"):
        _require_sha256(f"monitor contract {key}", contract[key])
    for key in (
        "monitor_interval_steps",
        "confirmation_count",
        "settling_grace_observations",
        "progress_window_observations",
        "progress_evidence_ttl_policy_steps",
        "goal_regression_evidence_ttl_policy_steps",
        "max_active_attempts_per_object",
        "max_attempt_records_per_episode",
        "max_evidence_records_per_episode",
    ):
        if type(contract[key]) is not int or contract[key] <= 0:
            raise ValueError(f"monitor contract {key} must be positive")
    rules = contract["action_event_rules"]
    if not isinstance(rules, list):
        raise ValueError("monitor contract action rules must be a list")
    identities: set[tuple[Any, ...]] = set()
    rule_ids: set[str] = set()
    contract_object_ids = set(contract["object_ids"])
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != _RULE_FIELDS:
            raise ValueError("monitor contract action rule fields mismatch")
        for key in ("rule_id", "object_id", "attempt_kind", "attempted_effect"):
            if not isinstance(rule[key], str) or not rule[key]:
                raise ValueError(f"monitor contract action rule {key} must be nonempty")
        for key in ("source_region", "destination_region"):
            if rule[key] is not None and (
                not isinstance(rule[key], str) or not rule[key]
            ):
                raise ValueError(f"monitor contract action rule {key} type mismatch")
        rule_id = rule["rule_id"]
        if not isinstance(rule_id, str) or not rule_id or rule_id.startswith("__"):
            raise ValueError("monitor contract action rule ID is invalid")
        if rule_id in rule_ids:
            raise ValueError("duplicate monitor contract action rule ID")
        rule_ids.add(rule_id)
        if rule["object_id"] not in contract_object_ids:
            raise ValueError("monitor contract rule object is outside contract object IDs")
        identity = (
            rule["object_id"],
            rule["attempted_effect"],
            rule["source_region"],
            rule["destination_region"],
        )
        if identity in identities:
            raise ValueError("duplicate monitor contract action rule identity")
        identities.add(identity)
        for key in (
            "contact_min_count",
            "effect_due_after_policy_steps",
            "evidence_ttl_policy_steps",
            "manipulation_attribution_ttl_policy_steps",
        ):
            if type(rule[key]) is not int or rule[key] <= 0:
                raise ValueError(f"monitor contract rule {key} must be positive")
        for key in (
            "gripper_close_threshold",
            "gripper_open_threshold",
            "motion_correlation_min",
            "region_distance_max",
        ):
            if type(rule[key]) is not float or not math.isfinite(rule[key]):
                raise ValueError(f"monitor contract rule {key} must be finite")
        if rule["gripper_close_threshold"] >= rule["gripper_open_threshold"]:
            raise ValueError("monitor contract gripper thresholds are unordered")
    return contract


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if isinstance(record, str):
        try:
            payload = json.loads(record)
        except json.JSONDecodeError as error:
            raise ValueError(
                "historical evidence must be a typed record, not a bare kind/string"
            ) from error
        if _canonical_json(payload) != record or not isinstance(payload, dict):
            raise ValueError("historical evidence record JSON is not canonical")
        return payload
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    raise ValueError("historical evidence must be a typed record, not a bare kind/string")


def _validate_evidence_record(
    record: dict[str, Any],
    contract: Mapping[str, Any],
    detector_sha256: str,
    *,
    root_policy_step: int,
    root_object_ids: frozenset[str],
) -> dict[str, Any]:
    if set(record) != _EVIDENCE_FIELDS:
        raise ValueError("historical evidence fields mismatch")
    for key in (
        "evidence_id",
        "evidence_kind",
        "rule_id",
        "object_id",
        "attempted_effect",
        "attempt_id",
        "detector_sha256",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"historical evidence {key} type mismatch")
    for key in ("source_region", "destination_region"):
        if record[key] is not None and (
            not isinstance(record[key], str) or not record[key]
        ):
            raise ValueError(f"historical evidence {key} type mismatch")
    for key in ("evidence_id", "attempt_id", "detector_sha256"):
        _require_sha256(f"evidence {key}", record[key])
    if record["detector_sha256"] != detector_sha256:
        raise ValueError("historical evidence detector hash mismatch")
    for key in (
        "start_policy_step",
        "effect_due_policy_step",
        "emitted_policy_step",
        "evidence_expires_policy_step",
    ):
        _require_nonnegative(f"evidence {key}", record[key])
    start = record["start_policy_step"]
    due = record["effect_due_policy_step"]
    emitted = record["emitted_policy_step"]
    expires = record["evidence_expires_policy_step"]
    if not start <= due <= emitted <= expires:
        raise ValueError("historical evidence due/emission/expiry ordering is invalid")
    if emitted > root_policy_step:
        raise ValueError("historical evidence is from a future root policy step")
    if not due <= root_policy_step <= expires:
        raise ValueError("historical evidence is not active at the root policy step")
    if record["object_id"] not in root_object_ids:
        raise ValueError("historical evidence object is outside root object instances")
    hashes = record["supporting_transition_hashes"]
    if not isinstance(hashes, (list, tuple)) or not hashes:
        raise ValueError("historical evidence requires supporting transition hashes")
    if len(set(hashes)) != len(hashes):
        raise ValueError("historical evidence has duplicate supporting transition hashes")
    for value in hashes:
        _require_sha256("supporting transition hash", value)
    record["supporting_transition_hashes"] = list(hashes)
    kind = record["evidence_kind"]
    rules = {rule["rule_id"]: rule for rule in contract["action_event_rules"]}
    if kind in _ACTION_EVIDENCE_KINDS:
        rule = rules.get(record["rule_id"])
        if rule is None:
            raise ValueError("historical evidence rule is not in the embedded contract")
        for key in (
            "object_id",
            "attempted_effect",
            "source_region",
            "destination_region",
        ):
            if record[key] != rule[key]:
                raise ValueError("historical evidence rule identity mismatch")
        expected_due = (
            start + rule["effect_due_after_policy_steps"]
            if kind == "ATTEMPTED_EFFECT_TIMEOUT"
            else emitted
        )
        if due != expected_due:
            raise ValueError("historical evidence due step violates its rule")
        if expires != emitted + rule["evidence_ttl_policy_steps"]:
            raise ValueError("historical evidence expiry violates its rule")
        if emitted > start + rule["manipulation_attribution_ttl_policy_steps"]:
            raise ValueError("historical evidence is outside manipulation attribution window")
        expected_attempt_id = _domain_json_sha256(
            b"LOGIV_ACTION_ATTEMPT_ID_V1",
            {
                "object_id": record["object_id"],
                "rule_id": record["rule_id"],
                "start_policy_step": start,
                "start_transition_sha256": hashes[0],
            },
        )
        if record["attempt_id"] != expected_attempt_id:
            raise ValueError("historical evidence attempt ID hash mismatch")
    elif kind in _FACT_EVIDENCE_RULES:
        if record["rule_id"] != _FACT_EVIDENCE_RULES[kind]:
            raise ValueError("fact evidence uses the wrong reserved rule ID")
        if kind == "GOAL_REGRESSION":
            expected_due = emitted
            ttl = contract["goal_regression_evidence_ttl_policy_steps"]
        else:
            expected_due = start + (
                contract["progress_window_observations"]
                * contract["monitor_interval_steps"]
            )
            ttl = contract["progress_evidence_ttl_policy_steps"]
        if due != expected_due or expires != emitted + ttl:
            raise ValueError("fact evidence due/expiry violates the embedded contract")
        if kind == "GOAL_REGRESSION":
            expected_attempt_id = _domain_json_sha256(
                b"LOGIV_GOAL_REGRESSION_ATTEMPT_ID_V1",
                {
                    "achieved_fact_evidence_sha256": hashes[0],
                    "achieved_policy_step": start,
                    "goal_literal": record["attempted_effect"],
                },
            )
            if record["attempt_id"] != expected_attempt_id:
                raise ValueError("Goal-regression attempt ID hash mismatch")
    else:
        raise ValueError("unknown historical evidence kind")
    expected_evidence_id = _domain_json_sha256(
        b"LOGIV_ACTION_EVENT_EVIDENCE_ID_V1",
        {
            "attempt_id": record["attempt_id"],
            "emitted_policy_step": emitted,
            "evidence_kind": kind,
            "supporting_transition_hashes": list(hashes),
        },
    )
    if record["evidence_id"] != expected_evidence_id:
        raise ValueError("historical evidence ID hash mismatch")
    return record


def _evidence_json_records(
    records: tuple[Any, ...] | list[Any],
    contract: Mapping[str, Any],
    detector_sha256: str,
    *,
    root_policy_step: int,
    root_object_ids: frozenset[str],
) -> tuple[str, ...]:
    rendered: list[str] = []
    evidence_ids: set[str] = set()
    attempt_kinds: set[tuple[str, str]] = set()
    for value in records:
        record = _validate_evidence_record(
            _record_mapping(value),
            contract,
            detector_sha256,
            root_policy_step=root_policy_step,
            root_object_ids=root_object_ids,
        )
        if record["evidence_id"] in evidence_ids:
            raise ValueError("duplicate historical evidence ID")
        evidence_ids.add(record["evidence_id"])
        attempt_kind = (record["attempt_id"], record["evidence_kind"])
        if attempt_kind in attempt_kinds:
            raise ValueError("duplicate attempt/evidence kind")
        attempt_kinds.add(attempt_kind)
        rendered.append(_canonical_json(record))
    return tuple(sorted(rendered))


def _validate_contract_root_objects(
    contract: Mapping[str, Any], object_instance_ids: tuple[str, ...]
) -> None:
    root_objects = frozenset(object_instance_ids)
    for rule in contract["action_event_rules"]:
        if rule["object_id"] not in root_objects:
            raise ValueError("monitor contract rule object is outside root object instances")


def _envelope_json_and_hash(
    value: str | None,
    *,
    field: str,
    expected_seed: int,
    expected_index: int | None,
    expected_config_sha256: str,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} envelope is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise ValueError(f"{field} envelope JSON is not canonical")
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "episode_seed",
        "inference_index",
        "policy_client_config_sha256",
    }:
        raise ValueError(f"{field} envelope fields mismatch")
    if (
        not isinstance(payload["version"], str)
        or type(payload["episode_seed"]) is not int
        or type(payload["inference_index"]) is not int
        or not isinstance(payload["policy_client_config_sha256"], str)
    ):
        raise ValueError(f"{field} envelope scalar type mismatch")
    if payload["version"] != "BaseRequestEnvelopeV1":
        raise ValueError(f"{field} envelope version mismatch")
    if expected_index is None or payload["inference_index"] != expected_index:
        raise ValueError(f"{field} envelope request index mismatch")
    if payload["episode_seed"] != expected_seed:
        raise ValueError(f"{field} envelope seed mismatch")
    if payload["policy_client_config_sha256"] != expected_config_sha256:
        raise ValueError(f"{field} envelope policy config mismatch")
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive_live_eligibility(
    active_json: str | None,
    next_json: str | None,
    replay_contract_sha256: str | None,
) -> tuple[bool, str | None]:
    if active_json is None:
        return False, "missing active Base request envelope"
    if next_json is None:
        return False, "missing next Base replay envelope"
    if replay_contract_sha256 is None:
        return False, "missing policy replay contract"
    return True, None


def _id_payloads(
    *,
    task_id: int,
    scene_sha256: str,
    object_instance_ids: tuple[str, ...],
    initial_state_sha256: str,
    event_origin_parent_sha256: str,
    perturbation_family: str,
    parent_trajectory_lineage_sha256: str,
    perturbation_seed: int | None,
    branch_seed: int | None,
    policy_step: int,
    deviation_status: str,
    deviation_event_id: str,
    simulator_state_sha256: str,
) -> tuple[str, str, str]:
    group_payload = {
        "task_id": task_id,
        "scene_sha256": scene_sha256,
        "object_instance_ids": sorted(object_instance_ids),
        "initial_state_sha256": initial_state_sha256,
        "event_origin_parent_sha256": event_origin_parent_sha256,
        "perturbation_family": perturbation_family,
    }
    independence_payload = {
        "task_id": task_id,
        "scene_sha256": scene_sha256,
        "initial_state_sha256": initial_state_sha256,
        "parent_trajectory_lineage_sha256": parent_trajectory_lineage_sha256,
    }
    root_payload = {
        **group_payload,
        "perturbation_seed": perturbation_seed,
        "branch_seed": branch_seed,
        "policy_step": policy_step,
        "deviation_status": deviation_status,
        "deviation_event_id": deviation_event_id,
        "simulator_state_sha256": simulator_state_sha256,
    }
    return (
        _json_sha256(root_payload),
        _json_sha256(group_payload),
        _json_sha256(independence_payload),
    )


def make_recovery_root_manifest(
    *,
    split: RecoverySplit,
    collection_label: CollectionLabel,
    task_id: int,
    episode_idx: int,
    scene_sha256: str,
    object_instance_ids: tuple[str, ...],
    initial_state_sha256: str,
    source_parent_snapshot_sha256: str | None,
    event_origin_parent_sha256: str,
    parent_trajectory_lineage_sha256: str,
    base_prompt_sha256: str,
    base_checkpoint_sha256: str,
    policy_client_config_sha256: str,
    perturbation_family: str,
    perturbation_seed: int | None,
    branch_seed: int | None,
    master_seed: int,
    policy_seed: int,
    simulator_seed: int,
    trigger_class: str,
    deviation_status: str,
    deviation_event_id: str,
    historical_failure_evidence: tuple[Any, ...],
    relevant_fact_sha256: str,
    source_graph_version: str,
    source_observation_generation: int,
    certificate_state: str,
    grounding_rule_sha256: str,
    event_detector_sha256: str,
    monitor_contract_json: str,
    policy_step: int,
    policy_request_generation: int,
    base_policy_request_count: int,
    active_base_request_index: int | None,
    next_base_request_index: int,
    active_base_request_envelope_json: str | None,
    next_base_replay_envelope_json: str | None,
    policy_replay_contract_sha256: str | None,
    base_action_response_size: int | None,
    base_action_chunk_size: int,
    pending_base_action_offset: int,
    simulator_state: np.ndarray,
    observation: Mapping[str, np.ndarray],
    pending_base_actions: np.ndarray,
    base_action_prefix_sha256: str,
    snapshot: FactSnapshot,
) -> RecoveryRootManifest:
    if type(split) is not RecoverySplit or type(collection_label) is not CollectionLabel:
        raise ValueError("recovery split and collection label types are exact enums")
    if type(object_instance_ids) is not tuple or any(
        type(value) is not str for value in object_instance_ids
    ):
        raise ValueError("object instance IDs must be a tuple of strings")
    if type(historical_failure_evidence) is not tuple:
        raise ValueError("historical failure evidence must be a tuple")
    if type(snapshot) is not FactSnapshot:
        raise ValueError("recovery snapshot must be an exact FactSnapshot")
    contract = _parse_contract(monitor_contract_json)
    simulator_state = _validate_array("simulator_state", simulator_state)
    pending_base_actions = _validate_array(
        "pending_base_actions", pending_base_actions
    )
    if pending_base_actions.ndim == 0:
        raise ValueError("pending Base actions must have an action dimension")
    observation_values = _validated_observation(observation)
    simulator_state_hash = _array_sha256(
        b"LOGIV_SIMULATOR_STATE_V1", simulator_state
    )
    pending_hash = _array_sha256(
        b"LOGIV_PENDING_BASE_ACTIONS_V1", pending_base_actions
    )
    rounded_state = np.round(simulator_state, decimals=4)
    state_fingerprint = _array_sha256(
        b"LOGIV_STATE_FINGERPRINT_V1", rounded_state
    )
    observation_hash = observation_sha256(observation_values)
    if any(
        value is None
        for value in (
            snapshot.fact_universe,
            snapshot.fact_universe_version,
            snapshot.fact_universe_sha256,
            snapshot.evidence_payload_json,
        )
    ):
        raise ValueError("recovery roots require a complete audited fact universe and evidence")
    evidence_payload = json.loads(snapshot.evidence_payload_json)
    if evidence_payload["observation_hash"] != observation_hash:
        raise ValueError("fact evidence observation hash does not match recovery observation")
    if contract["task_id"] != task_id:
        raise ValueError("monitor contract task mismatch")
    if contract["grounding_rule_sha256"] != grounding_rule_sha256:
        raise ValueError("monitor contract grounding-rule hash mismatch")
    if contract["event_detector_sha256"] != event_detector_sha256:
        raise ValueError("monitor contract event-detector hash mismatch")
    object_ids = tuple(sorted(object_instance_ids))
    _validate_contract_root_objects(contract, object_ids)
    evidence_json = _evidence_json_records(
        historical_failure_evidence,
        contract,
        event_detector_sha256,
        root_policy_step=policy_step,
        root_object_ids=frozenset(object_ids),
    )
    active_json, active_hash = _envelope_json_and_hash(
        active_base_request_envelope_json,
        field="active Base request",
        expected_seed=policy_seed,
        expected_index=active_base_request_index,
        expected_config_sha256=policy_client_config_sha256,
    )
    next_json, next_hash = _envelope_json_and_hash(
        next_base_replay_envelope_json,
        field="next Base replay",
        expected_seed=policy_seed,
        expected_index=next_base_request_index,
        expected_config_sha256=policy_client_config_sha256,
    )
    eligible, ineligibility_reason = _derive_live_eligibility(
        active_json, next_json, policy_replay_contract_sha256
    )
    unknown = snapshot.fact_universe - snapshot.true_facts - snapshot.false_facts
    root_id, group_id, independence_id = _id_payloads(
        task_id=task_id,
        scene_sha256=scene_sha256,
        object_instance_ids=object_ids,
        initial_state_sha256=initial_state_sha256,
        event_origin_parent_sha256=event_origin_parent_sha256,
        perturbation_family=perturbation_family,
        parent_trajectory_lineage_sha256=parent_trajectory_lineage_sha256,
        perturbation_seed=perturbation_seed,
        branch_seed=branch_seed,
        policy_step=policy_step,
        deviation_status=deviation_status,
        deviation_event_id=deviation_event_id,
        simulator_state_sha256=simulator_state_hash,
    )
    manifest = RecoveryRootManifest(
        schema_version=1,
        root_id=root_id,
        recovery_group_id=group_id,
        independence_unit_id=independence_id,
        split=split,
        collection_label=collection_label,
        task_id=task_id,
        episode_idx=episode_idx,
        scene_sha256=scene_sha256,
        object_instance_ids=object_ids,
        initial_state_sha256=initial_state_sha256,
        source_parent_snapshot_sha256=source_parent_snapshot_sha256,
        event_origin_parent_sha256=event_origin_parent_sha256,
        parent_trajectory_lineage_sha256=parent_trajectory_lineage_sha256,
        base_prompt_sha256=base_prompt_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        policy_client_config_sha256=policy_client_config_sha256,
        perturbation_family=perturbation_family,
        perturbation_seed=perturbation_seed,
        branch_seed=branch_seed,
        master_seed=master_seed,
        policy_seed=policy_seed,
        simulator_seed=simulator_seed,
        trigger_class=trigger_class,
        deviation_status=deviation_status,
        deviation_event_id=deviation_event_id,
        historical_failure_evidence_json=evidence_json,
        relevant_fact_sha256=relevant_fact_sha256,
        source_graph_version=source_graph_version,
        source_observation_generation=source_observation_generation,
        certificate_state=certificate_state,
        grounding_rule_sha256=grounding_rule_sha256,
        event_detector_sha256=event_detector_sha256,
        monitor_contract_sha256=contract["contract_sha256"],
        monitor_contract_json=monitor_contract_json,
        policy_step=policy_step,
        policy_request_generation=policy_request_generation,
        base_policy_request_count=base_policy_request_count,
        active_base_request_index=active_base_request_index,
        next_base_request_index=next_base_request_index,
        active_base_request_envelope_json=active_json,
        active_base_request_envelope_sha256=active_hash,
        next_base_replay_envelope_json=next_json,
        next_base_replay_envelope_sha256=next_hash,
        policy_replay_contract_sha256=policy_replay_contract_sha256,
        base_action_response_size=base_action_response_size,
        base_action_chunk_size=base_action_chunk_size,
        pending_base_action_offset=pending_base_action_offset,
        pending_base_action_count=int(pending_base_actions.shape[0]),
        pending_base_actions_sha256=pending_hash,
        live_base_continuation_eligible=eligible,
        live_base_continuation_ineligibility_reason=ineligibility_reason,
        simulator_state_sha256=simulator_state_hash,
        observation_sha256=observation_hash,
        state_fingerprint=state_fingerprint,
        base_action_prefix_sha256=base_action_prefix_sha256,
        fact_epoch_id=snapshot.epoch_id,
        fact_universe_version=snapshot.fact_universe_version,
        fact_universe_sha256=snapshot.fact_universe_sha256,
        fact_evidence_hash=snapshot.evidence_hash,
        fact_evidence_payload_json=snapshot.evidence_payload_json,
        true_facts=tuple(
            fact.pddl()
            for fact in sorted(snapshot.true_facts, key=fact_pddl_sort_key)
        ),
        false_facts=tuple(
            fact.pddl()
            for fact in sorted(snapshot.false_facts, key=fact_pddl_sort_key)
        ),
        unknown_facts=tuple(
            fact.pddl() for fact in sorted(unknown, key=fact_pddl_sort_key)
        ),
    )
    _validate_manifest_state(
        manifest, simulator_state, observation_values, pending_base_actions
    )
    return manifest


def _validate_manifest_state(
    manifest: RecoveryRootManifest,
    simulator_state: np.ndarray,
    observation: Mapping[str, np.ndarray],
    pending_base_actions: np.ndarray,
) -> None:
    _validate_manifest_scalar_types(manifest)
    if manifest.schema_version != 1:
        raise ValueError("unknown recovery-root schema")
    if not isinstance(manifest.split, RecoverySplit) or not isinstance(
        manifest.collection_label, CollectionLabel
    ):
        raise ValueError("recovery split or collection label is invalid")
    for name in (
        "task_id",
        "episode_idx",
        "master_seed",
        "policy_seed",
        "simulator_seed",
        "source_observation_generation",
        "policy_step",
        "policy_request_generation",
        "base_policy_request_count",
        "next_base_request_index",
        "base_action_chunk_size",
        "pending_base_action_offset",
        "pending_base_action_count",
        "fact_epoch_id",
    ):
        _require_nonnegative(name, getattr(manifest, name))
    for name in (
        "perturbation_seed",
        "branch_seed",
        "active_base_request_index",
        "base_action_response_size",
    ):
        _require_nonnegative(name, getattr(manifest, name), optional=True)
    for name in (
        "root_id",
        "recovery_group_id",
        "independence_unit_id",
        "scene_sha256",
        "initial_state_sha256",
        "event_origin_parent_sha256",
        "parent_trajectory_lineage_sha256",
        "base_prompt_sha256",
        "base_checkpoint_sha256",
        "policy_client_config_sha256",
        "relevant_fact_sha256",
        "grounding_rule_sha256",
        "event_detector_sha256",
        "monitor_contract_sha256",
        "pending_base_actions_sha256",
        "simulator_state_sha256",
        "observation_sha256",
        "state_fingerprint",
        "base_action_prefix_sha256",
        "fact_universe_sha256",
        "fact_evidence_hash",
    ):
        _require_sha256(name, getattr(manifest, name))
    for name in (
        "source_parent_snapshot_sha256",
        "active_base_request_envelope_sha256",
        "next_base_replay_envelope_sha256",
        "policy_replay_contract_sha256",
    ):
        _require_sha256(name, getattr(manifest, name), optional=True)
    if manifest.perturbation_seed is not None and manifest.source_parent_snapshot_sha256 is None:
        raise ValueError("synthetic recovery roots require a source parent snapshot")
    if not manifest.object_instance_ids or manifest.object_instance_ids != tuple(
        sorted(set(manifest.object_instance_ids))
    ):
        raise ValueError("object instance IDs must be nonempty, unique, and sorted")
    if any(not isinstance(value, str) or not value for value in manifest.object_instance_ids):
        raise ValueError("object instance IDs must be nonempty strings")
    for name in (
        "perturbation_family",
        "trigger_class",
        "deviation_event_id",
        "source_graph_version",
        "fact_universe_version",
    ):
        if not isinstance(getattr(manifest, name), str) or not getattr(manifest, name):
            raise ValueError(f"{name} must be nonempty")
    if manifest.deviation_status not in _DEVIATION_STATUSES:
        raise ValueError("deviation status is invalid")
    if manifest.certificate_state not in _CERTIFICATE_STATES:
        raise ValueError("certificate state is invalid")
    if manifest.next_base_request_index != manifest.base_policy_request_count:
        raise ValueError("next Base request index/count mismatch")
    if manifest.policy_request_generation != 0:
        raise ValueError("Phase 0 policy request generation must remain zero")
    expected_active = (
        None if manifest.base_policy_request_count == 0 else manifest.base_policy_request_count - 1
    )
    if manifest.active_base_request_index != expected_active:
        raise ValueError("active Base request index/count mismatch")
    contract = _parse_contract(manifest.monitor_contract_json)
    if manifest.monitor_contract_sha256 != contract["contract_sha256"]:
        raise ValueError("monitor contract hash mismatch")
    if contract["task_id"] != manifest.task_id:
        raise ValueError("monitor contract task mismatch")
    if contract["grounding_rule_sha256"] != manifest.grounding_rule_sha256:
        raise ValueError("monitor contract grounding-rule hash mismatch")
    if contract["event_detector_sha256"] != manifest.event_detector_sha256:
        raise ValueError("monitor contract event-detector hash mismatch")
    _validate_contract_root_objects(contract, manifest.object_instance_ids)
    canonical_evidence = _evidence_json_records(
        list(manifest.historical_failure_evidence_json),
        contract,
        manifest.event_detector_sha256,
        root_policy_step=manifest.policy_step,
        root_object_ids=frozenset(manifest.object_instance_ids),
    )
    if canonical_evidence != manifest.historical_failure_evidence_json:
        raise ValueError("historical evidence order or canonical JSON mismatch")
    active_json, active_hash = _envelope_json_and_hash(
        manifest.active_base_request_envelope_json,
        field="active Base request",
        expected_seed=manifest.policy_seed,
        expected_index=manifest.active_base_request_index,
        expected_config_sha256=manifest.policy_client_config_sha256,
    )
    next_json, next_hash = _envelope_json_and_hash(
        manifest.next_base_replay_envelope_json,
        field="next Base replay",
        expected_seed=manifest.policy_seed,
        expected_index=manifest.next_base_request_index,
        expected_config_sha256=manifest.policy_client_config_sha256,
    )
    if (
        active_json != manifest.active_base_request_envelope_json
        or active_hash != manifest.active_base_request_envelope_sha256
        or next_json != manifest.next_base_replay_envelope_json
        or next_hash != manifest.next_base_replay_envelope_sha256
    ):
        raise ValueError("Base replay envelope hash mismatch")
    expected_eligibility = _derive_live_eligibility(
        active_json, next_json, manifest.policy_replay_contract_sha256
    )
    if expected_eligibility != (
        manifest.live_base_continuation_eligible,
        manifest.live_base_continuation_ineligibility_reason,
    ):
        raise ValueError("live Base continuation eligibility provenance mismatch")
    simulator_state = _validate_array("simulator_state", simulator_state)
    pending_base_actions = _validate_array(
        "pending_base_actions", pending_base_actions
    )
    observation_values = _validated_observation(observation)
    if simulator_state.ndim != 1:
        raise ValueError("simulator state must be flattened")
    if pending_base_actions.ndim != 2 or pending_base_actions.shape[1] != 7:
        raise ValueError("pending Base actions must have shape (count, 7)")
    if manifest.base_policy_request_count == 0:
        if (
            manifest.base_action_response_size is not None
            or manifest.base_action_chunk_size != 0
            or manifest.pending_base_action_offset != 0
            or manifest.pending_base_action_count != 0
        ):
            raise ValueError("step-0 Base request/response/chunk provenance mismatch")
    elif (
        manifest.base_action_response_size is None
        or manifest.base_action_chunk_size <= 0
        or manifest.base_action_chunk_size > manifest.base_action_response_size
    ):
        raise ValueError("Base response size must cover the logical action chunk")
    if pending_base_actions.shape[0] != manifest.pending_base_action_count:
        raise ValueError("pending Base action count mismatch")
    if (
        manifest.pending_base_action_offset + manifest.pending_base_action_count
        != manifest.base_action_chunk_size
    ):
        raise ValueError("pending Base action offset/count/chunk mismatch")
    if _array_sha256(b"LOGIV_SIMULATOR_STATE_V1", simulator_state) != manifest.simulator_state_sha256:
        raise ValueError("simulator state hash mismatch")
    if _array_sha256(b"LOGIV_PENDING_BASE_ACTIONS_V1", pending_base_actions) != manifest.pending_base_actions_sha256:
        raise ValueError("pending Base actions hash mismatch")
    if observation_sha256(observation_values) != manifest.observation_sha256:
        raise ValueError("observation hash mismatch")
    if _array_sha256(b"LOGIV_STATE_FINGERPRINT_V1", np.round(simulator_state, 4)) != manifest.state_fingerprint:
        raise ValueError("state fingerprint mismatch")
    partitions = (
        manifest.true_facts,
        manifest.false_facts,
        manifest.unknown_facts,
    )
    if any(len(values) != len(set(values)) for values in partitions):
        raise ValueError("duplicate fact in saved partition")
    true_facts = frozenset(parse_pddl_fact(value) for value in manifest.true_facts)
    false_facts = frozenset(parse_pddl_fact(value) for value in manifest.false_facts)
    unknown_facts = frozenset(parse_pddl_fact(value) for value in manifest.unknown_facts)
    if true_facts & false_facts or true_facts & unknown_facts or false_facts & unknown_facts:
        raise ValueError("saved fact partition has conflicting members")
    universe = true_facts | false_facts | unknown_facts
    if fact_universe_sha256(manifest.fact_universe_version, universe) != manifest.fact_universe_sha256:
        raise ValueError("fact universe hash mismatch")
    snapshot = FactSnapshot(
        epoch_id=manifest.fact_epoch_id,
        true_facts=true_facts,
        false_facts=false_facts,
        evidence_hash=manifest.fact_evidence_hash,
        fact_universe=universe,
        fact_universe_version=manifest.fact_universe_version,
        fact_universe_sha256=manifest.fact_universe_sha256,
        evidence_payload_json=manifest.fact_evidence_payload_json,
    )
    if snapshot.unknown(universe) != unknown_facts:
        raise ValueError("saved UNKNOWN partition does not complete the fact universe")
    payload = json.loads(manifest.fact_evidence_payload_json)
    if payload["observation_hash"] != manifest.observation_sha256:
        raise ValueError("fact evidence observation hash mismatch")
    expected_ids = _id_payloads(
        task_id=manifest.task_id,
        scene_sha256=manifest.scene_sha256,
        object_instance_ids=manifest.object_instance_ids,
        initial_state_sha256=manifest.initial_state_sha256,
        event_origin_parent_sha256=manifest.event_origin_parent_sha256,
        perturbation_family=manifest.perturbation_family,
        parent_trajectory_lineage_sha256=manifest.parent_trajectory_lineage_sha256,
        perturbation_seed=manifest.perturbation_seed,
        branch_seed=manifest.branch_seed,
        policy_step=manifest.policy_step,
        deviation_status=manifest.deviation_status,
        deviation_event_id=manifest.deviation_event_id,
        simulator_state_sha256=manifest.simulator_state_sha256,
    )
    if expected_ids != (
        manifest.root_id,
        manifest.recovery_group_id,
        manifest.independence_unit_id,
    ):
        raise ValueError("recovery root/group/independence ID mismatch")


def _manifest_payload(manifest: RecoveryRootManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["split"] = manifest.split.value
    payload["collection_label"] = manifest.collection_label.value
    return payload


_MANIFEST_INTEGER_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "episode_idx",
        "master_seed",
        "policy_seed",
        "simulator_seed",
        "source_observation_generation",
        "policy_step",
        "policy_request_generation",
        "base_policy_request_count",
        "next_base_request_index",
        "base_action_chunk_size",
        "pending_base_action_offset",
        "pending_base_action_count",
        "fact_epoch_id",
    }
)
_MANIFEST_OPTIONAL_INTEGER_FIELDS = frozenset(
    {
        "perturbation_seed",
        "branch_seed",
        "active_base_request_index",
        "base_action_response_size",
    }
)
_MANIFEST_STRING_FIELDS = frozenset(
    {
        "root_id",
        "recovery_group_id",
        "independence_unit_id",
        "scene_sha256",
        "initial_state_sha256",
        "event_origin_parent_sha256",
        "parent_trajectory_lineage_sha256",
        "base_prompt_sha256",
        "base_checkpoint_sha256",
        "policy_client_config_sha256",
        "perturbation_family",
        "trigger_class",
        "deviation_status",
        "deviation_event_id",
        "relevant_fact_sha256",
        "source_graph_version",
        "certificate_state",
        "grounding_rule_sha256",
        "event_detector_sha256",
        "monitor_contract_sha256",
        "monitor_contract_json",
        "pending_base_actions_sha256",
        "simulator_state_sha256",
        "observation_sha256",
        "state_fingerprint",
        "base_action_prefix_sha256",
        "fact_universe_version",
        "fact_universe_sha256",
        "fact_evidence_hash",
        "fact_evidence_payload_json",
    }
)
_MANIFEST_OPTIONAL_STRING_FIELDS = frozenset(
    {
        "source_parent_snapshot_sha256",
        "active_base_request_envelope_json",
        "active_base_request_envelope_sha256",
        "next_base_replay_envelope_json",
        "next_base_replay_envelope_sha256",
        "policy_replay_contract_sha256",
        "live_base_continuation_ineligibility_reason",
    }
)
_MANIFEST_STRING_TUPLE_FIELDS = frozenset(
    {
        "object_instance_ids",
        "historical_failure_evidence_json",
        "true_facts",
        "false_facts",
        "unknown_facts",
    }
)


def _validate_manifest_scalar_types(manifest: RecoveryRootManifest) -> None:
    for name in _MANIFEST_INTEGER_FIELDS:
        if type(getattr(manifest, name)) is not int:
            raise ValueError(f"recovery-root {name} scalar type mismatch")
    for name in _MANIFEST_OPTIONAL_INTEGER_FIELDS:
        value = getattr(manifest, name)
        if value is not None and type(value) is not int:
            raise ValueError(f"recovery-root {name} scalar type mismatch")
    for name in _MANIFEST_STRING_FIELDS:
        if type(getattr(manifest, name)) is not str:
            raise ValueError(f"recovery-root {name} scalar type mismatch")
    for name in _MANIFEST_OPTIONAL_STRING_FIELDS:
        value = getattr(manifest, name)
        if value is not None and type(value) is not str:
            raise ValueError(f"recovery-root {name} scalar type mismatch")
    for name in _MANIFEST_STRING_TUPLE_FIELDS:
        values = getattr(manifest, name)
        if type(values) is not tuple or any(type(value) is not str for value in values):
            raise ValueError(f"recovery-root {name} container type mismatch")
    if type(manifest.live_base_continuation_eligible) is not bool:
        raise ValueError("recovery-root eligibility must be boolean")


def _validate_manifest_json_types(payload: Mapping[str, Any]) -> None:
    for name in _MANIFEST_INTEGER_FIELDS:
        if type(payload[name]) is not int:
            raise ValueError(f"recovery-root {name} JSON type mismatch")
    for name in _MANIFEST_OPTIONAL_INTEGER_FIELDS:
        if payload[name] is not None and type(payload[name]) is not int:
            raise ValueError(f"recovery-root {name} JSON type mismatch")
    for name in _MANIFEST_STRING_FIELDS | {"split", "collection_label"}:
        if type(payload[name]) is not str:
            raise ValueError(f"recovery-root {name} JSON type mismatch")
    for name in _MANIFEST_OPTIONAL_STRING_FIELDS:
        if payload[name] is not None and type(payload[name]) is not str:
            raise ValueError(f"recovery-root {name} JSON type mismatch")
    for name in _MANIFEST_STRING_TUPLE_FIELDS:
        values = payload[name]
        if type(values) is not list or any(type(value) is not str for value in values):
            raise ValueError(f"recovery-root {name} JSON container type mismatch")
    if type(payload["live_base_continuation_eligible"]) is not bool:
        raise ValueError("recovery-root eligibility JSON type must be boolean")


def _manifest_from_payload(payload: Any) -> RecoveryRootManifest:
    expected = set(RECOVERY_ROOT_MANIFEST_V1_FIELDS)
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("recovery-root manifest schema fields mismatch")
    _validate_manifest_json_types(payload)
    values = dict(payload)
    values["split"] = RecoverySplit(values["split"])
    values["collection_label"] = CollectionLabel(values["collection_label"])
    for name in (
        "object_instance_ids",
        "historical_failure_evidence_json",
        "true_facts",
        "false_facts",
        "unknown_facts",
    ):
        values[name] = tuple(values[name])
    return RecoveryRootManifest(**values)


@contextmanager
def _registry_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / ".recovery-roots.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(source: Path, destination: Path) -> None:
    os.rename(source, destination)


def _same_arrays(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> bool:
    if set(first) != set(second):
        return False
    return all(
        first[key].dtype == second[key].dtype
        and first[key].shape == second[key].shape
        and np.ascontiguousarray(first[key]).tobytes(order="C")
        == np.ascontiguousarray(second[key]).tobytes(order="C")
        for key in first
    )


def write_recovery_root(
    output_dir: Path | str,
    manifest: RecoveryRootManifest,
    simulator_state: np.ndarray,
    observation: Mapping[str, np.ndarray],
    pending_base_actions: np.ndarray,
) -> RecoveryRootArtifacts:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    simulator_state = _validate_array("simulator_state", simulator_state)
    observation_values = _validated_observation(observation)
    pending_base_actions = _validate_array(
        "pending_base_actions", pending_base_actions
    )
    _validate_manifest_state(
        manifest, simulator_state, observation_values, pending_base_actions
    )
    final = output / manifest.root_id
    artifacts = RecoveryRootArtifacts(
        directory=final,
        manifest_json=final / "recovery_root.json",
        state_npz=final / "state.npz",
    )
    with _registry_lock(output):
        if final.exists():
            try:
                existing_manifest, existing_state = load_recovery_root(final)
            except (OSError, ValueError) as error:
                raise ValueError("existing recovery root is corrupt") from error
            requested_state = {
                "simulator_state": simulator_state,
                "pending_base_actions": pending_base_actions,
                **observation_values,
            }
            if existing_manifest != manifest or not _same_arrays(
                existing_state, requested_state
            ):
                raise ValueError("existing recovery root is non-identical corruption")
            return artifacts
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{manifest.root_id}.tmp.", dir=output)
        )
        state_path = temporary / "state.npz"
        manifest_path = temporary / "recovery_root.json"
        np.savez(
            state_path,
            simulator_state=simulator_state,
            pending_base_actions=pending_base_actions,
            **observation_values,
        )
        _fsync_file(state_path)
        manifest_path.write_bytes(
            _canonical_json(_manifest_payload(manifest)).encode("utf-8")
        )
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        _publish_directory(temporary, final)
        _fsync_directory(output)
    return artifacts


def load_recovery_root(
    directory: Path | str,
) -> tuple[RecoveryRootManifest, Mapping[str, np.ndarray]]:
    root = Path(directory)
    if ".tmp." in root.name or not root.is_dir():
        raise ValueError("only a final recovery-root directory can be loaded")
    manifest_path = root / "recovery_root.json"
    state_path = root / "state.npz"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError("recovery root is missing manifest or state")
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(manifest_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("recovery-root manifest is unreadable") from error
    if _canonical_json(payload) != manifest_text:
        raise ValueError("recovery-root manifest JSON is not canonical")
    manifest = _manifest_from_payload(payload)
    if root.name != manifest.root_id:
        raise ValueError("recovery-root directory/root ID mismatch")
    try:
        with np.load(state_path, allow_pickle=False) as archive:
            state = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("recovery-root NPZ is unreadable or uses pickle") from error
    if not _STATE_KEYS <= set(state):
        raise ValueError("recovery-root NPZ is missing required state arrays")
    observation = {key: value for key, value in state.items() if key not in _STATE_KEYS}
    _validate_manifest_state(
        manifest,
        state["simulator_state"],
        observation,
        state["pending_base_actions"],
    )
    return manifest, state


def find_orphan_root_temps(output_dir: Path | str) -> tuple[Path, ...]:
    output = Path(output_dir)
    if not output.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in output.iterdir()
            if path.is_dir() and path.name.startswith(".") and ".tmp." in path.name
        )
    )
