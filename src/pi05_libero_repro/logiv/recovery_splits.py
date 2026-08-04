from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from pi05_libero_repro.logiv import recovery_records
from pi05_libero_repro.logiv.recovery_records import (
    CollectionLabel,
    RecoveryRootManifest,
    RecoverySplit,
    load_recovery_root,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_ROLE_REGISTRY_ID = "logiv-r2m-recovery-roles-v1"
CANONICAL_ROLE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "logiv"
    / "recovery-role-registry.json"
)
_STRONG_EVIDENCE_KINDS = frozenset(
    {
        "GOAL_REGRESSION",
        "ATTEMPTED_EFFECT_TIMEOUT",
        "ABNORMAL_TRANSFER_AFTER_MANIPULATION",
    }
)


class RecoverySplitError(ValueError):
    pass


class DatasetRole(str, Enum):
    TRAIN = "TRAIN"
    DEV_SELECTION = "DEV_SELECTION"
    PERMIT_CALIBRATION = "PERMIT_CALIBRATION"
    PAPER_CONFIRMATION = "PAPER_CONFIRMATION"


@dataclass(frozen=True)
class RoleAllocationEntry:
    source_dataset_sha256: str
    independence_unit_id: str
    role_isolation_keys: tuple[str, ...]
    role: DatasetRole


@dataclass(frozen=True)
class RoleAllocationManifest:
    schema_version: int
    source_dataset_sha256: str
    source_independence_unit_ids: tuple[str, ...]
    entries: tuple[RoleAllocationEntry, ...]
    allocation_version: str
    registry_parent_head_sha256: str
    allocation_sha256: str


@dataclass(frozen=True)
class RoleRegistryHead:
    schema_version: int
    registry_id: str
    revision: int
    parent_head_sha256: str | None
    allocations: tuple[RoleAllocationEntry, ...]
    head_sha256: str


@dataclass(frozen=True)
class FreshRecoveryLabel:
    schema_version: int
    root_id: str
    simulator_state_sha256: str
    fresh_observation_sha256: str
    deviation_event_id: str
    historical_evidence_sha256: str
    relevant_fact_sha256: str
    fresh_fact_epoch_id: int
    fact_universe_sha256: str
    fresh_fact_evidence_hash: str
    graph_version: str
    certificate_sha256: str
    grounding_rule_sha256: str
    event_detector_sha256: str
    monitor_contract_sha256: str
    labeler_version: str
    label_sha256: str


@dataclass(frozen=True)
class TrainingManifestEntry:
    root_id: str
    recovery_group_id: str
    independence_unit_id: str
    fresh_label_sha256: str | None


@dataclass(frozen=True)
class TrainingManifest:
    schema_version: int
    role: DatasetRole
    entries: tuple[TrainingManifestEntry, ...]
    source_dataset_sha256: str
    role_allocation_sha256: str
    role_registry_head_sha256: str
    builder_version: str
    manifest_sha256: str


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


def _content_hash(value: Any, hash_field: str) -> str:
    payload = asdict(value)
    payload.pop(hash_field)
    return _json_sha256(payload)


def _require_sha256(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RecoverySplitError(f"{name} must be a lowercase SHA-256")


def _require_text(name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise RecoverySplitError(f"{name} must be nonempty")


def _manifest_payload(manifest: RecoveryRootManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["split"] = manifest.split.value
    payload["collection_label"] = manifest.collection_label.value
    return payload


def recovery_source_dataset_sha256(
    manifests: Sequence[RecoveryRootManifest],
) -> str:
    ordered = sorted((_manifest_payload(item) for item in manifests), key=lambda x: x["root_id"])
    return _domain_json_sha256(b"LOGIV_RECOVERY_SOURCE_DATASET_V1", ordered)


def historical_evidence_sha256(manifest: RecoveryRootManifest) -> str:
    return _json_sha256(list(manifest.historical_failure_evidence_json))


def fresh_fact_evidence_sha256(
    *,
    fresh_observation_sha256: str,
    fresh_fact_epoch_id: int,
    fact_universe_sha256: str,
    relevant_fact_sha256: str,
) -> str:
    return _domain_json_sha256(
        b"LOGIV_FRESH_FACT_EVIDENCE_V1",
        {
            "fact_epoch_id": fresh_fact_epoch_id,
            "fact_universe_sha256": fact_universe_sha256,
            "fresh_observation_sha256": fresh_observation_sha256,
            "relevant_fact_sha256": relevant_fact_sha256,
        },
    )


def _evidence_records(manifest: RecoveryRootManifest) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for encoded in manifest.historical_failure_evidence_json:
        try:
            record = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as error:
            raise RecoverySplitError("historical evidence is not valid JSON") from error
        if not isinstance(record, dict) or _canonical_json(record) != encoded:
            raise RecoverySplitError("historical evidence JSON is not canonical")
        records.append(record)
    if manifest.deviation_status == "CONFIRMED_DEVIATION" and not any(
        record.get("evidence_kind") in _STRONG_EVIDENCE_KINDS for record in records
    ):
        raise RecoverySplitError(
            "CONFIRMED_DEVIATION requires active exact strong evidence; "
            "PROGRESS_TIMEOUT alone is invalid"
        )
    try:
        contract = recovery_records._parse_contract(manifest.monitor_contract_json)
        canonical = recovery_records._evidence_json_records(
            list(manifest.historical_failure_evidence_json),
            contract,
            manifest.event_detector_sha256,
            root_policy_step=manifest.policy_step,
            root_object_ids=frozenset(manifest.object_instance_ids),
        )
    except (TypeError, ValueError) as error:
        raise RecoverySplitError(f"historical evidence is invalid: {error}") from error
    if canonical != manifest.historical_failure_evidence_json:
        raise RecoverySplitError("historical evidence order is not canonical")
    return tuple(records)


_RELATION_CHECKS = (
    ("recovery_group_id", lambda item: item.recovery_group_id),
    ("independence_unit_id", lambda item: item.independence_unit_id),
    ("initial_state_sha256", lambda item: item.initial_state_sha256),
    ("source_parent_snapshot_sha256", lambda item: item.source_parent_snapshot_sha256),
    ("parent_trajectory_lineage_sha256", lambda item: item.parent_trajectory_lineage_sha256),
    ("state_fingerprint", lambda item: item.state_fingerprint),
    (
        "perturbation_seed",
        lambda item: (
            (item.task_id, item.perturbation_seed)
            if item.perturbation_seed is not None
            else None
        ),
    ),
)


def _reject_relations_across(
    manifests: Sequence[RecoveryRootManifest],
    labels: Mapping[str, str],
) -> None:
    for name, relation in _RELATION_CHECKS:
        seen: dict[Any, set[str]] = {}
        for item in manifests:
            key = relation(item)
            if key is None:
                continue
            seen.setdefault(key, set()).add(labels[item.root_id])
        if any(len(values) > 1 for values in seen.values()):
            raise RecoverySplitError(f"{name} is shared across isolated datasets")


def validate_recovery_splits(
    manifests: Sequence[RecoveryRootManifest],
) -> Mapping[str, int]:
    if not manifests:
        raise RecoverySplitError("recovery dataset is empty")
    root_ids: set[str] = set()
    events: dict[
        tuple[int, str], list[tuple[str, tuple[str, str, str]]]
    ] = {}
    labels: dict[str, str] = {}
    for item in manifests:
        if not isinstance(item, RecoveryRootManifest):
            raise RecoverySplitError("dataset contains a non-recovery-root manifest")
        if item.root_id in root_ids:
            raise RecoverySplitError(f"duplicate root_id: {item.root_id}")
        root_ids.add(item.root_id)
        if not isinstance(item.split, RecoverySplit):
            raise RecoverySplitError("recovery split is invalid")
        if item.deviation_status not in {
            "ANOMALY_CANDIDATE",
            "CONFIRMED_DEVIATION",
        }:
            raise RecoverySplitError("deviation status is invalid")
        _evidence_records(item)
        labels[item.root_id] = item.split.value
        event_key = (item.task_id, item.deviation_event_id)
        lineage = (
            item.event_origin_parent_sha256,
            item.recovery_group_id,
            item.independence_unit_id,
        )
        events.setdefault(event_key, []).append((item.deviation_status, lineage))
    for event_items in events.values():
        statuses = {status for status, _lineage in event_items}
        lineages = {lineage for _status, lineage in event_items}
        if statuses == {"ANOMALY_CANDIDATE", "CONFIRMED_DEVIATION"} and len(lineages) != 1:
            raise RecoverySplitError(
                "candidate/confirmed semantic event changed event_origin, "
                "recovery_group_id, or independence_unit_id"
            )
    _reject_relations_across(manifests, labels)
    return {
        "TRAIN": sum(item.split is RecoverySplit.TRAIN for item in manifests),
        "DEV": sum(item.split is RecoverySplit.DEV for item in manifests),
        "HELDOUT": sum(item.split is RecoverySplit.HELDOUT for item in manifests),
        "ANOMALY_CANDIDATE": sum(
            item.deviation_status == "ANOMALY_CANDIDATE" for item in manifests
        ),
        "CONFIRMED_DEVIATION": sum(
            item.deviation_status == "CONFIRMED_DEVIATION" for item in manifests
        ),
        "unique_independence_units": len(
            {item.independence_unit_id for item in manifests}
        ),
        "total": len(manifests),
    }


def _role_isolation_key(name: str, value: Any) -> str:
    return f"{name}:{_json_sha256(value)}"


def _unit_isolation_keys(manifests: Sequence[RecoveryRootManifest]) -> tuple[str, ...]:
    keys: set[str] = set()
    for item in manifests:
        for name, relation in _RELATION_CHECKS:
            value = relation(item)
            if value is not None:
                keys.add(_role_isolation_key(name, value))
    return tuple(sorted(keys))


def _validate_role_entry(entry: RoleAllocationEntry) -> None:
    if not isinstance(entry, RoleAllocationEntry):
        raise RecoverySplitError("role allocation entry type is invalid")
    _require_sha256("entry source dataset hash", entry.source_dataset_sha256)
    _require_text("entry independence_unit_id", entry.independence_unit_id)
    if not isinstance(entry.role, DatasetRole):
        raise RecoverySplitError("dataset role is invalid")
    if (
        type(entry.role_isolation_keys) is not tuple
        or entry.role_isolation_keys != tuple(sorted(set(entry.role_isolation_keys)))
        or any(
            not isinstance(key, str)
            or ":" not in key
            or _SHA256.fullmatch(key.rsplit(":", 1)[-1]) is None
            for key in entry.role_isolation_keys
        )
    ):
        raise RecoverySplitError("role isolation keys must be sorted canonical hashes")


def _validate_allocation_envelope(allocation: RoleAllocationManifest) -> None:
    if not isinstance(allocation, RoleAllocationManifest):
        raise RecoverySplitError("role allocation manifest type is invalid")
    if type(allocation.schema_version) is not int or allocation.schema_version != 1:
        raise RecoverySplitError("unsupported role allocation schema")
    _require_sha256("source dataset hash", allocation.source_dataset_sha256)
    _require_sha256("registry parent head hash", allocation.registry_parent_head_sha256)
    _require_sha256("allocation hash", allocation.allocation_sha256)
    _require_text("allocation version", allocation.allocation_version)
    if (
        type(allocation.source_independence_unit_ids) is not tuple
        or allocation.source_independence_unit_ids
        != tuple(sorted(set(allocation.source_independence_unit_ids)))
    ):
        raise RecoverySplitError("source independence units must be sorted and unique")
    if type(allocation.entries) is not tuple or not allocation.entries:
        raise RecoverySplitError("role allocation entries must be a nonempty tuple")
    for entry in allocation.entries:
        _validate_role_entry(entry)
        if entry.source_dataset_sha256 != allocation.source_dataset_sha256:
            raise RecoverySplitError("role entry source dataset hash mismatch")
    if tuple(entry.independence_unit_id for entry in allocation.entries) != tuple(
        sorted(entry.independence_unit_id for entry in allocation.entries)
    ):
        raise RecoverySplitError("role allocation entries must be sorted")
    if tuple(entry.independence_unit_id for entry in allocation.entries) != (
        allocation.source_independence_unit_ids
    ):
        raise RecoverySplitError(
            "complete role allocation entries must match the declared source units"
        )
    if _content_hash(allocation, "allocation_sha256") != allocation.allocation_sha256:
        raise RecoverySplitError("role allocation self-hash mismatch")


def _validate_role_compatibility(
    manifest: RecoveryRootManifest, role: DatasetRole
) -> None:
    if manifest.collection_label is CollectionLabel.IMPORTED_FROZEN:
        compatible = {
            DatasetRole.TRAIN: RecoverySplit.TRAIN,
            DatasetRole.DEV_SELECTION: RecoverySplit.DEV,
            DatasetRole.PERMIT_CALIBRATION: RecoverySplit.HELDOUT,
            DatasetRole.PAPER_CONFIRMATION: RecoverySplit.HELDOUT,
        }
        if manifest.split is not compatible[role]:
            raise RecoverySplitError(
                "dataset role is incompatible with frozen recovery split"
            )
        if role in {
            DatasetRole.PERMIT_CALIBRATION,
            DatasetRole.PAPER_CONFIRMATION,
        }:
            for field_name in (
                "base_prompt_sha256",
                "base_checkpoint_sha256",
                "policy_client_config_sha256",
                "grounding_rule_sha256",
                "event_detector_sha256",
                "monitor_contract_sha256",
            ):
                _require_sha256(
                    f"frozen {field_name}", getattr(manifest, field_name)
                )
        return
    if manifest.collection_label is not CollectionLabel.DEV_COLLECTION:
        raise RecoverySplitError("unknown recovery collection label")
    if manifest.split is not RecoverySplit.DEV:
        raise RecoverySplitError("DEV_COLLECTION roots must remain in the DEV split")
    if role not in {DatasetRole.TRAIN, DatasetRole.DEV_SELECTION}:
        raise RecoverySplitError("DEV_COLLECTION roots cannot enter permit/paper roles")


def validate_role_allocation(
    manifests: Sequence[RecoveryRootManifest], allocation: RoleAllocationManifest
) -> None:
    validate_recovery_splits(manifests)
    _validate_allocation_envelope(allocation)
    expected_source = recovery_source_dataset_sha256(manifests)
    if allocation.source_dataset_sha256 != expected_source:
        raise RecoverySplitError("role allocation source dataset hash mismatch")
    units = tuple(sorted({item.independence_unit_id for item in manifests}))
    if allocation.source_independence_unit_ids != units:
        raise RecoverySplitError("role allocation does not list the complete source units")
    entry_units = tuple(entry.independence_unit_id for entry in allocation.entries)
    if entry_units != units:
        raise RecoverySplitError(
            "complete role allocation must cover each source unit exactly once"
        )
    by_unit: dict[str, list[RecoveryRootManifest]] = {unit: [] for unit in units}
    for item in manifests:
        by_unit[item.independence_unit_id].append(item)
    role_by_root: dict[str, str] = {}
    for entry in allocation.entries:
        expected_keys = _unit_isolation_keys(by_unit[entry.independence_unit_id])
        if entry.role_isolation_keys != expected_keys:
            raise RecoverySplitError(
                f"role isolation keys do not match source unit {entry.independence_unit_id}"
            )
        for item in by_unit[entry.independence_unit_id]:
            _validate_role_compatibility(item, entry.role)
            role_by_root[item.root_id] = entry.role.value
    _reject_relations_across(manifests, role_by_root)


def make_role_allocation(
    manifests: Sequence[RecoveryRootManifest],
    *,
    roles: Mapping[str, DatasetRole],
    allocation_version: str,
    registry_parent_head_sha256: str,
) -> RoleAllocationManifest:
    source_hash = recovery_source_dataset_sha256(manifests)
    units = tuple(sorted({item.independence_unit_id for item in manifests}))
    if set(roles) != set(units):
        raise RecoverySplitError("roles must provide a complete source-unit allocation")
    by_unit: dict[str, list[RecoveryRootManifest]] = {unit: [] for unit in units}
    for item in manifests:
        by_unit[item.independence_unit_id].append(item)
    entries = tuple(
        RoleAllocationEntry(
            source_dataset_sha256=source_hash,
            independence_unit_id=unit,
            role_isolation_keys=_unit_isolation_keys(by_unit[unit]),
            role=roles[unit],
        )
        for unit in units
    )
    unsigned = RoleAllocationManifest(
        schema_version=1,
        source_dataset_sha256=source_hash,
        source_independence_unit_ids=units,
        entries=entries,
        allocation_version=allocation_version,
        registry_parent_head_sha256=registry_parent_head_sha256,
        allocation_sha256="",
    )
    allocation = RoleAllocationManifest(
        **{
            **asdict(unsigned),
            "entries": entries,
            "allocation_sha256": _content_hash(unsigned, "allocation_sha256"),
        }
    )
    validate_role_allocation(manifests, allocation)
    return allocation


def _validate_registry_head(head: RoleRegistryHead) -> None:
    if (
        not isinstance(head, RoleRegistryHead)
        or type(head.schema_version) is not int
        or head.schema_version != 1
    ):
        raise RecoverySplitError("role registry head schema is invalid")
    _require_text("registry ID", head.registry_id)
    if type(head.revision) is not int or head.revision < 0:
        raise RecoverySplitError("registry revision must be nonnegative")
    _require_sha256("registry parent head", head.parent_head_sha256, optional=True)
    _require_sha256("registry head", head.head_sha256)
    if (head.revision == 0) != (head.parent_head_sha256 is None):
        raise RecoverySplitError("registry parent/revision linkage is invalid")
    if type(head.allocations) is not tuple:
        raise RecoverySplitError("registry allocations must be a tuple")
    seen_units: set[str] = set()
    key_roles: dict[str, DatasetRole] = {}
    for entry in head.allocations:
        _validate_role_entry(entry)
        if entry.independence_unit_id in seen_units:
            raise RecoverySplitError("historical independence_unit_id is duplicated")
        seen_units.add(entry.independence_unit_id)
        for key in entry.role_isolation_keys:
            previous = key_roles.setdefault(key, entry.role)
            if previous is not entry.role:
                raise RecoverySplitError("historical role-isolation key changed role")
    if _content_hash(head, "head_sha256") != head.head_sha256:
        raise RecoverySplitError("role registry head self-hash mismatch")


def _new_registry_head(
    registry_id: str,
    revision: int,
    parent: str | None,
    allocations: tuple[RoleAllocationEntry, ...],
) -> RoleRegistryHead:
    unsigned = RoleRegistryHead(
        schema_version=1,
        registry_id=registry_id,
        revision=revision,
        parent_head_sha256=parent,
        allocations=allocations,
        head_sha256="",
    )
    return RoleRegistryHead(
        **{
            **asdict(unsigned),
            "allocations": allocations,
            "head_sha256": _content_hash(unsigned, "head_sha256"),
        }
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        encoded = _canonical_json(value).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_json(path: Path, value: Any) -> None:
    encoded = _canonical_json(value).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if path.exists():
            _preflight_immutable_json(path, encoded)
            return
        _atomic_write_json(path, value)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _preflight_immutable_json(path: Path, encoded: bytes) -> None:
    if not path.exists():
        return
    try:
        existing = path.read_bytes()
    except OSError as error:
        raise RecoverySplitError(
            f"immutable manifest is unreadable: {path}"
        ) from error
    if existing != encoded:
        raise RecoverySplitError(
            f"immutable manifest already exists with different content: {path}"
        )


class RoleRegistry:
    def __init__(self, path: Path | str, registry_id: str) -> None:
        self.path = Path(path)
        _require_text("registry ID", registry_id)
        self.registry_id = registry_id
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def genesis_head(self) -> RoleRegistryHead:
        return _new_registry_head(self.registry_id, 0, None, ())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def initialize(self) -> RoleRegistryHead:
        with self._locked():
            if self.path.exists():
                return self._read_unlocked()
            head = self.genesis_head()
            _atomic_write_json(self.path, asdict(head))
            return head

    def _read_unlocked(self) -> RoleRegistryHead:
        if not self.path.is_file():
            raise RecoverySplitError("canonical role registry is absent")
        try:
            encoded = self.path.read_text(encoding="utf-8")
            payload = json.loads(encoded)
        except (OSError, json.JSONDecodeError) as error:
            raise RecoverySplitError("canonical role registry is unreadable") from error
        if _canonical_json(payload) != encoded:
            raise RecoverySplitError("role registry JSON is not canonical")
        head = _registry_head_from_payload(payload)
        if head.registry_id != self.registry_id:
            raise RecoverySplitError("canonical role registry ID mismatch")
        return head

    def read_head(self) -> RoleRegistryHead:
        with self._locked():
            return self._read_unlocked()

    def read_head_sha256(self) -> str:
        return self.read_head().head_sha256


def require_canonical_role_registry(
    path: Path | str, registry_id: str
) -> RoleRegistry:
    if Path(path).resolve() != CANONICAL_ROLE_REGISTRY_PATH.resolve():
        raise RecoverySplitError("canonical role registry path mismatch")
    if registry_id != CANONICAL_ROLE_REGISTRY_ID:
        raise RecoverySplitError("canonical role registry ID mismatch")
    return RoleRegistry(CANONICAL_ROLE_REGISTRY_PATH, CANONICAL_ROLE_REGISTRY_ID)


def append_role_allocation(
    registry: RoleRegistry,
    allocation: RoleAllocationManifest,
    *,
    expected_head: str,
) -> RoleRegistryHead:
    if not isinstance(registry, RoleRegistry):
        raise RecoverySplitError("append requires the canonical RoleRegistry")
    with registry._locked():
        current = registry._read_unlocked()
        updated = preview_role_allocation_append(
            current, allocation, expected_head=expected_head
        )
        _atomic_write_json(registry.path, asdict(updated))
        return updated


def preview_role_allocation_append(
    current: RoleRegistryHead,
    allocation: RoleAllocationManifest,
    *,
    expected_head: str,
) -> RoleRegistryHead:
    _validate_registry_head(current)
    _validate_allocation_envelope(allocation)
    _require_sha256("expected head", expected_head)
    if current.head_sha256 != expected_head:
        raise RecoverySplitError("role registry expected head CAS mismatch")
    if allocation.registry_parent_head_sha256 != current.head_sha256:
        raise RecoverySplitError("allocation does not bind the expected head")
    historical_units = {item.independence_unit_id for item in current.allocations}
    historical_key_roles = {
        key: item.role
        for item in current.allocations
        for key in item.role_isolation_keys
    }
    for entry in allocation.entries:
        if entry.independence_unit_id in historical_units:
            raise RecoverySplitError(
                "historical independence_unit_id cannot be repackaged or reassigned"
            )
        for key in entry.role_isolation_keys:
            prior_role = historical_key_roles.get(key)
            if prior_role is not None and prior_role is not entry.role:
                relation = key.split(":", 1)[0]
                raise RecoverySplitError(
                    f"historical {relation} relation cannot be reassigned"
                )
    updated = _new_registry_head(
        current.registry_id,
        current.revision + 1,
        current.head_sha256,
        current.allocations + allocation.entries,
    )
    _validate_registry_head(updated)
    return updated


def _registry_contains_allocation(
    head: RoleRegistryHead, allocation: RoleAllocationManifest
) -> None:
    registered = set(head.allocations)
    missing = [
        entry.independence_unit_id
        for entry in allocation.entries
        if entry not in registered
    ]
    if missing:
        raise RecoverySplitError(
            f"role allocation is absent from canonical registry head: {missing}"
        )


def _validate_fresh_label(
    label: FreshRecoveryLabel, manifest: RecoveryRootManifest
) -> None:
    if (
        not isinstance(label, FreshRecoveryLabel)
        or type(label.schema_version) is not int
        or label.schema_version != 1
    ):
        raise RecoverySplitError("fresh label schema is invalid")
    for name in (
        "root_id",
        "simulator_state_sha256",
        "fresh_observation_sha256",
        "historical_evidence_sha256",
        "relevant_fact_sha256",
        "fact_universe_sha256",
        "fresh_fact_evidence_hash",
        "certificate_sha256",
        "grounding_rule_sha256",
        "event_detector_sha256",
        "monitor_contract_sha256",
        "label_sha256",
    ):
        _require_sha256(f"fresh label {name}", getattr(label, name))
    for name in ("deviation_event_id", "graph_version", "labeler_version"):
        _require_text(f"fresh label {name}", getattr(label, name))
    if type(label.fresh_fact_epoch_id) is not int or label.fresh_fact_epoch_id < 0:
        raise RecoverySplitError("fresh fact epoch must be nonnegative")
    if _content_hash(label, "label_sha256") != label.label_sha256:
        raise RecoverySplitError("fresh label self-hash mismatch")
    exact = {
        "root_id": manifest.root_id,
        "simulator_state_sha256": manifest.simulator_state_sha256,
        "deviation_event_id": manifest.deviation_event_id,
        "historical_evidence_sha256": historical_evidence_sha256(manifest),
        "relevant_fact_sha256": manifest.relevant_fact_sha256,
        "fact_universe_sha256": manifest.fact_universe_sha256,
        "graph_version": manifest.source_graph_version,
        "grounding_rule_sha256": manifest.grounding_rule_sha256,
        "event_detector_sha256": manifest.event_detector_sha256,
        "monitor_contract_sha256": manifest.monitor_contract_sha256,
    }
    for name, expected in exact.items():
        if getattr(label, name) != expected:
            raise RecoverySplitError(f"fresh label {name} mismatch")
    if label.fresh_observation_sha256 == manifest.observation_sha256:
        raise RecoverySplitError("fresh label must bind a newly observed state image")
    expected_evidence = fresh_fact_evidence_sha256(
        fresh_observation_sha256=label.fresh_observation_sha256,
        fresh_fact_epoch_id=label.fresh_fact_epoch_id,
        fact_universe_sha256=label.fact_universe_sha256,
        relevant_fact_sha256=label.relevant_fact_sha256,
    )
    if label.fresh_fact_evidence_hash != expected_evidence:
        raise RecoverySplitError("fresh label fact evidence hash is not recomputable")


def _validate_training_manifest(manifest: TrainingManifest) -> None:
    if (
        not isinstance(manifest, TrainingManifest)
        or type(manifest.schema_version) is not int
        or manifest.schema_version != 1
    ):
        raise RecoverySplitError("training manifest schema is invalid")
    if not isinstance(manifest.role, DatasetRole):
        raise RecoverySplitError("training manifest role is invalid")
    _require_text("training builder version", manifest.builder_version)
    for name in (
        "source_dataset_sha256",
        "role_allocation_sha256",
        "role_registry_head_sha256",
        "manifest_sha256",
    ):
        _require_sha256(f"training manifest {name}", getattr(manifest, name))
    if type(manifest.entries) is not tuple or not manifest.entries:
        raise RecoverySplitError("training manifest entries must be nonempty")
    if manifest.entries != tuple(sorted(manifest.entries, key=lambda item: item.root_id)):
        raise RecoverySplitError("training manifest entries are not sorted")
    roots: set[str] = set()
    associations: set[tuple[str, str, str]] = set()
    for entry in manifest.entries:
        if not isinstance(entry, TrainingManifestEntry):
            raise RecoverySplitError("training manifest entry type is invalid")
        for name in ("root_id", "recovery_group_id", "independence_unit_id"):
            _require_text(f"training entry {name}", getattr(entry, name))
        _require_sha256(
            "training entry fresh label hash",
            entry.fresh_label_sha256,
            optional=True,
        )
        association = (
            entry.root_id,
            entry.recovery_group_id,
            entry.independence_unit_id,
        )
        if entry.root_id in roots or association in associations:
            raise RecoverySplitError("duplicate root/group/unit training association")
        roots.add(entry.root_id)
        associations.add(association)
    if _content_hash(manifest, "manifest_sha256") != manifest.manifest_sha256:
        raise RecoverySplitError("training manifest self-hash mismatch")


def build_training_manifest(
    manifests: Sequence[RecoveryRootManifest],
    fresh_labels: Sequence[FreshRecoveryLabel],
    *,
    role: DatasetRole,
    source_dataset_sha256: str,
    role_allocation: RoleAllocationManifest,
    role_registry_head: RoleRegistryHead,
    builder_version: str,
) -> TrainingManifest:
    if not isinstance(role, DatasetRole):
        raise RecoverySplitError("explicit dataset role is required")
    _require_text("builder version", builder_version)
    validate_role_allocation(manifests, role_allocation)
    expected_source = recovery_source_dataset_sha256(manifests)
    if source_dataset_sha256 != expected_source:
        raise RecoverySplitError("training source dataset hash mismatch")
    if role_allocation.source_dataset_sha256 != source_dataset_sha256:
        raise RecoverySplitError("training allocation source hash mismatch")
    _validate_registry_head(role_registry_head)
    _registry_contains_allocation(role_registry_head, role_allocation)
    label_by_root: dict[str, FreshRecoveryLabel] = {}
    source_by_root = {item.root_id: item for item in manifests}
    for label in fresh_labels:
        if label.root_id in label_by_root:
            raise RecoverySplitError(f"duplicate fresh label for root {label.root_id}")
        root = source_by_root.get(label.root_id)
        if root is None:
            raise RecoverySplitError("fresh label refers to a root outside the source dataset")
        _validate_fresh_label(label, root)
        label_by_root[label.root_id] = label
    role_by_unit = {
        entry.independence_unit_id: entry.role for entry in role_allocation.entries
    }
    selected = sorted(
        (
            item
            for item in manifests
            if role_by_unit[item.independence_unit_id] is role
        ),
        key=lambda item: item.root_id,
    )
    if not selected:
        raise RecoverySplitError(f"source dataset has no roots allocated to {role.value}")
    entries: list[TrainingManifestEntry] = []
    for item in selected:
        if item.deviation_status == "ANOMALY_CANDIDATE":
            raise RecoverySplitError("anomaly candidate roots are never training eligible")
        if item.deviation_status != "CONFIRMED_DEVIATION":
            raise RecoverySplitError("only confirmed deviation roots are eligible")
        label = label_by_root.get(item.root_id)
        if item.certificate_state == "STALE":
            if label is None:
                raise RecoverySplitError("stale roots require a matching signed fresh label")
        elif item.certificate_state in {"CURRENT", "RECONCILED"}:
            if label is not None:
                raise RecoverySplitError("fresh labels may rehabilitate stale roots only")
        else:
            raise RecoverySplitError("root certificate state is invalid")
        entries.append(
            TrainingManifestEntry(
                root_id=item.root_id,
                recovery_group_id=item.recovery_group_id,
                independence_unit_id=item.independence_unit_id,
                fresh_label_sha256=None if label is None else label.label_sha256,
            )
        )
    unsigned = TrainingManifest(
        schema_version=1,
        role=role,
        entries=tuple(entries),
        source_dataset_sha256=source_dataset_sha256,
        role_allocation_sha256=role_allocation.allocation_sha256,
        role_registry_head_sha256=role_registry_head.head_sha256,
        builder_version=builder_version,
        manifest_sha256="",
    )
    result = TrainingManifest(
        **{
            **asdict(unsigned),
            "role": role,
            "entries": unsigned.entries,
            "manifest_sha256": _content_hash(unsigned, "manifest_sha256"),
        }
    )
    _validate_training_manifest(result)
    return result


def validate_dataset_roles(
    manifests: Sequence[TrainingManifest],
    *,
    role_allocations: Sequence[RoleAllocationManifest],
    role_registry_head: RoleRegistryHead,
    source_datasets: Mapping[
        str, Sequence[RecoveryRootManifest]
    ] | None = None,
) -> Mapping[str, int]:
    if not manifests:
        raise RecoverySplitError("role manifest dataset is empty")
    _validate_registry_head(role_registry_head)
    allocations_by_source: dict[str, RoleAllocationManifest] = {}
    for allocation in role_allocations:
        _validate_allocation_envelope(allocation)
        if allocation.source_dataset_sha256 in allocations_by_source:
            raise RecoverySplitError("duplicate allocation for source dataset")
        allocations_by_source[allocation.source_dataset_sha256] = allocation
        _registry_contains_allocation(role_registry_head, allocation)
    registry_entries_by_source: dict[str, set[RoleAllocationEntry]] = {}
    for entry in role_registry_head.allocations:
        registry_entries_by_source.setdefault(
            entry.source_dataset_sha256, set()
        ).add(entry)
    if set(allocations_by_source) != set(registry_entries_by_source):
        raise RecoverySplitError(
            "complete role validation omitted a historical registry allocation"
        )
    for source_hash, allocation in allocations_by_source.items():
        if set(allocation.entries) != registry_entries_by_source[source_hash]:
            raise RecoverySplitError(
                "role allocation does not exactly match canonical registry history"
            )
    source_by_root: dict[str, dict[str, RecoveryRootManifest]] = {}
    if source_datasets is not None:
        if set(source_datasets) != set(registry_entries_by_source):
            raise RecoverySplitError(
                "complete immutable source datasets must cover registry history"
            )
        for source_hash, roots in source_datasets.items():
            if recovery_source_dataset_sha256(roots) != source_hash:
                raise RecoverySplitError("immutable source dataset hash mismatch")
            validate_role_allocation(roots, allocations_by_source[source_hash])
            source_by_root[source_hash] = {item.root_id: item for item in roots}
    relation_roles: dict[str, DatasetRole] = {}
    root_roles: dict[str, DatasetRole] = {}
    group_roles: dict[str, DatasetRole] = {}
    unit_roles: dict[str, DatasetRole] = {}
    covered_units: dict[str, set[str]] = {}
    covered_roots: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for manifest in manifests:
        _validate_training_manifest(manifest)
        if manifest.role_registry_head_sha256 != role_registry_head.head_sha256:
            raise RecoverySplitError("training manifest uses a stale or alternative registry head")
        allocation = allocations_by_source.get(manifest.source_dataset_sha256)
        if allocation is None or allocation.allocation_sha256 != manifest.role_allocation_sha256:
            raise RecoverySplitError("training manifest allocation is absent or mismatched")
        entry_by_unit = {
            entry.independence_unit_id: entry for entry in allocation.entries
        }
        for entry in allocation.entries:
            for key in entry.role_isolation_keys:
                previous = relation_roles.setdefault(key, entry.role)
                if previous is not entry.role:
                    relation = key.split(":", 1)[0]
                    raise RecoverySplitError(f"{relation} is shared across dataset roles")
        for entry in manifest.entries:
            allocation_entry = entry_by_unit.get(entry.independence_unit_id)
            if allocation_entry is None or allocation_entry.role is not manifest.role:
                raise RecoverySplitError("training independence_unit_id has the wrong role")
            expected_unit_key = _role_isolation_key(
                "independence_unit_id", entry.independence_unit_id
            )
            expected_group_key = _role_isolation_key(
                "recovery_group_id", entry.recovery_group_id
            )
            if expected_unit_key not in allocation_entry.role_isolation_keys:
                raise RecoverySplitError("training independence_unit_id is outside allocation")
            if expected_group_key not in allocation_entry.role_isolation_keys:
                raise RecoverySplitError("training recovery_group_id is outside allocation")
            if source_datasets is not None:
                source_root = source_by_root[manifest.source_dataset_sha256].get(
                    entry.root_id
                )
                if source_root is None:
                    raise RecoverySplitError(
                        "training root_id is outside the immutable source dataset"
                    )
                if (
                    entry.recovery_group_id != source_root.recovery_group_id
                    or entry.independence_unit_id
                    != source_root.independence_unit_id
                ):
                    raise RecoverySplitError(
                        "training root/group/unit association mismatches source"
                    )
                if source_root.deviation_status != "CONFIRMED_DEVIATION":
                    raise RecoverySplitError(
                        "candidate source root is not derived-manifest eligible"
                    )
                if source_root.certificate_state == "STALE":
                    if entry.fresh_label_sha256 is None:
                        raise RecoverySplitError(
                            "stale source root is missing its fresh label hash"
                        )
                elif source_root.certificate_state in {"CURRENT", "RECONCILED"}:
                    if entry.fresh_label_sha256 is not None:
                        raise RecoverySplitError(
                            "current source root has an unexpected fresh label hash"
                        )
                else:
                    raise RecoverySplitError("source root certificate state is invalid")
                source_covered = covered_roots.setdefault(
                    manifest.source_dataset_sha256, set()
                )
                if entry.root_id in source_covered:
                    raise RecoverySplitError(
                        "duplicate training root_id across role manifests"
                    )
                source_covered.add(entry.root_id)
            covered_units.setdefault(manifest.source_dataset_sha256, set()).add(
                entry.independence_unit_id
            )
            for name, value, seen in (
                ("root_id", entry.root_id, root_roles),
                ("recovery_group_id", entry.recovery_group_id, group_roles),
                ("independence_unit_id", entry.independence_unit_id, unit_roles),
            ):
                previous = seen.setdefault(value, manifest.role)
                if previous is not manifest.role:
                    raise RecoverySplitError(f"{name} is shared across dataset roles")
        counts[manifest.role.value] = counts.get(manifest.role.value, 0) + len(
            manifest.entries
        )
    for source_hash, allocation in allocations_by_source.items():
        expected_units = {
            entry.independence_unit_id for entry in allocation.entries
        }
        if covered_units.get(source_hash, set()) != expected_units:
            raise RecoverySplitError(
                "complete dataset-role validation detected an omitted allocated unit"
            )
    if source_datasets is None:
        raise RecoverySplitError(
            "immutable source datasets are required for complete role validation"
        )
    for source_hash, roots in source_datasets.items():
        expected_roots = {item.root_id for item in roots}
        if covered_roots.get(source_hash, set()) != expected_roots:
            raise RecoverySplitError(
                "complete dataset-role validation omitted an immutable source root"
            )
    counts["total"] = sum(counts.values())
    return counts


def _exact_mapping(payload: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise RecoverySplitError(f"{name} schema fields mismatch")
    return payload


def _role_entry_from_payload(payload: Any) -> RoleAllocationEntry:
    values = _exact_mapping(
        payload,
        {"source_dataset_sha256", "independence_unit_id", "role_isolation_keys", "role"},
        "role allocation entry",
    )
    try:
        entry = RoleAllocationEntry(
            source_dataset_sha256=values["source_dataset_sha256"],
            independence_unit_id=values["independence_unit_id"],
            role_isolation_keys=tuple(values["role_isolation_keys"]),
            role=DatasetRole(values["role"]),
        )
    except (TypeError, ValueError) as error:
        raise RecoverySplitError("role allocation entry types are invalid") from error
    _validate_role_entry(entry)
    return entry


def _allocation_from_payload(payload: Any) -> RoleAllocationManifest:
    values = _exact_mapping(
        payload,
        {
            "schema_version",
            "source_dataset_sha256",
            "source_independence_unit_ids",
            "entries",
            "allocation_version",
            "registry_parent_head_sha256",
            "allocation_sha256",
        },
        "role allocation manifest",
    )
    try:
        allocation = RoleAllocationManifest(
            schema_version=values["schema_version"],
            source_dataset_sha256=values["source_dataset_sha256"],
            source_independence_unit_ids=tuple(values["source_independence_unit_ids"]),
            entries=tuple(_role_entry_from_payload(item) for item in values["entries"]),
            allocation_version=values["allocation_version"],
            registry_parent_head_sha256=values["registry_parent_head_sha256"],
            allocation_sha256=values["allocation_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise RecoverySplitError("role allocation manifest types are invalid") from error
    _validate_allocation_envelope(allocation)
    return allocation


def _registry_head_from_payload(payload: Any) -> RoleRegistryHead:
    values = _exact_mapping(
        payload,
        {
            "schema_version",
            "registry_id",
            "revision",
            "parent_head_sha256",
            "allocations",
            "head_sha256",
        },
        "role registry head",
    )
    try:
        head = RoleRegistryHead(
            schema_version=values["schema_version"],
            registry_id=values["registry_id"],
            revision=values["revision"],
            parent_head_sha256=values["parent_head_sha256"],
            allocations=tuple(_role_entry_from_payload(item) for item in values["allocations"]),
            head_sha256=values["head_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise RecoverySplitError("role registry head types are invalid") from error
    _validate_registry_head(head)
    return head


def _fresh_label_from_payload(payload: Any) -> FreshRecoveryLabel:
    fields = {
        "schema_version",
        "root_id",
        "simulator_state_sha256",
        "fresh_observation_sha256",
        "deviation_event_id",
        "historical_evidence_sha256",
        "relevant_fact_sha256",
        "fresh_fact_epoch_id",
        "fact_universe_sha256",
        "fresh_fact_evidence_hash",
        "graph_version",
        "certificate_sha256",
        "grounding_rule_sha256",
        "event_detector_sha256",
        "monitor_contract_sha256",
        "labeler_version",
        "label_sha256",
    }
    values = _exact_mapping(payload, fields, "fresh recovery label")
    try:
        return FreshRecoveryLabel(**values)
    except TypeError as error:
        raise RecoverySplitError("fresh recovery label types are invalid") from error


def _training_manifest_from_payload(payload: Any) -> TrainingManifest:
    values = _exact_mapping(
        payload,
        {
            "schema_version",
            "role",
            "entries",
            "source_dataset_sha256",
            "role_allocation_sha256",
            "role_registry_head_sha256",
            "builder_version",
            "manifest_sha256",
        },
        "training manifest",
    )
    entries: list[TrainingManifestEntry] = []
    try:
        for item in values["entries"]:
            entry_values = _exact_mapping(
                item,
                {
                    "root_id",
                    "recovery_group_id",
                    "independence_unit_id",
                    "fresh_label_sha256",
                },
                "training manifest entry",
            )
            entries.append(TrainingManifestEntry(**entry_values))
        manifest = TrainingManifest(
            schema_version=values["schema_version"],
            role=DatasetRole(values["role"]),
            entries=tuple(entries),
            source_dataset_sha256=values["source_dataset_sha256"],
            role_allocation_sha256=values["role_allocation_sha256"],
            role_registry_head_sha256=values["role_registry_head_sha256"],
            builder_version=values["builder_version"],
            manifest_sha256=values["manifest_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise RecoverySplitError("training manifest types are invalid") from error
    _validate_training_manifest(manifest)
    return manifest


def _load_canonical_json(path: Path, name: str) -> Any:
    if not path.is_file():
        raise RecoverySplitError(f"{name} file is absent")
    try:
        encoded = path.read_text(encoding="utf-8")
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as error:
        raise RecoverySplitError(f"{name} is unreadable") from error
    if _canonical_json(payload) != encoded:
        raise RecoverySplitError(f"{name} JSON is not canonical")
    return payload


def load_role_allocation(path: Path | str) -> RoleAllocationManifest:
    return _allocation_from_payload(_load_canonical_json(Path(path), "role allocation"))


def load_fresh_recovery_labels(path: Path | str) -> tuple[FreshRecoveryLabel, ...]:
    payload = _load_canonical_json(Path(path), "fresh-label file")
    if not isinstance(payload, list):
        raise RecoverySplitError("fresh-label file must contain a JSON list")
    return tuple(_fresh_label_from_payload(item) for item in payload)


def load_training_manifest(path: Path | str) -> TrainingManifest:
    source = Path(path)
    if source.is_dir():
        raise RecoverySplitError(
            "a training manifest file is required; raw root directories are forbidden"
        )
    return _training_manifest_from_payload(
        _load_canonical_json(source, "training manifest")
    )


def write_role_allocation(path: Path | str, allocation: RoleAllocationManifest) -> None:
    _validate_allocation_envelope(allocation)
    _write_immutable_json(Path(path), asdict(allocation))


def write_training_manifest(path: Path | str, manifest: TrainingManifest) -> None:
    _validate_training_manifest(manifest)
    _write_immutable_json(Path(path), asdict(manifest))


def _run_registry_provenance_payload(
    path: Path,
    *,
    source_dataset_sha256: str,
    role_allocation_sha256: str,
    role_registry_head_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("source dataset", source_dataset_sha256),
        ("role allocation", role_allocation_sha256),
        ("role registry head", role_registry_head_sha256),
    ):
        _require_sha256(name, value)
    payload: dict[str, Any] = {}
    if path.exists():
        loaded = _load_canonical_json(path, "run manifest")
        if not isinstance(loaded, dict):
            raise RecoverySplitError("run manifest must be a JSON object")
        payload.update(loaded)
    values = {
        "recovery_source_dataset_sha256": source_dataset_sha256,
        "recovery_role_allocation_sha256": role_allocation_sha256,
        "recovery_role_registry_head_sha256": role_registry_head_sha256,
    }
    for key, value in values.items():
        if key in payload and payload[key] != value:
            raise RecoverySplitError(f"run manifest already binds a different {key}")
        payload[key] = value
    return payload


def preflight_role_allocation_outputs(
    allocation_path: Path | str,
    allocation: RoleAllocationManifest,
    run_path: Path | str,
    *,
    role_registry_head_sha256: str,
) -> None:
    _validate_allocation_envelope(allocation)
    _preflight_immutable_json(
        Path(allocation_path), _canonical_json(asdict(allocation)).encode("utf-8")
    )
    _run_registry_provenance_payload(
        Path(run_path),
        source_dataset_sha256=allocation.source_dataset_sha256,
        role_allocation_sha256=allocation.allocation_sha256,
        role_registry_head_sha256=role_registry_head_sha256,
    )


def write_run_registry_provenance(
    path: Path | str,
    *,
    source_dataset_sha256: str,
    role_allocation_sha256: str,
    role_registry_head_sha256: str,
) -> None:
    destination = Path(path)
    payload = _run_registry_provenance_payload(
        destination,
        source_dataset_sha256=source_dataset_sha256,
        role_allocation_sha256=role_allocation_sha256,
        role_registry_head_sha256=role_registry_head_sha256,
    )
    _atomic_write_json(destination, payload)


def load_recovery_manifests(paths: Sequence[Path | str]) -> tuple[RecoveryRootManifest, ...]:
    manifest_paths: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir():
            raise RecoverySplitError(f"recovery dataset directory is absent: {path}")
        direct = path / "recovery_root.json"
        if direct.is_file():
            manifest_paths.add(direct.resolve())
        manifest_paths.update(item.resolve() for item in path.rglob("recovery_root.json"))
    if not manifest_paths:
        raise RecoverySplitError("recovery dataset is empty")
    manifests: list[RecoveryRootManifest] = []
    for manifest_path in sorted(manifest_paths):
        try:
            manifest, _state = load_recovery_root(manifest_path.parent)
        except (OSError, TypeError, ValueError) as error:
            raise RecoverySplitError(
                f"invalid recovery root {manifest_path.parent}: {error}"
            ) from error
        manifests.append(manifest)
    return tuple(manifests)


__all__ = [
    "CANONICAL_ROLE_REGISTRY_ID",
    "CANONICAL_ROLE_REGISTRY_PATH",
    "DatasetRole",
    "FreshRecoveryLabel",
    "RecoverySplitError",
    "RoleAllocationEntry",
    "RoleAllocationManifest",
    "RoleRegistry",
    "RoleRegistryHead",
    "TrainingManifest",
    "TrainingManifestEntry",
    "append_role_allocation",
    "build_training_manifest",
    "fresh_fact_evidence_sha256",
    "historical_evidence_sha256",
    "load_fresh_recovery_labels",
    "load_recovery_manifests",
    "load_role_allocation",
    "load_training_manifest",
    "make_role_allocation",
    "preflight_role_allocation_outputs",
    "preview_role_allocation_append",
    "recovery_source_dataset_sha256",
    "require_canonical_role_registry",
    "validate_dataset_roles",
    "validate_recovery_splits",
    "validate_role_allocation",
    "write_role_allocation",
    "write_run_registry_provenance",
    "write_training_manifest",
]
