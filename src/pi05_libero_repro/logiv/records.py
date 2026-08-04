from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pi05_libero_repro.logiv.model import ContextEnvelope


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZERO_HASH = "0" * 64
METHOD_ARMS = frozenset(
    {
        "BASE",
        "SHADOW_LOGIV",
        "STAGE_ONLY",
        "GRAPH_WITHOUT_VAL",
        "VAL_WITHOUT_LOCALIZED_REPAIR",
        "FULL_LOGIV",
    }
)
GOAL_MODES = frozenset({"METADATA_ASSISTED", "GOAL_PREDICTION"})
TERMINAL_STATUSES = frozenset(
    {
        "EPISODE_SUCCESS",
        "EPISODE_FAIL",
        "EVALUATOR_ERROR",
        "EVALUATOR_TIMEOUT",
        "TERMINAL_NO_FURTHER_DISPATCH",
        "SAFE_STOPPED",
        "UNSAFE_TERMINAL",
    }
)
FAILURE_CAUSES = frozenset(
    {
        "EPISODE_FAIL",
        "EVALUATOR_ERROR",
        "EVALUATOR_TIMEOUT",
        "STATE_GROUNDING_FAILURE",
        "POST_STOP_GROUNDING_FAILURE",
        "SUBTASK_GROUNDING_ERROR",
        "RECOVERY_SUBTASK_GROUNDING_ERROR",
        "PLAN_GROUNDING_INCOMPLETE",
        "PLAN_SCHEMA_ERROR",
        "TRACE_ERROR",
        "VALIDATION_ERROR",
        "COMPILER_ERROR",
        "LIVE_INSTALL_CONFLICT",
        "BUDGET_EXHAUSTED",
        "NO_CERTIFIED_REPAIR_WITHIN_BUDGET",
        "EXECUTOR_REJECTED_NOT_ENQUEUED",
        "EXECUTOR_OUTCOME_UNKNOWN",
        "EXECUTOR_TIMEOUT",
        "FENCE_FAILURE",
        "SETTLING_TIMEOUT",
        "SAFETY_VETO",
        "WATCHDOG_EXPIRED",
        "SAFETY_CONTEXT_MISMATCH",
        "PRECONDITION_FAILURE",
        "EFFECT_FAILURE",
        "FINAL_GOAL_FAILURE",
        "EXCEPTION",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class EventRecord:
    sequence: int
    event_type: str
    context: Mapping[str, str | int | None]
    details: Mapping[str, Any]
    monotonic_ns: int
    previous_hash: str
    event_hash: str


class EventJournal:
    """Append-only, hash-chained per-episode event journal."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.exists():
            count, head = validate_event_chain(self.path)
            self._sequence = count
            self._head = head
        else:
            self._sequence = 0
            self._head = _ZERO_HASH

    def append(
        self,
        event_type: str,
        context: ContextEnvelope,
        details: Mapping[str, Any],
    ) -> EventRecord:
        if not event_type:
            raise ValueError("event_type must be nonempty")
        base = {
            "sequence": self._sequence,
            "event_type": event_type,
            "context": context.payload(),
            "details": dict(details),
            "monotonic_ns": time.monotonic_ns(),
            "previous_hash": self._head,
        }
        event_hash = hashlib.sha256(_canonical_json(base)).hexdigest()
        record = EventRecord(**base, event_hash=event_hash)
        _locked_append(self.path, _canonical_json(asdict(record)) + b"\n")
        self._sequence += 1
        self._head = event_hash
        return record

    @property
    def count(self) -> int:
        return self._sequence

    @property
    def head(self) -> str:
        return self._head


def validate_event_chain(path: Path | str) -> tuple[int, str]:
    path = Path(path)
    if not path.exists():
        return 0, _ZERO_HASH
    previous = _ZERO_HASH
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank event line: {path}:{line_number}")
            payload = json.loads(line)
            event_hash = payload.pop("event_hash", None)
            if payload.get("sequence") != count:
                raise ValueError(f"event sequence mismatch: {path}:{line_number}")
            if payload.get("previous_hash") != previous:
                raise ValueError(f"event previous hash mismatch: {path}:{line_number}")
            expected = hashlib.sha256(_canonical_json(payload)).hexdigest()
            if event_hash != expected:
                raise ValueError(f"event hash mismatch: {path}:{line_number}")
            previous = event_hash
            count += 1
    return count, previous


@dataclass(frozen=True)
class LogivEpisodeRecord:
    schema_version: int
    run_id: str
    checkpoint: str
    method_arm: str
    goal_mode: str
    deviation_mode: str
    task_id: int
    task_name: str
    episode_idx: int
    seed: int
    allocated: bool
    valid: bool
    success: bool
    terminal_status: str
    terminal_cause: str
    evaluator_status: str
    init_state_sha256: str
    first_frame_sha256: str
    prompt_version: str
    prompt_config_sha256: str
    proposal_config_sha256: str
    coverage_manifest_sha256: str
    domain_sha256: str
    initial_certificate_hash: str | None
    initial_graph_hash: str | None
    initial_graph_width: int | None
    final_graph_hash: str | None
    physical_attempts: int
    repair_rounds: int
    total_val_calls: int
    graph_installs: int
    steps: int
    inference_requests: int
    wall_seconds: float
    safety_permits: int
    halt_acknowledged: bool | None
    receipts_count: int
    event_count: int
    event_chain_head: str
    artifact_dir: str
    video_path: str | None
    exception: str | None
    oracle_grounding: bool
    development_only: bool
    committed_receipts: int = 0
    failed_receipts: int = 0
    unknown_receipts: int = 0
    precondition_gate_rejections: int = 0
    effect_gate_rejections: int = 0
    final_goal_gate_rejections: int = 0
    base_policy_requests: int = 0
    initial_proposal_requests: int = 0
    initial_proposal_status: str = "NOT_APPLICABLE"
    initial_proposal_reason_code: str | None = None
    shadow_vlm_requests: int = 0
    recovery_policy_requests: int = 0
    shadow_monitor_calls: int = 0
    shadow_monitor_errors: int = 0
    shadow_monitor_seconds: float = 0.0
    shadow_parity_valid: bool = True

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (
            self.run_id,
            self.method_arm,
            self.goal_mode,
            self.task_id,
            self.episode_idx,
        )

    @property
    def key_text(self) -> str:
        return "/".join(map(str, self.key))


def _locked_append(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short append: {written} != {len(encoded)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_episode_records(path: Path | str) -> list[LogivEpisodeRecord]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank episode line: {path}:{line_number}")
            records.append(LogivEpisodeRecord(**json.loads(line)))
    return records


def append_episode_record(path: Path | str, record: LogivEpisodeRecord) -> None:
    errors = validate_episode_records([record])
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(path)
    existing = load_episode_records(path)
    if record.key in {item.key for item in existing}:
        raise ValueError(f"duplicate LOGIV episode: {record.key_text}")
    _locked_append(path, _canonical_json(asdict(record)) + b"\n")


def _hash_valid(value: str | None, *, optional: bool = False) -> bool:
    return (optional and value is None) or (
        isinstance(value, str) and _SHA256.fullmatch(value) is not None
    )


def validate_episode_records(records: Sequence[LogivEpisodeRecord]) -> list[str]:
    errors: list[str] = []
    seen = set()
    for record in records:
        label = record.key_text
        if record.key in seen:
            errors.append(f"duplicate LOGIV episode: {label}")
        seen.add(record.key)
        if record.schema_version not in {1, 2, 3}:
            errors.append(f"unsupported schema version: {label}")
        if record.method_arm not in METHOD_ARMS:
            errors.append(f"unknown method arm: {label}")
        if record.goal_mode not in GOAL_MODES:
            errors.append(f"unknown goal mode: {label}")
        if record.terminal_status not in TERMINAL_STATUSES:
            errors.append(f"unknown terminal status: {label}")
        if not record.allocated:
            errors.append(f"episode must remain allocated: {label}")
        if record.success != (
            record.terminal_status == "EPISODE_SUCCESS"
            and record.evaluator_status == "EPISODE_SUCCESS"
        ):
            errors.append(f"success/evaluator mismatch: {label}")
        if not record.success and (
            record.terminal_cause not in FAILURE_CAUSES
            and not any(record.terminal_cause.startswith(prefix) for prefix in FAILURE_CAUSES)
        ):
            errors.append(f"unknown failure cause: {label}")
        for field_name in (
            "init_state_sha256",
            "first_frame_sha256",
            "prompt_config_sha256",
            "proposal_config_sha256",
            "coverage_manifest_sha256",
            "domain_sha256",
            "event_chain_head",
        ):
            if not _hash_valid(getattr(record, field_name)):
                errors.append(f"invalid {field_name}: {label}")
        for field_name in (
            "initial_certificate_hash",
            "initial_graph_hash",
            "final_graph_hash",
        ):
            if not _hash_valid(getattr(record, field_name), optional=True):
                errors.append(f"invalid {field_name}: {label}")
        counters = (
            record.physical_attempts,
            record.repair_rounds,
            record.total_val_calls,
            record.graph_installs,
            record.steps,
            record.inference_requests,
            record.safety_permits,
            record.receipts_count,
            record.event_count,
            record.committed_receipts,
            record.failed_receipts,
            record.unknown_receipts,
            record.precondition_gate_rejections,
            record.effect_gate_rejections,
            record.final_goal_gate_rejections,
            record.base_policy_requests,
            record.initial_proposal_requests,
            record.shadow_vlm_requests,
            record.recovery_policy_requests,
            record.shadow_monitor_calls,
            record.shadow_monitor_errors,
        )
        if any(value < 0 for value in counters):
            errors.append(f"negative counter: {label}")
        if not math.isfinite(record.wall_seconds) or record.wall_seconds < 0:
            errors.append(f"invalid wall_seconds: {label}")
        if (
            not math.isfinite(record.shadow_monitor_seconds)
            or record.shadow_monitor_seconds < 0
        ):
            errors.append(f"invalid shadow_monitor_seconds: {label}")
        if record.initial_graph_hash is None and record.initial_graph_width is not None:
            errors.append(f"width without graph: {label}")
        if record.schema_version >= 2 and (
            record.committed_receipts
            + record.failed_receipts
            + record.unknown_receipts
            != record.receipts_count
        ):
            errors.append(f"receipt taxonomy/count mismatch: {label}")
        if record.schema_version == 3:
            if record.inference_requests != record.base_policy_requests:
                errors.append(f"Base policy request/count mismatch: {label}")
            if record.shadow_vlm_requests != 0 or record.recovery_policy_requests != 0:
                errors.append(f"nonzero Phase 0 non-Base requests: {label}")
            reason = record.initial_proposal_reason_code
            stable_reason = (
                reason is not None
                and re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_.]*(?::[A-Za-z][A-Za-z0-9_.]*)?",
                    reason,
                )
                is not None
            )
            if record.method_arm == "SHADOW_LOGIV":
                valid_status = (
                    record.initial_proposal_status == "ACCEPTED"
                    and record.initial_proposal_requests == 1
                    and reason is None
                ) or (
                    record.initial_proposal_status == "REJECTED"
                    and record.initial_proposal_requests == 1
                    and stable_reason
                ) or (
                    record.initial_proposal_status == "NOT_ATTEMPTED"
                    and record.initial_proposal_requests == 0
                    and stable_reason
                )
                if not valid_status:
                    errors.append(f"invalid Shadow proposal accounting: {label}")
            elif (
                record.initial_proposal_status != "NOT_APPLICABLE"
                or record.initial_proposal_requests != 0
                or reason is not None
            ):
                errors.append(f"invalid non-Shadow proposal accounting: {label}")
            if record.method_arm != "SHADOW_LOGIV" and (
                record.shadow_monitor_calls != 0
                or record.shadow_monitor_errors != 0
                or record.shadow_monitor_seconds != 0
            ):
                errors.append(f"non-Shadow monitor accounting is nonzero: {label}")
            if record.method_arm == "BASE" and not record.shadow_parity_valid:
                errors.append(f"Base parity must remain valid: {label}")
        if record.method_arm == "FULL_LOGIV" and record.task_id == 8:
            if record.initial_graph_hash is not None and (
                record.initial_graph_width is None or record.initial_graph_width < 2
            ):
                errors.append("task 8 Full LOGIV graph must have width >= 2")
    return errors


def paired_task_stratified_bootstrap(
    full_records: Iterable[LogivEpisodeRecord],
    comparator_records: Iterable[LogivEpisodeRecord],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    full = {(item.task_id, item.episode_idx): item for item in full_records}
    comparator = {(item.task_id, item.episode_idx): item for item in comparator_records}
    if full.keys() != comparator.keys() or not full:
        raise ValueError("paired records must contain the same nonempty task/episode keys")
    by_task: dict[int, list[float]] = {}
    for key in sorted(full):
        left, right = full[key], comparator[key]
        if left.init_state_sha256 != right.init_state_sha256:
            raise ValueError(f"paired initial-state hash mismatch: {key}")
        if hasattr(left, "first_frame_sha256") and hasattr(
            right, "first_frame_sha256"
        ) and left.first_frame_sha256 != right.first_frame_sha256:
            raise ValueError(f"paired first-frame hash mismatch: {key}")
        for field_name in (
            "seed",
            "checkpoint",
            "prompt_version",
            "prompt_config_sha256",
        ):
            if hasattr(left, field_name) and hasattr(right, field_name) and (
                getattr(left, field_name) != getattr(right, field_name)
            ):
                raise ValueError(f"paired {field_name} mismatch: {key}")
        by_task.setdefault(key[0], []).append(float(left.success) - float(right.success))
    estimate = float(np.mean([np.mean(values) for values in by_task.values()]))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        task_differences = []
        for values in by_task.values():
            array = np.asarray(values, dtype=np.float64)
            indices = generator.integers(0, len(array), size=len(array))
            task_differences.append(float(array[indices].mean()))
        draws[draw] = float(np.mean(task_differences))
    return {
        "estimate": estimate,
        "percentile_95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "bootstrap_samples": samples,
        "tasks": len(by_task),
        "paired_episodes": len(full),
        "seed": seed,
    }
