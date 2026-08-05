#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from pi05_libero_repro.logiv.model import FactSnapshot, parse_pddl_fact
from pi05_libero_repro.logiv.task5_terminal_recovery import (
    TerminalAssessment,
    _assessment_sha256,
)
from scripts.eval_logiv_libero import _write_json


_FROZEN_CASES = {
    "t05-r02": {"episode_idx": 36, "seed": 54804909},
    "t05-r03": {"episode_idx": 18, "seed": 3546047300},
    "t05-r04": {"episode_idx": 42, "seed": 1564298395},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CAPTURE_AGE = timedelta(hours=24)
_MAX_FUTURE_SKEW = timedelta(minutes=5)
_ASSESSMENT_FIELDS = frozenset(TerminalAssessment.__dataclass_fields__)


@dataclass(frozen=True)
class PreflightCase:
    case_id: str
    eligible: bool
    reason: str
    captured_at_utc: str
    assessment_sha256: str
    base_policy_steps: int
    base_policy_requests: int
    snapshot_sha256: str | None
    graph_hash: str
    certificate_hash: str
    monitor_contract_sha256: str
    capability_sha256: str


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load preflight artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"preflight artifact is not an object: {path}")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    return None if value is None else _require_sha256(value, label)


def _optional_string(value: object, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"terminal assessment {label} is invalid")
    return value


def _authenticated_assessment(payload: object) -> TerminalAssessment:
    if not isinstance(payload, dict) or set(payload) != _ASSESSMENT_FIELDS:
        raise ValueError("terminal assessment schema is invalid")
    if type(payload["eligible"]) is not bool:
        raise ValueError("terminal assessment eligible is invalid")
    if not isinstance(payload["reason"], str) or not payload["reason"]:
        raise ValueError("terminal assessment reason is invalid")
    if (
        type(payload["option_action_cap"]) is not int
        or payload["option_action_cap"] < 0
    ):
        raise ValueError("terminal assessment action cap is invalid")
    if (
        type(payload["base_policy_steps"]) is not int
        or payload["base_policy_steps"] <= 0
    ):
        raise ValueError("terminal assessment Base steps are invalid")
    protected_payload = payload["protected_true_facts"]
    if (
        not isinstance(protected_payload, list)
        or any(not isinstance(item, str) for item in protected_payload)
        or protected_payload != sorted(protected_payload)
        or len(protected_payload) != len(set(protected_payload))
    ):
        raise ValueError("terminal assessment protected facts are invalid")
    try:
        protected = frozenset(parse_pddl_fact(item) for item in protected_payload)
        assessment = TerminalAssessment(
            eligible=payload["eligible"],
            reason=payload["reason"],
            event_id=_optional_sha256(payload["event_id"], "event ID"),
            event_type=_optional_string(payload["event_type"], "event type"),
            option_action_cap=payload["option_action_cap"],
            snapshot_sha256=_optional_sha256(
                payload["snapshot_sha256"], "snapshot hash"
            ),
            graph_hash=_optional_sha256(payload["graph_hash"], "graph hash"),
            certificate_hash=_optional_sha256(
                payload["certificate_hash"], "certificate hash"
            ),
            monitor_contract_sha256=_optional_sha256(
                payload["monitor_contract_sha256"], "monitor contract hash"
            ),
            protected_true_facts=protected,
            capability_sha256=_require_sha256(
                payload["capability_sha256"], "assessment capability hash"
            ),
            base_policy_steps=payload["base_policy_steps"],
            place_node_id=_optional_string(
                payload["place_node_id"], "place node ID"
            ),
            assessment_sha256=_require_sha256(
                payload["assessment_sha256"], "assessment digest"
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"terminal assessment is malformed: {error}") from error
    if assessment.assessment_sha256 != _assessment_sha256(assessment):
        raise ValueError("terminal assessment digest mismatch")
    if assessment.eligible:
        if (
            assessment.reason != "ELIGIBLE"
            or assessment.event_id is None
            or assessment.event_type is None
            or assessment.option_action_cap <= 0
            or assessment.snapshot_sha256 is None
            or assessment.place_node_id is None
        ):
            raise ValueError("terminal assessment eligible fields are inconsistent")
    elif (
        assessment.reason == "ELIGIBLE"
        or assessment.event_id is not None
        or assessment.event_type is not None
        or assessment.option_action_cap != 0
        or assessment.protected_true_facts
        or assessment.place_node_id is not None
    ):
        raise ValueError("terminal assessment denial fields are inconsistent")
    return assessment


def _parse_fresh_timestamp(value: object, *, now: datetime) -> str:
    if not isinstance(value, str):
        raise ValueError("preflight capture timestamp is missing")
    try:
        captured = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("preflight capture timestamp is invalid") from error
    if captured.tzinfo is None:
        raise ValueError("preflight capture timestamp must be timezone-aware")
    age = now - captured.astimezone(timezone.utc)
    if age > _MAX_CAPTURE_AGE or age < -_MAX_FUTURE_SKEW:
        raise ValueError("preflight capture timestamp is not fresh")
    return value


def _strict_snapshot(payload: dict[str, Any]) -> FactSnapshot:
    if payload.get("schema_version") != 1 or payload.get("status") != "STRICT_AUDITED":
        raise ValueError("preflight requires a strict audited terminal snapshot")
    fact_fields = ("fact_universe", "true", "false")
    if any(
        not isinstance(payload.get(field), list)
        or payload[field] != sorted(payload[field])
        or len(payload[field]) != len(set(payload[field]))
        for field in fact_fields
    ):
        raise ValueError("strict terminal facts are not sorted unique PDDL strings")
    try:
        universe = frozenset(parse_pddl_fact(item) for item in payload["fact_universe"])
        true_facts = frozenset(parse_pddl_fact(item) for item in payload["true"])
        false_facts = frozenset(parse_pddl_fact(item) for item in payload["false"])
        return FactSnapshot(
            epoch_id=payload["epoch_id"],
            true_facts=true_facts,
            false_facts=false_facts,
            evidence_hash=payload["evidence_hash"],
            fact_universe=universe,
            fact_universe_version=payload["fact_universe_version"],
            fact_universe_sha256=payload["fact_universe_sha256"],
            evidence_payload_json=payload["evidence_payload_json"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"strict terminal snapshot audit failed: {error}") from error


def load_case(case_dir: Path) -> PreflightCase:
    case_dir = Path(case_dir)
    case_id = case_dir.name
    frozen = _FROZEN_CASES.get(case_id)
    if frozen is None:
        raise ValueError(f"unknown Task 5 preflight case: {case_id}")
    run = _load_object(case_dir / "run.json")
    expected_run_id = f"task5-terminal-preflight-{case_id}"
    expected_run = {
        "capture_task5_terminal_preflight": True,
        "development_only": True,
        "episode_indices": [frozen["episode_idx"]],
        "goal_mode": "METADATA_ASSISTED",
        "method_arm": "SHADOW_LOGIV",
        "no_video": True,
        "run_id": expected_run_id,
        "seed": frozen["seed"],
        "task_ids": [5],
    }
    if any(run.get(key) != value for key, value in expected_run.items()):
        raise ValueError(f"{case_id} does not match its frozen run configuration")

    try:
        episode_lines = [
            line
            for line in (case_dir / "episodes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise ValueError(f"cannot load {case_id} episode record: {error}") from error
    if len(episode_lines) != 1:
        raise ValueError(f"{case_id} must contain exactly one episode record")
    try:
        record = json.loads(episode_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"{case_id} episode record is invalid JSON") from error
    expected_record = {
        "development_only": True,
        "episode_idx": frozen["episode_idx"],
        "evaluator_status": "EPISODE_FAIL",
        "method_arm": "SHADOW_LOGIV",
        "run_id": expected_run_id,
        "success": False,
        "task_id": 5,
        "terminal_status": "EPISODE_FAIL",
        "valid": True,
        "video_path": None,
    }
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected_record.items()
    ):
        raise ValueError(f"{case_id} is not a valid native Base failure")
    artifact_relative = record.get("artifact_dir")
    if not isinstance(artifact_relative, str):
        raise ValueError(f"{case_id} episode artifact directory is missing")
    artifact_dir = (case_dir / artifact_relative).resolve()
    if not artifact_dir.is_relative_to(case_dir.resolve()):
        raise ValueError(f"{case_id} artifact directory escapes the run")

    base = _load_object(artifact_dir / "base_execution.json")
    compute = _load_object(artifact_dir / "compute_accounting.json")
    snapshot_payload = _load_object(artifact_dir / "current_snapshot.json")
    terminal = _load_object(artifact_dir / "terminal_deviation.json")
    now = datetime.now(timezone.utc)
    timestamp = _parse_fresh_timestamp(terminal.get("captured_at_utc"), now=now)
    if _parse_fresh_timestamp(snapshot_payload.get("captured_at_utc"), now=now) != timestamp:
        raise ValueError(f"{case_id} terminal artifact timestamps disagree")
    expected_episode_id = (
        f"{expected_run_id}/SHADOW_LOGIV/METADATA_ASSISTED/"
        f"task-5/episode-{frozen['episode_idx']}"
    )
    expected_identity = {
        "task_id": 5,
        "episode_idx": frozen["episode_idx"],
        "episode_id": expected_episode_id,
    }
    if any(
        payload.get(field) != value
        for payload in (snapshot_payload, terminal)
        for field, value in expected_identity.items()
    ):
        raise ValueError(f"{case_id} terminal artifact identity mismatch")
    artifact_case_ids = {
        snapshot_payload.get("case_id"),
        terminal.get("case_id"),
    }
    if len(artifact_case_ids) != 1 or not artifact_case_ids <= {case_id, "outputs"}:
        raise ValueError(f"{case_id} terminal artifact identity mismatch")
    grounding_error = snapshot_payload.get("status") == "GROUNDING_ERROR"
    if grounding_error:
        if (
            snapshot_payload.get("schema_version") != 1
            or terminal.get("status") != "GROUNDING_ERROR"
            or terminal.get("reason") != "GROUNDING_ERROR"
            or not isinstance(snapshot_payload.get("grounding_error"), str)
            or snapshot_payload.get("grounding_error")
            != terminal.get("grounding_error")
        ):
            raise ValueError(f"{case_id} grounding-error denial is malformed")
        snapshot = None
    else:
        snapshot = _strict_snapshot(snapshot_payload)

    if (
        base.get("post_settling_success") is not False
        or terminal.get("base_success") is not False
        or terminal.get("native_terminal_status") != "EPISODE_FAIL"
    ):
        raise ValueError(f"{case_id} did not preserve native Base failure")
    observed_steps = terminal.get("base_policy_steps")
    observed_requests = terminal.get("base_policy_requests")
    if (
        type(observed_steps) is not int
        or observed_steps <= 0
        or type(observed_requests) is not int
        or observed_requests <= 0
        or any(
            value != observed_steps
            for value in (
                record.get("steps"),
                base.get("steps"),
            )
        )
        or any(
            value != observed_requests
            for value in (
                record.get("inference_requests"),
                record.get("base_policy_requests"),
                base.get("inference_requests"),
                base.get("base_policy_requests"),
                compute.get("base_policy_requests"),
            )
        )
    ):
        raise ValueError(f"{case_id} Base accounting anchor mismatch")
    if (
        record.get("recovery_policy_requests") != 0
        or compute.get("recovery_policy_requests") != 0
        or terminal.get("recovery_policy_requests") != 0
        or terminal.get("recovery_actions") != 0
    ):
        raise ValueError(f"{case_id} contains a recovery action or request")

    assessment = _authenticated_assessment(terminal.get("assessment"))
    assessment_sha256 = _require_sha256(
        terminal.get("assessment_sha256"), "assessment anchor"
    )
    if (
        assessment.assessment_sha256 != assessment_sha256
        or assessment.base_policy_steps != observed_steps
    ):
        raise ValueError(f"{case_id} assessment anchor mismatch")
    hashes = {}
    for field in (
        "graph_hash",
        "certificate_hash",
        "monitor_contract_sha256",
        "capability_sha256",
    ):
        value = _require_sha256(terminal.get(field), field)
        if snapshot_payload.get(field) != value:
            raise ValueError(f"{case_id} terminal {field} mismatch")
        hashes[field] = value
    snapshot_sha256 = None if snapshot is None else snapshot.evidence_hash
    if (
        assessment.snapshot_sha256 != snapshot_sha256
        or assessment.graph_hash != hashes["graph_hash"]
        or assessment.certificate_hash != hashes["certificate_hash"]
        or assessment.monitor_contract_sha256
        != hashes["monitor_contract_sha256"]
        or assessment.capability_sha256 != hashes["capability_sha256"]
    ):
        raise ValueError(f"{case_id} assessment evidence anchor mismatch")
    if (
        snapshot is not None
        and not assessment.protected_true_facts <= snapshot.true_facts
    ):
        raise ValueError(f"{case_id} assessment protected facts are not observed true")
    if grounding_error:
        if assessment.eligible:
            raise ValueError(f"{case_id} grounding-error assessment is eligible")
    elif (
        terminal.get("status")
        != ("ELIGIBLE" if assessment.eligible else "DENIED")
        or terminal.get("reason") != assessment.reason
        or terminal.get("grounding_error") is not None
    ):
        raise ValueError(f"{case_id} terminal assessment semantics mismatch")

    return PreflightCase(
        case_id=case_id,
        eligible=assessment.eligible,
        reason="GROUNDING_ERROR" if grounding_error else assessment.reason,
        captured_at_utc=timestamp,
        assessment_sha256=assessment_sha256,
        base_policy_steps=observed_steps,
        base_policy_requests=observed_requests,
        snapshot_sha256=snapshot_sha256,
        **hashes,
    )


def summarize_preflight(case_dirs: Sequence[Path]) -> dict[str, object]:
    rows = tuple(load_case(path) for path in case_dirs)
    if {row.case_id for row in rows} != set(_FROZEN_CASES) or len(rows) != 3:
        raise ValueError("preflight requires the three frozen Task 5 failures")
    eligible = sum(row.eligible for row in rows)
    return {
        "schema_version": 1,
        "case_ids": [row.case_id for row in rows],
        "eligible_held_roots": eligible,
        "required_held_roots": 2,
        "go": eligible >= 2,
        "cases": [asdict(row) for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate Task 5 recovery on three strict terminal roots"
    )
    parser.add_argument("case_dirs", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = summarize_preflight(args.case_dirs)
    except ValueError as error:
        parser.error(str(error))
    _write_json(args.output, summary)
    return 0 if summary["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
