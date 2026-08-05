from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.model import (
    Fact,
    fact_universe_sha256,
    parse_pddl_fact,
)
from pi05_libero_repro.logiv.task5_terminal_recovery import (
    TerminalAssessment,
    _assessment_sha256,
)
from scripts.report_task5_terminal_preflight import load_case, summarize_preflight


CASES = {
    "t05-r02": (36, 54804909),
    "t05-r03": (18, 3546047300),
    "t05-r04": (42, 1564298395),
}
HASH = "a" * 64
ASSESSMENT_FIELDS = frozenset(TerminalAssessment.__dataclass_fields__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _assessment_payload(
    *,
    eligible: bool,
    snapshot_sha256: str | None,
    reason: str | None = None,
) -> dict[str, object]:
    provisional = TerminalAssessment(
        eligible=eligible,
        reason=reason or ("ELIGIBLE" if eligible else "PLACE_NODE_NOT_ACTIVE"),
        event_id="e" * 64 if eligible else None,
        event_type="TERMINAL_GOAL_FAILURE" if eligible else None,
        option_action_cap=8 if eligible else 0,
        snapshot_sha256=snapshot_sha256,
        graph_hash=HASH,
        certificate_hash=HASH,
        monitor_contract_sha256=HASH,
        protected_true_facts=(
            frozenset({Fact("holding", ("black_book_1",))})
            if eligible
            else frozenset()
        ),
        capability_sha256=HASH,
        base_policy_steps=340,
        place_node_id="place-node" if eligible else None,
        assessment_sha256="",
    )
    assessment = replace(
        provisional,
        assessment_sha256=_assessment_sha256(provisional),
    )
    return {
        "eligible": assessment.eligible,
        "reason": assessment.reason,
        "event_id": assessment.event_id,
        "event_type": assessment.event_type,
        "option_action_cap": assessment.option_action_cap,
        "snapshot_sha256": assessment.snapshot_sha256,
        "graph_hash": assessment.graph_hash,
        "certificate_hash": assessment.certificate_hash,
        "monitor_contract_sha256": assessment.monitor_contract_sha256,
        "protected_true_facts": sorted(
            fact.pddl() for fact in assessment.protected_true_facts
        ),
        "capability_sha256": assessment.capability_sha256,
        "base_policy_steps": assessment.base_policy_steps,
        "place_node_id": assessment.place_node_id,
        "assessment_sha256": assessment.assessment_sha256,
    }


def _resign_assessment(payload: dict[str, object]) -> None:
    assessment = TerminalAssessment(
        eligible=payload["eligible"],
        reason=payload["reason"],
        event_id=payload["event_id"],
        event_type=payload["event_type"],
        option_action_cap=payload["option_action_cap"],
        snapshot_sha256=payload["snapshot_sha256"],
        graph_hash=payload["graph_hash"],
        certificate_hash=payload["certificate_hash"],
        monitor_contract_sha256=payload["monitor_contract_sha256"],
        protected_true_facts=frozenset(
            parse_pddl_fact(item) for item in payload["protected_true_facts"]
        ),
        capability_sha256=payload["capability_sha256"],
        base_policy_steps=payload["base_policy_steps"],
        place_node_id=payload["place_node_id"],
        assessment_sha256="",
    )
    payload["assessment_sha256"] = _assessment_sha256(assessment)


def _write_case(root: Path, case_id: str, *, eligible: bool = True) -> Path:
    episode_idx, seed = CASES[case_id]
    case_dir = root / case_id
    artifact = case_dir / "artifacts/task_05" / f"episode_{episode_idx:03d}"
    timestamp = datetime.now(timezone.utc).isoformat()
    universe = frozenset(
        {
            Fact("holding", ("black_book_1",)),
            Fact(
                "at",
                ("black_book_1", "desk_caddy_1_back_contain_region"),
            ),
        }
    )
    version = "task5-terminal-test-v1"
    values = [
        [fact.pddl(), "TRUE" if fact.predicate == "holding" else "FALSE"]
        for fact in sorted(universe)
    ]
    evidence_payload_json = json.dumps(
        {
            "dominance_overrides": [],
            "epoch_id": 7,
            "observation_hash": HASH,
            "values": values,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    evidence_hash = hashlib.sha256(evidence_payload_json.encode()).hexdigest()
    run_id = f"task5-terminal-preflight-{case_id}"
    episode_id = (
        f"{run_id}/SHADOW_LOGIV/METADATA_ASSISTED/"
        f"task-5/episode-{episode_idx}"
    )
    _write_json(
        case_dir / "run.json",
        {
            "capture_task5_terminal_preflight": True,
            "development_only": True,
            "episode_indices": [episode_idx],
            "goal_mode": "METADATA_ASSISTED",
            "method_arm": "SHADOW_LOGIV",
            "no_video": True,
            "run_id": run_id,
            "seed": seed,
            "task_ids": [5],
        },
    )
    record = {
        "artifact_dir": str(artifact.relative_to(case_dir)),
        "base_policy_requests": 68,
        "development_only": True,
        "episode_idx": episode_idx,
        "evaluator_status": "EPISODE_FAIL",
        "inference_requests": 68,
        "method_arm": "SHADOW_LOGIV",
        "recovery_policy_requests": 0,
        "run_id": run_id,
        "steps": 340,
        "success": False,
        "task_id": 5,
        "terminal_status": "EPISODE_FAIL",
        "valid": True,
        "video_path": None,
    }
    (case_dir / "episodes.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (case_dir / "episodes.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(
        artifact / "base_execution.json",
        {
            "base_policy_requests": 68,
            "inference_requests": 68,
            "post_settling_success": False,
            "steps": 340,
        },
    )
    _write_json(
        artifact / "compute_accounting.json",
        {"base_policy_requests": 68, "recovery_policy_requests": 0},
    )
    _write_json(
        artifact / "current_snapshot.json",
        {
            "capability_sha256": HASH,
            "captured_at_utc": timestamp,
            "case_id": case_id,
            "certificate_hash": HASH,
            "episode_id": episode_id,
            "episode_idx": episode_idx,
            "epoch_id": 7,
            "evidence_hash": evidence_hash,
            "evidence_payload_json": evidence_payload_json,
            "fact_universe": [fact.pddl() for fact in sorted(universe)],
            "fact_universe_sha256": fact_universe_sha256(version, universe),
            "fact_universe_version": version,
            "false": [values[0][0] if values[0][1] == "FALSE" else values[1][0]],
            "graph_hash": HASH,
            "monitor_contract_sha256": HASH,
            "schema_version": 1,
            "status": "STRICT_AUDITED",
            "task_id": 5,
            "true": [values[0][0] if values[0][1] == "TRUE" else values[1][0]],
        },
    )
    assessment = _assessment_payload(
        eligible=eligible,
        snapshot_sha256=evidence_hash,
    )
    assessment_sha256 = assessment["assessment_sha256"]
    _write_json(
        artifact / "terminal_deviation.json",
        {
            "assessment": assessment,
            "assessment_sha256": assessment_sha256,
            "base_policy_requests": 68,
            "base_policy_steps": 340,
            "base_success": False,
            "capability_sha256": HASH,
            "captured_at_utc": timestamp,
            "case_id": case_id,
            "certificate_hash": HASH,
            "episode_id": episode_id,
            "episode_idx": episode_idx,
            "graph_hash": HASH,
            "monitor_contract_sha256": HASH,
            "native_terminal_status": "EPISODE_FAIL",
            "recovery_actions": 0,
            "recovery_policy_requests": 0,
            "reason": assessment["reason"],
            "schema_version": 1,
            "status": "ELIGIBLE" if eligible else "DENIED",
            "task_id": 5,
        },
    )
    return case_dir


def _replace_with_grounding_error(case_dir: Path) -> None:
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    snapshot_path = artifact / "current_snapshot.json"
    terminal_path = artifact / "terminal_deviation.json"
    snapshot = json.loads(snapshot_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    snapshot = {
        key: value
        for key, value in snapshot.items()
        if key
        in {
            "capability_sha256",
            "captured_at_utc",
            "case_id",
            "certificate_hash",
            "episode_id",
            "episode_idx",
            "graph_hash",
            "monitor_contract_sha256",
            "schema_version",
            "task_id",
        }
    }
    snapshot.update(
        {
            "status": "GROUNDING_ERROR",
            "grounding_error": "GroundingError: exactly-one violation",
        }
    )
    terminal.update(
        {
            "status": "GROUNDING_ERROR",
            "reason": "GROUNDING_ERROR",
            "grounding_error": "GroundingError: exactly-one violation",
        }
    )
    terminal["assessment"] = _assessment_payload(
        eligible=False,
        snapshot_sha256=None,
        reason="STRICT_AUDITED_SNAPSHOT_REQUIRED",
    )
    terminal["assessment_sha256"] = terminal["assessment"]["assessment_sha256"]
    _write_json(snapshot_path, snapshot)
    _write_json(terminal_path, terminal)


def test_terminal_preflight_report_gates_exact_three_frozen_failures(tmp_path: Path) -> None:
    case_dirs = [_write_case(tmp_path, case_id) for case_id in CASES]

    summary = summarize_preflight(case_dirs)

    assert summary["schema_version"] == 1
    assert summary["case_ids"] == list(CASES)
    assert summary["eligible_held_roots"] == 3
    assert summary["required_held_roots"] == 2
    assert summary["go"] is True
    assert all(case["assessment_sha256"] for case in summary["cases"])
    assert [case["base_policy_steps"] for case in summary["cases"]] == [340] * 3


def test_terminal_preflight_report_counts_fail_closed_grounding_error_as_denial(
    tmp_path: Path,
) -> None:
    case_dirs = [_write_case(tmp_path, case_id) for case_id in CASES]
    _replace_with_grounding_error(case_dirs[0])

    summary = summarize_preflight(case_dirs)

    assert summary["eligible_held_roots"] == 2
    assert summary["go"] is True
    assert summary["cases"][0]["reason"] == "GROUNDING_ERROR"
    assert summary["cases"][0]["snapshot_sha256"] is None


def test_terminal_preflight_report_uses_run_id_not_container_output_label(
    tmp_path: Path,
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    for name in ("current_snapshot.json", "terminal_deviation.json"):
        path = artifact / name
        payload = json.loads(path.read_text())
        payload["case_id"] = "outputs"
        _write_json(path, payload)

    row = load_case(case_dir)

    assert row.case_id == "t05-r02"


def test_terminal_preflight_report_blocks_when_fewer_than_two_are_eligible(
    tmp_path: Path,
) -> None:
    case_dirs = [
        _write_case(tmp_path, case_id, eligible=index == 0)
        for index, case_id in enumerate(CASES)
    ]

    summary = summarize_preflight(case_dirs)

    assert summary["eligible_held_roots"] == 1
    assert summary["go"] is False


@pytest.mark.parametrize("field", ["assessment_sha256", "base_policy_steps"])
def test_terminal_preflight_report_rejects_replaced_assessment_anchors(
    tmp_path: Path, field: str
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload["assessment"][field] = "c" * 64 if field.endswith("sha256") else 339
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


def test_terminal_preflight_report_requires_exact_case_identity(tmp_path: Path) -> None:
    case_dirs = [_write_case(tmp_path, case_id) for case_id in CASES]
    duplicate = case_dirs[:2] + [case_dirs[1]]

    with pytest.raises(ValueError, match="three frozen Task 5 failures"):
        summarize_preflight(duplicate)


def test_terminal_preflight_report_rejects_false_go_digest_forgery(
    tmp_path: Path,
) -> None:
    case_dirs = [
        _write_case(tmp_path, case_id, eligible=False) for case_id in CASES
    ]
    for case_dir in case_dirs[:2]:
        artifact = next((case_dir / "artifacts/task_05").iterdir())
        path = artifact / "terminal_deviation.json"
        payload = json.loads(path.read_text())
        payload["assessment"].update(
            {
                "eligible": True,
                "reason": "ELIGIBLE",
                "assessment_sha256": "f" * 64,
            }
        )
        payload.update(
            {
                "status": "ELIGIBLE",
                "reason": "ELIGIBLE",
                "assessment_sha256": "f" * 64,
            }
        )
        _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        summarize_preflight(case_dirs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("eligible", False),
        ("reason", "OTHER"),
        ("event_id", "f" * 64),
        ("event_type", "OTHER_EVENT"),
        ("option_action_cap", 7),
        ("snapshot_sha256", "c" * 64),
        ("graph_hash", "c" * 64),
        ("certificate_hash", "d" * 64),
        ("monitor_contract_sha256", "e" * 64),
        ("protected_true_facts", []),
        ("capability_sha256", "f" * 64),
        ("base_policy_steps", 339),
        ("place_node_id", "other-node"),
        ("assessment_sha256", "c" * 64),
    ],
)
def test_terminal_preflight_report_authenticates_every_assessment_field(
    tmp_path: Path, field: str, replacement: object
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload["assessment"][field] = replacement
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("snapshot_sha256", "c" * 64),
        ("graph_hash", "c" * 64),
        ("certificate_hash", "d" * 64),
        ("monitor_contract_sha256", "e" * 64),
        ("capability_sha256", "f" * 64),
        ("base_policy_steps", 339),
    ],
)
def test_terminal_preflight_report_binds_resigned_assessment_to_observation(
    tmp_path: Path, field: str, replacement: object
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload["assessment"][field] = replacement
    _resign_assessment(payload["assessment"])
    payload["assessment_sha256"] = payload["assessment"]["assessment_sha256"]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


def test_terminal_preflight_report_rejects_resigned_unobserved_protected_fact(
    tmp_path: Path,
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload["assessment"]["protected_true_facts"] = [
        "(at black_book_1 desk_caddy_1_back_contain_region)"
    ]
    _resign_assessment(payload["assessment"])
    payload["assessment_sha256"] = payload["assessment"]["assessment_sha256"]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="protected facts"):
        load_case(case_dir)


@pytest.mark.parametrize("field", sorted(ASSESSMENT_FIELDS))
def test_terminal_preflight_report_requires_complete_assessment_schema(
    tmp_path: Path, field: str
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    del payload["assessment"][field]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("eligible", 1),
        ("reason", None),
        ("event_id", 7),
        ("event_type", 7),
        ("option_action_cap", True),
        ("snapshot_sha256", 7),
        ("graph_hash", 7),
        ("certificate_hash", 7),
        ("monitor_contract_sha256", 7),
        ("protected_true_facts", "(holding black_book_1)"),
        ("protected_true_facts", [{}]),
        ("capability_sha256", 7),
        ("base_policy_steps", True),
        ("place_node_id", 7),
        ("assessment_sha256", 7),
    ],
)
def test_terminal_preflight_report_requires_typed_assessment_schema(
    tmp_path: Path, field: str, replacement: object
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload["assessment"][field] = replacement
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("status", "DENIED"), ("reason", "OTHER")],
)
def test_terminal_preflight_report_binds_terminal_assessment_semantics(
    tmp_path: Path, field: str, replacement: object
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    path = artifact / "terminal_deviation.json"
    payload = json.loads(path.read_text())
    payload[field] = replacement
    _write_json(path, payload)

    with pytest.raises(ValueError, match="assessment"):
        load_case(case_dir)


@pytest.mark.parametrize(
    "artifact_name", ["current_snapshot.json", "terminal_deviation.json"]
)
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("task_id", 4),
        ("episode_idx", 35),
        ("episode_id", "task5-terminal-preflight-t05-r02/other"),
    ],
)
def test_terminal_preflight_report_binds_each_artifact_identity_field(
    tmp_path: Path,
    artifact_name: str,
    field: str,
    replacement: object,
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = next((case_dir / "artifacts/task_05").iterdir())
    for name in ("current_snapshot.json", "terminal_deviation.json"):
        path = artifact / name
        payload = json.loads(path.read_text())
        payload["case_id"] = "outputs"
        if name == artifact_name:
            payload[field] = replacement
        _write_json(path, payload)

    with pytest.raises(ValueError, match="identity"):
        load_case(case_dir)


def test_terminal_preflight_report_rejects_swapped_outputs_artifacts(
    tmp_path: Path,
) -> None:
    target = _write_case(tmp_path, "t05-r02")
    source = _write_case(tmp_path, "t05-r03")
    target_artifact = next((target / "artifacts/task_05").iterdir())
    source_artifact = next((source / "artifacts/task_05").iterdir())
    for name in ("current_snapshot.json", "terminal_deviation.json"):
        payload = json.loads((source_artifact / name).read_text())
        payload["case_id"] = "outputs"
        _write_json(target_artifact / name, payload)

    with pytest.raises(ValueError, match="identity"):
        load_case(target)
