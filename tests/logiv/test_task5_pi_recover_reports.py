from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.model import Fact, fact_universe_sha256
from scripts.report_task5_terminal_preflight import load_case, summarize_preflight


CASES = {
    "t05-r02": (36, 54804909),
    "t05-r03": (18, 3546047300),
    "t05-r04": (42, 1564298395),
}
HASH = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


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
    _write_json(
        case_dir / "run.json",
        {
            "capture_task5_terminal_preflight": True,
            "development_only": True,
            "episode_indices": [episode_idx],
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
            "true": [values[0][0] if values[0][1] == "TRUE" else values[1][0]],
        },
    )
    assessment_sha256 = "b" * 64
    _write_json(
        artifact / "terminal_deviation.json",
        {
            "assessment": {
                "assessment_sha256": assessment_sha256,
                "base_policy_steps": 340,
                "eligible": eligible,
                "reason": "ELIGIBLE" if eligible else "HOLDING_NOT_EXPLICITLY_TRUE",
            },
            "assessment_sha256": assessment_sha256,
            "base_policy_requests": 68,
            "base_policy_steps": 340,
            "base_success": False,
            "capability_sha256": HASH,
            "captured_at_utc": timestamp,
            "case_id": case_id,
            "certificate_hash": HASH,
            "graph_hash": HASH,
            "monitor_contract_sha256": HASH,
            "native_terminal_status": "EPISODE_FAIL",
            "recovery_actions": 0,
            "recovery_policy_requests": 0,
            "schema_version": 1,
            "status": "ELIGIBLE" if eligible else "DENIED",
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
            "graph_hash",
            "monitor_contract_sha256",
            "schema_version",
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
    terminal["assessment"].update(
        {
            "eligible": False,
            "reason": "STRICT_AUDITED_SNAPSHOT_REQUIRED",
        }
    )
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
    assert [case["assessment_sha256"] for case in summary["cases"]] == [
        "b" * 64
    ] * 3
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

    with pytest.raises(ValueError, match="anchor"):
        load_case(case_dir)


def test_terminal_preflight_report_requires_exact_case_identity(tmp_path: Path) -> None:
    case_dirs = [_write_case(tmp_path, case_id) for case_id in CASES]
    duplicate = case_dirs[:2] + [case_dirs[1]]

    with pytest.raises(ValueError, match="three frozen Task 5 failures"):
        summarize_preflight(duplicate)
