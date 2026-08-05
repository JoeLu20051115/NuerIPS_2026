from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

import scripts.report_task5_terminal_preflight as preflight_report
from pi05_libero_repro.logiv.model import (
    Fact,
    fact_universe_sha256,
    parse_pddl_fact,
)
from pi05_libero_repro.logiv.task5_terminal_recovery import (
    TerminalAssessment,
    _assessment_sha256,
    load_task5_recovery_capability,
)
from scripts.report_task5_terminal_preflight import (
    _authenticated_assessment,
    _replay_eligible_assessment,
    _strict_snapshot,
    load_case,
    summarize_preflight,
)


CASES = {
    "t05-r02": (36, 54804909),
    "t05-r03": (18, 3546047300),
    "t05-r04": (42, 1564298395),
}
HASH = "a" * 64
ASSESSMENT_FIELDS = frozenset(TerminalAssessment.__dataclass_fields__)
CAPABILITY = load_task5_recovery_capability(
    Path(__file__).parents[2] / "configs/logiv/task5-terminal-pi-recover-v1.json"
)
CAPABILITY_HASH = CAPABILITY.capability_sha256
PLACE_NODE_ID = "place-node"
WRONG_LOCATION = Fact("at", ("black_book_1", "study_table_recovery_surface"))


@pytest.fixture(autouse=True)
def _isolate_synthetic_capture_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight_report,
        "_FROZEN_CASES",
        {
            case_id: dict(anchors)
            for case_id, anchors in preflight_report._FROZEN_CASES.items()
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _assessment_payload(
    *,
    eligible: bool,
    snapshot_sha256: str | None,
    episode_id: str,
    reason: str | None = None,
) -> dict[str, object]:
    event_payload = (
        f"{episode_id}:{snapshot_sha256}:{HASH}:{HASH}:{HASH}:"
        f"{CAPABILITY.event_type}"
    )
    provisional = TerminalAssessment(
        eligible=eligible,
        reason=reason or ("ELIGIBLE" if eligible else "CERTIFICATE_NOT_CURRENT"),
        event_id=(
            hashlib.sha256(
                b"LOGIV_TERMINAL_DEVIATION_V1\0"
                + event_payload.encode("utf-8")
            ).hexdigest()
            if eligible
            else None
        ),
        event_type=CAPABILITY.event_type if eligible else None,
        option_action_cap=(
            min(
                CAPABILITY.max_recovery_actions,
                CAPABILITY.max_combined_actions - 340,
            )
            if eligible
            else 0
        ),
        snapshot_sha256=snapshot_sha256,
        graph_hash=HASH,
        certificate_hash=HASH,
        monitor_contract_sha256=HASH,
        protected_true_facts=(
            CAPABILITY.protected_invariants
            if eligible
            else frozenset()
        ),
        capability_sha256=CAPABILITY_HASH,
        base_policy_steps=340,
        place_node_id=PLACE_NODE_ID if eligible else None,
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
            CAPABILITY.holding_fact,
            CAPABILITY.target_fact,
            WRONG_LOCATION,
            *CAPABILITY.protected_invariants,
        }
    )
    version = "task5-terminal-test-v1"
    true_facts = frozenset(
        {CAPABILITY.holding_fact, *CAPABILITY.protected_invariants}
    )
    values = [
        [fact.pddl(), "TRUE" if fact in true_facts else "FALSE"]
        for fact in sorted(universe, key=lambda item: item.pddl())
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
            "capability_sha256": CAPABILITY_HASH,
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
            "false": sorted(
                fact.pddl() for fact in universe if fact not in true_facts
            ),
            "graph_hash": HASH,
            "monitor_contract_sha256": HASH,
            "schema_version": 1,
            "status": "STRICT_AUDITED",
            "task_id": 5,
            "true": sorted(fact.pddl() for fact in true_facts),
        },
    )
    certificate_state = "CURRENT" if eligible else "STALE"
    place_status = "ACTIVE" if eligible else "BLOCKED"
    graph_version = f"graph-{HASH[:16]}"
    _write_json(
        artifact / "graph.json",
        {
            "action_layer_width": 1,
            "canonical_agenda": [PLACE_NODE_ID],
            "certificate_hash": HASH,
            "edges": [],
            "graph_hash": HASH,
            "graph_version": graph_version,
            "nodes": [
                {
                    "action": CAPABILITY.action,
                    "canonical_rank": 0,
                    "kind": "ACTION",
                    "lineage_root": "root",
                    "node_id": PLACE_NODE_ID,
                }
            ],
            "source_epoch": 0,
            "state_trace": [
                {
                    "certificate_state": certificate_state,
                    "graph_hash": HASH,
                    "graph_version": graph_version,
                    "nodes": [
                        {"node_id": PLACE_NODE_ID, "status": place_status}
                    ],
                }
            ],
        },
    )
    assessment = _assessment_payload(
        eligible=eligible,
        snapshot_sha256=evidence_hash,
        episode_id=episode_id,
    )
    assessment_sha256 = assessment["assessment_sha256"]
    preflight_report._FROZEN_CASES[case_id].update(
        {
            "assessment_sha256": assessment_sha256,
            "base_policy_steps": 340,
            "base_policy_requests": 68,
        }
    )
    _write_json(
        artifact / "terminal_deviation.json",
        {
            "assessment": assessment,
            "assessment_sha256": assessment_sha256,
            "base_policy_requests": 68,
            "base_policy_steps": 340,
            "base_success": False,
            "capability_sha256": CAPABILITY_HASH,
            "captured_at_utc": timestamp,
            "case_id": case_id,
            "certificate_hash": HASH,
            "certificate_state": certificate_state,
            "episode_id": episode_id,
            "episode_idx": episode_idx,
            "graph_hash": HASH,
            "monitor_contract_sha256": HASH,
            "native_terminal_status": "EPISODE_FAIL",
            "place_node_action": CAPABILITY.action,
            "place_node_status": place_status,
            "recovery_actions": 0,
            "recovery_policy_requests": 0,
            "reason": assessment["reason"],
            "schema_version": 1,
            "status": "ELIGIBLE" if eligible else "DENIED",
            "task_id": 5,
        },
    )
    _trust_current_artifacts(case_dir)
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
        episode_id=terminal["episode_id"],
        reason="STRICT_AUDITED_SNAPSHOT_REQUIRED",
    )
    terminal["assessment_sha256"] = terminal["assessment"]["assessment_sha256"]
    preflight_report._FROZEN_CASES[case_dir.name]["assessment_sha256"] = terminal[
        "assessment_sha256"
    ]
    _write_json(snapshot_path, snapshot)
    _write_json(terminal_path, terminal)
    _trust_current_artifacts(case_dir)


def _artifact_dir(case_dir: Path) -> Path:
    return next((case_dir / "artifacts/task_05").iterdir())


def _trust_current_artifacts(case_dir: Path) -> None:
    artifact = _artifact_dir(case_dir)
    terminal = json.loads((artifact / "terminal_deviation.json").read_text())
    preflight_report._FROZEN_CASES[case_dir.name].update(
        {
            "captured_at_utc": terminal["captured_at_utc"],
            "graph_artifact_sha256": hashlib.sha256(
                (artifact / "graph.json").read_bytes()
            ).hexdigest(),
            "snapshot_artifact_sha256": hashlib.sha256(
                (artifact / "current_snapshot.json").read_bytes()
            ).hexdigest(),
            "terminal_artifact_sha256": hashlib.sha256(
                (artifact / "terminal_deviation.json").read_bytes()
            ).hexdigest(),
        }
    )


def _refresh_event_id(terminal: dict[str, object]) -> None:
    assessment = terminal["assessment"]
    assert isinstance(assessment, dict)
    event_payload = (
        f"{terminal['episode_id']}:{assessment['snapshot_sha256']}:"
        f"{assessment['graph_hash']}:{assessment['certificate_hash']}:"
        f"{assessment['monitor_contract_sha256']}:{assessment['event_type']}"
    )
    assessment["event_id"] = hashlib.sha256(
        b"LOGIV_TERMINAL_DEVIATION_V1\0" + event_payload.encode("utf-8")
    ).hexdigest()


def _resign_terminal(path: Path, terminal: dict[str, object]) -> None:
    assessment = terminal["assessment"]
    assert isinstance(assessment, dict)
    _resign_assessment(assessment)
    terminal["assessment_sha256"] = assessment["assessment_sha256"]
    _write_json(path, terminal)


def _rewrite_snapshot_values(
    case_dir: Path, updates: dict[str, str]
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = _artifact_dir(case_dir)
    snapshot_path = artifact / "current_snapshot.json"
    terminal_path = artifact / "terminal_deviation.json"
    snapshot = json.loads(snapshot_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    evidence = json.loads(snapshot["evidence_payload_json"])
    values = {fact: state for fact, state in evidence["values"]}
    values.update(updates)
    evidence["values"] = [[fact, values[fact]] for fact in sorted(values)]
    evidence_json = json.dumps(
        evidence, separators=(",", ":"), sort_keys=True
    )
    evidence_hash = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
    snapshot.update(
        {
            "evidence_hash": evidence_hash,
            "evidence_payload_json": evidence_json,
            "true": sorted(
                fact for fact, state in values.items() if state == "TRUE"
            ),
            "false": sorted(
                fact for fact, state in values.items() if state == "FALSE"
            ),
        }
    )
    assessment = terminal["assessment"]
    assert isinstance(assessment, dict)
    assessment["snapshot_sha256"] = evidence_hash
    _refresh_event_id(terminal)
    _resign_terminal(terminal_path, terminal)
    _write_json(snapshot_path, snapshot)
    return snapshot, terminal


def _replay_fixture(case_dir: Path) -> None:
    artifact = _artifact_dir(case_dir)
    snapshot_payload = json.loads(
        (artifact / "current_snapshot.json").read_text()
    )
    terminal = json.loads((artifact / "terminal_deviation.json").read_text())
    assessment = _authenticated_assessment(terminal["assessment"])
    _replay_eligible_assessment(
        case_id=case_dir.name,
        artifact_dir=artifact,
        episode_id=terminal["episode_id"],
        snapshot=_strict_snapshot(snapshot_payload),
        terminal=terminal,
        assessment=assessment,
        observed_steps=terminal["base_policy_steps"],
        hashes={
            field: terminal[field]
            for field in (
                "graph_hash",
                "certificate_hash",
                "monitor_contract_sha256",
                "capability_sha256",
            )
        },
        expected_graph_artifact_sha256=preflight_report._FROZEN_CASES[
            case_dir.name
        ]["graph_artifact_sha256"],
    )


def test_terminal_preflight_replay_accepts_legitimate_eligible_evidence(
    tmp_path: Path,
) -> None:
    _replay_fixture(_write_case(tmp_path, "t05-r02"))


def test_terminal_preflight_frozen_capture_anchors_match_committed_result() -> None:
    result = json.loads(
        (
            Path(__file__).parents[2]
            / "results/task5-terminal-preflight-20260805.json"
        ).read_text()
    )

    for row in result["cases"]:
        frozen = preflight_report._FROZEN_CASES[row["case_id"]]
        assert row["assessment_sha256"] == frozen["assessment_sha256"]
        assert row["base_policy_steps"] == frozen["base_policy_steps"]
        assert row["base_policy_requests"] == frozen["base_policy_requests"]


@pytest.mark.parametrize("accounting", ["steps", "requests"])
def test_terminal_preflight_rejects_fully_rewritten_base_accounting(
    tmp_path: Path, accounting: str
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = _artifact_dir(case_dir)
    record_path = case_dir / "episodes.jsonl"
    base_path = artifact / "base_execution.json"
    compute_path = artifact / "compute_accounting.json"
    terminal_path = artifact / "terminal_deviation.json"
    record = json.loads(record_path.read_text())
    base = json.loads(base_path.read_text())
    compute = json.loads(compute_path.read_text())
    terminal = json.loads(terminal_path.read_text())

    if accounting == "steps":
        record["steps"] = 341
        base["steps"] = 341
        terminal["base_policy_steps"] = 341
        terminal["assessment"]["base_policy_steps"] = 341
        _resign_terminal(terminal_path, terminal)
    else:
        record["inference_requests"] = 69
        record["base_policy_requests"] = 69
        base["inference_requests"] = 69
        base["base_policy_requests"] = 69
        compute["base_policy_requests"] = 69
        terminal["base_policy_requests"] = 69
        _write_json(terminal_path, terminal)
    record_path.write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(base_path, base)
    _write_json(compute_path, compute)
    _trust_current_artifacts(case_dir)

    with pytest.raises(ValueError, match="frozen Base capture anchor"):
        load_case(case_dir)


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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(case_dir)

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
        _trust_current_artifacts(case_dir)

    with pytest.raises(ValueError, match="assessment"):
        summarize_preflight(case_dirs)


def test_terminal_preflight_report_rejects_fully_resigned_false_go(
    tmp_path: Path,
) -> None:
    case_dirs = [
        _write_case(tmp_path, case_id, eligible=False) for case_id in CASES
    ]
    for case_dir in case_dirs[:2]:
        artifact = _artifact_dir(case_dir)
        path = artifact / "terminal_deviation.json"
        payload = json.loads(path.read_text())
        payload["assessment"] = _assessment_payload(
            eligible=True,
            snapshot_sha256=payload["assessment"]["snapshot_sha256"],
            episode_id=payload["episode_id"],
        )
        payload.update(
            {
                "assessment_sha256": payload["assessment"]["assessment_sha256"],
                "certificate_state": "CURRENT",
                "place_node_action": CAPABILITY.action,
                "place_node_status": "ACTIVE",
                "reason": "ELIGIBLE",
                "status": "ELIGIBLE",
            }
        )
        _write_json(path, payload)

    with pytest.raises(ValueError, match="frozen artifact custody"):
        summarize_preflight(case_dirs)


@pytest.mark.parametrize(
    "mutation",
    [
        "graph_node_body",
        "graph_source_epoch",
        "graph_topology",
        "graph_final_trace",
        "capture_timestamp",
    ],
)
def test_terminal_preflight_report_rejects_post_capture_artifact_rewrites(
    tmp_path: Path, mutation: str
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02", eligible=False)
    artifact = _artifact_dir(case_dir)
    graph_path = artifact / "graph.json"
    snapshot_path = artifact / "current_snapshot.json"
    terminal_path = artifact / "terminal_deviation.json"
    graph = json.loads(graph_path.read_text())

    if mutation == "graph_node_body":
        graph["nodes"][0]["lineage_root"] = "forged-root"
        _write_json(graph_path, graph)
    elif mutation == "graph_source_epoch":
        graph["source_epoch"] = 7
        _write_json(graph_path, graph)
    elif mutation == "graph_topology":
        graph["canonical_agenda"] = []
        _write_json(graph_path, graph)
    elif mutation == "graph_final_trace":
        graph["state_trace"][-1]["certificate_state"] = "CURRENT"
        graph["state_trace"][-1]["nodes"][0]["status"] = "ACTIVE"
        _write_json(graph_path, graph)
    elif mutation == "capture_timestamp":
        forged_timestamp = "2026-08-05T13:36:18.270672+00:00"
        for path in (snapshot_path, terminal_path):
            payload = json.loads(path.read_text())
            payload["captured_at_utc"] = forged_timestamp
            _write_json(path, payload)
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="frozen artifact custody"):
        load_case(case_dir)


@pytest.mark.parametrize(
    "semantic",
    [
        "frozen_capability",
        "snapshot_source_epoch",
        "location_exactly_one",
        "location_unknown",
        "holding_true",
        "target_false",
        "protected_invariant_true",
        "protected_set_exact",
        "certificate_current",
        "graph_hash_anchor",
        "certificate_hash_anchor",
        "graph_version_anchor",
        "graph_membership",
        "terminal_action",
        "terminal_node_status",
        "assessment_node_id",
        "event_type",
        "event_id",
        "action_cap",
        "remaining_budget",
    ],
)
def test_terminal_preflight_report_replays_each_eligibility_semantic(
    tmp_path: Path, semantic: str
) -> None:
    case_dir = _write_case(tmp_path, "t05-r02")
    artifact = _artifact_dir(case_dir)
    snapshot_path = artifact / "current_snapshot.json"
    terminal_path = artifact / "terminal_deviation.json"
    graph_path = artifact / "graph.json"
    snapshot = json.loads(snapshot_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    graph = json.loads(graph_path.read_text())
    assessment = terminal["assessment"]

    if semantic == "frozen_capability":
        forged_hash = "f" * 64
        snapshot["capability_sha256"] = forged_hash
        terminal["capability_sha256"] = forged_hash
        assessment["capability_sha256"] = forged_hash
        _write_json(snapshot_path, snapshot)
        _resign_terminal(terminal_path, terminal)
    elif semantic == "snapshot_source_epoch":
        graph["source_epoch"] = snapshot["epoch_id"] + 1
        _write_json(graph_path, graph)
    elif semantic == "location_exactly_one":
        _rewrite_snapshot_values(
            case_dir,
            {
                CAPABILITY.holding_fact.pddl(): "FALSE",
                CAPABILITY.target_fact.pddl(): "FALSE",
                WRONG_LOCATION.pddl(): "FALSE",
            },
        )
    elif semantic == "location_unknown":
        _rewrite_snapshot_values(
            case_dir, {WRONG_LOCATION.pddl(): "UNKNOWN"}
        )
    elif semantic == "holding_true":
        _rewrite_snapshot_values(
            case_dir,
            {
                CAPABILITY.holding_fact.pddl(): "FALSE",
                WRONG_LOCATION.pddl(): "TRUE",
            },
        )
    elif semantic == "target_false":
        _rewrite_snapshot_values(
            case_dir, {CAPABILITY.target_fact.pddl(): "UNKNOWN"}
        )
    elif semantic == "protected_invariant_true":
        invariant = min(CAPABILITY.protected_invariants)
        _, terminal = _rewrite_snapshot_values(
            case_dir, {invariant.pddl(): "FALSE"}
        )
        terminal["assessment"]["protected_true_facts"].remove(invariant.pddl())
        _resign_terminal(terminal_path, terminal)
    elif semantic == "protected_set_exact":
        assessment["protected_true_facts"] = []
        _resign_terminal(terminal_path, terminal)
    elif semantic == "certificate_current":
        terminal["certificate_state"] = "STALE"
        graph["state_trace"][-1]["certificate_state"] = "STALE"
        _write_json(graph_path, graph)
        _write_json(terminal_path, terminal)
    elif semantic == "graph_hash_anchor":
        graph["graph_hash"] = "b" * 64
        _write_json(graph_path, graph)
    elif semantic == "certificate_hash_anchor":
        graph["certificate_hash"] = "b" * 64
        _write_json(graph_path, graph)
    elif semantic == "graph_version_anchor":
        graph["graph_version"] = "graph-forged"
        _write_json(graph_path, graph)
    elif semantic == "graph_membership":
        duplicate = dict(graph["nodes"][0])
        duplicate.update({"canonical_rank": 1, "node_id": "duplicate-place"})
        graph["nodes"].append(duplicate)
        graph["canonical_agenda"].append("duplicate-place")
        graph["state_trace"][-1]["nodes"].append(
            {"node_id": "duplicate-place", "status": "ACTIVE"}
        )
        _write_json(graph_path, graph)
    elif semantic == "terminal_action":
        terminal["place_node_action"] = "(pick black_book_1)"
        _write_json(terminal_path, terminal)
    elif semantic == "terminal_node_status":
        terminal["place_node_status"] = "BLOCKED"
        graph["state_trace"][-1]["nodes"][0]["status"] = "BLOCKED"
        _write_json(graph_path, graph)
        _write_json(terminal_path, terminal)
    elif semantic == "assessment_node_id":
        assessment["place_node_id"] = "forged-place-node"
        _resign_terminal(terminal_path, terminal)
    elif semantic == "event_type":
        assessment["event_type"] = "OTHER_EVENT"
        _refresh_event_id(terminal)
        _resign_terminal(terminal_path, terminal)
    elif semantic == "event_id":
        assessment["event_id"] = "f" * 64
        _resign_terminal(terminal_path, terminal)
    elif semantic == "action_cap":
        assessment["option_action_cap"] -= 1
        _resign_terminal(terminal_path, terminal)
    elif semantic == "remaining_budget":
        record = json.loads((case_dir / "episodes.jsonl").read_text())
        record["steps"] = CAPABILITY.max_combined_actions
        (case_dir / "episodes.jsonl").write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        base_path = artifact / "base_execution.json"
        base = json.loads(base_path.read_text())
        base["steps"] = CAPABILITY.max_combined_actions
        _write_json(base_path, base)
        terminal["base_policy_steps"] = CAPABILITY.max_combined_actions
        assessment["base_policy_steps"] = CAPABILITY.max_combined_actions
        assessment["option_action_cap"] = 1
        _resign_terminal(terminal_path, terminal)
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(semantic)

    with pytest.raises(ValueError, match="eligibility"):
        _replay_fixture(case_dir)


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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(case_dir)

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

    with pytest.raises(ValueError, match="eligibility"):
        _replay_fixture(case_dir)


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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(case_dir)

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
    _trust_current_artifacts(target)

    with pytest.raises(ValueError, match="identity"):
        load_case(target)
