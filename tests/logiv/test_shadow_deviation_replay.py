from __future__ import annotations

from scripts.report_shadow_deviation_replay import analyze_case


def _frame(step: int, certificate: str, action: str, goal: str = "BLOCKED") -> dict:
    return {
        "policy_step": step,
        "phase": "POLICY",
        "certificate_state": certificate,
        "nodes": [
            {"node_id": "INIT", "status": "COMPLETED"},
            {"node_id": "a0", "status": action},
            {"node_id": "GOAL", "status": goal},
        ],
    }


GRAPH = {
    "nodes": [
        {"node_id": "INIT", "kind": "INIT"},
        {"node_id": "a0", "kind": "ACTION"},
        {"node_id": "GOAL", "kind": "GOAL"},
    ]
}


def test_successful_stale_trace_is_retrospective_self_recovery_not_confirmation() -> None:
    case = {"case_id": "success", "base_success": True, "base_steps": 30}
    graph = {
        **GRAPH,
        "state_trace": [
            _frame(0, "CURRENT", "READY"),
            _frame(10, "STALE", "BLOCKED"),
            _frame(20, "STALE", "COMPLETED", "COMPLETED"),
        ],
    }

    result = analyze_case(case, graph)

    assert result["retrospective_label"] == "TRANSIENT_OR_SELF_RECOVERED"
    assert result["first_stale_policy_step"] == 10
    assert result["policy_steps_remaining"] == 20
    assert result["progress_after_stale"] is True
    assert result["confirmed_deviation"] is False
    assert result["requires_action_evidence"] is True


def test_failed_current_trace_is_a_stall_candidate_without_fake_confirmation() -> None:
    case = {"case_id": "failure", "base_success": False, "base_steps": 30}
    graph = {
        **GRAPH,
        "state_trace": [
            _frame(0, "CURRENT", "READY"),
            _frame(30, "CURRENT", "READY"),
        ],
    }

    result = analyze_case(case, graph)

    assert result["retrospective_label"] == "STALL_WITHOUT_STALE"
    assert result["first_stale_policy_step"] is None
    assert result["confirmed_deviation"] is False
    assert result["requires_action_evidence"] is True
