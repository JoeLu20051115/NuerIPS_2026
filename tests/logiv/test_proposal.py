from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.domain import FixedDomain, apply_action
from pi05_libero_repro.logiv.model import Fact, GoalMode
from pi05_libero_repro.logiv.proposal import (
    ProposalError,
    ScriptedProposalProvider,
    write_proposal_artifacts,
)


FIXTURE = Path("configs/logiv/libero10-scripted-proposals.json")
COVERAGE = Path("configs/logiv/libero10-coverage.json")


def test_metadata_assisted_task3_has_no_vlm_goal_and_uses_fine_grained_plan() -> None:
    package = ScriptedProposalProvider(FIXTURE).propose(task_id=3, epoch_id=17)

    assert package.proposal.goal_mode is GoalMode.METADATA_ASSISTED
    assert package.proposal.grounded_goal is None
    assert [item.action.schema for item in package.proposal.candidate_subtasks] == [
        "pick",
        "place-held-in",
        "close-access",
    ]
    assert Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")) in (
        package.problem.goal
    )
    assert Fact("closed", ("white_cabinet_1_bottom_access",)) in package.problem.goal
    assert package.frozen_goal.source == "official_bddl_metadata"


def test_task8_keeps_two_distinct_unordered_placement_occurrences() -> None:
    package = ScriptedProposalProvider(FIXTURE).propose(task_id=8, epoch_id=23)
    candidates = package.proposal.candidate_subtasks

    assert [item.action.schema for item in candidates] == ["place-on", "place-on"]
    assert [item.action.arguments[0] for item in candidates] == ["moka_pot_2", "moka_pot_1"]
    assert Fact("at", ("moka_pot_2", "kitchen_table_recovery_surface")) in (
        package.problem.initial_false
    )
    assert candidates[0].occurrence_id != candidates[1].occurrence_id
    assert all(item.rough_rank in {0, 1} for item in candidates)


def test_writer_outputs_auditable_consistent_artifacts(tmp_path: Path) -> None:
    package = ScriptedProposalProvider(FIXTURE).propose(task_id=3, epoch_id=31)

    artifacts = write_proposal_artifacts(tmp_path, package, FixedDomain())

    proposal = json.loads(artifacts.proposal_json.read_text(encoding="utf-8"))
    occurrences = json.loads(artifacts.occurrences_json.read_text(encoding="utf-8"))
    plan_lines = artifacts.candidate_plan.read_text(encoding="utf-8").splitlines()
    problem = artifacts.initial_problem.read_text(encoding="utf-8")
    assert proposal["grounded_goal"] is None
    assert proposal["initial_state"]["evidence_source"] == "scripted-vlm/oracle-grounding"
    assert proposal["source_bddl_sha256"] == package.proposal.source_bddl_sha256
    assert proposal["goal_contract"]["goal_id"] == package.frozen_goal.goal_id
    assert [item["occurrence_id"] for item in occurrences] == [
        item.occurrence_id for item in package.proposal.candidate_subtasks
    ]
    assert plan_lines == [item.action.pddl() for item in package.proposal.candidate_subtasks]
    assert "(:domain logiv-libero)" in problem
    assert "(closed white_cabinet_1_bottom_access)" in problem


def test_all_ten_fixtures_ground_and_coverage_is_frozen() -> None:
    provider = ScriptedProposalProvider(FIXTURE)
    packages = [provider.propose(task_id=task_id, epoch_id=task_id) for task_id in range(10)]
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))

    assert [item.proposal.task_id for item in packages] == list(range(10))
    assert coverage["frozen"] is True
    assert coverage["goal_mode"] == "METADATA_ASSISTED"
    assert [item["task_id"] for item in coverage["tasks"]] == list(range(10))
    assert all(item.proposal.candidate_subtasks for item in packages)
    for package, coverage_item in zip(packages, coverage["tasks"]):
        assert coverage_item["registered_objects"] == [
            item.name for item in package.proposal.registered_objects
        ]
        assert coverage_item["bddl_file"] == package.proposal.source_bddl
        assert coverage_item["bddl_sha256"] == package.proposal.source_bddl_sha256
        assert {item.action.schema for item in package.proposal.candidate_subtasks} <= set(
            coverage_item["supported_action_schemas"]
        )
        assert set(coverage_item["symbol_bindings"]) <= set(
            coverage_item["registered_objects"]
        )


def test_all_ten_candidate_plans_reach_the_frozen_goal() -> None:
    provider = ScriptedProposalProvider(FIXTURE)

    for task_id in range(10):
        package = provider.propose(task_id=task_id, epoch_id=task_id)
        state = package.problem.initial_state
        for candidate in package.proposal.candidate_subtasks:
            state = apply_action(package.problem, state, candidate.action)
        assert package.problem.goal <= state, task_id
        assert not (package.problem.negative_goal & state), task_id


def test_unknown_task_and_schema_drift_fail_closed(tmp_path: Path) -> None:
    provider = ScriptedProposalProvider(FIXTURE)
    with pytest.raises(ProposalError, match="unknown LIBERO-10 task_id"):
        provider.propose(task_id=10, epoch_id=0)

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["tasks"][0]["candidate_subtasks"][0]["schema"] = "invented-action"
    bad_fixture = tmp_path / "bad.json"
    bad_fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProposalError, match="unknown action schema"):
        ScriptedProposalProvider(bad_fixture).propose(task_id=0, epoch_id=0)


def test_incomplete_initial_exactly_one_state_fails_during_proposal(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["tasks"][0]["initial_true"] = [
        fact
        for fact in payload["tasks"][0]["initial_true"]
        if fact[:2] != ["at", "alphabet_soup_1"]
    ]
    bad_fixture = tmp_path / "bad-state.json"
    bad_fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProposalError, match="exactly one location"):
        ScriptedProposalProvider(bad_fixture).propose(task_id=0, epoch_id=0)
