from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.domain import FixedDomain, apply_action
from pi05_libero_repro.logiv.configuration import resolved_json_sha256
from pi05_libero_repro.logiv.model import Fact, GoalMode
from pi05_libero_repro.logiv.proposal import (
    ProposalError,
    ScriptedProposalProvider,
    write_proposal_artifacts,
)


FIXTURE = Path("configs/logiv/origin/proposals.json")
COVERAGE = Path("configs/logiv/origin/coverage.json")


def test_metadata_assisted_task3_has_no_vlm_goal_and_uses_origin_macro() -> None:
    package = ScriptedProposalProvider(FIXTURE).propose(task_id=3, epoch_id=17)

    assert package.proposal.goal_mode is GoalMode.METADATA_ASSISTED
    assert package.proposal.grounded_goal is None
    assert [item.action.schema for item in package.proposal.candidate_subtasks] == [
        "place-in",
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


def test_task5_origin_macro_keeps_the_official_instruction_atomic() -> None:
    package = ScriptedProposalProvider(FIXTURE).propose(task_id=5, epoch_id=19)
    candidates = package.proposal.candidate_subtasks

    assert [item.action.schema for item in candidates] == ["place-in"]
    assert candidates[0].instruction == (
        "Pick up the black book and place it in the back compartment of the desk caddy."
    )


def test_extended_proposal_config_adds_distinct_task0_recovery_location(
    tmp_path: Path,
) -> None:
    source = "living_room_table_secondary_recovery_surface"
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    task = base["tasks"][0]
    task["objects"].append([source, "surface"])
    task["initial_false"].extend(
        ["at", object_name, source]
        for object_name in ("alphabet_soup_1", "tomato_sauce_1")
    )
    overlay = tmp_path / "proposal-overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "extends": str(FIXTURE.resolve()),
                "tasks": [
                    {
                        key: task[key]
                        for key in (
                            "task_id",
                            "objects",
                            "initial_true",
                            "initial_false",
                            "candidate_subtasks",
                        )
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    package = ScriptedProposalProvider(overlay).propose(task_id=0, epoch_id=0)

    assert source in package.problem.object_types
    assert Fact("at", ("alphabet_soup_1", source)) in package.problem.initial_false
    assert Fact("at", ("tomato_sauce_1", source)) in package.problem.initial_false
    assert [candidate.action.arguments[1] for candidate in package.proposal.candidate_subtasks] == [
        "living_room_table_alphabet_soup_init_region",
        "living_room_table_tomato_sauce_init_region",
    ]


def test_extended_config_fingerprint_binds_parent_content(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    overlay = tmp_path / "overlay.json"
    parent.write_text('{"tasks":[{"task_id":0,"value":1}]}', encoding="utf-8")
    overlay.write_text(
        '{"extends":"parent.json","tasks":[{"task_id":0,"extra":2}]}',
        encoding="utf-8",
    )
    original = resolved_json_sha256(overlay)

    parent.write_text('{"tasks":[{"task_id":0,"value":3}]}', encoding="utf-8")

    assert resolved_json_sha256(overlay) != original


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

    task5 = packages[5]
    recovery_locations = {
        "study_table_recovery_surface",
        "desk_caddy_1_front_contain_region",
        "desk_caddy_1_left_contain_region",
        "desk_caddy_1_right_contain_region",
    }
    assert recovery_locations <= set(task5.problem.object_types)
    assert {
        Fact("at", ("black_book_1", location)) for location in recovery_locations
    } <= task5.problem.initial_false
    task6 = packages[6]
    assert "living_room_table_recovery_surface" in task6.problem.object_types
    assert {
        Fact("at", (object_name, "living_room_table_recovery_surface"))
        for object_name in ("porcelain_mug_1", "chocolate_pudding_1")
    } <= task6.problem.initial_false
    task9 = packages[9]
    assert "kitchen_table_recovery_surface" in task9.problem.object_types
    assert Fact(
        "at", ("white_yellow_mug_1", "kitchen_table_recovery_surface")
    ) in task9.problem.initial_false


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
