from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pi05_libero_repro.logiv.dag import SchemaOnlyCausalDagCompiler
from pi05_libero_repro.logiv.domain import FixedDomain
from pi05_libero_repro.logiv.configuration import resolved_json_sha256
from pi05_libero_repro.logiv.evaluation import (
    EvaluationContract,
    EvaluationContractError,
    GlobalRepairOperator,
    MethodArm,
    certify_initial_package,
)
from pi05_libero_repro.logiv.initial_proposal import (
    InitialProposalResult,
    InitialProposalStatus,
)
from pi05_libero_repro.logiv.model import Fact, FactSnapshot, GoalMode
from pi05_libero_repro.logiv.shadow_runtime import (
    ShadowRuntime,
    ShadowRuntimeCounters,
)
from pi05_libero_repro.logiv.task5_terminal_recovery import TerminalAssessment
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.repair import (
    PddlPlanner,
    RepairBounds,
    RepairOperator,
    RetryPolicy,
)
from pi05_libero_repro.logiv.val import ValWrapper
from scripts.eval_logiv_libero import (
    _allocate_episode_artifact_dir,
    _base_physical_attempts,
    _capture_task5_terminal_preflight,
    _gpt4o_request_accounting,
    _parser,
    _place_effect_confirmation_steps_for_task,
    _place_effect_stabilization_steps_for_task,
    _post_stop_reobservation_steps_for_task,
    _rebase_package_on_snapshot,
    _remaining_online_action_budget,
    _replace_episode_environment,
    _run_config,
    _online_detector_settings_for_task,
    _shadow_artifact_payloads,
    _symbolic_record_accounting,
    _validate_shadow_options,
    _verified_recovery_surface_facts,
)
from pi05_libero_repro.protocol import EpisodeOutcome, ShadowFailureRecord
import scripts.eval_logiv_libero as evaluator_script


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")


def test_gpt4o_request_accounting_reports_only_state_gate() -> None:
    assert _gpt4o_request_accounting(None) == {
        "shadow_vlm_requests": 0,
        "recovery_policy_requests": 0,
    }
    client = SimpleNamespace(request_counts={"state_gate": 3})
    assert _gpt4o_request_accounting(client) == {
        "shadow_vlm_requests": 3,
        "recovery_policy_requests": 0,
    }


def test_gpt4o_request_accounting_rejects_planning_purposes() -> None:
    client = SimpleNamespace(
        request_counts={"state_gate": 3, "local_repair": 1}
    )
    with pytest.raises(ValueError, match="non-State-Gate"):
        _gpt4o_request_accounting(client)


def test_direct_symbolic_accounting_keeps_state_gate_requests() -> None:
    client = SimpleNamespace(request_counts={"state_gate": 4})

    accounting = _symbolic_record_accounting(
        base_policy_requests=9,
        gpt4o_client=client,
    )

    assert accounting["base_policy_requests"] == 9
    assert accounting["shadow_vlm_requests"] == 4
    assert accounting["recovery_policy_requests"] == 0


def test_online_tuning_configs_resolve_for_all_ten_tasks() -> None:
    root = Path(__file__).parents[2]
    proposal_path = root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    coverage_path = root / "configs/logiv/libero10-coverage-online-v1.json"
    prompt_path = root / "configs/logiv/prompts/pi05-subtasks-online-v1.json"

    provider = ScriptedProposalProvider(proposal_path)
    renderer = evaluator_script.SubtaskPromptRenderer(prompt_path)
    for task_id in range(10):
        package = provider.propose(task_id, epoch_id=0)
        binding = evaluator_script.TaskBinding.from_manifest(
            coverage_path, task_id
        )
        assert package.proposal.candidate_subtasks
        assert binding.supported_action_schemas
    assert renderer.prompt_version == "pi05-subtasks-online-v1"


def test_task0_direct_recovery_manifest_keeps_the_place_in_macro() -> None:
    root = Path(__file__).parents[2]
    binding = evaluator_script.TaskBinding.from_manifest(
        root
        / "configs/logiv/libero10-coverage-online-v2-task0-direct.json",
        0,
    )

    assert binding.decompose_macro_sources == frozenset()


def test_online_v2_disambiguates_task8_recovery_objects() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v2.json"
    )
    domain = FixedDomain()

    left = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )
    right = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render(left) == "Put the left moka pot on the stove."
    assert renderer.render(right) == "Put the right moka pot on the stove."


def test_online_v3_disambiguates_task8_initial_objects() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v3.json"
    )
    domain = FixedDomain()

    left = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_moka_pot_left_init_region",
            "flat_stove_1_cook_region",
        ),
    )
    right = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_moka_pot_right_init_region",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render(left) == "Put the left moka pot on the stove."
    assert renderer.render(right) == "Put the right moka pot on the stove."


def test_online_v4_uses_whole_task_prompt_for_task8_recovery() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v4.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render(action) == "Put both moka pots on the stove."


def test_online_v5_splits_task8_recovery_into_pick_and_place_phases() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Pick up the moka pot on the left side of the table and hold it."
    )
    assert renderer.render_phase(action, "finish") == (
        "Place the moka pot you are holding on the stove and release it."
    )


def test_online_v6_uses_direct_task8_placement_without_split_phases() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v6-direct.json"
    )
    domain = FixedDomain()
    actions = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ] + [
        domain.ground(
            package.problem,
            "place-on",
            (
                object_name,
                "kitchen_table_recovery_surface",
                "flat_stove_1_cook_region",
            ),
        )
        for object_name in ("moka_pot_2", "moka_pot_1")
    ]

    assert [renderer.render_phase(action, "acquire") for action in actions] == [
        "Put the left moka pot on the stove.",
        "Put the right moka pot on the stove.",
        "Put the left moka pot on the stove.",
        "Put the right moka pot on the stove.",
    ]
    assert all(not renderer.has_phase(action, "finish") for action in actions)


def test_online_v7_ports_only_contextual_task8_recovery_behavior() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v7-contextual.json"
    )
    domain = FixedDomain()
    nominal = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]
    recovery = [
        domain.ground(
            package.problem,
            "place-on",
            (
                object_name,
                "kitchen_table_recovery_surface",
                "flat_stove_1_cook_region",
            ),
        )
        for object_name in ("moka_pot_2", "moka_pot_1")
    ]

    assert {
        renderer.render_recovery_frontier(action)
        for action in nominal + recovery
    } == {"put both moka pots on the stove"}
    assert "remaining moka pot" in renderer.render_phase(nominal[1], "acquire")
    assert "without moving the other moka pot" in renderer.render_phase(
        nominal[1], "acquire"
    )
    assert "remaining moka pot" in renderer.render_phase(recovery[0], "acquire")


def test_online_v8_changes_only_the_hard_left_moka_pot_repairs() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v8-left-context.json"
    )
    domain = FixedDomain()
    nominal = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]
    left_recovery = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )
    right_recovery = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    for action in (nominal[0], left_recovery):
        prompt = renderer.render_phase(action, "acquire")
        assert "remaining moka pot" in prompt
        assert "without moving the other moka pot" in prompt
    for action in (nominal[1], right_recovery):
        assert renderer.render_phase(action, "acquire") == parent.render_phase(
            action, "acquire"
        )


def test_online_v9_uses_a_short_visual_reference_only_for_left_moka_pot() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v9-left-table.json"
    )
    domain = FixedDomain()
    nominal = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]
    recovery = [
        domain.ground(
            package.problem,
            "place-on",
            (
                object_name,
                "kitchen_table_recovery_surface",
                "flat_stove_1_cook_region",
            ),
        )
        for object_name in ("moka_pot_2", "moka_pot_1")
    ]

    expected = "Put the moka pot that is still on the table on the stove."
    assert renderer.render_phase(nominal[0], "acquire") == expected
    assert renderer.render_phase(recovery[0], "acquire") == expected
    for action in (nominal[1], recovery[1]):
        assert renderer.render_phase(action, "acquire") == parent.render_phase(
            action, "acquire"
        )


def test_online_v10_targets_only_the_verified_moved_left_moka_pot() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v10-moved-left.json"
    )
    domain = FixedDomain()
    nominal_left, nominal_right = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]
    moved_left = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )
    moved_right = domain.ground(
        package.problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render_phase(moved_left, "acquire") == (
        "Put the moka pot that was moved to another part of the kitchen table "
        "on the stove."
    )
    for action in (nominal_left, nominal_right, moved_right):
        assert renderer.render_phase(action, "acquire") == parent.render_phase(
            action, "acquire"
        )


def test_online_v11_uses_a_direct_task0_tomato_repair_only() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(0, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v11-task0-direct.json"
    )
    alphabet, tomato = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]

    assert renderer.render_phase(tomato, "acquire") == (
        "Put the tomato sauce can in the basket."
    )
    assert renderer.render_phase(alphabet, "acquire") == parent.render_phase(
        alphabet, "acquire"
    )


def test_online_v12_uses_the_complete_task0_goal_for_either_repair_node() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(0, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v12-task0-goal.json"
    )
    actions = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]

    assert [renderer.render_phase(action, "acquire") for action in actions] == [
        "put both the alphabet soup and the tomato sauce in the basket",
        "put both the alphabet soup and the tomato sauce in the basket",
    ]
    assert all(not renderer.has_phase(action, "finish") for action in actions)


def test_online_v13_uses_a_direct_task9_recovery_from_the_table() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(9, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v13-task9-recovery.json"
    )
    domain = FixedDomain()
    nominal = package.proposal.candidate_subtasks[0].action
    recovery = domain.ground(
        package.problem,
        "place-in",
        (
            "white_yellow_mug_1",
            "kitchen_table_recovery_surface",
            "microwave_1_heating_region",
            "microwave_1_access",
        ),
    )

    assert renderer.render_phase(recovery, "acquire") == (
        "Put the yellow and white mug in the microwave and close it."
    )
    assert not renderer.has_phase(recovery, "finish")
    assert renderer.render_phase(nominal, "acquire") == parent.render_phase(
        nominal, "acquire"
    )


def test_online_v14_uses_a_short_task6_fallen_mug_repair() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(6, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v14-task6-mug.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-on",
        ("porcelain_mug_1", "living_room_table_recovery_surface", "plate_1"),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Put the white mug upright on the plate."
    )
    assert not renderer.has_phase(action, "finish")


def test_online_v15_changes_only_the_task8_nominal_right_pot_prompt() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(8, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v15-task8-right.json"
    )
    left, right = [
        candidate.action for candidate in package.proposal.candidate_subtasks
    ]
    recovery_right = FixedDomain().ground(
        package.problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert renderer.render_phase(right, "acquire") == (
        "Put the right moka pot on the stove."
    )
    assert not renderer.has_phase(right, "finish")
    for action in (left, recovery_right):
        assert renderer.render_phase(action, "acquire") == parent.render_phase(
            action, "acquire"
        )


def test_online_v16_uses_a_single_effect_task9_recovery_prompt() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(9, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v16-task9-place.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-in",
        (
            "white_yellow_mug_1",
            "kitchen_table_recovery_surface",
            "microwave_1_heating_region",
            "microwave_1_access",
        ),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Put the yellow and white mug in the microwave."
    )
    assert not renderer.has_phase(action, "finish")


def test_online_v17_uses_the_shortest_task6_mug_repair() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(6, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v17-task6-mug.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-on",
        ("porcelain_mug_1", "living_room_table_recovery_surface", "plate_1"),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Put the white mug on the plate."
    )
    assert not renderer.has_phase(action, "finish")


def test_online_v18_shortens_only_the_task0_tomato_recovery_actions() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(0, epoch_id=0)
    parent = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v5.json"
    )
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v18-task0-recovery.json"
    )
    domain = FixedDomain()
    pick = domain.ground(
        package.problem,
        "pick",
        ("tomato_sauce_1", "living_room_table_recovery_surface"),
    )
    place = domain.ground(
        package.problem,
        "place-held-in",
        ("tomato_sauce_1", "basket_1_contain_region", "basket_1_access"),
    )
    nominal = package.proposal.candidate_subtasks[0].action

    assert renderer.render(pick) == "Pick up the tomato sauce can."
    assert renderer.render(place) == "Put the tomato sauce can in the basket."
    assert renderer.render(nominal) == parent.render(nominal)


def test_online_v19_executes_task0_tomato_recovery_as_one_macro() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(0, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v19-task0-macro.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-in",
        (
            "tomato_sauce_1",
            "living_room_table_recovery_surface",
            "basket_1_contain_region",
            "basket_1_access",
        ),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Put the tomato sauce can in the basket."
    )
    assert not renderer.has_phase(action, "finish")


def test_online_v20_uses_the_native_task3_goal_for_bowl_repair() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(3, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v20-task3-macro.json"
    )
    domain = FixedDomain()
    nominal = package.proposal.candidate_subtasks[0].action
    recovery = domain.ground(
        package.problem,
        "place-in",
        (
            "akita_black_bowl_1",
            "kitchen_table_recovery_surface",
            "white_cabinet_1_bottom_region",
            "white_cabinet_1_bottom_access",
        ),
    )

    expected = "Put the black bowl in the bottom drawer and close it."
    assert renderer.render_phase(nominal, "acquire") == expected
    assert renderer.render_phase(recovery, "acquire") == expected
    assert not renderer.has_phase(nominal, "finish")
    assert not renderer.has_phase(recovery, "finish")


def test_online_v21_uses_the_native_task6_goal_for_recovery() -> None:
    root = Path(__file__).parents[2]
    package = ScriptedProposalProvider(
        root / "configs/logiv/libero10-scripted-proposals-online-v1.json"
    ).propose(6, epoch_id=0)
    renderer = evaluator_script.SubtaskPromptRenderer(
        root / "configs/logiv/prompts/pi05-subtasks-online-v21-task6-goal.json"
    )
    action = FixedDomain().ground(
        package.problem,
        "place-on",
        ("porcelain_mug_1", "living_room_table_recovery_surface", "plate_1"),
    )

    assert renderer.render_phase(action, "acquire") == (
        "Put the white mug on the plate and put the chocolate pudding to the "
        "right of the plate."
    )
    assert not renderer.has_phase(action, "finish")


def test_evaluator_accepts_run_scoped_proposal_and_coverage_configs() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "config-isolation",
            "--method-arm",
            "BASE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/config-isolation",
            "--proposal-config",
            "/tmp/proposals.json",
            "--coverage-manifest",
            "/tmp/coverage.json",
        ]
    )

    assert args.proposal_config == Path("/tmp/proposals.json")
    assert args.coverage_manifest == Path("/tmp/coverage.json")


def test_shadow_snapshot_reader_uses_advisory_grounding_during_manipulation() -> None:
    class Grounder:
        def __init__(self) -> None:
            self.strict_calls = 0
            self.advisory_calls = 0

        def peek_snapshot(self):
            self.strict_calls += 1
            return "strict"

        def peek_advisory_partial_snapshot(self):
            self.advisory_calls += 1
            return "advisory"

    grounder = Grounder()
    peek = getattr(evaluator_script, "_shadow_snapshot_peek", None)
    assert peek is not None

    assert peek(grounder) == "advisory"
    assert (grounder.strict_calls, grounder.advisory_calls) == (0, 1)


def test_terminal_snapshot_reader_always_updates_and_uses_strict_grounding() -> None:
    updates = []

    class Store:
        def update(self, observation):
            updates.append(observation)

    class Grounder:
        def __init__(self) -> None:
            self.strict_calls = 0

        def peek_snapshot(self):
            self.strict_calls += 1
            return "strict"

    reader = getattr(evaluator_script, "_strict_terminal_snapshot_reader", None)
    assert reader is not None
    observation = {"frame": 2}
    grounder = Grounder()

    assert reader(Store(), grounder, observation) == "strict"
    assert updates == [observation]
    assert grounder.strict_calls == 1


def test_terminal_preflight_policy_reads_advisory_without_strict_fallback() -> None:
    updates = []

    class Store:
        def update(self, observation):
            updates.append(observation)

    class Grounder:
        def __init__(self) -> None:
            self.advisory_calls = 0
            self.strict_calls = 0

        def peek_advisory_partial_snapshot(self):
            self.advisory_calls += 1
            return "advisory"

        def peek_snapshot(self):
            self.strict_calls += 1
            raise ValueError("ambiguous final location")

    policy_reader = getattr(evaluator_script, "_policy_time_snapshot_peek", None)
    strict_reader = getattr(
        evaluator_script, "_strict_terminal_snapshot_reader", None
    )
    assert policy_reader is not None
    assert strict_reader is not None
    grounder = Grounder()

    assert policy_reader(grounder) == "advisory"
    with pytest.raises(ValueError, match="ambiguous final location"):
        strict_reader(Store(), grounder, {"frame": 2})

    assert updates == [{"frame": 2}]
    assert (grounder.advisory_calls, grounder.strict_calls) == (1, 1)


def _terminal_preflight_args(*extra: str):
    return _parser().parse_args(
        [
            "--run-id",
            "terminal-preflight",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "5",
            "--episode-indices",
            "36",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/terminal-preflight",
            "--capture-task5-terminal-preflight",
            *extra,
        ]
    )


def test_terminal_preflight_flag_is_frozen_into_run_config() -> None:
    args = _terminal_preflight_args()

    contracts = _validate_shadow_options(args, (5,))
    config = _run_config(args, (5,), (36,))

    assert contracts[5].task_id == 5
    assert config["capture_task5_terminal_preflight"] is True


@pytest.mark.parametrize(
    ("argv", "task_ids", "message"),
    [
        (("--method-arm", "BASE"), (5,), "requires SHADOW_LOGIV"),
        (("--development-only",), (5,), "requires development-only"),
        ((), (4,), "exactly Task 5"),
        ((), (5, 6), "exactly Task 5"),
        (("--collect-recovery-roots",), (5,), "cannot collect recovery roots"),
        (
            ("--shadow-topology-only",),
            (5,),
            "cannot enable recovery",
        ),
    ],
)
def test_terminal_preflight_rejects_every_nonfrozen_scope(
    argv, task_ids, message
) -> None:
    args = _terminal_preflight_args(*argv)
    if argv == ("--development-only",):
        args.development_only = False
    if argv[:1] == ("--method-arm",):
        args.method_arm = argv[1]

    with pytest.raises(ValueError, match=message):
        _validate_shadow_options(args, task_ids)


def test_terminal_preflight_grounding_error_is_fail_closed_and_anchors_base(
    tmp_path: Path, monkeypatch
) -> None:
    assessment = TerminalAssessment(
        eligible=False,
        reason="STRICT_AUDITED_SNAPSHOT_REQUIRED",
        event_id=None,
        event_type=None,
        option_action_cap=0,
        snapshot_sha256=None,
        graph_hash="a" * 64,
        certificate_hash="b" * 64,
        monitor_contract_sha256="c" * 64,
        protected_true_facts=frozenset(),
        capability_sha256="d" * 64,
        base_policy_steps=340,
        place_node_id=None,
        assessment_sha256="e" * 64,
    )
    received = {}

    def assess(*args, **kwargs):
        received.update(kwargs)
        return assessment

    def strict_reader(observation):
        assert observation == {"frame": 2}
        raise ValueError("ambiguous final location")

    certified = SimpleNamespace(
        problem=object(),
        graph=SimpleNamespace(nodes=(), graph_hash="a" * 64),
        certificate=SimpleNamespace(certificate_hash="b" * 64),
    )
    runtime = SimpleNamespace(
        initial_proposal=SimpleNamespace(
            validation=SimpleNamespace(
                certified_episode=certified,
                strict_terminal_snapshot_reader=strict_reader,
            )
        ),
        state_trace=[],
    )
    outcome = SimpleNamespace(
        check_success=False,
        final_observation={"frame": 2},
        inference_requests=68,
        steps=340,
    )
    monkeypatch.setattr(evaluator_script, "assess_task5_terminal", assess)

    _capture_task5_terminal_preflight(
        artifact_dir=tmp_path,
        case_id="t05-r02",
        task_id=5,
        episode_idx=36,
        episode_id="task5-terminal-preflight-t05-r02",
        outcome=outcome,
        native_terminal_status="EPISODE_FAIL",
        runtime=runtime,
        monitor_contract=SimpleNamespace(contract_sha256="c" * 64),
        capability=SimpleNamespace(action="(place-held-in x y z)", capability_sha256="d" * 64),
    )

    snapshot = json.loads((tmp_path / "current_snapshot.json").read_text())
    terminal = json.loads((tmp_path / "terminal_deviation.json").read_text())
    assert snapshot["status"] == "GROUNDING_ERROR"
    assert terminal["status"] == "GROUNDING_ERROR"
    assert terminal["reason"] == "GROUNDING_ERROR"
    assert terminal["assessment_sha256"] == assessment.assessment_sha256
    assert terminal["base_policy_steps"] == outcome.steps
    assert terminal["base_policy_requests"] == outcome.inference_requests
    assert terminal["recovery_actions"] == 0
    assert terminal["recovery_policy_requests"] == 0
    assert received["base_policy_steps"] == outcome.steps
    assert received["snapshot"] is None


def test_evaluator_parser_accepts_shadow_data_collection_arm() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "shadow",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--episode-indices",
            "0",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/shadow",
            "--collect-recovery-roots",
            "--recovery-root-split",
            "DEV",
        ]
    )

    assert args.collect_recovery_roots
    assert args.recovery_root_split == "DEV"
    assert MethodArm(args.method_arm) is MethodArm.SHADOW_LOGIV
    assert args.shadow_monitor_interval_steps == 5
    assert args.shadow_confirmations == 3


def test_repair_overlay_has_a_separate_bounded_repair_action_budget() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "repair-overlay",
            "--method-arm",
            "LOGIV_REPAIR_OVERLAY",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "VERIFIED_BASE_FAILURE",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/repair-overlay",
            "--overlay-repair-max-steps",
            "180",
            "--overlay-monitor-recovery-surface",
            "--overlay-monitor-interval-steps",
            "5",
        ]
    )

    assert args.base_max_steps == 520
    assert args.overlay_repair_max_steps == 180
    config = _run_config(args, (8,), (0,))
    assert config["overlay_repair_max_steps"] == 180
    assert config["overlay_monitor_recovery_surface"] is True
    assert config["overlay_monitor_interval_steps"] == 5


def test_online_logiv_uses_topology_monitor_and_shared_base_budget() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "online-logiv",
            "--method-arm",
            "LOGIV_ONLINE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "ONLINE_VERIFIED_DEVIATION",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "0:10",
            "--episode-indices",
            "0:50",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/online-logiv",
        ]
    )

    assert MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
    assert _validate_shadow_options(args, tuple(range(10))) == {}
    assert _remaining_online_action_budget(args.base_max_steps, 123) == 397
    config = _run_config(args, tuple(range(10)), tuple(range(50)))
    assert config["online_monitor_interval_steps"] == 5
    assert config["online_confirmations"] == 3
    assert config["online_recovery_surface_confirmations"] == 1
    assert config["online_recovery_surface_confirmation_task_ids"] == []
    assert config["online_min_intervention_step"] == 120
    assert config["online_stall_steps"] == 120
    assert args.online_recovery_requires_goal_task_ids == (6,)
    assert config["online_recovery_requires_goal_task_ids"] == [6]
    assert args.online_stall_requires_handempty_task_ids == (8,)
    assert config["online_stall_requires_handempty_task_ids"] == [8]


def test_online_detector_settings_route_task_scoped_safety_gates() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "online-routing",
            "--method-arm",
            "LOGIV_ONLINE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "ONLINE_VERIFIED_DEVIATION",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/online-routing",
            "--online-stall-requires-goal-task-ids",
            "8",
            "--online-stall-ignores-holding-task-ids",
            "4",
            "--online-recovery-surface-confirmations",
            "1",
            "--online-recovery-surface-confirmation-task-ids",
            "8",
        ]
    )

    task6 = _online_detector_settings_for_task(args, task_id=6)
    task8 = _online_detector_settings_for_task(args, task_id=8)
    task4 = _online_detector_settings_for_task(args, task_id=4)
    control = _online_detector_settings_for_task(args, task_id=0)

    assert task6["recovery_requires_achieved_goal"] is True
    assert task6["stall_requires_achieved_goal"] is False
    assert task6["stall_ignores_holding"] is False
    assert task6["stall_requires_handempty"] is False
    assert task8["recovery_requires_achieved_goal"] is False
    assert task8["stall_requires_achieved_goal"] is True
    assert task8["stall_ignores_holding"] is False
    assert task8["stall_requires_handempty"] is True
    assert task8["recovery_surface_confirmation_count"] == 1
    assert task4["stall_ignores_holding"] is True
    assert control["recovery_requires_achieved_goal"] is False
    assert control["stall_requires_achieved_goal"] is False
    assert control["stall_ignores_holding"] is False
    assert control["stall_requires_handempty"] is False
    assert control["recovery_surface_confirmation_count"] is None


def test_online_logiv_shared_budget_rejects_prefix_overrun() -> None:
    with pytest.raises(ValueError, match="exceeds total action budget"):
        _remaining_online_action_budget(520, 521)


def test_recovery_package_is_rebased_on_the_observed_handoff_state() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    handoff = FactSnapshot(
        epoch_id=7,
        true_facts=package.proposal.initial_snapshot.true_facts,
        false_facts=package.proposal.initial_snapshot.false_facts,
        evidence_hash="handoff-evidence",
    )

    rebased = _rebase_package_on_snapshot(package, handoff)

    assert rebased.proposal.epoch_id == 7
    assert rebased.proposal.initial_snapshot == handoff
    assert rebased.problem.initial_state == handoff.true_facts
    assert rebased.problem.initial_false == handoff.false_facts
    assert rebased.frozen_goal == package.frozen_goal


def test_task8_recovery_certification_skips_a_pot_already_on_the_stove() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    moved = Fact("at", ("moka_pot_2", "flat_stove_1_cook_region"))
    old = Fact(
        "at", ("moka_pot_2", "kitchen_table_moka_pot_left_init_region")
    )
    handoff = FactSnapshot(
        epoch_id=0,
        true_facts=(package.proposal.initial_snapshot.true_facts - {old}) | {moved},
        false_facts=(package.proposal.initial_snapshot.false_facts - {moved}) | {old},
        evidence_hash="one-pot-complete",
    )
    rebased = _rebase_package_on_snapshot(package, handoff)

    certified = certify_initial_package(
        rebased,
        handoff,
        episode_id="task8-repair-overlay",
        val_wrapper=ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=frozenset(
            {"pick", "put-down", "place-on", "place-held-on", "turn-on"}
        ),
        repair_bounds=RepairBounds(
            max_edits=5, max_candidates=10000, max_val_calls=20
        ),
        decompose_macro_sources=frozenset({"kitchen_table_recovery_surface"}),
    )

    assert len(certified.plan) == 1
    assert certified.plan[0].arguments[0] == "moka_pot_1"


def test_task8_recovery_certification_can_resume_with_a_pot_already_held() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    held = Fact("holding", ("moka_pot_2",))
    handempty = Fact("handempty")
    old = Fact(
        "at", ("moka_pot_2", "kitchen_table_moka_pot_left_init_region")
    )
    handoff = FactSnapshot(
        epoch_id=0,
        true_facts=(
            package.proposal.initial_snapshot.true_facts - {old, handempty}
        )
        | {held},
        false_facts=(package.proposal.initial_snapshot.false_facts - {held})
        | {old, handempty},
        evidence_hash="held-pot",
    )
    rebased = _rebase_package_on_snapshot(package, handoff)

    certified = certify_initial_package(
        rebased,
        handoff,
        episode_id="task8-held-repair-overlay",
        val_wrapper=ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=frozenset(
            {"pick", "put-down", "place-on", "place-held-on", "turn-on"}
        ),
        repair_bounds=RepairBounds(
            max_edits=5, max_candidates=10000, max_val_calls=20
        ),
        decompose_macro_sources=frozenset({"kitchen_table_recovery_surface"}),
    )

    assert certified.plan[0].schema == "place-held-on"
    assert certified.plan[0].arguments[0] == "moka_pot_2"


def test_recovery_reorders_a_held_second_object_before_recovery_surface_work() -> None:
    package = ScriptedProposalProvider(
        Path("configs/logiv/libero10-scripted-proposals-online-v1.json")
    ).propose(8, epoch_id=0)
    handempty = Fact("handempty")
    held = Fact("holding", ("moka_pot_1",))
    pot1_source = Fact(
        "at", ("moka_pot_1", "kitchen_table_moka_pot_right_init_region")
    )
    pot2_source = Fact(
        "at", ("moka_pot_2", "kitchen_table_moka_pot_left_init_region")
    )
    pot2_recovery = Fact(
        "at", ("moka_pot_2", "kitchen_table_recovery_surface")
    )
    initial = package.proposal.initial_snapshot
    handoff = FactSnapshot(
        epoch_id=0,
        true_facts=(
            initial.true_facts - {handempty, pot1_source, pot2_source}
        )
        | {held, pot2_recovery},
        false_facts=(initial.false_facts - {held, pot2_recovery})
        | {handempty, pot1_source, pot2_source},
        evidence_hash="held-second-object",
    )

    certified = certify_initial_package(
        _rebase_package_on_snapshot(package, handoff),
        handoff,
        episode_id="task8-held-second-reorder",
        val_wrapper=ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=frozenset(
            {"pick", "put-down", "place-on", "place-held-on", "turn-on"}
        ),
        repair_bounds=RepairBounds(
            max_edits=5, max_candidates=10000, max_val_calls=20
        ),
        decompose_macro_sources=frozenset({"kitchen_table_recovery_surface"}),
    )

    assert certified.plan[0].schema == "place-held-on"
    assert certified.plan[0].arguments[0] == "moka_pot_1"


def test_overlay_intervenes_only_on_verified_recovery_surface_facts() -> None:
    recovery = Fact(
        "at", ("moka_pot_1", "kitchen_table_recovery_surface")
    )
    normal = Fact(
        "at", ("moka_pot_2", "kitchen_table_moka_pot_left_init_region")
    )
    snapshot = FactSnapshot(
        epoch_id=12,
        true_facts=frozenset({normal, recovery}),
        false_facts=frozenset(),
        evidence_hash="verified-deviation",
    )

    assert _verified_recovery_surface_facts(snapshot) == (recovery,)
    assert _verified_recovery_surface_facts(
        replace(snapshot, true_facts=frozenset({normal}))
    ) == ()


def test_place_effect_stabilization_can_be_scoped_to_task4() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "task-scoped-place-stabilization",
            "--method-arm",
            "FULL_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/task-scoped-place-stabilization",
            "--place-effect-stabilization-steps",
            "10",
            "--place-effect-stabilization-task-ids",
            "4",
        ]
    )

    assert args.place_effect_stabilization_task_ids == (4,)
    assert _place_effect_stabilization_steps_for_task(args, task_id=4) == 10
    assert _place_effect_stabilization_steps_for_task(args, task_id=5) == 0


def test_fast_place_confirmation_can_be_scoped_to_task5() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "task-scoped-place-confirmation",
            "--method-arm",
            "FULL_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/task-scoped-place-confirmation",
            "--place-effect-confirmation-steps",
            "1",
            "--place-effect-confirmation-task-ids",
            "5",
        ]
    )

    assert args.place_effect_confirmation_task_ids == (5,)
    assert _place_effect_confirmation_steps_for_task(args, task_id=5) == 1
    assert _place_effect_confirmation_steps_for_task(args, task_id=4) is None


def test_post_stop_reobservation_can_be_scoped_to_task9() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "task-scoped-reobservation",
            "--method-arm",
            "FULL_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/task-scoped-reobservation",
            "--post-stop-grounding-reobservation-steps",
            "10",
            "--post-stop-grounding-reobservation-task-ids",
            "9",
        ]
    )

    assert args.post_stop_grounding_reobservation_task_ids == (9,)
    assert _post_stop_reobservation_steps_for_task(args, task_id=9) == 10
    assert _post_stop_reobservation_steps_for_task(args, task_id=3) == 0


def test_run_config_hashes_resolved_extended_configs(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.json"
    coverage = tmp_path / "coverage.json"
    proposal.write_text(
        '{"extends":"' + str(Path("configs/logiv/libero10-scripted-proposals.json").resolve()) + '"}',
        encoding="utf-8",
    )
    coverage.write_text(
        '{"extends":"' + str(Path("configs/logiv/libero10-coverage.json").resolve()) + '"}',
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--run-id",
            "resolved-config-hash",
            "--method-arm",
            "BASE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            str(tmp_path / "output"),
            "--proposal-config",
            str(proposal),
            "--coverage-manifest",
            str(coverage),
        ]
    )

    config = _run_config(args, (0,), (0,))

    assert config["proposal_config_sha256"] == resolved_json_sha256(proposal)
    assert config["coverage_manifest_sha256"] == resolved_json_sha256(coverage)


def test_run_config_freezes_shadow_collection_contract() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "shadow-config",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--episode-indices",
            "0",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/shadow-config",
            "--collect-recovery-roots",
        ]
    )

    contracts = _validate_shadow_options(args, (8,))
    config = _run_config(args, (8,), (0,))

    assert contracts[8].task_id == 8
    assert config["collect_recovery_roots"] is True
    assert config["recovery_root_split"] == "DEV"
    assert config["shadow_monitor_interval_steps"] == 5
    assert config["shadow_confirmations"] == 3
    assert config["shadow_monitor_contract"] == str(args.shadow_monitor_contract)
    assert config["shadow_monitor_contract_sha256"] == contracts[8].contract_sha256
    assert config["shadow_monitor_contract_sha256_by_task"] == {
        "8": contracts[8].contract_sha256
    }


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--method-arm", "BASE"), "SHADOW_LOGIV"),
        (("--recovery-root-split", "TRAIN"), "DEV"),
    ],
)
def test_phase0_recovery_collection_rejects_unsafe_scope(extra, message) -> None:
    values = [
        "--run-id",
        "bad-shadow",
        "--method-arm",
        "SHADOW_LOGIV",
        "--goal-mode",
        "METADATA_ASSISTED",
        "--deviation-mode",
        "NOMINAL",
        "--oracle-grounding",
        "--development-only",
        "--port",
        "8010",
        "--output-dir",
        "/tmp/bad-shadow",
        "--collect-recovery-roots",
    ]
    option, value = extra
    if option in values:
        values[values.index(option) + 1] = value
    else:
        values.extend((option, value))
    args = _parser().parse_args(values)

    with pytest.raises(ValueError, match=message):
        _validate_shadow_options(args, (8,))


def test_shadow_monitor_cli_cannot_override_frozen_detector_semantics() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "bad-monitor",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/bad-monitor",
            "--shadow-monitor-interval-steps",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="interval"):
        _validate_shadow_options(args, (8,))


def test_phase0_collection_requires_development_mode() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "bad-holdout",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--prompt-locked",
            "--task-ids",
            "8",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/bad-holdout",
            "--collect-recovery-roots",
        ]
    )

    with pytest.raises(ValueError, match="development-only"):
        _validate_shadow_options(args, (8,))


def test_shadow_artifacts_render_step0_containment_as_not_attempted() -> None:
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=1,
        inference_requests=1,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[np.zeros(7)],
        shadow_calls=2,
        shadow_errors=1,
        shadow_failure_records=(
            ShadowFailureRecord(0, "INPUT_COPY", "MemoryError"),
        ),
        shadow_wall_seconds=0.25,
        shadow_parity_valid=True,
    )
    runtime = ShadowRuntime(None, None, None, ShadowRuntimeCounters())
    runtime.deviation_trace.append(
        {
            "certificate_state": "STALE",
            "deviation_status": "CONFIRMED_DEVIATION",
            "evidence_kinds": ["ATTEMPTED_EFFECT_TIMEOUT"],
            "policy_step": 15,
            "signature": ["effect:(at object target)"],
            "trigger_class": "ATTEMPTED_EFFECT_TIMEOUT_STABLE",
        }
    )

    proposal, monitor, compute, accounting = _shadow_artifact_payloads(
        outcome, runtime
    )

    assert proposal["status"] == "NOT_ATTEMPTED"
    assert proposal["request_count"] == 0
    assert proposal["reason"] == "INPUT_COPY:MemoryError"
    assert accounting["initial_proposal_status"] == "NOT_ATTEMPTED"
    assert accounting["initial_proposal_reason_code"] == "INPUT_COPY:MemoryError"
    assert monitor["callback_errors"] == 1
    assert monitor["deviation_decisions"] == runtime.deviation_trace
    assert monitor["attempt_records"] == 0
    assert monitor["evidence_records"] == 0
    assert monitor["terminal_topology_status"] is None
    assert monitor["terminal_topology_success"] is None
    assert monitor["base_self_recovered_after_confirmed_deviation"] is False
    assert monitor["aggregate_errors"] == 1
    assert compute["base_policy_requests"] == 1


def test_shadow_error_aggregate_sums_each_sole_owner_once() -> None:
    metrics = SimpleNamespace(
        snapshot_calls=7,
        snapshot_errors=2,
        event_tracker_errors=3,
        evidence_overflows=4,
        trigger_callback_errors=5,
        anomaly_candidates=1,
        confirmed_deviations=1,
        stale_certificates=0,
    )
    runtime = ShadowRuntime(
        None,
        None,
        SimpleNamespace(
            metrics=metrics,
            action_event_tracker=SimpleNamespace(
                attempt_record_count=11,
                evidence_record_count=4,
            ),
        ),
        ShadowRuntimeCounters(
            root_count=2,
            root_write_errors=6,
            proposal_callback_errors=7,
            provenance_errors=8,
            trace_errors=9,
        ),
    )
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=0,
        inference_requests=0,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[],
        shadow_errors=1,
    )

    _, monitor, _, accounting = _shadow_artifact_payloads(outcome, runtime)

    assert monitor["trace_errors"] == 9
    assert monitor["attempt_records"] == 11
    assert monitor["evidence_records"] == 4
    assert monitor["aggregate_errors"] == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
    assert accounting["shadow_monitor_errors"] == monitor["aggregate_errors"]


def test_shadow_artifacts_distinguish_confirmed_event_from_terminal_recovery() -> None:
    runtime = ShadowRuntime(
        None,
        None,
        SimpleNamespace(
            metrics=SimpleNamespace(confirmed_deviations=1),
            action_event_tracker=SimpleNamespace(
                attempt_record_count=1,
                evidence_record_count=1,
            ),
        ),
        ShadowRuntimeCounters(),
        state_trace=[
            {
                "policy_step": 12,
                "nodes": [
                    {"node_id": "INIT", "status": "COMPLETED"},
                    {"node_id": "GOAL", "status": "COMPLETED"},
                ],
            }
        ],
    )
    outcome = EpisodeOutcome(
        success=True,
        done=True,
        check_success=True,
        steps=12,
        inference_requests=3,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[],
    )

    _, monitor, _, _ = _shadow_artifact_payloads(outcome, runtime)

    assert monitor["terminal_topology_status"] == "COMPLETED"
    assert monitor["terminal_topology_success"] is True
    assert monitor["base_self_recovered_after_confirmed_deviation"] is True


def test_shadow_graph_artifact_is_written_only_for_accepted_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = getattr(evaluator_script, "_write_shadow_graph_artifact", None)
    assert writer is not None
    graph = SimpleNamespace(graph_hash="g" * 64)
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=graph,
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
        state_trace=[{"policy_step": 0}],
    )
    writes = []
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: writes.append((path, payload)),
    )

    assert writer(tmp_path, runtime) is graph
    assert writes == [
        (
            tmp_path / "graph.json",
            {"graph": graph, "state_trace": [{"policy_step": 0}]},
        )
    ]

    runtime.initial_proposal = replace(
        runtime.initial_proposal,
        status=InitialProposalStatus.REJECTED,
        validation=None,
    )
    assert writer(tmp_path, runtime) is None
    assert len(writes) == 1


def test_shadow_graph_artifact_write_failure_is_contained_and_accounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = SimpleNamespace(graph_hash="g" * 64)
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=graph,
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert evaluator_script._write_shadow_graph_artifact(tmp_path, runtime) is None
    assert runtime.counters.trace_errors == 1
    outcome = EpisodeOutcome(
        success=True,
        done=True,
        check_success=True,
        steps=1,
        inference_requests=1,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[np.zeros(7)],
    )
    _, monitor, _, accounting = _shadow_artifact_payloads(outcome, runtime)
    assert monitor["trace_errors"] == 1
    assert accounting["shadow_monitor_errors"] == 1


def test_shadow_outcome_none_fallback_counts_contained_graph_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounting_for = getattr(evaluator_script, "_shadow_record_accounting", None)
    assert accounting_for is not None
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=SimpleNamespace(graph_hash="g" * 64),
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert evaluator_script._write_shadow_graph_artifact(tmp_path, runtime) is None
    accounting = accounting_for(
        outcome=None,
        runtime=runtime,
        exception_text="RuntimeError: Base evaluator failed",
        base_policy_requests=0,
    )

    assert accounting["initial_proposal_status"] == "ACCEPTED"
    assert accounting["initial_proposal_requests"] == 1
    assert accounting["shadow_monitor_errors"] == 1


@pytest.mark.parametrize(
    ("runtime", "expected_status", "expected_requests", "expected_reason"),
    [
        (None, "NOT_ATTEMPTED", 0, "ROLLOUT:RuntimeError"),
        (
            ShadowRuntime(
                InitialProposalResult(
                    status=InitialProposalStatus.REJECTED,
                    provider="test-provider",
                    request_count=1,
                    elapsed_seconds=0.1,
                    package=None,
                    validation=None,
                    reason="ValueError: rejected",
                ),
                None,
                None,
                ShadowRuntimeCounters(trace_errors=2),
            ),
            "REJECTED",
            1,
            "ValueError",
        ),
    ],
)
def test_shadow_outcome_none_fallback_preserves_proposal_accounting(
    runtime: ShadowRuntime | None,
    expected_status: str,
    expected_requests: int,
    expected_reason: str,
) -> None:
    accounting_for = getattr(evaluator_script, "_shadow_record_accounting", None)
    assert accounting_for is not None

    accounting = accounting_for(
        outcome=None,
        runtime=runtime,
        exception_text="RuntimeError: Base evaluator failed",
        base_policy_requests=0,
    )

    assert accounting["initial_proposal_status"] == expected_status
    assert accounting["initial_proposal_requests"] == expected_requests
    assert accounting["initial_proposal_reason_code"] == expected_reason
    assert accounting["shadow_monitor_errors"] == (
        0 if runtime is None else runtime.counters.trace_errors
    )


def test_shadow_compute_seconds_separate_initial_proposal_from_monitor() -> None:
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.REJECTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=None,
            validation=None,
            reason="ValueError: rejected",
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=0,
        inference_requests=0,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[],
        shadow_wall_seconds=0.25,
    )

    _, monitor, compute, accounting = _shadow_artifact_payloads(outcome, runtime)

    assert monitor["callback_seconds"] == 0.25
    assert compute["initial_proposal_seconds"] == 0.1
    assert compute["shadow_monitor_seconds"] == pytest.approx(0.15)
    assert accounting["shadow_monitor_seconds"] == pytest.approx(0.15)


def test_base_rollout_is_one_physical_attempt_not_one_attempt_per_control_step() -> None:
    assert _base_physical_attempts(444) == 1
    assert _base_physical_attempts(0) == 0
    with pytest.raises(ValueError, match="steps"):
        _base_physical_attempts(-1)


def test_each_episode_replaces_and_closes_the_previous_simulator_environment() -> None:
    class FakeEnv:
        def __init__(self, marker):
            self.marker = marker
            self.closed = False

        def close(self):
            self.closed = True

    created = []

    def factory(**kwargs):
        env = FakeEnv(kwargs)
        created.append(env)
        return env

    old = FakeEnv("old")
    first = _replace_episode_environment(
        old,
        factory=factory,
        bddl_file="task.bddl",
    )
    second = _replace_episode_environment(
        first,
        factory=factory,
        bddl_file="task.bddl",
    )

    assert old.closed is True
    assert first.closed is True
    assert second is created[1]
    assert second.marker == {
        "bddl_file_name": "task.bddl",
        "camera_heights": 256,
        "camera_widths": 256,
    }


def test_resume_preserves_orphan_artifacts_and_allocates_a_generation(tmp_path: Path) -> None:
    first = _allocate_episode_artifact_dir(tmp_path, task_id=8, episode_idx=3)
    (first / "partial.json").write_text("interrupted")

    resumed = _allocate_episode_artifact_dir(tmp_path, task_id=8, episode_idx=3)

    assert first.name == "episode_003"
    assert resumed.name == "episode_003_resume_001"
    assert (first / "partial.json").read_text() == "interrupted"
    assert (resumed / "resume.json").exists()


def test_initial_certification_preserves_task8_parallel_graph() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=5)
    wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)
    bounds = RepairBounds(max_edits=3, max_candidates=1000, max_val_calls=20)

    certified = certify_initial_package(
        package,
        package.proposal.initial_snapshot,
        episode_id="test-task8-episode0",
        val_wrapper=wrapper,
        allowed_schemas=frozenset({"pick", "put-down", "place-on", "place-held-on"}),
        repair_bounds=bounds,
        decompose_macro_sources=frozenset({"kitchen_table_recovery_surface"}),
    )

    assert certified.graph.action_layer_width() == 2
    first, second = certified.graph.canonical_agenda
    assert certified.graph.edge(first, second) is None
    assert certified.graph.edge(second, first) is None
    assert certified.certificate.certificate_hash == certified.graph.certificate_hash
    assert certified.repair_operator.decompose_macro_sources == frozenset(
        {"kitchen_table_recovery_surface"}
    )


def test_initial_and_repair_share_one_pddl_planner() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=5)
    wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)
    planner = PddlPlanner(
        wrapper,
        allowed_schemas=frozenset(
            {"pick", "put-down", "place-on", "place-held-on"}
        ),
        bounds=RepairBounds(max_edits=3, max_candidates=1000, max_val_calls=20),
    )

    certified = certify_initial_package(
        package,
        package.proposal.initial_snapshot,
        episode_id="test-task8-shared-planner",
        val_wrapper=wrapper,
        allowed_schemas=frozenset(
            {"pick", "put-down", "place-on", "place-held-on"}
        ),
        repair_bounds=RepairBounds(
            max_edits=3, max_candidates=1000, max_val_calls=20
        ),
        planner=planner,
    )

    assert certified.planner is planner
    assert certified.repair_operator is planner
    assert certified.certificate is not None
    assert (
        certified.graph.certificate_hash
        == certified.certificate.certificate_hash
    )


def test_schema_only_arm_never_reuses_full_certificate() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=5)
    certified = certify_initial_package(
        package,
        package.proposal.initial_snapshot,
        episode_id="test-task8-episode0",
        val_wrapper=ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=frozenset({"pick", "put-down", "place-on", "place-held-on"}),
        repair_bounds=RepairBounds(max_edits=3, max_candidates=1000, max_val_calls=20),
    )
    schema_graph = SchemaOnlyCausalDagCompiler().compile(
        certified.problem,
        certified.plan,
        certified.occurrence_sidecar,
        certified.certificate.context,
    )
    assert schema_graph.certificate_hash != certified.certificate.certificate_hash
    assert schema_graph.action_layer_width() == 2


def test_global_repair_ablation_discards_causal_slice() -> None:
    class CapturingRepair:
        def __init__(self):
            self.slice = "unset"

        def repair(self, *args, causal_slice=None, **kwargs):
            self.slice = causal_slice
            return "result"

    wrapped = CapturingRepair()
    result = GlobalRepairOperator(wrapped).repair("problem", causal_slice="localized")
    assert result == "result"
    assert wrapped.slice is None


def test_evaluation_contract_requires_explicit_modes_oracle_and_locked_holdout() -> None:
    base = EvaluationContract(
        method_arm=MethodArm.FULL_LOGIV,
        goal_mode=GoalMode.METADATA_ASSISTED,
        deviation_mode="NOMINAL",
        oracle_grounding=True,
        development_only=True,
        prompt_locked=False,
        task_ids=(8,),
        episode_indices=(0, 1),
    )
    base.validate()

    with pytest.raises(EvaluationContractError, match="oracle grounding acknowledgement"):
        replace(base, oracle_grounding=False).validate()
    replace(base, method_arm=MethodArm.BASE, oracle_grounding=False).validate()
    with pytest.raises(EvaluationContractError, match="locked prompt"):
        replace(base, development_only=False, prompt_locked=False).validate()
    with pytest.raises(EvaluationContractError, match="not supported by the no-API provider"):
        replace(base, goal_mode=GoalMode.GOAL_PREDICTION).validate()
    with pytest.raises(EvaluationContractError, match="frozen 10-task manifest"):
        replace(base, task_ids=(10,)).validate()


def test_evaluation_contract_accepts_explicit_gpt4o_without_oracle() -> None:
    contract = EvaluationContract(
        method_arm=MethodArm.LOGIV_ONLINE,
        goal_mode=GoalMode.METADATA_ASSISTED,
        deviation_mode="NOMINAL",
        oracle_grounding=False,
        development_only=True,
        prompt_locked=False,
        task_ids=(0,),
        episode_indices=(0,),
        perception_backend="gpt4o",
    )
    contract.validate()
    with pytest.raises(EvaluationContractError, match="must not enable oracle"):
        replace(contract, oracle_grounding=True).validate()
