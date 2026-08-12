from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pi05_libero_repro.logiv.robotwin import (
    ROBOTWIN_TASKS,
    RobotwinEpisodeController,
    RobotwinFactGrounder,
    RobotwinPddlPlanner,
    RECOVERY_POLICY_PROMPTS,
    SCENE_BOUND_REPAIR_TASKS,
    TruthValue,
    bind_canonical_policy_prompt,
    extract_robotwin_images,
)


REAL_VAL = Path("artifacts/tools/val-ubuntu22/Validate").resolve()


def test_registry_is_exactly_the_frozen_ten_task_protocol() -> None:
    assert tuple(ROBOTWIN_TASKS) == (
        "handover_block",
        "open_microwave",
        "place_dual_shoes",
        "stamp_seal",
        "blocks_ranking_size",
        "move_can_pot",
        "turn_switch",
        "stack_blocks_three",
        "stack_bowls_three",
        "beat_block_hammer",
    )
    for task in ROBOTWIN_TASKS.values():
        assert task.stages
        assert len({stage.fact for stage in task.stages}) == len(task.stages)
        assert all(stage.policy_prompt and stage.observer_question for stage in task.stages)


def test_capability_routed_tasks_keep_dag_but_use_training_style_prompt() -> None:
    task = ROBOTWIN_TASKS["stack_blocks_three"]

    bound = bind_canonical_policy_prompt(task)

    assert len(bound.stages) == len(task.stages)
    assert len({stage.policy_prompt for stage in bound.stages}) == 1
    assert "stack blue on green" in bound.stages[0].policy_prompt
    assert [stage.fact for stage in bound.stages] == [
        stage.fact for stage in task.stages
    ]


def test_planner_replans_from_confirmed_prefix_and_real_val_certifies() -> None:
    task = ROBOTWIN_TASKS["stack_blocks_three"]
    planner = RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5)

    initial = {stage.fact: TruthValue.FALSE for stage in task.stages}
    complete = planner.plan(task, initial)

    assert complete.valid
    assert complete.searched
    assert "Plan valid" in complete.val_stdout
    assert tuple(action.stage_index for action in complete.actions) == tuple(
        range(len(task.stages))
    )
    assert len(complete.certificate_sha256) == 64

    progressed = dict(initial)
    progressed[task.stages[0].fact] = TruthValue.TRUE
    suffix = planner.plan(task, progressed)

    assert suffix.valid
    assert tuple(action.stage_index for action in suffix.actions) == tuple(
        range(1, len(task.stages))
    )

    later_visible = dict(initial)
    later_visible[task.stages[1].fact] = TruthValue.TRUE
    inferred_suffix = planner.plan(task, later_visible)
    assert tuple(action.stage_index for action in inferred_suffix.actions) == (2,)


def test_robotwin_image_extraction_has_front_and_both_wrists() -> None:
    obs = {
        "observation": {
            "head_camera": {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.ones((4, 5, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.full((4, 5, 3), 2, dtype=np.uint8)},
        }
    }

    images = extract_robotwin_images(obs)

    assert len(images) == 3
    assert [int(image[0, 0, 0]) for image in images] == [0, 1, 2]


class _Client:
    def __init__(self) -> None:
        self.kwargs = None

    def complete_json(self, **kwargs):
        self.kwargs = kwargs
        return {
            "facts": [
                {"name": "microwave-open", "value": "FALSE"},
                {"name": "handle-grasped", "value": "UNKNOWN"},
            ]
        }


def test_grounder_restricts_gpt4o_to_visual_fact_confirmation() -> None:
    client = _Client()
    task = ROBOTWIN_TASKS["open_microwave"]
    obs = {
        "observation": {
            name: {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)}
            for name in ("head_camera", "right_camera", "left_camera")
        }
    }

    result = RobotwinFactGrounder(client).observe(task, obs, epoch=3)

    assert result["microwave-open"] is TruthValue.FALSE
    assert result["handle-grasped"] is TruthValue.UNKNOWN
    assert client.kwargs["purpose"] == "state_gate"
    assert "Do not plan" in client.kwargs["system"]
    assert len(client.kwargs["images"]) == 3
    assert set(client.kwargs["schema"]["properties"]["facts"]["items"]
               ["properties"]["name"]["enum"]) == {
        stage.fact for stage in task.stages
    }


class _SequenceGrounder:
    def __init__(self, values):
        self.values = iter(values)
        self.epochs = []

    def observe(self, task, observation, *, epoch):
        self.epochs.append(epoch)
        return next(self.values)


def test_controller_observes_before_first_dispatch_and_after_every_chunk() -> None:
    task = ROBOTWIN_TASKS["open_microwave"]
    false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    first_done = dict(false)
    first_done[task.stages[0].fact] = TruthValue.TRUE
    complete = {stage.fact: TruthValue.TRUE for stage in task.stages}
    grounder = _SequenceGrounder([false, first_done, complete])
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
    )
    prompts = []

    outcome = controller.run(
        initial_observation="obs-0",
        dispatch=lambda prompt: prompts.append(prompt) or f"obs-{len(prompts)}",
        native_success=lambda: len(prompts) == 2,
        budget_exhausted=lambda: False,
    )

    assert outcome.success
    assert prompts == [stage.policy_prompt for stage in task.stages]
    assert grounder.epochs == [0, 1, 2]
    assert all(event.plan.valid for event in outcome.events)
    assert all(event.control_mode == "DAG_EXECUTION" for event in outcome.events)


def test_full_dag_control_dispatches_ready_node_before_task_prompt() -> None:
    task = ROBOTWIN_TASKS["open_microwave"]
    false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    first_done = dict(false)
    first_done[task.stages[0].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder([false, first_done, first_done])
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 2,
        budget_exhausted=lambda: False,
        base_prompt="Open the gray microwave using the left arm.",
        dag_from_start=True,
    )

    assert outcome.success
    assert prompts == [
        RECOVERY_POLICY_PROMPTS[task.name][0],
        RECOVERY_POLICY_PROMPTS[task.name][1],
    ]
    assert all(event.control_mode == "DAG_EXECUTION" for event in outcome.events)


def test_controller_retries_false_stage_without_asking_gpt_to_repair() -> None:
    task = ROBOTWIN_TASKS["turn_switch"]
    false = {task.goal_fact: TruthValue.FALSE}
    true = {task.goal_fact: TruthValue.TRUE}
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        _SequenceGrounder([false, false, true]),
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 2,
        budget_exhausted=lambda: False,
    )

    assert outcome.success
    assert prompts == [task.stages[0].policy_prompt] * 2


def test_controller_latches_confirmed_milestones_instead_of_regressing() -> None:
    task = ROBOTWIN_TASKS["handover_block"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    left_confirmed = dict(all_false)
    left_confirmed[task.stages[0].fact] = TruthValue.TRUE
    transient_disappeared = dict(all_false)
    final = dict(all_false)
    final[task.goal_fact] = TruthValue.TRUE
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        _SequenceGrounder(
            [all_false, left_confirmed, transient_disappeared, final]
        ),
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 3,
        budget_exhausted=lambda: False,
    )

    assert outcome.success
    assert prompts[0] == task.stages[0].policy_prompt
    assert prompts[1:] == [task.stages[1].policy_prompt] * 2


def test_repair_reopens_a_dropped_transient_grasp_after_two_observations() -> None:
    task = ROBOTWIN_TASKS["handover_block"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    left_confirmed = dict(all_false)
    left_confirmed[task.stages[0].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder(
        [all_false, left_confirmed, all_false, all_false, all_false, all_false]
    )
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=1,
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 5,
        budget_exhausted=lambda: False,
        base_prompt="Transfer the red block and put it on the blue pad.",
    )

    assert outcome.success
    assert task.stages[0].transient
    assert prompts == [
        "Transfer the red block and put it on the blue pad.",
        "Transfer the red block and put it on the blue pad.",
        RECOVERY_POLICY_PROMPTS[task.name][1],
        RECOVERY_POLICY_PROMPTS[task.name][1],
        RECOVERY_POLICY_PROMPTS[task.name][0],
    ]


def test_repair_keeps_confirmed_persistent_geometry_latched() -> None:
    task = ROBOTWIN_TASKS["stack_blocks_three"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    base_confirmed = dict(all_false)
    base_confirmed[task.stages[0].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder(
        [all_false, base_confirmed, base_confirmed, all_false, all_false, all_false]
    )
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=1,
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 5,
        budget_exhausted=lambda: False,
        base_prompt="Stack the three blocks.",
    )

    assert outcome.success
    assert prompts[-1] == RECOVERY_POLICY_PROMPTS[task.name][1]


def test_monitored_base_prefix_keeps_the_original_scene_prompt() -> None:
    task = ROBOTWIN_TASKS["open_microwave"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    first_done = dict(all_false)
    first_done[task.stages[0].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder(
        [all_false, first_done, first_done, first_done]
    )
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=2,
    )
    prompts = []
    original = "Open the gray microwave using the left arm."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 3,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts == [original, original, original]
    assert all(event.control_mode == "BASE_MONITORED" for event in outcome.events)


def test_stable_false_frontier_preserves_scene_specific_repair_prompt() -> None:
    task = ROBOTWIN_TASKS["turn_switch"]
    false = {task.goal_fact: TruthValue.FALSE}
    grounder = _SequenceGrounder([false, false, false, false])
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=2,
    )
    prompts = []
    original = "Use the left arm to press the flat tan switch."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 3,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts[:2] == [original, original]
    assert prompts[2] == original
    assert [event.control_mode for event in outcome.events] == [
        "BASE_MONITORED",
        "BASE_MONITORED",
        "REPAIR",
        "REPAIR",
    ]


def test_dispatch_receives_control_mode_for_repair_chunk_sizing() -> None:
    task = ROBOTWIN_TASKS["turn_switch"]
    false = {task.goal_fact: TruthValue.FALSE}
    grounder = _SequenceGrounder([false] * 4)
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=2,
    )
    calls = []
    original = "Use the left arm to press the flat tan switch."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: None,
        dispatch_with_mode=lambda prompt, mode: calls.append((prompt, mode)),
        native_success=lambda: len(calls) == 3,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert calls == [
        (original, "BASE_MONITORED"),
        (original, "BASE_MONITORED"),
        (original, "REPAIR"),
    ]


def test_dispatch_receives_active_frontier_for_macro_continuation() -> None:
    task = ROBOTWIN_TASKS["handover_block"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    left_done = dict(all_false)
    left_done[task.stages[0].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder([all_false, left_done, left_done])
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=1,
    )
    calls = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: None,
        dispatch_with_context=lambda prompt, mode, active: calls.append(
            (prompt, mode, active)
        ),
        native_success=lambda: len(calls) == 2,
        budget_exhausted=lambda: False,
        base_prompt="Transfer the red block.",
    )

    assert outcome.success
    assert [active for _, _, active in calls] == [0, 1]
    assert [mode for _, mode, _ in calls] == ["BASE_MONITORED", "BASE_MONITORED"]


def test_base_protection_window_monitors_but_delays_prompt_replacement() -> None:
    task = ROBOTWIN_TASKS["turn_switch"]
    false = {task.goal_fact: TruthValue.FALSE}
    grounder = _SequenceGrounder([false] * 5)
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=1,
        min_base_dispatches=3,
    )
    prompts = []
    original = "Use the left arm to press the flat tan switch."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 4,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts == [original] * 4
    assert [event.control_mode for event in outcome.events[:3]] == [
        "BASE_MONITORED"
    ] * 3
    assert outcome.events[3].control_mode == "REPAIR"


def test_recovery_prompts_preserve_required_arm_and_release_constraints() -> None:
    assert "right arm" in RECOVERY_POLICY_PROMPTS["handover_block"][2]
    assert "left arm" in RECOVERY_POLICY_PROMPTS["open_microwave"][1]
    assert "release" in RECOVERY_POLICY_PROMPTS["stamp_seal"][1]
    assert "release" in RECOVERY_POLICY_PROMPTS["stack_blocks_three"][2]


def test_bowl_repair_preserves_randomized_scene_description() -> None:
    assert "stack_bowls_three" in SCENE_BOUND_REPAIR_TASKS


def test_ranking_and_bowl_dags_follow_pi05_training_order() -> None:
    ranking = ROBOTWIN_TASKS["blocks_ranking_size"]
    bowls = ROBOTWIN_TASKS["stack_bowls_three"]

    assert [stage.fact for stage in ranking.stages] == [
        "large-block-left",
        "medium-block-center",
        "blocks-ranked-large-to-small",
    ]
    assert "largest block" in RECOVERY_POLICY_PROMPTS[ranking.name][0]
    assert "medium block" in RECOVERY_POLICY_PROMPTS[ranking.name][1]
    assert RECOVERY_POLICY_PROMPTS[ranking.name][2] == (
        "Arrange blocks large block, medium block, and small block in decreasing size order."
    )
    assert "largest bowl" in RECOVERY_POLICY_PROMPTS[bowls.name][0]
    assert "medium bowl" in RECOVERY_POLICY_PROMPTS[bowls.name][1]
    assert "smallest bowl" in RECOVERY_POLICY_PROMPTS[bowls.name][2]


def test_stage_specific_threshold_protects_handoff_then_repairs_placement() -> None:
    task = ROBOTWIN_TASKS["handover_block"]
    all_false = {stage.fact: TruthValue.FALSE for stage in task.stages}
    left_done = dict(all_false)
    left_done[task.stages[0].fact] = TruthValue.TRUE
    right_done = dict(left_done)
    right_done[task.stages[1].fact] = TruthValue.TRUE
    grounder = _SequenceGrounder(
        [all_false, left_done, left_done, right_done, right_done, right_done]
    )
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=4,
        stage_stall_observations=(4, 4, 1),
    )
    prompts = []
    original = "Transfer the red block and put it on the blue pad."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 5,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts[:4] == [original] * 4
    assert prompts[4] == RECOVERY_POLICY_PROMPTS[task.name][2]


def test_stage_specific_threshold_shape_is_validated() -> None:
    task = ROBOTWIN_TASKS["handover_block"]
    planner = RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5)
    grounder = _SequenceGrounder([])

    with pytest.raises(ValueError, match="match task stages"):
        RobotwinEpisodeController(
            task, planner, grounder, stage_stall_observations=(1, 2)
        )


def test_visual_goal_cannot_override_native_failure() -> None:
    task = ROBOTWIN_TASKS["turn_switch"]
    visual_true = {task.goal_fact: TruthValue.TRUE}
    grounder = _SequenceGrounder([visual_true, visual_true])
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
    )
    prompts = []

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 1,
        budget_exhausted=lambda: False,
        base_prompt="Click the switch.",
    )

    assert outcome.success
    assert prompts == ["Click the switch."]


def test_persistent_visual_goal_native_conflict_enters_goal_repair() -> None:
    task = ROBOTWIN_TASKS["move_can_pot"]
    visual_true = {task.goal_fact: TruthValue.TRUE}
    grounder = _SequenceGrounder([visual_true] * 4)
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=2,
    )
    prompts = []
    original = "Move the sauce can beside the cooking pot."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 3,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts[0] == original
    assert prompts[1] == original
    assert outcome.events[1].control_mode == "REPAIR"
    assert outcome.events[1].active_stage_index == 0


def test_visual_goal_conflict_reopens_goal_immediately_during_repair() -> None:
    task = ROBOTWIN_TASKS["move_can_pot"]
    false = {task.goal_fact: TruthValue.FALSE}
    visual_true = {task.goal_fact: TruthValue.TRUE}
    grounder = _SequenceGrounder(
        [false, false, false, visual_true, visual_true]
    )
    controller = RobotwinEpisodeController(
        task,
        RobotwinPddlPlanner(REAL_VAL, timeout_seconds=5),
        grounder,
        base_stall_observations=2,
    )
    prompts = []
    original = "Move the sauce can beside the cooking pot."

    outcome = controller.run(
        initial_observation=None,
        dispatch=lambda prompt: prompts.append(prompt),
        native_success=lambda: len(prompts) == 4,
        budget_exhausted=lambda: False,
        base_prompt=original,
    )

    assert outcome.success
    assert prompts == [
        original,
        original,
        original,
        original,
    ]
    assert outcome.events[3].control_mode == "REPAIR"
    assert outcome.events[3].active_stage_index == 0


def test_grounder_compares_current_views_to_episode_initial_views() -> None:
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def complete_json(self, **kwargs):
            self.calls.append(kwargs)
            return {"facts": [{"name": "switch-activated", "value": "UNKNOWN"}]}

    client = RecordingClient()
    grounder = RobotwinFactGrounder(client)
    task = ROBOTWIN_TASKS["turn_switch"]

    def obs(pixel):
        return {
            "observation": {
                name: {
                    "rgb": np.full((4, 5, 3), pixel, dtype=np.uint8)
                }
                for name in ("head_camera", "right_camera", "left_camera")
            }
        }

    grounder.observe(task, obs(1), epoch=0)
    grounder.observe(task, obs(2), epoch=1)

    assert len(client.calls[0]["images"]) == 3
    assert len(client.calls[1]["images"]) == 6
    assert "first three images are the initial" in client.calls[1]["text"]
