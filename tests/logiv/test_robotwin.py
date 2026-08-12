from __future__ import annotations

from pathlib import Path

import numpy as np

from pi05_libero_repro.logiv.robotwin import (
    ROBOTWIN_TASKS,
    RobotwinEpisodeController,
    RobotwinFactGrounder,
    RobotwinPddlPlanner,
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
