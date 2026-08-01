from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pi05_libero_repro.logiv.controller import (
    DispatchStatus,
    ExecutorStatus,
    GroundingStatus,
)
from pi05_libero_repro.logiv.domain import FixedDomain
from pi05_libero_repro.logiv.libero_adapter import (
    LiberoOracleGrounder,
    LiberoObservationStore,
    Pi05MacroExecutor,
    SimulatorSafetySupervisor,
    TaskBinding,
    monitored_fact_universe,
)
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    GoalMode,
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider


ROOT = Path(__file__).resolve().parents[2]


class FakeObject:
    def __init__(self, name: str) -> None:
        self.contact_geoms = [f"{name}_geom"]


class FakeRobot:
    gripper = object()


class FakeInnerEnv:
    def __init__(self) -> None:
        self.relations: dict[tuple[str, ...], bool] = {}
        self.held: set[str] = set()
        self.objects_dict = {
            "moka_pot_1": FakeObject("moka_pot_1"),
            "moka_pot_2": FakeObject("moka_pot_2"),
        }
        self.robots = [FakeRobot()]

    def _eval_predicate(self, state) -> bool:
        key = tuple(str(item) for item in state)
        if key not in self.relations:
            raise KeyError(key)
        return self.relations[key]

    def _check_grasp(self, gripper, geoms) -> bool:
        del gripper
        object_name = str(geoms[0]).removesuffix("_geom")
        return object_name in self.held


class FakeEnv:
    def __init__(self) -> None:
        self.env = FakeInnerEnv()
        self.actions: list[np.ndarray] = []
        self.step_hook = None
        self.obs = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 0.5]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        }

    def step(self, action):
        value = np.asarray(action, dtype=np.float64)
        self.actions.append(value)
        if self.step_hook is not None:
            self.step_hook(len(self.actions), value)
        return dict(self.obs), 0.0, False, {}


class FakeImageTools:
    @staticmethod
    def resize_with_pad(image, height, width):
        del height, width
        return image

    @staticmethod
    def convert_to_uint8(image):
        return np.asarray(image, dtype=np.uint8)


class FakeClient:
    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.requests = []

    def infer(self, element):
        self.requests.append(element)
        return {"actions": self.chunks.pop(0)}


def _context(epoch: int = 4) -> ContextEnvelope:
    return ContextEnvelope(
        phase=ContextPhase.PRE_DISPATCH_FACTS,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id="request-1",
        request_generation=0,
        episode_id="episode-1",
        goal_id="goal-1",
        goal_epoch=0,
        epoch_id=epoch,
        graph_version="graph-1",
        occurrence_id="task08-o000",
        attempt_id=None,
        certificate_hash="certificate-1",
        safety_epoch=None,
    )


def _task8():
    package = ScriptedProposalProvider().propose(8, epoch_id=4)
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage.json", 8
    )
    return package, binding


def _grounder(env: FakeEnv):
    package, binding = _task8()
    store = LiberoObservationStore(dict(env.obs), epoch_id=4)
    facts = monitored_fact_universe(package.problem)
    grounder = LiberoOracleGrounder(env, store, binding, facts)
    return package, store, grounder


def test_oracle_grounder_maps_relations_holding_and_exactly_one_location() -> None:
    env = FakeEnv()
    env.env.relations.update(
        {
            ("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", "flat_stove_1_cook_region"): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_right_init_region"): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"): True,
            ("on", "moka_pot_2", "flat_stove_1_cook_region"): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )
    package, _, grounder = _grounder(env)
    context = _context()
    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        context,
        monitored_fact_universe(package.problem),
    )

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact("at", ("moka_pot_1", "kitchen_table_moka_pot_right_init_region")) in response.snapshot.true_facts
    assert Fact("at", ("moka_pot_1", "flat_stove_1_cook_region")) in response.snapshot.false_facts
    assert Fact("holding", ("moka_pot_1",)) in response.snapshot.false_facts
    assert Fact("handempty") in response.snapshot.true_facts
    assert Fact("powered-on", ("flat_stove_1_power",)) in response.snapshot.true_facts

    env.env.held.add("moka_pot_1")
    env.env.relations[("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region")] = False
    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        context,
        frozenset({Fact("holding", ("moka_pot_1",)), Fact("handempty")}),
    )
    assert response.snapshot is not None
    assert Fact("holding", ("moka_pot_1",)) in response.snapshot.true_facts
    assert Fact("handempty") in response.snapshot.false_facts


def test_conflicting_locations_fail_closed() -> None:
    env = FakeEnv()
    for object_name in ("moka_pot_1", "moka_pot_2"):
        for location in (
            "kitchen_table_moka_pot_right_init_region",
            "kitchen_table_moka_pot_left_init_region",
            "flat_stove_1_cook_region",
        ):
            env.env.relations[("on", object_name, location)] = False
    env.env.relations[("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region")] = True
    env.env.relations[("on", "moka_pot_1", "flat_stove_1_cook_region")] = True
    env.env.relations[("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region")] = True
    env.env.relations[("turnon", "flat_stove_1")] = True
    env.env.relations[("turnoff", "flat_stove_1")] = False
    package, _, grounder = _grounder(env)

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        monitored_fact_universe(package.problem),
    )
    assert response.status is GroundingStatus.STATE_GROUNDING_FAILURE
    assert "exactly-one" in response.reason


def test_prompt_for_task8_names_only_one_branch_and_forbids_advancing() -> None:
    package, _ = _task8()
    renderer = SubtaskPromptRenderer()
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    first_prompt = renderer.render(first)
    second_prompt = renderer.render(second)

    assert "right moka pot" in first_prompt.lower()
    assert "left moka pot" not in first_prompt.lower()
    assert "both" not in first_prompt.lower()
    assert "do not start another subtask" in first_prompt.lower()
    assert "left moka pot" in second_prompt.lower()
    assert "right moka pot" not in second_prompt.lower()


def test_macro_executor_stops_on_observed_effect_flushes_chunk_and_preserves_gripper() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = package.proposal.candidate_subtasks[0].action
    start_location = action.arguments[1]
    target = action.arguments[2]
    env.env.relations[("on", "moka_pot_1", start_location)] = True
    env.env.relations[("on", "moka_pot_1", target)] = False
    env.env.relations[("on", "moka_pot_2", "kitchen_table_moka_pot_right_init_region")] = False
    env.env.relations[("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region")] = True
    env.env.relations[("on", "moka_pot_2", target)] = False
    env.env.relations[("turnon", "flat_stove_1")] = True
    env.env.relations[("turnoff", "flat_stove_1")] = False

    def change_world(step: int, low_level_action: np.ndarray) -> None:
        if step == 2:
            env.env.relations[("on", "moka_pot_1", start_location)] = False
            env.env.relations[("on", "moka_pot_1", target)] = True

    env.step_hook = change_world
    action_chunk = np.array(
        [
            [0, 0, 0, 0, 0, 0, -0.75],
            [0, 0, 0, 0, 0, 0, 0.65],
            [1, 0, 0, 0, 0, 0, 0.1],
            [1, 0, 0, 0, 0, 0, 0.1],
        ],
        dtype=np.float64,
    )
    client = FakeClient([action_chunk])
    executor = Pi05MacroExecutor(
        env=env,
        client=client,
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=4,
        max_action_steps=12,
        settling_steps=2,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    assert dispatch.status is DispatchStatus.ENQUEUED
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert outcome.stopped and outcome.stop_ack_attempt_id == dispatch.attempt_id
    assert outcome.settled_epoch == store.epoch_id
    assert len(client.requests) == 1
    # Two task actions, then two settling holds: the unused two chunk actions are flushed.
    assert len(env.actions) == 4
    assert np.allclose(env.actions[-1][:6], 0.0)
    assert env.actions[-1][-1] == pytest.approx(0.65)
    assert executor.results[-1].unused_actions_flushed == 2
    assert executor.results[-1].detector_calls == 2
    assert Fact("at", ("moka_pot_1", target)) in grounder.peek_snapshot().true_facts


@pytest.mark.parametrize(
    "bad_chunk",
    [
        np.zeros((1, 7)),
        np.full((4, 7), np.nan),
        np.zeros((4, 6)),
        np.full((4, 7), 1.5),
    ],
)
def test_invalid_policy_actions_fail_closed(bad_chunk: np.ndarray) -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = package.proposal.candidate_subtasks[0].action
    client = FakeClient([bad_chunk])
    executor = Pi05MacroExecutor(
        env=env,
        client=client,
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=4,
        max_action_steps=4,
        settling_steps=0,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.OUTCOME_UNKNOWN
    assert not outcome.stopped
    assert env.actions == []


def test_safety_veto_creates_no_attempt_and_no_command() -> None:
    env = FakeEnv()
    env.obs["robot0_eef_pos"] = np.array([np.nan, 0.0, 0.5])
    package, store, grounder = _grounder(env)
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        settling_steps=0,
    )

    result = executor.consume_permit_and_enqueue(
        package.proposal.candidate_subtasks[0].action,
        _context(),
        package.proposal.initial_snapshot,
    )
    assert result.status is DispatchStatus.SAFETY_VETO
    assert result.attempt_id is None
    assert env.actions == []


def test_manifest_is_frozen_and_prompt_configuration_is_valid_json() -> None:
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage.json", 8
    )
    prompt_payload = json.loads(
        (ROOT / "configs/logiv/prompts/pi05-subtasks-v1.json").read_text()
    )
    assert binding.task_id == 8
    assert binding.frozen is True
    assert prompt_payload["prompt_version"] == "pi05-subtasks-v1"
    assert set(FixedDomain().schemas) <= set(prompt_payload["templates"])
