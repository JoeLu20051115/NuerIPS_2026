from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

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
        self.bottom_offset = np.array([0.0, 0.0, -0.05])


class FakeRobot:
    gripper = object()


class FakeInnerEnv:
    def __init__(self) -> None:
        self.relations: dict[tuple[str, ...], bool] = {}
        self.held: set[str] = set()
        self.objects_dict = {
            "black_book_1": FakeObject("black_book_1"),
            "moka_pot_1": FakeObject("moka_pot_1"),
            "moka_pot_2": FakeObject("moka_pot_2"),
            "white_yellow_mug_1": FakeObject("white_yellow_mug_1"),
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
        self.done = False
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
        return dict(self.obs), 0.0, self.done, {}


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


def test_task5_wrong_compartment_is_a_grounded_recovery_location() -> None:
    env = FakeEnv()
    package = ScriptedProposalProvider().propose(5, epoch_id=4)
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage.json", 5
    )
    locations = {
        fact.arguments[1]
        for fact in monitored_fact_universe(package.problem)
        if fact.predicate == "at" and fact.arguments[0] == "black_book_1"
    }
    wrong = "desk_caddy_1_front_contain_region"
    for location in locations:
        relation = LiberoOracleGrounder._location_predicate(location)
        env.env.relations[(relation, "black_book_1", location)] = location == wrong
    grounder = LiberoOracleGrounder(
        env,
        LiberoObservationStore(dict(env.obs), epoch_id=4),
        binding,
        monitored_fact_universe(package.problem),
    )

    snapshot = grounder.peek_snapshot()

    assert Fact("at", ("black_book_1", wrong)) in snapshot.true_facts
    assert Fact("handempty") in snapshot.true_facts


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


def test_reliable_holding_overrides_stale_on_relation_before_exactly_one_lint() -> None:
    env = FakeEnv()
    source = "kitchen_table_moka_pot_left_init_region"
    target = "flat_stove_1_cook_region"
    env.env.relations.update(
        {
            ("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"): True,
            ("on", "moka_pot_1", source): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_right_init_region"): False,
            # Contact can leave LIBERO's On predicate true just after grasp.
            ("on", "moka_pot_2", source): True,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )
    env.env.held.add("moka_pot_2")
    package, _, grounder = _grounder(env)

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        monitored_fact_universe(package.problem),
    )

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact("holding", ("moka_pot_2",)) in response.snapshot.true_facts
    assert Fact("at", ("moka_pot_2", source)) in response.snapshot.false_facts
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


def test_oracle_grounder_recovers_object_on_registered_table_surface() -> None:
    env = FakeEnv()
    recovery = "kitchen_table_recovery_surface"
    source_1 = "kitchen_table_moka_pot_right_init_region"
    source_2 = "kitchen_table_moka_pot_left_init_region"
    target = "flat_stove_1_cook_region"
    env.env.relations.update(
        {
            ("on", "moka_pot_1", source_1): True,
            ("on", "moka_pot_1", source_2): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", source_1): False,
            ("on", "moka_pot_2", source_2): False,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )
    # ``kitchen_table`` is an arena workspace and cannot be queried through
    # LIBERO's ordinary object-state predicate table.  The current simulator
    # contact pair is the positive recovery-surface evidence.
    geom_names = ("moka_pot_2_geom", "table_collision")
    geom_ids = {name: index for index, name in enumerate(geom_names)}
    env.env.get_object = lambda name: env.env.objects_dict[name]
    env.env.sim = SimpleNamespace(
        model=SimpleNamespace(
            ngeom=len(geom_names),
            geom_name2id=lambda name: geom_ids[name],
            geom_id2name=lambda index: geom_names[index],
        ),
        data=SimpleNamespace(
            ncon=1,
            contact=[SimpleNamespace(geom1=geom_ids["moka_pot_2_geom"], geom2=geom_ids["table_collision"])],
        ),
    )
    package, _, grounder = _grounder(env)

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        monitored_fact_universe(package.problem),
    )

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact("at", ("moka_pot_2", recovery)) in response.snapshot.true_facts
    assert Fact("holding", ("moka_pot_2",)) in response.snapshot.false_facts


def test_oracle_grounder_uses_workspace_geometry_when_table_contact_is_missing() -> None:
    env = FakeEnv()
    source_1 = "kitchen_table_moka_pot_right_init_region"
    source_2 = "kitchen_table_moka_pot_left_init_region"
    target = "flat_stove_1_cook_region"
    env.env.relations.update(
        {
            ("on", "moka_pot_1", source_1): True,
            ("on", "moka_pot_1", source_2): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", source_1): False,
            ("on", "moka_pot_2", source_2): False,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )
    geom_names = ("moka_pot_2_geom",)
    env.env.get_object = lambda name: env.env.objects_dict[name]
    env.env.obj_body_id = {"moka_pot_1": 0, "moka_pot_2": 1}
    env.env.workspace_offset = np.array([0.0, 0.0, 0.9])
    env.env.kitchen_table_full_size = (1.0, 1.2, 0.05)
    env.env.sim = SimpleNamespace(
        model=SimpleNamespace(
            ngeom=1,
            geom_name2id=lambda name: geom_names.index(name),
            geom_id2name=lambda index: geom_names[index],
        ),
        data=SimpleNamespace(
            ncon=0,
            contact=[],
            body_xpos=np.array(
                [
                    [0.2, 0.1, 0.95],
                    [-0.2, -0.1, 0.95],
                ]
            ),
        ),
    )
    package, _, grounder = _grounder(env)

    response = grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        monitored_fact_universe(package.problem),
    )

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact("at", ("moka_pot_2", "kitchen_table_recovery_surface")) in (
        response.snapshot.true_facts
    )


def test_prompt_for_task8_names_only_one_branch_and_forbids_advancing() -> None:
    package, _ = _task8()
    renderer = SubtaskPromptRenderer()
    by_object = {
        item.action.arguments[0]: item.action
        for item in package.proposal.candidate_subtasks
    }

    first_prompt = renderer.render(by_object["moka_pot_1"])
    second_prompt = renderer.render(by_object["moka_pot_2"])

    assert "right moka pot" in first_prompt.lower()
    assert "left moka pot" not in first_prompt.lower()
    assert "both" not in first_prompt.lower()
    assert "do not start another subtask" in first_prompt.lower()
    assert "left moka pot" in second_prompt.lower()
    assert "right moka pot" not in second_prompt.lower()


def test_macro_executor_stops_on_observed_effect_flushes_chunk_and_settles_released() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.arguments[0] == "moka_pot_1"
    )
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
    assert env.actions[-1][-1] == pytest.approx(-1.0)
    assert executor.results[-1].unused_actions_flushed == 2
    assert executor.results[-1].detector_calls == 2
    assert executor.total_action_steps == 2
    assert Fact("at", ("moka_pot_1", target)) in grounder.peek_snapshot().true_facts


def test_effect_gated_macro_stops_when_object_lands_on_recovery_surface() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = package.proposal.candidate_subtasks[0].action
    assert action.arguments[0] == "moka_pot_2"
    object_name, source, target = action.arguments
    recovery = "kitchen_table_recovery_surface"
    env.env.relations.update(
        {
            ("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", target): False,
            ("on", object_name, "kitchen_table_moka_pot_right_init_region"): False,
            ("on", object_name, source): True,
            ("on", object_name, target): False,
            ("on", object_name, "kitchen_table"): True,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def land_outside_nominal_regions(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", object_name, source)] = False

    env.step_hook = land_outside_nominal_regions
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((4, 7), dtype=np.float64)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v5.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=4,
        max_action_steps=12,
        max_total_action_steps=12,
        settling_steps=0,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert outcome.reason == "observed target-location divergence"
    assert len(executor.results[0].actions) == 1
    assert executor.results[0].unused_actions_flushed == 3
    assert executor.results[0].post_snapshot is not None
    assert Fact("at", (object_name, recovery)) in (
        executor.results[0].post_snapshot.true_facts
    )


def test_v2_task8_prompt_uses_concise_training_style_and_explicit_release() -> None:
    package, _ = _task8()
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v2.json"
    )
    by_object = {
        item.action.arguments[0]: item.action
        for item in package.proposal.candidate_subtasks
    }

    assert renderer.render(by_object["moka_pot_1"]).startswith(
        "Put the moka pot closest to the stove"
    )
    assert "release" in renderer.render(by_object["moka_pot_1"]).lower()
    assert renderer.render(by_object["moka_pot_2"]).startswith(
        "Put the remaining moka pot on the stove"
    )


def test_v3_macro_switches_from_acquire_to_finish_on_verified_holding() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.arguments[0] == "moka_pot_1"
    )
    source, target = action.arguments[1:]
    env.env.relations.update(
        {
            ("on", "moka_pot_1", source): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_right_init_region"): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"): True,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def transition(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.held.add("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", source)] = False
        elif step == 2:
            env.env.held.remove("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", target)] = True

    env.step_hook = transition
    client = FakeClient(
        [
            np.zeros((2, 7), dtype=np.float64),
            np.zeros((2, 7), dtype=np.float64),
        ]
    )
    executor = Pi05MacroExecutor(
        env=env,
        client=client,
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v3.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=2,
        max_action_steps=4,
        max_total_action_steps=4,
        settling_steps=0,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert len(client.requests) == 2
    assert client.requests[0]["prompt"].startswith("Pick up")
    assert client.requests[1]["prompt"].startswith("Put the moka pot you are holding")
    assert len(executor.results[0].prompt_history) == 2
    assert executor.results[0].unused_actions_flushed == 2
    assert len(executor.results[0].actions) == 2


def test_v4_task8_order_and_prompts_bind_the_same_objects() -> None:
    package, _ = _task8()
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v4.json"
    )
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert first.arguments[0] == "moka_pot_2"
    assert second.arguments[0] == "moka_pot_1"
    assert "closest to the stove" in renderer.render_phase(first, "acquire")
    assert "remaining moka pot" in renderer.render_phase(second, "acquire")
    assert renderer.has_phase(first, "finish")
    assert renderer.has_phase(second, "finish")


def test_v5_uses_training_task_prompt_but_keeps_occurrence_effect_boundaries() -> None:
    package, _ = _task8()
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v5.json"
    )
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render(first) == "Put both moka pots on the stove."
    assert renderer.render(second) == "Put both moka pots on the stove."
    assert not renderer.has_phase(first, "finish")
    assert not renderer.has_phase(second, "finish")
    assert first != second


def test_v6_uses_natural_prompts_for_observed_task5_and_task9_failures() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v6.json"
    )
    provider = ScriptedProposalProvider()
    book = provider.propose(5, epoch_id=0).proposal.candidate_subtasks[0].action
    microwave = provider.propose(9, epoch_id=0).proposal.candidate_subtasks

    assert renderer.render(book) == (
        "Pick up the book and place it in the back compartment of the caddy."
    )
    assert [renderer.render(item.action) for item in microwave] == [
        "Put the yellow and white mug in the microwave.",
        "Close the microwave.",
    ]


def test_v7_requests_stable_book_placement_and_full_microwave_task() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v7.json"
    )
    provider = ScriptedProposalProvider()
    book = provider.propose(5, epoch_id=0).proposal.candidate_subtasks[0].action
    microwave = provider.propose(9, epoch_id=0).proposal.candidate_subtasks[0].action

    assert renderer.render(book) == (
        "Lay the book flat and fully inside the back compartment of the caddy, "
        "release it gently, and stop."
    )
    assert renderer.render(microwave) == (
        "Put the yellow and white mug in the microwave and close it."
    )


def test_v8_recovery_prompts_cover_new_locations_and_reuse_microwave_task() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v8.json"
    )
    provider = ScriptedProposalProvider()
    task5 = provider.propose(5, epoch_id=0)
    moved_book = FixedDomain().ground(
        task5.problem,
        "place-in",
        (
            "black_book_1",
            "desk_caddy_1_front_contain_region",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_access",
        ),
    )
    close_microwave = provider.propose(
        9, epoch_id=0
    ).proposal.candidate_subtasks[1].action

    assert renderer.render(moved_book) == (
        "Pick up the book and place it in the back compartment of the caddy."
    )
    assert renderer.render(close_microwave) == (
        "Put the yellow and white mug in the microwave and close it."
    )


def test_v9_recovery_pick_and_microwave_phase_prompts_are_explicit() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v9.json"
    )
    provider = ScriptedProposalProvider()
    task5 = provider.propose(5, epoch_id=0)
    pick_book = FixedDomain().ground(
        task5.problem,
        "pick",
        ("black_book_1", "study_table_recovery_surface"),
    )
    place_mug = provider.propose(
        9, epoch_id=0
    ).proposal.candidate_subtasks[0].action

    assert renderer.render(pick_book) == (
        "Pick up the book and place it in the back compartment of the caddy."
    )
    assert renderer.render_phase(place_mug, "acquire") == (
        "Pick up the yellow and white mug."
    )
    assert renderer.render_phase(place_mug, "finish") == (
        "Put the mug you are holding in the microwave and release it."
    )


def test_v10_task8_recovery_prompt_targets_only_the_failed_branch() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v10.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    problem = package.problem
    domain = FixedDomain()
    nominal = [item.action for item in package.proposal.candidate_subtasks]
    nearer = domain.ground(
        problem,
        "place-on",
        (
            "moka_pot_2",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )
    remaining = domain.ground(
        problem,
        "place-on",
        (
            "moka_pot_1",
            "kitchen_table_recovery_surface",
            "flat_stove_1_cook_region",
        ),
    )

    assert [renderer.render(action) for action in nominal] == [
        "Put both moka pots on the stove.",
        "Put both moka pots on the stove.",
    ]
    assert renderer.render_phase(nominal[0], "acquire") == (
        "Put both moka pots on the stove."
    )
    assert renderer.render_phase(nominal[1], "acquire") == (
        "Pick up the remaining moka pot that is not on the stove."
    )
    assert renderer.render_phase(nominal[1], "finish") == (
        "Put the moka pot you are holding on the stove and release it, then stop."
    )
    assert renderer.render_phase(nearer, "acquire") == (
        "Pick up the moka pot closest to the stove."
    )
    assert renderer.render_phase(remaining, "acquire") == (
        "Pick up the remaining moka pot that is not on the stove."
    )
    for action in (nearer, remaining):
        assert renderer.render_phase(action, "finish") == (
            "Put the moka pot you are holding on the stove and release it, then stop."
        )
        assert "both" not in renderer.render_phase(action, "acquire").lower()


def test_v11_task8_uses_visual_left_right_branch_names() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v11.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    domain = FixedDomain()
    nominal = [item.action for item in package.proposal.candidate_subtasks]
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

    assert renderer.render_phase(nominal[0], "acquire") == (
        "Put both moka pots on the stove."
    )
    assert renderer.render_phase(nominal[1], "acquire") == (
        "Pick up the right moka pot from the table, then hold it securely."
    )
    assert renderer.render_phase(left_recovery, "acquire") == (
        "Pick up the left moka pot from the table, then hold it securely."
    )
    assert renderer.render_phase(right_recovery, "acquire") == (
        "Pick up the right moka pot from the table, then hold it securely."
    )


def test_v12_keeps_nominal_training_prompt_and_targets_only_recovery() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v12.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    domain = FixedDomain()
    nominal = [item.action for item in package.proposal.candidate_subtasks]
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

    assert [renderer.render_phase(action, "acquire") for action in nominal] == [
        "Put both moka pots on the stove.",
        "Put both moka pots on the stove.",
    ]
    assert renderer.render_phase(left_recovery, "acquire") == (
        "Pick up the left moka pot from the table, then hold it securely."
    )
    assert renderer.render_phase(right_recovery, "acquire") == (
        "Pick up the right moka pot from the table, then hold it securely."
    )


def test_effect_gated_macro_does_not_confuse_libero_success_with_termination() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.arguments[0] == "moka_pot_1"
    )
    source, target = action.arguments[1:]
    env.env.relations.update(
        {
            ("on", "moka_pot_1", source): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_right_init_region"): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"): True,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def transition(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.held.add("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", source)] = False
            # LIBERO's done is task success, not an absorbing terminal state.
            env.done = True
        elif step == 2:
            env.env.held.remove("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", target)] = True

    env.step_hook = transition
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient(
            [
                np.zeros((1, 7), dtype=np.float64),
                np.zeros((1, 7), dtype=np.float64),
            ]
        ),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v3.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=2,
        max_total_action_steps=2,
        settling_steps=0,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert outcome.reason == "observed declared effects"
    assert len(executor.results[0].actions) == 2
    assert len(executor.results[0].prompt_history) == 2


def test_effect_gated_macro_requires_consecutive_confirmation_before_stop() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.arguments[0] == "moka_pot_1"
    )
    source, target = action.arguments[1:]
    env.env.relations.update(
        {
            ("on", "moka_pot_1", source): True,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"): True,
            ("on", "moka_pot_2", target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def transition(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", "moka_pot_1", source)] = False
            env.env.relations[("on", "moka_pot_1", target)] = True
        elif step == 2:
            env.env.relations[("on", "moka_pot_1", target)] = False
            env.env.held.add("moka_pot_1")
        elif step == 3:
            env.env.held.remove("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", target)] = True

    env.step_hook = transition
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient(
            [np.zeros((1, 7), dtype=np.float64) for _ in range(4)]
        ),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=4,
        max_total_action_steps=4,
        settling_steps=0,
        effect_confirmation_steps=2,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert outcome.reason == "observed declared effects"
    assert len(executor.results[0].actions) == 4


def test_episode_action_budget_rejects_before_creating_another_attempt() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.arguments[0] == "moka_pot_1"
    )
    for object_name in ("moka_pot_1", "moka_pot_2"):
        for location in (
            "kitchen_table_moka_pot_right_init_region",
            "kitchen_table_moka_pot_left_init_region",
            "flat_stove_1_cook_region",
        ):
            env.env.relations[("on", object_name, location)] = False
    env.env.relations[("on", "moka_pot_1", action.arguments[1])] = True
    env.env.relations[("on", "moka_pot_2", "kitchen_table_moka_pot_left_init_region")] = True
    env.env.relations[("turnon", "flat_stove_1")] = True
    env.env.relations[("turnoff", "flat_stove_1")] = False
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        max_total_action_steps=1,
        settling_steps=0,
    )
    first = executor.consume_permit_and_enqueue(action, _context(), package.proposal.initial_snapshot)
    assert executor.await_outcome(first).status is ExecutorStatus.SUCCEEDED
    current = grounder.peek_snapshot()
    second_context = replace(_context(), epoch_id=store.epoch_id, request_id="request-2")

    second = executor.consume_permit_and_enqueue(action, second_context, current)

    assert second.status is DispatchStatus.ACTION_BUDGET_EXHAUSTED
    assert second.attempt_id is None


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
