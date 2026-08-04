from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pi05_libero_repro.logiv.controller import (
    DispatchStatus,
    ExecutionCompletionHint,
    ExecutorStatus,
    GroundingStatus,
)
from pi05_libero_repro.logiv.domain import FixedDomain
from pi05_libero_repro.logiv.libero_adapter import (
    build_libero_transition_feature_reader,
    GroundingError,
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
    TruthValue,
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.shadow_monitor import load_monitor_evidence_contract
from pi05_libero_repro.protocol import ShadowStepContext


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


def _audited_grounder():
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
    package, binding = _task8()
    store = LiberoObservationStore(dict(env.obs), epoch_id=4)
    unknown = Fact("unregistered-state", ("moka_pot_1",))
    grounder = LiberoOracleGrounder(
        env,
        store,
        binding,
        monitored_fact_universe(package.problem) | {unknown},
    )
    return env, store, grounder, unknown


def _rehash_evidence_payload(payload: dict) -> tuple[str, str]:
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload_json, hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def test_task_binding_accepts_a_task_scoped_extended_manifest(tmp_path: Path) -> None:
    source = "living_room_table_recovery_surface"
    overlay = tmp_path / "coverage-overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "extends": str(
                    (ROOT / "configs/logiv/libero10-coverage.json").resolve()
                ),
                "tasks": [
                    {
                        "task_id": 0,
                        "registered_objects": [
                            "alphabet_soup_1",
                            "tomato_sauce_1",
                            source,
                            "basket_1_contain_region",
                            "basket_1_access",
                        ],
                        "symbol_bindings": {
                            source: {
                                "libero_id": "living_room_table",
                                "kind": "support_surface_alias",
                            },
                            "basket_1_access": {
                                "libero_id": "basket_1_contain_region",
                                "kind": "always_open_access",
                            },
                        },
                        "decompose_macro_sources": [source],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    binding = TaskBinding.from_manifest(overlay, 0)

    assert binding.resolve(source) == ("living_room_table", "support_surface_alias")
    assert binding.supported_action_schemas == frozenset({"place-in"})
    assert binding.decompose_macro_sources == frozenset({source})


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


def test_v53_task6_uses_a_safe_tie_break_without_removing_dag_independence() -> None:
    provider = ScriptedProposalProvider(
        ROOT / "configs/logiv/libero10-scripted-proposals-v53-task6-order.json"
    )
    package = provider.propose(6, epoch_id=4)
    actions = [item.action for item in package.proposal.candidate_subtasks]

    assert [action.arguments[0] for action in actions] == [
        "chocolate_pudding_1",
        "porcelain_mug_1",
    ]
    assert all(action.preconditions <= package.problem.initial_state for action in actions)


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


def test_oracle_grounder_emits_a_complete_audited_true_false_unknown_partition() -> None:
    _, _, grounder, unknown = _audited_grounder()

    snapshot = grounder.peek_snapshot()

    assert snapshot.fact_universe is not None
    assert snapshot.fact_universe_version is not None
    assert snapshot.fact_universe_sha256 is not None
    assert snapshot.evidence_payload_json is not None
    assert snapshot.fact_universe == (
        snapshot.true_facts | snapshot.false_facts | snapshot.unknown(snapshot.fact_universe)
    )
    assert unknown in snapshot.unknown(snapshot.fact_universe)
    payload = json.loads(snapshot.evidence_payload_json)
    assert {value for _, value in payload["values"]} == {"TRUE", "FALSE", "UNKNOWN"}


def test_epoch_and_observation_noise_change_evidence_but_not_fact_universe() -> None:
    _, store, grounder, _ = _audited_grounder()
    first = grounder.peek_snapshot()
    noisy = dict(store.read()[1])
    noisy["agentview_image"] = noisy["agentview_image"].copy()
    noisy["agentview_image"][0, 0, 0] = 255
    store.update(noisy)

    second = grounder.peek_snapshot()

    assert second.epoch_id == first.epoch_id + 1
    assert second.fact_universe == first.fact_universe
    assert second.fact_universe_version == first.fact_universe_version
    assert second.fact_universe_sha256 == first.fact_universe_sha256
    assert second.evidence_hash != first.evidence_hash
    assert second.evidence_payload_json != first.evidence_payload_json


def test_observation_store_owns_immutable_array_copies() -> None:
    observation = dict(FakeEnv().obs)
    original_pixel = int(observation["agentview_image"][0, 0, 0])
    store = LiberoObservationStore(observation, epoch_id=4)

    observation["agentview_image"][0, 0, 0] = original_pixel + 1
    _, stored, _ = store.read()

    assert int(stored["agentview_image"][0, 0, 0]) == original_pixel
    assert stored["agentview_image"].flags.writeable is False
    with pytest.raises(ValueError, match="read-only|writeable|assignment"):
        stored["agentview_image"][0, 0, 0] = original_pixel + 2


def test_observation_store_reads_cannot_mutate_owned_arrays() -> None:
    observation = dict(FakeEnv().obs)
    original_pixel = int(observation["agentview_image"][0, 0, 0])
    store = LiberoObservationStore(observation, epoch_id=4)

    read_epoch, returned, _ = store.read()
    returned["agentview_image"].setflags(write=True)
    returned["agentview_image"][0, 0, 0] = original_pixel + 1
    later_epoch, later, _ = store.read()

    assert read_epoch == later_epoch == 4
    assert int(later["agentview_image"][0, 0, 0]) == original_pixel


def test_snapshot_and_synchronous_simulator_advance_cannot_mix_epochs() -> None:
    env = FakeEnv()
    source = "kitchen_table_moka_pot_right_init_region"
    env.env.relations[("on", "moka_pot_1", source)] = True
    _, binding = _task8()
    store = LiberoObservationStore(dict(env.obs), epoch_id=4)
    monitored = frozenset(
        {
            Fact("at", ("moka_pot_1", source)),
            Fact("holding", ("moka_pot_1",)),
            Fact("handempty"),
        }
    )
    grounder = LiberoOracleGrounder(env, store, binding, monitored)
    original_truth = grounder._truth
    grounding_started = threading.Event()
    release_grounding = threading.Event()
    first_call = True

    def blocking_truth(fact):
        nonlocal first_call
        if first_call:
            first_call = False
            grounding_started.set()
            assert release_grounding.wait(1.0)
        return original_truth(fact)

    grounder._truth = blocking_truth
    captured: list = []
    grounding_thread = threading.Thread(target=lambda: captured.append(grounder.peek_snapshot()))
    grounding_thread.start()
    assert grounding_started.wait(1.0)

    advance_finished = threading.Event()

    def advance() -> None:
        def step():
            env.env.held.add("moka_pot_1")
            env.env.relations[("on", "moka_pot_1", source)] = False
            observation = dict(env.obs)
            observation["robot0_eef_pos"] = np.array([0.4, 0.0, 0.5])
            return observation, 0.0, False, {}

        store.advance(step)
        advance_finished.set()

    advance_thread = threading.Thread(target=advance)
    advance_thread.start()
    assert advance_finished.wait(0.05) is False
    release_grounding.set()
    grounding_thread.join(1.0)
    advance_thread.join(1.0)

    assert advance_finished.is_set()
    assert captured[0].epoch_id == 4
    assert Fact("at", ("moka_pot_1", source)) in captured[0].true_facts
    assert Fact("holding", ("moka_pot_1",)) in captured[0].false_facts
    current = grounder.peek_snapshot()
    assert current.epoch_id == 5
    assert Fact("holding", ("moka_pot_1",)) in current.true_facts


def test_required_facts_cannot_expand_the_frozen_registered_universe() -> None:
    _, _, grounder, _ = _audited_grounder()
    unregistered = Fact("never-registered", ("moka_pot_1",))

    with pytest.raises(GroundingError, match="registered|universe"):
        grounder._snapshot(frozenset({unregistered}))


def test_audited_snapshot_rejects_tampered_missing_duplicate_or_conflicting_evidence() -> None:
    _, _, grounder, _ = _audited_grounder()
    snapshot = grounder.peek_snapshot()
    assert snapshot.evidence_payload_json is not None

    payload = json.loads(snapshot.evidence_payload_json)
    payload["epoch_id"] += 1
    with pytest.raises(ValueError, match="evidence|epoch|hash"):
        replace(snapshot, evidence_payload_json=_rehash_evidence_payload(payload)[0])

    payload = json.loads(snapshot.evidence_payload_json)
    payload["values"].pop()
    missing_json, missing_hash = _rehash_evidence_payload(payload)
    with pytest.raises(ValueError, match="universe|partition|missing"):
        replace(
            snapshot,
            evidence_payload_json=missing_json,
            evidence_hash=missing_hash,
        )

    payload = json.loads(snapshot.evidence_payload_json)
    payload["values"].append(payload["values"][0])
    duplicate_json, duplicate_hash = _rehash_evidence_payload(payload)
    with pytest.raises(ValueError, match="duplicate|partition"):
        replace(
            snapshot,
            evidence_payload_json=duplicate_json,
            evidence_hash=duplicate_hash,
        )

    payload = json.loads(snapshot.evidence_payload_json)
    payload["values"].reverse()
    reordered_json, reordered_hash = _rehash_evidence_payload(payload)
    with pytest.raises(ValueError, match="canonical|sorted"):
        replace(
            snapshot,
            evidence_payload_json=reordered_json,
            evidence_hash=reordered_hash,
        )

    known = next(iter(snapshot.true_facts))
    with pytest.raises(ValueError, match="TRUE and FALSE|conflict"):
        replace(snapshot, false_facts=snapshot.false_facts | {known})


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


def _task0_grounding_after_alphabet_shift(alphabet_x: float):
    env = FakeEnv()
    env.env.objects_dict.update(
        {
            "alphabet_soup_1": FakeObject("alphabet_soup_1"),
            "tomato_sauce_1": FakeObject("tomato_sauce_1"),
        }
    )
    alphabet_source = "living_room_table_alphabet_soup_init_region"
    tomato_source = "living_room_table_tomato_sauce_init_region"
    target = "basket_1_contain_region"
    env.env.relations.update(
        {
            ("on", "alphabet_soup_1", alphabet_source): False,
            ("on", "alphabet_soup_1", tomato_source): False,
            ("in", "alphabet_soup_1", target): False,
            ("on", "tomato_sauce_1", alphabet_source): False,
            ("on", "tomato_sauce_1", tomato_source): True,
            ("in", "tomato_sauce_1", target): False,
        }
    )
    env.env.parsed_problem = {
        "initial_state": [
            ["on", "alphabet_soup_1", alphabet_source],
            ["on", "tomato_sauce_1", tomato_source],
        ],
        "regions": {
            alphabet_source: {"target": "living_room_table"},
            tomato_source: {"target": "living_room_table"},
        },
    }
    env.env.get_object = lambda name: env.env.objects_dict.get(name)
    env.env.object_sites_dict = {
        alphabet_source: SimpleNamespace(size=np.array([0.025, 0.025, 0.007])),
        tomato_source: SimpleNamespace(size=np.array([0.025, 0.025, 0.007])),
    }
    env.env.obj_body_id = {"alphabet_soup_1": 0, "tomato_sauce_1": 1}
    env.env.workspace_offset = np.array([0.0, 0.0, 0.9])
    env.env.living_room_table_full_size = (1.0, 1.2, 0.05)
    geom_names = ("alphabet_soup_1_geom", "tomato_sauce_1_geom")
    env.env.sim = SimpleNamespace(
        model=SimpleNamespace(
            ngeom=len(geom_names),
            geom_name2id=lambda name: geom_names.index(name),
            geom_id2name=lambda index: geom_names[index],
        ),
        data=SimpleNamespace(
            ncon=0,
            contact=[],
            body_xpos=np.array([[alphabet_x, 0.0, 0.95], [-0.08, 0.05, 0.95]]),
            get_site_xpos=lambda name: np.array([0.0, 0.0, 0.9]),
            get_site_xmat=lambda name: np.eye(3),
        ),
    )
    package = ScriptedProposalProvider(
        ROOT / "configs/logiv/libero10-scripted-proposals-v38-task0-recovery.json"
    ).propose(0, epoch_id=4)
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage-v38-task0-recovery.json", 0
    )
    grounder = LiberoOracleGrounder(
        env,
        LiberoObservationStore(dict(env.obs), epoch_id=4),
        binding,
        monitored_fact_universe(package.problem),
    )

    return grounder.ground(
        ContextPhase.PRE_DISPATCH_FACTS,
        _context(),
        monitored_fact_universe(package.problem),
    )


def test_oracle_grounder_tolerates_millimetric_init_region_settling() -> None:
    response = _task0_grounding_after_alphabet_shift(0.026)

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact(
        "at",
        ("alphabet_soup_1", "living_room_table_alphabet_soup_init_region"),
    ) in response.snapshot.true_facts
    assert Fact(
        "at", ("alphabet_soup_1", "living_room_table_recovery_surface")
    ) in response.snapshot.false_facts


def test_oracle_grounder_distinguishes_far_drop_as_recovery_surface() -> None:
    response = _task0_grounding_after_alphabet_shift(0.2)

    assert response.status is GroundingStatus.OK
    assert response.snapshot is not None
    assert Fact(
        "at", ("alphabet_soup_1", "living_room_table_recovery_surface")
    ) in (
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


def test_identical_prompt_dag_frontier_waits_for_union_of_ready_effects() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_frontier_in_sequence(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True
        elif step == 3:
            env.env.relations[("on", second.arguments[0], second.arguments[1])] = False
            env.env.relations[("on", second.arguments[0], target)] = True

    env.step_hook = complete_frontier_in_sequence
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(4)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v14.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=4,
        max_total_action_steps=4,
        settling_steps=0,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    assert outcome.reason == "observed frontier effects"
    result = executor.results[-1]
    assert len(result.actions) == 3
    assert result.completion_mode == "DAG_FRONTIER"
    assert result.completion_occurrence_ids == hint.occurrence_ids
    assert result.completion_actions == hint.actions
    assert result.completion_positive == first.add_effects | second.add_effects
    assert result.completion_negative == first.del_effects | second.del_effects
    from scripts.eval_logiv_libero import _attempt_json

    artifact = _attempt_json(result)
    assert artifact["completion_mode"] == "DAG_FRONTIER"
    assert artifact["completion_occurrence_ids"] == list(hint.occurrence_ids)
    assert artifact["completion_actions"] == [first.pddl(), second.pddl()]
    assert artifact["completion_positive"] == sorted(
        fact.pddl() for fact in first.add_effects | second.add_effects
    )


def test_frontier_followup_deadline_reserves_budget_after_primary_completion() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_only_primary(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True

    env.step_hook = complete_only_primary
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(4)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v14.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=4,
        max_total_action_steps=4,
        settling_steps=0,
        frontier_followup_steps=2,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "frontier follow-up deadline reached"
    result = executor.results[-1]
    assert len(result.actions) == 2
    assert result.primary_effect_first_step == 1
    assert result.frontier_followup_limit == 2


def test_invalid_frontier_hint_does_not_consume_safety_permit_or_attempt_id() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v26.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        settling_steps=0,
    )
    mismatched = ExecutionCompletionHint(
        occurrence_ids=("task08-o001", "task08-o000"),
        actions=(second, first),
    )

    rejected = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=mismatched,
    )
    accepted = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
    )

    assert rejected.status is DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED
    assert accepted.status is DispatchStatus.ENQUEUED
    assert accepted.attempt_id == "sim-attempt-000000"
    assert accepted.context.safety_epoch == 1


def test_verified_primary_effect_switches_to_bounded_sibling_completion() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_frontier_in_order(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True
        elif step == 3:
            env.env.relations[("on", second.arguments[0], second.arguments[1])] = False
            env.env.relations[("on", second.arguments[0], target)] = True

    env.step_hook = complete_frontier_in_order
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(3)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v25.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=3,
        max_total_action_steps=3,
        settling_steps=0,
        effect_confirmation_steps=1,
        frontier_followup_steps=1,
        frontier_completion_followup_steps=3,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed frontier effects"
    result = executor.results[-1]
    assert len(result.actions) == 3
    assert result.prompt_history == (
        "put the moka pot closest to the stove on the stove",
        "put the remaining moka pot on the stove beside the other moka pot without moving the other moka pot",
    )
    assert result.frontier_completion_step == 1
    assert result.frontier_completion_prompt == result.prompt_history[-1]
    assert result.frontier_followup_limit == 3


def test_recovery_only_completion_does_not_change_initial_frontier() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_only_primary(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True

    env.step_hook = complete_only_primary
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v26.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        max_total_action_steps=1,
        settling_steps=0,
        effect_confirmation_steps=1,
        frontier_followup_steps=1,
        frontier_completion_followup_steps=3,
        frontier_completion_recovery_only=True,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "frontier follow-up deadline reached"
    result = executor.results[-1]
    assert result.prompt_history == ("put the moka pot closest to the stove on the stove",)
    assert result.frontier_completion_step is None


def test_recovery_frontier_budget_gate_preserves_late_single_occurrence() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    first = FixedDomain().ground(
        package.problem,
        "place-on",
        (
            first.arguments[0],
            "kitchen_table_recovery_surface",
            first.arguments[2],
        ),
    )
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            (
                "on",
                first.arguments[0],
                "kitchen_table_moka_pot_left_init_region",
            ): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64)] * 2),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v26.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        max_total_action_steps=520,
        settling_steps=0,
        frontier_completion_followup_steps=120,
        frontier_completion_recovery_only=True,
        frontier_recovery_max_consumed_steps=180,
    )
    initial_dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
    )
    executor.await_outcome(initial_dispatch)
    executor.total_action_steps = 181
    snapshot = grounder.peek_snapshot()
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        replace(
            _context(),
            request_id="request-2",
            epoch_id=snapshot.epoch_id,
            graph_version="graph-2",
        ),
        snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.status is ExecutorStatus.SUCCEEDED
    result = executor.results[-1]
    assert result.recovery_frontier is False
    assert result.completion_mode == "OCCURRENCE"
    renderer_prompt = executor.prompt_renderer.render_phase(first, "acquire")
    assert result.prompt == renderer_prompt
    assert renderer_prompt != "put both moka pots on the stove"


def test_stalled_frontier_switches_to_fallback_and_uses_fallback_followup() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_after_fallback(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 3:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True
        elif step == 5:
            env.env.relations[("on", second.arguments[0], second.arguments[1])] = False
            env.env.relations[("on", second.arguments[0], target)] = True

    env.step_hook = complete_after_fallback
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(5)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v23.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=5,
        max_total_action_steps=5,
        settling_steps=0,
        effect_confirmation_steps=1,
        frontier_followup_steps=1,
        frontier_fallback_after_steps=2,
        frontier_fallback_followup_steps=3,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed frontier effects"
    result = executor.results[-1]
    assert result.prompt_history == (
        "put the moka pot closest to the stove on the stove",
        "put both moka pots on the stove",
    )
    assert result.frontier_fallback_step == 2
    assert result.frontier_fallback_prompt == "put both moka pots on the stove"
    assert result.frontier_followup_limit == 3


@pytest.mark.parametrize("recovery_graph", [False, True])
def test_only_a_fresh_graph_uses_recovery_frontier_prompt(
    recovery_graph: bool,
) -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    completion_step = 2 if recovery_graph else 1

    def complete_both(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == completion_step:
            for action in (first, second):
                env.env.relations[("on", action.arguments[0], action.arguments[1])] = False
                env.env.relations[("on", action.arguments[0], target)] = True

    env.step_hook = complete_both
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient(
            [np.zeros((1, 7), dtype=np.float64)] * completion_step
        ),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v24.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=1,
        max_total_action_steps=completion_step,
        settling_steps=0,
    )
    if recovery_graph:
        initial_dispatch = executor.consume_permit_and_enqueue(
            first,
            _context(),
            package.proposal.initial_snapshot,
        )
        executor.await_outcome(initial_dispatch)
    else:
        # A sensor refresh before the first dispatch is not a recovery.
        store.update(dict(env.obs))
    snapshot = grounder.peek_snapshot()
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )
    context = replace(
        _context(),
        request_id="request-2" if recovery_graph else "request-1",
        epoch_id=snapshot.epoch_id,
        graph_version="graph-2" if recovery_graph else "graph-1",
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        context,
        snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed frontier effects"
    assert executor.results[-1].completion_mode == "DAG_FRONTIER"
    expected = (
        "put both moka pots on the stove"
        if recovery_graph
        else "put the moka pot closest to the stove on the stove"
    )
    assert executor.results[-1].prompt == expected


def test_frontier_switches_to_primary_finish_prompt_after_verified_holding() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, primary = [item.action for item in package.proposal.candidate_subtasks]
    object_name, source, target = primary.arguments
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", object_name, source): True,
            ("on", object_name, target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def acquire_then_finish(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", object_name, source)] = False
            env.env.held.add(object_name)
        elif step == 2:
            env.env.held.remove(object_name)
            env.env.relations[("on", object_name, target)] = True

    env.step_hook = acquire_then_finish
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(2)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(
            ROOT / "configs/logiv/prompts/pi05-subtasks-v10.json"
        ),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=2,
        max_total_action_steps=2,
        settling_steps=0,
        frontier_followup_steps=1,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o001", "task08-o000"),
        actions=(primary, first),
    )
    context = replace(_context(), occurrence_id="task08-o001")

    dispatch = executor.consume_permit_and_enqueue(
        primary,
        context,
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "frontier follow-up deadline reached"
    assert executor.results[-1].completion_mode == "DAG_FRONTIER"
    assert executor.results[-1].prompt_history == (
        "Put both moka pots on the stove.",
        "Put the moka pot you are holding on the stove and release it, then stop.",
    )


def test_transient_target_divergence_requires_consecutive_confirmation() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = package.proposal.candidate_subtasks[0].action
    object_name, source, target = action.arguments
    recovery = "kitchen_table_recovery_surface"
    env.env.relations.update(
        {
            ("on", object_name, source): True,
            ("on", object_name, "kitchen_table_moka_pot_right_init_region"): False,
            ("on", object_name, target): False,
            ("on", object_name, recovery): False,
            ("on", object_name, "kitchen_table"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_1", "kitchen_table"): True,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def transient_divergence_then_success(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", object_name, source)] = False
            env.env.relations[("on", object_name, recovery)] = True
        elif step == 2:
            env.env.relations[("on", object_name, recovery)] = False
            env.env.held.add(object_name)
        elif step == 3:
            env.env.held.remove(object_name)
            env.env.relations[("on", object_name, target)] = True

    env.step_hook = transient_divergence_then_success
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(4)]),
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

    assert outcome.reason == "observed declared effects"
    assert len(executor.results[-1].actions) == 4


def test_target_divergence_confirmation_is_independent_from_effect_confirmation() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    action = package.proposal.candidate_subtasks[0].action
    object_name, source, target = action.arguments
    recovery = "kitchen_table_recovery_surface"
    env.env.relations.update(
        {
            ("on", object_name, source): True,
            ("on", object_name, "kitchen_table_moka_pot_right_init_region"): False,
            ("on", object_name, target): False,
            ("on", object_name, recovery): False,
            ("on", object_name, "kitchen_table"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"): True,
            ("on", "moka_pot_1", "kitchen_table_moka_pot_left_init_region"): False,
            ("on", "moka_pot_1", target): False,
            ("on", "moka_pot_1", "kitchen_table"): True,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def recover_after_two_divergent_steps(
        step: int, low_level_action: np.ndarray
    ) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", object_name, source)] = False
            env.env.relations[("on", object_name, recovery)] = True
        elif step == 3:
            env.env.relations[("on", object_name, recovery)] = False
            env.env.relations[("on", object_name, target)] = True

    env.step_hook = recover_after_two_divergent_steps
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64) for _ in range(3)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=3,
        max_total_action_steps=3,
        settling_steps=0,
        effect_confirmation_steps=1,
        target_divergence_confirmation_steps=3,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed declared effects"
    assert len(executor.results[-1].actions) == 3


def test_different_frontier_prompts_fall_back_to_primary_occurrence_effects() -> None:
    env = FakeEnv()
    package, store, grounder = _grounder(env)
    first, second = [item.action for item in package.proposal.candidate_subtasks]
    target = first.arguments[2]
    env.env.relations.update(
        {
            ("on", first.arguments[0], first.arguments[1]): True,
            ("on", first.arguments[0], target): False,
            ("on", second.arguments[0], second.arguments[1]): True,
            ("on", second.arguments[0], target): False,
            ("turnon", "flat_stove_1"): True,
            ("turnoff", "flat_stove_1"): False,
        }
    )

    def complete_primary(step: int, low_level_action: np.ndarray) -> None:
        del low_level_action
        if step == 1:
            env.env.relations[("on", first.arguments[0], first.arguments[1])] = False
            env.env.relations[("on", first.arguments[0], target)] = True

    env.step_hook = complete_primary
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([np.zeros((1, 7), dtype=np.float64)]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=4,
        max_total_action_steps=4,
        settling_steps=0,
    )
    hint = ExecutionCompletionHint(
        occurrence_ids=("task08-o000", "task08-o001"),
        actions=(first, second),
    )

    dispatch = executor.consume_permit_and_enqueue(
        first,
        _context(),
        package.proposal.initial_snapshot,
        completion_hint=hint,
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed declared effects"
    assert len(executor.results[-1].actions) == 1
    assert executor.results[-1].completion_mode == "OCCURRENCE"
    assert executor.results[-1].completion_actions == (first,)


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
    task5 = provider.propose(5, epoch_id=0)
    book = FixedDomain().ground(
        task5.problem,
        "place-in",
        (
            "black_book_1",
            "study_table_black_book_init_region",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_access",
        ),
    )
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
    task5 = provider.propose(5, epoch_id=0)
    book = FixedDomain().ground(
        task5.problem,
        "place-in",
        (
            "black_book_1",
            "study_table_black_book_init_region",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_access",
        ),
    )
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


def test_v13_matches_official_task8_prompt_exactly() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v13.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    nominal = [item.action for item in package.proposal.candidate_subtasks]

    assert [renderer.render_phase(action, "acquire") for action in nominal] == [
        "put both moka pots on the stove",
        "put both moka pots on the stove",
    ]
    assert all(not renderer.has_phase(action, "finish") for action in nominal)


def test_v14_uses_exact_frontier_prompt_and_targeted_single_branch_recovery() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v14.json"
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

    assert renderer.prompt_version == "pi05-subtasks-v14"
    assert [renderer.render(action) for action in nominal] == [
        "put both moka pots on the stove",
        "put both moka pots on the stove",
    ]
    assert renderer.render_phase(nominal[1], "acquire") == (
        "Pick up the right moka pot from the table, then hold it securely."
    )
    assert renderer.has_phase(nominal[1], "finish")
    assert renderer.render_phase(left_recovery, "acquire") == (
        "Pick up the left moka pot from the table, then hold it securely."
    )
    assert renderer.render_phase(right_recovery, "acquire") == (
        "Pick up the right moka pot from the table, then hold it securely."
    )


def test_v15_separates_shared_frontier_prompt_from_relational_single_action_prompt() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v15.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == "put both moka pots on the stove"
    assert renderer.render_frontier(second) == "put both moka pots on the stove"
    assert renderer.render(second) == "put the remaining moka pot on the stove"
    assert "right" not in renderer.render(second)


def test_v16_uses_canonical_primary_prompt_then_remaining_object_prompt() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v16.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == (
        "put the moka pot closest to the stove on the stove"
    )
    assert renderer.render_frontier(second) == renderer.render_frontier(first)
    assert renderer.render(second) == "put the remaining moka pot on the stove"


def test_v17_adds_verified_holding_finish_phases_to_both_task8_branches() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v17.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == (
        "put the moka pot closest to the stove on the stove"
    )
    assert renderer.render_phase(first, "finish") == (
        "put the moka pot you are holding on the stove and release it"
    )
    assert renderer.render_phase(second, "acquire") == (
        "put the remaining moka pot on the stove"
    )
    assert renderer.render_phase(second, "finish") == renderer.render_phase(
        first, "finish"
    )


def test_v18_protects_the_already_placed_sibling_during_second_branch() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v18.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == (
        "put the moka pot closest to the stove on the stove"
    )
    assert "without moving the other moka pot" in renderer.render_phase(
        second, "acquire"
    )
    assert "beside the other moka pot" in renderer.render_phase(second, "finish")


def test_v22_keeps_official_task_prompt_continuous_across_initial_frontier() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v22.json"
    )
    v18 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v18.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == "put both moka pots on the stove"
    assert renderer.render_frontier(second) == renderer.render_frontier(first)
    assert not renderer.has_phase(first, "finish")
    assert renderer.render_phase(second, "acquire") == v18.render_phase(
        second, "acquire"
    )
    assert renderer.render_phase(second, "finish") == v18.render_phase(
        second, "finish"
    )


def test_v23_adds_official_fallback_without_changing_v18_fast_frontier() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v23.json"
    )
    v18 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v18.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == v18.render_frontier(first)
    assert renderer.render_frontier(second) == v18.render_frontier(second)
    assert renderer.render_frontier_fallback(first) == "put both moka pots on the stove"
    assert renderer.render_frontier_fallback(second) == (
        renderer.render_frontier_fallback(first)
    )


def test_v24_limits_official_recovery_prompt_to_post_initial_frontiers() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v24.json"
    )
    v23 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v23.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == v23.render_frontier(first)
    assert renderer.render_frontier(second) == v23.render_frontier(second)
    assert renderer.render_recovery_frontier(first) == "put both moka pots on the stove"
    assert renderer.render_recovery_frontier(second) == (
        renderer.render_recovery_frontier(first)
    )


def test_v25_keeps_initial_targeting_and_adds_recovery_and_sibling_prompts() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v25.json"
    )
    v23 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v23.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    assert renderer.render_frontier(first) == v23.render_frontier(first)
    assert renderer.render_frontier(second) == v23.render_frontier(second)
    assert renderer.render_recovery_frontier(first) == (
        "put the moka pot that was moved to another part of the kitchen table on the stove"
    )
    assert renderer.render_recovery_frontier(second) == (
        renderer.render_recovery_frontier(first)
    )
    assert renderer.render_frontier_completion(first) == (
        "put the remaining moka pot on the stove beside the other moka pot without moving the other moka pot"
    )


def test_v26_combines_v23_initial_v24_recovery_and_v25_completion() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v26.json"
    )
    v23 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v23.json"
    )
    v24 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v24.json"
    )
    v25 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v25.json"
    )
    package = ScriptedProposalProvider().propose(8, epoch_id=0)
    first, second = [item.action for item in package.proposal.candidate_subtasks]

    for action in (first, second):
        assert renderer.render_frontier(action) == v23.render_frontier(action)
        assert renderer.render_recovery_frontier(action) == (
            v24.render_recovery_frontier(action)
        )
        assert renderer.render_frontier_completion(action) == (
            v25.render_frontier_completion(action)
        )


def test_v27_preserves_v12_non_task8_and_v26_task8_frontier_prompts() -> None:
    merged = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v27.json"
    )
    general = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v12.json"
    )
    task8 = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v26.json"
    )
    provider = ScriptedProposalProvider()

    assert merged.templates == general.templates
    assert merged.labels == general.labels
    assert merged.suffix == general.suffix
    for key, value in general.overrides.items():
        if key not in task8.overrides:
            assert merged.overrides[key] == value
    for key, value in general.phase_overrides.items():
        if key not in task8.phase_overrides:
            assert merged.phase_overrides[key] == value

    for task_id in (*range(8), 9):
        package = provider.propose(task_id, epoch_id=0)
        for candidate in package.proposal.candidate_subtasks:
            action = candidate.action
            assert merged.render(action) == general.render(action)
            assert merged.render_phase(action, "acquire") == general.render_phase(
                action, "acquire"
            )
            assert merged.render_phase(action, "finish") == general.render_phase(
                action, "finish"
            )

    package = provider.propose(8, epoch_id=0)
    for candidate in package.proposal.candidate_subtasks:
        action = candidate.action
        assert merged.render(action) == task8.render(action)
        assert merged.render_frontier(action) == task8.render_frontier(action)
        assert merged.render_recovery_frontier(action) == (
            task8.render_recovery_frontier(action)
        )
        assert merged.render_frontier_completion(action) == (
            task8.render_frontier_completion(action)
        )


def test_v28_adds_state_specific_task5_and_task6_recovery_prompts() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v28.json"
    )
    provider = ScriptedProposalProvider()

    task5 = provider.propose(5, epoch_id=0)
    acquire, place = [
        item.action for item in task5.proposal.candidate_subtasks
    ]
    assert renderer.render(acquire) == (
        "Pick up the black book from the study table and keep holding it."
    )
    assert renderer.render(place) == (
        "Put the black book you are holding fully inside the back compartment "
        "of the desk caddy, release it, and move the gripper away."
    )

    task6 = provider.propose(6, epoch_id=0)
    mug_initial, pudding_initial = [
        item.action for item in task6.proposal.candidate_subtasks
    ]
    assert renderer.render(mug_initial) == "Put the white mug upright on the plate."
    assert renderer.render(pudding_initial) == (
        "Put the chocolate pudding to the right of the plate without touching "
        "the white mug."
    )
    mug_recovery = FixedDomain().ground(
        task6.problem,
        "place-on",
        ("porcelain_mug_1", "living_room_table_recovery_surface", "plate_1"),
    )
    pudding_recovery = FixedDomain().ground(
        task6.problem,
        "place-relative",
        (
            "chocolate_pudding_1",
            "living_room_table_recovery_surface",
            "living_room_table_plate_right_region",
        ),
    )
    assert renderer.render(mug_recovery) == (
        "Put the fallen white mug upright on the plate without moving the "
        "chocolate pudding."
    )
    assert renderer.render(pudding_recovery) == (
        "Put the chocolate pudding to the right of the plate without touching "
        "the white mug."
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


def test_close_access_stabilization_recovers_from_rebound_before_stop() -> None:
    env = FakeEnv()
    package = ScriptedProposalProvider().propose(3, epoch_id=4)
    action = next(
        item.action
        for item in package.proposal.candidate_subtasks
        if item.action.schema == "close-access"
    )
    store = LiberoObservationStore(dict(env.obs), epoch_id=4)

    class ReboundingGrounder:
        detector_calls = 0

        def __init__(self) -> None:
            self.progress = iter(
                [
                    (True, True, False),
                    (False, False, False),
                    (True, True, False),
                    (True, True, False),
                    (True, True, False),
                ]
            )

        def observe_action_progress_details(self, *args, **kwargs):
            del args, kwargs
            self.detector_calls += 1
            return next(self.progress)

        def peek_snapshot(self):
            return package.proposal.initial_snapshot

    grounder = ReboundingGrounder()
    policy_action = np.full((1, 7), 0.1, dtype=np.float64)
    executor = Pi05MacroExecutor(
        env=env,
        client=FakeClient([policy_action, policy_action]),
        image_tools=FakeImageTools(),
        observation_store=store,
        grounder=grounder,
        prompt_renderer=SubtaskPromptRenderer(),
        safety_supervisor=SimulatorSafetySupervisor(watchdog_seconds=60.0),
        replan_steps=1,
        max_action_steps=5,
        max_total_action_steps=5,
        settling_steps=0,
        effect_confirmation_steps=1,
        access_effect_stabilization_steps=2,
    )

    dispatch = executor.consume_permit_and_enqueue(
        action, _context(), package.proposal.initial_snapshot
    )
    outcome = executor.await_outcome(dispatch)

    assert outcome.reason == "observed stabilized declared effects"
    assert len(executor.results[-1].actions) == 5
    for index in (1, 3, 4):
        hold = executor.results[-1].actions[index]
        assert np.allclose(hold[:6], 0.0)
        assert hold[-1] == pytest.approx(0.1)


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


def test_v41_switches_only_task0_tomato_after_verified_holding() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v41-task0-holding-phase.json"
    )
    package = ScriptedProposalProvider(
        ROOT / "configs/logiv/libero10-scripted-proposals-v38-task0-recovery.json"
    ).propose(0, epoch_id=0)
    alphabet, tomato = [
        item.action for item in package.proposal.candidate_subtasks
    ]

    assert not renderer.has_phase(alphabet, "finish")
    assert renderer.render_phase(tomato, "acquire") == (
        "Pick up the tomato sauce can."
    )
    assert renderer.render_phase(tomato, "finish") == (
        "Put the tomato sauce can you are holding in the basket and release it."
    )


def test_v44_names_task2_by_appearance_without_a_false_side() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v44-task2-visual-identity.json"
    )
    prior = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v41-task0-holding-phase.json"
    )
    provider = ScriptedProposalProvider()
    task2 = provider.propose(2, epoch_id=0)
    place = next(
        item.action
        for item in task2.proposal.candidate_subtasks
        if item.action.schema == "place-on"
    )

    assert renderer.render(place) == (
        "put the silver moka pot with the black handle on the stove; "
        "do not move the frying pan"
    )
    for candidate in provider.propose(8, epoch_id=0).proposal.candidate_subtasks:
        assert renderer.render(candidate.action) == prior.render(candidate.action)


def test_v45_closes_task3_drawer_fully_without_changing_task2() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v45-task3-full-close.json"
    )
    prior = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v44-task2-visual-identity.json"
    )
    provider = ScriptedProposalProvider()
    task3 = provider.propose(3, epoch_id=0)
    close = next(
        item.action
        for item in task3.proposal.candidate_subtasks
        if item.action.schema == "close-access"
    )

    assert renderer.render(close) == (
        "Push the bottom drawer fully closed until its front is flush with the cabinet."
    )
    for candidate in provider.propose(2, epoch_id=0).proposal.candidate_subtasks:
        assert renderer.render(candidate.action) == prior.render(candidate.action)


def test_v51_keeps_task9_downstream_context_without_skipping_gates() -> None:
    renderer = SubtaskPromptRenderer(
        ROOT / "configs/logiv/prompts/pi05-subtasks-v51-task9-context.json"
    )
    actions = [
        item.action
        for item in ScriptedProposalProvider()
        .propose(9, epoch_id=0)
        .proposal.candidate_subtasks
    ]
    place = next(action for action in actions if action.schema == "place-in")
    close = next(action for action in actions if action.schema == "close-access")
    task_prompt = "Put the yellow and white mug in the microwave and close it."

    assert renderer.render_phase(place, "acquire") == task_prompt
    assert renderer.render(close) == task_prompt


class _ReadOnlyTransitionInner:
    def __init__(self) -> None:
        self.holding: dict[str, bool | None] = {}
        self.contacts: dict[str, int | None] = {}
        self.region_truth: dict[tuple[str, str], bool | None] = {}
        self.region_distance: dict[tuple[str, str], float | None] = {}
        self.object_positions: dict[str, np.ndarray | None] = {}

    def _eval_predicate(self, state):
        object_id, region = str(state[1]), str(state[2])
        value = self.region_truth.get((object_id, region))
        if value is None:
            raise KeyError((object_id, region))
        return value

    def read_logiv_holding(self, object_id: str):
        return self.holding.get(object_id)

    def read_logiv_contact_count(self, object_id: str):
        return self.contacts.get(object_id)

    def read_logiv_region_truth(self, object_id: str, region: str):
        return self.region_truth.get((object_id, region))

    def read_logiv_region_distance(self, object_id: str, region: str):
        return self.region_distance.get((object_id, region))

    def read_logiv_object_position(self, object_id: str):
        value = self.object_positions.get(object_id)
        return None if value is None else np.array(value, copy=True)


class _MutationTrapTransitionEnv:
    def __init__(self) -> None:
        self.env = _ReadOnlyTransitionInner()
        self.step_calls = 0
        self.reset_calls = 0
        self.set_state_calls = 0

    def step(self, action):
        del action
        self.step_calls += 1
        raise AssertionError("transition reader must not step")

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("transition reader must not reset")

    def set_state(self, state):
        del state
        self.set_state_calls += 1
        raise AssertionError("transition reader must not mutate simulator state")


def _transition_context(
    step: int,
    *,
    gripper: float | None = 0.02,
    eef_x: float = 0.0,
    pixel: int = 0,
) -> ShadowStepContext:
    observation = {
        "agentview_image": np.full((2, 2, 3), pixel, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([eef_x, 0.0, 0.5]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    if gripper is not None:
        observation["robot0_gripper_qpos"] = np.array([gripper])
    return ShadowStepContext(
        observation=observation,
        last_action=None if step == 0 else np.zeros(7),
        policy_step=step,
        base_policy_request_count=1,
        active_base_request_index=0,
        next_base_request_index=1,
        active_base_request_envelope_json=None,
        next_base_replay_envelope_json=None,
        base_action_response_size=1,
        base_action_chunk_size=1,
        pending_base_action_offset=0,
        pending_base_actions=np.empty((0, 7)),
        base_action_prefix_sha256="1" * 64,
    )


def _transition_reader_fixture():
    env = _MutationTrapTransitionEnv()
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage.json", 5
    )
    contract = load_monitor_evidence_contract(
        ROOT / "configs/logiv/r2m-monitor-evidence-v1.json", task_id=5
    )
    for rule in contract.action_event_rules:
        env.env.holding[rule.object_id] = False
        env.env.contacts[rule.object_id] = 0
        env.env.object_positions[rule.object_id] = np.array([0.0, 0.0, 0.0])
        for region in (
            rule.source_region,
            rule.destination_region,
            *contract.abnormal_support_surfaces,
        ):
            if region is None:
                continue
            env.env.region_truth[(rule.object_id, region)] = False
            env.env.region_distance[(rule.object_id, region)] = 1.0
    reader = build_libero_transition_feature_reader(env, binding, contract)
    return env, contract, reader


def test_transition_feature_reader_is_read_only_version_bound_and_complete() -> None:
    env, contract, reader = _transition_reader_fixture()
    rows = reader(_transition_context(0))

    assert reader.monitor_contract_sha256 == contract.contract_sha256
    assert reader.tracker_version == contract.tracker_version
    assert reader.rule_ids == tuple(rule.rule_id for rule in contract.action_event_rules)
    assert tuple(item.rule_id for item in rows) == reader.rule_ids
    assert all(item.policy_step == 0 for item in rows)
    assert all(item.gripper_qpos == 0.02 for item in rows)
    assert all(item.transition_sha256 for item in rows)
    assert "env" not in vars(reader)
    assert (env.step_calls, env.reset_calls, env.set_state_calls) == (0, 0, 0)


def test_transition_hash_ignores_pixels_but_binds_registered_values() -> None:
    _, _, first = _transition_reader_fixture()
    _, _, noisy = _transition_reader_fixture()
    _, _, changed = _transition_reader_fixture()

    first_hashes = tuple(
        item.transition_sha256 for item in first(_transition_context(0, pixel=0))
    )
    noisy_hashes = tuple(
        item.transition_sha256 for item in noisy(_transition_context(0, pixel=255))
    )
    changed_hashes = tuple(
        item.transition_sha256
        for item in changed(_transition_context(0, gripper=-0.02, pixel=0))
    )
    assert first_hashes == noisy_hashes
    assert first_hashes != changed_hashes


def test_transition_reader_missing_keys_return_null_unknown_without_mutation() -> None:
    env, _, reader = _transition_reader_fixture()
    env.env.holding.clear()
    env.env.contacts.clear()
    env.env.object_positions.clear()
    env.env.region_truth.clear()
    env.env.region_distance.clear()

    rows = reader(_transition_context(0, gripper=None))
    assert rows
    assert all(item.gripper_qpos is None for item in rows)
    assert all(item.contact_count is None for item in rows)
    assert all(item.holding is TruthValue.UNKNOWN for item in rows)
    assert all(item.source_region_truth is TruthValue.UNKNOWN for item in rows)
    assert all(item.destination_region_truth is TruthValue.UNKNOWN for item in rows)
    assert (env.step_calls, env.reset_calls, env.set_state_calls) == (0, 0, 0)


def test_transition_reader_computes_registered_motion_correlation_after_step_zero() -> None:
    env, _, reader = _transition_reader_fixture()
    reader(_transition_context(0, eef_x=0.0))
    for object_id in env.env.object_positions:
        env.env.object_positions[object_id] = np.array([0.2, 0.0, 0.0])
    rows = reader(_transition_context(1, eef_x=0.2))
    assert all(item.object_eef_motion_correlation == pytest.approx(1.0) for item in rows)


def test_transition_reader_grounds_support_alias_from_read_only_workspace_geometry() -> None:
    class GeometryInner:
        workspace_offset = np.array([0.0, 0.0, 0.8])
        table_full_size = np.array([1.0, 1.0, 0.1])
        obj_body_id = {"black_book_1": 0}
        sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.array([[0.0, 0.0, 0.85]]),
                get_site_xpos=lambda name: (_ for _ in ()).throw(KeyError(name)),
            )
        )
        objects_dict = {"black_book_1": FakeObject("black_book_1")}
        robots = [FakeRobot()]

        def _eval_predicate(self, state):
            raise KeyError(tuple(state))

        def _check_grasp(self, gripper, geoms):
            del gripper, geoms
            return False

        def get_object(self, object_id):
            return self.objects_dict[object_id]

    env = SimpleNamespace(env=GeometryInner())
    binding = TaskBinding.from_manifest(
        ROOT / "configs/logiv/libero10-coverage.json", 5
    )
    contract = load_monitor_evidence_contract(
        ROOT / "configs/logiv/r2m-monitor-evidence-v1.json", task_id=5
    )
    rows = build_libero_transition_feature_reader(env, binding, contract)(
        _transition_context(0)
    )

    assert all(item.abnormal_region_truth is TruthValue.TRUE for item in rows)
    assert all(item.abnormal_region_id == "study_table_recovery_surface" for item in rows)
    assert all(item.abnormal_region_distance == pytest.approx(0.0) for item in rows)
