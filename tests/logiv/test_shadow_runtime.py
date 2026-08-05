from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from pi05_libero_repro.logiv.dag import (
    CausalGraph,
    CausalLink,
    GraphEdge,
    GraphNode,
    NodeKind,
    SignedLiteral,
)
from pi05_libero_repro.logiv.initial_proposal import InitialProposalStatus
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    GoalMode,
    GroundAction,
    ObjectDecl,
    TaskProblem,
    TruthValue,
    fact_universe_sha256,
)
from pi05_libero_repro.logiv.recovery_records import CollectionLabel
from pi05_libero_repro.logiv.shadow_monitor import MonitorEvidenceContract
from pi05_libero_repro.logiv.shadow_runtime import (
    ShadowEpisodeContext,
    ShadowRuntime,
    ShadowRuntimeCounters,
    ShadowValidatedProposal,
    build_shadow_runtime,
)
from pi05_libero_repro.protocol import (
    BaseActionPrefixHasher,
    ShadowSettlingContext,
    ShadowStepContext,
    run_episode,
)


OBJECT = "black_book_1"
SOURCE = "table_region"
TARGET = "caddy_region"
ABNORMAL = "recovery_surface"
AT_SOURCE = Fact("at", (OBJECT, SOURCE))
AT_TARGET = Fact("at", (OBJECT, TARGET))
AT_ABNORMAL = Fact("at", (OBJECT, ABNORMAL))
HOLDING = Fact("holding", (OBJECT,))
HANDEMPTY = Fact("handempty")
UNIVERSE = frozenset({AT_SOURCE, AT_TARGET, AT_ABNORMAL, HOLDING, HANDEMPTY})


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot(
    step: int, *, abnormal: bool = False, goal: bool = False
) -> FactSnapshot:
    true_facts = frozenset(
        {AT_TARGET, HANDEMPTY}
        if goal
        else {AT_ABNORMAL, HANDEMPTY}
        if abnormal
        else {AT_SOURCE, HANDEMPTY}
    )
    false_facts = UNIVERSE - true_facts
    values = [
        [
            fact.pddl(),
            (
                TruthValue.TRUE.value
                if fact in true_facts
                else TruthValue.FALSE.value
            ),
        ]
        for fact in sorted(UNIVERSE, key=lambda item: item.pddl())
    ]
    payload_json = _canonical_json(
        {
            "dominance_overrides": [],
            "epoch_id": step,
            "observation_hash": _sha256(f"observation:{step}:{abnormal}:{goal}"),
            "values": values,
        }
    )
    return FactSnapshot(
        epoch_id=step,
        true_facts=true_facts,
        false_facts=false_facts,
        evidence_hash=_sha256(payload_json),
        fact_universe=UNIVERSE,
        fact_universe_version="runtime-test-v1",
        fact_universe_sha256=fact_universe_sha256("runtime-test-v1", UNIVERSE),
        evidence_payload_json=payload_json,
    )


def _certified(*, source_epoch: int = 0) -> Any:
    action = GroundAction(
        schema="place-in",
        arguments=(OBJECT, SOURCE, TARGET),
        preconditions=frozenset({AT_SOURCE}),
        add_effects=frozenset({AT_TARGET}),
        del_effects=frozenset({AT_SOURCE}),
        repeatable=False,
    )
    problem = TaskProblem(
        name="runtime_test",
        objects=(
            ObjectDecl(OBJECT, "movable"),
            ObjectDecl(SOURCE, "relative-region"),
            ObjectDecl(TARGET, "container-region"),
            ObjectDecl(ABNORMAL, "surface"),
        ),
        initial_state=frozenset({AT_SOURCE, HANDEMPTY}),
        initial_false=frozenset({AT_TARGET, AT_ABNORMAL, HOLDING}),
        goal=frozenset({AT_TARGET}),
    )
    certificate_hash = "c" * 64
    graph = CausalGraph(
        graph_version="runtime-graph-v1",
        graph_hash="d" * 64,
        source_epoch=source_epoch,
        certificate_hash=certificate_hash,
        nodes=(
            GraphNode("INIT", NodeKind.INIT, 0),
            GraphNode("a0", NodeKind.ACTION, 1, action=action),
            GraphNode("GOAL", NodeKind.GOAL, 2),
        ),
        edges=(
            GraphEdge(
                "INIT",
                "a0",
                support_literals=frozenset({SignedLiteral(AT_SOURCE, True)}),
            ),
            GraphEdge(
                "a0",
                "GOAL",
                support_literals=frozenset({SignedLiteral(AT_TARGET, True)}),
            ),
        ),
        causal_links=(
            CausalLink("INIT", SignedLiteral(AT_SOURCE, True), "a0"),
            CausalLink("a0", SignedLiteral(AT_TARGET, True), "GOAL"),
        ),
        canonical_agenda=("a0",),
    )
    return SimpleNamespace(
        problem=problem,
        plan=(action,),
        graph=graph,
        certificate=SimpleNamespace(certificate_hash=certificate_hash),
    )


def _contract(*, interval: int = 1, confirmations: int = 2) -> MonitorEvidenceContract:
    payload = {
        "contract_id": "runtime-test-monitor-v1",
        "task_id": 5,
        "object_ids": [OBJECT],
        "nominal_source_facts": [AT_SOURCE.pddl()],
        "abnormal_support_surfaces": [ABNORMAL],
        "task_relevant_effects": [AT_TARGET.pddl()],
        "tracker_version": "runtime-test-tracker-v1",
        "action_event_rules": [],
        "monitor_interval_steps": interval,
        "confirmation_count": confirmations,
        "settling_grace_observations": 1,
        "progress_window_observations": 2,
        "progress_evidence_ttl_policy_steps": 5,
        "goal_regression_evidence_ttl_policy_steps": 5,
        "max_active_attempts_per_object": 2,
        "max_attempt_records_per_episode": 4,
        "max_evidence_records_per_episode": 4,
        "grounding_rule_sha256": "1" * 64,
        "event_detector_sha256": "2" * 64,
    }
    payload["contract_sha256"] = _sha256(_canonical_json(payload))
    return MonitorEvidenceContract.from_mapping(payload)


class _Provider:
    def __init__(self, *, epoch_id: int = 0) -> None:
        self.calls = 0
        self.package = SimpleNamespace(
            proposal=SimpleNamespace(epoch_id=epoch_id)
        )

    def propose(self, task_id: int, epoch_id: int, goal_mode: GoalMode) -> Any:
        self.calls += 1
        return self.package


class _FeatureReader:
    tracker_version = "runtime-test-tracker-v1"
    rule_ids: tuple[str, ...] = ()

    def __init__(self, contract_sha256: str) -> None:
        self.monitor_contract_sha256 = contract_sha256

    def __call__(self, context: ShadowStepContext) -> tuple[()]:
        return ()


def _episode_context(contract: MonitorEvidenceContract) -> ShadowEpisodeContext:
    return ShadowEpisodeContext(
        task_id=5,
        episode_idx=3,
        initial_epoch_id=0,
        scene_sha256="3" * 64,
        object_instance_ids=(OBJECT, SOURCE, TARGET, ABNORMAL),
        initial_state_sha256="4" * 64,
        parent_trajectory_lineage_sha256="5" * 64,
        base_prompt_sha256="6" * 64,
        base_checkpoint_sha256="7" * 64,
        policy_client_config_sha256="8" * 64,
        policy_replay_contract_sha256="9" * 64,
        master_seed=11,
        policy_seed=12,
        simulator_seed=13,
        replan_steps=2,
        collect_recovery_roots=True,
        collection_label=CollectionLabel.DEV_COLLECTION,
        root_output_dir=Path("/tmp/runtime-test-roots"),
        simulator_state_reader=lambda: np.zeros(4),
        transition_feature_reader=_FeatureReader(contract.contract_sha256),
    )


def _context(
    step: int,
    hasher: BaseActionPrefixHasher,
    *,
    abnormal: bool = False,
    digest: str | None = None,
) -> ShadowStepContext:
    action = None if step == 0 else np.full(7, step, dtype=np.float32)
    expected_digest = hasher.update_and_hexdigest(action)
    request_index = None if step == 0 else (step - 1) // 2
    request_count = 0 if request_index is None else request_index + 1
    chunk_offset = 0 if step == 0 else (step - 1) % 2 + 1
    pending_count = 0 if step == 0 else 2 - chunk_offset
    return ShadowStepContext(
        observation={"policy_step": step, "abnormal": abnormal},
        last_action=action,
        policy_step=step,
        base_policy_request_count=request_count,
        active_base_request_index=request_index,
        next_base_request_index=request_count,
        active_base_request_envelope_json=(
            None if request_index is None else f"envelope-{request_index}"
        ),
        next_base_replay_envelope_json=f"envelope-{request_count}",
        base_action_response_size=None if step == 0 else 4,
        base_action_chunk_size=0 if step == 0 else 2,
        pending_base_action_offset=chunk_offset,
        pending_base_actions=np.empty((pending_count, 7), dtype=np.float32),
        base_action_prefix_sha256=digest or expected_digest,
    )


def _build(
    provider: _Provider,
    *,
    validator_error: Exception | None = None,
    source_epoch: int = 0,
    collector=lambda trigger, context, episode_context: object(),
    topology_only: bool = False,
    interval_steps: int = 1,
):
    contract = _contract()

    def snapshot_reader(observation: Mapping[str, Any]) -> FactSnapshot:
        return _snapshot(
            int(observation["policy_step"]),
            abnormal=bool(observation["abnormal"]),
            goal=bool(observation.get("goal", False)),
        )

    def live_validator(package: Any, observation: Mapping[str, Any]) -> ShadowValidatedProposal:
        if validator_error is not None:
            raise validator_error
        assert observation["policy_step"] == 0
        return ShadowValidatedProposal(_certified(source_epoch=source_epoch), snapshot_reader)

    runtime = build_shadow_runtime(
        provider=provider,
        provider_name="runtime-test-provider",
        episode_context=_episode_context(contract),
        goal_mode=GoalMode.METADATA_ASSISTED,
        live_validator=live_validator,
        monitor_contract=None if topology_only else contract,
        root_collector=collector,
        interval_steps=interval_steps,
        confirmation_count=2,
        topology_only=topology_only,
    )
    return runtime


def test_provider_is_lazy_and_acceptance_seeds_monitor_at_step_zero() -> None:
    provider = _Provider()
    runtime = _build(provider)
    assert provider.calls == 0
    assert runtime.initial_proposal is None
    assert runtime.monitor is None

    hasher = BaseActionPrefixHasher()
    runtime.observer(_context(0, hasher))

    assert provider.calls == 1
    assert runtime.initial_proposal is not None
    assert runtime.initial_proposal.status is InitialProposalStatus.ACCEPTED
    assert runtime.monitor is not None
    assert runtime.monitor.metrics.snapshot_calls == 1
    assert runtime.monitor.action_event_tracker.attempt_record_count == 0
    assert runtime.monitor.metrics.anomaly_candidates == 0
    assert runtime.monitor.metrics.confirmed_deviations == 0


def test_topology_only_records_fixed_graph_states_without_recovery_monitoring() -> None:
    runtime = _build(_Provider(), topology_only=True)
    hasher = BaseActionPrefixHasher()

    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher))

    assert runtime.monitor is None
    assert runtime.counters.root_count == 0
    assert [state["policy_step"] for state in runtime.state_trace] == [0, 1]
    assert [state["observation_generation"] for state in runtime.state_trace] == [0, 1]
    assert all(
        [node["node_id"] for node in state["nodes"]] == ["INIT", "a0", "GOAL"]
        for state in runtime.state_trace
    )


def test_topology_only_records_every_callback_even_with_sparse_monitor_interval() -> None:
    runtime = _build(_Provider(), topology_only=True, interval_steps=5)
    hasher = BaseActionPrefixHasher()

    for step in range(4):
        runtime.observer(_context(step, hasher))

    assert [state["policy_step"] for state in runtime.state_trace] == [0, 1, 2, 3]
    assert [state["observation_generation"] for state in runtime.state_trace] == [
        0,
        1,
        2,
        3,
    ]


def test_shadow_runtime_fifth_positional_argument_remains_state_trace() -> None:
    counters = ShadowRuntimeCounters()
    trace = [{"policy_step": 0}]

    runtime = ShadowRuntime(None, None, None, counters, trace)

    assert runtime.state_trace is trace
    assert runtime.settling_observer is None


def test_topology_only_settling_regresses_a_transient_goal_in_the_same_graph() -> None:
    runtime = _build(_Provider(), topology_only=True)
    hasher = BaseActionPrefixHasher()
    initial = _context(0, hasher)
    goal = _context(1, hasher)
    goal.observation["goal"] = True

    runtime.observer(initial)
    runtime.observer(goal)
    assert runtime.state_trace[-1]["nodes"][-1]["status"] == "COMPLETED"

    assert runtime.settling_observer is not None
    runtime.settling_observer(
        ShadowSettlingContext(
            observation={"policy_step": 1, "abnormal": False, "goal": False},
            policy_step=1,
            settling_step=1,
            settling_steps=1,
        )
    )

    assert runtime.state_trace[-1]["phase"] == "SETTLING"
    assert runtime.state_trace[-1]["settling_step"] == 1
    assert runtime.state_trace[-1]["nodes"][-1]["status"] == "BLOCKED"


def test_live_validation_rejection_is_stored_and_later_callbacks_are_noops() -> None:
    provider = _Provider()
    runtime = _build(provider, validator_error=ValueError("VAL rejected"))
    hasher = BaseActionPrefixHasher()

    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher))

    assert provider.calls == 1
    assert runtime.initial_proposal is not None
    assert runtime.initial_proposal.status is InitialProposalStatus.REJECTED
    assert runtime.initial_proposal.reason == "ValueError: VAL rejected"
    assert runtime.monitor is None
    assert runtime.counters == type(runtime.counters)()


def test_epoch_mismatch_is_one_fail_open_rejection() -> None:
    provider = _Provider(epoch_id=1)
    runtime = _build(provider)
    hasher = BaseActionPrefixHasher()

    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher))

    assert provider.calls == 1
    assert runtime.initial_proposal is not None
    assert runtime.initial_proposal.status is InitialProposalStatus.REJECTED
    assert runtime.initial_proposal.reason == "ValueError: proposal epoch mismatch"
    assert runtime.monitor is None


def test_root_writer_failure_has_exactly_one_owner_and_runtime_remains_callable() -> None:
    provider = _Provider()

    def fail_write(trigger, context, episode_context):
        raise OSError("disk")

    runtime = _build(provider, collector=fail_write)
    hasher = BaseActionPrefixHasher()
    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher, abnormal=True))
    runtime.observer(_context(2, hasher, abnormal=True))
    runtime.observer(_context(3, hasher, abnormal=False))

    assert runtime.monitor is not None
    assert runtime.counters.root_write_errors == 1
    assert runtime.counters.root_count == 0
    assert runtime.monitor.metrics.snapshot_errors == 0
    assert runtime.monitor.metrics.event_tracker_errors == 0
    assert runtime.monitor.metrics.trigger_callback_errors == 0


def test_collector_gets_unmodified_protocol_context_and_episode_lineage() -> None:
    received = []
    provider = _Provider()

    def collect(trigger, context, episode_context):
        received.append((trigger, context, episode_context))
        return object()

    runtime = _build(provider, collector=collect)
    hasher = BaseActionPrefixHasher()
    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher, abnormal=True))
    emitted_context = _context(2, hasher, abnormal=True)
    runtime.observer(emitted_context)

    assert len(received) == 1
    _, captured, episode = received[0]
    assert captured is emitted_context
    assert captured.base_policy_request_count == 1
    assert captured.active_base_request_index == 0
    assert captured.next_base_request_index == 1
    assert captured.pending_base_action_offset == 2
    assert captured.active_base_request_envelope_json == "envelope-0"
    assert captured.base_action_prefix_sha256 == emitted_context.base_action_prefix_sha256
    assert episode.parent_trajectory_lineage_sha256 == "5" * 64
    assert (episode.master_seed, episode.policy_seed, episode.simulator_seed) == (11, 12, 13)
    assert runtime.counters.root_count == 1


def test_action_prefix_mismatch_disables_monitoring_in_provenance_bucket_only() -> None:
    provider = _Provider()
    runtime = _build(provider)
    hasher = BaseActionPrefixHasher()
    runtime.observer(_context(0, hasher))
    runtime.observer(_context(1, hasher, digest="f" * 64))
    runtime.observer(_context(2, hasher, abnormal=True))

    assert runtime.counters.provenance_errors == 1
    assert runtime.counters.proposal_callback_errors == 0
    assert runtime.counters.root_write_errors == 0
    assert runtime.monitor is not None
    assert runtime.monitor.metrics.snapshot_calls == 1


def test_rng_consuming_proposal_keeps_base_rollout_bitwise_identical() -> None:
    class ImageTools:
        @staticmethod
        def resize_with_pad(image, height, width):
            return image

        @staticmethod
        def convert_to_uint8(image):
            return image.astype(np.uint8)

    def observation():
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    class Env:
        def __init__(self):
            self.policy_steps = 0

        def reset(self):
            return observation()

        def set_init_state(self, initial_state):
            return observation()

        def step(self, action):
            self.policy_steps += 1
            return observation(), 0.0, self.policy_steps >= 3, {}

        def check_success(self):
            return self.policy_steps >= 3

    class Client:
        def __init__(self):
            self.inputs = []

        def infer(self, element):
            self.inputs.append(
                {
                    key: np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in element.items()
                }
            )
            return {"actions": np.ones((3, 7), dtype=np.float32)}

    class ConsumingProvider:
        provider = "rng-consuming-provider"

        def __init__(self):
            self.calls = 0

        def propose(self, task_id, epoch_id, goal_mode):
            self.calls += 1
            random.random()
            np.random.random()
            raise RuntimeError("offline")

    baseline_client = Client()
    random.seed(123)
    np.random.seed(123)
    baseline = run_episode(
        Env(),
        baseline_client,
        np.zeros(1),
        "prompt",
        ImageTools(),
        wait_steps=0,
        max_steps=3,
        replan_steps=3,
    )
    baseline_draws = (random.random(), np.random.random())

    contract = _contract()
    provider = ConsumingProvider()
    runtime = build_shadow_runtime(
        provider=provider,
        provider_name=provider.provider,
        episode_context=_episode_context(contract),
        goal_mode=GoalMode.METADATA_ASSISTED,
        live_validator=lambda package, observation: (_ for _ in ()).throw(
            AssertionError("provider failure must skip validation")
        ),
        monitor_contract=contract,
        root_collector=lambda trigger, context, episode_context: None,
        interval_steps=1,
        confirmation_count=2,
    )
    shadow_client = Client()
    random.seed(123)
    np.random.seed(123)
    shadow = run_episode(
        Env(),
        shadow_client,
        np.zeros(1),
        "prompt",
        ImageTools(),
        wait_steps=0,
        max_steps=3,
        replan_steps=3,
        shadow_observer=runtime.observer,
    )
    shadow_draws = (random.random(), np.random.random())

    assert provider.calls == 1
    assert runtime.initial_proposal is not None
    assert runtime.initial_proposal.status is InitialProposalStatus.REJECTED
    assert shadow_draws == baseline_draws
    assert shadow.inference_requests == baseline.inference_requests
    np.testing.assert_array_equal(shadow.actions, baseline.actions)
    assert len(shadow_client.inputs) == len(baseline_client.inputs)
    for shadow_input, baseline_input in zip(
        shadow_client.inputs, baseline_client.inputs
    ):
        assert shadow_input.keys() == baseline_input.keys()
        for key in shadow_input:
            if isinstance(shadow_input[key], np.ndarray):
                np.testing.assert_array_equal(
                    shadow_input[key], baseline_input[key]
                )
            else:
                assert shadow_input[key] == baseline_input[key]
