from __future__ import annotations

import hashlib
import json

from pi05_libero_repro.logiv import shadow_runtime
from pi05_libero_repro.logiv.dag import CausalGraph, GraphEdge, GraphNode, NodeKind, SignedLiteral
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    GroundAction,
    ObjectDecl,
    TaskProblem,
    TruthValue,
    fact_universe_sha256,
)
import pytest

from scripts.eval_logiv_libero import _graph_json, _parser, _run_config, _validate_shadow_options


AT_SOURCE = Fact("at", ("book", "table"))
AT_TARGET = Fact("at", ("book", "caddy"))
HELD = Fact("holding", ("book",))
BLOCKED = Fact("open", ("drawer",))
UNKNOWN = Fact("clear", ("caddy",))
INSPECTED = Fact("inspected", ("caddy",))
BLOCKED_EFFECT = Fact("blocked-effect", ("drawer",))
UNIVERSE = frozenset({AT_SOURCE, AT_TARGET, HELD, BLOCKED, UNKNOWN, INSPECTED, BLOCKED_EFFECT})


def _graph() -> tuple[CausalGraph, TaskProblem]:
    first = GroundAction(
        schema="pick",
        arguments=("book", "table"),
        preconditions=frozenset({AT_SOURCE}),
        add_effects=frozenset({HELD}),
        del_effects=frozenset({AT_SOURCE}),
        repeatable=False,
    )
    ready = GroundAction(
        schema="place",
        arguments=("book", "caddy"),
        preconditions=frozenset({HELD}),
        negative_preconditions=frozenset({BLOCKED}),
        add_effects=frozenset({AT_TARGET}),
        del_effects=frozenset({HELD}),
        repeatable=False,
    )
    unknown = GroundAction(
        schema="inspect",
        arguments=("caddy",),
        preconditions=frozenset({UNKNOWN}),
        add_effects=frozenset({INSPECTED}),
        del_effects=frozenset(),
        repeatable=False,
    )
    blocked = GroundAction(
        schema="open",
        arguments=("drawer",),
        preconditions=frozenset({BLOCKED}),
        add_effects=frozenset({BLOCKED_EFFECT}),
        del_effects=frozenset(),
        repeatable=False,
    )
    graph = CausalGraph(
        graph_version="trace-v1",
        graph_hash="g" * 64,
        source_epoch=0,
        certificate_hash="c" * 64,
        nodes=(
            GraphNode("INIT", NodeKind.INIT, 0),
            GraphNode("first", NodeKind.ACTION, 1, action=first),
            GraphNode("ready", NodeKind.ACTION, 2, action=ready),
            GraphNode("unknown", NodeKind.ACTION, 3, action=unknown),
            GraphNode("blocked", NodeKind.ACTION, 4, action=blocked),
            GraphNode("GOAL", NodeKind.GOAL, 5),
        ),
        edges=(
            GraphEdge("INIT", "first", frozenset({SignedLiteral(AT_SOURCE, True)})),
            GraphEdge("first", "ready", frozenset({SignedLiteral(HELD, True)})),
            GraphEdge("ready", "GOAL", frozenset({SignedLiteral(AT_TARGET, True)})),
        ),
        causal_links=(),
        canonical_agenda=("first", "ready", "unknown", "blocked"),
    )
    return graph, TaskProblem(
        name="trace",
        objects=(),
        initial_state=frozenset(),
        initial_false=frozenset(),
        goal=frozenset({AT_TARGET}),
    )


def _snapshot(*, true: frozenset[Fact], false: frozenset[Fact]) -> FactSnapshot:
    return FactSnapshot(
        epoch_id=0,
        true_facts=true,
        false_facts=false,
        evidence_hash="e" * 64,
    )


def _temporal_graph() -> tuple[CausalGraph, TaskProblem]:
    action = GroundAction(
        schema="place-on",
        arguments=("book", "table", "caddy"),
        preconditions=frozenset({AT_SOURCE, Fact("handempty")}),
        add_effects=frozenset({AT_TARGET}),
        del_effects=frozenset({AT_SOURCE}),
        repeatable=False,
    )
    graph = CausalGraph(
        graph_version="temporal-v1",
        graph_hash="t" * 64,
        source_epoch=0,
        certificate_hash="c" * 64,
        nodes=(
            GraphNode("INIT", NodeKind.INIT, 0),
            GraphNode("place", NodeKind.ACTION, 1, action=action),
            GraphNode("GOAL", NodeKind.GOAL, 2),
        ),
        edges=(
            GraphEdge("INIT", "place", frozenset({SignedLiteral(AT_SOURCE, True)})),
            GraphEdge("place", "GOAL", frozenset({SignedLiteral(AT_TARGET, True)})),
        ),
        causal_links=(),
        canonical_agenda=("place",),
    )
    problem = TaskProblem(
        name="temporal",
        objects=(
            ObjectDecl("book", "movable"),
            ObjectDecl("table", "surface"),
            ObjectDecl("caddy", "surface"),
        ),
        initial_state=frozenset({AT_SOURCE, Fact("handempty")}),
        initial_false=frozenset({AT_TARGET, HELD}),
        goal=frozenset({AT_TARGET}),
    )
    return graph, problem


def _audited_temporal_snapshot(
    epoch: int,
    *,
    true: frozenset[Fact],
    unknown: frozenset[Fact] = frozenset(),
    raw_target_while_held: bool = False,
) -> FactSnapshot:
    universe = frozenset({AT_SOURCE, AT_TARGET, HELD, Fact("handempty")})
    false = universe - true - unknown
    overrides = (
        [[HELD.pddl(), AT_TARGET.pddl(), "reliable-holding-over-at"]]
        if raw_target_while_held
        else []
    )
    payload = {
        "dominance_overrides": overrides,
        "epoch_id": epoch,
        "observation_hash": hashlib.sha256(f"temporal:{epoch}".encode()).hexdigest(),
        "values": [
            [
                fact.pddl(),
                (
                    TruthValue.TRUE
                    if fact in true
                    else TruthValue.UNKNOWN
                    if fact in unknown
                    else TruthValue.FALSE
                ).value,
            ]
            for fact in sorted(universe, key=lambda item: item.pddl())
        ],
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    version = "temporal-test-v1"
    return FactSnapshot(
        epoch_id=epoch,
        true_facts=true,
        false_facts=false,
        evidence_hash=hashlib.sha256(payload_json.encode()).hexdigest(),
        fact_universe=universe,
        fact_universe_version=version,
        fact_universe_sha256=fact_universe_sha256(version, universe),
        evidence_payload_json=payload_json,
    )


def test_shadow_trace_projects_completed_ready_blocked_unknown_and_goal_in_graph_order() -> None:
    project = getattr(shadow_runtime, "project_graph_state", None)
    assert project is not None
    graph, problem = _graph()
    state = project(
        graph,
        problem,
        _snapshot(
            true=frozenset({HELD}),
            false=frozenset({AT_SOURCE, AT_TARGET, BLOCKED}),
        ),
        policy_step=10,
        observation_generation=2,
        certificate_state="CURRENT",
    )

    assert state == {
        "policy_step": 10,
        "observation_generation": 2,
        "certificate_state": "CURRENT",
        "graph_version": "trace-v1",
        "graph_hash": "g" * 64,
        "nodes": [
            {"node_id": "INIT", "status": "COMPLETED"},
            {"node_id": "first", "status": "COMPLETED"},
            {"node_id": "ready", "status": "READY"},
            {"node_id": "unknown", "status": "PRECONDITION_UNKNOWN"},
            {"node_id": "blocked", "status": "BLOCKED"},
            {"node_id": "GOAL", "status": "BLOCKED"},
        ],
    }


def test_shadow_trace_marks_goal_completed_when_signed_goal_is_satisfied() -> None:
    project = getattr(shadow_runtime, "project_graph_state", None)
    assert project is not None
    graph, problem = _graph()
    state = project(
        graph,
        problem,
        _snapshot(
            true=frozenset({AT_TARGET}),
            false=frozenset({AT_SOURCE, HELD, BLOCKED, UNKNOWN}),
        ),
        policy_step=15,
        observation_generation=3,
        certificate_state="STALE",
    )
    assert state["nodes"][-1] == {"node_id": "GOAL", "status": "COMPLETED"}


def test_temporal_shadow_graph_tracks_macro_transport_and_raw_goal_truth() -> None:
    tracker_type = getattr(shadow_runtime, "ShadowGraphTracker", None)
    assert tracker_type is not None
    graph, problem = _temporal_graph()
    tracker = tracker_type(graph, problem)
    handempty = Fact("handempty")
    snapshots = (
        _audited_temporal_snapshot(0, true=frozenset({AT_SOURCE, handempty})),
        _audited_temporal_snapshot(1, true=frozenset({HELD})),
        _audited_temporal_snapshot(
            2,
            true=frozenset({HELD}),
            raw_target_while_held=True,
        ),
        _audited_temporal_snapshot(3, true=frozenset({AT_TARGET, handempty})),
    )

    states = [
        tracker.project(
            snapshot,
            policy_step=index,
            observation_generation=index,
            certificate_state="CURRENT",
            phase="POLICY",
        )
        for index, snapshot in enumerate(snapshots)
    ]

    assert [state["nodes"][1]["status"] for state in states] == [
        "READY",
        "ACTIVE",
        "EFFECT_OBSERVED",
        "COMPLETED",
    ]
    assert [state["nodes"][-1]["status"] for state in states] == [
        "BLOCKED",
        "BLOCKED",
        "COMPLETED",
        "COMPLETED",
    ]


def test_temporal_shadow_graph_treats_unlocated_ready_object_as_active() -> None:
    tracker_type = getattr(shadow_runtime, "ShadowGraphTracker", None)
    assert tracker_type is not None
    graph, problem = _temporal_graph()
    tracker = tracker_type(graph, problem)
    handempty = Fact("handempty")
    tracker.project(
        _audited_temporal_snapshot(0, true=frozenset({AT_SOURCE, handempty})),
        policy_step=0,
        observation_generation=0,
        certificate_state="CURRENT",
        phase="POLICY",
    )

    state = tracker.project(
        _audited_temporal_snapshot(1, true=frozenset({handempty})),
        policy_step=1,
        observation_generation=1,
        certificate_state="CURRENT",
        phase="POLICY",
    )

    assert state["nodes"][1] == {"node_id": "place", "status": "ACTIVE"}


def test_temporal_shadow_graph_does_not_treat_unknown_holding_as_transport() -> None:
    tracker_type = getattr(shadow_runtime, "ShadowGraphTracker", None)
    assert tracker_type is not None
    graph, problem = _temporal_graph()
    tracker = tracker_type(graph, problem)
    handempty = Fact("handempty")
    tracker.project(
        _audited_temporal_snapshot(0, true=frozenset({AT_SOURCE, handempty})),
        policy_step=0,
        observation_generation=0,
        certificate_state="CURRENT",
        phase="POLICY",
    )

    state = tracker.project(
        _audited_temporal_snapshot(
            1,
            true=frozenset({handempty}),
            unknown=frozenset({HELD}),
        ),
        policy_step=1,
        observation_generation=1,
        certificate_state="CURRENT",
        phase="POLICY",
    )

    assert state["nodes"][1] == {
        "node_id": "place",
        "status": "PRECONDITION_UNKNOWN",
    }


def test_temporal_shadow_graph_rejects_incidental_raw_effect_without_progress() -> None:
    tracker_type = getattr(shadow_runtime, "ShadowGraphTracker", None)
    assert tracker_type is not None
    graph, problem = _temporal_graph()
    tracker = tracker_type(graph, problem)

    state = tracker.project(
        _audited_temporal_snapshot(
            0,
            true=frozenset({HELD}),
            raw_target_while_held=True,
        ),
        policy_step=0,
        observation_generation=0,
        certificate_state="CURRENT",
        phase="POLICY",
    )

    assert state["nodes"][1] == {"node_id": "place", "status": "BLOCKED"}


def test_parser_exposes_topology_only_and_no_video_batch_flags() -> None:
    args = _parser().parse_args(
        [
            "--run-id", "trace", "--method-arm", "SHADOW_LOGIV",
            "--goal-mode", "METADATA_ASSISTED", "--deviation-mode", "NOMINAL",
            "--port", "8010", "--output-dir", "/tmp/trace",
            "--shadow-topology-only", "--no-video",
        ]
    )
    assert args.shadow_topology_only is True
    assert args.no_video is True


def test_topology_only_bypasses_recovery_contracts_for_all_tasks() -> None:
    args = _parser().parse_args(
        [
            "--run-id", "all-trace", "--method-arm", "SHADOW_LOGIV",
            "--goal-mode", "METADATA_ASSISTED", "--deviation-mode", "NOMINAL",
            "--port", "8010", "--output-dir", "/tmp/all-trace",
            "--shadow-topology-only", "--task-ids", "all",
        ]
    )

    assert _validate_shadow_options(args, tuple(range(10))) == {}
    config = _run_config(args, tuple(range(10)), (0,))
    assert config["shadow_monitor_contract"] is None
    assert config["shadow_monitor_contract_sha256_by_task"] == {}


def test_topology_only_rejects_recovery_root_collection() -> None:
    args = _parser().parse_args(
        [
            "--run-id", "bad-trace", "--method-arm", "SHADOW_LOGIV",
            "--goal-mode", "METADATA_ASSISTED", "--deviation-mode", "NOMINAL",
            "--port", "8010", "--output-dir", "/tmp/bad-trace",
            "--shadow-topology-only", "--collect-recovery-roots",
        ]
    )

    with pytest.raises(ValueError, match="cannot collect recovery roots"):
        _validate_shadow_options(args, tuple(range(10)))


def test_topology_only_rejects_nonpositive_audit_interval() -> None:
    args = _parser().parse_args(
        [
            "--run-id", "bad-interval", "--method-arm", "SHADOW_LOGIV",
            "--goal-mode", "METADATA_ASSISTED", "--deviation-mode", "NOMINAL",
            "--port", "8010", "--output-dir", "/tmp/bad-interval",
            "--shadow-topology-only", "--shadow-monitor-interval-steps", "0",
        ]
    )

    with pytest.raises(ValueError, match="interval"):
        _validate_shadow_options(args, tuple(range(10)))


def test_graph_artifact_keeps_fixed_structure_and_orders_the_state_trace() -> None:
    graph, _ = _graph()
    trace = [{"policy_step": 0, "nodes": [{"node_id": "INIT", "status": "COMPLETED"}]}]

    artifact = _graph_json(graph, state_trace=trace)

    assert [node["node_id"] for node in artifact["nodes"]] == [
        "INIT", "first", "ready", "unknown", "blocked", "GOAL"
    ]
    assert artifact["state_trace"] == trace
