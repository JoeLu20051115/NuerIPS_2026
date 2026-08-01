from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.dag import (
    CausalDagCompiler,
    CompilerError,
    GraphEdge,
    NodeKind,
    SchemaOnlyCausalDagCompiler,
    SignedLiteral,
    validate_graph,
)
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    GoalMode,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.val import ValidationStatus, ValWrapper


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")


def certified(task_id: int):
    package = ScriptedProposalProvider().propose(task_id, epoch_id=41)
    plan = tuple(item.action for item in package.proposal.candidate_subtasks)
    sidecar = json.dumps(
        [
            {
                "occurrence_id": item.occurrence_id,
                "schema": item.action.schema,
                "arguments": list(item.action.arguments),
                "lineage_root": item.lineage_root,
                "instruction": item.instruction,
            }
            for item in package.proposal.candidate_subtasks
        ],
        sort_keys=True,
    ).encode("utf-8")
    context = ContextEnvelope(
        phase=ContextPhase.PREINSTALL_VAL,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id=f"dag-cert-{task_id}",
        request_generation=0,
        episode_id=f"dag-episode-{task_id}",
        goal_id=package.frozen_goal.goal_id,
        goal_epoch=package.frozen_goal.goal_epoch,
        epoch_id=41,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )
    result = ValWrapper(REAL_VAL, timeout_seconds=5.0).validate(
        package.problem, plan, sidecar, context
    )
    assert result.status is ValidationStatus.VALID
    return package, plan, sidecar, context, result.certificate


def test_task8_dag_has_two_unordered_action_nodes_and_width_two() -> None:
    package, plan, sidecar, context, certificate = certified(8)

    graph = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0).compile(
        package.problem, plan, sidecar, certificate, context
    )

    actions = [node for node in graph.nodes if node.kind is NodeKind.ACTION]
    assert graph.canonical_agenda == tuple(item.node_id for item in actions)
    assert graph.action_layer_width() == 2
    assert graph.edge(actions[0].node_id, actions[1].node_id) is None
    assert graph.edge(actions[1].node_id, actions[0].node_id) is None
    assert all(graph.edge(node.node_id, "GOAL") is not None for node in actions)

    schema_only = SchemaOnlyCausalDagCompiler().compile(
        package.problem, plan, sidecar, context
    )
    assert schema_only.action_layer_width() == 2
    assert schema_only.certificate_hash != certificate.certificate_hash


def test_task3_support_and_open_threat_create_real_place_before_close_edge() -> None:
    package, plan, sidecar, context, certificate = certified(3)

    graph = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0).compile(
        package.problem, plan, sidecar, certificate, context
    )

    pick, place, close = graph.canonical_agenda
    assert graph.edge(pick, place).support_literals == frozenset(
        {SignedLiteral(Fact("holding", ("akita_black_bowl_1",)), positive=True)}
    )
    protected = graph.edge(place, close)
    assert SignedLiteral(Fact("handempty"), positive=True) in protected.support_literals
    assert any(
        reason.literal == SignedLiteral(
            Fact("open", ("white_cabinet_1_bottom_access",)), positive=True
        )
        for reason in protected.conflict_reasons
    )
    assert graph.canonical_agenda == (pick, place, close)


def test_support_literals_between_same_nodes_are_merged_not_parallel_edges() -> None:
    package, plan, sidecar, context, certificate = certified(9)

    graph = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0).compile(
        package.problem, plan, sidecar, certificate, context
    )
    place = graph.canonical_agenda[0]
    edge = graph.edge("INIT", place)

    assert edge is not None
    assert len(edge.support_literals) >= 4
    assert len([item for item in graph.edges if item.source == "INIT" and item.target == place]) == 1


def test_negative_goal_literal_is_supported_by_delete_effect() -> None:
    package, plan, sidecar, context, _ = certified(8)
    start = Fact("at", ("moka_pot_1", "kitchen_table_moka_pot_right_init_region"))
    problem = replace(package.problem, negative_goal=frozenset({start}))
    context = replace(context, goal_id="negative-goal-test")
    result = ValWrapper(REAL_VAL, timeout_seconds=5.0).validate(
        problem, plan, sidecar, context
    )
    assert result.status is ValidationStatus.VALID

    graph = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0).compile(
        problem, plan, sidecar, result.certificate, context
    )
    first = graph.canonical_agenda[0]

    assert SignedLiteral(start, positive=False) in graph.edge(first, "GOAL").support_literals


def test_compiler_rejects_certificate_mismatch_and_invalid_graph_edges() -> None:
    package, plan, sidecar, context, certificate = certified(8)
    compiler = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0)

    with pytest.raises(CompilerError, match="certificate mismatch"):
        compiler.compile(package.problem, plan, sidecar + b" ", certificate, context)

    graph = compiler.compile(package.problem, plan, sidecar, certificate, context)
    first = graph.canonical_agenda[0]
    self_loop = GraphEdge(source=first, target=first)
    with pytest.raises(CompilerError, match="self-loop"):
        validate_graph(replace(graph, edges=graph.edges + (self_loop,)))
    reverse = GraphEdge(source="GOAL", target=first)
    with pytest.raises(CompilerError, match="reverse canonical rank"):
        validate_graph(replace(graph, edges=graph.edges + (reverse,)))
