from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from pi05_libero_repro.logiv.dag import CausalDagCompiler, SignedLiteral
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GoalMode,
    TaskProblem,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.repair import (
    FailureObligation,
    RepairBounds,
    RepairOperator,
    RepairStatus,
    RetryLedger,
    RetryPolicy,
    TraceKind,
    build_causal_slice,
    trace_invalid_plan,
)
from pi05_libero_repro.logiv.val import ValidationStatus, ValWrapper


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")
TASK3_SCHEMAS = frozenset(
    {"pick", "put-down", "open-access", "place-held-in", "close-access"}
)


def nominal(task_id: int):
    package = ScriptedProposalProvider().propose(task_id, epoch_id=50)
    plan = tuple(item.action for item in package.proposal.candidate_subtasks)
    sidecar = json.dumps(
        [
            {
                "occurrence_id": item.occurrence_id,
                "schema": item.action.schema,
                "arguments": list(item.action.arguments),
                "lineage_root": item.lineage_root,
            }
            for item in package.proposal.candidate_subtasks
        ],
        sort_keys=True,
    ).encode("utf-8")
    context = ContextEnvelope(
        phase=ContextPhase.PREINSTALL_VAL,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id=f"initial-{task_id}",
        request_generation=0,
        episode_id=f"episode-{task_id}",
        goal_id=package.frozen_goal.goal_id,
        goal_epoch=0,
        epoch_id=50,
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
    graph = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0).compile(
        package.problem, plan, sidecar, result.certificate, context
    )
    return package, plan, sidecar, context, result.certificate, graph


def current_problem(
    base: TaskProblem,
    true_facts: set[Fact],
    false_facts: set[Fact],
) -> TaskProblem:
    return replace(
        base,
        initial_state=frozenset(true_facts),
        initial_false=frozenset(false_facts),
    )


def recovery_context(graph, certificate, epoch: int) -> ContextEnvelope:
    return ContextEnvelope(
        phase=ContextPhase.RECOVERY_VAL,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id=f"repair-{epoch}",
        request_generation=0,
        episode_id="episode-3",
        goal_id="libero10-task-3-official-goal-v1",
        goal_epoch=0,
        epoch_id=epoch,
        graph_version=graph.graph_version,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=certificate.certificate_hash,
        safety_epoch=None,
    )


def repair_task3(problem: TaskProblem, remaining_plan, context, **kwargs):
    operator = RepairOperator(
        ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=TASK3_SCHEMAS,
        bounds=RepairBounds(max_edits=4, max_candidates=4096, max_val_calls=8),
    )
    return operator.repair(problem, remaining_plan, context=context, **kwargs)


def test_trace_is_exactly_earliest_action_failure_or_final_goal_failure() -> None:
    package3, plan3, *_ = nominal(3)
    action_trace = trace_invalid_plan(package3.problem, plan3[1:])
    package8, plan8, *_ = nominal(8)
    goal_trace = trace_invalid_plan(package8.problem, plan8[:1])

    assert action_trace.kind is TraceKind.ACTION_FAILURE
    assert len(action_trace.obligations) == 1
    assert action_trace.obligations[0].literal == SignedLiteral(
        Fact("holding", ("akita_black_bowl_1",)), positive=True
    )
    assert goal_trace.kind is TraceKind.GOAL_FAILURE
    assert all(item.consumer_id == "GOAL" for item in goal_trace.obligations)


def test_causal_slice_for_one_task8_goal_does_not_pull_in_sibling_branch() -> None:
    package, _, _, _, _, graph = nominal(8)
    first, second = graph.canonical_agenda
    obligation = FailureObligation(
        consumer_id="GOAL",
        literal=SignedLiteral(
            Fact("at", ("moka_pot_1", "flat_stove_1_cook_region")), positive=True
        ),
    )

    causal_slice = build_causal_slice(graph, (obligation,))

    assert first in causal_slice.node_ids
    assert second not in causal_slice.node_ids
    assert causal_slice.canonical_seed == (first,)
    assert package.problem.goal


def test_repair_dropped_bowl_rebuilds_pick_place_close() -> None:
    package, plan, _, _, certificate, graph = nominal(3)
    true_facts = {
        Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
        Fact("handempty"),
        Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
        Fact("open", ("white_cabinet_1_bottom_access",)),
    }
    false_facts = {
        Fact("holding", ("akita_black_bowl_1",)),
        Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
        Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
        Fact("closed", ("white_cabinet_1_bottom_access",)),
    }
    problem = current_problem(package.problem, true_facts, false_facts)

    result = repair_task3(
        problem,
        plan[1:],
        recovery_context(graph, certificate, epoch=51),
    )

    assert result.status is RepairStatus.CERTIFIED
    assert [action.schema for action in result.plan] == ["pick", "place-held-in", "close-access"]
    assert result.val_calls == 1


def test_repair_held_bowl_uses_remaining_place_then_close() -> None:
    package, plan, _, _, certificate, graph = nominal(3)
    problem = current_problem(
        package.problem,
        {
            Fact("holding", ("akita_black_bowl_1",)),
            Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
            Fact("open", ("white_cabinet_1_bottom_access",)),
        },
        {
            Fact("handempty"),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
            Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            Fact("closed", ("white_cabinet_1_bottom_access",)),
        },
    )

    result = repair_task3(
        problem,
        plan[1:],
        recovery_context(graph, certificate, epoch=52),
    )

    assert result.status is RepairStatus.CERTIFIED
    assert [action.schema for action in result.plan] == ["place-held-in", "close-access"]


def test_repair_closed_drawer_while_holding_requires_putdown_open_repick() -> None:
    package, plan, _, _, certificate, graph = nominal(3)
    problem = current_problem(
        package.problem,
        {
            Fact("holding", ("akita_black_bowl_1",)),
            Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
            Fact("closed", ("white_cabinet_1_bottom_access",)),
        },
        {
            Fact("handempty"),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
            Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            Fact("open", ("white_cabinet_1_bottom_access",)),
        },
    )

    result = repair_task3(
        problem,
        plan[1:],
        recovery_context(graph, certificate, epoch=53),
    )

    assert result.status is RepairStatus.CERTIFIED
    assert [action.schema for action in result.plan] == [
        "put-down",
        "open-access",
        "pick",
        "place-held-in",
        "close-access",
    ]


def test_final_open_drawer_repairs_with_close_only() -> None:
    package, _, _, _, certificate, graph = nominal(3)
    problem = current_problem(
        package.problem,
        {
            Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            Fact("handempty"),
            Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
            Fact("open", ("white_cabinet_1_bottom_access",)),
        },
        {
            Fact("holding", ("akita_black_bowl_1",)),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
            Fact("closed", ("white_cabinet_1_bottom_access",)),
        },
    )

    result = repair_task3(
        problem,
        (),
        recovery_context(graph, certificate, epoch=54),
    )

    assert result.status is RepairStatus.CERTIFIED
    assert [action.schema for action in result.plan] == ["close-access"]


def test_forbidden_retry_key_survives_renamed_occurrence_and_graph() -> None:
    package, plan, _, _, certificate, graph = nominal(3)
    problem = current_problem(
        package.problem,
        {
            Fact("holding", ("akita_black_bowl_1",)),
            Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
            Fact("open", ("white_cabinet_1_bottom_access",)),
        },
        {
            Fact("handempty"),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
            Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            Fact("closed", ("white_cabinet_1_bottom_access",)),
        },
    )
    snapshot = FactSnapshot(55, problem.initial_state, problem.initial_false, "sha256:retry")
    failed_action = plan[1]
    ledger = RetryLedger()
    original = ledger.record_dispatch(failed_action, snapshot, lineage_root="original-lineage")
    ledger.record_effect_failure(failed_action)
    inherited = ledger.ensure_lineage(failed_action, snapshot, lineage_root="renamed-lineage")
    assert inherited.retry_key == original.retry_key
    assert inherited.lineage_root == "original-lineage"

    result = repair_task3(
        problem,
        plan[1:],
        recovery_context(graph, certificate, epoch=55),
        retry_ledger=ledger,
        retry_policy=RetryPolicy(max_retries_per_lineage=1),
        forbidden_retry_keys=frozenset({original.retry_key.digest}),
    )

    assert result.status is RepairStatus.NO_CERTIFIED_REPAIR_WITHIN_BUDGET
    assert result.plan == ()


def test_retry_requires_stopped_known_facts_and_respects_lineage_limit() -> None:
    package, plan, *_ = nominal(3)
    action = plan[1]
    snapshot = FactSnapshot(
        57,
        frozenset(
            {
                Fact("holding", ("akita_black_bowl_1",)),
                Fact(
                    "accessible",
                    ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access"),
                ),
                Fact("open", ("white_cabinet_1_bottom_access",)),
            }
        ),
        frozenset(
            {
                Fact("handempty"),
                Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            }
        ),
        "sha256:retry-policy",
    )
    ledger = RetryLedger()
    policy = RetryPolicy(max_retries_per_lineage=1)
    ledger.record_dispatch(action, snapshot, lineage_root="place-lineage")
    ledger.record_effect_failure(action)

    assert policy.retry_allowed(
        action,
        snapshot,
        ledger,
        stopped=True,
        receipt_has_unknown_partial_effect=False,
    )
    assert not policy.retry_allowed(
        action,
        snapshot,
        ledger,
        stopped=False,
        receipt_has_unknown_partial_effect=False,
    )
    unknown = replace(
        snapshot,
        false_facts=snapshot.false_facts
        - {Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region"))},
    )
    assert not policy.retry_allowed(
        action,
        unknown,
        ledger,
        stopped=True,
        receipt_has_unknown_partial_effect=False,
    )
    ledger.record_dispatch(action, snapshot, lineage_root="renamed")
    ledger.record_effect_failure(action)
    assert not policy.retry_allowed(
        action,
        snapshot,
        ledger,
        stopped=True,
        receipt_has_unknown_partial_effect=False,
    )


def test_small_search_bound_reports_bounded_failure_not_global_no_solution() -> None:
    package, plan, _, _, certificate, graph = nominal(3)
    problem = current_problem(
        package.problem,
        {
            Fact("holding", ("akita_black_bowl_1",)),
            Fact("accessible", ("white_cabinet_1_bottom_region", "white_cabinet_1_bottom_access")),
            Fact("closed", ("white_cabinet_1_bottom_access",)),
        },
        {
            Fact("handempty"),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")),
            Fact("at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")),
            Fact("at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            Fact("open", ("white_cabinet_1_bottom_access",)),
        },
    )
    operator = RepairOperator(
        ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=TASK3_SCHEMAS,
        bounds=RepairBounds(max_edits=4, max_candidates=1, max_val_calls=1),
    )

    result = operator.repair(
        problem,
        plan[1:],
        context=recovery_context(graph, certificate, epoch=56),
    )

    assert result.status is RepairStatus.NO_CERTIFIED_REPAIR_WITHIN_BUDGET
    assert "within configured bounds" in result.reason
