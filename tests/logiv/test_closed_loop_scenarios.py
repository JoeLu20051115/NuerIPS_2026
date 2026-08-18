from __future__ import annotations

from dataclasses import replace

import pytest

from pi05_libero_repro.logiv.controller import (
    ControllerStatus,
    ExecutorStatus,
    GroundingStatus,
    RuntimeBudgetLimits,
)
from pi05_libero_repro.logiv.model import ContextPhase
from pi05_libero_repro.logiv.val import (
    SignedTraceResult,
    SignedTraceStatus,
    ValidationResult,
    ValidationStatus,
)
from tests.logiv.test_controller import controller_for


def _assert_every_dispatch_has_current_authorization(result, dispatcher) -> None:
    events = list(result.events)
    running = [event for event in events if event.startswith("ATTEMPT_RUNNING:")]
    authorized = [event for event in events if event.startswith("EXECUTION_AUTHORIZED:")]
    fact_authorized = [event for event in events if event.startswith("FACTS_AUTHORIZED:")]
    assert len(running) == len(dispatcher.dispatches)
    assert len(authorized) == len(dispatcher.dispatches)
    assert len(fact_authorized) >= len(dispatcher.dispatches)
    for running_event in running:
        attempt_id = running_event.split(":", 1)[1]
        running_index = events.index(running_event)
        permit_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith(f"EXECUTION_AUTHORIZED:{attempt_id}:")
        )
        assert permit_index < running_index
        assert any(
            event.startswith("FACTS_AUTHORIZED:")
            for event in events[:permit_index]
        )
    installed_certificates = [
        event for event in events if event.startswith("CERTIFICATE_INSTALLED:")
    ]
    assert len(installed_certificates) == result.graph_installs - 1
    assert events[0].startswith("CERTIFICATE_ACTIVE:")


def test_nominal_task8_is_parallel_and_every_dispatch_is_fact_authorized() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    assert controller.graph.action_layer_width() == 2

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    _assert_every_dispatch_has_current_authorization(result, dispatcher)


@pytest.mark.parametrize(
    "scenario, expected_dispatches, expected_installs",
    [
        (
            "repeatable_effect_failure",
            ["pick", "place-held-in", "place-held-in", "close-access"],
            2,
        ),
        (
            "dropped_bowl",
            ["pick", "place-held-in", "pick", "place-held-in", "close-access"],
            2,
        ),
        (
            "held_closed_drawer",
            ["pick", "put-down", "open-access", "pick", "place-held-in", "close-access"],
            2,
        ),
        (
            "multiple_producers",
            ["pick", "place-held-in", "open-access", "pick", "place-held-in", "close-access"],
            2,
        ),
        (
            "final_goal_reopened",
            ["pick", "place-held-in", "close-access", "close-access"],
            2,
        ),
    ],
)
def test_recovery_matrix_installs_only_certified_graphs(
    scenario: str,
    expected_dispatches: list[str],
    expected_installs: int,
) -> None:
    controller, world, grounder, dispatcher, _ = controller_for(3)
    if scenario == "repeatable_effect_failure":
        dispatcher.fail_once_schema = "place-held-in"
    elif scenario == "dropped_bowl":
        dispatcher.drop_once_schema = "place-held-in"
    elif scenario == "held_closed_drawer":
        dispatcher.close_after_schema = "pick"
    elif scenario == "multiple_producers":
        dispatcher.drop_once_schema = "place-held-in"
        dispatcher.close_after_schema = "place-held-in"
    elif scenario == "final_goal_reopened":
        grounder.final_hook = lambda: world.set_access(
            "white_cabinet_1_bottom_access", opened=True
        )

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert [action.schema for action in dispatcher.dispatches] == expected_dispatches
    assert result.graph_installs == expected_installs
    _assert_every_dispatch_has_current_authorization(result, dispatcher)


def test_forbidden_retry_never_reappears_under_new_occurrence_or_graph() -> None:
    controller, _, _, dispatcher, _ = controller_for(3, max_retries_per_lineage=0)
    dispatcher.fail_once_schema = "place-held-in"

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.terminal_cause == "NO_CERTIFIED_REPAIR_WITHIN_BUDGET"
    assert [action.schema for action in dispatcher.dispatches].count("place-held-in") == 1
    _assert_every_dispatch_has_current_authorization(result, dispatcher)


def test_grounding_validation_fence_and_budget_fail_closed() -> None:
    grounding_controller, _, grounding, grounding_dispatcher, _ = controller_for(8)
    grounding.failed_phase = ContextPhase.POST_STOP_FACTS
    grounding_result = grounding_controller.run()
    assert grounding_result.terminal_cause == "POST_STOP_GROUNDING_FAILURE"
    assert len(grounding_dispatcher.dispatches) == 1

    fence_controller, _, _, fence_dispatcher, _ = controller_for(8)
    fence_dispatcher.outcome_status = ExecutorStatus.FENCE_FAILURE
    fence_result = fence_controller.run()
    assert fence_result.status is ControllerStatus.SAFE_STOPPED
    assert fence_result.terminal_cause == "FENCE_FAILURE"

    repair_budget_controller, _, _, repair_budget_dispatcher, _ = controller_for(
        3,
        limits=RuntimeBudgetLimits(20, 0, 20),
    )
    repair_budget_dispatcher.close_after_schema = "pick"
    repair_budget_result = repair_budget_controller.run()
    assert repair_budget_result.terminal_cause == "BUDGET_EXHAUSTED"

    val_budget_controller, _, _, val_budget_dispatcher, _ = controller_for(
        3,
        limits=RuntimeBudgetLimits(20, 8, 1),
    )
    val_budget_dispatcher.fail_once_schema = "place-held-in"
    val_budget_result = val_budget_controller.run()
    assert val_budget_result.terminal_cause == "BUDGET_EXHAUSTED"


class _RecoveryValidationError:
    def validate(self, problem, plan, occurrence_sidecar, context, **kwargs):
        del plan, occurrence_sidecar, context, kwargs
        trace = SignedTraceResult(
            SignedTraceStatus.VALIDATION_ERROR,
            problem.initial_state,
            problem.initial_false,
            reason="injected wrapper error",
        )
        return ValidationResult(
            ValidationStatus.VALIDATION_ERROR,
            trace,
            None,
            reason="injected wrapper error",
        )


def test_validation_error_is_not_treated_as_invalid_and_repaired() -> None:
    controller, _, _, dispatcher, _ = controller_for(3)
    dispatcher.fail_once_schema = "place-held-in"
    controller.val_wrapper = _RecoveryValidationError()

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.terminal_cause == "VALIDATION_ERROR"
    assert [action.schema for action in dispatcher.dispatches] == ["pick", "place-held-in"]
