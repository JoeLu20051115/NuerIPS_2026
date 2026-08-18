from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from pi05_libero_repro.logiv.controller import (
    AttemptReceiptStatus,
    ControllerInstallation,
    ControllerStatus,
    DispatchStart,
    DispatchStatus,
    EvaluatorStatus,
    ExecutorOutcome,
    ExecutorStatus,
    GroundingResponse,
    GroundingStatus,
    LogivController,
    RuntimeBudgetLimits,
)
from pi05_libero_repro.logiv.dag import CausalDagCompiler
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GoalMode,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.repair import RepairBounds, RepairOperator, RetryPolicy
from pi05_libero_repro.logiv.val import ValidationStatus, ValWrapper


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")
TEST_PROPOSALS = Path(__file__).resolve().parents[1] / "fixtures/controller-proposals.json"
TASK3_SCHEMAS = frozenset(
    {"pick", "put-down", "open-access", "place-held-in", "close-access"}
)


class SymbolicWorld:
    def __init__(self, snapshot: FactSnapshot) -> None:
        self.true = set(snapshot.true_facts)
        self.false = set(snapshot.false_facts)
        self.epoch = snapshot.epoch_id

    def snapshot(self) -> FactSnapshot:
        return FactSnapshot(
            self.epoch,
            frozenset(self.true),
            frozenset(self.false),
            f"world:{self.epoch}",
        )

    def apply(self, action) -> None:
        self.true.difference_update(action.del_effects)
        self.true.update(action.add_effects)
        self.false.difference_update(action.add_effects)
        self.false.update(action.del_effects)
        self.epoch += 1

    def set_access(self, access: str, *, opened: bool) -> None:
        open_fact = Fact("open", (access,))
        closed_fact = Fact("closed", (access,))
        if opened:
            self.true.discard(closed_fact)
            self.false.discard(open_fact)
            self.true.add(open_fact)
            self.false.add(closed_fact)
        else:
            self.true.discard(open_fact)
            self.false.discard(closed_fact)
            self.true.add(closed_fact)
            self.false.add(open_fact)
        self.epoch += 1

    def drop_bowl_to_start(self) -> None:
        holding = Fact("holding", ("akita_black_bowl_1",))
        handempty = Fact("handempty")
        start = Fact(
            "at", ("akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region")
        )
        target = Fact(
            "at", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")
        )
        recovery = Fact(
            "at", ("akita_black_bowl_1", "kitchen_table_recovery_surface")
        )
        self.true.discard(holding)
        self.false.add(holding)
        self.true.add(handempty)
        self.false.discard(handempty)
        self.true.add(start)
        self.false.discard(start)
        self.true.discard(target)
        self.false.add(target)
        self.true.discard(recovery)
        self.false.add(recovery)
        self.epoch += 1


class ScriptedGrounder:
    def __init__(self, world: SymbolicWorld) -> None:
        self.world = world
        self.stale_once_phase = None
        self.failed_phase = None
        self.final_hook = None
        self.calls = []

    def acquire_epoch(self, phase: ContextPhase) -> int:
        if phase is ContextPhase.FINAL_GOAL and self.final_hook is not None:
            hook, self.final_hook = self.final_hook, None
            hook()
        return self.world.epoch

    def ground(self, phase, context, required_facts) -> GroundingResponse:
        self.calls.append((phase, context, frozenset(required_facts)))
        if self.stale_once_phase is phase:
            self.stale_once_phase = None
            return GroundingResponse(
                status=GroundingStatus.OK,
                context=replace(context, request_id="stale-callback"),
                snapshot=self.world.snapshot(),
            )
        if self.failed_phase is phase:
            status = (
                GroundingStatus.POST_STOP_GROUNDING_FAILURE
                if phase is ContextPhase.POST_STOP_FACTS
                else GroundingStatus.STATE_GROUNDING_FAILURE
            )
            return GroundingResponse(status=status, context=context, reason="scripted failure")
        return GroundingResponse(
            status=GroundingStatus.OK,
            context=context,
            snapshot=self.world.snapshot(),
        )


class SymbolicDispatcher:
    def __init__(self, world: SymbolicWorld) -> None:
        self.world = world
        self.dispatches = []
        self.completion_hints = []
        self.attempts = {}
        self.attempt_hints = {}
        self.next_id = 0
        self.fail_once_schema = None
        self.drop_once_schema = None
        self.close_after_schema = None
        self.outcome_status = None
        self.dispatch_status = DispatchStatus.ENQUEUED
        self.halt_ack = True
        self.halt_calls = 0
        self.apply_frontier_hint = False

    def consume_permit_and_enqueue(
        self, action, context, snapshot, *, completion_hint=None
    ) -> DispatchStart:
        if self.dispatch_status is not DispatchStatus.ENQUEUED:
            return DispatchStart(status=self.dispatch_status, context=context)
        attempt_id = f"attempt-{self.next_id}"
        self.next_id += 1
        self.dispatches.append(action)
        self.completion_hints.append(completion_hint)
        self.attempts[attempt_id] = action
        self.attempt_hints[attempt_id] = completion_hint
        attempt_context = replace(
            context,
            attempt_id=attempt_id,
            safety_epoch=self.next_id,
        )
        return DispatchStart(
            status=DispatchStatus.ENQUEUED,
            attempt_id=attempt_id,
            safety_epoch=self.next_id,
            context=attempt_context,
        )

    def await_outcome(self, start: DispatchStart) -> ExecutorOutcome:
        action = self.attempts[start.attempt_id]
        if self.outcome_status is not None:
            return ExecutorOutcome(
                status=self.outcome_status,
                attempt_id=start.attempt_id,
                stopped=self.outcome_status is ExecutorStatus.EPISODE_SUCCESS,
                stop_ack_attempt_id=(
                    start.attempt_id
                    if self.outcome_status is ExecutorStatus.EPISODE_SUCCESS
                    else None
                ),
                settled_epoch=(
                    self.world.epoch
                    if self.outcome_status is ExecutorStatus.EPISODE_SUCCESS
                    else None
                ),
                context=start.context,
            )
        if self.drop_once_schema == action.schema:
            self.drop_once_schema = None
            self.world.drop_bowl_to_start()
        elif self.fail_once_schema == action.schema:
            self.fail_once_schema = None
            self.world.epoch += 1
        else:
            hint = self.attempt_hints[start.attempt_id]
            actions = hint.actions if self.apply_frontier_hint else (action,)
            for hinted_action in actions:
                self.world.apply(hinted_action)
        if self.close_after_schema == action.schema:
            self.close_after_schema = None
            self.world.set_access("white_cabinet_1_bottom_access", opened=False)
        return ExecutorOutcome(
            status=ExecutorStatus.SUCCEEDED,
            attempt_id=start.attempt_id,
            stopped=True,
            stop_ack_attempt_id=start.attempt_id,
            settled_epoch=self.world.epoch,
            context=start.context,
        )

    def request_emergency_halt(self, attempt_id) -> bool:
        self.halt_calls += 1
        return self.halt_ack


class FixedEvaluator:
    def __init__(self, status: EvaluatorStatus) -> None:
        self.status = status
        self.calls = 0

    def evaluate(self, handle) -> EvaluatorStatus:
        self.calls += 1
        return self.status


def installed(task_id: int):
    package = ScriptedProposalProvider(TEST_PROPOSALS).propose(task_id, epoch_id=70)
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
        request_id=f"controller-initial-{task_id}",
        request_generation=0,
        episode_id=f"controller-episode-{task_id}",
        goal_id=package.frozen_goal.goal_id,
        goal_epoch=0,
        epoch_id=70,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )
    val_wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)
    result = val_wrapper.validate(package.problem, plan, sidecar, context)
    assert result.status is ValidationStatus.VALID
    compiler = CausalDagCompiler(REAL_VAL, timeout_seconds=5.0)
    graph = compiler.compile(package.problem, plan, sidecar, result.certificate, context)
    installation = ControllerInstallation(
        problem=package.problem,
        plan=plan,
        occurrence_sidecar=sidecar,
        certificate=result.certificate,
        graph=graph,
        goal_context=context,
        initial_snapshot=package.proposal.initial_snapshot,
    )
    return package, installation, val_wrapper, compiler


def controller_for(
    task_id: int,
    *,
    evaluator_status: EvaluatorStatus = EvaluatorStatus.EPISODE_SUCCESS,
    limits: RuntimeBudgetLimits | None = None,
    max_retries_per_lineage: int = 1,
):
    package, installation, val_wrapper, compiler = installed(task_id)
    world = SymbolicWorld(installation.initial_snapshot)
    grounder = ScriptedGrounder(world)
    dispatcher = SymbolicDispatcher(world)
    evaluator = FixedEvaluator(evaluator_status)
    allowed = TASK3_SCHEMAS if task_id == 3 else frozenset({"place-on", "pick", "place-held-on"})
    repair = RepairOperator(
        val_wrapper,
        allowed_schemas=allowed,
        bounds=RepairBounds(max_edits=4, max_candidates=4096, max_val_calls=8),
    )
    controller = LogivController(
        installation,
        grounder=grounder,
        dispatcher=dispatcher,
        evaluator=evaluator,
        evaluator_handle=f"evaluator-{task_id}",
        val_wrapper=val_wrapper,
        compiler=compiler,
        repair_operator=repair,
        retry_policy=RetryPolicy(max_retries_per_lineage=max_retries_per_lineage),
        budget_limits=limits
        or RuntimeBudgetLimits(
            max_physical_attempts=20,
            max_repair_rounds=8,
            max_total_val_calls=20,
        ),
        initial_val_calls=1,
    )
    return controller, world, grounder, dispatcher, evaluator


def test_normal_task8_commits_both_branches_and_external_evaluator_succeeds() -> None:
    controller, _, _, dispatcher, evaluator = controller_for(8)

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert [action.arguments[0] for action in dispatcher.dispatches] == [
        "moka_pot_2",
        "moka_pot_1",
    ]
    assert [receipt.status for receipt in result.receipts] == [
        AttemptReceiptStatus.COMMITTED,
        AttemptReceiptStatus.COMMITTED,
    ]
    first_hint, second_hint = dispatcher.completion_hints
    assert first_hint.occurrence_ids == controller.graph.canonical_agenda
    assert [action.arguments[0] for action in first_hint.actions] == [
        "moka_pot_2",
        "moka_pot_1",
    ]
    assert second_hint.occurrence_ids == (controller.graph.canonical_agenda[1],)
    assert evaluator.calls == 1


def test_serial_graph_dispatch_hint_never_crosses_action_precedence() -> None:
    controller, _, _, dispatcher, _ = controller_for(3)

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert all(len(hint.actions) == 1 for hint in dispatcher.completion_hints)
    assert [hint.actions[0] for hint in dispatcher.completion_hints] == dispatcher.dispatches


def test_incidentally_completed_frontier_sibling_requires_fresh_empty_repair() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    dispatcher.apply_frontier_hint = True

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert len(dispatcher.dispatches) == 1
    assert len(result.receipts) == 1
    assert result.receipts[0].status is AttemptReceiptStatus.COMMITTED
    assert result.events.count("PRECONDITION_GATE_REJECTED") == 1
    assert result.graph_installs == 2


def test_effect_failure_recertifies_suffix_and_retries_same_uncommitted_occurrence() -> None:
    controller, _, _, dispatcher, _ = controller_for(3)
    dispatcher.fail_once_schema = "place-held-in"

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert [action.schema for action in dispatcher.dispatches] == [
        "pick",
        "place-held-in",
        "place-held-in",
        "close-access",
    ]
    failed, retried = result.receipts[1:3]
    assert failed.status is AttemptReceiptStatus.FAILED
    assert retried.status is AttemptReceiptStatus.COMMITTED
    assert failed.occurrence_id == retried.occurrence_id
    assert result.budget_usage.total_val_calls == 2
    assert result.events.count("EFFECT_GATE_REJECTED") == 1


def test_valid_suffix_with_retry_forbidden_never_renames_and_redispatches_action() -> None:
    controller, _, _, dispatcher, _ = controller_for(3, max_retries_per_lineage=0)
    dispatcher.fail_once_schema = "place-held-in"

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.terminal_cause == "NO_CERTIFIED_REPAIR_WITHIN_BUDGET"
    assert [action.schema for action in dispatcher.dispatches] == ["pick", "place-held-in"]
    assert result.budget_usage.total_val_calls == 2


def test_effect_failure_drop_invalidates_suffix_and_installs_certified_repair() -> None:
    controller, _, _, dispatcher, _ = controller_for(3)
    dispatcher.drop_once_schema = "place-held-in"

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert [action.schema for action in dispatcher.dispatches] == [
        "pick",
        "place-held-in",
        "pick",
        "place-held-in",
        "close-access",
    ]
    assert result.graph_installs == 2
    assert result.budget_usage.total_val_calls == 3


def test_precondition_failure_repairs_directly_without_suffix_recertification() -> None:
    controller, _, _, dispatcher, _ = controller_for(3)
    dispatcher.close_after_schema = "pick"

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert [action.schema for action in dispatcher.dispatches] == [
        "pick",
        "put-down",
        "open-access",
        "pick",
        "place-held-in",
        "close-access",
    ]
    assert result.budget_usage.total_val_calls == 2
    assert result.events.count("PRECONDITION_GATE_REJECTED") == 1


def test_empty_agenda_goal_failure_repairs_close_without_empty_val_call() -> None:
    controller, world, grounder, _, _ = controller_for(3)
    grounder.final_hook = lambda: world.set_access(
        "white_cabinet_1_bottom_access", opened=True
    )

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert result.budget_usage.total_val_calls == 2
    assert result.receipts[-1].schema == "close-access"
    assert result.events.count("FINAL_GOAL_GATE_REJECTED") == 1


def test_stale_post_stop_callback_is_noop_then_correct_response_commits() -> None:
    controller, _, grounder, _, _ = controller_for(8)
    grounder.stale_once_phase = ContextPhase.POST_STOP_FACTS

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert "STALE_CALLBACK_NOOP" in result.events
    assert len(result.receipts) == 2


def test_post_stop_grounding_failure_keeps_cursor_and_prevents_retry() -> None:
    controller, _, grounder, dispatcher, _ = controller_for(8)
    grounder.failed_phase = ContextPhase.POST_STOP_FACTS

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.receipts[0].status is AttemptReceiptStatus.POST_STOP_UNKNOWN
    assert len(dispatcher.dispatches) == 1
    assert dispatcher.halt_calls == 0
    assert result.active_attempt_id == "attempt-0"
    assert controller.run() == result
    assert len(dispatcher.dispatches) == 1


def test_unknown_executor_outcome_halts_and_never_replaces_running_attempt() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    dispatcher.outcome_status = ExecutorStatus.OUTCOME_UNKNOWN
    dispatcher.halt_ack = False

    result = controller.run()

    assert result.status is ControllerStatus.UNSAFE_TERMINAL
    assert len(dispatcher.dispatches) == 1
    assert dispatcher.halt_calls == 1
    assert result.receipts[0].status is AttemptReceiptStatus.POST_STOP_UNKNOWN
    assert result.active_attempt_id == "attempt-0"


def test_external_evaluator_failure_is_not_upgraded_by_internal_goal() -> None:
    controller, _, _, _, evaluator = controller_for(
        8, evaluator_status=EvaluatorStatus.EPISODE_FAIL
    )

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_FAIL
    assert evaluator.calls == 1


def test_native_done_absorbing_skips_post_stop_gates_and_external_evaluator() -> None:
    controller, _, grounder, dispatcher, evaluator = controller_for(
        8, evaluator_status=EvaluatorStatus.EPISODE_FAIL
    )
    dispatcher.outcome_status = ExecutorStatus.EPISODE_SUCCESS

    result = controller.run()

    assert result.status is ControllerStatus.EPISODE_SUCCESS
    assert result.terminal_cause == "EPISODE_SUCCESS"
    assert len(dispatcher.dispatches) == 1
    assert evaluator.calls == 0
    assert all(call[0] is not ContextPhase.POST_STOP_FACTS for call in grounder.calls)


def test_physical_budget_blocks_next_dispatch() -> None:
    controller, _, _, dispatcher, _ = controller_for(
        8,
        limits=RuntimeBudgetLimits(
            max_physical_attempts=1,
            max_repair_rounds=4,
            max_total_val_calls=4,
        ),
    )

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.terminal_cause == "BUDGET_EXHAUSTED"
    assert len(dispatcher.dispatches) == 1


def test_safety_veto_creates_no_attempt_and_waits_for_halt_ack() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    dispatcher.dispatch_status = DispatchStatus.SAFETY_VETO

    result = controller.run()

    assert result.status is ControllerStatus.SAFE_STOPPED
    assert dispatcher.dispatches == []
    assert result.receipts == ()
    assert result.active_attempt_id is None
    assert dispatcher.halt_calls == 1
    assert result.budget_usage.physical_attempts == 0


def test_executor_rejected_not_enqueued_is_terminal_without_halt() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    dispatcher.dispatch_status = DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert dispatcher.dispatches == []
    assert dispatcher.halt_calls == 0
    assert result.budget_usage.physical_attempts == 0


def test_executor_action_budget_rejection_creates_no_physical_attempt() -> None:
    controller, _, _, dispatcher, _ = controller_for(8)
    dispatcher.dispatch_status = DispatchStatus.ACTION_BUDGET_EXHAUSTED

    result = controller.run()

    assert result.status is ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH
    assert result.terminal_cause == "BUDGET_EXHAUSTED"
    assert dispatcher.dispatches == []
    assert result.budget_usage.physical_attempts == 0
