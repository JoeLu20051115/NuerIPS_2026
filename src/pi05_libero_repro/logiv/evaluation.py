from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, FrozenSet, Sequence

from pi05_libero_repro.logiv.controller import (
    AttemptReceipt,
    AttemptReceiptStatus,
    ControllerResult,
    ControllerStatus,
    DispatchStatus,
    ControllerInstallation,
    EvaluatorStatus,
    ExecutorStatus,
    GroundingStatus,
    RuntimeBudgetUsage,
)
from pi05_libero_repro.logiv.dag import CausalDagCompiler, CausalGraph
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    FactSnapshot,
    GoalMode,
    GroundAction,
    ProposalPackage,
    TaskProblem,
)
from pi05_libero_repro.logiv.repair import (
    RepairBounds,
    RepairOperator,
    RepairStatus,
    RetryLedger,
    RetryPolicy,
)
from pi05_libero_repro.logiv.val import PlanCertificate, ValWrapper


class MethodArm(str, Enum):
    BASE = "BASE"
    SHADOW_LOGIV = "SHADOW_LOGIV"
    STAGE_ONLY = "STAGE_ONLY"
    GRAPH_WITHOUT_VAL = "GRAPH_WITHOUT_VAL"
    VAL_WITHOUT_LOCALIZED_REPAIR = "VAL_WITHOUT_LOCALIZED_REPAIR"
    FULL_LOGIV = "FULL_LOGIV"


class EvaluationContractError(ValueError):
    pass


class EpisodePreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationContract:
    method_arm: MethodArm
    goal_mode: GoalMode
    deviation_mode: str
    oracle_grounding: bool
    development_only: bool
    prompt_locked: bool
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]

    def validate(self) -> None:
        if self.method_arm is not MethodArm.BASE and not self.oracle_grounding:
            raise EvaluationContractError(
                "the development implementation requires an explicit oracle grounding acknowledgement"
            )
        if self.goal_mode is not GoalMode.METADATA_ASSISTED:
            raise EvaluationContractError(
                "GOAL_PREDICTION is not supported by the no-API provider"
            )
        if not self.deviation_mode or self.deviation_mode == "UNSET":
            raise EvaluationContractError("deviation_mode must be explicit")
        if not self.development_only and not self.prompt_locked:
            raise EvaluationContractError("holdout evaluation requires a locked prompt")
        if (
            not self.task_ids
            or len(self.task_ids) != len(set(self.task_ids))
            or any(task_id < 0 or task_id > 9 for task_id in self.task_ids)
        ):
            raise EvaluationContractError("task IDs must come from the frozen 10-task manifest")
        if (
            not self.episode_indices
            or len(self.episode_indices) != len(set(self.episode_indices))
            or any(index < 0 or index >= 50 for index in self.episode_indices)
        ):
            raise EvaluationContractError("episode indices must be unique values in [0, 49]")


@dataclass(frozen=True)
class CertifiedEpisode:
    problem: TaskProblem
    plan: tuple[GroundAction, ...]
    occurrence_sidecar: bytes
    certificate: PlanCertificate
    graph: CausalGraph
    installation: ControllerInstallation
    repair_operator: RepairOperator
    initial_val_calls: int


def certify_initial_package(
    package: ProposalPackage,
    grounded_snapshot: FactSnapshot,
    *,
    episode_id: str,
    val_wrapper: ValWrapper,
    allowed_schemas: FrozenSet[str],
    repair_bounds: RepairBounds,
    retry_policy: RetryPolicy | None = None,
    decompose_macro_sources: FrozenSet[str] = frozenset(),
) -> CertifiedEpisode:
    proposal = package.proposal
    if grounded_snapshot.epoch_id != proposal.epoch_id:
        raise EpisodePreparationError("proposal/GroundFacts epoch mismatch")
    if not proposal.initial_snapshot.true_facts <= grounded_snapshot.true_facts:
        raise EpisodePreparationError("proposal TRUE fact conflicts with GroundFacts")
    if not proposal.initial_snapshot.false_facts <= grounded_snapshot.false_facts:
        raise EpisodePreparationError("proposal FALSE fact conflicts with GroundFacts")
    problem = replace(
        package.problem,
        initial_state=grounded_snapshot.true_facts,
        initial_false=grounded_snapshot.false_facts,
    )
    rough_plan = tuple(item.action for item in proposal.candidate_subtasks)
    context = ContextEnvelope(
        phase=ContextPhase.PREINSTALL_VAL,
        goal_mode=proposal.goal_mode,
        request_id=f"{episode_id}-preinstall-val",
        request_generation=0,
        episode_id=episode_id,
        goal_id=package.frozen_goal.goal_id,
        goal_epoch=package.frozen_goal.goal_epoch,
        epoch_id=grounded_snapshot.epoch_id,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )
    policy = retry_policy or RetryPolicy(max_retries_per_lineage=1)
    repair = RepairOperator(
        val_wrapper,
        allowed_schemas=allowed_schemas,
        bounds=repair_bounds,
        retry_policy=policy,
        decompose_macro_sources=decompose_macro_sources,
    )
    lineage_roots = {
        item.action.retry_key: item.lineage_root for item in proposal.candidate_subtasks
    }
    result = repair.repair(
        problem,
        rough_plan,
        context=context,
        retry_ledger=RetryLedger(),
        retry_policy=policy,
        lineage_roots=lineage_roots,
    )
    if result.status is not RepairStatus.CERTIFIED or result.certificate is None:
        raise EpisodePreparationError(
            f"initial plan was not certified: {result.status.value}:{result.reason}"
        )
    compiler = CausalDagCompiler(
        val_wrapper.val_binary,
        timeout_seconds=val_wrapper.timeout_seconds,
    )
    graph = compiler.compile(
        problem,
        result.plan,
        result.occurrence_sidecar,
        result.certificate,
        result.certificate.context,
    )
    installation = ControllerInstallation(
        problem=problem,
        plan=result.plan,
        occurrence_sidecar=result.occurrence_sidecar,
        certificate=result.certificate,
        graph=graph,
        goal_context=result.certificate.context,
        initial_snapshot=grounded_snapshot,
    )
    return CertifiedEpisode(
        problem=problem,
        plan=result.plan,
        occurrence_sidecar=result.occurrence_sidecar,
        certificate=result.certificate,
        graph=graph,
        installation=installation,
        repair_operator=repair,
        initial_val_calls=result.val_calls,
    )


class GlobalRepairOperator:
    """VAL ablation: keep certification but remove topology-localized repair hints."""

    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped

    def repair(self, *args: Any, causal_slice: Any = None, **kwargs: Any) -> Any:
        del causal_slice
        return self.wrapped.repair(*args, causal_slice=None, **kwargs)


class NativeLiberoTaskEvaluator:
    """Independent terminal evaluator; it never receives the controller's symbolic goal."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_status: EvaluatorStatus | None = None

    def evaluate(self, evaluator_handle: Any) -> EvaluatorStatus:
        self.calls += 1
        try:
            successful = bool(evaluator_handle.check_success())
        except TimeoutError:
            status = EvaluatorStatus.EVALUATOR_TIMEOUT
        except Exception:
            status = EvaluatorStatus.EVALUATOR_ERROR
        else:
            status = (
                EvaluatorStatus.EPISODE_SUCCESS
                if successful
                else EvaluatorStatus.EPISODE_FAIL
            )
        self.last_status = status
        return status


def _controller_status(evaluator_status: EvaluatorStatus) -> ControllerStatus:
    return {
        EvaluatorStatus.EPISODE_SUCCESS: ControllerStatus.EPISODE_SUCCESS,
        EvaluatorStatus.EPISODE_FAIL: ControllerStatus.EPISODE_FAIL,
        EvaluatorStatus.EVALUATOR_ERROR: ControllerStatus.EVALUATOR_ERROR,
        EvaluatorStatus.EVALUATOR_TIMEOUT: ControllerStatus.EVALUATOR_TIMEOUT,
    }[evaluator_status]


def _simple_result(
    status: ControllerStatus,
    cause: str,
    receipts: Sequence[AttemptReceipt],
    events: Sequence[str],
    *,
    physical_attempts: int,
    graph_installs: int,
) -> ControllerResult:
    return ControllerResult(
        status=status,
        terminal_cause=cause,
        receipts=tuple(receipts),
        events=tuple(events),
        budget_usage=RuntimeBudgetUsage(
            physical_attempts=physical_attempts,
            repair_rounds=0,
            total_val_calls=0,
        ),
        graph_installs=graph_installs,
        active_attempt_id=None,
    )


def run_stage_only(
    plan: Sequence[GroundAction],
    *,
    initial_snapshot: FactSnapshot,
    dispatcher: Any,
    evaluator: NativeLiberoTaskEvaluator,
    evaluator_handle: Any,
    base_context: ContextEnvelope,
    max_physical_attempts: int,
) -> ControllerResult:
    """Stage decomposition ablation: no VAL, DAG, fact gates, or repair."""

    snapshot = initial_snapshot
    receipts: list[AttemptReceipt] = []
    events = ["STAGE_ONLY_NO_VAL_NO_DAG_NO_FACT_GATE"]
    for index, action in enumerate(plan):
        if index >= max_physical_attempts:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "BUDGET_EXHAUSTED",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=0,
            )
        context = replace(
            base_context,
            phase=ContextPhase.PRE_DISPATCH_FACTS,
            request_id=f"{base_context.episode_id}-stage-{index}",
            epoch_id=int(dispatcher.store.epoch_id),
            graph_version=None,
            occurrence_id=f"stage-o{index:03d}",
            attempt_id=None,
            certificate_hash=None,
            safety_epoch=None,
        )
        snapshot = replace(snapshot, epoch_id=context.epoch_id)
        start = dispatcher.consume_permit_and_enqueue(action, context, snapshot)
        if start.status is DispatchStatus.ACTION_BUDGET_EXHAUSTED:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "BUDGET_EXHAUSTED",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=0,
            )
        if start.status is not DispatchStatus.ENQUEUED or start.attempt_id is None:
            acknowledged = dispatcher.request_emergency_halt(None)
            return _simple_result(
                ControllerStatus.SAFE_STOPPED if acknowledged else ControllerStatus.UNSAFE_TERMINAL,
                start.status.value,
                receipts,
                events,
                physical_attempts=index,
                graph_installs=0,
            )
        outcome = dispatcher.await_outcome(start)
        stopped = (
            outcome.stopped
            and outcome.stop_ack_attempt_id == start.attempt_id
            and outcome.attempt_id == start.attempt_id
        )
        if outcome.status is not ExecutorStatus.SUCCEEDED or not stopped:
            acknowledged = dispatcher.request_emergency_halt(start.attempt_id)
            return _simple_result(
                ControllerStatus.SAFE_STOPPED if acknowledged else ControllerStatus.UNSAFE_TERMINAL,
                outcome.status.value,
                receipts,
                events,
                physical_attempts=index + 1,
                graph_installs=0,
            )
        receipts.append(
            AttemptReceipt(
                attempt_id=start.attempt_id,
                occurrence_id=f"stage-o{index:03d}",
                graph_version="NO_GRAPH",
                schema=action.schema,
                status=AttemptReceiptStatus.COMMITTED,
                pre_epoch=context.epoch_id,
                post_epoch=outcome.settled_epoch,
                stop_confirmed=True,
                effect_status="NOT_CHECKED",
            )
        )
    evaluated = evaluator.evaluate(evaluator_handle)
    return _simple_result(
        _controller_status(evaluated),
        evaluated.value,
        receipts,
        events,
        physical_attempts=len(receipts),
        graph_installs=0,
    )


def run_schema_only_graph(
    problem: TaskProblem,
    graph: CausalGraph,
    *,
    initial_snapshot: FactSnapshot,
    grounder: Any,
    dispatcher: Any,
    evaluator: NativeLiberoTaskEvaluator,
    evaluator_handle: Any,
    base_context: ContextEnvelope,
    max_physical_attempts: int,
) -> ControllerResult:
    """Graph-without-VAL arm with schema checks and fact gates, but no repair."""

    receipts: list[AttemptReceipt] = []
    events = ["GRAPH_WITHOUT_VAL_SCHEMA_ONLY"]
    snapshot = initial_snapshot

    def context(phase: ContextPhase, index: int, epoch: int, **values: Any) -> ContextEnvelope:
        return replace(
            base_context,
            phase=phase,
            request_id=f"{base_context.episode_id}-schema-only-{phase.value}-{index}",
            epoch_id=epoch,
            graph_version=graph.graph_version,
            certificate_hash=graph.certificate_hash,
            occurrence_id=values.get("occurrence_id"),
            attempt_id=values.get("attempt_id"),
            safety_epoch=values.get("safety_epoch"),
        )

    for index, occurrence_id in enumerate(graph.canonical_agenda):
        if index >= max_physical_attempts:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "BUDGET_EXHAUSTED",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=1,
            )
        action = graph.node_map[occurrence_id].action
        assert action is not None
        required = frozenset(
            action.preconditions
            | action.negative_preconditions
            | action.add_effects
            | action.del_effects
        )
        epoch = int(grounder.acquire_epoch(ContextPhase.PRE_DISPATCH_FACTS))
        pre_context = context(
            ContextPhase.PRE_DISPATCH_FACTS, index, epoch, occurrence_id=occurrence_id
        )
        grounded = grounder.ground(ContextPhase.PRE_DISPATCH_FACTS, pre_context, required)
        if grounded.status is not GroundingStatus.OK or grounded.snapshot is None:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "STATE_GROUNDING_FAILURE",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=1,
            )
        snapshot = grounded.snapshot
        if not snapshot.satisfies(
            positive=action.preconditions,
            negative=action.negative_preconditions,
        ):
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "PRECONDITION_FAILURE",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=1,
            )
        start = dispatcher.consume_permit_and_enqueue(action, pre_context, snapshot)
        if start.status is DispatchStatus.ACTION_BUDGET_EXHAUSTED:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "BUDGET_EXHAUSTED",
                receipts,
                events,
                physical_attempts=index,
                graph_installs=1,
            )
        if start.status is not DispatchStatus.ENQUEUED or start.attempt_id is None:
            acknowledged = dispatcher.request_emergency_halt(None)
            return _simple_result(
                ControllerStatus.SAFE_STOPPED if acknowledged else ControllerStatus.UNSAFE_TERMINAL,
                start.status.value,
                receipts,
                events,
                physical_attempts=index,
                graph_installs=1,
            )
        outcome = dispatcher.await_outcome(start)
        stopped = (
            outcome.status is ExecutorStatus.SUCCEEDED
            and outcome.stopped
            and outcome.stop_ack_attempt_id == start.attempt_id
        )
        if not stopped or outcome.settled_epoch is None:
            acknowledged = dispatcher.request_emergency_halt(start.attempt_id)
            return _simple_result(
                ControllerStatus.SAFE_STOPPED if acknowledged else ControllerStatus.UNSAFE_TERMINAL,
                outcome.status.value,
                receipts,
                events,
                physical_attempts=index + 1,
                graph_installs=1,
            )
        post_context = context(
            ContextPhase.POST_STOP_FACTS,
            index,
            outcome.settled_epoch,
            occurrence_id=occurrence_id,
            attempt_id=start.attempt_id,
            safety_epoch=start.safety_epoch,
        )
        post = grounder.ground(
            ContextPhase.POST_STOP_FACTS,
            post_context,
            frozenset(action.add_effects | action.del_effects),
        )
        if post.status is not GroundingStatus.OK or post.snapshot is None:
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "POST_STOP_GROUNDING_FAILURE",
                receipts,
                events,
                physical_attempts=index + 1,
                graph_installs=1,
            )
        if not post.snapshot.satisfies(
            positive=action.add_effects, negative=action.del_effects
        ):
            return _simple_result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "EFFECT_FAILURE",
                receipts,
                events,
                physical_attempts=index + 1,
                graph_installs=1,
            )
        snapshot = post.snapshot
        receipts.append(
            AttemptReceipt(
                attempt_id=start.attempt_id,
                occurrence_id=occurrence_id,
                graph_version=graph.graph_version,
                schema=action.schema,
                status=AttemptReceiptStatus.COMMITTED,
                pre_epoch=pre_context.epoch_id,
                post_epoch=post.snapshot.epoch_id,
                stop_confirmed=True,
                effect_status="VERIFIED",
            )
        )

    goal_context = context(
        ContextPhase.FINAL_GOAL,
        len(graph.canonical_agenda),
        int(grounder.acquire_epoch(ContextPhase.FINAL_GOAL)),
    )
    final = grounder.ground(
        ContextPhase.FINAL_GOAL,
        goal_context,
        frozenset(problem.goal | problem.negative_goal),
    )
    if (
        final.status is not GroundingStatus.OK
        or final.snapshot is None
        or not final.snapshot.satisfies(
            positive=problem.goal, negative=problem.negative_goal
        )
    ):
        return _simple_result(
            ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
            "FINAL_GOAL_FAILURE",
            receipts,
            events,
            physical_attempts=len(receipts),
            graph_installs=1,
        )
    evaluated = evaluator.evaluate(evaluator_handle)
    return _simple_result(
        _controller_status(evaluated),
        evaluated.value,
        receipts,
        events,
        physical_attempts=len(receipts),
        graph_installs=1,
    )
