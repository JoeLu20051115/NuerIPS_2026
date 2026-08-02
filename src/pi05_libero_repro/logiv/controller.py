from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
from typing import Any, Sequence, Tuple

from pi05_libero_repro.logiv.dag import (
    CausalDagCompiler,
    CausalGraph,
    CompilerError,
    SignedLiteral,
)
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
)
from pi05_libero_repro.logiv.repair import (
    CausalSlice,
    FailureObligation,
    RepairOperator,
    RepairError,
    RepairStatus,
    RetryLedger,
    RetryPolicy,
    build_causal_slice,
    build_excluded_retry_slice,
    trace_invalid_plan,
)
from pi05_libero_repro.logiv.val import PlanCertificate, ValidationStatus, ValWrapper


_STALE_INSTALL = object()


class GroundingStatus(str, Enum):
    OK = "OK"
    STATE_GROUNDING_FAILURE = "STATE_GROUNDING_FAILURE"
    POST_STOP_GROUNDING_FAILURE = "POST_STOP_GROUNDING_FAILURE"


class DispatchStatus(str, Enum):
    ENQUEUED = "ENQUEUED"
    SAFETY_VETO = "SAFETY_VETO"
    WATCHDOG_EXPIRED = "WATCHDOG_EXPIRED"
    EXECUTOR_REJECTED_NOT_ENQUEUED = "EXECUTOR_REJECTED_NOT_ENQUEUED"
    ACTION_BUDGET_EXHAUSTED = "ACTION_BUDGET_EXHAUSTED"


class ExecutorStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    EXECUTOR_FAILED = "EXECUTOR_FAILED"
    OUTCOME_UNKNOWN = "EXECUTOR_OUTCOME_UNKNOWN"
    TIMEOUT = "EXECUTOR_TIMEOUT"
    FENCE_FAILURE = "FENCE_FAILURE"
    SETTLING_TIMEOUT = "SETTLING_TIMEOUT"


class EvaluatorStatus(str, Enum):
    EPISODE_SUCCESS = "EPISODE_SUCCESS"
    EPISODE_FAIL = "EPISODE_FAIL"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    EVALUATOR_TIMEOUT = "EVALUATOR_TIMEOUT"


class AttemptReceiptStatus(str, Enum):
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    POST_STOP_UNKNOWN = "POST_STOP_UNKNOWN"
    REJECTED = "REJECTED"


class ControllerStatus(str, Enum):
    EPISODE_SUCCESS = "EPISODE_SUCCESS"
    EPISODE_FAIL = "EPISODE_FAIL"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    EVALUATOR_TIMEOUT = "EVALUATOR_TIMEOUT"
    TERMINAL_NO_FURTHER_DISPATCH = "TERMINAL_NO_FURTHER_DISPATCH"
    SAFE_STOPPED = "SAFE_STOPPED"
    UNSAFE_TERMINAL = "UNSAFE_TERMINAL"


class RecoveryReason(str, Enum):
    PRECONDITION_FAILURE = "PRECONDITION_FAILURE"
    EFFECT_FAILURE = "EFFECT_FAILURE"
    EXECUTOR_FAILED = "EXECUTOR_FAILED"
    FINAL_GOAL_FAILURE = "FINAL_GOAL_FAILURE"


@dataclass(frozen=True)
class GroundingResponse:
    status: GroundingStatus
    context: ContextEnvelope
    snapshot: FactSnapshot | None = None
    reason: str = ""


@dataclass(frozen=True)
class DispatchStart:
    status: DispatchStatus
    attempt_id: str | None = None
    safety_epoch: int | None = None
    context: ContextEnvelope | None = None


@dataclass(frozen=True)
class ExecutorOutcome:
    status: ExecutorStatus
    attempt_id: str
    stopped: bool
    stop_ack_attempt_id: str | None = None
    settled_epoch: int | None = None
    reason: str = ""
    context: ContextEnvelope | None = None


@dataclass(frozen=True)
class AttemptReceipt:
    attempt_id: str
    occurrence_id: str
    graph_version: str
    schema: str
    status: AttemptReceiptStatus
    pre_epoch: int
    post_epoch: int | None
    stop_confirmed: bool
    effect_status: str
    reason: str = ""


@dataclass(frozen=True)
class RuntimeBudgetLimits:
    max_physical_attempts: int
    max_repair_rounds: int
    max_total_val_calls: int

    def __post_init__(self) -> None:
        if min(
            self.max_physical_attempts,
            self.max_repair_rounds,
            self.max_total_val_calls,
        ) < 0:
            raise ValueError("runtime budgets must be nonnegative")


@dataclass(frozen=True)
class RuntimeBudgetUsage:
    physical_attempts: int
    repair_rounds: int
    total_val_calls: int


@dataclass(frozen=True)
class ControllerInstallation:
    problem: TaskProblem
    plan: Tuple[GroundAction, ...]
    occurrence_sidecar: bytes
    certificate: PlanCertificate
    graph: CausalGraph
    goal_context: ContextEnvelope
    initial_snapshot: FactSnapshot

    def __post_init__(self) -> None:
        if self.graph.certificate_hash != self.certificate.certificate_hash:
            raise ValueError("installation graph/certificate mismatch")
        if self.graph.source_epoch != self.initial_snapshot.epoch_id:
            raise ValueError("installation graph/snapshot epoch mismatch")
        if self.certificate.context != self.goal_context:
            raise ValueError("installation certificate/context mismatch")
        if self.problem.initial_state != self.initial_snapshot.true_facts or (
            self.problem.initial_false != self.initial_snapshot.false_facts
        ):
            raise ValueError("installation Problem does not match initial GroundFacts")
        if len(self.plan) != len(self.graph.canonical_agenda):
            raise ValueError("installation plan/agenda mismatch")


@dataclass(frozen=True)
class ControllerResult:
    status: ControllerStatus
    terminal_cause: str | None
    receipts: Tuple[AttemptReceipt, ...]
    events: Tuple[str, ...]
    budget_usage: RuntimeBudgetUsage
    graph_installs: int
    active_attempt_id: str | None


class _BudgetCounter:
    def __init__(self, limits: RuntimeBudgetLimits, initial_val_calls: int) -> None:
        if initial_val_calls < 0 or initial_val_calls > limits.max_total_val_calls:
            raise ValueError("initial_val_calls exceeds the total VAL budget")
        self.limits = limits
        self.physical_attempts = 0
        self.repair_rounds = 0
        self.total_val_calls = initial_val_calls

    def consume_physical(self) -> bool:
        if self.physical_attempts >= self.limits.max_physical_attempts:
            return False
        self.physical_attempts += 1
        return True

    def refund_unenqueued_physical(self) -> None:
        if self.physical_attempts <= 0:
            raise RuntimeError("cannot refund an unreserved physical attempt")
        self.physical_attempts -= 1

    def consume_repair(self) -> bool:
        if self.repair_rounds >= self.limits.max_repair_rounds:
            return False
        self.repair_rounds += 1
        return True

    def consume_val(self) -> bool:
        if self.total_val_calls >= self.limits.max_total_val_calls:
            return False
        self.total_val_calls += 1
        return True

    def snapshot(self) -> RuntimeBudgetUsage:
        return RuntimeBudgetUsage(
            physical_attempts=self.physical_attempts,
            repair_rounds=self.repair_rounds,
            total_val_calls=self.total_val_calls,
        )


def _action_relevant_facts(action: GroundAction) -> frozenset[Fact]:
    return frozenset(
        action.preconditions
        | action.negative_preconditions
        | action.add_effects
        | action.del_effects
    )


class LogivController:
    def __init__(
        self,
        installation: ControllerInstallation,
        *,
        grounder: Any,
        dispatcher: Any,
        evaluator: Any,
        evaluator_handle: Any,
        val_wrapper: ValWrapper,
        compiler: CausalDagCompiler,
        repair_operator: RepairOperator,
        retry_policy: RetryPolicy,
        budget_limits: RuntimeBudgetLimits,
        initial_val_calls: int,
    ) -> None:
        self.problem = installation.problem
        self.plan = installation.plan
        self.sidecar = installation.occurrence_sidecar
        self.certificate = installation.certificate
        self.graph = installation.graph
        self.goal_context = installation.goal_context
        self.snapshot = installation.initial_snapshot
        self.cursor = 0
        self.grounder = grounder
        self.dispatcher = dispatcher
        self.evaluator = evaluator
        self.evaluator_handle = evaluator_handle
        self.val_wrapper = val_wrapper
        self.compiler = compiler
        self.repair_operator = repair_operator
        self.retry_policy = retry_policy
        self.budgets = _BudgetCounter(budget_limits, initial_val_calls)
        self.retry_ledger = RetryLedger()
        self.forbidden_retry_keys: set[str] = set()
        self.receipts: list[AttemptReceipt] = []
        self.events: list[str] = []
        self.graph_installs = 1
        self._request_counter = 0
        self._active_attempt_id: str | None = None
        self._latched_result: ControllerResult | None = None
        verified_graph = self.compiler.compile(
            self.problem,
            self.plan,
            self.sidecar,
            self.certificate,
            self.certificate.context,
            forbidden_retry_keys=frozenset(),
            retry_ledger_version=0,
        )
        if verified_graph.graph_hash != self.graph.graph_hash:
            raise ValueError("initial installation graph does not match certified inputs")
        self.events.append(f"CERTIFICATE_ACTIVE:{self.certificate.certificate_hash}")
        self.events.append(f"GRAPH_ACTIVE:{self.graph.graph_version}")

    def _result(self, status: ControllerStatus, cause: str | None = None) -> ControllerResult:
        result = ControllerResult(
            status=status,
            terminal_cause=cause,
            receipts=tuple(self.receipts),
            events=tuple(self.events),
            budget_usage=self.budgets.snapshot(),
            graph_installs=self.graph_installs,
            active_attempt_id=self._active_attempt_id,
        )
        self._latched_result = result
        return result

    def _new_context(
        self,
        phase: ContextPhase,
        epoch_id: int,
        *,
        occurrence_id: str | None = None,
        attempt_id: str | None = None,
        safety_epoch: int | None = None,
    ) -> ContextEnvelope:
        self._request_counter += 1
        return ContextEnvelope(
            phase=phase,
            goal_mode=self.goal_context.goal_mode,
            request_id=f"{self.goal_context.episode_id}-request-{self._request_counter:06d}",
            request_generation=0,
            episode_id=self.goal_context.episode_id,
            goal_id=self.goal_context.goal_id,
            goal_epoch=self.goal_context.goal_epoch,
            epoch_id=epoch_id,
            graph_version=self.graph.graph_version,
            occurrence_id=occurrence_id,
            attempt_id=attempt_id,
            certificate_hash=self.certificate.certificate_hash,
            safety_epoch=safety_epoch,
        )

    def _ground(
        self,
        phase: ContextPhase,
        required_facts: frozenset[Fact],
        *,
        occurrence_id: str | None = None,
        attempt_id: str | None = None,
        safety_epoch: int | None = None,
        forced_epoch: int | None = None,
    ) -> GroundingResponse:
        epoch = (
            forced_epoch
            if forced_epoch is not None
            else int(self.grounder.acquire_epoch(phase))
        )
        context = self._new_context(
            phase,
            epoch,
            occurrence_id=occurrence_id,
            attempt_id=attempt_id,
            safety_epoch=safety_epoch,
        )
        for _ in range(16):
            response = self.grounder.ground(phase, context, required_facts)
            if response.context != context:
                self.events.append("STALE_CALLBACK_NOOP")
                continue
            if response.status is not GroundingStatus.OK:
                return response
            if response.snapshot is None or response.snapshot.epoch_id != context.epoch_id:
                return GroundingResponse(
                    status=(
                        GroundingStatus.POST_STOP_GROUNDING_FAILURE
                        if phase is ContextPhase.POST_STOP_FACTS
                        else GroundingStatus.STATE_GROUNDING_FAILURE
                    ),
                    context=context,
                    reason="snapshot/context epoch mismatch",
                )
            return response
        return GroundingResponse(
            status=(
                GroundingStatus.POST_STOP_GROUNDING_FAILURE
                if phase is ContextPhase.POST_STOP_FACTS
                else GroundingStatus.STATE_GROUNDING_FAILURE
            ),
            context=context,
            reason="too many stale callbacks",
        )

    def _current_problem(self, snapshot: FactSnapshot) -> TaskProblem:
        return replace(
            self.problem,
            initial_state=snapshot.true_facts,
            initial_false=snapshot.false_facts,
        )

    def _sidecar_for_cursor(self) -> bytes:
        payload = []
        node_map = self.graph.node_map
        for occurrence_id in self.graph.canonical_agenda[self.cursor :]:
            node = node_map[occurrence_id]
            assert node.action is not None
            payload.append(
                {
                    "occurrence_id": occurrence_id,
                    "schema": node.action.schema,
                    "arguments": list(node.action.arguments),
                    "lineage_root": node.lineage_root,
                    "instruction": node.instruction,
                }
            )
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _lineage_roots(self) -> dict[tuple[str, tuple[str, ...]], str]:
        result = {}
        for node in self.graph.nodes:
            if node.action is not None and node.lineage_root is not None:
                result[node.action.retry_key] = node.lineage_root
        return result

    def _install(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        sidecar: bytes,
        certificate: PlanCertificate,
        graph: CausalGraph,
        snapshot: FactSnapshot,
    ) -> None:
        self.problem = problem
        self.plan = tuple(plan)
        self.sidecar = sidecar
        self.certificate = certificate
        self.graph = graph
        self.snapshot = snapshot
        self.cursor = 0
        self.graph_installs += 1
        self.events.append(f"CERTIFICATE_INSTALLED:{certificate.certificate_hash}")
        self.events.append("GRAPH_INSTALLED")

    def _halt(self, attempt_id: str | None, cause: str) -> ControllerResult:
        self.events.append("HALT_PENDING")
        acknowledged = bool(self.dispatcher.request_emergency_halt(attempt_id))
        return self._result(
            ControllerStatus.SAFE_STOPPED if acknowledged else ControllerStatus.UNSAFE_TERMINAL,
            cause,
        )

    def _known_gate(
        self,
        snapshot: FactSnapshot,
        positive: frozenset[Fact],
        negative: frozenset[Fact],
    ) -> tuple[bool, bool]:
        required = positive | negative
        if snapshot.unknown(required):
            return False, False
        return snapshot.satisfies(positive=positive, negative=negative), True

    def _install_certified(
        self,
        problem: TaskProblem,
        plan: Sequence[GroundAction],
        sidecar: bytes,
        certificate: PlanCertificate,
        snapshot: FactSnapshot,
    ) -> ControllerResult | object | None:
        if certificate.context.epoch_id != snapshot.epoch_id:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "LIVE_INSTALL_CONFLICT",
            )
        if int(self.grounder.acquire_epoch(ContextPhase.PRE_DISPATCH_FACTS)) != snapshot.epoch_id:
            self.events.append("STALE_CALLBACK_NOOP")
            return _STALE_INSTALL
        try:
            graph = self.compiler.compile(
                problem,
                plan,
                sidecar,
                certificate,
                certificate.context,
                forbidden_retry_keys=frozenset(self.forbidden_retry_keys),
                retry_ledger_version=self.retry_ledger.version,
            )
        except CompilerError as error:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                f"COMPILER_ERROR:{error}",
            )
        self._install(problem, plan, sidecar, certificate, graph, snapshot)
        return None

    def _restart_stale_recovery(
        self,
        reason: RecoveryReason,
    ) -> ControllerResult | None:
        required = set(
            self.problem.initial_state
            | self.problem.initial_false
            | self.problem.goal
            | self.problem.negative_goal
        )
        for action in self.plan[self.cursor :]:
            required.update(_action_relevant_facts(action))
        phase = (
            ContextPhase.FINAL_GOAL
            if reason is RecoveryReason.FINAL_GOAL_FAILURE
            else ContextPhase.PRE_DISPATCH_FACTS
        )
        occurrence_id = (
            self.graph.canonical_agenda[self.cursor]
            if self.cursor < len(self.graph.canonical_agenda)
            else None
        )
        refreshed = self._ground(
            phase,
            frozenset(required),
            occurrence_id=occurrence_id,
        )
        if refreshed.status is not GroundingStatus.OK or refreshed.snapshot is None:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "STATE_GROUNDING_FAILURE",
            )
        self.snapshot = refreshed.snapshot
        obligations: tuple[FailureObligation, ...] = ()
        if reason is RecoveryReason.PRECONDITION_FAILURE and occurrence_id is not None:
            action = self.plan[self.cursor]
            obligations = tuple(
                FailureObligation(occurrence_id, SignedLiteral(fact, True))
                for fact in sorted(action.preconditions - self.snapshot.true_facts)
            ) + tuple(
                FailureObligation(occurrence_id, SignedLiteral(fact, False))
                for fact in sorted(
                    action.negative_preconditions - self.snapshot.false_facts
                )
            )
        elif reason is RecoveryReason.FINAL_GOAL_FAILURE:
            obligations = tuple(
                FailureObligation("GOAL", SignedLiteral(fact, True))
                for fact in sorted(self.problem.goal - self.snapshot.true_facts)
            ) + tuple(
                FailureObligation("GOAL", SignedLiteral(fact, False))
                for fact in sorted(self.problem.negative_goal - self.snapshot.false_facts)
            )
        return self._recover(reason, self.snapshot, obligations)

    def _recover(
        self,
        reason: RecoveryReason,
        snapshot: FactSnapshot,
        obligations: Sequence[FailureObligation] = (),
    ) -> ControllerResult | None:
        if not self.budgets.consume_repair():
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH, "BUDGET_EXHAUSTED"
            )
        current_problem = self._current_problem(snapshot)
        remaining_plan = self.plan[self.cursor :]
        remaining_ids = self.graph.canonical_agenda[self.cursor :]
        suffix_sidecar = self._sidecar_for_cursor()
        recovery_context = self._new_context(ContextPhase.RECOVERY_VAL, snapshot.epoch_id)
        derived_obligations = tuple(obligations)
        excluded_retry_slice: CausalSlice | None = None

        if reason in {RecoveryReason.EFFECT_FAILURE, RecoveryReason.EXECUTOR_FAILED}:
            if not self.budgets.consume_val():
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "BUDGET_EXHAUSTED",
                )
            recertified = self.val_wrapper.validate(
                current_problem,
                remaining_plan,
                suffix_sidecar,
                recovery_context,
                forbidden_retry_keys=frozenset(self.forbidden_retry_keys),
                retry_ledger_version=self.retry_ledger.version,
            )
            if recertified.status is ValidationStatus.VALID:
                current_action = remaining_plan[0]
                if self.retry_policy.retry_allowed(
                    current_action,
                    snapshot,
                    self.retry_ledger,
                    stopped=True,
                    receipt_has_unknown_partial_effect=False,
                ):
                    installation = self._install_certified(
                        current_problem,
                        remaining_plan,
                        suffix_sidecar,
                        recertified.certificate,
                        snapshot,
                    )
                    if installation is _STALE_INSTALL:
                        return self._restart_stale_recovery(reason)
                    return installation
                lineage = self.retry_ledger.lookup(current_action)
                if lineage is not None:
                    self.forbidden_retry_keys.add(lineage.retry_key.digest)
                excluded_retry_slice = build_excluded_retry_slice(
                    self.graph, remaining_ids[0]
                )
            elif recertified.status is ValidationStatus.INVALID:
                try:
                    analysis = trace_invalid_plan(
                        current_problem,
                        remaining_plan,
                        occurrence_ids=remaining_ids,
                    )
                except RepairError as error:
                    return self._result(
                        ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                        f"TRACE_ERROR:{error}",
                    )
                derived_obligations = analysis.obligations
            else:
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "VALIDATION_ERROR",
                )

        causal_slice = excluded_retry_slice or build_causal_slice(
            self.graph, derived_obligations
        )
        result = self.repair_operator.repair(
            current_problem,
            remaining_plan,
            context=recovery_context,
            retry_ledger=self.retry_ledger,
            retry_policy=self.retry_policy,
            forbidden_retry_keys=frozenset(self.forbidden_retry_keys),
            lineage_roots=self._lineage_roots(),
            causal_slice=causal_slice,
            val_call_guard=self.budgets.consume_val,
        )
        if result.status is RepairStatus.BUDGET_EXHAUSTED:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH, "BUDGET_EXHAUSTED"
            )
        if result.status is RepairStatus.NO_CERTIFIED_REPAIR_WITHIN_BUDGET:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "NO_CERTIFIED_REPAIR_WITHIN_BUDGET",
            )
        if result.status is not RepairStatus.CERTIFIED or result.certificate is None:
            return self._result(
                ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                "VALIDATION_ERROR",
            )
        installation = self._install_certified(
            current_problem,
            result.plan,
            result.occurrence_sidecar,
            result.certificate,
            snapshot,
        )
        if installation is _STALE_INSTALL:
            return self._restart_stale_recovery(reason)
        return installation

    def run(self) -> ControllerResult:
        if self._latched_result is not None:
            return self._latched_result
        for _ in range(10000):
            if self.cursor >= len(self.graph.canonical_agenda):
                final = self._ground(
                    ContextPhase.FINAL_GOAL,
                    frozenset(self.problem.goal | self.problem.negative_goal),
                )
                if final.status is not GroundingStatus.OK or final.snapshot is None:
                    return self._result(
                        ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                        "STATE_GROUNDING_FAILURE",
                    )
                self.snapshot = final.snapshot
                satisfied, known = self._known_gate(
                    final.snapshot,
                    self.problem.goal,
                    self.problem.negative_goal,
                )
                if not known:
                    return self._result(
                        ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                        "PLAN_GROUNDING_INCOMPLETE",
                    )
                if not satisfied:
                    self.events.append("FINAL_GOAL_GATE_REJECTED")
                    obligations = tuple(
                        FailureObligation("GOAL", SignedLiteral(fact, True))
                        for fact in sorted(self.problem.goal - final.snapshot.true_facts)
                    ) + tuple(
                        FailureObligation("GOAL", SignedLiteral(fact, False))
                        for fact in sorted(self.problem.negative_goal - final.snapshot.false_facts)
                    )
                    terminal = self._recover(
                        RecoveryReason.FINAL_GOAL_FAILURE,
                        final.snapshot,
                        obligations,
                    )
                    if terminal is not None:
                        return terminal
                    continue
                evaluator_status = self.evaluator.evaluate(self.evaluator_handle)
                mapping = {
                    EvaluatorStatus.EPISODE_SUCCESS: ControllerStatus.EPISODE_SUCCESS,
                    EvaluatorStatus.EPISODE_FAIL: ControllerStatus.EPISODE_FAIL,
                    EvaluatorStatus.EVALUATOR_ERROR: ControllerStatus.EVALUATOR_ERROR,
                    EvaluatorStatus.EVALUATOR_TIMEOUT: ControllerStatus.EVALUATOR_TIMEOUT,
                }
                return self._result(mapping[evaluator_status], evaluator_status.value)

            occurrence_id = self.graph.canonical_agenda[self.cursor]
            node = self.graph.node_map[occurrence_id]
            action = node.action
            assert action is not None
            required = _action_relevant_facts(action)
            current_epoch = int(self.grounder.acquire_epoch(ContextPhase.PRE_DISPATCH_FACTS))
            if self.snapshot.epoch_id != current_epoch or self.snapshot.unknown(required):
                grounded = self._ground(
                    ContextPhase.PRE_DISPATCH_FACTS,
                    required,
                    occurrence_id=occurrence_id,
                    forced_epoch=current_epoch,
                )
                if grounded.status is not GroundingStatus.OK or grounded.snapshot is None:
                    return self._result(
                        ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                        "STATE_GROUNDING_FAILURE",
                    )
                self.snapshot = grounded.snapshot
            if self.snapshot.unknown(required):
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "PLAN_GROUNDING_INCOMPLETE",
                )
            preconditions_hold, known = self._known_gate(
                self.snapshot,
                action.preconditions,
                action.negative_preconditions,
            )
            if not known:
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "PLAN_GROUNDING_INCOMPLETE",
                )
            if not preconditions_hold:
                self.events.append("PRECONDITION_GATE_REJECTED")
                obligations = tuple(
                    FailureObligation(occurrence_id, SignedLiteral(fact, True))
                    for fact in sorted(action.preconditions - self.snapshot.true_facts)
                ) + tuple(
                    FailureObligation(occurrence_id, SignedLiteral(fact, False))
                    for fact in sorted(
                        action.negative_preconditions - self.snapshot.false_facts
                    )
                )
                terminal = self._recover(
                    RecoveryReason.PRECONDITION_FAILURE,
                    self.snapshot,
                    obligations,
                )
                if terminal is not None:
                    return terminal
                continue

            self.events.append(
                f"FACTS_AUTHORIZED:{occurrence_id}:epoch-{self.snapshot.epoch_id}"
            )
            if not self.budgets.consume_physical():
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "BUDGET_EXHAUSTED",
                )
            dispatch_context = self._new_context(
                ContextPhase.PRE_DISPATCH_FACTS,
                self.snapshot.epoch_id,
                occurrence_id=occurrence_id,
            )
            start = self.dispatcher.consume_permit_and_enqueue(
                action, dispatch_context, self.snapshot
            )
            if start.status is not DispatchStatus.ENQUEUED:
                self.budgets.refund_unenqueued_physical()
            if start.status is DispatchStatus.EXECUTOR_REJECTED_NOT_ENQUEUED:
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "EXECUTOR_REJECTED_NOT_ENQUEUED",
                )
            if start.status is DispatchStatus.ACTION_BUDGET_EXHAUSTED:
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "BUDGET_EXHAUSTED",
                )
            if (
                start.status is not DispatchStatus.ENQUEUED
                or start.attempt_id is None
                or start.safety_epoch is None
            ):
                return self._halt(None, start.status.value)
            attempt_context = replace(
                dispatch_context,
                attempt_id=start.attempt_id,
                safety_epoch=start.safety_epoch,
            )
            if start.context != attempt_context:
                return self._halt(start.attempt_id, "SAFETY_CONTEXT_MISMATCH")
            self.events.append(
                f"EXECUTION_AUTHORIZED:{start.attempt_id}:"
                f"epoch-{self.snapshot.epoch_id}:safety-{start.safety_epoch}"
            )
            try:
                self.retry_ledger.record_dispatch(
                    action,
                    self.snapshot,
                    lineage_root=node.lineage_root or f"lineage:{occurrence_id}",
                )
            except RepairError as error:
                self._active_attempt_id = start.attempt_id
                return self._halt(start.attempt_id, f"RETRY_POLICY_ERROR:{error}")
            self._active_attempt_id = start.attempt_id
            self.events.append(f"ATTEMPT_RUNNING:{start.attempt_id}")
            outcome = None
            for _ in range(16):
                candidate_outcome = self.dispatcher.await_outcome(start)
                if candidate_outcome.context != attempt_context:
                    self.events.append("STALE_CALLBACK_NOOP")
                    continue
                outcome = candidate_outcome
                break
            if outcome is None:
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.POST_STOP_UNKNOWN,
                        pre_epoch=self.snapshot.epoch_id,
                        post_epoch=None,
                        stop_confirmed=False,
                        effect_status="UNKNOWN",
                        reason="STALE_EXECUTOR_CALLBACKS",
                    )
                )
                return self._halt(start.attempt_id, "STALE_EXECUTOR_CALLBACKS")
            stopped = (
                outcome.stopped
                and outcome.stop_ack_attempt_id == start.attempt_id
                and outcome.attempt_id == start.attempt_id
            )
            if outcome.status not in {
                ExecutorStatus.SUCCEEDED,
                ExecutorStatus.EXECUTOR_FAILED,
            } or not stopped:
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.POST_STOP_UNKNOWN,
                        pre_epoch=self.snapshot.epoch_id,
                        post_epoch=outcome.settled_epoch,
                        stop_confirmed=stopped,
                        effect_status="UNKNOWN",
                        reason=outcome.status.value,
                    )
                )
                return self._halt(start.attempt_id, outcome.status.value)
            if outcome.settled_epoch is None:
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.POST_STOP_UNKNOWN,
                        pre_epoch=self.snapshot.epoch_id,
                        post_epoch=None,
                        stop_confirmed=True,
                        effect_status="UNKNOWN",
                        reason="SETTLING_TIMEOUT",
                    )
                )
                return self._halt(start.attempt_id, "SETTLING_TIMEOUT")

            self.events.append(f"STOPPED_UNCOMMITTED:{start.attempt_id}")
            pre_epoch = self.snapshot.epoch_id
            post = self._ground(
                ContextPhase.POST_STOP_FACTS,
                frozenset(action.add_effects | action.del_effects),
                occurrence_id=occurrence_id,
                attempt_id=start.attempt_id,
                safety_epoch=start.safety_epoch,
                forced_epoch=outcome.settled_epoch,
            )
            if post.status is not GroundingStatus.OK or post.snapshot is None:
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.POST_STOP_UNKNOWN,
                        pre_epoch=pre_epoch,
                        post_epoch=outcome.settled_epoch,
                        stop_confirmed=True,
                        effect_status="UNKNOWN",
                        reason=post.reason,
                    )
                )
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "POST_STOP_GROUNDING_FAILURE",
                )
            self.snapshot = post.snapshot
            effects_hold, effects_known = self._known_gate(
                self.snapshot,
                action.add_effects,
                action.del_effects,
            )
            if not effects_known:
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.POST_STOP_UNKNOWN,
                        pre_epoch=pre_epoch,
                        post_epoch=self.snapshot.epoch_id,
                        stop_confirmed=True,
                        effect_status="UNKNOWN",
                    )
                )
                return self._result(
                    ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                    "POST_STOP_GROUNDING_FAILURE",
                )

            if outcome.status is ExecutorStatus.EXECUTOR_FAILED or not effects_hold:
                self.events.append(
                    "EXECUTOR_FAILED"
                    if outcome.status is ExecutorStatus.EXECUTOR_FAILED
                    else "EFFECT_GATE_REJECTED"
                )
                self.retry_ledger.record_effect_failure(action)
                self.receipts.append(
                    AttemptReceipt(
                        attempt_id=start.attempt_id,
                        occurrence_id=occurrence_id,
                        graph_version=self.graph.graph_version,
                        schema=action.schema,
                        status=AttemptReceiptStatus.FAILED,
                        pre_epoch=pre_epoch,
                        post_epoch=self.snapshot.epoch_id,
                        stop_confirmed=True,
                        effect_status="FAILED",
                        reason=(
                            "EXECUTOR_FAILED"
                            if outcome.status is ExecutorStatus.EXECUTOR_FAILED
                            else "EFFECT_FAILURE"
                        ),
                    )
                )
                self._active_attempt_id = None
                terminal = self._recover(
                    RecoveryReason.EXECUTOR_FAILED
                    if outcome.status is ExecutorStatus.EXECUTOR_FAILED
                    else RecoveryReason.EFFECT_FAILURE,
                    self.snapshot,
                )
                if terminal is not None:
                    return terminal
                continue

            self.receipts.append(
                AttemptReceipt(
                    attempt_id=start.attempt_id,
                    occurrence_id=occurrence_id,
                    graph_version=self.graph.graph_version,
                    schema=action.schema,
                    status=AttemptReceiptStatus.COMMITTED,
                    pre_epoch=pre_epoch,
                    post_epoch=self.snapshot.epoch_id,
                    stop_confirmed=True,
                    effect_status="CONFIRMED",
                )
            )
            self._active_attempt_id = None
            self.cursor += 1
        return self._result(
            ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
            "CONTROLLER_STEP_LIMIT",
        )
