from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import heapq
from itertools import product
import json
from typing import Callable, FrozenSet, Mapping, Sequence, Tuple

from pi05_libero_repro.logiv.dag import CausalGraph, SignedLiteral
from pi05_libero_repro.logiv.domain import DomainError, FixedDomain, validate_state
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
    TruthValue,
)
from pi05_libero_repro.logiv.val import (
    PlanCertificate,
    SignedTraceStatus,
    ValidationStatus,
    ValWrapper,
    run_signed_trace,
)


ActionSignature = Tuple[str, Tuple[str, ...]]


class RepairError(RuntimeError):
    pass


class TraceKind(str, Enum):
    ACTION_FAILURE = "ACTION_FAILURE"
    GOAL_FAILURE = "GOAL_FAILURE"


@dataclass(frozen=True)
class FailureObligation:
    consumer_id: str
    literal: SignedLiteral


@dataclass(frozen=True)
class TraceAnalysis:
    kind: TraceKind
    obligations: Tuple[FailureObligation, ...]


@dataclass(frozen=True)
class CausalSlice:
    node_ids: FrozenSet[str]
    canonical_nodes: Tuple[str, ...]
    action_signatures: Tuple[ActionSignature, ...]


def trace_invalid_plan(
    problem: TaskProblem,
    plan: Sequence[GroundAction],
    occurrence_ids: Sequence[str] | None = None,
) -> TraceAnalysis:
    trace = run_signed_trace(problem, plan)
    if trace.status not in {
        SignedTraceStatus.ACTION_PRECONDITION_FAILURE,
        SignedTraceStatus.FINAL_GOAL_FAILURE,
    }:
        raise RepairError(f"trace cannot form a repair obligation: {trace.status.value}")
    obligations = []
    for item in trace.obligations:
        if item.consumer_index is None:
            consumer_id = "GOAL"
        elif occurrence_ids is None:
            consumer_id = item.consumer
        else:
            if item.consumer_index >= len(occurrence_ids):
                raise RepairError("trace occurrence index is out of range")
            consumer_id = occurrence_ids[item.consumer_index]
        obligations.append(
            FailureObligation(
                consumer_id=consumer_id,
                literal=SignedLiteral(item.fact, positive=item.positive),
            )
        )
    kind = (
        TraceKind.ACTION_FAILURE
        if trace.status is SignedTraceStatus.ACTION_PRECONDITION_FAILURE
        else TraceKind.GOAL_FAILURE
    )
    return TraceAnalysis(kind=kind, obligations=tuple(obligations))


def build_causal_slice(
    graph: CausalGraph,
    obligations: Sequence[FailureObligation],
) -> CausalSlice:
    included: set[str] = set()
    expanded: set[str] = set()

    def include_predecessors(target: str) -> None:
        if target in expanded or target in {"INIT", "GOAL"}:
            return
        expanded.add(target)
        for link in graph.causal_links:
            if link.target == target:
                included.add(link.source)
                include_predecessors(link.source)

    for obligation in obligations:
        included.add(obligation.consumer_id)
        matching = [
            link
            for link in graph.causal_links
            if link.target == obligation.consumer_id and link.literal == obligation.literal
        ]
        for link in matching:
            included.add(link.source)
            include_predecessors(link.source)
    canonical_nodes = tuple(
        node_id for node_id in graph.canonical_agenda if node_id in included
    )
    signatures = tuple(
        graph.node_map[node_id].action.retry_key
        for node_id in canonical_nodes
        if graph.node_map[node_id].action is not None
    )
    return CausalSlice(
        node_ids=frozenset(included),
        canonical_nodes=canonical_nodes,
        action_signatures=signatures,
    )


def build_excluded_retry_slice(graph: CausalGraph, occurrence_id: str) -> CausalSlice:
    if occurrence_id not in graph.node_map or occurrence_id in {"INIT", "GOAL"}:
        raise RepairError(f"unknown failed occurrence: {occurrence_id}")
    included = {occurrence_id}
    frontier = [occurrence_id]
    while frontier:
        source = frontier.pop()
        for edge in graph.edges:
            if edge.source == source and edge.target not in included:
                included.add(edge.target)
                frontier.append(edge.target)
    canonical_nodes = tuple(
        node_id for node_id in graph.canonical_agenda if node_id in included
    )
    signatures = tuple(
        graph.node_map[node_id].action.retry_key
        for node_id in canonical_nodes
        if graph.node_map[node_id].action is not None
    )
    return CausalSlice(frozenset(included), canonical_nodes, signatures)


@dataclass(frozen=True)
class RetryKey:
    digest: str
    schema: str
    arguments: Tuple[str, ...]
    relevant_facts: Tuple[Tuple[Fact, TruthValue], ...]
    lineage_root: str


@dataclass
class RetryLineage:
    signature: ActionSignature
    lineage_root: str
    retry_key: RetryKey
    physical_attempts: int = 0
    effect_failures: int = 0


def _relevant_facts(action: GroundAction) -> Tuple[Fact, ...]:
    return tuple(
        sorted(
            action.preconditions
            | action.negative_preconditions
            | action.add_effects
            | action.del_effects
        )
    )


class RetryLedger:
    def __init__(self) -> None:
        self._by_signature: dict[ActionSignature, RetryLineage] = {}
        self.version = 0

    def lookup(self, action: GroundAction) -> RetryLineage | None:
        return self._by_signature.get(action.retry_key)

    def ensure_lineage(
        self,
        action: GroundAction,
        snapshot: FactSnapshot,
        *,
        lineage_root: str,
    ) -> RetryLineage:
        existing = self.lookup(action)
        if existing is not None:
            return existing
        relevant = tuple((fact, snapshot.truth(fact)) for fact in _relevant_facts(action))
        unknown = [fact for fact, truth in relevant if truth is TruthValue.UNKNOWN]
        if unknown:
            raise RepairError(
                "cannot mint retry key with UNKNOWN relevant facts: "
                + ", ".join(str(fact) for fact in unknown)
            )
        payload = {
            "schema": action.schema,
            "arguments": action.arguments,
            "relevant_facts": [
                (fact.predicate, fact.arguments, truth.value) for fact, truth in relevant
            ],
            "lineage_root": lineage_root,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        lineage = RetryLineage(
            signature=action.retry_key,
            lineage_root=lineage_root,
            retry_key=RetryKey(
                digest=digest,
                schema=action.schema,
                arguments=action.arguments,
                relevant_facts=relevant,
                lineage_root=lineage_root,
            ),
        )
        self._by_signature[action.retry_key] = lineage
        self.version += 1
        return lineage

    def record_dispatch(
        self,
        action: GroundAction,
        snapshot: FactSnapshot,
        *,
        lineage_root: str,
    ) -> RetryLineage:
        lineage = self.ensure_lineage(action, snapshot, lineage_root=lineage_root)
        lineage.physical_attempts += 1
        self.version += 1
        return lineage

    def record_effect_failure(self, action: GroundAction) -> RetryLineage:
        lineage = self.lookup(action)
        if lineage is None or lineage.physical_attempts <= lineage.effect_failures:
            raise RepairError("effect failure has no unmatched physical dispatch")
        lineage.effect_failures += 1
        self.version += 1
        return lineage


@dataclass(frozen=True)
class RetryPolicy:
    max_retries_per_lineage: int

    def __post_init__(self) -> None:
        if self.max_retries_per_lineage < 0:
            raise ValueError("max_retries_per_lineage must be nonnegative")

    def plan_allowed(
        self,
        plan: Sequence[GroundAction],
        ledger: RetryLedger,
        forbidden_retry_keys: FrozenSet[str],
    ) -> bool:
        planned_counts: dict[ActionSignature, int] = {}
        for action in plan:
            planned_counts[action.retry_key] = planned_counts.get(action.retry_key, 0) + 1
            lineage = ledger.lookup(action)
            if lineage is not None:
                if lineage.retry_key.digest in forbidden_retry_keys:
                    return False
                if lineage.physical_attempts > 0 and not action.repeatable:
                    return False
                total = lineage.physical_attempts + planned_counts[action.retry_key]
                if total > 1 + self.max_retries_per_lineage:
                    return False
            elif planned_counts[action.retry_key] > 1 + self.max_retries_per_lineage:
                return False
        return True

    def retry_allowed(
        self,
        action: GroundAction,
        snapshot: FactSnapshot,
        ledger: RetryLedger,
        *,
        stopped: bool,
        receipt_has_unknown_partial_effect: bool,
    ) -> bool:
        lineage = ledger.lookup(action)
        if lineage is None or not stopped or receipt_has_unknown_partial_effect:
            return False
        if not action.repeatable or lineage.effect_failures > self.max_retries_per_lineage:
            return False
        return not any(
            snapshot.truth(fact) is TruthValue.UNKNOWN for fact in _relevant_facts(action)
        )


@dataclass(frozen=True)
class RepairBounds:
    max_edits: int
    max_candidates: int
    max_val_calls: int

    def __post_init__(self) -> None:
        if min(self.max_edits, self.max_candidates, self.max_val_calls) < 0:
            raise ValueError("repair bounds must be nonnegative")


class RepairStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    NO_CERTIFIED_REPAIR_WITHIN_BUDGET = "NO_CERTIFIED_REPAIR_WITHIN_BUDGET"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class RepairResult:
    status: RepairStatus
    plan: Tuple[GroundAction, ...] = ()
    occurrence_sidecar: bytes = b""
    certificate: PlanCertificate | None = None
    explored_candidates: int = 0
    val_calls: int = 0
    reason: str = ""


def _edit_distance(left: Sequence[ActionSignature], right: Sequence[ActionSignature]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _goal_satisfied(
    problem: TaskProblem,
    true_facts: FrozenSet[Fact],
    false_facts: FrozenSet[Fact],
) -> bool:
    return problem.goal <= true_facts and problem.negative_goal <= false_facts


def _transition(
    problem: TaskProblem,
    action: GroundAction,
    true_facts: FrozenSet[Fact],
    false_facts: FrozenSet[Fact],
) -> tuple[FrozenSet[Fact], FrozenSet[Fact]] | None:
    if not action.preconditions <= true_facts:
        return None
    if not action.negative_preconditions <= false_facts:
        return None
    next_true = frozenset((true_facts - action.del_effects) | action.add_effects)
    next_false = frozenset((false_facts - action.add_effects) | action.del_effects)
    if next_true & next_false:
        return None
    try:
        validate_state(problem, next_true)
    except DomainError:
        return None
    if next_true == true_facts and next_false == false_facts:
        return None
    return next_true, next_false


class PddlPlanner:
    """Bounded symbolic search over the fixed PDDL domain, followed by VAL."""

    def __init__(
        self,
        val_wrapper: ValWrapper,
        *,
        allowed_schemas: FrozenSet[str],
        bounds: RepairBounds,
        retry_policy: RetryPolicy | None = None,
        decompose_macro_sources: FrozenSet[str] = frozenset(),
    ) -> None:
        self.val_wrapper = val_wrapper
        self.allowed_schemas = allowed_schemas
        self.bounds = bounds
        self.retry_policy = retry_policy or RetryPolicy(max_retries_per_lineage=1)
        self.decompose_macro_sources = decompose_macro_sources

    def _catalog(self, problem: TaskProblem) -> Tuple[GroundAction, ...]:
        domain = FixedDomain()
        unknown = self.allowed_schemas - domain.schemas.keys()
        if unknown:
            raise RepairError(f"unknown allowed schemas: {sorted(unknown)}")
        names = tuple(sorted(problem.object_types))
        actions = set()
        for schema_name in sorted(self.allowed_schemas):
            schema = domain.schemas[schema_name]
            for arguments in product(names, repeat=len(schema.parameters)):
                try:
                    action = domain.ground(problem, schema_name, arguments)
                except DomainError:
                    continue
                if (
                    schema_name in {"place-in", "place-on", "place-relative"}
                    and action.arguments[1] in self.decompose_macro_sources
                ):
                    continue
                actions.add(action)
        return tuple(sorted(actions, key=lambda action: action.pddl()))

    def _sidecar(
        self,
        plan: Sequence[GroundAction],
        context: ContextEnvelope,
        ledger: RetryLedger,
        lineage_roots: Mapping[ActionSignature, str],
    ) -> bytes:
        payload = []
        for index, action in enumerate(plan):
            prior = ledger.lookup(action)
            lineage_root = (
                prior.lineage_root
                if prior is not None
                else lineage_roots.get(
                    action.retry_key,
                    "repair:" + hashlib.sha256(action.pddl().encode("utf-8")).hexdigest()[:16],
                )
            )
            occurrence_hash = hashlib.sha256(
                f"{context.request_id}:{index}:{action.pddl()}".encode("utf-8")
            ).hexdigest()[:12]
            payload.append(
                {
                    "occurrence_id": f"repair-o{index:03d}-{occurrence_hash}",
                    "schema": action.schema,
                    "arguments": list(action.arguments),
                    "lineage_root": lineage_root,
                }
            )
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def repair(
        self,
        problem: TaskProblem,
        remaining_plan: Sequence[GroundAction],
        *,
        context: ContextEnvelope,
        retry_ledger: RetryLedger | None = None,
        retry_policy: RetryPolicy | None = None,
        forbidden_retry_keys: FrozenSet[str] = frozenset(),
        lineage_roots: Mapping[ActionSignature, str] | None = None,
        causal_slice: CausalSlice | None = None,
        val_call_guard: Callable[[], bool] | None = None,
    ) -> RepairResult:
        ledger = retry_ledger or RetryLedger()
        policy = retry_policy or self.retry_policy
        lineage_roots = lineage_roots or {}
        try:
            validate_state(problem, problem.initial_state)
            catalog = self._catalog(problem)
        except (DomainError, RepairError) as error:
            return RepairResult(RepairStatus.VALIDATION_ERROR, reason=str(error))

        remaining_signatures = tuple(action.retry_key for action in remaining_plan)
        ranked_signatures = tuple(
            dict.fromkeys(
                (causal_slice.action_signatures if causal_slice is not None else ())
                + remaining_signatures
            )
        )
        old_rank = {signature: index for index, signature in enumerate(ranked_signatures)}
        initially_satisfied = {
            SignedLiteral(fact, True) for fact in problem.goal if fact in problem.initial_state
        }
        initially_satisfied.update(
            SignedLiteral(fact, False)
            for fact in problem.negative_goal
            if fact in problem.initial_false
        )

        def priority(
            path: Tuple[GroundAction, ...],
            true_facts: FrozenSet[Fact],
            false_facts: FrozenSet[Fact],
        ) -> tuple:
            still_satisfied = {
                literal
                for literal in initially_satisfied
                if (
                    literal.fact in true_facts
                    if literal.positive
                    else literal.fact in false_facts
                )
            }
            signatures = tuple(action.retry_key for action in path)
            return (
                len(initially_satisfied - still_satisfied),
                _edit_distance(signatures, remaining_signatures),
                len(path),
                tuple(old_rank.get(signature, len(old_rank) + 1) for signature in signatures),
                tuple(action.pddl() for action in path),
            )

        initial_true = frozenset(problem.initial_state)
        initial_false = frozenset(problem.initial_false)
        initial_path: Tuple[GroundAction, ...] = ()
        queue = [
            (
                priority(initial_path, initial_true, initial_false),
                0,
                initial_path,
                initial_true,
                initial_false,
            )
        ]
        visited = {(initial_true, initial_false): queue[0][0]}
        counter = 0
        explored = 0
        val_calls = 0
        max_depth = len(remaining_plan) + self.bounds.max_edits

        while queue and explored < self.bounds.max_candidates:
            path_priority, _, path, true_facts, false_facts = heapq.heappop(queue)
            if visited.get((true_facts, false_facts)) != path_priority:
                continue
            explored += 1
            if _goal_satisfied(problem, true_facts, false_facts):
                edits = _edit_distance(
                    tuple(action.retry_key for action in path), remaining_signatures
                )
                if edits <= self.bounds.max_edits and policy.plan_allowed(
                    path, ledger, forbidden_retry_keys
                ):
                    if val_calls >= self.bounds.max_val_calls:
                        break
                    if val_call_guard is not None and not val_call_guard():
                        return RepairResult(
                            status=RepairStatus.BUDGET_EXHAUSTED,
                            explored_candidates=explored,
                            val_calls=val_calls,
                            reason="episode-global VAL budget exhausted",
                        )
                    val_calls += 1
                    sidecar = self._sidecar(path, context, ledger, lineage_roots)
                    candidate_context = replace(
                        context,
                        request_id=f"{context.request_id}-candidate-{val_calls}",
                        request_generation=0,
                    )
                    validation = self.val_wrapper.validate(
                        problem,
                        path,
                        sidecar,
                        candidate_context,
                        forbidden_retry_keys=frozenset(forbidden_retry_keys),
                        retry_ledger_version=ledger.version,
                    )
                    if validation.status is ValidationStatus.VALID:
                        return RepairResult(
                            status=RepairStatus.CERTIFIED,
                            plan=path,
                            occurrence_sidecar=sidecar,
                            certificate=validation.certificate,
                            explored_candidates=explored,
                            val_calls=val_calls,
                        )
                    if validation.status is ValidationStatus.VALIDATION_ERROR:
                        return RepairResult(
                            status=RepairStatus.VALIDATION_ERROR,
                            explored_candidates=explored,
                            val_calls=val_calls,
                            reason=validation.reason or "VAL validation error",
                        )
            if len(path) >= max_depth:
                continue
            for action in catalog:
                next_path = path + (action,)
                if not policy.plan_allowed(next_path, ledger, forbidden_retry_keys):
                    continue
                successor = _transition(problem, action, true_facts, false_facts)
                if successor is None:
                    continue
                next_true, next_false = successor
                next_priority = priority(next_path, next_true, next_false)
                state_key = (next_true, next_false)
                previous = visited.get(state_key)
                if previous is not None and previous <= next_priority:
                    continue
                visited[state_key] = next_priority
                counter += 1
                heapq.heappush(
                    queue,
                    (next_priority, counter, next_path, next_true, next_false),
                )

        return RepairResult(
            status=RepairStatus.NO_CERTIFIED_REPAIR_WITHIN_BUDGET,
            explored_candidates=explored,
            val_calls=val_calls,
            reason="no certified repair found within configured bounds",
        )


RepairOperator = PddlPlanner
