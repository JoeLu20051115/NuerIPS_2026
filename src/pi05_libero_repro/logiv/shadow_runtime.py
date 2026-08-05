from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from pi05_libero_repro.logiv.dag import CausalGraph, NodeKind
from pi05_libero_repro.logiv.initial_proposal import (
    InitialProposalResult,
    InitialProposalStatus,
    run_initial_proposal,
)
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    GoalMode,
    GroundAction,
    ProposalPackage,
    TaskProblem,
    TruthValue,
)
from pi05_libero_repro.logiv.recovery_records import (
    CollectionLabel,
    RecoveryRootArtifacts,
)
from pi05_libero_repro.logiv.shadow_monitor import (
    MonitorEvidenceContract,
    ShadowCertificateReconciler,
    ShadowPlanContext,
    ShadowTrigger,
    StableRecoveryObserver,
    TransitionFeatureReader,
    VersionedActionEventTracker,
)
from pi05_libero_repro.logiv.val import PlanCertificate
from pi05_libero_repro.protocol import (
    BaseActionPrefixHasher,
    ShadowSettlingContext,
    ShadowStepContext,
)


class CertifiedEpisodeLike(Protocol):
    problem: TaskProblem
    plan: tuple[GroundAction, ...]
    graph: CausalGraph
    certificate: PlanCertificate


@dataclass(frozen=True)
class ShadowValidatedProposal:
    certified_episode: CertifiedEpisodeLike
    snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot]


@dataclass(frozen=True)
class ShadowEpisodeContext:
    task_id: int
    episode_idx: int
    initial_epoch_id: int
    scene_sha256: str
    object_instance_ids: tuple[str, ...]
    initial_state_sha256: str
    parent_trajectory_lineage_sha256: str
    base_prompt_sha256: str
    base_checkpoint_sha256: str
    policy_client_config_sha256: str
    policy_replay_contract_sha256: str | None
    master_seed: int
    policy_seed: int
    simulator_seed: int
    replan_steps: int
    collect_recovery_roots: bool
    collection_label: CollectionLabel
    root_output_dir: Path
    simulator_state_reader: Callable[[], np.ndarray]
    transition_feature_reader: TransitionFeatureReader


@dataclass
class ShadowRuntimeCounters:
    root_count: int = 0
    root_write_errors: int = 0
    proposal_callback_errors: int = 0
    provenance_errors: int = 0
    trace_errors: int = 0


@dataclass
class ShadowRuntime:
    initial_proposal: InitialProposalResult[ShadowValidatedProposal] | None
    observer: Callable[[ShadowStepContext], None] | None
    settling_observer: Callable[[ShadowSettlingContext], None] | None
    monitor: StableRecoveryObserver | None
    counters: ShadowRuntimeCounters
    state_trace: list[dict[str, Any]] = field(default_factory=list)


class ShadowGraphTracker:
    """Stateful projection of physical macro progress onto one fixed graph."""

    _TRANSPORT_SCHEMAS = frozenset({"place-on", "place-in", "place-relative"})

    def __init__(self, graph: CausalGraph, problem: TaskProblem) -> None:
        self.graph = graph
        self.problem = problem
        self._statuses: dict[str, str] = {"INIT": "COMPLETED"}
        self._predecessors = {node.node_id: set() for node in graph.nodes}
        for edge in graph.edges:
            source = graph.node_map[edge.source]
            if source.kind is NodeKind.ACTION:
                self._predecessors[edge.target].add(edge.source)
        relevant = set(problem.initial_state | problem.initial_false)
        relevant.update(problem.goal | problem.negative_goal)
        for node in graph.nodes:
            if node.action is not None:
                relevant.update(
                    node.action.preconditions
                    | node.action.negative_preconditions
                    | node.action.add_effects
                    | node.action.del_effects
                )
        self._relevant_facts = frozenset(relevant)

    @staticmethod
    def _raw_satisfies(
        snapshot: FactSnapshot,
        *,
        positive: frozenset,
        negative: frozenset,
    ) -> bool:
        return all(
            snapshot.raw_truth(fact) is TruthValue.TRUE for fact in positive
        ) and all(
            snapshot.raw_truth(fact) is TruthValue.FALSE for fact in negative
        )

    def _object_candidates(
        self, snapshot: FactSnapshot, object_name: str
    ) -> frozenset:
        universe = snapshot.fact_universe or self._relevant_facts
        return frozenset(
            fact
            for fact in universe
            if (
                fact.predicate == "holding"
                and fact.arguments == (object_name,)
            )
            or (
                fact.predicate == "at"
                and fact.arguments
                and fact.arguments[0] == object_name
            )
        )

    def _transport_state(
        self, snapshot: FactSnapshot, object_name: str
    ) -> tuple[bool, bool]:
        candidates = self._object_candidates(snapshot, object_name)
        values = tuple(snapshot.truth(fact) for fact in candidates)
        unlocated = bool(values) and all(value is TruthValue.FALSE for value in values)
        unresolved = bool(values) and not any(
            value is TruthValue.TRUE for value in values
        ) and any(value is TruthValue.UNKNOWN for value in values)
        return unlocated, unresolved

    def project(
        self,
        snapshot: FactSnapshot,
        *,
        policy_step: int,
        observation_generation: int,
        certificate_state: str,
        phase: str | None = None,
        settling_step: int | None = None,
    ) -> dict[str, Any]:
        statuses: dict[str, str] = {"INIT": "COMPLETED"}
        for node in self.graph.nodes:
            if node.kind is not NodeKind.ACTION or node.action is None:
                continue
            action = node.action
            previous = self._statuses.get(node.node_id)
            completed = snapshot.satisfies(
                positive=action.add_effects, negative=action.del_effects
            )
            if completed:
                statuses[node.node_id] = "COMPLETED"
                continue
            raw_effect = self._raw_satisfies(
                snapshot,
                positive=action.add_effects,
                negative=action.del_effects,
            )
            if raw_effect:
                statuses[node.node_id] = "EFFECT_OBSERVED"
                continue

            object_name = (
                action.arguments[0]
                if action.schema in self._TRANSPORT_SCHEMAS and action.arguments
                else None
            )
            unlocated = unresolved = False
            held = False
            if object_name is not None:
                unlocated, unresolved = self._transport_state(snapshot, object_name)
                held = snapshot.truth(
                    Fact("holding", (object_name,))
                ) is TruthValue.TRUE
            if object_name is not None and previous in {
                "READY",
                "ACTIVE",
                "EFFECT_OBSERVED",
            } and (
                held or unlocated or unresolved
            ):
                statuses[node.node_id] = "ACTIVE"
                continue

            preconditions = action.preconditions | action.negative_preconditions
            has_unknown = unresolved or any(
                snapshot.truth(fact) is TruthValue.UNKNOWN for fact in preconditions
            )
            ready = (
                all(
                    statuses.get(item) == "COMPLETED"
                    for item in self._predecessors[node.node_id]
                )
                and snapshot.satisfies(
                    positive=action.preconditions,
                    negative=action.negative_preconditions,
                )
            )
            statuses[node.node_id] = (
                "READY"
                if ready
                else "PRECONDITION_UNKNOWN"
                if has_unknown
                else "BLOCKED"
            )

        goal_completed = self._raw_satisfies(
            snapshot,
            positive=self.problem.goal,
            negative=self.problem.negative_goal,
        )
        statuses["GOAL"] = (
            "COMPLETED"
            if goal_completed
            else "READY"
            if all(
                statuses.get(item) == "COMPLETED"
                for item in self._predecessors["GOAL"]
            )
            else "BLOCKED"
        )
        self._statuses = statuses
        state: dict[str, Any] = {
            "policy_step": policy_step,
            "observation_generation": observation_generation,
            "certificate_state": certificate_state,
            "graph_version": self.graph.graph_version,
            "graph_hash": self.graph.graph_hash,
            "nodes": [
                {"node_id": node.node_id, "status": statuses[node.node_id]}
                for node in self.graph.nodes
            ],
        }
        if phase is not None:
            state["phase"] = phase
        if settling_step is not None:
            state["settling_step"] = settling_step
        return state


def project_graph_state(
    graph: CausalGraph,
    problem: TaskProblem,
    snapshot: FactSnapshot,
    *,
    policy_step: int,
    observation_generation: int,
    certificate_state: str,
) -> dict[str, Any]:
    """Project one snapshot without retaining temporal action progress."""

    return ShadowGraphTracker(graph, problem).project(
        snapshot,
        policy_step=policy_step,
        observation_generation=observation_generation,
        certificate_state=certificate_state,
    )


def _same_value(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            return False
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and np.array_equal(first, second, equal_nan=True)
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return False
        return set(first) == set(second) and all(
            _same_value(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        if not isinstance(first, (tuple, list)) or not isinstance(second, (tuple, list)):
            return False
        return len(first) == len(second) and all(
            _same_value(left, right) for left, right in zip(first, second)
        )
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _same_context(first: ShadowStepContext, second: ShadowStepContext) -> bool:
    return all(
        _same_value(getattr(first, name), getattr(second, name))
        for name in first.__dataclass_fields__
    )


def build_shadow_runtime(
    *,
    provider: Any,
    provider_name: str,
    episode_context: ShadowEpisodeContext,
    goal_mode: GoalMode,
    live_validator: Callable[
        [ProposalPackage, Mapping[str, Any]], ShadowValidatedProposal
    ],
    monitor_contract: MonitorEvidenceContract | None,
    root_collector: Callable[
        [ShadowTrigger, ShadowStepContext, ShadowEpisodeContext],
        RecoveryRootArtifacts | None,
    ],
    interval_steps: int,
    confirmation_count: int,
    topology_only: bool = False,
) -> ShadowRuntime:
    """Build a fail-open observer without touching the proposal provider."""

    if topology_only and interval_steps <= 0:
        raise ValueError("topology-only interval must be positive")
    counters = ShadowRuntimeCounters()
    action_prefix = BaseActionPrefixHasher()
    previous_context: ShadowStepContext | None = None
    disabled = False
    envelopes: dict[int, str] = {}
    active_trigger_context: ShadowStepContext | None = None
    topology_auditor: Callable[[ShadowStepContext], None] | None = None
    settling_auditor: Callable[[ShadowSettlingContext], None] | None = None

    runtime = ShadowRuntime(
        initial_proposal=None,
        observer=None,
        settling_observer=None,
        monitor=None,
        counters=counters,
    )

    def disable_for_provenance() -> None:
        nonlocal disabled
        if not disabled:
            counters.provenance_errors += 1
        disabled = True

    def remember_envelope(index: int, value: str | None) -> None:
        if value is None:
            return
        existing = envelopes.get(index)
        if existing is not None and existing != value:
            raise ValueError("Base request envelope changed for one request index")
        envelopes[index] = value

    def validate_step(context: ShadowStepContext) -> bool:
        nonlocal previous_context
        if previous_context is not None:
            if context.policy_step == previous_context.policy_step:
                if _same_context(previous_context, context):
                    return False
                raise ValueError("duplicate policy step changed content")
            if context.policy_step != previous_context.policy_step + 1:
                raise ValueError("shadow policy steps are not contiguous")
        elif context.policy_step != 0:
            raise ValueError("first shadow callback must be policy step zero")

        if context.policy_step == 0:
            if (
                context.last_action is not None
                or context.pending_base_actions.size != 0
                or context.base_policy_request_count != 0
                or context.active_base_request_index is not None
                or context.next_base_request_index != 0
            ):
                raise ValueError("step-zero Base provenance is not pristine")
        else:
            if context.last_action is None:
                raise ValueError("action step is missing its executed Base action")
            if (
                context.base_policy_request_count <= 0
                or context.next_base_request_index
                != context.base_policy_request_count
                or context.active_base_request_index
                != context.base_policy_request_count - 1
            ):
                raise ValueError("Base request indices are inconsistent")
            if previous_context is not None and (
                context.base_policy_request_count
                < previous_context.base_policy_request_count
                or (
                    previous_context.active_base_request_index is not None
                    and context.active_base_request_index
                    < previous_context.active_base_request_index
                )
            ):
                raise ValueError("Base request indices regressed")
            pending = context.pending_base_actions
            if pending.ndim != 2 or pending.shape[1] != 7:
                raise ValueError("pending Base actions must have shape Nx7")
            if (
                context.base_action_response_size is None
                or context.base_action_chunk_size <= 0
                or context.base_action_response_size < context.base_action_chunk_size
                or context.pending_base_action_offset < 0
                or context.pending_base_action_offset > context.base_action_chunk_size
                or pending.shape[0]
                != context.base_action_chunk_size
                - context.pending_base_action_offset
            ):
                raise ValueError("pending Base action suffix is inconsistent")

        expected_digest = action_prefix.update_and_hexdigest(context.last_action)
        if expected_digest != context.base_action_prefix_sha256:
            raise ValueError("Base action-prefix digest mismatch")

        if context.active_base_request_index is not None:
            remember_envelope(
                context.active_base_request_index,
                context.active_base_request_envelope_json,
            )
        remember_envelope(
            context.next_base_request_index,
            context.next_base_replay_envelope_json,
        )
        previous_context = context
        return True

    def collect_root(trigger: ShadowTrigger) -> None:
        context = active_trigger_context
        if context is None or not episode_context.collect_recovery_roots:
            return
        try:
            artifacts = root_collector(trigger, context, episode_context)
        except Exception:
            counters.root_write_errors += 1
            return
        if artifacts is not None:
            counters.root_count += 1

    def validate_initial(
        package: ProposalPackage,
        observation: Mapping[str, Any],
    ) -> ShadowValidatedProposal:
        if episode_context.initial_epoch_id != 0:
            raise ValueError("Phase 0 initial epoch must be zero")
        if package.proposal.epoch_id != episode_context.initial_epoch_id:
            raise ValueError("proposal epoch mismatch")
        validated = live_validator(package, observation)
        if not isinstance(validated, ShadowValidatedProposal):
            raise ValueError("live validator returned an invalid result")
        certified = validated.certified_episode
        if certified.graph.source_epoch != episode_context.initial_epoch_id:
            raise ValueError("certified graph source epoch mismatch")
        snapshot = validated.snapshot_reader(observation)
        if not isinstance(snapshot, FactSnapshot):
            raise ValueError("live snapshot reader returned an invalid record")
        if snapshot.epoch_id != episode_context.initial_epoch_id:
            raise ValueError("live snapshot epoch mismatch")
        return validated

    def observe(context: ShadowStepContext) -> None:
        nonlocal active_trigger_context, disabled, topology_auditor, settling_auditor
        if disabled:
            return
        # If Task 2 could not prepare the step-zero callback, that protocol
        # failure is its sole owner.  A later callback must not manufacture a
        # second provenance error for the missing initial observation.
        if previous_context is None and context.policy_step != 0:
            disabled = True
            return
        try:
            should_process = validate_step(context)
        except Exception:
            disable_for_provenance()
            return
        if not should_process:
            return

        if runtime.initial_proposal is None:
            result = run_initial_proposal(
                provider,
                provider_name=provider_name,
                task_id=episode_context.task_id,
                epoch_id=episode_context.initial_epoch_id,
                goal_mode=goal_mode,
                validator=lambda package: validate_initial(
                    package, context.observation
                ),
            )
            runtime.initial_proposal = result
            if (
                result.status is InitialProposalStatus.REJECTED
                or result.validation is None
            ):
                disabled = True
                return
            try:
                certified = result.validation.certified_episode
                plan_context = ShadowPlanContext(
                    problem=certified.problem,
                    plan=certified.plan,
                    graph=certified.graph,
                    certificate_hash=certified.certificate.certificate_hash,
                )
                graph_tracker = ShadowGraphTracker(
                    certified.graph, certified.problem
                )
                def record_state(
                    policy_step: int,
                    snapshot: FactSnapshot,
                    reconciliation: Any,
                    *,
                    phase: str = "POLICY",
                    settling_step: int | None = None,
                ) -> None:
                    try:
                        runtime.state_trace.append(
                            graph_tracker.project(
                                snapshot,
                                policy_step=policy_step,
                                observation_generation=reconciliation.observation_generation,
                                certificate_state=reconciliation.certificate_state.value,
                                phase=phase,
                                settling_step=settling_step,
                            )
                        )
                    except Exception:
                        counters.trace_errors += 1

                if topology_only:
                    reconciler = ShadowCertificateReconciler(plan_context)
                    previous_snapshot: list[FactSnapshot | None] = [None]

                    def audit_observation(
                        observation: Mapping[str, Any],
                        *,
                        policy_step: int,
                        phase: str,
                        settling_step: int | None = None,
                    ) -> None:
                        snapshot = result.validation.snapshot_reader(observation)
                        reconciliation = (
                            reconciler.initial(snapshot)
                            if previous_snapshot[0] is None
                            else reconciler.reconcile(previous_snapshot[0], snapshot)
                        )
                        previous_snapshot[0] = snapshot
                        record_state(
                            policy_step,
                            snapshot,
                            reconciliation,
                            phase=phase,
                            settling_step=settling_step,
                        )

                    def audit_topology(trace_context: ShadowStepContext) -> None:
                        if trace_context.policy_step % interval_steps:
                            return
                        try:
                            audit_observation(
                                trace_context.observation,
                                policy_step=trace_context.policy_step,
                                phase="POLICY",
                            )
                        except Exception:
                            counters.trace_errors += 1

                    def audit_settling(
                        settling_context: ShadowSettlingContext,
                    ) -> None:
                        try:
                            audit_observation(
                                settling_context.observation,
                                policy_step=settling_context.policy_step,
                                phase="SETTLING",
                                settling_step=settling_context.settling_step,
                            )
                        except Exception:
                            counters.trace_errors += 1

                    topology_auditor = audit_topology
                    settling_auditor = audit_settling
                else:
                    if monitor_contract is None:
                        raise ValueError("shadow monitor contract is required")
                    tracker = VersionedActionEventTracker(
                        monitor_contract,
                        episode_context.transition_feature_reader,
                    )
                    runtime.monitor = StableRecoveryObserver(
                        plan_context=plan_context,
                        monitor_contract=monitor_contract,
                        snapshot_reader=result.validation.snapshot_reader,
                        action_event_tracker=tracker,
                        on_trigger=collect_root,
                        interval_steps=interval_steps,
                        confirmation_count=confirmation_count,
                        on_snapshot=lambda context, snapshot, reconciliation: record_state(
                            context.policy_step, snapshot, reconciliation
                        ),
                    )
            except Exception:
                counters.proposal_callback_errors += 1
                disabled = True
                return

        if runtime.monitor is not None:
            active_trigger_context = context
            try:
                runtime.monitor(context)
            finally:
                active_trigger_context = None
        elif topology_auditor is not None:
            topology_auditor(context)

    runtime.observer = observe

    def observe_settling(context: ShadowSettlingContext) -> None:
        if disabled or settling_auditor is None:
            return
        settling_auditor(context)

    runtime.settling_observer = observe_settling if topology_only else None
    return runtime
