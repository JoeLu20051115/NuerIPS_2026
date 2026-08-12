#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np

from pi05_libero_repro.logiv.controller import (
    AttemptReceiptStatus,
    ControllerResult,
    ControllerStatus,
    EvaluatorStatus,
    GroundingStatus,
    LogivController,
    RuntimeBudgetLimits,
    RuntimeBudgetUsage,
)
from pi05_libero_repro.logiv.configuration import resolved_json_sha256
from pi05_libero_repro.logiv.dag import CausalDagCompiler, CausalGraph, SchemaOnlyCausalDagCompiler
from pi05_libero_repro.logiv.domain import render_domain_pddl, validate_state
from pi05_libero_repro.logiv.evaluation import (
    EvaluationContract,
    GlobalRepairOperator,
    MethodArm,
    NativeLiberoTaskEvaluator,
    certify_initial_package,
    run_schema_only_graph,
    run_stage_only,
)
from pi05_libero_repro.logiv.gpt4o import Gpt4oClient
from pi05_libero_repro.logiv.gpt4o_grounding import Gpt4oGrounder
from pi05_libero_repro.logiv.libero_adapter import (
    build_libero_transition_feature_reader,
    LiberoObservationStore,
    LiberoOracleGrounder,
    Pi05MacroExecutor,
    SimulatorSafetySupervisor,
    TaskBinding,
    monitored_fact_universe,
)
from pi05_libero_repro.logiv.model import (
    CandidateSubtask,
    ContextEnvelope,
    ContextPhase,
    FactSnapshot,
    GoalMode,
    parse_pddl_fact,
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.logiv.proposal import DEFAULT_FIXTURE, ScriptedProposalProvider
from pi05_libero_repro.logiv.records import (
    EventJournal,
    LogivEpisodeRecord,
    append_episode_record,
    load_episode_records,
)
from pi05_libero_repro.logiv.recovery_records import (
    CollectionLabel,
    RecoveryRootArtifacts,
    RecoverySplit,
    make_recovery_root_manifest,
    observation_sha256,
    write_recovery_root,
)
from pi05_libero_repro.logiv.repair import CausalSlice, RepairBounds, RetryPolicy
from pi05_libero_repro.logiv.shadow_monitor import (
    MonitorEvidenceContract,
    ShadowTrigger,
    load_monitor_evidence_contract,
)
from pi05_libero_repro.logiv.shadow_runtime import (
    ShadowEpisodeContext,
    ShadowRuntime,
    ShadowValidatedProposal,
    build_shadow_runtime,
)
from pi05_libero_repro.logiv.task5_terminal_recovery import (
    Task5RecoveryCapability,
    TerminalAssessment,
    assess_task5_terminal,
    load_task5_recovery_capability,
)
from pi05_libero_repro.logiv.val import ValWrapper
from pi05_libero_repro.protocol import (
    BaseActionPrefixHasher,
    LIBERO_DUMMY_ACTION,
    derive_episode_seed,
    EpisodeSeededClient,
    prepare_observation,
    run_episode,
    seed_episode_runtime,
    ShadowStepContext,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COVERAGE_MANIFEST = REPOSITORY_ROOT / "configs/logiv/libero10-coverage.json"
DOMAIN_PATH = REPOSITORY_ROOT / "configs/logiv/logiv-libero-domain.pddl"
TASK5_RECOVERY_CAPABILITY = (
    REPOSITORY_ROOT / "configs/logiv/task5-terminal-pi-recover-v1.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _domain_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(
        domain + b"\0" + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _base_physical_attempts(steps: int) -> int:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    return int(steps > 0)


def _remaining_online_action_budget(total_steps: int, prefix_steps: int) -> int:
    if total_steps <= 0 or prefix_steps < 0:
        raise ValueError("total_steps must be positive and prefix_steps nonnegative")
    if prefix_steps > total_steps:
        raise ValueError("Base prefix exceeds total action budget")
    return total_steps - prefix_steps


def _effective_evaluator_status(
    result: ControllerResult, evaluator: NativeLiberoTaskEvaluator | None
) -> str:
    if result.status is ControllerStatus.EPISODE_SUCCESS:
        return EvaluatorStatus.EPISODE_SUCCESS.value
    if evaluator is None or evaluator.last_status is None:
        return "NOT_CALLED"
    return evaluator.last_status.value


def _replace_episode_environment(
    current: Any | None,
    *,
    factory: Any,
    bddl_file: Path | str,
) -> Any:
    """Create an episode-local simulator so hidden RNG state cannot cross episodes."""
    if current is not None:
        current.close()
    return factory(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _allocate_episode_artifact_dir(
    output_dir: Path, *, task_id: int, episode_idx: int
) -> Path:
    """Preserve an interrupted episode directory and allocate a retry generation."""

    task_dir = output_dir / "artifacts" / f"task_{task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"episode_{episode_idx:03d}"
    generation = 0
    while True:
        name = base_name if generation == 0 else f"{base_name}_resume_{generation:03d}"
        candidate = task_dir / name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            generation += 1
            continue
        if generation:
            predecessor = (
                base_name
                if generation == 1
                else f"{base_name}_resume_{generation - 1:03d}"
            )
            _write_json(
                candidate / "resume.json",
                {
                    "reason": "orphan artifacts without an episode record",
                    "predecessor": predecessor,
                    "resume_generation": generation,
                },
            )
        return candidate


def _parse_ids(value: str, *, maximum: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(maximum))
    result: list[int] = []
    for component in value.split(","):
        if ":" in component:
            parts = component.split(":")
            if len(parts) != 2:
                raise ValueError(f"invalid range: {component}")
            result.extend(range(int(parts[0]), int(parts[1])))
        else:
            result.append(int(component))
    if not result or len(result) != len(set(result)) or any(item < 0 or item >= maximum for item in result):
        raise ValueError(f"invalid IDs: {value}")
    return tuple(result)


def _place_effect_stabilization_steps_for_task(
    args: argparse.Namespace, *, task_id: int
) -> int:
    if task_id not in args.place_effect_stabilization_task_ids:
        return 0
    return args.place_effect_stabilization_steps


def _place_effect_confirmation_steps_for_task(
    args: argparse.Namespace, *, task_id: int
) -> int | None:
    if args.place_effect_confirmation_steps <= 0:
        return None
    if task_id not in args.place_effect_confirmation_task_ids:
        return None
    return args.place_effect_confirmation_steps


def _post_stop_reobservation_steps_for_task(
    args: argparse.Namespace, *, task_id: int
) -> int:
    if task_id not in args.post_stop_grounding_reobservation_task_ids:
        return 0
    return args.post_stop_grounding_reobservation_steps


def _initial_context(package, episode_id: str) -> ContextEnvelope:
    return ContextEnvelope(
        phase=ContextPhase.INITIAL_FACTS,
        goal_mode=package.proposal.goal_mode,
        request_id=f"{episode_id}-initial-facts",
        request_generation=0,
        episode_id=episode_id,
        goal_id=package.frozen_goal.goal_id,
        goal_epoch=package.frozen_goal.goal_epoch,
        epoch_id=package.proposal.epoch_id,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )


def _candidate_sidecar(package) -> bytes:
    return json.dumps(
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
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ground_initial(
    package,
    grounder: LiberoOracleGrounder,
    context: ContextEnvelope,
    *,
    require_proposal_consistency: bool = True,
) -> FactSnapshot:
    required = monitored_fact_universe(package.problem)
    response = grounder.ground(ContextPhase.INITIAL_FACTS, context, required)
    if response.status is not GroundingStatus.OK or response.snapshot is None:
        raise RuntimeError(f"STATE_GROUNDING_FAILURE:{response.reason}")
    if require_proposal_consistency:
        proposal = package.proposal.initial_snapshot
        if not proposal.true_facts <= response.snapshot.true_facts:
            raise RuntimeError("STATE_GROUNDING_FAILURE:proposal TRUE conflict")
        if not proposal.false_facts <= response.snapshot.false_facts:
            raise RuntimeError("STATE_GROUNDING_FAILURE:proposal FALSE conflict")
    return response.snapshot

def _rebase_package_on_snapshot(package, snapshot: FactSnapshot):
    """Use the observed Base handoff state as the recovery problem's initial state."""

    return replace(
        package,
        proposal=replace(
            package.proposal,
            epoch_id=snapshot.epoch_id,
            initial_snapshot=snapshot,
        ),
        problem=replace(
            package.problem,
            initial_state=snapshot.true_facts,
            initial_false=snapshot.false_facts,
        ),
    )


def _online_recovery_inputs(
    runtime: ShadowRuntime,
) -> tuple[Any, CausalSlice]:
    proposal_result = runtime.initial_proposal
    request = runtime.online_repair_request
    if (
        proposal_result is None
        or proposal_result.package is None
        or proposal_result.validation is None
        or request is None
    ):
        raise RuntimeError("online recovery inputs are incomplete")
    certified = proposal_result.validation.certified_episode
    graph = certified.graph
    latest = runtime.state_trace[-1] if runtime.state_trace else {}
    statuses = {
        item.get("node_id"): item.get("status")
        for item in latest.get("nodes", ())
        if isinstance(item, Mapping)
    }
    remaining_ids = tuple(
        node_id
        for node_id in graph.canonical_agenda
        if statuses.get(node_id) != "COMPLETED"
    )
    if not remaining_ids:
        remaining_ids = graph.canonical_agenda
    node_map = graph.node_map
    candidates = tuple(
        CandidateSubtask(
            occurrence_id=node_id,
            rough_rank=rank,
            action=node_map[node_id].action,
            instruction=node_map[node_id].instruction or node_map[node_id].action.pddl(),
            evidence_source="pddl-planner-certified-dag-remainder",
            lineage_root=node_map[node_id].lineage_root or f"online:{node_id}",
        )
        for rank, node_id in enumerate(remaining_ids)
        if node_map[node_id].action is not None
    )
    if not candidates:
        raise RuntimeError("online recovery has no remaining action nodes")
    package = replace(
        proposal_result.package,
        proposal=replace(
            proposal_result.package.proposal,
            candidate_subtasks=candidates,
        ),
    )
    requested_ids = tuple(
        node_id for node_id in request.signature if node_id in remaining_ids
    )
    affected_ids = requested_ids or tuple(item.occurrence_id for item in candidates)
    causal_slice = CausalSlice(
        node_ids=frozenset(affected_ids),
        canonical_seed=affected_ids,
        action_signatures=tuple(
            node_map[node_id].action.retry_key
            for node_id in affected_ids
            if node_map[node_id].action is not None
        ),
    )
    return package, causal_slice


def _verified_recovery_surface_facts(snapshot: FactSnapshot) -> tuple:
    return tuple(
        sorted(
            fact
            for fact in snapshot.true_facts
            if fact.predicate == "at"
            and len(fact.arguments) == 2
            and fact.arguments[1].endswith("recovery_surface")
        )
    )


class _RecoverySurfaceMonitor:
    """Fail-open, read-only trigger for an observed harmful object drop."""

    def __init__(
        self,
        *,
        env: Any,
        binding: TaskBinding,
        required_facts,
        interval_steps: int,
    ) -> None:
        if interval_steps <= 0:
            raise ValueError("overlay monitor interval must be positive")
        self.env = env
        self.binding = binding
        self.required_facts = required_facts
        self.interval_steps = interval_steps
        self.store: LiberoObservationStore | None = None
        self.grounder: LiberoOracleGrounder | None = None
        self.trigger_step: int | None = None
        self.trigger_facts: tuple = ()
        self.grounding_errors: list[str] = []

    def __call__(self, observation: dict, _action: np.ndarray, step: int) -> bool:
        if self.store is None:
            self.store = LiberoObservationStore(observation, epoch_id=0)
            self.grounder = LiberoOracleGrounder(
                self.env,
                self.store,
                self.binding,
                self.required_facts,
            )
        else:
            self.store.update(observation)
        if step % self.interval_steps:
            return False
        assert self.grounder is not None
        try:
            snapshot = self.grounder.peek_snapshot()
        except Exception as error:
            self.grounding_errors.append(f"step-{step}:{type(error).__name__}:{error}")
            return False
        recovery_facts = _verified_recovery_surface_facts(snapshot)
        if not recovery_facts:
            return False
        self.trigger_step = step
        self.trigger_facts = recovery_facts
        return True


def _graph_json(
    graph: CausalGraph, *, state_trace: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload = {
        "graph_version": graph.graph_version,
        "graph_hash": graph.graph_hash,
        "certificate_hash": graph.certificate_hash,
        "source_epoch": graph.source_epoch,
        "action_layer_width": graph.action_layer_width(),
        "canonical_agenda": list(graph.canonical_agenda),
        "nodes": [
            {
                "node_id": node.node_id,
                "kind": node.kind.value,
                "canonical_rank": node.canonical_rank,
                "action": node.action.pddl() if node.action else None,
                "lineage_root": node.lineage_root,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "support_literals": sorted(str(item) for item in edge.support_literals),
                "conflict_reasons": sorted(str(item) for item in edge.conflict_reasons),
            }
            for edge in graph.edges
        ],
    }
    if state_trace is not None:
        payload["state_trace"] = state_trace
    return payload


def _write_shadow_graph_artifact(
    artifact_dir: Path, runtime: ShadowRuntime
) -> CausalGraph | None:
    proposal = runtime.initial_proposal
    if (
        proposal is None
        or proposal.status.value != "ACCEPTED"
        or proposal.validation is None
    ):
        return None
    graph = proposal.validation.certified_episode.graph
    try:
        _write_json(
            artifact_dir / "graph.json",
            _graph_json(graph, state_trace=runtime.state_trace),
        )
    except Exception:
        runtime.counters.trace_errors += 1
        return None
    return graph


def _certificate_json(certificate) -> dict[str, Any]:
    payload = asdict(certificate)
    payload["context"] = certificate.context.payload()
    return payload


def _attempt_json(result) -> dict[str, Any]:
    return {
        "attempt_id": result.attempt_id,
        "context": result.context.payload(),
        "action": result.action.pddl(),
        "prompt": result.prompt,
        "prompt_history": list(result.prompt_history),
        "completion_mode": result.completion_mode,
        "recovery_frontier": result.recovery_frontier,
        "completion_occurrence_ids": list(result.completion_occurrence_ids),
        "completion_actions": [action.pddl() for action in result.completion_actions],
        "completion_positive": sorted(
            fact.pddl() for fact in result.completion_positive
        ),
        "completion_negative": sorted(
            fact.pddl() for fact in result.completion_negative
        ),
        "primary_effect_first_step": result.primary_effect_first_step,
        "frontier_followup_limit": result.frontier_followup_limit,
        "frontier_completion_step": result.frontier_completion_step,
        "frontier_completion_prompt": result.frontier_completion_prompt,
        "frontier_fallback_step": result.frontier_fallback_step,
        "frontier_fallback_prompt": result.frontier_fallback_prompt,
        "pre_epoch": result.pre_epoch,
        "post_epoch": result.post_epoch,
        "executor_status": result.executor_status.value,
        "stopped": result.stopped,
        "stop_evidence": result.stop_evidence,
        "actions": [np.asarray(action).tolist() for action in result.actions],
        "inference_requests": result.inference_requests,
        "detector_calls": result.detector_calls,
        "unused_actions_flushed": result.unused_actions_flushed,
        "post_stop_reobservation_steps": result.post_stop_reobservation_steps,
        "post_snapshot_evidence_hash": (
            result.post_snapshot.evidence_hash if result.post_snapshot is not None else None
        ),
        "post_snapshot_error": result.post_snapshot_error,
        "post_snapshot_true": (
            sorted(fact.pddl() for fact in result.post_snapshot.true_facts)
            if result.post_snapshot is not None
            else None
        ),
        "post_snapshot_false": (
            sorted(fact.pddl() for fact in result.post_snapshot.false_facts)
            if result.post_snapshot is not None
            else None
        ),
        "reason": result.reason,
    }


def _record_events(
    journal: EventJournal,
    initial_context: ContextEnvelope,
    result: ControllerResult,
    executor: Pi05MacroExecutor | None,
    graph: CausalGraph | None,
    certificate: Any | None,
    evaluator: NativeLiberoTaskEvaluator | None,
) -> None:
    journal.append("INITIAL_FACTS_ACCEPTED", initial_context, {"epoch": initial_context.epoch_id})
    if certificate is not None:
        journal.append(
            "PLAN_CERTIFIED",
            certificate.context,
            {"certificate_hash": certificate.certificate_hash},
        )
    if graph is not None:
        journal.append(
            "GRAPH_COMPILED",
            replace(
                initial_context,
                graph_version=graph.graph_version,
                certificate_hash=graph.certificate_hash,
            ),
            {
                "graph_hash": graph.graph_hash,
                "action_layer_width": graph.action_layer_width(),
            },
        )
    if executor is not None:
        for attempt in executor.results:
            journal.append(
                "EXECUTION_AUTHORIZED",
                attempt.context,
                {"action": attempt.action.pddl(), "prompt": attempt.prompt},
            )
            journal.append(
                "EXECUTOR_OUTCOME",
                attempt.context,
                {
                    "status": attempt.executor_status.value,
                    "stopped": attempt.stopped,
                    "post_epoch": attempt.post_epoch,
                },
            )
    for event in result.events:
        journal.append("CONTROLLER_EVENT", initial_context, {"event": event})
    journal.append(
        "TERMINAL_RECEIPT",
        initial_context,
        {
            "controller_status": result.status.value,
            "terminal_cause": result.terminal_cause,
            "evaluator_status": _effective_evaluator_status(result, evaluator),
        },
    )


def _reset_episode(env: Any, initial_state: np.ndarray, wait_steps: int) -> dict[str, Any]:
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, _, _ = env.step(list(LIBERO_DUMMY_ACTION))
    return observation


def _execute_symbolic_arm(
    args: argparse.Namespace,
    env: Any,
    client: Any,
    image_tools: Any,
    task_id: int,
    episode_id: str,
    initial_observation: dict[str, Any],
    *,
    recovery_state: bool = False,
    max_total_action_steps: int | None = None,
    max_physical_attempts: int | None = None,
    gpt4o_client: Gpt4oClient | None = None,
    recovery_package: Any | None = None,
    causal_slice: CausalSlice | None = None,
):
    binding = TaskBinding.from_manifest(args.coverage_manifest, task_id)
    scaffold = ScriptedProposalProvider(args.proposal_config)
    if recovery_package is not None:
        package = recovery_package
    else:
        package = scaffold.propose(
            task_id, epoch_id=0, goal_mode=GoalMode(args.goal_mode)
        )
    store = LiberoObservationStore(initial_observation, epoch_id=0)
    if args.perception_backend == "gpt4o":
        if gpt4o_client is None:
            gpt4o_client = Gpt4oClient.from_env()
        grounder = Gpt4oGrounder(
            client=gpt4o_client,
            observation_store=store,
            problem=package.problem,
            monitored_facts=monitored_fact_universe(package.problem),
            task_instruction=package.proposal.task_name,
            graph_version=None,
            image_tools=image_tools,
        )
    else:
        grounder = LiberoOracleGrounder(
            env,
            store,
            binding,
            monitored_fact_universe(package.problem),
        )
    initial_context = _initial_context(package, episode_id)
    initial_snapshot = _ground_initial(
        package,
        grounder,
        initial_context,
        require_proposal_consistency=not recovery_state,
    )
    if recovery_state:
        package = _rebase_package_on_snapshot(package, initial_snapshot)
    prompt_renderer = SubtaskPromptRenderer(args.prompt_config)
    evaluator = NativeLiberoTaskEvaluator()
    arm = MethodArm(args.method_arm)
    executor = Pi05MacroExecutor(
        env=env,
        client=client,
        image_tools=image_tools,
        observation_store=store,
        grounder=grounder,
        prompt_renderer=prompt_renderer,
        safety_supervisor=SimulatorSafetySupervisor(
            watchdog_seconds=args.watchdog_seconds,
            action_limit=args.action_limit,
        ),
        replan_steps=args.replan_steps,
        max_action_steps=args.max_action_steps,
        settling_steps=args.settling_steps,
        post_stop_grounding_reobservation_steps=(
            _post_stop_reobservation_steps_for_task(args, task_id=task_id)
        ),
        effect_confirmation_steps=args.effect_confirmation_steps,
        place_effect_confirmation_steps=(
            _place_effect_confirmation_steps_for_task(args, task_id=task_id)
        ),
        access_effect_stabilization_steps=args.access_effect_stabilization_steps,
        place_effect_stabilization_steps=(
            _place_effect_stabilization_steps_for_task(args, task_id=task_id)
        ),
        target_divergence_confirmation_steps=(
            args.target_divergence_confirmation_steps
        ),
        held_target_divergence_confirmation_steps=(
            args.held_target_divergence_confirmation_steps
        ),
        frontier_followup_steps=args.frontier_followup_steps,
        frontier_completion_followup_steps=(
            args.frontier_completion_followup_steps or None
        ),
        frontier_completion_recovery_only=(
            args.frontier_completion_recovery_only
        ),
        frontier_recovery_max_consumed_steps=(
            args.frontier_recovery_max_consumed_steps or None
        ),
        frontier_fallback_after_steps=(args.frontier_fallback_after_steps or None),
        frontier_fallback_followup_steps=(
            args.frontier_fallback_followup_steps or None
        ),
        stop_on_effects=arm is not MethodArm.STAGE_ONLY,
        max_total_action_steps=(
            args.base_max_steps
            if max_total_action_steps is None
            else max_total_action_steps
        ),
    )
    plan = tuple(item.action for item in package.proposal.candidate_subtasks)
    graph = None
    final_graph = None
    certificate = None
    initial_val_calls = 0

    if arm is MethodArm.STAGE_ONLY:
        result = run_stage_only(
            plan,
            initial_snapshot=initial_snapshot,
            dispatcher=executor,
            evaluator=evaluator,
            evaluator_handle=env,
            base_context=initial_context,
            max_physical_attempts=args.max_physical_attempts,
        )
    elif arm is MethodArm.GRAPH_WITHOUT_VAL:
        problem = replace(
            package.problem,
            initial_state=initial_snapshot.true_facts,
            initial_false=initial_snapshot.false_facts,
        )
        validate_state(problem, problem.initial_state)
        graph_context = replace(
            initial_context,
            phase=ContextPhase.PREINSTALL_VAL,
            request_id=f"{episode_id}-schema-only-compile",
        )
        graph = SchemaOnlyCausalDagCompiler().compile(
            problem,
            plan,
            _candidate_sidecar(package),
            graph_context,
        )
        result = run_schema_only_graph(
            problem,
            graph,
            initial_snapshot=initial_snapshot,
            grounder=grounder,
            dispatcher=executor,
            evaluator=evaluator,
            evaluator_handle=env,
            base_context=initial_context,
            max_physical_attempts=args.max_physical_attempts,
        )
        final_graph = graph
    else:
        val_wrapper = ValWrapper(args.val_binary, timeout_seconds=args.val_timeout)
        policy = RetryPolicy(max_retries_per_lineage=args.max_retries_per_lineage)
        repair_bounds = RepairBounds(
            max_edits=args.max_edits,
            max_candidates=args.max_candidates,
            max_val_calls=args.max_repair_val_calls,
        )
        certified = certify_initial_package(
            package,
            initial_snapshot,
            episode_id=episode_id,
            val_wrapper=val_wrapper,
            allowed_schemas=binding.supported_action_schemas | binding.recovery_schemas,
            repair_bounds=repair_bounds,
            retry_policy=policy,
            decompose_macro_sources=binding.decompose_macro_sources,
            causal_slice=causal_slice,
        )
        if args.perception_backend == "gpt4o":
            grounder.set_graph_version(certified.graph.graph_version)
        graph = certified.graph
        certificate = certified.certificate
        initial_val_calls = certified.initial_val_calls
        compiler = CausalDagCompiler(args.val_binary, timeout_seconds=args.val_timeout)
        repair_operator = (
            GlobalRepairOperator(certified.repair_operator)
            if arm is MethodArm.VAL_WITHOUT_LOCALIZED_REPAIR
            else certified.repair_operator
        )
        controller = LogivController(
            certified.installation,
            grounder=grounder,
            dispatcher=executor,
            evaluator=evaluator,
            evaluator_handle=env,
            val_wrapper=val_wrapper,
            compiler=compiler,
            repair_operator=repair_operator,
            retry_policy=policy,
            budget_limits=RuntimeBudgetLimits(
                max_physical_attempts=(
                    args.max_physical_attempts
                    if max_physical_attempts is None
                    else max_physical_attempts
                ),
                max_repair_rounds=args.max_repair_rounds,
                max_total_val_calls=args.max_total_val_calls,
            ),
            initial_val_calls=initial_val_calls,
        )
        result = controller.run()
        final_graph = controller.graph

    return (
        package,
        initial_context,
        initial_snapshot,
        result,
        executor,
        evaluator,
        graph,
        final_graph,
        certificate,
        initial_val_calls,
    )


def _validate_shadow_options(
    args: argparse.Namespace,
    task_ids: tuple[int, ...],
) -> dict[int, MonitorEvidenceContract]:
    arm = MethodArm(args.method_arm)
    if arm is MethodArm.LOGIV_ONLINE:
        if not args.development_only:
            raise ValueError("LOGIV_ONLINE tuning requires development-only mode")
        if args.collect_recovery_roots:
            raise ValueError("LOGIV_ONLINE does not collect Phase 0 recovery roots")
        if args.capture_task5_terminal_preflight:
            raise ValueError("LOGIV_ONLINE does not run terminal preflight")
        if args.shadow_topology_only:
            raise ValueError("LOGIV_ONLINE selects topology monitoring internally")
        if args.online_monitor_interval_steps <= 0:
            raise ValueError("online monitor interval must be positive")
        return {}
    if args.capture_task5_terminal_preflight:
        if arm is not MethodArm.SHADOW_LOGIV:
            raise ValueError("terminal preflight requires SHADOW_LOGIV")
        if not args.development_only:
            raise ValueError("terminal preflight requires development-only mode")
        if task_ids != (5,):
            raise ValueError("terminal preflight requires exactly Task 5")
        if args.collect_recovery_roots:
            raise ValueError("terminal preflight cannot collect recovery roots")
        if (
            getattr(args, "overlay_monitor_recovery_surface", False)
            or args.shadow_topology_only
        ):
            raise ValueError("terminal preflight cannot enable recovery or topology-only mode")
    if args.shadow_topology_only and arm is not MethodArm.SHADOW_LOGIV:
        raise ValueError("shadow topology-only mode requires SHADOW_LOGIV")
    if args.shadow_topology_only and args.shadow_monitor_interval_steps <= 0:
        raise ValueError("shadow topology-only interval must be positive")
    if args.shadow_topology_only and args.collect_recovery_roots:
        raise ValueError("shadow topology-only mode cannot collect recovery roots")
    if args.collect_recovery_roots:
        if arm is not MethodArm.SHADOW_LOGIV:
            raise ValueError("Phase 0 recovery collection requires SHADOW_LOGIV")
        if not args.development_only:
            raise ValueError("Phase 0 recovery collection requires development-only mode")
        if args.recovery_root_split != RecoverySplit.DEV.value:
            raise ValueError("Phase 0 recovery collection is restricted to DEV")
    if arm is not MethodArm.SHADOW_LOGIV or args.shadow_topology_only:
        return {}
    contracts = {
        task_id: load_monitor_evidence_contract(
            args.shadow_monitor_contract,
            task_id=task_id,
        )
        for task_id in task_ids
    }
    for contract in contracts.values():
        if contract.monitor_interval_steps != args.shadow_monitor_interval_steps:
            raise ValueError("shadow monitor interval does not match frozen contract")
        if contract.confirmation_count != args.shadow_confirmations:
            raise ValueError("shadow confirmations do not match frozen contract")
    return contracts


def _gpt4o_request_accounting(client: Any | None) -> dict[str, int]:
    counts = client.request_counts if client is not None else {}
    unexpected = set(counts) - {"state_gate"}
    if unexpected:
        raise ValueError(
            "LOGIV recorded non-State-Gate GPT-4o purposes: "
            + ", ".join(sorted(unexpected))
        )
    return {
        "shadow_vlm_requests": int(counts.get("state_gate", 0)),
        "recovery_policy_requests": 0,
    }


def _symbolic_record_accounting(
    *, base_policy_requests: int, gpt4o_client: Any | None
) -> dict[str, Any]:
    gpt_accounting = _gpt4o_request_accounting(gpt4o_client)
    return {
        "base_policy_requests": base_policy_requests,
        "initial_proposal_requests": 0,
        "initial_proposal_status": "NOT_APPLICABLE",
        "initial_proposal_reason_code": None,
        "shadow_vlm_requests": gpt_accounting["shadow_vlm_requests"],
        "recovery_policy_requests": gpt_accounting[
            "recovery_policy_requests"
        ],
        "shadow_monitor_calls": 0,
        "shadow_monitor_errors": 0,
        "shadow_monitor_seconds": 0.0,
        "shadow_parity_valid": True,
    }


def _shadow_artifact_payloads(
    outcome: Any,
    runtime: ShadowRuntime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    proposal = runtime.initial_proposal
    if proposal is None:
        failure = next(
            (
                item
                for item in outcome.shadow_failure_records
                if item.policy_step == 0
            ),
            None,
        )
        reason = (
            f"{failure.stage}:{failure.reason}"
            if failure is not None
            else "RUNTIME:NO_PROPOSAL_RESULT"
        )
        status = "NOT_ATTEMPTED"
        reason_code = reason
        provider = None
        request_count = 0
        elapsed_seconds = 0.0
        certificate_hash = None
        graph_hash = None
    else:
        status = proposal.status.value
        reason = proposal.reason
        reason_code = (
            proposal.reason.split(":", 1)[0]
            if status == "REJECTED" and proposal.reason
            else None
        )
        provider = proposal.provider
        request_count = proposal.request_count
        elapsed_seconds = proposal.elapsed_seconds
        validation = proposal.validation
        certificate_hash = (
            validation.certified_episode.certificate.certificate_hash
            if validation is not None
            else None
        )
        graph_hash = (
            validation.certified_episode.graph.graph_hash
            if validation is not None
            else None
        )

    metrics = runtime.monitor.metrics if runtime.monitor is not None else None
    action_event_tracker = (
        runtime.monitor.action_event_tracker
        if runtime.monitor is not None
        else None
    )

    def metric(name: str) -> int:
        return int(getattr(metrics, name, 0))

    def tracker_metric(name: str) -> int:
        return int(getattr(action_event_tracker, name, 0))

    terminal_topology_status = None
    if runtime.state_trace:
        terminal_topology_status = next(
            (
                node.get("status")
                for node in runtime.state_trace[-1].get("nodes", ())
                if node.get("node_id") == "GOAL"
            ),
            None,
        )
    terminal_topology_success = (
        terminal_topology_status == "COMPLETED"
        if terminal_topology_status is not None
        else None
    )

    aggregate_errors = sum(
        (
            int(outcome.shadow_errors),
            runtime.counters.proposal_callback_errors,
            runtime.counters.provenance_errors,
            runtime.counters.trace_errors,
            metric("snapshot_errors"),
            metric("event_tracker_errors"),
            metric("evidence_overflows"),
            metric("trigger_callback_errors"),
            runtime.counters.root_write_errors,
        )
    )
    monitor_seconds = max(
        0.0,
        float(outcome.shadow_wall_seconds) - float(elapsed_seconds),
    )
    proposal_payload = {
        "status": status,
        "provider": provider,
        "request_count": request_count,
        "elapsed_seconds": elapsed_seconds,
        "reason": reason,
        "certificate_hash": certificate_hash,
        "graph_hash": graph_hash,
    }
    monitor_payload = {
        "callback_calls": outcome.shadow_calls,
        "callback_errors": outcome.shadow_errors,
        "callback_seconds": outcome.shadow_wall_seconds,
        "shadow_parity_valid": outcome.shadow_parity_valid,
        "failure_records": [asdict(item) for item in outcome.shadow_failure_records],
        "proposal_callback_errors": runtime.counters.proposal_callback_errors,
        "provenance_errors": runtime.counters.provenance_errors,
        "trace_errors": runtime.counters.trace_errors,
        "snapshot_calls": metric("snapshot_calls"),
        "snapshot_errors": metric("snapshot_errors"),
        "event_tracker_errors": metric("event_tracker_errors"),
        "evidence_overflows": metric("evidence_overflows"),
        "trigger_callback_errors": metric("trigger_callback_errors"),
        "root_write_errors": runtime.counters.root_write_errors,
        "anomaly_candidates": metric("anomaly_candidates"),
        "confirmed_deviations": metric("confirmed_deviations"),
        "stale_certificates": metric("stale_certificates"),
        "attempt_records": tracker_metric("attempt_record_count"),
        "evidence_records": tracker_metric("evidence_record_count"),
        "terminal_topology_status": terminal_topology_status,
        "terminal_topology_success": terminal_topology_success,
        "base_self_recovered_after_confirmed_deviation": bool(
            metric("confirmed_deviations") and outcome.check_success
        ),
        "deviation_decisions": runtime.deviation_trace,
        "root_count": runtime.counters.root_count,
        "aggregate_errors": aggregate_errors,
    }
    gpt_accounting = _gpt4o_request_accounting(runtime.gpt4o_client)
    shadow_vlm_requests = gpt_accounting["shadow_vlm_requests"]
    recovery_policy_requests = gpt_accounting["recovery_policy_requests"]
    compute_payload = {
        "base_policy_requests": outcome.inference_requests,
        "initial_proposal_requests": request_count,
        "shadow_vlm_requests": shadow_vlm_requests,
        "recovery_policy_requests": recovery_policy_requests,
        "initial_proposal_seconds": elapsed_seconds,
        "shadow_monitor_seconds": monitor_seconds,
    }
    record_accounting = {
        "base_policy_requests": outcome.inference_requests,
        "initial_proposal_requests": request_count,
        "initial_proposal_status": status,
        "initial_proposal_reason_code": reason_code,
        "shadow_vlm_requests": shadow_vlm_requests,
        "recovery_policy_requests": recovery_policy_requests,
        "shadow_monitor_calls": outcome.shadow_calls,
        "shadow_monitor_errors": aggregate_errors,
        "shadow_monitor_seconds": monitor_seconds,
        "shadow_parity_valid": outcome.shadow_parity_valid,
    }
    return proposal_payload, monitor_payload, compute_payload, record_accounting


def _shadow_record_accounting(
    *,
    outcome: Any | None,
    runtime: ShadowRuntime | None,
    exception_text: str | None,
    base_policy_requests: int,
) -> dict[str, Any]:
    if outcome is not None and runtime is not None:
        return _shadow_artifact_payloads(outcome, runtime)[3]

    proposal = runtime.initial_proposal if runtime is not None else None
    if proposal is None:
        exception_code = (
            exception_text.split(":", 1)[0] if exception_text else "RuntimeError"
        )
        status = "NOT_ATTEMPTED"
        request_count = 0
        reason_code = f"ROLLOUT:{exception_code}"
    else:
        status = proposal.status.value
        request_count = proposal.request_count
        reason_code = (
            proposal.reason.split(":", 1)[0] if proposal.reason else None
        )
    gpt_accounting = _gpt4o_request_accounting(
        runtime.gpt4o_client if runtime is not None else None
    )
    return {
        "base_policy_requests": base_policy_requests,
        "initial_proposal_requests": request_count,
        "initial_proposal_status": status,
        "initial_proposal_reason_code": reason_code,
        "shadow_vlm_requests": gpt_accounting["shadow_vlm_requests"],
        "recovery_policy_requests": gpt_accounting[
            "recovery_policy_requests"
        ],
        "shadow_monitor_calls": 0,
        "shadow_monitor_errors": (
            runtime.counters.trace_errors if runtime is not None else 0
        ),
        "shadow_monitor_seconds": 0.0,
        "shadow_parity_valid": True,
    }


def _base_execution_payload(
    outcome: Any,
    episode_client: EpisodeSeededClient,
    *,
    initial_state_sha256: str,
    base_prompt_sha256: str,
    base_checkpoint_sha256: str,
    policy_client_config_sha256: str,
) -> dict[str, Any]:
    action_prefix = BaseActionPrefixHasher()
    actions_sha256 = action_prefix.update_and_hexdigest(None)
    for action in outcome.actions:
        actions_sha256 = action_prefix.update_and_hexdigest(action)
    request_envelope_log_sha256 = _domain_sha256(
        b"LOGIV_BASE_REQUEST_ENVELOPE_LOG_V1",
        list(episode_client.issued_request_envelopes),
    )
    return {
        "steps": outcome.steps,
        "inference_requests": outcome.inference_requests,
        "base_policy_requests": outcome.inference_requests,
        "done_signal": outcome.done,
        "post_settling_success": outcome.check_success,
        "intervention_requested": getattr(outcome, "intervention_requested", False),
        "discarded_pending_actions": getattr(
            outcome, "discarded_pending_actions", 0
        ),
        "initial_state_sha256": initial_state_sha256,
        "base_prompt_sha256": base_prompt_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "policy_client_config_sha256": policy_client_config_sha256,
        "request_envelope_log_sha256": request_envelope_log_sha256,
        "actions_sha256": actions_sha256,
        "actions_prefix_520_sha256": _sha256_array(
            np.asarray(outcome.actions[:520], dtype=np.float64)
        ),
    }


def _read_simulator_state(env: Any) -> np.ndarray:
    reader = getattr(env, "get_sim_state", None)
    if callable(reader):
        return np.asarray(reader())
    current = env
    seen: set[int] = set()
    while not hasattr(current, "sim") and hasattr(current, "env"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.env
    state = current.sim.get_state()
    if hasattr(state, "flatten"):
        state = state.flatten()
    return np.asarray(state)


def _shadow_snapshot_peek(grounder: LiberoOracleGrounder) -> FactSnapshot:
    return grounder.peek_advisory_partial_snapshot()


def _policy_time_snapshot_peek(
    grounder: LiberoOracleGrounder,
) -> FactSnapshot:
    return grounder.peek_advisory_partial_snapshot()


def _strict_terminal_snapshot_reader(
    store: LiberoObservationStore,
    grounder: LiberoOracleGrounder,
    observation: Mapping[str, Any],
) -> FactSnapshot:
    store.update(observation)
    strict_reader = getattr(grounder, "read_strict_snapshot", None)
    return strict_reader() if strict_reader is not None else grounder.peek_snapshot()


def _terminal_assessment_payload(
    assessment: TerminalAssessment,
) -> dict[str, Any]:
    return {
        "eligible": assessment.eligible,
        "reason": assessment.reason,
        "event_id": assessment.event_id,
        "event_type": assessment.event_type,
        "option_action_cap": assessment.option_action_cap,
        "snapshot_sha256": assessment.snapshot_sha256,
        "graph_hash": assessment.graph_hash,
        "certificate_hash": assessment.certificate_hash,
        "monitor_contract_sha256": assessment.monitor_contract_sha256,
        "protected_true_facts": sorted(
            fact.pddl() for fact in assessment.protected_true_facts
        ),
        "capability_sha256": assessment.capability_sha256,
        "base_policy_steps": assessment.base_policy_steps,
        "place_node_id": assessment.place_node_id,
        "assessment_sha256": assessment.assessment_sha256,
    }


def _capture_task5_terminal_preflight(
    *,
    artifact_dir: Path,
    case_id: str,
    task_id: int,
    episode_idx: int,
    episode_id: str,
    outcome: Any,
    native_terminal_status: str,
    runtime: ShadowRuntime,
    monitor_contract: MonitorEvidenceContract,
    capability: Task5RecoveryCapability,
) -> None:
    proposal = runtime.initial_proposal
    if proposal is None or proposal.validation is None:
        raise ValueError("terminal preflight requires an accepted Shadow proposal")
    validation = proposal.validation
    certified = validation.certified_episode
    terminal_state = runtime.state_trace[-1] if runtime.state_trace else {}
    node_statuses = {
        item.get("node_id"): item.get("status")
        for item in terminal_state.get("nodes", ())
    }
    place_nodes = tuple(
        node
        for node in certified.graph.nodes
        if node.action is not None and node.action.pddl() == capability.action
    )
    place_node_action = place_nodes[0].action if len(place_nodes) == 1 else None
    place_node_status = (
        node_statuses.get(place_nodes[0].node_id) if len(place_nodes) == 1 else None
    )
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    strict_snapshot = None
    grounding_error = None
    try:
        strict_snapshot = validation.strict_terminal_snapshot_reader(
            outcome.final_observation
        )
    except Exception as error:
        grounding_error = f"{type(error).__name__}: {error}"

    observed_base_policy_steps = outcome.steps
    observed_base_policy_requests = outcome.inference_requests
    assessment = assess_task5_terminal(
        capability,
        task_id=task_id,
        episode_id=episode_id,
        base_success=outcome.check_success,
        native_terminal_status=native_terminal_status,
        base_policy_steps=observed_base_policy_steps,
        snapshot=strict_snapshot,
        problem=certified.problem,
        graph=certified.graph,
        certificate=certified.certificate,
        monitor_contract_sha256=monitor_contract.contract_sha256,
        certificate_state=terminal_state.get("certificate_state"),
        place_node_action=place_node_action,
        place_node_status=place_node_status,
    )
    original_assessment_sha256 = assessment.assessment_sha256
    common = {
        "schema_version": 1,
        "case_id": case_id,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "episode_id": episode_id,
        "captured_at_utc": captured_at_utc,
        "graph_hash": certified.graph.graph_hash,
        "certificate_hash": certified.certificate.certificate_hash,
        "monitor_contract_sha256": monitor_contract.contract_sha256,
        "capability_sha256": capability.capability_sha256,
    }
    if strict_snapshot is None:
        snapshot_payload = {
            **common,
            "status": "GROUNDING_ERROR",
            "grounding_error": grounding_error,
        }
    else:
        snapshot_payload = {
            **common,
            "status": "STRICT_AUDITED",
            "epoch_id": strict_snapshot.epoch_id,
            "true": sorted(fact.pddl() for fact in strict_snapshot.true_facts),
            "false": sorted(fact.pddl() for fact in strict_snapshot.false_facts),
            "fact_universe": sorted(
                fact.pddl() for fact in strict_snapshot.fact_universe or ()
            ),
            "fact_universe_version": strict_snapshot.fact_universe_version,
            "fact_universe_sha256": strict_snapshot.fact_universe_sha256,
            "evidence_hash": strict_snapshot.evidence_hash,
            "evidence_payload_json": strict_snapshot.evidence_payload_json,
        }
    terminal_status = (
        "GROUNDING_ERROR"
        if grounding_error is not None
        else "ELIGIBLE"
        if assessment.eligible
        else "DENIED"
    )
    terminal_payload = {
        **common,
        "status": terminal_status,
        "reason": "GROUNDING_ERROR" if grounding_error is not None else assessment.reason,
        "grounding_error": grounding_error,
        "base_success": outcome.check_success,
        "native_terminal_status": native_terminal_status,
        "certificate_state": terminal_state.get("certificate_state"),
        "place_node_action": (
            place_node_action.pddl() if place_node_action is not None else None
        ),
        "place_node_status": place_node_status,
        "base_policy_steps": observed_base_policy_steps,
        "base_policy_requests": observed_base_policy_requests,
        "recovery_actions": 0,
        "recovery_policy_requests": 0,
        "assessment_sha256": original_assessment_sha256,
        "assessment": _terminal_assessment_payload(assessment),
    }
    _write_json(artifact_dir / "current_snapshot.json", snapshot_payload)
    _write_json(artifact_dir / "terminal_deviation.json", terminal_payload)


def _online_detector_settings_for_task(
    args: argparse.Namespace, *, task_id: int
) -> dict[str, int | bool | None]:
    return {
        "confirmation_count": args.online_confirmations,
        "recovery_surface_confirmation_count": (
            args.online_recovery_surface_confirmations
            if task_id in args.online_recovery_surface_confirmation_task_ids
            else None
        ),
        "min_intervention_step": args.online_min_intervention_step,
        "stall_steps": args.online_stall_steps,
        "recovery_requires_achieved_goal": (
            task_id in args.online_recovery_requires_goal_task_ids
        ),
        "stall_requires_achieved_goal": (
            task_id in args.online_stall_requires_goal_task_ids
        ),
        "stall_ignores_holding": (
            task_id in args.online_stall_ignores_holding_task_ids
        ),
        "stall_requires_handempty": (
            task_id in args.online_stall_requires_handempty_task_ids
        ),
    }


def _build_evaluator_shadow_runtime(
    args: argparse.Namespace,
    *,
    env: Any,
    task_id: int,
    episode_idx: int,
    episode_id: str,
    initial_state_sha256: str,
    scene_sha256: str,
    base_prompt_sha256: str,
    base_checkpoint_sha256: str,
    policy_client_config_sha256: str,
    policy_seed: int,
    simulator_seed: int,
    artifact_dir: Path,
    monitor_contract: MonitorEvidenceContract | None,
    image_tools: Any | None = None,
) -> ShadowRuntime:
    binding = TaskBinding.from_manifest(args.coverage_manifest, task_id)
    scaffold = ScriptedProposalProvider(args.proposal_config)
    gpt4o_client = (
        Gpt4oClient.from_env() if args.perception_backend == "gpt4o" else None
    )
    provider = scaffold
    transition_reader = (
        build_libero_transition_feature_reader(env, binding, monitor_contract)
        if monitor_contract is not None
        else lambda context: ()
    )
    lineage_sha256 = _domain_sha256(
        b"LOGIV_PARENT_TRAJECTORY_LINEAGE_V1",
        {
            "task_id": task_id,
            "scene_sha256": scene_sha256,
            "initial_state_sha256": initial_state_sha256,
            "base_prompt_sha256": base_prompt_sha256,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "policy_client_config_sha256": policy_client_config_sha256,
            "master_seed": args.seed,
            "policy_seed": policy_seed,
            "simulator_seed": simulator_seed,
        },
    )
    episode_context = ShadowEpisodeContext(
        task_id=task_id,
        episode_idx=episode_idx,
        initial_epoch_id=0,
        scene_sha256=scene_sha256,
        object_instance_ids=tuple(sorted(binding.registered_objects)),
        initial_state_sha256=initial_state_sha256,
        parent_trajectory_lineage_sha256=lineage_sha256,
        base_prompt_sha256=base_prompt_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        policy_client_config_sha256=policy_client_config_sha256,
        policy_replay_contract_sha256=None,
        master_seed=args.seed,
        policy_seed=policy_seed,
        simulator_seed=simulator_seed,
        replan_steps=args.replan_steps,
        collect_recovery_roots=args.collect_recovery_roots,
        collection_label=CollectionLabel.DEV_COLLECTION,
        root_output_dir=artifact_dir / "recovery_roots",
        simulator_state_reader=lambda: _read_simulator_state(env),
        transition_feature_reader=transition_reader,
    )

    def live_validator(
        package: Any, observation: Mapping[str, Any]
    ) -> ShadowValidatedProposal:
        registered = tuple(item.name for item in package.proposal.registered_objects)
        if registered != binding.registered_objects:
            raise ValueError("proposal/binding registered-object mismatch")
        registered_set = frozenset(registered)
        if monitor_contract is not None:
            required_contract_ids = set(monitor_contract.object_ids) | set(
                monitor_contract.abnormal_support_surfaces
            )
            for effect in monitor_contract.task_relevant_effects:
                required_contract_ids.update(parse_pddl_fact(effect).arguments)
            if not required_contract_ids <= registered_set:
                raise ValueError("monitor contract references unregistered proposal IDs")

        store = LiberoObservationStore(observation, epoch_id=0)
        if gpt4o_client is not None:
            grounder = Gpt4oGrounder(
                client=gpt4o_client,
                observation_store=store,
                problem=package.problem,
                monitored_facts=monitored_fact_universe(package.problem),
                task_instruction=package.proposal.task_name,
                graph_version=None,
                image_tools=image_tools,
            )
        else:
            grounder = LiberoOracleGrounder(
                env,
                store,
                binding,
                monitored_fact_universe(package.problem),
            )
        initial_context = _initial_context(package, episode_id)
        initial_snapshot = _ground_initial(package, grounder, initial_context)
        val_wrapper = ValWrapper(
            args.val_binary, timeout_seconds=args.val_timeout
        )
        repair_bounds = RepairBounds(
            max_edits=args.max_edits,
            max_candidates=args.max_candidates,
            max_val_calls=args.max_repair_val_calls,
        )
        policy = RetryPolicy(
            max_retries_per_lineage=args.max_retries_per_lineage
        )
        certified = certify_initial_package(
            package,
            initial_snapshot,
            episode_id=episode_id,
            val_wrapper=val_wrapper,
            allowed_schemas=(
                binding.supported_action_schemas | binding.recovery_schemas
            ),
            repair_bounds=repair_bounds,
            retry_policy=policy,
            decompose_macro_sources=binding.decompose_macro_sources,
        )
        if gpt4o_client is not None:
            grounder.set_graph_version(certified.graph.graph_version)
        previous_observation_sha256 = observation_sha256(observation)
        previous_snapshot = initial_snapshot

        def snapshot_reader(
            current_observation: Mapping[str, Any]
        ) -> FactSnapshot:
            nonlocal previous_observation_sha256, previous_snapshot
            current_hash = observation_sha256(current_observation)
            if current_hash == previous_observation_sha256:
                return previous_snapshot
            store.update(current_observation)
            if args.capture_task5_terminal_preflight:
                previous_snapshot = _policy_time_snapshot_peek(grounder)
            else:
                previous_snapshot = _shadow_snapshot_peek(grounder)
            previous_observation_sha256 = current_hash
            return previous_snapshot

        def strict_terminal_snapshot_reader(
            current_observation: Mapping[str, Any],
        ) -> FactSnapshot:
            return _strict_terminal_snapshot_reader(
                store, grounder, current_observation
            )

        return ShadowValidatedProposal(
            certified, snapshot_reader, strict_terminal_snapshot_reader
        )

    event_origins: dict[str, str] = {}

    def collect_root(
        trigger: ShadowTrigger,
        context: ShadowStepContext,
        runtime_context: ShadowEpisodeContext,
    ) -> RecoveryRootArtifacts:
        previous_origin = event_origins.get(trigger.deviation_event_id)
        if (
            previous_origin is not None
            and previous_origin != trigger.event_origin_parent_sha256
        ):
            raise ValueError("deviation event origin changed across captures")
        event_origins[trigger.deviation_event_id] = trigger.event_origin_parent_sha256
        simulator_state = runtime_context.simulator_state_reader()
        observation = {
            key: np.array(value, copy=True)
            for key, value in trigger.observation.items()
        }
        pending_actions = np.array(context.pending_base_actions, copy=True)
        manifest = make_recovery_root_manifest(
            split=RecoverySplit(args.recovery_root_split),
            collection_label=runtime_context.collection_label,
            task_id=runtime_context.task_id,
            episode_idx=runtime_context.episode_idx,
            scene_sha256=runtime_context.scene_sha256,
            object_instance_ids=runtime_context.object_instance_ids,
            initial_state_sha256=runtime_context.initial_state_sha256,
            source_parent_snapshot_sha256=None,
            event_origin_parent_sha256=trigger.event_origin_parent_sha256,
            parent_trajectory_lineage_sha256=(
                runtime_context.parent_trajectory_lineage_sha256
            ),
            base_prompt_sha256=runtime_context.base_prompt_sha256,
            base_checkpoint_sha256=runtime_context.base_checkpoint_sha256,
            policy_client_config_sha256=(
                runtime_context.policy_client_config_sha256
            ),
            perturbation_family="observed_nominal_failure",
            perturbation_seed=None,
            branch_seed=None,
            master_seed=runtime_context.master_seed,
            policy_seed=runtime_context.policy_seed,
            simulator_seed=runtime_context.simulator_seed,
            trigger_class=trigger.trigger_class,
            deviation_status=trigger.deviation_status.value,
            deviation_event_id=trigger.deviation_event_id,
            historical_failure_evidence=trigger.historical_failure_evidence,
            relevant_fact_sha256=trigger.relevant_fact_sha256,
            source_graph_version=trigger.source_graph_version,
            source_observation_generation=trigger.observation_generation,
            certificate_state=trigger.certificate_state.value,
            grounding_rule_sha256=monitor_contract.grounding_rule_sha256,
            event_detector_sha256=monitor_contract.event_detector_sha256,
            monitor_contract_json=monitor_contract.canonical_json(),
            policy_step=trigger.policy_step,
            policy_request_generation=0,
            base_policy_request_count=context.base_policy_request_count,
            active_base_request_index=context.active_base_request_index,
            next_base_request_index=context.next_base_request_index,
            active_base_request_envelope_json=(
                context.active_base_request_envelope_json
            ),
            next_base_replay_envelope_json=(
                context.next_base_replay_envelope_json
            ),
            policy_replay_contract_sha256=(
                runtime_context.policy_replay_contract_sha256
            ),
            base_action_response_size=context.base_action_response_size,
            base_action_chunk_size=context.base_action_chunk_size,
            pending_base_action_offset=context.pending_base_action_offset,
            simulator_state=simulator_state,
            observation=observation,
            pending_base_actions=pending_actions,
            base_action_prefix_sha256=context.base_action_prefix_sha256,
            snapshot=trigger.snapshot,
        )
        return write_recovery_root(
            runtime_context.root_output_dir,
            manifest,
            simulator_state,
            observation,
            pending_actions,
        )

    runtime = build_shadow_runtime(
        provider=provider,
        provider_name=provider.provider,
        episode_context=episode_context,
        goal_mode=GoalMode(args.goal_mode),
        live_validator=live_validator,
        monitor_contract=monitor_contract,
        root_collector=collect_root,
        interval_steps=(
            args.online_monitor_interval_steps
            if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
            else args.shadow_monitor_interval_steps
        ),
        confirmation_count=(
            args.online_confirmations
            if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
            else args.shadow_confirmations
        ),
        topology_only=(
            MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
            or args.shadow_topology_only
        ),
        observation_interval_steps=(
            args.online_monitor_interval_steps
            if gpt4o_client is not None
            else None
        ),
        online_detector_settings=(
            _online_detector_settings_for_task(args, task_id=task_id)
            if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
            else None
        ),
    )
    runtime.gpt4o_client = gpt4o_client
    return runtime


def _run_config(args: argparse.Namespace, task_ids: tuple[int, ...], episode_indices: tuple[int, ...]) -> dict[str, Any]:
    online_topology = MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
    shadow_contract_hashes = (
        {
            str(task_id): load_monitor_evidence_contract(
                args.shadow_monitor_contract,
                task_id=task_id,
            ).contract_sha256
            for task_id in task_ids
        }
        if MethodArm(args.method_arm) is MethodArm.SHADOW_LOGIV
        and not args.shadow_topology_only
        else {}
    )
    return {
        "schema_version": 7,
        "run_id": args.run_id,
        "checkpoint": args.checkpoint_name,
        "method_arm": args.method_arm,
        "goal_mode": args.goal_mode,
        "deviation_mode": args.deviation_mode,
        "task_ids": list(task_ids),
        "episode_indices": list(episode_indices),
        "seed": args.seed,
        "policy_rng_protocol": "episode-seeded-v1",
        "policy_rng_seed_derivation": "uint32(sha256('LOGIV-policy-seed-v1:master:task:episode')[:4])",
        "simulator_rng_protocol": "episode-seeded-v1",
        "simulator_rng_seed_derivation": "uint32(sha256('LOGIV-simulator-seed-v1:master:task:episode')[:4])",
        "simulator_env_lifecycle": "fresh-env-per-episode-v1",
        "terminal_evaluator_protocol": "post-settling-native-check-success-v1",
        "proposal_config_sha256": resolved_json_sha256(args.proposal_config),
        "coverage_manifest_sha256": resolved_json_sha256(args.coverage_manifest),
        "prompt_version": args.prompt_version,
        "prompt_locked": args.prompt_locked,
        "development_only": args.development_only,
        "oracle_grounding": args.oracle_grounding,
        "perception_backend": args.perception_backend,
        "collect_recovery_roots": args.collect_recovery_roots,
        "capture_task5_terminal_preflight": (
            args.capture_task5_terminal_preflight
        ),
        "shadow_topology_only": args.shadow_topology_only or online_topology,
        "recovery_root_split": args.recovery_root_split,
        "shadow_monitor_interval_steps": args.shadow_monitor_interval_steps,
        "shadow_confirmations": args.shadow_confirmations,
        "online_monitor_interval_steps": args.online_monitor_interval_steps,
        "online_confirmations": args.online_confirmations,
        "online_recovery_surface_confirmations": (
            args.online_recovery_surface_confirmations
        ),
        "online_recovery_surface_confirmation_task_ids": list(
            args.online_recovery_surface_confirmation_task_ids
        ),
        "online_min_intervention_step": args.online_min_intervention_step,
        "online_stall_steps": args.online_stall_steps,
        "online_recovery_requires_goal_task_ids": list(
            args.online_recovery_requires_goal_task_ids
        ),
        "online_stall_requires_goal_task_ids": list(
            args.online_stall_requires_goal_task_ids
        ),
        "online_stall_ignores_holding_task_ids": list(
            args.online_stall_ignores_holding_task_ids
        ),
        "online_stall_requires_handempty_task_ids": list(
            args.online_stall_requires_handempty_task_ids
        ),
        "shadow_monitor_contract": (
            None
            if args.shadow_topology_only or online_topology
            else str(args.shadow_monitor_contract)
        ),
        "shadow_monitor_contract_registry_sha256": (
            None
            if args.shadow_topology_only or online_topology
            else resolved_json_sha256(args.shadow_monitor_contract)
        ),
        "shadow_monitor_contract_sha256": (
            next(iter(shadow_contract_hashes.values()))
            if len(shadow_contract_hashes) == 1
            else None
        ),
        "shadow_monitor_contract_sha256_by_task": shadow_contract_hashes,
        "replan_steps": args.replan_steps,
        "no_video": args.no_video,
        "max_action_steps": args.max_action_steps,
        "max_total_action_steps": args.base_max_steps,
        "overlay_repair_max_steps": args.overlay_repair_max_steps,
        "overlay_monitor_recovery_surface": (
            args.overlay_monitor_recovery_surface
        ),
        "overlay_monitor_interval_steps": args.overlay_monitor_interval_steps,
        "settling_steps": args.settling_steps,
        "post_stop_grounding_reobservation_steps": (
            args.post_stop_grounding_reobservation_steps
        ),
        "post_stop_grounding_reobservation_task_ids": list(
            args.post_stop_grounding_reobservation_task_ids
        ),
        "effect_confirmation_steps": args.effect_confirmation_steps,
        "place_effect_confirmation_steps": args.place_effect_confirmation_steps,
        "place_effect_confirmation_task_ids": list(
            args.place_effect_confirmation_task_ids
        ),
        "access_effect_stabilization_steps": (
            args.access_effect_stabilization_steps
        ),
        "place_effect_stabilization_steps": (
            args.place_effect_stabilization_steps
        ),
        "place_effect_stabilization_task_ids": list(
            args.place_effect_stabilization_task_ids
        ),
        "target_divergence_confirmation_steps": (
            args.target_divergence_confirmation_steps
        ),
        "held_target_divergence_confirmation_steps": (
            args.held_target_divergence_confirmation_steps
        ),
        "frontier_followup_steps": args.frontier_followup_steps,
        "frontier_completion_followup_steps": (
            args.frontier_completion_followup_steps or None
        ),
        "frontier_completion_recovery_only": (
            args.frontier_completion_recovery_only
        ),
        "frontier_recovery_max_consumed_steps": (
            args.frontier_recovery_max_consumed_steps or None
        ),
        "frontier_fallback_after_steps": (
            args.frontier_fallback_after_steps or None
        ),
        "frontier_fallback_followup_steps": (
            args.frontier_fallback_followup_steps or None
        ),
        "max_physical_attempts": args.max_physical_attempts,
        "max_repair_rounds": args.max_repair_rounds,
        "max_total_val_calls": args.max_total_val_calls,
        "max_retries_per_lineage": args.max_retries_per_lineage,
        "max_edits": args.max_edits,
        "max_candidates": args.max_candidates,
        "max_repair_val_calls": args.max_repair_val_calls,
        "val_timeout": args.val_timeout,
    }


def evaluate(args: argparse.Namespace) -> int:
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    task_ids = _parse_ids(args.task_ids, maximum=10)
    episode_indices = _parse_ids(args.episode_indices, maximum=50)
    contract = EvaluationContract(
        method_arm=MethodArm(args.method_arm),
        goal_mode=GoalMode(args.goal_mode),
        deviation_mode=args.deviation_mode,
        oracle_grounding=args.oracle_grounding,
        development_only=args.development_only,
        prompt_locked=args.prompt_locked,
        task_ids=task_ids,
        episode_indices=episode_indices,
        perception_backend=args.perception_backend,
    )
    contract.validate()
    shadow_contracts = _validate_shadow_options(args, task_ids)
    task5_preflight_capability = (
        load_task5_recovery_capability(TASK5_RECOVERY_CAPABILITY)
        if args.capture_task5_terminal_preflight
        else None
    )
    prompt_renderer = SubtaskPromptRenderer(args.prompt_config)
    if prompt_renderer.prompt_version != args.prompt_version:
        raise ValueError("prompt version/config mismatch")

    output_dir = args.output_dir.resolve()
    run_path = output_dir / "run.json"
    episodes_path = output_dir / "episodes.jsonl"
    config = _run_config(args, task_ids, episode_indices)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not run_path.exists() or json.loads(run_path.read_text()) != config:
            raise ValueError("nonempty output directory is not this exact recognized run")
        existing = load_episode_records(episodes_path)
        expected = len(task_ids) * len(episode_indices)
        if len(existing) > expected:
            raise ValueError("episode log exceeds configured allocation")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_path, config)
    existing_keys = {record.key for record in load_episode_records(episodes_path)}

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if client.get_server_metadata().get("logiv_episode_rng_protocol") != 1:
        raise ValueError("policy server does not support LOGIV episode RNG protocol v1")
    common_hashes = {
        "prompt_config_sha256": _sha256_file(Path(args.prompt_config)),
        "proposal_config_sha256": resolved_json_sha256(args.proposal_config),
        "coverage_manifest_sha256": resolved_json_sha256(args.coverage_manifest),
        "domain_sha256": _sha256_bytes(render_domain_pddl().encode("utf-8")),
    }
    base_checkpoint_sha256 = _sha256_file(
        REPOSITORY_ROOT
        / "artifacts"
        / "manifests"
        / f"{args.checkpoint_name}-checkpoint.json"
    )
    policy_client_config_sha256 = _domain_sha256(
        b"LOGIV_POLICY_CLIENT_CONFIG_V1",
        {
            "checkpoint_name": args.checkpoint_name,
            "host": args.host,
            "port": args.port,
            "replan_steps": args.replan_steps,
            "rng_protocol": "episode-seeded-v1",
        },
    )

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        scene_sha256 = _sha256_file(task_bddl_file)
        base_prompt_sha256 = _sha256_bytes(str(task.language).encode("utf-8"))
        env = None
        try:
            for episode_idx in episode_indices:
                key = (args.run_id, args.method_arm, args.goal_mode, task_id, episode_idx)
                if key in existing_keys:
                    continue
                episode_started = time.monotonic()
                episode_id = (
                    f"{args.run_id}/{args.method_arm}/{args.goal_mode}/task-{task_id}/episode-{episode_idx}"
                )
                artifact_dir = _allocate_episode_artifact_dir(
                    output_dir, task_id=task_id, episode_idx=episode_idx
                )
                policy_seed = derive_episode_seed(
                    "policy",
                    master_seed=args.seed,
                    task_id=task_id,
                    episode_idx=episode_idx,
                )
                simulator_seed = derive_episode_seed(
                    "simulator",
                    master_seed=args.seed,
                    task_id=task_id,
                    episode_idx=episode_idx,
                )
                env = _replace_episode_environment(
                    env,
                    factory=OffScreenRenderEnv,
                    bddl_file=task_bddl_file,
                )
                episode_client = EpisodeSeededClient(
                    client,
                    episode_seed=policy_seed,
                    policy_client_config_sha256=policy_client_config_sha256,
                )
                _write_json(
                    artifact_dir / "policy_rng.json",
                    {
                        "protocol": "episode-seeded-v1",
                        "master_seed": args.seed,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "episode_seed": policy_seed,
                        "simulator_protocol": "episode-seeded-v1",
                        "simulator_seed": simulator_seed,
                    },
                )
                seed_episode_runtime(env, simulator_seed)
                journal = EventJournal(artifact_dir / "events.jsonl")
                initial_state = np.asarray(initial_states[episode_idx])
                init_hash = _sha256_array(initial_state)
                first_frame_hash = "0" * 64
                graph = None
                final_graph = None
                certificate = None
                executor = None
                evaluator = None
                result = None
                outcome = None
                shadow_runtime = None
                exception_text = None
                valid = True
                frames: list[np.ndarray] = []
                base_prefix_steps = 0
                repair_steps = 0
                online_trigger = None
                episode_gpt4o_client = None
                try:
                    arm = MethodArm(args.method_arm)
                    if args.perception_backend == "gpt4o" and arm not in {
                        MethodArm.BASE,
                        MethodArm.SHADOW_LOGIV,
                        MethodArm.LOGIV_ONLINE,
                    }:
                        episode_gpt4o_client = Gpt4oClient.from_env()
                    if arm in {
                        MethodArm.BASE,
                        MethodArm.SHADOW_LOGIV,
                        MethodArm.LOGIV_ONLINE,
                        MethodArm.LOGIV_REPAIR_OVERLAY,
                    }:
                        if arm in {
                            MethodArm.SHADOW_LOGIV,
                            MethodArm.LOGIV_ONLINE,
                        }:
                            shadow_runtime = _build_evaluator_shadow_runtime(
                                args,
                                env=env,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                episode_id=episode_id,
                                initial_state_sha256=init_hash,
                                scene_sha256=scene_sha256,
                                base_prompt_sha256=base_prompt_sha256,
                                base_checkpoint_sha256=base_checkpoint_sha256,
                                policy_client_config_sha256=(
                                    policy_client_config_sha256
                                ),
                                policy_seed=policy_seed,
                                simulator_seed=simulator_seed,
                                artifact_dir=artifact_dir,
                                monitor_contract=shadow_contracts.get(task_id),
                                image_tools=image_tools,
                            )
                            episode_gpt4o_client = shadow_runtime.gpt4o_client
                        online_initialization_guard = None
                        if arm is MethodArm.LOGIV_ONLINE:
                            assert shadow_runtime is not None

                            def require_certified_online_graph() -> None:
                                assert shadow_runtime is not None
                                proposal_result = shadow_runtime.initial_proposal
                                if (
                                    proposal_result is None
                                    or proposal_result.validation is None
                                ):
                                    reason = (
                                        proposal_result.reason
                                        if proposal_result is not None
                                        else "proposal callback did not run"
                                    )
                                    raise RuntimeError(
                                        f"initial DAG was not certified: {reason}"
                                    )

                            online_initialization_guard = (
                                require_certified_online_graph
                            )
                        overlay_monitor = None
                        if (
                            arm is MethodArm.LOGIV_REPAIR_OVERLAY
                            and args.overlay_monitor_recovery_surface
                        ):
                            monitor_package = ScriptedProposalProvider(
                                args.proposal_config
                            ).propose(
                                task_id,
                                epoch_id=0,
                                goal_mode=GoalMode(args.goal_mode),
                            )
                            monitor_binding = TaskBinding.from_manifest(
                                args.coverage_manifest, task_id
                            )
                            overlay_monitor = _RecoverySurfaceMonitor(
                                env=env,
                                binding=monitor_binding,
                                required_facts=monitored_fact_universe(
                                    monitor_package.problem
                                ),
                                interval_steps=args.overlay_monitor_interval_steps,
                            )
                        intervention_monitor = overlay_monitor
                        if arm is MethodArm.LOGIV_ONLINE:
                            assert shadow_runtime is not None
                            intervention_monitor = (
                                lambda _obs, _action, _step: (
                                    shadow_runtime.online_repair_request is not None
                                )
                            )
                        outcome = run_episode(
                            env,
                            episode_client,
                            initial_state,
                            str(task.language),
                            image_tools,
                            max_steps=args.base_max_steps,
                            wait_steps=args.wait_steps,
                            replan_steps=args.replan_steps,
                            settling_steps=args.settling_steps,
                            shadow_observer=(
                                shadow_runtime.observer
                                if shadow_runtime is not None
                                else None
                            ),
                            shadow_settling_observer=(
                                shadow_runtime.settling_observer
                                if shadow_runtime is not None
                                else None
                            ),
                            request_envelope_reader=(
                                episode_client.request_envelope_reader
                            ),
                            capture_replay_frames=not args.no_video,
                            intervention_monitor=intervention_monitor,
                            shadow_initialization_guard=(
                                online_initialization_guard
                            ),
                        )
                        base_prefix_steps = outcome.steps
                        if shadow_runtime is not None:
                            online_trigger = shadow_runtime.online_repair_request
                        evaluator = NativeLiberoTaskEvaluator()
                        if outcome.done:
                            evaluated = EvaluatorStatus.EPISODE_SUCCESS
                            evaluator.last_status = evaluated
                        else:
                            evaluated = evaluator.evaluate(env)
                        if not outcome.done and outcome.check_success != (
                            evaluated.value == ControllerStatus.EPISODE_SUCCESS.value
                        ):
                            raise RuntimeError(
                                "Base post-settling result/evaluator disagreement"
                            )
                        first_frame_hash = _sha256_array(outcome.first_frame)
                        frames = list(outcome.replay_frames)
                        _write_json(
                            artifact_dir / "base_execution.json",
                            _base_execution_payload(
                                outcome,
                                episode_client,
                                initial_state_sha256=init_hash,
                                base_prompt_sha256=base_prompt_sha256,
                                base_checkpoint_sha256=base_checkpoint_sha256,
                                policy_client_config_sha256=(
                                    policy_client_config_sha256
                                ),
                            ),
                        )
                        if args.capture_task5_terminal_preflight:
                            assert shadow_runtime is not None
                            assert task5_preflight_capability is not None
                            monitor_contract = shadow_contracts[task_id]
                            _capture_task5_terminal_preflight(
                                artifact_dir=artifact_dir,
                                case_id=args.run_id.removeprefix(
                                    "task5-terminal-preflight-"
                                ),
                                task_id=task_id,
                                episode_idx=episode_idx,
                                episode_id=episode_id,
                                outcome=outcome,
                                native_terminal_status=evaluated.value,
                                runtime=shadow_runtime,
                                monitor_contract=monitor_contract,
                                capability=task5_preflight_capability,
                            )
                        if overlay_monitor is not None:
                            _write_json(
                                artifact_dir / "overlay_monitor.json",
                                {
                                    "interval_steps": (
                                        overlay_monitor.interval_steps
                                    ),
                                    "trigger_step": overlay_monitor.trigger_step,
                                    "trigger_facts": [
                                        fact.pddl()
                                        for fact in overlay_monitor.trigger_facts
                                    ],
                                    "grounding_errors": (
                                        overlay_monitor.grounding_errors
                                    ),
                                },
                            )
                        remaining_online_steps = (
                            _remaining_online_action_budget(
                                args.base_max_steps, outcome.steps
                            )
                            if arm is MethodArm.LOGIV_ONLINE
                            else None
                        )
                        direct_base_result = (
                            arm in {MethodArm.BASE, MethodArm.SHADOW_LOGIV}
                            or outcome.success
                            or (
                                arm is MethodArm.LOGIV_ONLINE
                                and (
                                    online_trigger is None
                                    or remaining_online_steps == 0
                                )
                            )
                        )
                        if direct_base_result:
                            status = (
                                ControllerStatus.EPISODE_SUCCESS
                                if outcome.success
                                else ControllerStatus(evaluated.value)
                            )
                            if (
                                arm is MethodArm.LOGIV_ONLINE
                                and online_trigger is not None
                                and remaining_online_steps == 0
                            ):
                                terminal_cause = "BUDGET_EXHAUSTED"
                            else:
                                terminal_cause = status.value
                            if arm in {
                                MethodArm.BASE,
                                MethodArm.SHADOW_LOGIV,
                            }:
                                direct_event = "BASE_DIRECT_EXECUTION"
                            elif arm is MethodArm.LOGIV_ONLINE:
                                if outcome.success:
                                    direct_event = "ONLINE_NATIVE_SUCCESS_ABSORBED"
                                elif online_trigger is None:
                                    direct_event = "ONLINE_NO_VERIFIED_DEVIATION"
                                else:
                                    direct_event = "ONLINE_REPAIR_BUDGET_EXHAUSTED"
                            else:
                                direct_event = "OVERLAY_BASE_PREFIX_SUCCESS"
                            result = ControllerResult(
                                status=status,
                                terminal_cause=terminal_cause,
                                receipts=(),
                                events=(
                                    direct_event,
                                    f"BASE_DONE_SIGNAL:{outcome.done}",
                                    f"BASE_POST_SETTLING_SUCCESS:{outcome.check_success}",
                                ),
                                budget_usage=RuntimeBudgetUsage(
                                    _base_physical_attempts(outcome.steps), 0, 0
                                ),
                                graph_installs=0,
                                active_attempt_id=None,
                            )
                            initial_context = ContextEnvelope(
                                ContextPhase.INITIAL,
                                GoalMode(args.goal_mode),
                                f"{episode_id}-base",
                                0,
                                episode_id,
                                f"libero10-task-{task_id}-external-only",
                                0,
                                0,
                                None,
                                None,
                                None,
                                None,
                                None,
                            )
                            evaluator_status = evaluated.value
                            steps = outcome.steps
                            inference_requests = outcome.inference_requests
                            initial_snapshot = None
                            package = None
                            if shadow_runtime is not None:
                                (
                                    proposal_payload,
                                    monitor_payload,
                                    compute_payload,
                                    _,
                                ) = _shadow_artifact_payloads(
                                    outcome, shadow_runtime
                                )
                                _write_json(
                                    artifact_dir / "initial_proposal.json",
                                    proposal_payload,
                                )
                                _write_json(
                                    artifact_dir / "shadow_monitor.json",
                                    monitor_payload,
                                )
                                _write_json(
                                    artifact_dir / "compute_accounting.json",
                                    compute_payload,
                                )
                                proposal_result = shadow_runtime.initial_proposal
                                if (
                                    proposal_result is not None
                                    and proposal_result.validation is not None
                                ):
                                    package = proposal_result.package
                                    certified = (
                                        proposal_result.validation.certified_episode
                                    )
                                    graph = certified.graph
                                    final_graph = certified.graph
                                    certificate = certified.certificate
                        else:
                            recovery_package = None
                            recovery_causal_slice = None
                            if (
                                arm is MethodArm.LOGIV_ONLINE
                                and args.perception_backend == "gpt4o"
                            ):
                                assert shadow_runtime is not None
                                (
                                    recovery_package,
                                    recovery_causal_slice,
                                ) = _online_recovery_inputs(shadow_runtime)
                            (
                                package,
                                initial_context,
                                initial_snapshot,
                                result,
                                executor,
                                evaluator,
                                graph,
                                final_graph,
                                certificate,
                                _,
                            ) = _execute_symbolic_arm(
                                args,
                                env,
                                episode_client,
                                image_tools,
                                task_id,
                                episode_id,
                                outcome.final_observation,
                                recovery_state=True,
                                max_total_action_steps=(
                                    remaining_online_steps
                                    if arm is MethodArm.LOGIV_ONLINE
                                    else args.base_max_steps
                                    + args.overlay_repair_max_steps
                                    - outcome.steps
                                ),
                                max_physical_attempts=max(
                                    0, args.max_physical_attempts - 1
                                ),
                                gpt4o_client=episode_gpt4o_client,
                                recovery_package=recovery_package,
                                causal_slice=recovery_causal_slice,
                            )
                            frames.extend(
                                frame
                                for attempt in executor.results
                                for frame in attempt.frames
                            )
                            repair_steps = sum(
                                len(attempt.actions) for attempt in executor.results
                            )
                            repair_inference_requests = sum(
                                attempt.inference_requests
                                for attempt in executor.results
                            )
                            intervention_event = (
                                "ONLINE_VERIFIED_DEVIATION_INTERVENTION"
                                if arm is MethodArm.LOGIV_ONLINE
                                else (
                                    "OVERLAY_VERIFIED_DEVIATION_INTERVENTION"
                                    if outcome.intervention_requested
                                    else "OVERLAY_BASE_PREFIX_FAILED"
                                )
                            )
                            prefix_event = (
                                f"ONLINE_BASE_PREFIX_STEPS:{outcome.steps}"
                                if arm is MethodArm.LOGIV_ONLINE
                                else f"OVERLAY_BASE_PREFIX_STEPS:{outcome.steps}"
                            )
                            result = replace(
                                result,
                                events=(
                                    intervention_event,
                                    prefix_event,
                                )
                                + result.events,
                                budget_usage=replace(
                                    result.budget_usage,
                                    physical_attempts=(
                                        result.budget_usage.physical_attempts
                                        + _base_physical_attempts(outcome.steps)
                                    ),
                                ),
                            )
                            steps = outcome.steps + repair_steps
                            inference_requests = (
                                outcome.inference_requests
                                + repair_inference_requests
                            )
                            evaluator_status = _effective_evaluator_status(
                                result, evaluator
                            )
                    else:
                        initial_observation = _reset_episode(env, initial_state, args.wait_steps)
                        _, first_frame = prepare_observation(
                            initial_observation, str(task.language), image_tools
                        )
                        first_frame_hash = _sha256_array(first_frame)
                        (
                            package,
                            initial_context,
                            initial_snapshot,
                            result,
                            executor,
                            evaluator,
                            graph,
                            final_graph,
                            certificate,
                            _,
                        ) = _execute_symbolic_arm(
                            args,
                            env,
                            episode_client,
                            image_tools,
                            task_id,
                            episode_id,
                            initial_observation,
                            gpt4o_client=episode_gpt4o_client,
                        )
                        frames = [frame for attempt in executor.results for frame in attempt.frames]
                        steps = sum(len(attempt.actions) for attempt in executor.results)
                        inference_requests = sum(
                            attempt.inference_requests for attempt in executor.results
                        )
                        evaluator_status = _effective_evaluator_status(
                            result, evaluator
                        )
                    if initial_snapshot is not None:
                        _write_json(
                            artifact_dir / "initial_snapshot.json",
                            {
                                "epoch_id": initial_snapshot.epoch_id,
                                "true": sorted(fact.pddl() for fact in initial_snapshot.true_facts),
                                "false": sorted(fact.pddl() for fact in initial_snapshot.false_facts),
                                "evidence_hash": initial_snapshot.evidence_hash,
                            },
                        )
                        if graph is not None:
                            _write_json(artifact_dir / "graph.json", _graph_json(graph))
                        if certificate is not None:
                            _write_json(
                                artifact_dir / "certificate.json",
                                _certificate_json(certificate),
                            )
                        assert executor is not None
                        for attempt_index, attempt in enumerate(executor.results):
                            _write_json(
                                artifact_dir / f"attempt_{attempt_index:03d}.json",
                                _attempt_json(attempt),
                            )
                    assert result is not None
                    _record_events(
                        journal,
                        initial_context,
                        result,
                        executor,
                        graph,
                        certificate,
                        evaluator,
                    )
                except Exception as error:
                    valid = False
                    exception_text = f"{type(error).__name__}: {error}"
                    logging.exception("episode failed with an unhandled exception")
                    initial_context = ContextEnvelope(
                        ContextPhase.INITIAL,
                        GoalMode(args.goal_mode),
                        f"{episode_id}-exception",
                        0,
                        episode_id,
                        f"libero10-task-{task_id}-goal",
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    result = ControllerResult(
                        ControllerStatus.TERMINAL_NO_FURTHER_DISPATCH,
                        "EXCEPTION",
                        (),
                        (),
                        RuntimeBudgetUsage(0, 0, 0),
                        0,
                        None,
                    )
                    evaluator_status = "NOT_CALLED"
                    steps = 0
                    inference_requests = 0
                    journal.append(
                        "UNHANDLED_EXCEPTION",
                        initial_context,
                        {"exception": exception_text, "traceback": traceback.format_exc()},
                    )

                if shadow_runtime is not None:
                    _write_shadow_graph_artifact(artifact_dir, shadow_runtime)
                    if (
                        MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        and final_graph is not None
                    ):
                        _write_json(
                            artifact_dir / "final_graph.json",
                            _graph_json(final_graph),
                        )
                    if outcome is not None:
                        (
                            proposal_payload,
                            monitor_payload,
                            compute_payload,
                            _,
                        ) = _shadow_artifact_payloads(outcome, shadow_runtime)
                        _write_json(
                            artifact_dir / "initial_proposal.json", proposal_payload
                        )
                        _write_json(
                            artifact_dir / "shadow_monitor.json", monitor_payload
                        )
                        _write_json(
                            artifact_dir / "compute_accounting.json", compute_payload
                        )
                    if online_trigger is not None:
                        _write_json(
                            artifact_dir / "online_repair_trigger.json",
                            {
                                "kind": online_trigger.kind.value,
                                "policy_step": online_trigger.policy_step,
                                "first_observed_step": (
                                    online_trigger.first_observed_step
                                ),
                                "signature": list(online_trigger.signature),
                                "source_graph_hash": (
                                    online_trigger.source_graph_hash
                                ),
                                "request_sha256": online_trigger.request_sha256,
                                "snapshot_evidence_hash": (
                                    online_trigger.snapshot.evidence_hash
                                ),
                                "discarded_pending_actions": (
                                    outcome.discarded_pending_actions
                                    if outcome is not None
                                    else 0
                                ),
                            },
                        )

                record_accounting = _symbolic_record_accounting(
                    base_policy_requests=inference_requests,
                    gpt4o_client=episode_gpt4o_client,
                )
                if MethodArm(args.method_arm) in {
                    MethodArm.SHADOW_LOGIV,
                    MethodArm.LOGIV_ONLINE,
                }:
                    record_accounting = _shadow_record_accounting(
                        outcome=outcome,
                        runtime=shadow_runtime,
                        exception_text=exception_text,
                        base_policy_requests=inference_requests,
                    )
                    if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE:
                        record_accounting["base_policy_requests"] = (
                            inference_requests
                        )

                video_path = None
                if frames and not args.no_video:
                    relative = Path("videos") / f"task_{task_id:02d}_episode_{episode_idx:03d}.mp4"
                    final_video = output_dir / relative
                    final_video.parent.mkdir(parents=True, exist_ok=True)
                    temporary = final_video.with_name(f".{final_video.name}.tmp.mp4")
                    imageio.mimwrite(temporary, frames, fps=args.video_fps)
                    temporary.replace(final_video)
                    video_path = str(relative)
                success = result.status is ControllerStatus.EPISODE_SUCCESS
                receipt_statuses = [receipt.status for receipt in result.receipts]
                record_initial_graph = graph
                record_initial_certificate = certificate
                if (
                    MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                    and shadow_runtime is not None
                    and shadow_runtime.initial_proposal is not None
                    and shadow_runtime.initial_proposal.validation is not None
                ):
                    online_certified = (
                        shadow_runtime.initial_proposal.validation.certified_episode
                    )
                    record_initial_graph = online_certified.graph
                    record_initial_certificate = online_certified.certificate
                record = LogivEpisodeRecord(
                    schema_version=(
                        5
                        if args.perception_backend == "gpt4o"
                        else 4
                        if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        else 3
                    ),
                    run_id=args.run_id,
                    checkpoint=args.checkpoint_name,
                    method_arm=args.method_arm,
                    goal_mode=args.goal_mode,
                    deviation_mode=args.deviation_mode,
                    task_id=task_id,
                    task_name=str(task.language),
                    episode_idx=episode_idx,
                    seed=args.seed,
                    allocated=True,
                    valid=valid,
                    success=success,
                    terminal_status=result.status.value,
                    terminal_cause=result.terminal_cause or result.status.value,
                    evaluator_status=evaluator_status,
                    init_state_sha256=init_hash,
                    first_frame_sha256=first_frame_hash,
                    prompt_version=args.prompt_version,
                    initial_certificate_hash=(
                        record_initial_certificate.certificate_hash
                        if record_initial_certificate is not None
                        else None
                    ),
                    initial_graph_hash=(
                        record_initial_graph.graph_hash
                        if record_initial_graph is not None
                        else None
                    ),
                    initial_graph_width=(
                        record_initial_graph.action_layer_width()
                        if record_initial_graph is not None
                        else None
                    ),
                    final_graph_hash=(
                        final_graph.graph_hash if final_graph is not None else None
                    ),
                    physical_attempts=result.budget_usage.physical_attempts,
                    repair_rounds=result.budget_usage.repair_rounds,
                    total_val_calls=result.budget_usage.total_val_calls,
                    graph_installs=result.graph_installs,
                    steps=steps,
                    inference_requests=inference_requests,
                    wall_seconds=time.monotonic() - episode_started,
                    safety_permits=len(executor.results) if executor is not None else 0,
                    halt_acknowledged=(
                        True
                        if result.status is ControllerStatus.SAFE_STOPPED
                        else False if result.status is ControllerStatus.UNSAFE_TERMINAL else None
                    ),
                    receipts_count=len(result.receipts),
                    event_count=journal.count,
                    event_chain_head=journal.head,
                    artifact_dir=str(artifact_dir.relative_to(output_dir)),
                    video_path=video_path,
                    exception=exception_text,
                    oracle_grounding=(
                        MethodArm(args.method_arm) is not MethodArm.BASE
                        and args.perception_backend == "scripted-oracle"
                    ),
                    perception_backend=args.perception_backend,
                    development_only=args.development_only,
                    committed_receipts=receipt_statuses.count(
                        AttemptReceiptStatus.COMMITTED
                    ),
                    failed_receipts=receipt_statuses.count(AttemptReceiptStatus.FAILED),
                    unknown_receipts=sum(
                        status
                        in {
                            AttemptReceiptStatus.POST_STOP_UNKNOWN,
                            AttemptReceiptStatus.REJECTED,
                        }
                        for status in receipt_statuses
                    ),
                    precondition_gate_rejections=result.events.count(
                        "PRECONDITION_GATE_REJECTED"
                    ),
                    effect_gate_rejections=result.events.count(
                        "EFFECT_GATE_REJECTED"
                    ),
                    final_goal_gate_rejections=result.events.count(
                        "FINAL_GOAL_GATE_REJECTED"
                    ),
                    base_prefix_steps=(
                        base_prefix_steps
                        if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        else 0
                    ),
                    repair_steps=(
                        repair_steps
                        if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        else 0
                    ),
                    combined_actions=(
                        steps
                        if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        else 0
                    ),
                    discarded_pending_actions=(
                        outcome.discarded_pending_actions
                        if MethodArm(args.method_arm) is MethodArm.LOGIV_ONLINE
                        and outcome is not None
                        else 0
                    ),
                    online_trigger_kind=(
                        online_trigger.kind.value
                        if online_trigger is not None
                        else None
                    ),
                    online_trigger_step=(
                        online_trigger.policy_step
                        if online_trigger is not None
                        else None
                    ),
                    online_trigger_sha256=(
                        online_trigger.request_sha256
                        if online_trigger is not None
                        else None
                    ),
                    **record_accounting,
                    **common_hashes,
                )
                append_episode_record(episodes_path, record)
                existing_keys.add(record.key)
                logging.info(
                    "arm=%s task=%d episode=%d status=%s steps=%d",
                    args.method_arm,
                    task_id,
                    episode_idx,
                    result.status.value,
                    steps,
                )
        finally:
            if env is not None:
                env.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable interactive LOGIV evaluation on frozen LIBERO-10 tasks"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-name", default="full", choices=("full",))
    parser.add_argument("--method-arm", required=True, choices=tuple(item.value for item in MethodArm))
    parser.add_argument("--goal-mode", required=True, choices=tuple(item.value for item in GoalMode))
    parser.add_argument("--deviation-mode", required=True)
    parser.add_argument("--oracle-grounding", action="store_true")
    parser.add_argument(
        "--perception-backend",
        choices=("scripted-oracle", "gpt4o"),
        default="scripted-oracle",
    )
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--collect-recovery-roots", action="store_true")
    parser.add_argument("--capture-task5-terminal-preflight", action="store_true")
    parser.add_argument("--shadow-topology-only", action="store_true")
    parser.add_argument(
        "--recovery-root-split",
        choices=("TRAIN", "DEV", "HELDOUT"),
        default="DEV",
    )
    parser.add_argument("--shadow-monitor-interval-steps", default=5, type=int)
    parser.add_argument("--shadow-confirmations", default=3, type=int)
    parser.add_argument("--online-monitor-interval-steps", default=5, type=int)
    parser.add_argument("--online-confirmations", default=3, type=int)
    parser.add_argument(
        "--online-recovery-surface-confirmations", default=1, type=int
    )
    parser.add_argument(
        "--online-recovery-surface-confirmation-task-ids",
        default=(),
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument("--online-min-intervention-step", default=120, type=int)
    parser.add_argument("--online-stall-steps", default=120, type=int)
    parser.add_argument(
        "--online-recovery-requires-goal-task-ids",
        default=(6,),
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument(
        "--online-stall-requires-handempty-task-ids",
        default=(8,),
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument(
        "--online-stall-requires-goal-task-ids",
        default=(),
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument(
        "--online-stall-ignores-holding-task-ids",
        default=(),
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument(
        "--shadow-monitor-contract",
        default=REPOSITORY_ROOT / "configs/logiv/r2m-monitor-evidence-v1.json",
        type=Path,
    )
    parser.add_argument("--prompt-locked", action="store_true")
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--episode-indices", default="0:50")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--wait-steps", default=10, type=int)
    parser.add_argument("--replan-steps", default=5, type=int)
    parser.add_argument("--max-action-steps", default=260, type=int)
    parser.add_argument("--base-max-steps", default=520, type=int)
    parser.add_argument("--overlay-repair-max-steps", default=180, type=int)
    parser.add_argument("--overlay-monitor-recovery-surface", action="store_true")
    parser.add_argument("--overlay-monitor-interval-steps", default=5, type=int)
    parser.add_argument("--settling-steps", default=10, type=int)
    parser.add_argument(
        "--post-stop-grounding-reobservation-steps", default=0, type=int
    )
    parser.add_argument(
        "--post-stop-grounding-reobservation-task-ids",
        default="all",
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument("--effect-confirmation-steps", default=5, type=int)
    parser.add_argument("--place-effect-confirmation-steps", default=0, type=int)
    parser.add_argument(
        "--place-effect-confirmation-task-ids",
        default="all",
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument("--access-effect-stabilization-steps", default=0, type=int)
    parser.add_argument("--place-effect-stabilization-steps", default=0, type=int)
    parser.add_argument(
        "--place-effect-stabilization-task-ids",
        default="all",
        type=lambda value: _parse_ids(value, maximum=10),
    )
    parser.add_argument("--target-divergence-confirmation-steps", default=5, type=int)
    parser.add_argument(
        "--held-target-divergence-confirmation-steps", default=0, type=int
    )
    parser.add_argument("--frontier-followup-steps", default=180, type=int)
    parser.add_argument("--frontier-completion-followup-steps", default=0, type=int)
    parser.add_argument("--frontier-completion-recovery-only", action="store_true")
    parser.add_argument("--frontier-recovery-max-consumed-steps", default=0, type=int)
    parser.add_argument("--frontier-fallback-after-steps", default=0, type=int)
    parser.add_argument("--frontier-fallback-followup-steps", default=120, type=int)
    parser.add_argument("--video-fps", default=10, type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--watchdog-seconds", default=60.0, type=float)
    parser.add_argument("--action-limit", default=1.1, type=float)
    parser.add_argument("--max-physical-attempts", default=12, type=int)
    parser.add_argument("--max-repair-rounds", default=4, type=int)
    parser.add_argument("--max-total-val-calls", default=40, type=int)
    parser.add_argument("--max-retries-per-lineage", default=1, type=int)
    parser.add_argument("--max-edits", default=5, type=int)
    parser.add_argument("--max-candidates", default=10000, type=int)
    parser.add_argument("--max-repair-val-calls", default=20, type=int)
    parser.add_argument("--val-timeout", default=5.0, type=float)
    parser.add_argument("--val-binary", default="/val/Validate", type=Path)
    parser.add_argument(
        "--prompt-config",
        default=REPOSITORY_ROOT / "configs/logiv/prompts/pi05-subtasks-v1.json",
        type=Path,
    )
    parser.add_argument(
        "--proposal-config",
        default=DEFAULT_FIXTURE,
        type=Path,
    )
    parser.add_argument(
        "--coverage-manifest",
        default=COVERAGE_MANIFEST,
        type=Path,
    )
    parser.add_argument("--prompt-version", default="pi05-subtasks-v1")
    parser.add_argument("--diagnostic-resume", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return evaluate(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
