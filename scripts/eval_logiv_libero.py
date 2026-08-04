#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from pi05_libero_repro.logiv.repair import RepairBounds, RetryPolicy
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


def _ground_initial(package, grounder: LiberoOracleGrounder, context: ContextEnvelope) -> FactSnapshot:
    required = monitored_fact_universe(package.problem)
    response = grounder.ground(ContextPhase.INITIAL_FACTS, context, required)
    if response.status is not GroundingStatus.OK or response.snapshot is None:
        raise RuntimeError(f"STATE_GROUNDING_FAILURE:{response.reason}")
    proposal = package.proposal.initial_snapshot
    if not proposal.true_facts <= response.snapshot.true_facts:
        raise RuntimeError("STATE_GROUNDING_FAILURE:proposal TRUE conflict")
    if not proposal.false_facts <= response.snapshot.false_facts:
        raise RuntimeError("STATE_GROUNDING_FAILURE:proposal FALSE conflict")
    return response.snapshot


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
            "evaluator_status": (
                evaluator.last_status.value
                if evaluator is not None and evaluator.last_status is not None
                else "NOT_CALLED"
            ),
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
):
    provider = ScriptedProposalProvider(args.proposal_config)
    package = provider.propose(task_id, epoch_id=0, goal_mode=GoalMode(args.goal_mode))
    binding = TaskBinding.from_manifest(args.coverage_manifest, task_id)
    store = LiberoObservationStore(initial_observation, epoch_id=0)
    grounder = LiberoOracleGrounder(
        env,
        store,
        binding,
        monitored_fact_universe(package.problem),
    )
    initial_context = _initial_context(package, episode_id)
    initial_snapshot = _ground_initial(package, grounder, initial_context)
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
        effect_confirmation_steps=args.effect_confirmation_steps,
        access_effect_stabilization_steps=args.access_effect_stabilization_steps,
        target_divergence_confirmation_steps=(
            args.target_divergence_confirmation_steps
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
        max_total_action_steps=args.base_max_steps,
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
        certified = certify_initial_package(
            package,
            initial_snapshot,
            episode_id=episode_id,
            val_wrapper=val_wrapper,
            allowed_schemas=binding.supported_action_schemas | binding.recovery_schemas,
            repair_bounds=RepairBounds(
                max_edits=args.max_edits,
                max_candidates=args.max_candidates,
                max_val_calls=args.max_repair_val_calls,
            ),
            retry_policy=policy,
            decompose_macro_sources=binding.decompose_macro_sources,
        )
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
                max_physical_attempts=args.max_physical_attempts,
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

    def metric(name: str) -> int:
        return int(getattr(metrics, name, 0))

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
        "root_count": runtime.counters.root_count,
        "aggregate_errors": aggregate_errors,
    }
    compute_payload = {
        "base_policy_requests": outcome.inference_requests,
        "initial_proposal_requests": request_count,
        "shadow_vlm_requests": 0,
        "recovery_policy_requests": 0,
        "initial_proposal_seconds": elapsed_seconds,
        "shadow_monitor_seconds": monitor_seconds,
    }
    record_accounting = {
        "base_policy_requests": outcome.inference_requests,
        "initial_proposal_requests": request_count,
        "initial_proposal_status": status,
        "initial_proposal_reason_code": reason_code,
        "shadow_vlm_requests": 0,
        "recovery_policy_requests": 0,
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
    return {
        "base_policy_requests": base_policy_requests,
        "initial_proposal_requests": request_count,
        "initial_proposal_status": status,
        "initial_proposal_reason_code": reason_code,
        "shadow_vlm_requests": 0,
        "recovery_policy_requests": 0,
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
        "base_policy_requests": outcome.inference_requests,
        "done_signal": outcome.done,
        "post_settling_success": outcome.check_success,
        "initial_state_sha256": initial_state_sha256,
        "base_prompt_sha256": base_prompt_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "policy_client_config_sha256": policy_client_config_sha256,
        "request_envelope_log_sha256": request_envelope_log_sha256,
        "actions_sha256": actions_sha256,
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
) -> ShadowRuntime:
    binding = TaskBinding.from_manifest(args.coverage_manifest, task_id)
    provider = ScriptedProposalProvider(args.proposal_config)
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
        grounder = LiberoOracleGrounder(
            env,
            store,
            binding,
            monitored_fact_universe(package.problem),
        )
        initial_context = _initial_context(package, episode_id)
        initial_snapshot = _ground_initial(package, grounder, initial_context)
        certified = certify_initial_package(
            package,
            initial_snapshot,
            episode_id=episode_id,
            val_wrapper=ValWrapper(args.val_binary, timeout_seconds=args.val_timeout),
            allowed_schemas=(
                binding.supported_action_schemas | binding.recovery_schemas
            ),
            repair_bounds=RepairBounds(
                max_edits=args.max_edits,
                max_candidates=args.max_candidates,
                max_val_calls=args.max_repair_val_calls,
            ),
            retry_policy=RetryPolicy(
                max_retries_per_lineage=args.max_retries_per_lineage
            ),
            decompose_macro_sources=binding.decompose_macro_sources,
        )
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
            previous_snapshot = grounder.peek_snapshot()
            previous_observation_sha256 = current_hash
            return previous_snapshot

        return ShadowValidatedProposal(certified, snapshot_reader)

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

    return build_shadow_runtime(
        provider=provider,
        provider_name=provider.provider,
        episode_context=episode_context,
        goal_mode=GoalMode(args.goal_mode),
        live_validator=live_validator,
        monitor_contract=monitor_contract,
        root_collector=collect_root,
        interval_steps=args.shadow_monitor_interval_steps,
        confirmation_count=args.shadow_confirmations,
        topology_only=args.shadow_topology_only,
    )


def _run_config(args: argparse.Namespace, task_ids: tuple[int, ...], episode_indices: tuple[int, ...]) -> dict[str, Any]:
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
        "collect_recovery_roots": args.collect_recovery_roots,
        "shadow_topology_only": args.shadow_topology_only,
        "recovery_root_split": args.recovery_root_split,
        "shadow_monitor_interval_steps": args.shadow_monitor_interval_steps,
        "shadow_confirmations": args.shadow_confirmations,
        "shadow_monitor_contract": (
            None if args.shadow_topology_only else str(args.shadow_monitor_contract)
        ),
        "shadow_monitor_contract_registry_sha256": (
            None
            if args.shadow_topology_only
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
        "settling_steps": args.settling_steps,
        "effect_confirmation_steps": args.effect_confirmation_steps,
        "target_divergence_confirmation_steps": (
            args.target_divergence_confirmation_steps
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
    )
    contract.validate()
    shadow_contracts = _validate_shadow_options(args, task_ids)
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
                try:
                    arm = MethodArm(args.method_arm)
                    if arm in {MethodArm.BASE, MethodArm.SHADOW_LOGIV}:
                        if arm is MethodArm.SHADOW_LOGIV:
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
                            request_envelope_reader=(
                                episode_client.request_envelope_reader
                            ),
                            capture_replay_frames=not args.no_video,
                        )
                        evaluator = NativeLiberoTaskEvaluator()
                        evaluated = evaluator.evaluate(env)
                        if outcome.check_success != (
                            evaluated.value == ControllerStatus.EPISODE_SUCCESS.value
                        ):
                            raise RuntimeError(
                                "Base post-settling result/evaluator disagreement"
                            )
                        first_frame_hash = _sha256_array(outcome.first_frame)
                        frames = list(outcome.replay_frames)
                        status = ControllerStatus(evaluated.value)
                        result = ControllerResult(
                            status=status,
                            terminal_cause=status.value,
                            receipts=(),
                            events=(
                                "BASE_DIRECT_EXECUTION",
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
                        )
                        frames = [frame for attempt in executor.results for frame in attempt.frames]
                        steps = sum(len(attempt.actions) for attempt in executor.results)
                        inference_requests = sum(
                            attempt.inference_requests for attempt in executor.results
                        )
                        evaluator_status = (
                            evaluator.last_status.value
                            if evaluator.last_status is not None
                            else "NOT_CALLED"
                        )
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

                record_accounting = {
                    "base_policy_requests": inference_requests,
                    "initial_proposal_requests": 0,
                    "initial_proposal_status": "NOT_APPLICABLE",
                    "initial_proposal_reason_code": None,
                    "shadow_vlm_requests": 0,
                    "recovery_policy_requests": 0,
                    "shadow_monitor_calls": 0,
                    "shadow_monitor_errors": 0,
                    "shadow_monitor_seconds": 0.0,
                    "shadow_parity_valid": True,
                }
                if MethodArm(args.method_arm) is MethodArm.SHADOW_LOGIV:
                    record_accounting = _shadow_record_accounting(
                        outcome=outcome,
                        runtime=shadow_runtime,
                        exception_text=exception_text,
                        base_policy_requests=inference_requests,
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
                record = LogivEpisodeRecord(
                    schema_version=3,
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
                        certificate.certificate_hash if certificate is not None else None
                    ),
                    initial_graph_hash=graph.graph_hash if graph is not None else None,
                    initial_graph_width=graph.action_layer_width() if graph is not None else None,
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
                    oracle_grounding=MethodArm(args.method_arm) is not MethodArm.BASE,
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
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--collect-recovery-roots", action="store_true")
    parser.add_argument("--shadow-topology-only", action="store_true")
    parser.add_argument(
        "--recovery-root-split",
        choices=("TRAIN", "DEV", "HELDOUT"),
        default="DEV",
    )
    parser.add_argument("--shadow-monitor-interval-steps", default=5, type=int)
    parser.add_argument("--shadow-confirmations", default=3, type=int)
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
    parser.add_argument("--settling-steps", default=10, type=int)
    parser.add_argument("--effect-confirmation-steps", default=5, type=int)
    parser.add_argument("--access-effect-stabilization-steps", default=0, type=int)
    parser.add_argument("--target-divergence-confirmation-steps", default=5, type=int)
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
