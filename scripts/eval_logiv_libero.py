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
from typing import Any, Iterable

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
)
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.logiv.proposal import DEFAULT_FIXTURE, ScriptedProposalProvider
from pi05_libero_repro.logiv.records import (
    EventJournal,
    LogivEpisodeRecord,
    append_episode_record,
    load_episode_records,
)
from pi05_libero_repro.logiv.repair import RepairBounds, RetryPolicy
from pi05_libero_repro.logiv.val import ValWrapper
from pi05_libero_repro.protocol import (
    LIBERO_DUMMY_ACTION,
    derive_episode_seed,
    EpisodeSeededClient,
    prepare_observation,
    run_episode,
    seed_episode_runtime,
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


def _graph_json(graph: CausalGraph) -> dict[str, Any]:
    return {
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
    provider = ScriptedProposalProvider()
    package = provider.propose(task_id, epoch_id=0, goal_mode=GoalMode(args.goal_mode))
    binding = TaskBinding.from_manifest(COVERAGE_MANIFEST, task_id)
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


def _run_config(args: argparse.Namespace, task_ids: tuple[int, ...], episode_indices: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema_version": 4,
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
        "prompt_version": args.prompt_version,
        "prompt_locked": args.prompt_locked,
        "development_only": args.development_only,
        "oracle_grounding": args.oracle_grounding,
        "replan_steps": args.replan_steps,
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
        "proposal_config_sha256": _sha256_file(DEFAULT_FIXTURE),
        "coverage_manifest_sha256": _sha256_file(COVERAGE_MANIFEST),
        "domain_sha256": _sha256_bytes(render_domain_pddl().encode("utf-8")),
    }

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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
                episode_client = EpisodeSeededClient(client, episode_seed=policy_seed)
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
                exception_text = None
                valid = True
                frames: list[np.ndarray] = []
                try:
                    if MethodArm(args.method_arm) is MethodArm.BASE:
                        outcome = run_episode(
                            env,
                            episode_client,
                            initial_state,
                            str(task.language),
                            image_tools,
                            max_steps=args.base_max_steps,
                            wait_steps=args.wait_steps,
                            replan_steps=args.replan_steps,
                        )
                        first_frame_hash = _sha256_array(outcome.first_frame)
                        frames = list(outcome.replay_frames)
                        status = (
                            ControllerStatus.EPISODE_SUCCESS
                            if outcome.success
                            else ControllerStatus.EPISODE_FAIL
                        )
                        result = ControllerResult(
                            status=status,
                            terminal_cause=status.value,
                            receipts=(),
                            events=("BASE_DIRECT_EXECUTION",),
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
                        evaluator_status = status.value
                        steps = outcome.steps
                        inference_requests = outcome.inference_requests
                        initial_snapshot = None
                        package = None
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

                video_path = None
                if frames:
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
                    schema_version=2,
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
    parser.add_argument("--target-divergence-confirmation-steps", default=5, type=int)
    parser.add_argument("--frontier-followup-steps", default=180, type=int)
    parser.add_argument("--frontier-completion-followup-steps", default=0, type=int)
    parser.add_argument("--frontier-completion-recovery-only", action="store_true")
    parser.add_argument("--frontier-recovery-max-consumed-steps", default=0, type=int)
    parser.add_argument("--frontier-fallback-after-steps", default=0, type=int)
    parser.add_argument("--frontier-fallback-followup-steps", default=120, type=int)
    parser.add_argument("--video-fps", default=10, type=int)
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
    parser.add_argument("--prompt-version", default="pi05-subtasks-v1")
    parser.add_argument("--diagnostic-resume", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return evaluate(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
