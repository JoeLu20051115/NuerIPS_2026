from __future__ import annotations

from dataclasses import replace
from itertools import product
import json
from typing import Any, Callable, FrozenSet, Mapping, Sequence

import numpy as np

from pi05_libero_repro.logiv.domain import (
    DomainError,
    FixedDomain,
    render_domain_pddl,
    render_problem_pddl,
    validate_state,
)
from pi05_libero_repro.logiv.gpt4o import Gpt4oClient
from pi05_libero_repro.logiv.model import (
    CandidateSubtask,
    ContextEnvelope,
    GoalMode,
    GroundAction,
    ProposalPackage,
    TaskProblem,
)
from pi05_libero_repro.logiv.repair import (
    CausalSlice,
    RepairBounds,
    RepairOperator,
    RepairResult,
    RepairStatus,
    RetryLedger,
    RetryPolicy,
)
from pi05_libero_repro.logiv.val import ValidationStatus, ValWrapper


class Gpt4oPlanningError(RuntimeError):
    """GPT-4o returned an unusable symbolic plan."""


def _action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "schema": {"type": "string"},
                        "arguments": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "instruction": {"type": "string"},
                    },
                    "required": ["schema", "arguments", "instruction"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def _oriented_images(observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        main = np.asarray(observation["agentview_image"])[::-1, ::-1]
        wrist = np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1]
    except (KeyError, TypeError, ValueError) as error:
        raise Gpt4oPlanningError("observation is missing the two RGB cameras") from error
    images = []
    for image in (main, wrist):
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise Gpt4oPlanningError("camera image must have shape HxWx3")
        images.append(image)
    return images[0], images[1]


def _catalog(
    problem: TaskProblem, allowed_schemas: FrozenSet[str]
) -> tuple[GroundAction, ...]:
    domain = FixedDomain()
    unknown = allowed_schemas - domain.schemas.keys()
    if unknown:
        raise Gpt4oPlanningError(f"unknown allowed schemas: {sorted(unknown)}")
    names = tuple(sorted(problem.object_types))
    actions: set[GroundAction] = set()
    for schema_name in sorted(allowed_schemas):
        schema = domain.schemas[schema_name]
        for arguments in product(names, repeat=len(schema.parameters)):
            try:
                actions.add(domain.ground(problem, schema_name, arguments))
            except DomainError:
                continue
    return tuple(sorted(actions, key=lambda action: action.pddl()))


def _parse_actions(
    response: Mapping[str, Any],
    *,
    problem: TaskProblem,
    allowed_schemas: FrozenSet[str],
) -> tuple[tuple[GroundAction, str], ...]:
    if set(response) != {"actions"} or not isinstance(response["actions"], list):
        raise Gpt4oPlanningError("plan response fields mismatch")
    if not response["actions"]:
        raise Gpt4oPlanningError("plan must not be empty")
    domain = FixedDomain()
    parsed: list[tuple[GroundAction, str]] = []
    seen = set()
    for record in response["actions"]:
        if not isinstance(record, Mapping) or set(record) != {
            "schema",
            "arguments",
            "instruction",
        }:
            raise Gpt4oPlanningError("action record fields mismatch")
        schema = record["schema"]
        arguments = record["arguments"]
        instruction = record["instruction"]
        if (
            not isinstance(schema, str)
            or schema not in allowed_schemas
            or not isinstance(arguments, list)
            or not all(isinstance(item, str) for item in arguments)
            or not isinstance(instruction, str)
            or not instruction.strip()
        ):
            raise Gpt4oPlanningError("action record is invalid")
        try:
            action = domain.ground(problem, schema, tuple(arguments))
        except DomainError as error:
            raise Gpt4oPlanningError(str(error)) from error
        if action.retry_key in seen:
            raise Gpt4oPlanningError("duplicate action record")
        seen.add(action.retry_key)
        parsed.append((action, instruction.strip()))
    return tuple(parsed)


def _render_catalog(actions: Sequence[GroundAction]) -> str:
    return "\n".join(action.pddl() for action in actions)


class Gpt4oProposalProvider:
    """Initial high-level PDDL plan proposal with immutable metadata/goal."""

    requires_observation = True
    provider = "gpt-4o-logiv"
    prompt_version = "gpt4o-initial-plan-v1"

    def __init__(
        self,
        *,
        client: Gpt4oClient,
        scaffold_provider: Any,
        allowed_schemas: FrozenSet[str],
    ) -> None:
        self.client = client
        self.scaffold_provider = scaffold_provider
        self.allowed_schemas = allowed_schemas

    def propose(
        self,
        task_id: int,
        epoch_id: int,
        goal_mode: GoalMode = GoalMode.METADATA_ASSISTED,
        *,
        observation: Mapping[str, Any],
    ) -> ProposalPackage:
        scaffold = self.scaffold_provider.propose(task_id, epoch_id, goal_mode)
        legal = _catalog(scaffold.problem, self.allowed_schemas)
        response = self.client.complete_json(
            purpose="initial_plan",
            system=(
                "You are LOGIV's PDDL plan proposer. Use only listed grounded "
                "actions. Never change objects, state, goal, schemas, or graph edges."
            ),
            text=(
                f"Task instruction: {scaffold.proposal.task_name}\n\n"
                f"Fixed Domain:\n{render_domain_pddl()}\n"
                f"Current Problem:\n{render_problem_pddl(scaffold.problem)}\n"
                f"Legal grounded actions:\n{_render_catalog(legal)}"
            ),
            images=_oriented_images(observation),
            schema_name="logiv_initial_plan",
            schema=_action_schema(),
        )
        parsed = _parse_actions(
            response,
            problem=scaffold.problem,
            allowed_schemas=self.allowed_schemas,
        )
        candidates = tuple(
            CandidateSubtask(
                occurrence_id=f"gpt4o-t{task_id:02d}-o{rank:03d}",
                rough_rank=rank,
                action=action,
                instruction=instruction,
                evidence_source="gpt-4o-initial-plan",
                lineage_root=(
                    f"gpt4o:t{task_id:02d}:{action.schema}:"
                    + "|".join(action.arguments)
                ),
            )
            for rank, (action, instruction) in enumerate(parsed)
        )
        proposal = replace(
            scaffold.proposal,
            provider=self.provider,
            prompt_version=self.prompt_version,
            candidate_subtasks=candidates,
        )
        return replace(scaffold, proposal=proposal)


def _edit_distance(
    left: Sequence[tuple[str, tuple[str, ...]]],
    right: Sequence[tuple[str, tuple[str, ...]]],
) -> int:
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


class Gpt4oRepairOperator:
    """Model-proposed causal-slice edits with programmatic merge and full VAL."""

    def __init__(
        self,
        *,
        client: Gpt4oClient,
        val_wrapper: ValWrapper,
        allowed_schemas: FrozenSet[str],
        bounds: RepairBounds,
        retry_policy: RetryPolicy | None = None,
        decompose_macro_sources: FrozenSet[str] = frozenset(),
        observation_reader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.val_wrapper = val_wrapper
        self.allowed_schemas = allowed_schemas
        self.bounds = bounds
        self.retry_policy = retry_policy or RetryPolicy(max_retries_per_lineage=1)
        self.observation_reader = observation_reader
        self._helper = RepairOperator(
            val_wrapper,
            allowed_schemas=allowed_schemas,
            bounds=bounds,
            retry_policy=self.retry_policy,
            decompose_macro_sources=decompose_macro_sources,
        )

    @staticmethod
    def _merge(
        remaining_plan: Sequence[GroundAction],
        replacement: Sequence[GroundAction],
        causal_slice: CausalSlice | None,
    ) -> tuple[GroundAction, ...]:
        if causal_slice is None:
            return tuple(replacement)
        affected = set(causal_slice.action_signatures)
        indices = [
            index
            for index, action in enumerate(remaining_plan)
            if action.retry_key in affected
        ]
        if not indices:
            raise Gpt4oPlanningError("causal slice does not intersect remaining plan")
        first = indices[0]
        merged: list[GroundAction] = []
        for index, action in enumerate(remaining_plan):
            if index == first:
                merged.extend(replacement)
            if index not in indices:
                merged.append(action)
        return tuple(merged)

    def repair(
        self,
        problem: TaskProblem,
        remaining_plan: Sequence[GroundAction],
        *,
        context: ContextEnvelope,
        retry_ledger: RetryLedger | None = None,
        retry_policy: RetryPolicy | None = None,
        forbidden_retry_keys: FrozenSet[str] = frozenset(),
        lineage_roots: Mapping[tuple[str, tuple[str, ...]], str] | None = None,
        causal_slice: CausalSlice | None = None,
        val_call_guard: Callable[[], bool] | None = None,
    ) -> RepairResult:
        ledger = retry_ledger or RetryLedger()
        policy = retry_policy or self.retry_policy
        lineage_roots = lineage_roots or {}
        try:
            validate_state(problem, problem.initial_state)
            legal = _catalog(problem, self.allowed_schemas)
        except (DomainError, Gpt4oPlanningError) as error:
            return RepairResult(RepairStatus.VALIDATION_ERROR, reason=str(error))
        affected = (
            set(causal_slice.action_signatures)
            if causal_slice is not None
            else {action.retry_key for action in remaining_plan}
        )
        protected = [
            action.pddl() for action in remaining_plan if action.retry_key not in affected
        ]
        feedback = "none"
        val_calls = 0
        explored = 0
        max_candidates = self.bounds.max_candidates
        if causal_slice is None and remaining_plan and self.bounds.max_val_calls > 0:
            if policy.plan_allowed(remaining_plan, ledger, forbidden_retry_keys):
                if val_call_guard is not None and not val_call_guard():
                    return RepairResult(
                        RepairStatus.BUDGET_EXHAUSTED,
                        reason="episode-global VAL budget exhausted",
                    )
                val_calls += 1
                existing = tuple(remaining_plan)
                sidecar = self._helper._sidecar(
                    existing, context, ledger, lineage_roots
                )
                candidate_context = replace(
                    context,
                    request_id=f"{context.request_id}-gpt4o-candidate-{val_calls}",
                    request_generation=0,
                )
                validation = self.val_wrapper.validate(
                    problem,
                    existing,
                    sidecar,
                    candidate_context,
                    forbidden_retry_keys=frozenset(forbidden_retry_keys),
                    retry_ledger_version=ledger.version,
                )
                if validation.status is ValidationStatus.VALID:
                    return RepairResult(
                        RepairStatus.CERTIFIED,
                        plan=existing,
                        occurrence_sidecar=sidecar,
                        certificate=validation.certificate,
                        explored_candidates=0,
                        val_calls=val_calls,
                    )
                if validation.status is ValidationStatus.VALIDATION_ERROR:
                    return RepairResult(
                        RepairStatus.VALIDATION_ERROR,
                        explored_candidates=0,
                        val_calls=val_calls,
                        reason=validation.reason or "VAL validation error",
                    )
                feedback = validation.reason or validation.stderr or "VAL rejected plan"
        while explored < max_candidates and val_calls < self.bounds.max_val_calls:
            explored += 1
            observation = self.observation_reader() if self.observation_reader else None
            images = _oriented_images(observation) if observation is not None else ()
            response = self.client.complete_json(
                purpose="local_repair",
                system=(
                    "You are LOGIV's local PDDL repair proposer. Return only a "
                    "replacement for the affected slice. Do not edit protected "
                    "actions, objects, goals, schemas, DAG edges, or controls."
                ),
                text=(
                    f"Current Problem:\n{render_problem_pddl(problem)}\n"
                    "Remaining plan:\n"
                    + "\n".join(action.pddl() for action in remaining_plan)
                    + "\nAffected action signatures:\n"
                    + json.dumps(sorted((schema, list(args)) for schema, args in affected))
                    + "\nProtected actions (program code will preserve these):\n"
                    + "\n".join(protected)
                    + f"\nPrior VAL feedback: {feedback}\n"
                    + f"Legal grounded actions:\n{_render_catalog(legal)}"
                ),
                images=images,
                schema_name="logiv_local_repair",
                schema=_action_schema(),
            )
            try:
                replacement = tuple(
                    action
                    for action, _ in _parse_actions(
                        response,
                        problem=problem,
                        allowed_schemas=self.allowed_schemas,
                    )
                )
                merged = self._merge(remaining_plan, replacement, causal_slice)
            except Gpt4oPlanningError as error:
                feedback = str(error)
                continue
            edits = _edit_distance(
                tuple(action.retry_key for action in merged),
                tuple(action.retry_key for action in remaining_plan),
            )
            if edits > self.bounds.max_edits:
                feedback = f"edit budget exceeded: {edits}>{self.bounds.max_edits}"
                continue
            if not policy.plan_allowed(merged, ledger, forbidden_retry_keys):
                feedback = "retry policy rejected the merged plan"
                continue
            if val_call_guard is not None and not val_call_guard():
                return RepairResult(
                    RepairStatus.BUDGET_EXHAUSTED,
                    explored_candidates=explored,
                    val_calls=val_calls,
                    reason="episode-global VAL budget exhausted",
                )
            val_calls += 1
            sidecar = self._helper._sidecar(merged, context, ledger, lineage_roots)
            candidate_context = replace(
                context,
                request_id=f"{context.request_id}-gpt4o-candidate-{val_calls}",
                request_generation=0,
            )
            validation = self.val_wrapper.validate(
                problem,
                merged,
                sidecar,
                candidate_context,
                forbidden_retry_keys=frozenset(forbidden_retry_keys),
                retry_ledger_version=ledger.version,
            )
            if validation.status is ValidationStatus.VALID:
                return RepairResult(
                    RepairStatus.CERTIFIED,
                    plan=merged,
                    occurrence_sidecar=sidecar,
                    certificate=validation.certificate,
                    explored_candidates=explored,
                    val_calls=val_calls,
                )
            if validation.status is ValidationStatus.VALIDATION_ERROR:
                return RepairResult(
                    RepairStatus.VALIDATION_ERROR,
                    explored_candidates=explored,
                    val_calls=val_calls,
                    reason=validation.reason or "VAL validation error",
                )
            feedback = validation.reason or validation.stderr or "VAL rejected plan"
        return RepairResult(
            RepairStatus.NO_CERTIFIED_REPAIR_WITHIN_BUDGET,
            explored_candidates=explored,
            val_calls=val_calls,
            reason=feedback,
        )
