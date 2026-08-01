from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pi05_libero_repro.logiv.domain import (
    DomainError,
    FixedDomain,
    render_problem_pddl,
    validate_state,
)
from pi05_libero_repro.logiv.model import (
    CandidateSubtask,
    Fact,
    FactSnapshot,
    FrozenGoal,
    GoalMode,
    ObjectDecl,
    Proposal,
    ProposalPackage,
    SignedGoal,
    TaskProblem,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = REPOSITORY_ROOT / "configs/logiv/libero10-scripted-proposals.json"


class ProposalError(ValueError):
    pass


@dataclass(frozen=True)
class ProposalArtifacts:
    proposal_json: Path
    initial_problem: Path
    candidate_plan: Path
    occurrences_json: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fact(value: Sequence[Any]) -> Fact:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ProposalError(f"invalid fact literal: {value!r}")
    try:
        return Fact(value[0], tuple(value[1:]))
    except ValueError as error:
        raise ProposalError(str(error)) from error


def _facts(values: Any) -> frozenset[Fact]:
    if not isinstance(values, list):
        raise ProposalError("fact collection must be a list")
    return frozenset(_fact(value) for value in values)


class ScriptedProposalProvider:
    """No-API proposal source, explicitly labeled as scripted/oracle-grounded."""

    def __init__(self, fixture_path: Path | str = DEFAULT_FIXTURE) -> None:
        self.fixture_path = Path(fixture_path)
        try:
            payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProposalError(f"cannot load proposal fixture: {error}") from error
        if payload.get("schema_version") != 1:
            raise ProposalError("unsupported proposal fixture schema_version")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise ProposalError("proposal fixture tasks must be a list")
        self.provider = str(payload.get("provider", ""))
        self.prompt_version = str(payload.get("prompt_version", ""))
        self.bddl_root = REPOSITORY_ROOT / str(payload.get("bddl_root", ""))
        self._tasks = {item.get("task_id"): item for item in tasks if isinstance(item, dict)}
        if sorted(self._tasks) != list(range(10)) or len(tasks) != 10:
            raise ProposalError("fixture must contain exactly task IDs 0..9")

    def propose(
        self,
        task_id: int,
        epoch_id: int,
        goal_mode: GoalMode = GoalMode.METADATA_ASSISTED,
    ) -> ProposalPackage:
        if goal_mode is not GoalMode.METADATA_ASSISTED:
            raise ProposalError("scripted no-API provider currently supports METADATA_ASSISTED only")
        try:
            item = self._tasks[task_id]
        except KeyError as error:
            raise ProposalError(f"unknown LIBERO-10 task_id: {task_id}") from error
        if epoch_id < 0:
            raise ProposalError("epoch_id must be nonnegative")

        bddl_file = str(item.get("bddl_file", ""))
        bddl_path = self.bddl_root / bddl_file
        try:
            bddl_hash = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ProposalError(f"task {task_id}: cannot read official BDDL: {error}") from error
        if bddl_hash != item.get("bddl_sha256"):
            raise ProposalError(f"task {task_id}: official BDDL hash mismatch")

        try:
            objects = tuple(ObjectDecl(name, type_name) for name, type_name in item["objects"])
            initial_true = _facts(item["initial_true"])
            initial_false = _facts(item["initial_false"])
            goal_positive = _facts(item["official_goal"]["positive"])
            goal_negative = _facts(item["official_goal"].get("negative", []))
        except (KeyError, TypeError, ValueError) as error:
            raise ProposalError(f"invalid task {task_id} fixture: {error}") from error

        problem = TaskProblem(
            name=f"libero10-task-{task_id}",
            objects=objects,
            initial_state=initial_true,
            initial_false=initial_false,
            goal=goal_positive,
            negative_goal=goal_negative,
        )
        try:
            validate_state(problem, problem.initial_state)
        except DomainError as error:
            raise ProposalError(f"task {task_id}: {error}") from error
        domain = FixedDomain()
        candidates = []
        try:
            for rank, raw in enumerate(item["candidate_subtasks"]):
                schema = str(raw["schema"])
                arguments = tuple(str(value) for value in raw["arguments"])
                action = domain.ground(problem, schema, arguments)
                occurrence_id = f"proposal-t{task_id:02d}-o{rank:03d}"
                lineage = f"t{task_id:02d}:{schema}:{'|'.join(arguments)}"
                candidates.append(
                    CandidateSubtask(
                        occurrence_id=occurrence_id,
                        rough_rank=rank,
                        action=action,
                        instruction=str(raw["instruction"]),
                        evidence_source=str(raw["evidence_source"]),
                        lineage_root=lineage,
                    )
                )
        except (KeyError, TypeError, DomainError) as error:
            raise ProposalError(f"task {task_id}: {error}") from error
        if not candidates:
            raise ProposalError(f"task {task_id}: candidate_subtasks must not be empty")

        evidence_payload = {
            "task_id": task_id,
            "epoch_id": epoch_id,
            "initial_true": item["initial_true"],
            "initial_false": item["initial_false"],
            "provider": self.provider,
        }
        snapshot = FactSnapshot(
            epoch_id=epoch_id,
            true_facts=initial_true,
            false_facts=initial_false,
            evidence_hash=_hash_json(evidence_payload),
        )
        signed_goal = SignedGoal(goal_positive, goal_negative)
        goal_id = f"libero10-task-{task_id}-official-goal-v1"
        frozen_goal = FrozenGoal(
            goal_id=goal_id,
            goal_epoch=0,
            literals=signed_goal,
            source="official_bddl_metadata",
        )
        proposal = Proposal(
            task_id=task_id,
            task_name=str(item["task_instruction"]),
            source_bddl=bddl_file,
            source_bddl_sha256=bddl_hash,
            epoch_id=epoch_id,
            goal_mode=goal_mode,
            provider=self.provider,
            prompt_version=self.prompt_version,
            registered_objects=objects,
            initial_snapshot=snapshot,
            candidate_subtasks=tuple(candidates),
            grounded_goal=None,
        )
        return ProposalPackage(proposal=proposal, frozen_goal=frozen_goal, problem=problem)


def _fact_json(fact: Fact) -> dict[str, Any]:
    return {"predicate": fact.predicate, "arguments": list(fact.arguments)}


def _write_exact(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ProposalError(f"refusing to overwrite different artifact: {path}")
    path.write_text(content, encoding="utf-8")


def write_proposal_artifacts(
    directory: Path | str,
    package: ProposalPackage,
    domain: FixedDomain,
) -> ProposalArtifacts:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    proposal = package.proposal
    for candidate in proposal.candidate_subtasks:
        regrounded = domain.ground(
            package.problem, candidate.action.schema, candidate.action.arguments
        )
        if regrounded != candidate.action:
            raise ProposalError(f"grounded action drift: {candidate.occurrence_id}")

    goal_payload = {
        "positive": [_fact_json(fact) for fact in sorted(package.frozen_goal.literals.positive)],
        "negative": [_fact_json(fact) for fact in sorted(package.frozen_goal.literals.negative)],
    }
    proposal_payload = {
        "schema_version": 1,
        "task_id": proposal.task_id,
        "task_name": proposal.task_name,
        "source_bddl": proposal.source_bddl,
        "source_bddl_sha256": proposal.source_bddl_sha256,
        "epoch_id": proposal.epoch_id,
        "goal_mode": proposal.goal_mode.value,
        "provider": proposal.provider,
        "prompt_version": proposal.prompt_version,
        "registered_objects": [
            {"name": item.name, "type": item.type_name} for item in proposal.registered_objects
        ],
        "initial_state": {
            "true": [_fact_json(fact) for fact in sorted(proposal.initial_snapshot.true_facts)],
            "false": [_fact_json(fact) for fact in sorted(proposal.initial_snapshot.false_facts)],
            "evidence_hash": proposal.initial_snapshot.evidence_hash,
            "evidence_source": "scripted-vlm/oracle-grounding",
        },
        "grounded_goal": None,
        "goal_contract": {
            "goal_id": package.frozen_goal.goal_id,
            "goal_epoch": package.frozen_goal.goal_epoch,
            "source": package.frozen_goal.source,
            "goal_hash": _hash_json(goal_payload),
        },
        "candidate_subtasks": [
            {
                "occurrence_id": item.occurrence_id,
                "rough_rank": item.rough_rank,
                "schema": item.action.schema,
                "arguments": list(item.action.arguments),
                "instruction": item.instruction,
                "evidence_source": item.evidence_source,
            }
            for item in proposal.candidate_subtasks
        ],
    }
    occurrences_payload = [
        {
            "occurrence_id": item.occurrence_id,
            "candidate_index": item.rough_rank,
            "schema": item.action.schema,
            "arguments": list(item.action.arguments),
            "lineage_root": item.lineage_root,
            "instruction": item.instruction,
        }
        for item in proposal.candidate_subtasks
    ]
    artifacts = ProposalArtifacts(
        proposal_json=output / "proposal.json",
        initial_problem=output / "initial_problem.pddl",
        candidate_plan=output / "candidate.plan",
        occurrences_json=output / "candidate.occurrences.json",
    )
    _write_exact(
        artifacts.proposal_json,
        json.dumps(proposal_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_exact(artifacts.initial_problem, render_problem_pddl(package.problem))
    _write_exact(
        artifacts.candidate_plan,
        "".join(f"{item.action.pddl()}\n" for item in proposal.candidate_subtasks),
    )
    _write_exact(
        artifacts.occurrences_json,
        json.dumps(occurrences_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return artifacts
