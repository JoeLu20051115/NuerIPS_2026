from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from pi05_libero_repro.logiv.dag import CausalGraph
from pi05_libero_repro.logiv.domain import DomainError, FixedDomain
from pi05_libero_repro.logiv.model import (
    ContextPhase,
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
    parse_pddl_fact,
)
from pi05_libero_repro.logiv.val import (
    PlanCertificate,
    SignedTraceStatus,
    ValidationStatus,
    ValWrapper,
    run_signed_trace,
)


RECOVERY_SEED_DOMAIN = "LOGIV-recovery-policy-seed-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_FIELDS = frozenset(
    {
        "schema_version",
        "capability_id",
        "task_id",
        "event_type",
        "prompt_version",
        "prompt",
        "holding_fact",
        "target_fact",
        "action",
        "active_node_statuses",
        "protected_invariants",
        "effect_confirmation_observations",
        "settling_steps",
        "max_recovery_actions",
        "max_combined_actions",
        "capability_required_successes",
        "capability_total",
        "capability_sha256",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


@dataclass(frozen=True)
class Task5RecoveryCapability:
    schema_version: int
    capability_id: str
    task_id: int
    event_type: str
    prompt_version: str
    prompt: str
    holding_fact: Fact
    target_fact: Fact
    action: str
    active_node_statuses: frozenset[str]
    protected_invariants: frozenset[Fact]
    effect_confirmation_observations: int
    settling_steps: int
    max_recovery_actions: int
    max_combined_actions: int
    capability_required_successes: int
    capability_total: int
    capability_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "prompt_version": self.prompt_version,
            "prompt": self.prompt,
            "holding_fact": self.holding_fact.pddl(),
            "target_fact": self.target_fact.pddl(),
            "action": self.action,
            "active_node_statuses": sorted(self.active_node_statuses),
            "protected_invariants": [
                fact.pddl() for fact in sorted(self.protected_invariants)
            ],
            "effect_confirmation_observations": self.effect_confirmation_observations,
            "settling_steps": self.settling_steps,
            "max_recovery_actions": self.max_recovery_actions,
            "max_combined_actions": self.max_combined_actions,
            "capability_required_successes": self.capability_required_successes,
            "capability_total": self.capability_total,
        }

    def recompute_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._payload()).encode("utf-8")
        ).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate capability field: {key}")
        result[key] = value
    return result


def load_task5_recovery_capability(path: Path) -> Task5RecoveryCapability:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load recovery capability: {error}") from error
    if not isinstance(payload, dict) or frozenset(payload) != _CAPABILITY_FIELDS:
        raise ValueError("recovery capability fields mismatch")

    integer_fields = (
        "schema_version",
        "task_id",
        "effect_confirmation_observations",
        "settling_steps",
        "max_recovery_actions",
        "max_combined_actions",
        "capability_required_successes",
        "capability_total",
    )
    string_fields = (
        "capability_id",
        "event_type",
        "prompt_version",
        "prompt",
        "holding_fact",
        "target_fact",
        "action",
        "capability_sha256",
    )
    if any(type(payload[name]) is not int for name in integer_fields):
        raise ValueError("recovery capability integer field is invalid")
    if any(not isinstance(payload[name], str) or not payload[name] for name in string_fields):
        raise ValueError("recovery capability string field is invalid")
    if payload["schema_version"] != 1 or any(
        payload[name] <= 0 for name in integer_fields if name not in {"schema_version", "task_id"}
    ):
        raise ValueError("recovery capability numeric domain is invalid")
    if payload["task_id"] != 5:
        raise ValueError("recovery capability task_id must be 5")
    statuses = payload["active_node_statuses"]
    invariants = payload["protected_invariants"]
    if (
        not isinstance(statuses, list)
        or not statuses
        or any(not isinstance(item, str) or not item for item in statuses)
        or len(statuses) != len(set(statuses))
    ):
        raise ValueError("recovery capability active node statuses are invalid")
    if (
        not isinstance(invariants, list)
        or not invariants
        or any(not isinstance(item, str) for item in invariants)
        or len(invariants) != len(set(invariants))
    ):
        raise ValueError("recovery capability protected invariants are invalid")
    if not _valid_sha256(payload["capability_sha256"]):
        raise ValueError("recovery capability SHA-256 is invalid")
    try:
        holding_fact = parse_pddl_fact(payload["holding_fact"])
        target_fact = parse_pddl_fact(payload["target_fact"])
        protected_invariants = frozenset(parse_pddl_fact(item) for item in invariants)
    except ValueError as error:
        raise ValueError(f"recovery capability fact is invalid: {error}") from error
    capability = Task5RecoveryCapability(
        schema_version=payload["schema_version"],
        capability_id=payload["capability_id"],
        task_id=payload["task_id"],
        event_type=payload["event_type"],
        prompt_version=payload["prompt_version"],
        prompt=payload["prompt"],
        holding_fact=holding_fact,
        target_fact=target_fact,
        action=payload["action"],
        active_node_statuses=frozenset(statuses),
        protected_invariants=protected_invariants,
        effect_confirmation_observations=payload["effect_confirmation_observations"],
        settling_steps=payload["settling_steps"],
        max_recovery_actions=payload["max_recovery_actions"],
        max_combined_actions=payload["max_combined_actions"],
        capability_required_successes=payload["capability_required_successes"],
        capability_total=payload["capability_total"],
        capability_sha256=payload["capability_sha256"],
    )
    if capability.capability_sha256 != capability.recompute_sha256():
        raise ValueError("recovery capability self-hash mismatch")
    return capability


def derive_recovery_policy_seed(
    master_seed: int, task_id: int, episode_idx: int, event_id: str
) -> int:
    if not (0 <= master_seed < 2**32 and 0 <= task_id <= 9 and 0 <= episode_idx < 50):
        raise ValueError("recovery seed inputs are outside the frozen evaluation domain")
    if not event_id:
        raise ValueError("event_id must be nonempty")
    payload = f"{RECOVERY_SEED_DOMAIN}:{master_seed}:{task_id}:{episode_idx}:{event_id}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True)
class TerminalAssessment:
    eligible: bool
    reason: str
    event_id: str | None
    event_type: str | None
    option_action_cap: int
    snapshot_sha256: str | None
    graph_hash: str | None
    certificate_hash: str | None
    monitor_contract_sha256: str | None
    protected_true_facts: frozenset[Fact]


def assess_task5_terminal(
    capability: Task5RecoveryCapability,
    *,
    task_id: int,
    episode_id: str,
    base_success: bool,
    native_terminal_status: str,
    base_policy_steps: int,
    snapshot: FactSnapshot | None,
    problem: TaskProblem | None,
    graph: CausalGraph | None,
    certificate: PlanCertificate | None,
    monitor_contract_sha256: str | None,
    certificate_state: str | None,
    place_node_action: GroundAction | None,
    place_node_status: str | None,
) -> TerminalAssessment:
    def denied(reason: str) -> TerminalAssessment:
        return TerminalAssessment(
            eligible=False,
            reason=reason,
            event_id=None,
            event_type=None,
            option_action_cap=0,
            snapshot_sha256=None if snapshot is None else snapshot.evidence_hash,
            graph_hash=None if graph is None else graph.graph_hash,
            certificate_hash=None if certificate is None else certificate.certificate_hash,
            monitor_contract_sha256=monitor_contract_sha256,
            protected_true_facts=frozenset(),
        )

    if base_success:
        return denied("BASE_SUCCEEDED")
    if task_id != capability.task_id:
        return denied("UNSUPPORTED_TASK")
    if native_terminal_status != "EPISODE_FAIL":
        return denied("NATIVE_TERMINAL_NOT_FAILED")
    if problem is None or graph is None or certificate is None:
        return denied("MISSING_GRAPH_OR_CERTIFICATE")
    if certificate_state != "CURRENT":
        return denied("CERTIFICATE_NOT_CURRENT")
    if graph.certificate_hash != certificate.certificate_hash:
        return denied("GRAPH_CERTIFICATE_HASH_MISMATCH")
    if not _valid_sha256(monitor_contract_sha256):
        return denied("MONITOR_CONTRACT_HASH_INVALID")
    if snapshot is None or snapshot.fact_universe is None:
        return denied("STRICT_AUDITED_SNAPSHOT_REQUIRED")
    location_facts = frozenset(
        fact
        for fact in snapshot.fact_universe
        if (fact.predicate == "holding" and fact.arguments == ("black_book_1",))
        or (fact.predicate == "at" and fact.arguments[:1] == ("black_book_1",))
    )
    if snapshot.unknown(location_facts) or len(snapshot.true_facts & location_facts) != 1:
        return denied("EXACTLY_ONE_LOCATION_NOT_PROVEN")
    if capability.holding_fact not in snapshot.true_facts:
        return denied("HOLDING_NOT_EXPLICITLY_TRUE")
    if capability.target_fact not in snapshot.false_facts:
        return denied("TARGET_NOT_EXPLICITLY_FALSE")
    try:
        expected_action = FixedDomain().ground(
            problem,
            "place-held-in",
            (
                "black_book_1",
                "desk_caddy_1_back_contain_region",
                "desk_caddy_1_access",
            ),
        )
    except DomainError:
        return denied("PLACE_NODE_ACTION_MISMATCH")
    if place_node_action != expected_action:
        return denied("PLACE_NODE_ACTION_MISMATCH")
    if place_node_status not in capability.active_node_statuses:
        return denied("PLACE_NODE_NOT_ACTIVE")
    if not capability.protected_invariants <= snapshot.true_facts:
        return denied("PROTECTED_INVARIANT_NOT_TRUE")
    remaining = capability.max_combined_actions - base_policy_steps
    if remaining <= 0:
        return denied("ACTION_BUDGET_EXHAUSTED")
    event_payload = (
        f"{episode_id}:{snapshot.evidence_hash}:{graph.graph_hash}:"
        f"{certificate.certificate_hash}:{monitor_contract_sha256}:"
        f"{capability.event_type}"
    )
    event_id = hashlib.sha256(
        b"LOGIV_TERMINAL_DEVIATION_V1\0" + event_payload.encode("utf-8")
    ).hexdigest()
    return TerminalAssessment(
        eligible=True,
        reason="ELIGIBLE",
        event_id=event_id,
        event_type=capability.event_type,
        option_action_cap=min(capability.max_recovery_actions, remaining),
        snapshot_sha256=snapshot.evidence_hash,
        graph_hash=graph.graph_hash,
        certificate_hash=certificate.certificate_hash,
        monitor_contract_sha256=monitor_contract_sha256,
        protected_true_facts=frozenset(
            capability.protected_invariants | (problem.goal & snapshot.true_facts)
        ),
    )


def _plan_sha256(plan: Sequence[GroundAction]) -> str:
    return _domain_hash(
        b"LOGIV_TASK5_RECOVERY_PLAN_V1\0", [action.pddl() for action in plan]
    )


@dataclass(frozen=True)
class RecoveryPermit:
    granted: bool
    reason: str
    permit_sha256: str
    event_id: str
    action_cap: int
    plan: Sequence[GroundAction]
    plan_sha256: str | None
    certificate: PlanCertificate | None
    capability_sha256: str
    recovery_checkpoint_sha256: str

    def recompute_sha256(self) -> str:
        return _domain_hash(
            b"LOGIV_TASK5_RECOVERY_PERMIT_V1\0",
            {
                "granted": self.granted,
                "reason": self.reason,
                "event_id": self.event_id,
                "action_cap": self.action_cap,
                "plan": [action.pddl() for action in self.plan],
                "plan_sha256": self.plan_sha256,
                "certificate_hash": None
                if self.certificate is None
                else self.certificate.certificate_hash,
                "capability_sha256": self.capability_sha256,
                "recovery_checkpoint_sha256": self.recovery_checkpoint_sha256,
            },
        )


def certify_task5_recovery(
    capability: Task5RecoveryCapability,
    *,
    assessment: TerminalAssessment,
    snapshot: FactSnapshot,
    problem: TaskProblem,
    graph: CausalGraph,
    certificate: PlanCertificate,
    candidate_plan: Sequence[GroundAction],
    val_wrapper: ValWrapper,
    recovery_checkpoint_sha256: str,
) -> RecoveryPermit:
    plan = tuple(candidate_plan)

    def denied(reason: str) -> RecoveryPermit:
        provisional = RecoveryPermit(
            granted=False,
            reason=reason,
            permit_sha256="",
            event_id=assessment.event_id or "",
            action_cap=0,
            plan=plan,
            plan_sha256=None if not plan else _plan_sha256(plan),
            certificate=None,
            capability_sha256=capability.capability_sha256,
            recovery_checkpoint_sha256=recovery_checkpoint_sha256,
        )
        return replace(provisional, permit_sha256=provisional.recompute_sha256())

    if not assessment.eligible or assessment.event_id is None:
        return denied("ASSESSMENT_NOT_ELIGIBLE")
    if capability.capability_sha256 != capability.recompute_sha256():
        return denied("CAPABILITY_HASH_MISMATCH")
    if snapshot.evidence_hash != assessment.snapshot_sha256:
        return denied("SNAPSHOT_HASH_CHANGED")
    if graph.graph_hash != assessment.graph_hash:
        return denied("GRAPH_HASH_CHANGED")
    if certificate.certificate_hash != assessment.certificate_hash:
        return denied("CERTIFICATE_HASH_CHANGED")
    if graph.certificate_hash != certificate.certificate_hash:
        return denied("GRAPH_CERTIFICATE_HASH_MISMATCH")
    if not _valid_sha256(recovery_checkpoint_sha256):
        return denied("RECOVERY_CHECKPOINT_HASH_INVALID")
    if len(plan) != 1:
        return denied("PLAN_ACTION_COUNT_NOT_ONE")
    try:
        expected_action = FixedDomain().ground(
            problem,
            "place-held-in",
            (
                "black_book_1",
                "desk_caddy_1_back_contain_region",
                "desk_caddy_1_access",
            ),
        )
    except DomainError:
        return denied("PLAN_ACTION_MISMATCH")
    if plan[0] != expected_action:
        return denied("PLAN_ACTION_MISMATCH")
    current_problem = replace(
        problem,
        initial_state=snapshot.true_facts,
        initial_false=snapshot.false_facts,
    )
    trace = run_signed_trace(current_problem, plan)
    if trace.status is not SignedTraceStatus.VALID:
        return denied("SIGNED_STATE_INVALID")
    context = replace(
        certificate.context,
        phase=ContextPhase.RECOVERY_VAL,
        request_id=f"task5-recovery-{assessment.event_id}",
        epoch_id=snapshot.epoch_id,
        graph_version=graph.graph_version,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=certificate.certificate_hash,
        safety_epoch=None,
    )
    sidecar = _canonical_json(
        [
            {
                "occurrence_id": "task5-terminal-recovery-0",
                "schema": plan[0].schema,
                "arguments": list(plan[0].arguments),
            }
        ]
    ).encode("utf-8")
    validation = val_wrapper.validate(current_problem, plan, sidecar, context)
    if validation.status is ValidationStatus.VALIDATION_ERROR:
        return denied("VAL_VALIDATION_ERROR")
    if validation.status is not ValidationStatus.VALID or validation.certificate is None:
        return denied("VAL_PLAN_INVALID")
    provisional = RecoveryPermit(
        granted=True,
        reason="GRANTED",
        permit_sha256="",
        event_id=assessment.event_id,
        action_cap=assessment.option_action_cap,
        plan=plan,
        plan_sha256=_plan_sha256(plan),
        certificate=validation.certificate,
        capability_sha256=capability.capability_sha256,
        recovery_checkpoint_sha256=recovery_checkpoint_sha256,
    )
    return replace(provisional, permit_sha256=provisional.recompute_sha256())


@dataclass(frozen=True)
class RecoveryCommit:
    committed: bool
    reason: str
    commit_sha256: str
    base_actions: int
    recovery_actions: int
    combined_actions: int


def verify_task5_recovery_commit(
    capability: Task5RecoveryCapability,
    *,
    permit: RecoveryPermit,
    post_snapshot: FactSnapshot,
    handoff_goal_facts: frozenset[Fact],
    event_id: str,
    permit_sha256: str,
    plan_sha256: str | None,
    recovery_checkpoint_sha256: str,
    capability_sha256: str,
    base_actions: int,
    recovery_actions: int,
    native_evaluator_success: bool,
) -> RecoveryCommit:
    combined_actions = base_actions + recovery_actions

    def denied(reason: str) -> RecoveryCommit:
        payload = {
            "committed": False,
            "reason": reason,
            "event_id": event_id,
            "permit_sha256": permit_sha256,
            "plan_sha256": plan_sha256,
            "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
            "capability_sha256": capability_sha256,
            "post_snapshot_sha256": post_snapshot.evidence_hash,
            "base_actions": base_actions,
            "recovery_actions": recovery_actions,
            "combined_actions": combined_actions,
            "native_evaluator_success": native_evaluator_success,
        }
        return RecoveryCommit(
            committed=False,
            reason=reason,
            commit_sha256=_domain_hash(b"LOGIV_TASK5_RECOVERY_COMMIT_V1\0", payload),
            base_actions=base_actions,
            recovery_actions=recovery_actions,
            combined_actions=combined_actions,
        )

    if not permit.granted:
        return denied("PERMIT_NOT_GRANTED")
    if event_id != permit.event_id:
        return denied("EVENT_ID_CHANGED")
    if permit_sha256 != permit.permit_sha256 or permit.recompute_sha256() != permit.permit_sha256:
        return denied("PERMIT_HASH_CHANGED")
    if (
        plan_sha256 is None
        or plan_sha256 != permit.plan_sha256
        or _plan_sha256(permit.plan) != permit.plan_sha256
    ):
        return denied("PLAN_HASH_CHANGED")
    if recovery_checkpoint_sha256 != permit.recovery_checkpoint_sha256:
        return denied("RECOVERY_CHECKPOINT_HASH_CHANGED")
    if (
        capability_sha256 != permit.capability_sha256
        or capability.capability_sha256 != permit.capability_sha256
        or capability.recompute_sha256() != capability.capability_sha256
    ):
        return denied("CAPABILITY_HASH_CHANGED")
    if post_snapshot.fact_universe is None:
        return denied("STRICT_AUDITED_POST_SNAPSHOT_REQUIRED")
    required = frozenset(
        {capability.target_fact} | handoff_goal_facts | capability.protected_invariants
    )
    if post_snapshot.unknown(required):
        return denied("REQUIRED_LITERAL_UNKNOWN")
    if capability.target_fact not in post_snapshot.true_facts:
        return denied("TARGET_NOT_EXPLICITLY_TRUE")
    if not handoff_goal_facts <= post_snapshot.true_facts:
        return denied("HANDOFF_GOAL_NOT_TRUE")
    if not capability.protected_invariants <= post_snapshot.true_facts:
        return denied("PROTECTED_INVARIANT_NOT_TRUE")
    if base_actions < 0 or recovery_actions < 0:
        return denied("ACTION_COUNT_INVALID")
    if recovery_actions > permit.action_cap:
        return denied("RECOVERY_ACTION_CAP_EXCEEDED")
    if combined_actions > capability.max_combined_actions:
        return denied("COMBINED_ACTION_BUDGET_EXCEEDED")
    if not native_evaluator_success:
        return denied("NATIVE_EVALUATOR_FAILED")
    payload = {
        "committed": True,
        "reason": "COMMITTED",
        "event_id": event_id,
        "permit_sha256": permit_sha256,
        "plan_sha256": plan_sha256,
        "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
        "capability_sha256": capability_sha256,
        "post_snapshot_sha256": post_snapshot.evidence_hash,
        "base_actions": base_actions,
        "recovery_actions": recovery_actions,
        "combined_actions": combined_actions,
        "native_evaluator_success": True,
    }
    return RecoveryCommit(
        committed=True,
        reason="COMMITTED",
        commit_sha256=_domain_hash(b"LOGIV_TASK5_RECOVERY_COMMIT_V1\0", payload),
        base_actions=base_actions,
        recovery_actions=recovery_actions,
        combined_actions=combined_actions,
    )
