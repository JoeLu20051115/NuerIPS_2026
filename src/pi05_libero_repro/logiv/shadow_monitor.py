from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from pi05_libero_repro.logiv.dag import (
    CausalGraph,
    CompilerError,
    NodeKind,
    SignedLiteral,
    validate_graph,
)
from pi05_libero_repro.logiv.domain import _is_subtype
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    GroundAction,
    TaskProblem,
    TruthValue,
    parse_pddl_fact,
)
from pi05_libero_repro.protocol import ShadowStepContext


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRONG_EVIDENCE = frozenset(
    {
        "GOAL_REGRESSION",
        "ATTEMPTED_EFFECT_TIMEOUT",
        "ABNORMAL_TRANSFER_AFTER_MANIPULATION",
    }
)
_WEAK_EVIDENCE = frozenset({"PROGRESS_TIMEOUT"})
_TRANSIENT_EVIDENCE = frozenset(
    {"TRANSIENT_HOLDING", "TRANSIENT_RELEASE", "NORMAL_PHASE"}
)
_EVIDENCE_KINDS = _STRONG_EVIDENCE | _WEAK_EVIDENCE | _TRANSIENT_EVIDENCE


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(
        domain + b"\0" + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _optional_finite(name: str, value: float | None, *, nonnegative: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite or null")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative or null")


def _truth(snapshot: FactSnapshot, literal: SignedLiteral) -> bool:
    value = snapshot.truth(literal.fact)
    return value is (TruthValue.TRUE if literal.positive else TruthValue.FALSE)


def _fact_object(fact: Fact) -> str:
    return fact.arguments[0] if fact.arguments else "__task__"


@dataclass(frozen=True)
class ShadowPlanContext:
    problem: TaskProblem
    plan: tuple[GroundAction, ...]
    graph: CausalGraph
    certificate_hash: str

    def __post_init__(self) -> None:
        try:
            validate_graph(self.graph)
        except CompilerError as error:
            raise ValueError(f"invalid shadow graph: {error}") from error
        if self.graph.certificate_hash != self.certificate_hash:
            raise ValueError("shadow plan certificate hash mismatch")
        if len(self.plan) != len(self.graph.canonical_agenda):
            raise ValueError("shadow plan length does not match graph agenda")
        node_map = self.graph.node_map
        agenda_actions = tuple(
            node_map[node_id].action for node_id in self.graph.canonical_agenda
        )
        if any(action is None for action in agenda_actions) or agenda_actions != self.plan:
            raise ValueError("shadow plan actions do not match graph agenda")


class ReservedFactEventRuleId(str, Enum):
    GOAL_REGRESSION = "__goal_regression__"
    PROGRESS_TIMEOUT = "__progress_timeout__"


@dataclass(frozen=True)
class ActionEventRule:
    rule_id: str
    object_id: str
    attempt_kind: str
    attempted_effect: str
    source_region: str | None
    destination_region: str | None
    gripper_close_threshold: float
    gripper_open_threshold: float
    contact_min_count: int
    motion_correlation_min: float
    region_distance_max: float
    effect_due_after_policy_steps: int
    evidence_ttl_policy_steps: int
    manipulation_attribution_ttl_policy_steps: int

    def __post_init__(self) -> None:
        if not self.rule_id or self.rule_id.startswith("__"):
            raise ValueError("action event rule ID is empty or reserved")
        if not self.object_id or self.attempt_kind not in {"grasp", "place"}:
            raise ValueError("action event rule identity is invalid")
        fact = parse_pddl_fact(self.attempted_effect)
        if not fact.arguments or fact.arguments[0] != self.object_id:
            raise ValueError("attempted effect object does not match its rule")
        for name, value in (
            ("source_region", self.source_region),
            ("destination_region", self.destination_region),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be nonempty or null")
        for name, value in (
            ("gripper_close_threshold", self.gripper_close_threshold),
            ("gripper_open_threshold", self.gripper_open_threshold),
            ("motion_correlation_min", self.motion_correlation_min),
            ("region_distance_max", self.region_distance_max),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if self.gripper_close_threshold >= self.gripper_open_threshold:
            raise ValueError("gripper thresholds are unordered")
        if not -1.0 <= self.motion_correlation_min <= 1.0:
            raise ValueError("motion correlation threshold is outside [-1, 1]")
        if self.region_distance_max < 0:
            raise ValueError("region distance threshold must be nonnegative")
        for name, value in (
            ("contact_min_count", self.contact_min_count),
            ("effect_due_after_policy_steps", self.effect_due_after_policy_steps),
            ("evidence_ttl_policy_steps", self.evidence_ttl_policy_steps),
            (
                "manipulation_attribution_ttl_policy_steps",
                self.manipulation_attribution_ttl_policy_steps,
            ),
        ):
            _require_positive(name, value)


@dataclass(frozen=True)
class MonitorEvidenceContract:
    contract_id: str
    task_id: int
    object_ids: tuple[str, ...]
    nominal_source_facts: tuple[str, ...]
    abnormal_support_surfaces: tuple[str, ...]
    task_relevant_effects: tuple[str, ...]
    tracker_version: str
    action_event_rules: tuple[ActionEventRule, ...]
    monitor_interval_steps: int
    confirmation_count: int
    settling_grace_observations: int
    progress_window_observations: int
    progress_evidence_ttl_policy_steps: int
    goal_regression_evidence_ttl_policy_steps: int
    max_active_attempts_per_object: int
    max_attempt_records_per_episode: int
    max_evidence_records_per_episode: int
    grounding_rule_sha256: str
    event_detector_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        if not self.contract_id or not self.tracker_version:
            raise ValueError("monitor contract ID and tracker version must be nonempty")
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int) or self.task_id < 0:
            raise ValueError("monitor contract task ID must be nonnegative")
        for name, values in (
            ("object_ids", self.object_ids),
            ("nominal_source_facts", self.nominal_source_facts),
            ("abnormal_support_surfaces", self.abnormal_support_surfaces),
            ("task_relevant_effects", self.task_relevant_effects),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"monitor contract {name} must be a unique string tuple")
        if not self.object_ids:
            raise ValueError("monitor contract object IDs must be nonempty")
        for value in self.nominal_source_facts + self.task_relevant_effects:
            parse_pddl_fact(value)
        for name, value in (
            ("monitor_interval_steps", self.monitor_interval_steps),
            ("confirmation_count", self.confirmation_count),
            ("settling_grace_observations", self.settling_grace_observations),
            ("progress_window_observations", self.progress_window_observations),
            (
                "progress_evidence_ttl_policy_steps",
                self.progress_evidence_ttl_policy_steps,
            ),
            (
                "goal_regression_evidence_ttl_policy_steps",
                self.goal_regression_evidence_ttl_policy_steps,
            ),
            ("max_active_attempts_per_object", self.max_active_attempts_per_object),
            ("max_attempt_records_per_episode", self.max_attempt_records_per_episode),
            ("max_evidence_records_per_episode", self.max_evidence_records_per_episode),
        ):
            _require_positive(name, value)
        _require_sha256("grounding rule hash", self.grounding_rule_sha256)
        _require_sha256("event detector hash", self.event_detector_sha256)
        _require_sha256("monitor contract hash", self.contract_sha256)
        rule_ids: set[str] = set()
        identities: set[tuple[Any, ...]] = set()
        for rule in self.action_event_rules:
            if not isinstance(rule, ActionEventRule):
                raise ValueError("monitor contract rules must be ActionEventRule records")
            if rule.rule_id in rule_ids:
                raise ValueError("duplicate monitor action rule ID")
            identity = (
                rule.object_id,
                rule.attempted_effect,
                rule.source_region,
                rule.destination_region,
            )
            if identity in identities:
                raise ValueError("duplicate monitor action rule identity")
            if rule.object_id not in self.object_ids:
                raise ValueError("monitor rule object is outside contract object IDs")
            if rule.attempted_effect not in self.task_relevant_effects:
                raise ValueError("monitor rule effect is outside task-relevant effects")
            rule_ids.add(rule.rule_id)
            identities.add(identity)
        signed = asdict(self)
        signed.pop("contract_sha256")
        expected = hashlib.sha256(_canonical_json(signed).encode("utf-8")).hexdigest()
        if expected != self.contract_sha256:
            raise ValueError("monitor contract self-hash mismatch")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MonitorEvidenceContract":
        expected = {item.name for item in cls.__dataclass_fields__.values()}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("monitor contract schema fields mismatch")
        payload = dict(value)
        rules = payload.get("action_event_rules")
        if not isinstance(rules, list):
            raise ValueError("monitor contract action rules must be a list")
        payload["action_event_rules"] = tuple(ActionEventRule(**item) for item in rules)
        for name in (
            "object_ids",
            "nominal_source_facts",
            "abnormal_support_surfaces",
            "task_relevant_effects",
        ):
            if not isinstance(payload[name], list):
                raise ValueError(f"monitor contract {name} must be a list")
            payload[name] = tuple(payload[name])
        return cls(**payload)

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


def load_monitor_evidence_contract(path: Path | str, *, task_id: int) -> MonitorEvidenceContract:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load monitor evidence contract: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "contracts"}:
        raise ValueError("monitor contract registry schema mismatch")
    if payload["schema_version"] != 1 or not isinstance(payload["contracts"], list):
        raise ValueError("monitor contract registry version mismatch")
    contracts = [MonitorEvidenceContract.from_mapping(item) for item in payload["contracts"]]
    if len({item.task_id for item in contracts}) != len(contracts):
        raise ValueError("duplicate monitor contract task ID")
    if {item.task_id for item in contracts} != {5, 8}:
        raise ValueError("monitor contract registry must freeze exactly tasks 5 and 8")
    try:
        contract = next(item for item in contracts if item.task_id == task_id)
    except StopIteration as error:
        raise ValueError(f"no monitor evidence contract for task {task_id}") from error
    coverage_path = Path(__file__).resolve().parents[3] / "configs/logiv/libero10-coverage.json"
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        task = next(
            item for item in coverage["tasks"] if item["task_id"] == contract.task_id
        )
        registered = frozenset(task["registered_objects"])
    except (OSError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        raise ValueError("cannot validate monitor contract against frozen coverage") from error
    required = set(contract.object_ids) | set(contract.abnormal_support_surfaces)
    for value in contract.nominal_source_facts + contract.task_relevant_effects:
        required.update(parse_pddl_fact(value).arguments)
    for rule in contract.action_event_rules:
        required.update(
            value
            for value in (rule.object_id, rule.source_region, rule.destination_region)
            if value is not None
        )
    missing = required - registered
    if missing:
        raise ValueError(
            "monitor contract IDs are outside frozen task coverage: "
            + ", ".join(sorted(missing))
        )
    return contract


def _transition_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.value if isinstance(value, Enum) else value
        for key, value in values.items()
        if key != "transition_sha256"
    }


@dataclass(frozen=True)
class ActionTransitionFeatures:
    policy_step: int
    monitor_contract_sha256: str
    tracker_version: str
    rule_id: str
    object_id: str
    source_region: str | None
    destination_region: str | None
    gripper_qpos: float | None
    contact_count: int | None
    holding: TruthValue
    source_region_truth: TruthValue
    destination_region_truth: TruthValue
    abnormal_region_truth: TruthValue
    source_region_distance: float | None
    destination_region_distance: float | None
    abnormal_region_id: str | None
    abnormal_region_distance: float | None
    object_eef_motion_correlation: float | None
    transition_sha256: str

    @classmethod
    def create(cls, **values: Any) -> "ActionTransitionFeatures":
        payload = _transition_payload(values)
        values["transition_sha256"] = _json_sha256(
            b"LOGIV_ACTION_TRANSITION_V1", payload
        )
        return cls(**values)

    def __post_init__(self) -> None:
        if isinstance(self.policy_step, bool) or not isinstance(self.policy_step, int) or self.policy_step < 0:
            raise ValueError("transition policy step must be nonnegative")
        _require_sha256("transition monitor contract hash", self.monitor_contract_sha256)
        if not self.tracker_version or not self.rule_id or not self.object_id:
            raise ValueError("transition feature identity is incomplete")
        for value in (
            self.holding,
            self.source_region_truth,
            self.destination_region_truth,
            self.abnormal_region_truth,
        ):
            if not isinstance(value, TruthValue):
                raise ValueError("transition truth value is invalid")
        if self.contact_count is not None and (
            isinstance(self.contact_count, bool)
            or not isinstance(self.contact_count, int)
            or self.contact_count < 0
        ):
            raise ValueError("contact count must be nonnegative or null")
        for name, value, nonnegative in (
            ("gripper_qpos", self.gripper_qpos, False),
            ("source_region_distance", self.source_region_distance, True),
            ("destination_region_distance", self.destination_region_distance, True),
            ("abnormal_region_distance", self.abnormal_region_distance, True),
            (
                "object_eef_motion_correlation",
                self.object_eef_motion_correlation,
                False,
            ),
        ):
            _optional_finite(name, value, nonnegative=nonnegative)
        if self.object_eef_motion_correlation is not None and not (
            -1.0 <= self.object_eef_motion_correlation <= 1.0
        ):
            raise ValueError("motion correlation is outside [-1, 1]")
        if self.abnormal_region_id is not None and not self.abnormal_region_id:
            raise ValueError("abnormal region ID must be nonempty or null")
        _require_sha256("transition hash", self.transition_sha256)
        if self.transition_sha256 != _json_sha256(
            b"LOGIV_ACTION_TRANSITION_V1", _transition_payload(asdict(self))
        ):
            raise ValueError("transition feature hash mismatch")


@dataclass(frozen=True)
class ActionEventEvidence:
    evidence_id: str
    evidence_kind: str
    rule_id: str
    object_id: str
    attempted_effect: str
    source_region: str | None
    destination_region: str | None
    attempt_id: str
    start_policy_step: int
    effect_due_policy_step: int
    emitted_policy_step: int
    evidence_expires_policy_step: int
    supporting_transition_hashes: tuple[str, ...]
    detector_sha256: str

    @classmethod
    def create(cls, **values: Any) -> "ActionEventEvidence":
        values["supporting_transition_hashes"] = tuple(
            values["supporting_transition_hashes"]
        )
        values["evidence_id"] = _json_sha256(
            b"LOGIV_ACTION_EVENT_EVIDENCE_ID_V1",
            {
                "attempt_id": values["attempt_id"],
                "evidence_kind": values["evidence_kind"],
                "emitted_policy_step": values["emitted_policy_step"],
                "supporting_transition_hashes": list(
                    values["supporting_transition_hashes"]
                ),
            },
        )
        return cls(**values)

    def __post_init__(self) -> None:
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError("unsupported action event evidence kind")
        if not self.rule_id or not self.object_id:
            raise ValueError("action event evidence identity is incomplete")
        attempted = parse_pddl_fact(self.attempted_effect)
        if attempted.arguments and attempted.arguments[0] != self.object_id:
            raise ValueError("event evidence object/effect mismatch")
        for name, value in (
            ("source_region", self.source_region),
            ("destination_region", self.destination_region),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"event evidence {name} must be nonempty or null")
        _require_sha256("event attempt ID", self.attempt_id)
        _require_sha256("event evidence ID", self.evidence_id)
        _require_sha256("event detector hash", self.detector_sha256)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.start_policy_step,
                self.effect_due_policy_step,
                self.emitted_policy_step,
                self.evidence_expires_policy_step,
            )
        ):
            raise ValueError("event policy steps must be nonnegative integers")
        if not (
            self.start_policy_step
            <= self.effect_due_policy_step
            <= self.emitted_policy_step
            <= self.evidence_expires_policy_step
        ):
            raise ValueError("event policy-step window is invalid")
        if self.evidence_kind in {
            "GOAL_REGRESSION",
            "ABNORMAL_TRANSFER_AFTER_MANIPULATION",
        } and self.effect_due_policy_step != self.emitted_policy_step:
            raise ValueError("immediate event evidence must be due when emitted")
        if not self.supporting_transition_hashes:
            raise ValueError("event evidence requires supporting hashes")
        if len(self.supporting_transition_hashes) != len(
            set(self.supporting_transition_hashes)
        ):
            raise ValueError("event evidence supporting hashes must be unique")
        for value in self.supporting_transition_hashes:
            _require_sha256("supporting transition hash", value)
        expected = _json_sha256(
            b"LOGIV_ACTION_EVENT_EVIDENCE_ID_V1",
            {
                "attempt_id": self.attempt_id,
                "evidence_kind": self.evidence_kind,
                "emitted_policy_step": self.emitted_policy_step,
                "supporting_transition_hashes": list(
                    self.supporting_transition_hashes
                ),
            },
        )
        if expected != self.evidence_id:
            raise ValueError("action event evidence hash mismatch")


class TransitionFeatureReader(Protocol):
    monitor_contract_sha256: str
    tracker_version: str
    rule_ids: tuple[str, ...]

    def __call__(
        self, context: ShadowStepContext
    ) -> tuple[ActionTransitionFeatures, ...]: ...


class ActionEventTracker(Protocol):
    @property
    def overflow_count(self) -> int: ...

    def observe(self, context: ShadowStepContext) -> None: ...

    def record_fact_event(self, evidence: ActionEventEvidence) -> bool: ...

    def active_for(
        self,
        *,
        object_id: str,
        attempted_effect: str,
        source_region: str | None,
        destination_region: str | None,
        policy_step: int,
    ) -> tuple[ActionEventEvidence, ...]: ...


@dataclass
class _Attempt:
    rule: ActionEventRule
    attempt_id: str
    start_policy_step: int
    effect_due_policy_step: int
    attribution_expires_policy_step: int
    supporting_hashes: list[str]
    timeout_state: str = "ACTIVE"
    emitted_kinds: set[str] = field(default_factory=set)


class VersionedActionEventTracker:
    def __init__(
        self,
        monitor_contract: MonitorEvidenceContract,
        transition_feature_reader: TransitionFeatureReader,
    ) -> None:
        expected_ids = tuple(rule.rule_id for rule in monitor_contract.action_event_rules)
        if transition_feature_reader.monitor_contract_sha256 != monitor_contract.contract_sha256:
            raise ValueError("transition reader monitor contract hash mismatch")
        if transition_feature_reader.tracker_version != monitor_contract.tracker_version:
            raise ValueError("transition reader tracker version mismatch")
        if transition_feature_reader.rule_ids != expected_ids:
            raise ValueError("transition reader rule IDs mismatch")
        self.monitor_contract = monitor_contract
        self.transition_feature_reader = transition_feature_reader
        self.rules = {rule.rule_id: rule for rule in monitor_contract.action_event_rules}
        self._previous: dict[str, ActionTransitionFeatures] = {}
        self._last_policy_step: int | None = None
        self.attempts: list[_Attempt] = []
        self._evidence: list[ActionEventEvidence] = []
        self._evidence_ids: set[str] = set()
        self._attempt_record_count = 0
        self._evidence_record_count = 0
        self._overflow_count = 0

    @property
    def attempt_record_count(self) -> int:
        return self._attempt_record_count

    @property
    def evidence_record_count(self) -> int:
        return self._evidence_record_count

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    def _effect_truth(
        self, rule: ActionEventRule, feature: ActionTransitionFeatures
    ) -> TruthValue:
        effect = parse_pddl_fact(rule.attempted_effect)
        if effect.predicate == "holding" and effect.arguments == (rule.object_id,):
            return feature.holding
        if (
            effect.arguments
            and effect.arguments[0] == rule.object_id
            and rule.destination_region is not None
            and effect.arguments[-1] == rule.destination_region
        ):
            return feature.destination_region_truth
        return TruthValue.UNKNOWN

    def _active_attempts_for(self, object_id: str, policy_step: int) -> int:
        return sum(
            attempt.rule.object_id == object_id
            and (
                attempt.timeout_state == "ACTIVE"
                or policy_step <= attempt.attribution_expires_policy_step
            )
            for attempt in self.attempts
        )

    def _start_attempt(
        self, rule: ActionEventRule, feature: ActionTransitionFeatures
    ) -> None:
        if self._attempt_record_count >= self.monitor_contract.max_attempt_records_per_episode:
            self._overflow_count += 1
            return
        if self._active_attempts_for(rule.object_id, feature.policy_step) >= self.monitor_contract.max_active_attempts_per_object:
            self._overflow_count += 1
            return
        attempt_id = _json_sha256(
            b"LOGIV_ACTION_ATTEMPT_ID_V1",
            {
                "rule_id": rule.rule_id,
                "object_id": rule.object_id,
                "start_policy_step": feature.policy_step,
                "start_transition_sha256": feature.transition_sha256,
            },
        )
        self.attempts.append(
            _Attempt(
                rule=rule,
                attempt_id=attempt_id,
                start_policy_step=feature.policy_step,
                effect_due_policy_step=(
                    feature.policy_step + rule.effect_due_after_policy_steps
                ),
                attribution_expires_policy_step=(
                    feature.policy_step
                    + rule.manipulation_attribution_ttl_policy_steps
                ),
                supporting_hashes=[feature.transition_sha256],
            )
        )
        self._attempt_record_count += 1

    def _accept_evidence(self, evidence: ActionEventEvidence) -> bool:
        if evidence.evidence_id in self._evidence_ids:
            return False
        if self._evidence_record_count >= self.monitor_contract.max_evidence_records_per_episode:
            self._overflow_count += 1
            return False
        self._evidence.append(evidence)
        self._evidence_ids.add(evidence.evidence_id)
        self._evidence_record_count += 1
        return True

    def _emit(
        self,
        attempt: _Attempt,
        evidence_kind: str,
        feature: ActionTransitionFeatures,
    ) -> None:
        if evidence_kind in attempt.emitted_kinds:
            return
        attempt.emitted_kinds.add(evidence_kind)
        hashes = tuple(dict.fromkeys(attempt.supporting_hashes + [feature.transition_sha256]))
        immediate = evidence_kind == "ABNORMAL_TRANSFER_AFTER_MANIPULATION"
        emitted = feature.policy_step
        evidence = ActionEventEvidence.create(
            evidence_kind=evidence_kind,
            rule_id=attempt.rule.rule_id,
            object_id=attempt.rule.object_id,
            attempted_effect=attempt.rule.attempted_effect,
            source_region=attempt.rule.source_region,
            destination_region=attempt.rule.destination_region,
            attempt_id=attempt.attempt_id,
            start_policy_step=attempt.start_policy_step,
            effect_due_policy_step=(emitted if immediate else attempt.effect_due_policy_step),
            emitted_policy_step=emitted,
            evidence_expires_policy_step=(
                emitted + attempt.rule.evidence_ttl_policy_steps
            ),
            supporting_transition_hashes=hashes,
            detector_sha256=self.monitor_contract.event_detector_sha256,
        )
        self._accept_evidence(evidence)

    def _advance_attempts(
        self, rule: ActionEventRule, feature: ActionTransitionFeatures
    ) -> None:
        for attempt in self.attempts:
            if attempt.rule.rule_id != rule.rule_id:
                continue
            relevant = (
                attempt.timeout_state == "ACTIVE"
                or feature.policy_step <= attempt.attribution_expires_policy_step
            )
            if relevant and feature.transition_sha256 not in attempt.supporting_hashes:
                attempt.supporting_hashes.append(feature.transition_sha256)
            if attempt.timeout_state == "ACTIVE":
                effect = self._effect_truth(rule, feature)
                if effect is TruthValue.TRUE and feature.policy_step <= attempt.effect_due_policy_step:
                    attempt.timeout_state = "SATISFIED"
                elif feature.policy_step >= attempt.effect_due_policy_step:
                    if effect is TruthValue.FALSE:
                        self._emit(attempt, "ATTEMPTED_EFFECT_TIMEOUT", feature)
                        attempt.timeout_state = "EMITTED"
                    else:
                        attempt.timeout_state = "INCONCLUSIVE"
            abnormal_registered = (
                feature.abnormal_region_id
                in self.monitor_contract.abnormal_support_surfaces
            )
            abnormal_near = (
                feature.abnormal_region_distance is not None
                and feature.abnormal_region_distance <= rule.region_distance_max
            )
            if (
                feature.policy_step <= attempt.attribution_expires_policy_step
                and feature.abnormal_region_truth is TruthValue.TRUE
                and abnormal_registered
                and abnormal_near
            ):
                self._emit(
                    attempt,
                    "ABNORMAL_TRANSFER_AFTER_MANIPULATION",
                    feature,
                )

    @staticmethod
    def _continuous(
        rule: ActionEventRule,
        feature: ActionTransitionFeatures,
        *,
        initial: bool,
    ) -> bool:
        if feature.gripper_qpos is None or feature.contact_count is None:
            return False
        if feature.holding is TruthValue.UNKNOWN:
            return False
        if rule.source_region is not None and (
            feature.source_region_truth is TruthValue.UNKNOWN
            or feature.source_region_distance is None
        ):
            return False
        if rule.destination_region is not None and (
            feature.destination_region_truth is TruthValue.UNKNOWN
            or feature.destination_region_distance is None
        ):
            return False
        return initial or feature.object_eef_motion_correlation is not None

    def _mark_due_inconclusive(self, policy_step: int) -> None:
        for attempt in self.attempts:
            if (
                attempt.timeout_state == "ACTIVE"
                and policy_step >= attempt.effect_due_policy_step
            ):
                attempt.timeout_state = "INCONCLUSIVE"

    def _can_start(
        self,
        rule: ActionEventRule,
        previous: ActionTransitionFeatures | None,
        feature: ActionTransitionFeatures,
        context: ShadowStepContext,
    ) -> bool:
        if previous is None or context.last_action is None:
            return False
        if previous.policy_step + 1 != feature.policy_step:
            return False
        if previous.gripper_qpos is None or feature.gripper_qpos is None:
            return False
        if rule.attempt_kind == "grasp":
            closed = (
                previous.gripper_qpos > rule.gripper_close_threshold
                and feature.gripper_qpos <= rule.gripper_close_threshold
            )
            return bool(
                closed
                and feature.contact_count is not None
                and feature.contact_count >= rule.contact_min_count
                and feature.object_eef_motion_correlation is not None
                and feature.object_eef_motion_correlation
                >= rule.motion_correlation_min
                and feature.source_region_truth is TruthValue.TRUE
            )
        opened = (
            previous.gripper_qpos < rule.gripper_open_threshold
            and feature.gripper_qpos >= rule.gripper_open_threshold
        )
        return bool(
            opened
            and previous.holding is TruthValue.TRUE
        )

    def observe(self, context: ShadowStepContext) -> None:
        if (
            self._last_policy_step is not None
            and context.policy_step <= self._last_policy_step
        ):
            raise ValueError("action event policy steps must be monotonic")
        if (
            self._last_policy_step is not None
            and context.policy_step != self._last_policy_step + 1
        ):
            self._previous.clear()
            self._mark_due_inconclusive(context.policy_step)
        try:
            rows = self.transition_feature_reader(context)
        except Exception:
            self._previous.clear()
            self._mark_due_inconclusive(context.policy_step)
            self._last_policy_step = context.policy_step
            raise
        if not isinstance(rows, tuple):
            self._previous.clear()
            self._last_policy_step = context.policy_step
            raise ValueError("transition reader must return a tuple")
        by_rule: dict[str, ActionTransitionFeatures] = {}
        for feature in rows:
            if not isinstance(feature, ActionTransitionFeatures):
                raise ValueError("transition reader returned an invalid record")
            if feature.rule_id in by_rule:
                raise ValueError("duplicate rule transition record")
            by_rule[feature.rule_id] = feature
        missing = set(self.rules) - set(by_rule)
        extra = set(by_rule) - set(self.rules)
        if missing or extra:
            self._previous.clear()
            self._mark_due_inconclusive(context.policy_step)
            self._last_policy_step = context.policy_step
            raise ValueError(
                f"missing rule transition records: {sorted(missing)}; extra={sorted(extra)}"
            )
        next_previous: dict[str, ActionTransitionFeatures] = {}
        for rule_id in tuple(self.rules):
            rule = self.rules[rule_id]
            feature = by_rule[rule_id]
            if (
                feature.policy_step != context.policy_step
                or feature.monitor_contract_sha256
                != self.monitor_contract.contract_sha256
                or feature.tracker_version != self.monitor_contract.tracker_version
                or feature.object_id != rule.object_id
                or feature.source_region != rule.source_region
                or feature.destination_region != rule.destination_region
            ):
                self._previous.pop(rule_id, None)
                self._last_policy_step = context.policy_step
                raise ValueError("transition rule identity mismatch")
            self._advance_attempts(rule, feature)
            previous = self._previous.get(rule_id)
            if self._can_start(rule, previous, feature, context):
                self._start_attempt(rule, feature)
            if self._continuous(
                rule, feature, initial=context.last_action is None
            ):
                next_previous[rule_id] = feature
        self._previous = next_previous
        self._last_policy_step = context.policy_step

    def record_fact_event(self, evidence: ActionEventEvidence) -> bool:
        reserved = {
            "GOAL_REGRESSION": ReservedFactEventRuleId.GOAL_REGRESSION.value,
            "PROGRESS_TIMEOUT": ReservedFactEventRuleId.PROGRESS_TIMEOUT.value,
        }
        if evidence.evidence_kind not in reserved:
            raise ValueError("record_fact_event accepts only reserved fact events")
        if evidence.rule_id != reserved[evidence.evidence_kind]:
            raise ValueError("fact event rule ID mismatch")
        if evidence.detector_sha256 != self.monitor_contract.event_detector_sha256:
            raise ValueError("fact event detector hash mismatch")
        if evidence.object_id not in self.monitor_contract.object_ids:
            raise ValueError("fact event object is outside the monitor contract")
        if evidence.attempted_effect not in self.monitor_contract.task_relevant_effects:
            raise ValueError("fact event effect is outside the monitor contract")
        rule = next(
            (
                item
                for item in self.monitor_contract.action_event_rules
                if item.object_id == evidence.object_id
                and item.attempted_effect == evidence.attempted_effect
            ),
            None,
        )
        if rule is None or (
            evidence.source_region,
            evidence.destination_region,
        ) != (rule.source_region, rule.destination_region):
            raise ValueError("fact event action-rule identity mismatch")
        expected_ttl = (
            self.monitor_contract.goal_regression_evidence_ttl_policy_steps
            if evidence.evidence_kind == "GOAL_REGRESSION"
            else self.monitor_contract.progress_evidence_ttl_policy_steps
        )
        if evidence.evidence_expires_policy_step != evidence.emitted_policy_step + expected_ttl:
            raise ValueError("fact event evidence TTL mismatch")
        if evidence.evidence_kind == "PROGRESS_TIMEOUT" and (
            evidence.effect_due_policy_step
            != evidence.start_policy_step
            + self.monitor_contract.progress_window_observations
            * self.monitor_contract.monitor_interval_steps
        ):
            raise ValueError("progress timeout due step mismatch")
        if evidence.evidence_kind == "GOAL_REGRESSION":
            expected_attempt_id = _json_sha256(
                b"LOGIV_GOAL_REGRESSION_ATTEMPT_ID_V1",
                {
                    "achieved_fact_evidence_sha256": (
                        evidence.supporting_transition_hashes[0]
                    ),
                    "achieved_policy_step": evidence.start_policy_step,
                    "goal_literal": evidence.attempted_effect,
                },
            )
        else:
            expected_attempt_id = _json_sha256(
                b"LOGIV_PROGRESS_TIMEOUT_ATTEMPT_ID_V1",
                {
                    "attempted_effect": evidence.attempted_effect,
                    "effect_due_policy_step": evidence.effect_due_policy_step,
                    "object_id": evidence.object_id,
                    "rule_id": evidence.rule_id,
                    "start_fact_evidence_sha256": (
                        evidence.supporting_transition_hashes[0]
                    ),
                    "start_policy_step": evidence.start_policy_step,
                },
            )
        if evidence.attempt_id != expected_attempt_id:
            raise ValueError("fact event attempt ID mismatch")
        return self._accept_evidence(evidence)

    def active_for(
        self,
        *,
        object_id: str,
        attempted_effect: str,
        source_region: str | None,
        destination_region: str | None,
        policy_step: int,
    ) -> tuple[ActionEventEvidence, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._evidence
                    if item.object_id == object_id
                    and item.attempted_effect == attempted_effect
                    and item.source_region == source_region
                    and item.destination_region == destination_region
                    and item.effect_due_policy_step
                    <= policy_step
                    <= item.evidence_expires_policy_step
                ),
                key=lambda item: item.evidence_id,
            )
        )


class DeviationStatus(str, Enum):
    ANOMALY_CANDIDATE = "ANOMALY_CANDIDATE"
    CONFIRMED_DEVIATION = "CONFIRMED_DEVIATION"


class CertificateState(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    RECONCILED = "RECONCILED"


@dataclass(frozen=True)
class CertificateReconciliation:
    observation_generation: int
    relevant_fact_sha256: str
    source_graph_version: str
    certificate_state: CertificateState


def _relevant_facts(plan_context: ShadowPlanContext) -> frozenset[Fact]:
    facts = set(plan_context.problem.goal | plan_context.problem.negative_goal)
    facts.add(Fact("handempty"))
    object_types = plan_context.problem.object_types
    movables = tuple(
        name for name, kind in object_types.items() if _is_subtype(kind, "movable")
    )
    locations = tuple(
        name
        for name, kind in object_types.items()
        if _is_subtype(kind, "location")
    )
    accesses = tuple(
        name for name, kind in object_types.items() if _is_subtype(kind, "access")
    )
    devices = tuple(
        name
        for name, kind in object_types.items()
        if _is_subtype(kind, "switchable")
    )
    facts.update(Fact("holding", (object_id,)) for object_id in movables)
    facts.update(
        Fact("at", (object_id, location))
        for object_id in movables
        for location in locations
    )
    facts.update(
        Fact(predicate, (access,))
        for access in accesses
        for predicate in ("open", "closed")
    )
    facts.update(
        Fact(predicate, (device,))
        for device in devices
        for predicate in ("powered-on", "powered-off")
    )
    for action in plan_context.plan:
        facts.update(
            action.preconditions
            | action.negative_preconditions
            | action.add_effects
            | action.del_effects
        )
    facts.update(link.literal.fact for link in plan_context.graph.causal_links)
    return frozenset(facts)


def _relevant_fact_hash(
    snapshot: FactSnapshot,
    relevant_facts: frozenset[Fact],
    grounding_rule_sha256: str,
) -> str:
    return _json_sha256(
        b"LOGIV_RELEVANT_FACTS_V1",
        {
            "grounding_rule_sha256": grounding_rule_sha256,
            "facts": [
                [fact.pddl(), snapshot.truth(fact).value]
                for fact in sorted(relevant_facts, key=lambda item: item.pddl())
            ],
        },
    )


class ShadowCertificateReconciler:
    _MANIPULATION_SCHEMAS = frozenset(
        {
            "pick",
            "place-on",
            "place-in",
            "place-relative",
            "put-down",
            "place-held-on",
            "place-held-in",
            "place-held-relative",
        }
    )
    _BINARY_TRANSITION_SCHEMAS = frozenset(
        {"open-access", "close-access", "turn-on", "turn-off"}
    )
    _TEMPORAL_SCHEMAS = _MANIPULATION_SCHEMAS | _BINARY_TRANSITION_SCHEMAS

    def __init__(
        self,
        plan_context: ShadowPlanContext,
        grounding_rule_sha256: str | None = None,
        *,
        effect_confirmation_steps: int = 5,
    ) -> None:
        if effect_confirmation_steps <= 0:
            raise ValueError("effect confirmation steps must be positive")
        self.plan_context = plan_context
        self.grounding_rule_sha256 = grounding_rule_sha256 or "0" * 64
        _require_sha256("grounding rule hash", self.grounding_rule_sha256)
        self.relevant_facts = _relevant_facts(plan_context)
        self._state = CertificateState.CURRENT
        self._generation = 0
        self._inflight_nodes: set[str] = set()
        self._effect_confirmation_steps = effect_confirmation_steps
        self._effect_streaks: dict[str, int] = {}
        self._pending_invalid_releases: dict[str, int] = {}

    def initial(self, snapshot: FactSnapshot) -> CertificateReconciliation:
        return CertificateReconciliation(
            observation_generation=self._generation,
            relevant_fact_sha256=_relevant_fact_hash(
                snapshot, self.relevant_facts, self.grounding_rule_sha256
            ),
            source_graph_version=self.plan_context.graph.graph_version,
            certificate_state=self._state,
        )

    def _predecessors_satisfied(
        self, node_id: str, previous: FactSnapshot
    ) -> bool:
        node_map = self.plan_context.graph.node_map
        for edge in self.plan_context.graph.edges:
            if edge.target != node_id or edge.source not in node_map:
                continue
            predecessor = node_map[edge.source]
            if predecessor.kind is not NodeKind.ACTION or predecessor.action is None:
                continue
            if not previous.satisfies(
                positive=predecessor.action.add_effects,
                negative=predecessor.action.del_effects,
            ):
                return False
        return True

    def _temporal_envelope(self, action: GroundAction) -> frozenset[SignedLiteral]:
        if action.schema in self._MANIPULATION_SCHEMAS:
            object_name = action.arguments[0]
            facts = {
                fact
                for fact in self.relevant_facts
                if (
                    fact.predicate == "holding"
                    and fact.arguments == (object_name,)
                )
                or (
                    fact.predicate == "at"
                    and fact.arguments
                    and fact.arguments[0] == object_name
                )
                or fact == Fact("handempty")
            }
        else:
            facts = set(
                action.preconditions
                | action.negative_preconditions
                | action.add_effects
                | action.del_effects
            )
        return frozenset(
            SignedLiteral(fact, truth) for fact in facts for truth in (False, True)
        )

    def _entered_temporal_transition(
        self, action: GroundAction, current: FactSnapshot
    ) -> bool:
        if action.schema in self._BINARY_TRANSITION_SCHEMAS:
            values = tuple(
                current.truth(fact)
                for fact in action.add_effects | action.del_effects
            )
            return bool(values) and all(
                value is TruthValue.FALSE for value in values
            )
        object_name = action.arguments[0]
        holding = current.truth(Fact("holding", (object_name,)))
        if holding is TruthValue.TRUE:
            return True
        if holding is not TruthValue.FALSE:
            return False
        locations = tuple(
            fact
            for fact in self.relevant_facts
            if fact.predicate == "at"
            and fact.arguments
            and fact.arguments[0] == object_name
        )
        return bool(locations) and all(
            current.truth(fact) is TruthValue.FALSE for fact in locations
        )

    def _grounded_non_target_release(
        self, action: GroundAction, current: FactSnapshot
    ) -> bool:
        if action.schema not in self._MANIPULATION_SCHEMAS:
            return False
        object_name = action.arguments[0]
        if current.truth(Fact("holding", (object_name,))) is not TruthValue.FALSE:
            return False
        target_locations = {
            fact
            for fact in action.add_effects
            if fact.predicate == "at" and fact.arguments[:1] == (object_name,)
        }
        return any(
            fact not in target_locations and current.truth(fact) is TruthValue.TRUE
            for fact in self.relevant_facts
            if fact.predicate == "at" and fact.arguments[:1] == (object_name,)
        )

    def _covered(
        self,
        changed: frozenset[SignedLiteral],
        became_unknown: frozenset[Fact],
        previous: FactSnapshot,
        current: FactSnapshot,
    ) -> bool:
        for node_id in tuple(self._inflight_nodes):
            action = self.plan_context.graph.node_map[node_id].action
            envelope = self._temporal_envelope(action) if action is not None else ()
            envelope_facts = frozenset(item.fact for item in envelope)
            unknown_envelope_facts = (
                action.add_effects | action.del_effects
                if action is not None
                and action.schema in self._BINARY_TRANSITION_SCHEMAS
                else envelope_facts
            )
            if (
                action is None
                or not changed <= envelope
                or not became_unknown <= unknown_envelope_facts
            ):
                continue
            if current.satisfies(
                positive=action.add_effects, negative=action.del_effects
            ):
                confirmations = self._effect_streaks.get(node_id, 0) + 1
                self._effect_streaks[node_id] = confirmations
                self._pending_invalid_releases.pop(node_id, None)
                if confirmations >= self._effect_confirmation_steps:
                    self._inflight_nodes.remove(node_id)
                    self._effect_streaks.pop(node_id, None)
                return True
            self._effect_streaks.pop(node_id, None)
            if self._entered_temporal_transition(action, current):
                self._pending_invalid_releases.pop(node_id, None)
                return True
            if current.satisfies(
                positive=action.preconditions,
                negative=action.negative_preconditions,
            ):
                self._inflight_nodes.remove(node_id)
                self._effect_streaks.pop(node_id, None)
                self._pending_invalid_releases.pop(node_id, None)
                return True
            if self._grounded_non_target_release(action, current):
                confirmations = self._pending_invalid_releases.get(node_id, 0) + 1
                self._pending_invalid_releases[node_id] = confirmations
                if confirmations < 2:
                    return True
                self._inflight_nodes.remove(node_id)
                self._effect_streaks.pop(node_id, None)
                self._pending_invalid_releases.pop(node_id, None)
                return False
            self._pending_invalid_releases.pop(node_id, None)
            return True
        for node_id in self.plan_context.graph.canonical_agenda:
            node = self.plan_context.graph.node_map[node_id]
            action = node.action
            if action is None or not previous.satisfies(
                positive=action.preconditions,
                negative=action.negative_preconditions,
            ) or not self._predecessors_satisfied(node_id, previous):
                continue
            effects = frozenset(
                {SignedLiteral(fact, True) for fact in action.add_effects}
                | {SignedLiteral(fact, False) for fact in action.del_effects}
            )
            envelope = self._temporal_envelope(action)
            binary_facts = action.add_effects | action.del_effects
            if (
                action.schema in self._BINARY_TRANSITION_SCHEMAS
                and became_unknown
                and changed <= envelope
                and became_unknown <= binary_facts
            ):
                self._inflight_nodes.add(node_id)
                self._effect_streaks.pop(node_id, None)
                self._pending_invalid_releases.pop(node_id, None)
                return True
            if changed and changed <= effects:
                if action.schema in self._TEMPORAL_SCHEMAS:
                    if current.satisfies(
                        positive=action.add_effects,
                        negative=action.del_effects,
                    ):
                        self._inflight_nodes.add(node_id)
                        self._effect_streaks[node_id] = 1
                        self._pending_invalid_releases.pop(node_id, None)
                    else:
                        self._inflight_nodes.add(node_id)
                        self._effect_streaks.pop(node_id, None)
                        self._pending_invalid_releases.pop(node_id, None)
                return True
            if (
                action.schema in self._MANIPULATION_SCHEMAS
                and changed <= self._temporal_envelope(action)
                and self._grounded_non_target_release(action, current)
            ):
                self._inflight_nodes.add(node_id)
                self._pending_invalid_releases[node_id] = 1
                return True
            if (
                action.schema in self._TEMPORAL_SCHEMAS
                and changed <= self._temporal_envelope(action)
            ):
                if current.satisfies(
                    positive=action.add_effects,
                    negative=action.del_effects,
                ):
                    self._inflight_nodes.add(node_id)
                    self._effect_streaks[node_id] = 1
                    self._pending_invalid_releases.pop(node_id, None)
                    return True
                if self._entered_temporal_transition(action, current):
                    self._inflight_nodes.add(node_id)
                    self._effect_streaks.pop(node_id, None)
                    self._pending_invalid_releases.pop(node_id, None)
                    return True
        return False

    def reconcile(
        self, previous: FactSnapshot, current: FactSnapshot
    ) -> CertificateReconciliation:
        self._generation += 1
        changed = frozenset(
            SignedLiteral(fact, current.truth(fact) is TruthValue.TRUE)
            for fact in self.relevant_facts
            if previous.truth(fact) is not current.truth(fact)
            and current.truth(fact) is not TruthValue.UNKNOWN
        )
        became_unknown = frozenset(
            fact
            for fact in self.relevant_facts
            if previous.truth(fact) is not current.truth(fact)
            and current.truth(fact) is TruthValue.UNKNOWN
        )
        if (
            self._state is CertificateState.CURRENT
            and (changed or became_unknown or self._inflight_nodes)
            and not self._covered(changed, became_unknown, previous, current)
        ):
            self._state = CertificateState.STALE
        return CertificateReconciliation(
            observation_generation=self._generation,
            relevant_fact_sha256=_relevant_fact_hash(
                current, self.relevant_facts, self.grounding_rule_sha256
            ),
            source_graph_version=self.plan_context.graph.graph_version,
            certificate_state=self._state,
        )


@dataclass(frozen=True)
class ShadowTrigger:
    trigger_class: str
    deviation_status: DeviationStatus
    signature: tuple[str, ...]
    historical_failure_evidence: tuple[ActionEventEvidence, ...]
    deviation_event_id: str
    event_origin_parent_sha256: str
    policy_step: int
    observation_generation: int
    relevant_fact_sha256: str
    source_graph_version: str
    certificate_state: CertificateState
    snapshot: FactSnapshot
    observation: Mapping[str, Any]


@dataclass
class ShadowMonitorMetrics:
    snapshot_calls: int = 0
    snapshot_errors: int = 0
    event_tracker_errors: int = 0
    evidence_overflows: int = 0
    trigger_callback_errors: int = 0
    anomaly_candidates: int = 0
    confirmed_deviations: int = 0
    stale_certificates: int = 0


def _copied_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in observation.items():
        result[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return result


class StableRecoveryObserver:
    def __init__(
        self,
        *,
        plan_context: ShadowPlanContext,
        monitor_contract: MonitorEvidenceContract,
        snapshot_reader: Callable[[Mapping[str, Any]], FactSnapshot],
        action_event_tracker: ActionEventTracker,
        on_trigger: Callable[[ShadowTrigger], None],
        interval_steps: int,
        confirmation_count: int,
        on_snapshot: Callable[[ShadowStepContext, FactSnapshot, CertificateReconciliation], None] | None = None,
    ) -> None:
        if interval_steps != monitor_contract.monitor_interval_steps:
            raise ValueError("monitor interval does not match the frozen contract")
        if confirmation_count != monitor_contract.confirmation_count:
            raise ValueError("monitor confirmation count does not match the frozen contract")
        if monitor_contract.task_id < 0:
            raise ValueError("monitor contract task mismatch")
        self.plan_context = plan_context
        self.monitor_contract = monitor_contract
        self.snapshot_reader = snapshot_reader
        self.action_event_tracker = action_event_tracker
        self.on_trigger = on_trigger
        self.interval_steps = interval_steps
        self.confirmation_count = confirmation_count
        self.on_snapshot = on_snapshot
        self.metrics = ShadowMonitorMetrics()
        self.reconciler = ShadowCertificateReconciler(
            plan_context, monitor_contract.grounding_rule_sha256
        )
        self._required_facts = self._build_required_facts()
        self._universe_version: str | None = None
        self._universe_sha256: str | None = None
        self._previous_snapshot: FactSnapshot | None = None
        self._reconciliation: CertificateReconciliation | None = None
        self._achieved: dict[SignedLiteral, tuple[int, str]] = {}
        self._regressed: set[SignedLiteral] = set()
        self._fact_events: dict[SignedLiteral, ActionEventEvidence] = {}
        self._progress_start: tuple[int, str] | None = None
        self._progress_recorded_for: set[tuple[int, str]] = set()
        self._streak_key: tuple[str, ...] | None = None
        self._streak_count = 0
        self._first_observed_step = 0
        self._provisional_origin: str | None = None
        self._recorded: dict[tuple[str, ...], DeviationStatus] = {}
        self._event_identities: dict[tuple[str, ...], tuple[str, str]] = {}

    def _build_required_facts(self) -> frozenset[Fact]:
        facts = set(_relevant_facts(self.plan_context))
        facts.update(parse_pddl_fact(value) for value in self.monitor_contract.nominal_source_facts)
        facts.update(parse_pddl_fact(value) for value in self.monitor_contract.task_relevant_effects)
        for object_id in self.monitor_contract.object_ids:
            facts.add(Fact("holding", (object_id,)))
            for surface in self.monitor_contract.abnormal_support_surfaces:
                facts.add(Fact("at", (object_id, surface)))
        for rule in self.monitor_contract.action_event_rules:
            if rule.source_region is not None:
                facts.add(Fact("at", (rule.object_id, rule.source_region)))
            if rule.destination_region is not None:
                facts.add(Fact("at", (rule.object_id, rule.destination_region)))
        return frozenset(facts)

    def _validate_snapshot(self, snapshot: FactSnapshot) -> None:
        if any(
            value is None
            for value in (
                snapshot.fact_universe,
                snapshot.fact_universe_version,
                snapshot.fact_universe_sha256,
                snapshot.evidence_payload_json,
            )
        ):
            raise ValueError("monitor snapshot lacks fact-universe audit fields")
        assert snapshot.fact_universe is not None
        if not self._required_facts <= snapshot.fact_universe:
            missing = self._required_facts - snapshot.fact_universe
            raise ValueError(
                "monitor snapshot misses registered facts: "
                + ", ".join(fact.pddl() for fact in sorted(missing))
            )
        if self._universe_version is None:
            self._universe_version = snapshot.fact_universe_version
            self._universe_sha256 = snapshot.fact_universe_sha256
        elif (
            snapshot.fact_universe_version != self._universe_version
            or snapshot.fact_universe_sha256 != self._universe_sha256
        ):
            raise ValueError("monitor fact-universe version/hash drift")

    def _rule_for_effect(
        self, object_id: str, attempted_effect: str
    ) -> ActionEventRule | None:
        return next(
            (
                rule
                for rule in self.monitor_contract.action_event_rules
                if rule.object_id == object_id
                and rule.attempted_effect == attempted_effect
            ),
            None,
        )

    def _fact_event(
        self,
        literal: SignedLiteral,
        achieved_step: int,
        achieved_hash: str,
        snapshot: FactSnapshot,
        policy_step: int,
    ) -> ActionEventEvidence:
        object_id = _fact_object(literal.fact)
        effect = literal.fact.pddl()
        rule = self._rule_for_effect(object_id, effect)
        attempt_id = _json_sha256(
            b"LOGIV_GOAL_REGRESSION_ATTEMPT_ID_V1",
            {
                "achieved_fact_evidence_sha256": achieved_hash,
                "achieved_policy_step": achieved_step,
                "goal_literal": literal.fact.pddl(),
            },
        )
        return ActionEventEvidence.create(
            evidence_kind="GOAL_REGRESSION",
            rule_id=ReservedFactEventRuleId.GOAL_REGRESSION.value,
            object_id=object_id,
            attempted_effect=effect,
            source_region=rule.source_region if rule else None,
            destination_region=rule.destination_region if rule else None,
            attempt_id=attempt_id,
            start_policy_step=achieved_step,
            effect_due_policy_step=policy_step,
            emitted_policy_step=policy_step,
            evidence_expires_policy_step=(
                policy_step
                + self.monitor_contract.goal_regression_evidence_ttl_policy_steps
            ),
            supporting_transition_hashes=(achieved_hash, snapshot.evidence_hash),
            detector_sha256=self.monitor_contract.event_detector_sha256,
        )

    def _update_goals(self, snapshot: FactSnapshot, policy_step: int) -> None:
        literals = tuple(
            [SignedLiteral(fact, True) for fact in self.plan_context.problem.goal]
            + [
                SignedLiteral(fact, False)
                for fact in self.plan_context.problem.negative_goal
            ]
        )
        for literal in literals:
            if _truth(snapshot, literal):
                self._achieved.setdefault(
                    literal, (policy_step, snapshot.evidence_hash)
                )
                continue
            opposite = snapshot.truth(literal.fact) is (
                TruthValue.FALSE if literal.positive else TruthValue.TRUE
            )
            if not opposite or literal not in self._achieved:
                continue
            self._regressed.add(literal)
            if literal in self._fact_events:
                continue
            achieved_step, achieved_hash = self._achieved[literal]
            evidence = self._fact_event(
                literal, achieved_step, achieved_hash, snapshot, policy_step
            )
            if self.action_event_tracker.record_fact_event(evidence):
                self._fact_events[literal] = evidence

    def _update_progress(
        self,
        snapshot: FactSnapshot,
        policy_step: int,
        relevant_hash: str,
    ) -> None:
        if self._progress_start is None or self._progress_start[1] != relevant_hash:
            self._progress_start = (policy_step, relevant_hash)
            return
        start_step, start_hash = self._progress_start
        due = (
            start_step
            + self.monitor_contract.progress_window_observations
            * self.monitor_contract.monitor_interval_steps
        )
        key = (start_step, start_hash)
        if policy_step < due or key in self._progress_recorded_for:
            return
        if not self.monitor_contract.action_event_rules:
            return
        rule = self.monitor_contract.action_event_rules[0]
        attempt_id = _json_sha256(
            b"LOGIV_PROGRESS_TIMEOUT_ATTEMPT_ID_V1",
            {
                "attempted_effect": rule.attempted_effect,
                "effect_due_policy_step": due,
                "object_id": rule.object_id,
                "rule_id": ReservedFactEventRuleId.PROGRESS_TIMEOUT.value,
                "start_fact_evidence_sha256": start_hash,
                "start_policy_step": start_step,
            },
        )
        evidence = ActionEventEvidence.create(
            evidence_kind="PROGRESS_TIMEOUT",
            rule_id=ReservedFactEventRuleId.PROGRESS_TIMEOUT.value,
            object_id=rule.object_id,
            attempted_effect=rule.attempted_effect,
            source_region=rule.source_region,
            destination_region=rule.destination_region,
            attempt_id=attempt_id,
            start_policy_step=start_step,
            effect_due_policy_step=due,
            emitted_policy_step=policy_step,
            evidence_expires_policy_step=(
                policy_step
                + self.monitor_contract.progress_evidence_ttl_policy_steps
            ),
            supporting_transition_hashes=(start_hash, snapshot.evidence_hash),
            detector_sha256=self.monitor_contract.event_detector_sha256,
        )
        self.action_event_tracker.record_fact_event(evidence)
        self._progress_recorded_for.add(key)

    def _active_evidence(
        self,
        object_id: str,
        attempted_effect: str,
        policy_step: int,
    ) -> tuple[ActionEventEvidence, ...]:
        rule = self._rule_for_effect(object_id, attempted_effect)
        source = rule.source_region if rule else None
        destination = rule.destination_region if rule else None
        return self.action_event_tracker.active_for(
            object_id=object_id,
            attempted_effect=attempted_effect,
            source_region=source,
            destination_region=destination,
            policy_step=policy_step,
        )

    def _candidate(
        self, snapshot: FactSnapshot, policy_step: int
    ) -> tuple[str, tuple[str, ...], tuple[ActionEventEvidence, ...]] | None:
        facts: list[str] = []
        evidence: dict[str, ActionEventEvidence] = {}
        regressed = tuple(
            sorted(
                (
                    literal
                    for literal in self._regressed
                    if snapshot.truth(literal.fact)
                    is (TruthValue.FALSE if literal.positive else TruthValue.TRUE)
                ),
                key=lambda item: (item.fact.pddl(), item.positive),
            )
        )
        if regressed:
            for literal in regressed:
                token = f"goal:{'+' if literal.positive else '-'}{literal.fact.pddl()}"
                facts.append(token)
                for item in self._active_evidence(
                    _fact_object(literal.fact), literal.fact.pddl(), policy_step
                ):
                    evidence[item.evidence_id] = item
            return (
                "COMPLETED_GOAL_REGRESSION_STABLE",
                tuple(sorted(facts)),
                tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
            )
        for object_id in self.monitor_contract.object_ids:
            for surface in self.monitor_contract.abnormal_support_surfaces:
                fact = Fact("at", (object_id, surface))
                if snapshot.truth(fact) is not TruthValue.TRUE:
                    continue
                if fact in self.plan_context.problem.initial_state or fact in self.plan_context.problem.goal:
                    continue
                object_evidence: dict[str, ActionEventEvidence] = {}
                for rule in self.monitor_contract.action_event_rules:
                    if rule.object_id != object_id:
                        continue
                    for item in self._active_evidence(
                        object_id, rule.attempted_effect, policy_step
                    ):
                        object_evidence[item.evidence_id] = item
                if any(
                    item.evidence_kind in _TRANSIENT_EVIDENCE
                    for item in object_evidence.values()
                ):
                    continue
                facts.append(fact.pddl())
                evidence.update(object_evidence)
        if not facts:
            timeout_evidence: dict[str, ActionEventEvidence] = {}
            timeout_effects: set[str] = set()
            for rule in self.monitor_contract.action_event_rules:
                for item in self._active_evidence(
                    rule.object_id, rule.attempted_effect, policy_step
                ):
                    if item.evidence_kind != "ATTEMPTED_EFFECT_TIMEOUT":
                        continue
                    timeout_evidence[item.evidence_id] = item
                    timeout_effects.add(item.attempted_effect)
            if not timeout_evidence:
                return None
            return (
                "ATTEMPTED_EFFECT_TIMEOUT_STABLE",
                tuple(f"effect:{effect}" for effect in sorted(timeout_effects)),
                tuple(
                    sorted(
                        timeout_evidence.values(),
                        key=lambda item: item.evidence_id,
                    )
                ),
            )
        return (
            "UNPLANNED_SUPPORT_STABLE",
            tuple(sorted(facts)),
            tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
        )

    def _event_ids(
        self,
        trigger_class: str,
        facts: tuple[str, ...],
        snapshot: FactSnapshot,
        first_observed_step: int,
    ) -> tuple[str, str]:
        goal_signature = sorted(
            [f"+{fact.pddl()}" for fact in self.plan_context.problem.goal]
            + [f"-{fact.pddl()}" for fact in self.plan_context.problem.negative_goal]
        )
        deviation_event_id = _json_sha256(
            b"LOGIV_DEVIATION_EVENT_V1",
            {
                "task_id": self.monitor_contract.task_id,
                "trigger_class": trigger_class,
                "facts": list(facts),
                "first_observed_policy_step": first_observed_step,
                "protected_goal": goal_signature,
            },
        )
        relevant_signed = [
            [fact.pddl(), snapshot.truth(fact).value]
            for fact in sorted(self._required_facts, key=lambda item: item.pddl())
        ]
        event_origin = _json_sha256(
            b"LOGIV_EVENT_ORIGIN_V1",
            {
                "task_id": self.monitor_contract.task_id,
                "trigger_class": trigger_class,
                "semantic_signature": list(facts),
                "protected_goal": goal_signature,
                "task_relevant_facts": relevant_signed,
            },
        )
        return deviation_event_id, event_origin

    def _reset_streak(self) -> None:
        self._streak_key = None
        self._streak_count = 0
        self._provisional_origin = None

    def observe_settling(
        self, observation: Mapping[str, Any]
    ) -> tuple[FactSnapshot, CertificateReconciliation] | None:
        """Refresh the fixed graph during no-action simulator settling."""

        self.metrics.snapshot_calls += 1
        try:
            snapshot = self.snapshot_reader(observation)
            if not isinstance(snapshot, FactSnapshot):
                raise ValueError("snapshot reader returned an invalid record")
            self._validate_snapshot(snapshot)
        except Exception:
            self.metrics.snapshot_errors += 1
            self._reset_streak()
            return None
        reconciliation = (
            self.reconciler.initial(snapshot)
            if self._previous_snapshot is None
            else self.reconciler.reconcile(self._previous_snapshot, snapshot)
        )
        self._previous_snapshot = snapshot
        self._reconciliation = reconciliation
        if reconciliation.certificate_state is CertificateState.STALE:
            self.metrics.stale_certificates = 1
        return snapshot, reconciliation

    def __call__(self, context: ShadowStepContext) -> None:
        tracker_ok = True
        try:
            self.action_event_tracker.observe(context)
        except Exception:
            tracker_ok = False
            self.metrics.event_tracker_errors += 1
        self.metrics.evidence_overflows = self.action_event_tracker.overflow_count
        if context.policy_step % self.interval_steps != 0:
            return
        self.metrics.snapshot_calls += 1
        try:
            snapshot = self.snapshot_reader(context.observation)
            if not isinstance(snapshot, FactSnapshot):
                raise ValueError("snapshot reader returned an invalid record")
            self._validate_snapshot(snapshot)
        except Exception:
            self.metrics.snapshot_errors += 1
            self._reset_streak()
            return
        if self._previous_snapshot is None:
            reconciliation = self.reconciler.initial(snapshot)
        else:
            reconciliation = self.reconciler.reconcile(
                self._previous_snapshot, snapshot
            )
        self._previous_snapshot = snapshot
        self._reconciliation = reconciliation
        if self.on_snapshot is not None:
            self.on_snapshot(context, snapshot, reconciliation)
        if reconciliation.certificate_state is CertificateState.STALE:
            self.metrics.stale_certificates = 1
        try:
            self._update_goals(snapshot, context.policy_step)
            self._update_progress(
                snapshot, context.policy_step, reconciliation.relevant_fact_sha256
            )
        except Exception:
            self.metrics.event_tracker_errors += 1
            tracker_ok = False
        candidate = self._candidate(snapshot, context.policy_step)
        if candidate is None:
            self._reset_streak()
            return
        trigger_class, facts, evidence = candidate
        if not tracker_ok or self.action_event_tracker.overflow_count:
            evidence = ()
        streak_key = (trigger_class, *facts)
        if streak_key != self._streak_key:
            self._streak_key = streak_key
            self._streak_count = 1
            self._first_observed_step = context.policy_step
            _, self._provisional_origin = self._event_ids(
                trigger_class,
                facts,
                snapshot,
                self._first_observed_step,
            )
        else:
            self._streak_count += 1
        strong = any(item.evidence_kind in _STRONG_EVIDENCE for item in evidence)
        status = (
            DeviationStatus.CONFIRMED_DEVIATION
            if strong
            else DeviationStatus.ANOMALY_CANDIDATE
        )
        previous_status = self._recorded.get(streak_key)
        should_emit = (
            self._streak_count >= self.confirmation_count
            and (
                previous_status is None
                or (
                    previous_status is DeviationStatus.ANOMALY_CANDIDATE
                    and status is DeviationStatus.CONFIRMED_DEVIATION
                )
            )
        )
        if not should_emit:
            return
        identity = self._event_identities.get(streak_key)
        if identity is None:
            deviation_event_id, event_origin = self._event_ids(
                trigger_class,
                facts,
                snapshot,
                self._first_observed_step,
            )
            if self._provisional_origin is not None:
                event_origin = self._provisional_origin
            identity = (deviation_event_id, event_origin)
            self._event_identities[streak_key] = identity
        deviation_event_id, event_origin = identity
        trigger = ShadowTrigger(
            trigger_class=trigger_class,
            deviation_status=status,
            signature=tuple(
                sorted((*facts, *(item.evidence_id for item in evidence)))
            ),
            historical_failure_evidence=evidence,
            deviation_event_id=deviation_event_id,
            event_origin_parent_sha256=event_origin,
            policy_step=context.policy_step,
            observation_generation=reconciliation.observation_generation,
            relevant_fact_sha256=reconciliation.relevant_fact_sha256,
            source_graph_version=reconciliation.source_graph_version,
            certificate_state=reconciliation.certificate_state,
            snapshot=snapshot,
            observation=_copied_observation(context.observation),
        )
        self._recorded[streak_key] = status
        if status is DeviationStatus.CONFIRMED_DEVIATION:
            self.metrics.confirmed_deviations += 1
        else:
            self.metrics.anomaly_candidates += 1
        try:
            self.on_trigger(trigger)
        except Exception:
            self.metrics.trigger_callback_errors += 1
