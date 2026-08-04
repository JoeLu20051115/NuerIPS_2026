from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import AbstractSet, FrozenSet, Tuple


_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_pddl_fact(value: str) -> Fact:
    if not isinstance(value, str) or not value.startswith("(") or not value.endswith(")"):
        raise ValueError(f"non-canonical PDDL fact: {value!r}")
    tokens = value[1:-1].split()
    if not tokens:
        raise ValueError(f"non-canonical PDDL fact: {value!r}")
    fact = Fact(tokens[0], tuple(tokens[1:]))
    if fact.pddl() != value:
        raise ValueError(f"non-canonical PDDL fact: {value!r}")
    return fact


def fact_pddl_sort_key(fact: Fact) -> str:
    return fact.pddl()


def fact_universe_sha256(version: str, facts: FrozenSet[Fact]) -> str:
    payload = {
        "facts": [fact.pddl() for fact in sorted(facts, key=fact_pddl_sort_key)],
        "version": version,
    }
    return hashlib.sha256(
        b"LOGIV_FACT_UNIVERSE_V1\0" + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, order=True)
class Fact:
    predicate: str
    arguments: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SYMBOL.fullmatch(self.predicate):
            raise ValueError(f"invalid predicate: {self.predicate!r}")
        if any(not _SYMBOL.fullmatch(argument) for argument in self.arguments):
            raise ValueError(f"invalid fact arguments: {self.arguments!r}")

    def pddl(self) -> str:
        suffix = " " + " ".join(self.arguments) if self.arguments else ""
        return f"({self.predicate}{suffix})"

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.arguments)})"


@dataclass(frozen=True, order=True)
class ObjectDecl:
    name: str
    type_name: str

    def __post_init__(self) -> None:
        if not _SYMBOL.fullmatch(self.name) or not _SYMBOL.fullmatch(self.type_name):
            raise ValueError(f"invalid object declaration: {self.name!r} - {self.type_name!r}")


class TruthValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class GoalMode(str, Enum):
    METADATA_ASSISTED = "METADATA_ASSISTED"
    GOAL_PREDICTION = "GOAL_PREDICTION"


class ContextPhase(str, Enum):
    INITIAL = "INITIAL"
    INITIAL_FACTS = "INITIAL_FACTS"
    PREINSTALL_VAL = "PREINSTALL_VAL"
    PRE_DISPATCH_FACTS = "PRE_DISPATCH_FACTS"
    POST_STOP_FACTS = "POST_STOP_FACTS"
    FINAL_GOAL = "FINAL_GOAL"
    RECOVERY_VAL = "RECOVERY_VAL"


@dataclass(frozen=True)
class ContextEnvelope:
    phase: ContextPhase
    goal_mode: GoalMode
    request_id: str
    request_generation: int
    episode_id: str
    goal_id: str
    goal_epoch: int
    epoch_id: int
    graph_version: str | None
    occurrence_id: str | None
    attempt_id: str | None
    certificate_hash: str | None
    safety_epoch: int | None

    def __post_init__(self) -> None:
        if not self.request_id or not self.episode_id or not self.goal_id:
            raise ValueError("context IDs must be nonempty")
        if min(self.request_generation, self.goal_epoch, self.epoch_id) < 0:
            raise ValueError("context generations and epochs must be nonnegative")
        if self.safety_epoch is not None and self.safety_epoch < 0:
            raise ValueError("safety_epoch must be nonnegative or None")

    def payload(self) -> dict[str, str | int | None]:
        return {
            "phase": self.phase.value,
            "goal_mode": self.goal_mode.value,
            "request_id": self.request_id,
            "request_generation": self.request_generation,
            "episode_id": self.episode_id,
            "goal_id": self.goal_id,
            "goal_epoch": self.goal_epoch,
            "epoch_id": self.epoch_id,
            "graph_version": self.graph_version,
            "occurrence_id": self.occurrence_id,
            "attempt_id": self.attempt_id,
            "certificate_hash": self.certificate_hash,
            "safety_epoch": self.safety_epoch,
        }


@dataclass(frozen=True)
class FactSnapshot:
    """Evidence-backed partial symbolic state for one physical epoch."""

    epoch_id: int
    true_facts: FrozenSet[Fact]
    false_facts: FrozenSet[Fact]
    evidence_hash: str
    fact_universe: FrozenSet[Fact] | None = None
    fact_universe_version: str | None = None
    fact_universe_sha256: str | None = None
    evidence_payload_json: str | None = None

    def __post_init__(self) -> None:
        conflict = self.true_facts & self.false_facts
        if conflict:
            rendered = ", ".join(str(fact) for fact in sorted(conflict))
            raise ValueError(f"facts cannot be both TRUE and FALSE: {rendered}")
        audit_fields = (
            self.fact_universe,
            self.fact_universe_version,
            self.fact_universe_sha256,
            self.evidence_payload_json,
        )
        if all(value is None for value in audit_fields):
            return
        if any(value is None for value in audit_fields):
            raise ValueError("FactSnapshot audit fields must be all present or all null")
        if self.epoch_id < 0:
            raise ValueError("audited fact epoch must be nonnegative")
        if not isinstance(self.true_facts, frozenset) or not isinstance(
            self.false_facts, frozenset
        ):
            raise ValueError("audited fact partitions must be frozensets")
        universe = self.fact_universe
        if not isinstance(universe, frozenset) or not all(
            isinstance(fact, Fact) for fact in universe
        ):
            raise ValueError("audited fact universe must be a frozenset of canonical facts")
        if not all(isinstance(fact, Fact) for fact in self.true_facts | self.false_facts):
            raise ValueError("audited fact partitions contain a non-canonical fact")
        known = self.true_facts | self.false_facts
        if not known <= universe:
            raise ValueError("known fact partition contains members outside the fact universe")
        version = self.fact_universe_version
        if not isinstance(version, str) or not version:
            raise ValueError("fact universe version must be nonempty")
        expected_universe_hash = fact_universe_sha256(version, universe)
        if self.fact_universe_sha256 != expected_universe_hash:
            raise ValueError("fact universe hash mismatch")
        if not _SHA256.fullmatch(self.evidence_hash):
            raise ValueError("audited fact evidence hash must be lowercase SHA-256")
        payload_json = self.evidence_payload_json
        if not isinstance(payload_json, str):
            raise ValueError("fact evidence payload must be canonical JSON")
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("fact evidence payload is not valid JSON") from error
        if _canonical_json(payload) != payload_json:
            raise ValueError("fact evidence payload is not canonical JSON")
        if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != self.evidence_hash:
            raise ValueError("fact evidence payload hash mismatch")
        expected_keys = {
            "epoch_id",
            "observation_hash",
            "values",
            "dominance_overrides",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("fact evidence payload fields mismatch")
        if payload["epoch_id"] != self.epoch_id:
            raise ValueError("fact evidence epoch mismatch")
        if not isinstance(payload["observation_hash"], str) or not _SHA256.fullmatch(
            payload["observation_hash"]
        ):
            raise ValueError("fact evidence observation hash is invalid")
        values = payload["values"]
        if not isinstance(values, list):
            raise ValueError("fact evidence values must be a complete partition")
        parsed: dict[Fact, TruthValue] = {}
        for item in values:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("fact evidence value record is malformed")
            fact = parse_pddl_fact(item[0])
            if fact in parsed:
                raise ValueError("duplicate fact in evidence partition")
            try:
                parsed[fact] = TruthValue(item[1])
            except (TypeError, ValueError) as error:
                raise ValueError("fact evidence contains an invalid truth value") from error
        if values != sorted(values, key=lambda item: item[0]):
            raise ValueError("fact evidence values are not in canonical sorted order")
        if frozenset(parsed) != universe:
            raise ValueError("fact evidence partition does not cover its universe")
        expected_true = frozenset(
            fact for fact, value in parsed.items() if value is TruthValue.TRUE
        )
        expected_false = frozenset(
            fact for fact, value in parsed.items() if value is TruthValue.FALSE
        )
        if expected_true != self.true_facts or expected_false != self.false_facts:
            raise ValueError("fact evidence partition conflicts with saved facts")
        overrides = payload["dominance_overrides"]
        if not isinstance(overrides, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) and value for value in item)
            for item in overrides
        ):
            raise ValueError("fact evidence dominance overrides are malformed")
        seen_overrides: set[tuple[Fact, Fact, str]] = set()
        for source_text, target_text, kind in overrides:
            source = parse_pddl_fact(source_text)
            target = parse_pddl_fact(target_text)
            if source not in universe or target not in universe:
                raise ValueError("fact evidence dominance override is outside its universe")
            if kind != "reliable-holding-over-at":
                raise ValueError("fact evidence dominance override kind is invalid")
            if (
                source.predicate != "holding"
                or target.predicate != "at"
                or source.arguments != target.arguments[:1]
            ):
                raise ValueError("fact evidence dominance override facts are incompatible")
            override = (source, target, kind)
            if override in seen_overrides:
                raise ValueError("duplicate fact evidence dominance override")
            seen_overrides.add(override)
        if overrides != sorted(overrides):
            raise ValueError("fact evidence dominance overrides are not canonical")

    def truth(self, fact: Fact) -> TruthValue:
        if fact in self.true_facts:
            return TruthValue.TRUE
        if fact in self.false_facts:
            return TruthValue.FALSE
        return TruthValue.UNKNOWN

    def satisfies(
        self,
        *,
        positive: AbstractSet[Fact] = frozenset(),
        negative: AbstractSet[Fact] = frozenset(),
    ) -> bool:
        return positive <= self.true_facts and negative <= self.false_facts

    def unknown(self, required: AbstractSet[Fact]) -> FrozenSet[Fact]:
        return frozenset(required - self.true_facts - self.false_facts)


@dataclass(frozen=True)
class TaskProblem:
    name: str
    objects: Tuple[ObjectDecl, ...]
    initial_state: FrozenSet[Fact]
    goal: FrozenSet[Fact]
    initial_false: FrozenSet[Fact] = frozenset()
    negative_goal: FrozenSet[Fact] = frozenset()

    def __post_init__(self) -> None:
        if not _SYMBOL.fullmatch(self.name):
            raise ValueError(f"invalid problem name: {self.name!r}")
        names = [item.name for item in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("duplicate object declaration")
        conflict = self.initial_state & self.initial_false
        if conflict:
            raise ValueError("initial facts cannot be both TRUE and FALSE")
        goal_conflict = self.goal & self.negative_goal
        if goal_conflict:
            raise ValueError("goal facts cannot be both positive and negative")

    @property
    def object_types(self) -> dict[str, str]:
        return {item.name: item.type_name for item in self.objects}


@dataclass(frozen=True)
class GroundAction:
    schema: str
    arguments: Tuple[str, ...]
    preconditions: FrozenSet[Fact]
    add_effects: FrozenSet[Fact]
    del_effects: FrozenSet[Fact]
    repeatable: bool
    negative_preconditions: FrozenSet[Fact] = frozenset()

    def pddl(self) -> str:
        suffix = " " + " ".join(self.arguments) if self.arguments else ""
        return f"({self.schema}{suffix})"

    @property
    def retry_key(self) -> Tuple[str, Tuple[str, ...]]:
        return (self.schema, self.arguments)


@dataclass(frozen=True)
class SignedGoal:
    positive: FrozenSet[Fact]
    negative: FrozenSet[Fact] = frozenset()

    def __post_init__(self) -> None:
        if self.positive & self.negative:
            raise ValueError("goal facts cannot be both positive and negative")


@dataclass(frozen=True)
class FrozenGoal:
    goal_id: str
    goal_epoch: int
    literals: SignedGoal
    source: str


@dataclass(frozen=True)
class CandidateSubtask:
    occurrence_id: str
    rough_rank: int
    action: GroundAction
    instruction: str
    evidence_source: str
    lineage_root: str


@dataclass(frozen=True)
class Proposal:
    task_id: int
    task_name: str
    source_bddl: str
    source_bddl_sha256: str
    epoch_id: int
    goal_mode: GoalMode
    provider: str
    prompt_version: str
    registered_objects: Tuple[ObjectDecl, ...]
    initial_snapshot: FactSnapshot
    candidate_subtasks: Tuple[CandidateSubtask, ...]
    grounded_goal: SignedGoal | None


@dataclass(frozen=True)
class ProposalPackage:
    proposal: Proposal
    frozen_goal: FrozenGoal
    problem: TaskProblem
