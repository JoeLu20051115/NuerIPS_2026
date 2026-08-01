from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import AbstractSet, FrozenSet, Tuple


_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


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


@dataclass(frozen=True)
class FactSnapshot:
    """Evidence-backed partial symbolic state for one physical epoch."""

    epoch_id: int
    true_facts: FrozenSet[Fact]
    false_facts: FrozenSet[Fact]
    evidence_hash: str

    def __post_init__(self) -> None:
        conflict = self.true_facts & self.false_facts
        if conflict:
            rendered = ", ".join(str(fact) for fact in sorted(conflict))
            raise ValueError(f"facts cannot be both TRUE and FALSE: {rendered}")

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
