from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import FrozenSet, Iterable, Mapping, Sequence, Tuple

from pi05_libero_repro.logiv.model import Fact, GroundAction, TaskProblem


class DomainError(ValueError):
    pass


class PreconditionsNotMet(DomainError):
    pass


TYPE_PARENT = {
    "object": None,
    "movable": "object",
    "location": "object",
    "surface": "location",
    "container-region": "location",
    "relative-region": "location",
    "access": "object",
    "switchable": "object",
}

PREDICATES: Mapping[str, Tuple[str, ...]] = {
    "at": ("movable", "location"),
    "holding": ("movable",),
    "handempty": (),
    "open": ("access",),
    "closed": ("access",),
    "powered-on": ("switchable",),
    "powered-off": ("switchable",),
    "accessible": ("container-region", "access"),
}


def _is_subtype(actual: str, expected: str) -> bool:
    while actual is not None:
        if actual == expected:
            return True
        actual = TYPE_PARENT.get(actual)
    return False


@dataclass(frozen=True)
class FactTemplate:
    predicate: str
    arguments: Tuple[str, ...] = ()

    def bind(self, values: Mapping[str, str]) -> Fact:
        return Fact(self.predicate, tuple(values.get(item, item) for item in self.arguments))

    def pddl(self, negated: bool = False) -> str:
        suffix = " " + " ".join(self.arguments) if self.arguments else ""
        atom = f"({self.predicate}{suffix})"
        return f"(not {atom})" if negated else atom


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: Tuple[Tuple[str, str], ...]
    preconditions: Tuple[FactTemplate, ...]
    add_effects: Tuple[FactTemplate, ...]
    del_effects: Tuple[FactTemplate, ...]
    repeatable: bool = True
    negative_preconditions: Tuple[FactTemplate, ...] = ()

    def ground(self, problem: TaskProblem, arguments: Sequence[str]) -> GroundAction:
        if len(arguments) != len(self.parameters):
            raise DomainError(
                f"{self.name} expects {len(self.parameters)} arguments, got {len(arguments)}"
            )
        object_types = problem.object_types
        values: dict[str, str] = {}
        for (parameter, expected), argument in zip(self.parameters, arguments):
            actual = object_types.get(argument)
            if actual is None:
                raise DomainError(f"unknown object: {argument}")
            if not _is_subtype(actual, expected):
                raise DomainError(
                    f"parameter {parameter} expects {expected}, got {argument} - {actual}"
                )
            values[parameter] = argument
        action = GroundAction(
            schema=self.name,
            arguments=tuple(arguments),
            preconditions=frozenset(item.bind(values) for item in self.preconditions),
            add_effects=frozenset(item.bind(values) for item in self.add_effects),
            del_effects=frozenset(item.bind(values) for item in self.del_effects),
            repeatable=self.repeatable,
            negative_preconditions=frozenset(
                item.bind(values) for item in self.negative_preconditions
            ),
        )
        if action.add_effects & action.del_effects:
            raise DomainError(f"grounded Add/Del overlap in {action.pddl()}")
        return action


def _template(predicate: str, *arguments: str) -> FactTemplate:
    return FactTemplate(predicate, tuple(arguments))


SCHEMAS = (
    ActionSchema(
        "pick",
        (("?object", "movable"), ("?from", "location")),
        (_template("at", "?object", "?from"), _template("handempty")),
        (_template("holding", "?object"),),
        (_template("at", "?object", "?from"), _template("handempty")),
    ),
    ActionSchema(
        "place-on",
        (("?object", "movable"), ("?from", "location"), ("?to", "surface")),
        (_template("at", "?object", "?from"), _template("handempty")),
        (_template("at", "?object", "?to"),),
        (_template("at", "?object", "?from"),),
    ),
    ActionSchema(
        "place-in",
        (
            ("?object", "movable"),
            ("?from", "location"),
            ("?to", "container-region"),
            ("?access", "access"),
        ),
        (
            _template("at", "?object", "?from"),
            _template("handempty"),
            _template("accessible", "?to", "?access"),
            _template("open", "?access"),
        ),
        (_template("at", "?object", "?to"),),
        (_template("at", "?object", "?from"),),
    ),
    ActionSchema(
        "place-relative",
        (
            ("?object", "movable"),
            ("?from", "location"),
            ("?to", "relative-region"),
        ),
        (_template("at", "?object", "?from"), _template("handempty")),
        (_template("at", "?object", "?to"),),
        (_template("at", "?object", "?from"),),
    ),
    ActionSchema(
        "open-access",
        (("?access", "access"),),
        (_template("closed", "?access"), _template("handempty")),
        (_template("open", "?access"),),
        (_template("closed", "?access"),),
    ),
    ActionSchema(
        "close-access",
        (("?access", "access"),),
        (_template("open", "?access"), _template("handempty")),
        (_template("closed", "?access"),),
        (_template("open", "?access"),),
    ),
    ActionSchema(
        "turn-on",
        (("?device", "switchable"),),
        (_template("powered-off", "?device"),),
        (_template("powered-on", "?device"),),
        (_template("powered-off", "?device"),),
    ),
    ActionSchema(
        "turn-off",
        (("?device", "switchable"),),
        (_template("powered-on", "?device"),),
        (_template("powered-off", "?device"),),
        (_template("powered-on", "?device"),),
    ),
    ActionSchema(
        "put-down",
        (("?object", "movable"), ("?to", "surface")),
        (_template("holding", "?object"),),
        (_template("at", "?object", "?to"), _template("handempty")),
        (_template("holding", "?object"),),
    ),
    ActionSchema(
        "place-held-on",
        (("?object", "movable"), ("?to", "surface")),
        (_template("holding", "?object"),),
        (_template("at", "?object", "?to"), _template("handempty")),
        (_template("holding", "?object"),),
    ),
    ActionSchema(
        "place-held-in",
        (
            ("?object", "movable"),
            ("?to", "container-region"),
            ("?access", "access"),
        ),
        (
            _template("holding", "?object"),
            _template("accessible", "?to", "?access"),
            _template("open", "?access"),
        ),
        (_template("at", "?object", "?to"), _template("handempty")),
        (_template("holding", "?object"),),
    ),
    ActionSchema(
        "place-held-relative",
        (("?object", "movable"), ("?to", "relative-region")),
        (_template("holding", "?object"),),
        (_template("at", "?object", "?to"), _template("handempty")),
        (_template("holding", "?object"),),
    ),
)


def lint_domain() -> None:
    names: set[str] = set()
    for schema in SCHEMAS:
        if schema.name in names:
            raise DomainError(f"duplicate action schema: {schema.name}")
        names.add(schema.name)
        parameters = dict(schema.parameters)
        if len(parameters) != len(schema.parameters):
            raise DomainError(f"duplicate parameter in schema: {schema.name}")
        for parameter, type_name in schema.parameters:
            if not parameter.startswith("?") or type_name not in TYPE_PARENT:
                raise DomainError(f"invalid parameter {parameter} - {type_name} in {schema.name}")
        for template in (
            schema.preconditions
            + schema.negative_preconditions
            + schema.add_effects
            + schema.del_effects
        ):
            expected = PREDICATES.get(template.predicate)
            if expected is None or len(expected) != len(template.arguments):
                raise DomainError(f"invalid predicate template in {schema.name}: {template}")
            for argument, expected_type in zip(template.arguments, expected):
                actual_type = parameters.get(argument)
                if actual_type is None or not _is_subtype(actual_type, expected_type):
                    raise DomainError(
                        f"invalid argument {argument} for {template.predicate} in {schema.name}"
                    )
        if set(schema.add_effects) & set(schema.del_effects):
            raise DomainError(f"Add/Del overlap in schema: {schema.name}")


class FixedDomain:
    def __init__(self) -> None:
        lint_domain()
        self.schemas = {item.name: item for item in SCHEMAS}

    def ground(self, problem: TaskProblem, schema: str, arguments: Sequence[str]) -> GroundAction:
        try:
            action_schema = self.schemas[schema]
        except KeyError as error:
            raise DomainError(f"unknown action schema: {schema}") from error
        return action_schema.ground(problem, arguments)


def _validate_fact(problem: TaskProblem, fact: Fact) -> None:
    expected = PREDICATES.get(fact.predicate)
    if expected is None:
        raise DomainError(f"unknown predicate: {fact.predicate}")
    if len(fact.arguments) != len(expected):
        raise DomainError(f"wrong arity for {fact.predicate}")
    object_types = problem.object_types
    for argument, expected_type in zip(fact.arguments, expected):
        actual = object_types.get(argument)
        if actual is None or not _is_subtype(actual, expected_type):
            raise DomainError(
                f"predicate {fact.predicate} expects {expected_type}, got {argument} - {actual}"
            )


def validate_state(problem: TaskProblem, state: FrozenSet[Fact]) -> None:
    for fact in state | problem.initial_false | problem.goal | problem.negative_goal:
        _validate_fact(problem, fact)

    types = problem.object_types
    movable = [name for name, type_name in types.items() if _is_subtype(type_name, "movable")]
    holdings = [fact.arguments[0] for fact in state if fact.predicate == "holding"]
    if len(holdings) > 1:
        raise DomainError("single-arm state cannot hold more than one object")
    for object_name in movable:
        locations = [
            fact
            for fact in state
            if fact.predicate == "at" and fact.arguments[0] == object_name
        ]
        count = len(locations) + holdings.count(object_name)
        if count != 1:
            raise DomainError(f"{object_name} must have exactly one location or holding fact")

    handempty = Fact("handempty") in state
    if handempty == bool(holdings):
        raise DomainError("handempty and holding facts are inconsistent")

    for name, type_name in types.items():
        if _is_subtype(type_name, "access"):
            count = int(Fact("open", (name,)) in state) + int(
                Fact("closed", (name,)) in state
            )
            if count != 1:
                raise DomainError(f"{name} must be exactly open or closed")
        if _is_subtype(type_name, "switchable"):
            count = int(Fact("powered-on", (name,)) in state) + int(
                Fact("powered-off", (name,)) in state
            )
            if count != 1:
                raise DomainError(f"{name} must be exactly powered-on or powered-off")


def apply_action(
    problem: TaskProblem, state: FrozenSet[Fact], action: GroundAction
) -> FrozenSet[Fact]:
    missing = action.preconditions - state
    if missing:
        rendered = ", ".join(str(fact) for fact in sorted(missing))
        raise PreconditionsNotMet(f"missing preconditions: {rendered}")
    violated_negative = action.negative_preconditions & state
    if violated_negative:
        rendered = ", ".join(str(fact) for fact in sorted(violated_negative))
        raise PreconditionsNotMet(f"violated negative preconditions: {rendered}")
    next_state = frozenset((state - action.del_effects) | action.add_effects)
    validate_state(problem, next_state)
    return next_state


def _conjunction(items: Iterable[str]) -> str:
    values = tuple(items)
    if not values:
        return "(and)"
    return f"(and {' '.join(values)})"


def _render_action(schema: ActionSchema) -> str:
    parameters = " ".join(f"{name} - {type_name}" for name, type_name in schema.parameters)
    preconditions = [item.pddl() for item in schema.preconditions]
    preconditions.extend(item.pddl(negated=True) for item in schema.negative_preconditions)
    precondition = _conjunction(preconditions)
    effects = [item.pddl() for item in schema.add_effects]
    effects.extend(item.pddl(negated=True) for item in schema.del_effects)
    return "\n".join(
        (
            f"  (:action {schema.name}",
            f"    :parameters ({parameters})",
            f"    :precondition {precondition}",
            f"    :effect {_conjunction(effects)}",
            "  )",
        )
    )


def render_domain_pddl() -> str:
    lint_domain()
    predicates = "\n".join(
        "    " + Fact(name, tuple(f"?x{index}" for index in range(len(types)))).pddl()
        if not types
        else "    ("
        + name
        + " "
        + " ".join(f"?x{index} - {type_name}" for index, type_name in enumerate(types))
        + ")"
        for name, types in PREDICATES.items()
    )
    actions = "\n".join(_render_action(item) for item in SCHEMAS)
    return (
        "(define (domain logiv-libero)\n"
        "  (:requirements :strips :typing :negative-preconditions)\n"
        "  (:types movable location access switchable - object\n"
        "          surface container-region relative-region - location)\n"
        "  (:predicates\n"
        f"{predicates}\n"
        "  )\n"
        f"{actions}\n"
        ")\n"
    )


def render_problem_pddl(problem: TaskProblem) -> str:
    validate_state(problem, problem.initial_state)
    object_lines = []
    ordered_objects = sorted(problem.objects, key=lambda item: (item.type_name, item.name))
    for type_name, values in groupby(ordered_objects, key=lambda item: item.type_name):
        object_lines.append("    " + " ".join(item.name for item in values) + f" - {type_name}")
    initial = "\n".join(f"    {fact.pddl()}" for fact in sorted(problem.initial_state))
    goal_literals = [fact.pddl() for fact in sorted(problem.goal)]
    goal_literals.extend(f"(not {fact.pddl()})" for fact in sorted(problem.negative_goal))
    goal = _conjunction(goal_literals)
    return (
        f"(define (problem {problem.name})\n"
        "  (:domain logiv-libero)\n"
        "  (:objects\n"
        + "\n".join(object_lines)
        + "\n  )\n"
        "  (:init\n"
        f"{initial}\n"
        "  )\n"
        f"  (:goal {goal})\n"
        ")\n"
    )
