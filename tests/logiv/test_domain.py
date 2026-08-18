from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.domain import (
    DomainError,
    FixedDomain,
    PreconditionsNotMet,
    apply_action,
    lint_domain,
    render_domain_pddl,
    render_problem_pddl,
    validate_state,
)
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    ObjectDecl,
    TaskProblem,
    TruthValue,
)


def problem() -> TaskProblem:
    return TaskProblem(
        name="domain-test",
        objects=(
            ObjectDecl("pot_1", "movable"),
            ObjectDecl("pot_2", "movable"),
            ObjectDecl("pot_1_start", "relative-region"),
            ObjectDecl("pot_2_start", "relative-region"),
            ObjectDecl("stove_surface", "surface"),
            ObjectDecl("drawer_region", "container-region"),
            ObjectDecl("drawer_access", "access"),
            ObjectDecl("stove", "switchable"),
        ),
        initial_state=frozenset(
            {
                Fact("at", ("pot_1", "pot_1_start")),
                Fact("at", ("pot_2", "pot_2_start")),
                Fact("handempty"),
                Fact("accessible", ("drawer_region", "drawer_access")),
                Fact("closed", ("drawer_access",)),
                Fact("powered-off", ("stove",)),
            }
        ),
        initial_false=frozenset(
            {
                Fact("at", ("pot_1", "stove_surface")),
                Fact("at", ("pot_2", "stove_surface")),
                Fact("holding", ("pot_1",)),
                Fact("holding", ("pot_2",)),
                Fact("open", ("drawer_access",)),
                Fact("powered-on", ("stove",)),
            }
        ),
        goal=frozenset(
            {
                Fact("at", ("pot_1", "stove_surface")),
                Fact("at", ("pot_2", "stove_surface")),
            }
        ),
    )


def test_place_on_moves_only_the_bound_object() -> None:
    domain = FixedDomain()
    task = problem()
    action = domain.ground(task, "place-on", ("pot_1", "pot_1_start", "stove_surface"))

    state = apply_action(task, task.initial_state, action)

    assert Fact("at", ("pot_1", "stove_surface")) in state
    assert Fact("at", ("pot_1", "pot_1_start")) not in state
    assert Fact("at", ("pot_2", "pot_2_start")) in state
    assert Fact("handempty") in state


def test_place_in_requires_open_access() -> None:
    domain = FixedDomain()
    task = problem()
    action = domain.ground(
        task,
        "place-in",
        ("pot_1", "pot_1_start", "drawer_region", "drawer_access"),
    )

    with pytest.raises(PreconditionsNotMet, match=r"open\(drawer_access\)"):
        apply_action(task, task.initial_state, action)


def test_place_held_in_releases_object_and_restores_handempty() -> None:
    domain = FixedDomain()
    task = problem()
    held_state = set(task.initial_state)
    held_state.remove(Fact("at", ("pot_1", "pot_1_start")))
    held_state.remove(Fact("handempty"))
    held_state.remove(Fact("closed", ("drawer_access",)))
    held_state.update({Fact("holding", ("pot_1",)), Fact("open", ("drawer_access",))})
    action = domain.ground(
        task,
        "place-held-in",
        ("pot_1", "drawer_region", "drawer_access"),
    )

    state = apply_action(task, frozenset(held_state), action)

    assert Fact("holding", ("pot_1",)) not in state
    assert Fact("at", ("pot_1", "drawer_region")) in state
    assert Fact("handempty") in state


def test_pick_establishes_holding_and_clears_location_and_handempty() -> None:
    domain = FixedDomain()
    task = problem()
    action = domain.ground(task, "pick", ("pot_1", "pot_1_start"))

    state = apply_action(task, task.initial_state, action)

    assert Fact("holding", ("pot_1",)) in state
    assert Fact("at", ("pot_1", "pot_1_start")) not in state
    assert Fact("handempty") not in state


def test_grounding_rejects_wrong_argument_type() -> None:
    with pytest.raises(DomainError, match="parameter .* expects surface"):
        FixedDomain().ground(problem(), "place-on", ("pot_1", "pot_1_start", "drawer_access"))


def test_grounding_rejects_add_delete_overlap_after_binding() -> None:
    with pytest.raises(DomainError, match="grounded Add/Del overlap"):
        FixedDomain().ground(
            problem(), "place-on", ("pot_1", "stove_surface", "stove_surface")
        )


@pytest.mark.parametrize(
    "mutated",
    [
        {Fact("at", ("pot_1", "pot_1_start")), Fact("at", ("pot_1", "stove_surface"))},
        {Fact("holding", ("pot_1",)), Fact("handempty")},
        {Fact("holding", ("pot_1",)), Fact("holding", ("pot_2",))},
    ],
)
def test_state_lint_rejects_exactly_one_and_gripper_conflicts(mutated: set[Fact]) -> None:
    task = problem()
    state = {
        fact
        for fact in task.initial_state
        if fact.predicate not in {"at", "handempty", "holding"}
    }
    state.update(mutated)

    with pytest.raises(DomainError):
        validate_state(task, frozenset(state))


def test_pddl_is_task_independent_and_problem_is_typed() -> None:
    domain_pddl = render_domain_pddl()
    problem_pddl = render_problem_pddl(problem())

    assert "(:requirements :strips :typing :negative-preconditions)" in domain_pddl
    assert "(:action place-held-in" in domain_pddl
    assert "(:action pick" in domain_pddl
    assert "pot_1" not in domain_pddl
    assert "pot_1 pot_2 - movable" in problem_pddl
    assert "(powered-off stove)" in problem_pddl
    assert "(:goal (and (at pot_1 stove_surface) (at pot_2 stove_surface)))" in problem_pddl
    assert "(not (at pot_1 stove_surface))" not in problem_pddl


def test_negative_goal_is_explicitly_rendered() -> None:
    task = replace(problem(), negative_goal=frozenset({Fact("holding", ("pot_1",))}))

    problem_pddl = render_problem_pddl(task)

    assert "(not (holding pot_1))" in problem_pddl


def test_domain_lint_accepts_fixed_schemas() -> None:
    lint_domain()


def test_snapshot_uses_explicit_three_valued_gate_semantics() -> None:
    known_true = Fact("open", ("drawer_access",))
    known_false = Fact("closed", ("drawer_access",))
    unknown = Fact("holding", ("pot_1",))
    snapshot = FactSnapshot(
        epoch_id=7,
        true_facts=frozenset({known_true}),
        false_facts=frozenset({known_false}),
        evidence_hash="sha256:test",
    )

    assert snapshot.truth(known_true) is TruthValue.TRUE
    assert snapshot.truth(known_false) is TruthValue.FALSE
    assert snapshot.truth(unknown) is TruthValue.UNKNOWN
    assert snapshot.satisfies(positive={known_true}, negative={known_false})
    assert not snapshot.satisfies(positive={unknown})
    assert not snapshot.satisfies(negative={unknown})


def test_snapshot_rejects_conflicting_true_and_false_evidence() -> None:
    fact = Fact("open", ("drawer_access",))

    with pytest.raises(ValueError, match="both TRUE and FALSE"):
        FactSnapshot(
            epoch_id=7,
            true_facts=frozenset({fact}),
            false_facts=frozenset({fact}),
            evidence_hash="sha256:test",
        )


def test_checked_in_domain_matches_the_python_semantics() -> None:
    artifact = Path("configs/logiv/origin/domain.pddl")

    assert artifact.read_text(encoding="utf-8") == render_domain_pddl()
