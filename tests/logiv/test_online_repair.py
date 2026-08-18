from __future__ import annotations

from dataclasses import replace

from pi05_libero_repro.logiv.dag import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeKind,
    SignedLiteral,
)
from pi05_libero_repro.logiv.model import (
    Fact,
    FactSnapshot,
    GroundAction,
    ObjectDecl,
    TaskProblem,
)
from pi05_libero_repro.logiv.online_repair import (
    OnlineDeviationKind,
    OnlineGraphDeviationDetector,
)


SOURCE = Fact("at", ("object_1", "table_init_region"))
TARGET = Fact("at", ("object_1", "target_region"))
RECOVERY = Fact("at", ("object_1", "table_recovery_surface"))
HOLDING = Fact("holding", ("object_1",))
HAND_EMPTY = Fact("handempty", ())
MILESTONE = Fact("at", ("object_2", "milestone_region"))


def _problem_and_graph() -> tuple[TaskProblem, CausalGraph]:
    action = GroundAction(
        schema="place-on",
        arguments=("object_1", "table_init_region", "target_region"),
        preconditions=frozenset({SOURCE}),
        add_effects=frozenset({TARGET}),
        del_effects=frozenset({SOURCE}),
        repeatable=True,
    )
    support = frozenset({SignedLiteral(SOURCE, True)})
    goal_support = frozenset({SignedLiteral(TARGET, True)})
    graph = CausalGraph(
        graph_version="graph-v1",
        graph_hash="a" * 64,
        source_epoch=0,
        certificate_hash="b" * 64,
        nodes=(
            GraphNode("INIT", NodeKind.INIT, 0),
            GraphNode("o000", NodeKind.ACTION, 1, action),
            GraphNode("GOAL", NodeKind.GOAL, 2),
        ),
        edges=(
            GraphEdge("INIT", "o000", support_literals=support),
            GraphEdge("o000", "GOAL", support_literals=goal_support),
        ),
        causal_links=(),
        canonical_agenda=("o000",),
    )
    problem = TaskProblem(
        name="online-test",
        objects=(ObjectDecl("object_1", "movable"),),
        initial_state=frozenset({SOURCE}),
        initial_false=frozenset({TARGET, RECOVERY, HOLDING}),
        goal=frozenset({TARGET}),
    )
    return problem, graph


def _snapshot(
    step: int,
    *,
    true: frozenset[Fact],
    false: frozenset[Fact],
) -> FactSnapshot:
    return FactSnapshot(step, true, false, f"evidence-{step}")


def _state(step: int, status: str, *, certificate: str = "CURRENT") -> dict:
    return {
        "policy_step": step,
        "graph_hash": "a" * 64,
        "certificate_state": certificate,
        "nodes": [
            {"node_id": "INIT", "status": "COMPLETED"},
            {"node_id": "o000", "status": status},
            {"node_id": "GOAL", "status": "BLOCKED"},
        ],
    }


def _detector(**overrides) -> OnlineGraphDeviationDetector:
    problem, graph = _problem_and_graph()
    settings = {
        "confirmation_count": 2,
        "min_intervention_step": 0,
        "stall_steps": 100,
    }
    settings.update(overrides)
    return OnlineGraphDeviationDetector(problem, graph, **settings)


def test_recovery_surface_requires_repeated_true_and_strict_confirmation() -> None:
    detector = _detector()
    observed = _snapshot(
        10,
        true=frozenset({RECOVERY}),
        false=frozenset({SOURCE, TARGET, HOLDING}),
    )
    strict_calls = 0

    def strict_reader(_observation):
        nonlocal strict_calls
        strict_calls += 1
        return observed

    assert detector.observe(
        snapshot=observed,
        graph_state=_state(10, "ACTIVE"),
        observation={"frame": 10},
        strict_snapshot_reader=strict_reader,
    ) is None
    request = detector.observe(
        snapshot=observed,
        graph_state=_state(15, "ACTIVE"),
        observation={"frame": 15},
        strict_snapshot_reader=strict_reader,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE
    assert request.first_observed_step == 10
    assert request.policy_step == 15
    assert request.signature == (RECOVERY.pddl(),)
    assert strict_calls == 1


def test_recovery_surface_can_confirm_faster_without_weakening_stall() -> None:
    detector = _detector(
        confirmation_count=3,
        recovery_surface_confirmation_count=1,
        stall_steps=5,
    )
    recovery = _snapshot(
        10,
        true=frozenset({RECOVERY}),
        false=frozenset({SOURCE, TARGET, HOLDING}),
    )

    request = detector.observe(
        snapshot=recovery,
        graph_state=_state(10, "ACTIVE"),
        observation={"frame": 10},
        strict_snapshot_reader=lambda _observation: recovery,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE

    stall_detector = _detector(
        confirmation_count=3,
        recovery_surface_confirmation_count=1,
        stall_steps=5,
    )
    stalled = _snapshot(
        0,
        true=frozenset({SOURCE, HAND_EMPTY}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )
    assert stall_detector.observe(
        snapshot=stalled,
        graph_state=_state(0, "ACTIVE"),
        observation={"frame": 0},
        strict_snapshot_reader=lambda _observation: stalled,
    ) is None
    assert stall_detector.observe(
        snapshot=stalled,
        graph_state=_state(5, "ACTIVE"),
        observation={"frame": 5},
        strict_snapshot_reader=lambda _observation: stalled,
    ) is None
    assert stall_detector.observe(
        snapshot=stalled,
        graph_state=_state(10, "ACTIVE"),
        observation={"frame": 10},
        strict_snapshot_reader=lambda _observation: stalled,
    ) is None
    stall_request = stall_detector.observe(
        snapshot=stalled,
        graph_state=_state(15, "ACTIVE"),
        observation={"frame": 15},
        strict_snapshot_reader=lambda _observation: stalled,
    )

    assert stall_request is not None
    assert stall_request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_recovery_surface_can_require_prior_goal_progress() -> None:
    problem, graph = _problem_and_graph()
    problem = replace(
        problem,
        goal=frozenset({TARGET, MILESTONE}),
        initial_false=problem.initial_false | frozenset({MILESTONE}),
    )
    detector = OnlineGraphDeviationDetector(
        problem,
        graph,
        confirmation_count=2,
        min_intervention_step=0,
        stall_steps=100,
        recovery_requires_achieved_goal=True,
    )
    no_progress = _snapshot(
        10,
        true=frozenset({RECOVERY}),
        false=frozenset({SOURCE, TARGET, MILESTONE, HOLDING}),
    )

    for step in (10, 15):
        assert detector.observe(
            snapshot=no_progress,
            graph_state=_state(step, "ACTIVE"),
            observation={"step": step},
            strict_snapshot_reader=lambda _obs: no_progress,
        ) is None

    progressed = _snapshot(
        20,
        true=frozenset({TARGET, RECOVERY}),
        false=frozenset({SOURCE, MILESTONE, HOLDING}),
    )
    assert detector.observe(
        snapshot=progressed,
        graph_state=_state(20, "ACTIVE"),
        observation={"step": 20},
        strict_snapshot_reader=lambda _obs: progressed,
    ) is None
    request = detector.observe(
        snapshot=progressed,
        graph_state=_state(25, "ACTIVE"),
        observation={"step": 25},
        strict_snapshot_reader=lambda _obs: progressed,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.UNPLANNED_RECOVERY_SURFACE
    assert request.signature == (RECOVERY.pddl(),)


def test_recovery_goal_progress_must_match_strict_snapshot() -> None:
    problem, graph = _problem_and_graph()
    problem = replace(
        problem,
        goal=frozenset({TARGET, MILESTONE}),
        initial_false=problem.initial_false | frozenset({MILESTONE}),
    )
    detector = OnlineGraphDeviationDetector(
        problem,
        graph,
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=100,
        recovery_requires_achieved_goal=True,
    )
    advisory = _snapshot(
        20,
        true=frozenset({TARGET, RECOVERY}),
        false=frozenset({SOURCE, MILESTONE, HOLDING}),
    )
    strict = _snapshot(
        20,
        true=frozenset({RECOVERY}),
        false=frozenset({SOURCE, TARGET, MILESTONE, HOLDING}),
    )

    assert detector.observe(
        snapshot=advisory,
        graph_state=_state(20, "ACTIVE"),
        observation={"step": 20},
        strict_snapshot_reader=lambda _obs: strict,
    ) is None


def test_goal_regression_requires_prior_confirmed_goal() -> None:
    detector = _detector()
    achieved = _snapshot(
        5,
        true=frozenset({TARGET}),
        false=frozenset({SOURCE, RECOVERY, HOLDING}),
    )
    regressed = _snapshot(
        10,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    assert detector.observe(
        snapshot=achieved,
        graph_state=_state(5, "COMPLETED"),
        observation={},
        strict_snapshot_reader=lambda _obs: achieved,
    ) is None
    assert detector.observe(
        snapshot=regressed,
        graph_state=_state(10, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: regressed,
    ) is None
    request = detector.observe(
        snapshot=regressed,
        graph_state=_state(15, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: regressed,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.GOAL_REGRESSION
    assert request.signature == (f"+{TARGET.pddl()}",)


def test_frontier_stall_requires_min_step_and_unchanged_graph_window() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=20,
        stall_steps=10,
    )
    stalled = _snapshot(
        1,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    for step in (5, 19, 20, 29):
        assert detector.observe(
            snapshot=stalled,
            graph_state=_state(step, "ACTIVE"),
            observation={"step": step},
            strict_snapshot_reader=lambda _obs: stalled,
        ) is None
    request = detector.observe(
        snapshot=stalled,
        graph_state=_state(30, "ACTIVE"),
        observation={"step": 30},
        strict_snapshot_reader=lambda _obs: stalled,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL
    assert request.policy_step == 30


def test_frontier_stall_can_require_a_previously_achieved_goal() -> None:
    problem, graph = _problem_and_graph()
    detector = OnlineGraphDeviationDetector(
        replace(problem, goal=frozenset({MILESTONE, TARGET})),
        graph,
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_achieved_goal=True,
    )
    no_goal = _snapshot(
        0,
        true=frozenset({SOURCE}),
        false=frozenset({MILESTONE, TARGET, RECOVERY, HOLDING}),
    )
    one_goal = _snapshot(
        20,
        true=frozenset({SOURCE, MILESTONE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    for step in (0, 10):
        assert detector.observe(
            snapshot=no_goal,
            graph_state=_state(step, "ACTIVE"),
            observation={"step": step},
            strict_snapshot_reader=lambda _obs: no_goal,
        ) is None
    assert detector.observe(
        snapshot=one_goal,
        graph_state=_state(20, "ACTIVE"),
        observation={"step": 20},
        strict_snapshot_reader=lambda _obs: one_goal,
    ) is None
    request = detector.observe(
        snapshot=one_goal,
        graph_state=_state(30, "ACTIVE"),
        observation={"step": 30},
        strict_snapshot_reader=lambda _obs: one_goal,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_frontier_stall_ignores_nonmilestone_grounding_jitter() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
    )
    stable = _snapshot(
        0,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )
    source_jitter = _snapshot(
        5,
        true=frozenset(),
        false=frozenset({SOURCE, TARGET, RECOVERY, HOLDING}),
    )

    assert detector.observe(
        snapshot=stable,
        graph_state=_state(0, "READY"),
        observation={},
        strict_snapshot_reader=lambda _obs: stable,
    ) is None
    assert detector.observe(
        snapshot=source_jitter,
        graph_state=_state(5, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: stable,
    ) is None
    request = detector.observe(
        snapshot=stable,
        graph_state=_state(10, "READY"),
        observation={},
        strict_snapshot_reader=lambda _obs: stable,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_frontier_stall_resets_on_confirmed_holding_progress() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
    )
    source = _snapshot(
        0,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )
    holding = _snapshot(
        5,
        true=frozenset({HOLDING}),
        false=frozenset({SOURCE, TARGET, RECOVERY}),
    )

    for step, snapshot, status in (
        (0, source, "READY"),
        (5, holding, "ACTIVE"),
        (10, holding, "ACTIVE"),
    ):
        assert detector.observe(
            snapshot=snapshot,
            graph_state=_state(step, status),
            observation={},
            strict_snapshot_reader=lambda _obs: holding,
        ) is None
    request = detector.observe(
        snapshot=holding,
        graph_state=_state(15, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: holding,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_frontier_stall_can_ignore_transient_holding_after_goal_progress() -> None:
    problem, graph = _problem_and_graph()
    detector = OnlineGraphDeviationDetector(
        replace(problem, goal=frozenset({MILESTONE, TARGET})),
        graph,
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_achieved_goal=True,
        stall_ignores_holding=True,
    )
    handempty = _snapshot(
        0,
        true=frozenset({SOURCE, MILESTONE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )
    holding = _snapshot(
        10,
        true=frozenset({MILESTONE, HOLDING}),
        false=frozenset({SOURCE, TARGET, RECOVERY}),
    )

    assert detector.observe(
        snapshot=handempty,
        graph_state=_state(0, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: handempty,
    ) is None
    request = detector.observe(
        snapshot=holding,
        graph_state=_state(10, "ACTIVE"),
        observation={},
        strict_snapshot_reader=lambda _obs: holding,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_frontier_stall_can_require_handempty() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_handempty=True,
    )
    holding = _snapshot(
        0,
        true=frozenset({HOLDING}),
        false=frozenset({SOURCE, TARGET, RECOVERY}),
    )

    for step in (0, 10, 20):
        assert detector.observe(
            snapshot=holding,
            graph_state=_state(step, "ACTIVE"),
            observation={"step": step},
            strict_snapshot_reader=lambda _obs: holding,
        ) is None


def test_frontier_stall_requires_explicit_advisory_handempty() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_handempty=True,
    )
    unknown_handempty = _snapshot(
        0,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    for step in (0, 10):
        assert detector.observe(
            snapshot=unknown_handempty,
            graph_state=_state(step, "ACTIVE"),
            observation={"step": step},
            strict_snapshot_reader=lambda _obs: unknown_handempty,
        ) is None


def test_frontier_stall_strictly_confirms_handempty() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_handempty=True,
    )
    advisory = _snapshot(
        0,
        true=frozenset({SOURCE, HAND_EMPTY}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )
    strict = _snapshot(
        10,
        true=frozenset({SOURCE}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    assert detector.observe(
        snapshot=advisory,
        graph_state=_state(0, "ACTIVE"),
        observation={"step": 0},
        strict_snapshot_reader=lambda _obs: strict,
    ) is None
    assert detector.observe(
        snapshot=advisory,
        graph_state=_state(10, "ACTIVE"),
        observation={"step": 10},
        strict_snapshot_reader=lambda _obs: strict,
    ) is None


def test_frontier_stall_triggers_when_both_snapshots_verify_handempty() -> None:
    detector = _detector(
        confirmation_count=1,
        min_intervention_step=0,
        stall_steps=10,
        stall_requires_handempty=True,
    )
    handempty = _snapshot(
        0,
        true=frozenset({SOURCE, HAND_EMPTY}),
        false=frozenset({TARGET, RECOVERY, HOLDING}),
    )

    assert detector.observe(
        snapshot=handempty,
        graph_state=_state(0, "ACTIVE"),
        observation={"step": 0},
        strict_snapshot_reader=lambda _obs: handempty,
    ) is None
    request = detector.observe(
        snapshot=handempty,
        graph_state=_state(10, "ACTIVE"),
        observation={"step": 10},
        strict_snapshot_reader=lambda _obs: handempty,
    )

    assert request is not None
    assert request.kind is OnlineDeviationKind.FRONTIER_STALL


def test_unknown_or_stale_certificate_alone_never_triggers() -> None:
    detector = _detector(confirmation_count=1, stall_steps=5)
    unknown = _snapshot(
        1,
        true=frozenset({SOURCE}),
        false=frozenset({RECOVERY, HOLDING}),
    )

    for step in range(0, 30, 5):
        assert detector.observe(
            snapshot=unknown,
            graph_state=_state(step, "BLOCKED", certificate="STALE"),
            observation={},
            strict_snapshot_reader=lambda _obs: unknown,
        ) is None


def test_detector_latches_only_the_first_request() -> None:
    detector = _detector(confirmation_count=1)
    recovery = _snapshot(
        10,
        true=frozenset({RECOVERY}),
        false=frozenset({SOURCE, TARGET, HOLDING}),
    )
    first = detector.observe(
        snapshot=recovery,
        graph_state=_state(10, "ACTIVE"),
        observation={"frame": 10},
        strict_snapshot_reader=lambda _obs: recovery,
    )
    second = detector.observe(
        snapshot=recovery,
        graph_state=_state(15, "ACTIVE"),
        observation={"frame": 15},
        strict_snapshot_reader=lambda _obs: recovery,
    )

    assert first is not None
    assert second is None
    assert detector.latched_request is first
