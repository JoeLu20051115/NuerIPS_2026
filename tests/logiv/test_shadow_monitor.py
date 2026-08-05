from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pytest

from pi05_libero_repro.logiv.dag import (
    CausalGraph,
    CausalLink,
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
    TruthValue,
    fact_universe_sha256,
)
from pi05_libero_repro.logiv.recovery_records import (
    _parse_contract,
    _validate_evidence_record,
)
from pi05_libero_repro.logiv.shadow_monitor import (
    ActionEventEvidence,
    ActionEventRule,
    ActionTransitionFeatures,
    CertificateState,
    DeviationStatus,
    MonitorEvidenceContract,
    ReservedFactEventRuleId,
    ShadowCertificateReconciler,
    ShadowPlanContext,
    StableRecoveryObserver,
    VersionedActionEventTracker,
    load_monitor_evidence_contract,
)
from pi05_libero_repro.protocol import ShadowStepContext


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "study_table_black_book_init_region"
TARGET = "desk_caddy_1_back_contain_region"
ACCESS = "desk_caddy_1_access"
ABNORMAL = "study_table_recovery_surface"
BOOK_1 = "black_book_1"
BOOK_2 = "black_book_2"
AT_SOURCE = Fact("at", (BOOK_1, SOURCE))
AT_TARGET = Fact("at", (BOOK_1, TARGET))
AT_ABNORMAL = Fact("at", (BOOK_1, ABNORMAL))
AT_SOURCE_2 = Fact("at", (BOOK_2, SOURCE))
AT_TARGET_2 = Fact("at", (BOOK_2, TARGET))
AT_ABNORMAL_2 = Fact("at", (BOOK_2, ABNORMAL))
HOLDING_1 = Fact("holding", (BOOK_1,))
HOLDING_2 = Fact("holding", (BOOK_2,))
HANDEMPTY = Fact("handempty")
OPEN = Fact("open", (ACCESS,))
CLOSED = Fact("closed", (ACCESS,))
POWERED_ON = Fact("powered-on", ("lamp_1",))
UNRELATED = Fact("powered-off", ("lamp_1",))
UNIVERSE = frozenset(
    {
        AT_SOURCE,
        AT_TARGET,
        AT_ABNORMAL,
        AT_SOURCE_2,
        AT_TARGET_2,
        AT_ABNORMAL_2,
        HOLDING_1,
        HOLDING_2,
        HANDEMPTY,
        OPEN,
        CLOSED,
        POWERED_ON,
        UNRELATED,
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


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(
        domain + b"\0" + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _snapshot(
    policy_step: int,
    *,
    true: Iterable[Fact],
    false: Iterable[Fact],
    universe: frozenset[Fact] = UNIVERSE,
    version: str = "test-fact-universe-v1",
    noise: str = "",
) -> FactSnapshot:
    true_facts = frozenset(true)
    false_facts = frozenset(false)
    values = []
    for fact in sorted(universe, key=lambda item: item.pddl()):
        value = (
            TruthValue.TRUE
            if fact in true_facts
            else TruthValue.FALSE
            if fact in false_facts
            else TruthValue.UNKNOWN
        )
        values.append([fact.pddl(), value.value])
    payload = {
        "epoch_id": policy_step,
        "observation_hash": _sha256(f"observation:{policy_step}:{noise}"),
        "values": values,
        "dominance_overrides": [],
    }
    payload_json = _canonical_json(payload)
    return FactSnapshot(
        epoch_id=policy_step,
        true_facts=true_facts,
        false_facts=false_facts,
        evidence_hash=_sha256(payload_json),
        fact_universe=universe,
        fact_universe_version=version,
        fact_universe_sha256=fact_universe_sha256(version, universe),
        evidence_payload_json=payload_json,
    )


def _initial_snapshot(step: int = 0, *, noise: str = "") -> FactSnapshot:
    return _snapshot(
        step,
        true={AT_SOURCE, HANDEMPTY, OPEN},
        false={AT_TARGET, AT_ABNORMAL, HOLDING_1},
        noise=noise,
    )


def _dropped_snapshot(step: int, *, noise: str = "") -> FactSnapshot:
    return _snapshot(
        step,
        true={AT_ABNORMAL, HANDEMPTY, OPEN},
        false={AT_SOURCE, AT_TARGET, HOLDING_1},
        noise=noise,
    )


def _dropped_snapshot_2(step: int) -> FactSnapshot:
    return _snapshot(
        step,
        true={AT_ABNORMAL_2, HANDEMPTY, OPEN},
        false={AT_SOURCE_2, AT_TARGET_2, HOLDING_2},
    )


def _goal_snapshot(truth: bool, step: int) -> FactSnapshot:
    return _snapshot(
        step,
        true={AT_TARGET if truth else AT_ABNORMAL, HANDEMPTY, OPEN},
        false={AT_ABNORMAL if truth else AT_TARGET, AT_SOURCE, HOLDING_1},
    )


def _step_context(policy_step: int, *, observation_noise: float = 0.0) -> ShadowStepContext:
    return ShadowStepContext(
        observation={
            "agentview_image": np.full((2, 2, 3), observation_noise),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3)),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(1),
        },
        last_action=None if policy_step == 0 else np.zeros(7),
        policy_step=policy_step,
        base_policy_request_count=1,
        active_base_request_index=0,
        next_base_request_index=1,
        active_base_request_envelope_json=None,
        next_base_replay_envelope_json=None,
        base_action_response_size=1,
        base_action_chunk_size=1,
        pending_base_action_offset=0,
        pending_base_actions=np.empty((0, 7)),
        base_action_prefix_sha256="a" * 64,
    )


def _rule(
    *,
    rule_id: str = "place_book_1",
    object_id: str = BOOK_1,
    attempt_kind: str = "place",
    attempted_effect: str | None = None,
    source_region: str | None = SOURCE,
    destination_region: str | None = TARGET,
    due_after: int = 3,
    evidence_ttl: int = 5,
    attribution_ttl: int = 4,
) -> ActionEventRule:
    effect = attempted_effect or Fact("at", (object_id, TARGET)).pddl()
    return ActionEventRule(
        rule_id=rule_id,
        object_id=object_id,
        attempt_kind=attempt_kind,
        attempted_effect=effect,
        source_region=source_region,
        destination_region=destination_region,
        gripper_close_threshold=-0.5,
        gripper_open_threshold=0.5,
        contact_min_count=1,
        motion_correlation_min=0.5,
        region_distance_max=0.1,
        effect_due_after_policy_steps=due_after,
        evidence_ttl_policy_steps=evidence_ttl,
        manipulation_attribution_ttl_policy_steps=attribution_ttl,
    )


def _monitor_contract(
    *,
    rules: tuple[ActionEventRule, ...] | None = None,
    object_ids: tuple[str, ...] = (BOOK_1, BOOK_2),
    abnormal_support_surfaces: tuple[str, ...] = (ABNORMAL,),
    monitor_interval_steps: int = 5,
    confirmation_count: int = 3,
    max_attempt_records: int = 128,
    max_evidence_records: int = 128,
) -> MonitorEvidenceContract:
    active_rules = rules or (_rule(),)
    payload = {
        "contract_id": "test-monitor-v1",
        "task_id": 5,
        "object_ids": list(object_ids),
        "nominal_source_facts": [AT_SOURCE.pddl(), AT_SOURCE_2.pddl()],
        "abnormal_support_surfaces": list(abnormal_support_surfaces),
        "task_relevant_effects": sorted(
            {rule.attempted_effect for rule in active_rules} | {AT_TARGET.pddl()}
        ),
        "tracker_version": "test-tracker-v1",
        "action_event_rules": [asdict(rule) for rule in active_rules],
        "monitor_interval_steps": monitor_interval_steps,
        "confirmation_count": confirmation_count,
        "settling_grace_observations": 2,
        "progress_window_observations": 4,
        "progress_evidence_ttl_policy_steps": 20,
        "goal_regression_evidence_ttl_policy_steps": 20,
        "max_active_attempts_per_object": 8,
        "max_attempt_records_per_episode": max_attempt_records,
        "max_evidence_records_per_episode": max_evidence_records,
        "grounding_rule_sha256": "1" * 64,
        "event_detector_sha256": "2" * 64,
    }
    payload["contract_sha256"] = _sha256(_canonical_json(payload))
    return MonitorEvidenceContract(
        **{
            **payload,
            "object_ids": tuple(payload["object_ids"]),
            "nominal_source_facts": tuple(payload["nominal_source_facts"]),
            "abnormal_support_surfaces": tuple(payload["abnormal_support_surfaces"]),
            "task_relevant_effects": tuple(payload["task_relevant_effects"]),
            "action_event_rules": active_rules,
        }
    )


def _shadow_plan_context(*, goal: Fact = AT_TARGET) -> ShadowPlanContext:
    action = GroundAction(
        schema="place-in",
        arguments=(BOOK_1, SOURCE, TARGET, ACCESS),
        preconditions=frozenset({AT_SOURCE, HANDEMPTY, OPEN}),
        add_effects=frozenset({AT_TARGET}),
        del_effects=frozenset({AT_SOURCE}),
        repeatable=False,
    )
    problem = TaskProblem(
        name="test_problem",
        objects=(
            ObjectDecl(BOOK_1, "movable"),
            ObjectDecl(BOOK_2, "movable"),
            ObjectDecl(SOURCE, "relative-region"),
            ObjectDecl(TARGET, "container-region"),
            ObjectDecl(ABNORMAL, "surface"),
            ObjectDecl(ACCESS, "access"),
            ObjectDecl("lamp_1", "switchable"),
        ),
        initial_state=frozenset({AT_SOURCE, AT_SOURCE_2, HANDEMPTY, OPEN}),
        initial_false=frozenset({AT_TARGET, AT_TARGET_2, AT_ABNORMAL, AT_ABNORMAL_2}),
        goal=frozenset({goal}),
    )
    certificate = "c" * 64
    graph = CausalGraph(
        graph_version="graph-v1",
        graph_hash="d" * 64,
        source_epoch=0,
        certificate_hash=certificate,
        nodes=(
            GraphNode("INIT", NodeKind.INIT, 0),
            GraphNode("a0", NodeKind.ACTION, 1, action=action),
            GraphNode("GOAL", NodeKind.GOAL, 2),
        ),
        edges=(
            GraphEdge(
                "INIT",
                "a0",
                support_literals=frozenset({SignedLiteral(AT_SOURCE, True)}),
            ),
            GraphEdge(
                "a0",
                "GOAL",
                support_literals=frozenset({SignedLiteral(goal, True)}),
            ),
        ),
        causal_links=(
            CausalLink("INIT", SignedLiteral(AT_SOURCE, True), "a0"),
            CausalLink("a0", SignedLiteral(goal, True), "GOAL"),
        ),
        canonical_agenda=("a0",),
    )
    return ShadowPlanContext(
        problem=problem,
        plan=(action,),
        graph=graph,
        certificate_hash=certificate,
    )


def _evidence(
    *,
    contract: MonitorEvidenceContract | None = None,
    evidence_kind: str = "ATTEMPTED_EFFECT_TIMEOUT",
    rule_id: str = "place_book_1",
    object_id: str = BOOK_1,
    attempted_effect: str | None = None,
    source_region: str | None = SOURCE,
    destination_region: str | None = TARGET,
    start: int = 1,
    due: int = 10,
    emitted: int = 10,
    expires: int = 30,
) -> ActionEventEvidence:
    active_contract = contract or _monitor_contract()
    if evidence_kind == "GOAL_REGRESSION":
        rule_id = ReservedFactEventRuleId.GOAL_REGRESSION.value
    elif evidence_kind == "PROGRESS_TIMEOUT":
        rule_id = ReservedFactEventRuleId.PROGRESS_TIMEOUT.value
    effect = attempted_effect or Fact("at", (object_id, TARGET)).pddl()
    supporting_hash = _sha256("transition")
    attempt_id = _sha256(f"attempt:{rule_id}:{object_id}:{start}")
    if evidence_kind == "GOAL_REGRESSION":
        attempt_id = _domain_sha256(
            b"LOGIV_GOAL_REGRESSION_ATTEMPT_ID_V1",
            {
                "achieved_fact_evidence_sha256": supporting_hash,
                "achieved_policy_step": start,
                "goal_literal": effect,
            },
        )
    elif evidence_kind == "PROGRESS_TIMEOUT":
        attempt_id = _domain_sha256(
            b"LOGIV_PROGRESS_TIMEOUT_ATTEMPT_ID_V1",
            {
                "attempted_effect": effect,
                "effect_due_policy_step": due,
                "object_id": object_id,
                "rule_id": rule_id,
                "start_fact_evidence_sha256": supporting_hash,
                "start_policy_step": start,
            },
        )
    return ActionEventEvidence.create(
        evidence_kind=evidence_kind,
        rule_id=rule_id,
        object_id=object_id,
        attempted_effect=effect,
        source_region=source_region,
        destination_region=destination_region,
        attempt_id=attempt_id,
        start_policy_step=start,
        effect_due_policy_step=due,
        emitted_policy_step=emitted,
        evidence_expires_policy_step=expires,
        supporting_transition_hashes=(supporting_hash,),
        detector_sha256=active_contract.event_detector_sha256,
    )


class _EvidenceTracker:
    def __init__(
        self,
        evidence: Iterable[ActionEventEvidence] = (),
        *,
        evidence_by_step: Mapping[int, Iterable[ActionEventEvidence]] | None = None,
        overflow_count: int = 0,
        fail_steps: Iterable[int] = (),
    ) -> None:
        self.records = list(evidence)
        self.evidence_by_step = {
            step: tuple(items) for step, items in (evidence_by_step or {}).items()
        }
        self._overflow_count = overflow_count
        self.fail_steps = set(fail_steps)
        self.policy_step = 0
        self.observed_policy_steps: list[int] = []

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    def observe(self, context: ShadowStepContext) -> None:
        self.policy_step = context.policy_step
        self.observed_policy_steps.append(context.policy_step)
        if context.policy_step in self.fail_steps:
            raise RuntimeError("tracker")

    def record_fact_event(self, evidence: ActionEventEvidence) -> bool:
        if evidence.evidence_id in {item.evidence_id for item in self.records}:
            return False
        self.records.append(evidence)
        return True

    def active_for(
        self,
        *,
        object_id: str,
        attempted_effect: str,
        source_region: str | None,
        destination_region: str | None,
        policy_step: int,
    ) -> tuple[ActionEventEvidence, ...]:
        candidates = self.records + list(self.evidence_by_step.get(policy_step, ()))
        return tuple(
            item
            for item in candidates
            if item.object_id == object_id
            and item.attempted_effect == attempted_effect
            and item.source_region == source_region
            and item.destination_region == destination_region
            and item.effect_due_policy_step <= policy_step <= item.evidence_expires_policy_step
        )


def _observer(
    snapshots: Iterable[FactSnapshot],
    *,
    contract: MonitorEvidenceContract | None = None,
    tracker: _EvidenceTracker | None = None,
    plan_context: ShadowPlanContext | None = None,
    callback=None,
) -> tuple[StableRecoveryObserver, list]:
    values = iter(snapshots)
    triggers: list = []
    active_contract = contract or _monitor_contract()
    observer = StableRecoveryObserver(
        plan_context=plan_context or _shadow_plan_context(),
        monitor_contract=active_contract,
        snapshot_reader=lambda observation: next(values),
        action_event_tracker=tracker or _EvidenceTracker(),
        on_trigger=callback or triggers.append,
        interval_steps=active_contract.monitor_interval_steps,
        confirmation_count=active_contract.confirmation_count,
    )
    return observer, triggers


def _run(
    snapshots: Iterable[FactSnapshot],
    *,
    steps: Iterable[int] = range(0, 21),
    contract: MonitorEvidenceContract | None = None,
    tracker: _EvidenceTracker | None = None,
    plan_context: ShadowPlanContext | None = None,
) -> tuple[list, StableRecoveryObserver]:
    observer, triggers = _observer(
        snapshots, contract=contract, tracker=tracker, plan_context=plan_context
    )
    for step in steps:
        observer(_step_context(step))
    return triggers, observer


def test_static_unplanned_support_is_only_an_anomaly_candidate() -> None:
    observer, triggers = _observer(
        [_dropped_snapshot(5), _dropped_snapshot(10), _dropped_snapshot(15), _dropped_snapshot(20)]
    )
    for step in range(1, 21):
        observer(_step_context(step))

    assert len(triggers) == 1
    assert triggers[0].policy_step == 15
    assert triggers[0].trigger_class == "UNPLANNED_SUPPORT_STABLE"
    assert triggers[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE
    assert triggers[0].historical_failure_evidence == ()
    assert observer.metrics.snapshot_calls == 4
    assert observer.metrics.anomaly_candidates == 1
    assert observer.metrics.confirmed_deviations == 0


def test_goal_regression_is_confirmed_only_after_goal_was_true() -> None:
    triggers, _ = _run(
        [
            _goal_snapshot(True, 0),
            _goal_snapshot(False, 5),
            _goal_snapshot(False, 10),
            _goal_snapshot(False, 15),
            _goal_snapshot(False, 20),
        ]
    )
    assert [item.trigger_class for item in triggers] == [
        "COMPLETED_GOAL_REGRESSION_STABLE"
    ]
    assert triggers[0].deviation_status is DeviationStatus.CONFIRMED_DEVIATION
    assert [item.evidence_kind for item in triggers[0].historical_failure_evidence] == [
        "GOAL_REGRESSION"
    ]
    evidence = triggers[0].historical_failure_evidence[0]
    _validate_evidence_record(
        asdict(evidence),
        _parse_contract(_monitor_contract().canonical_json()),
        _monitor_contract().event_detector_sha256,
        root_policy_step=triggers[0].policy_step,
        root_object_ids=frozenset({BOOK_1, BOOK_2}),
    )


def test_initial_and_transient_intermediate_surfaces_are_not_candidates() -> None:
    initial, _ = _run([_initial_snapshot(step) for step in (0, 5, 10, 15, 20)])
    transient = _evidence(
        evidence_kind="TRANSIENT_RELEASE",
        start=0,
        due=0,
        emitted=0,
        expires=30,
    )
    settling, _ = _run(
        [_initial_snapshot(0)] + [_dropped_snapshot(step) for step in (5, 10, 15, 20)],
        tracker=_EvidenceTracker([transient]),
    )
    assert initial == []
    assert settling == []


def test_weak_timeout_does_not_confirm_but_strong_timeout_does() -> None:
    snapshots = [_initial_snapshot(0)] + [_dropped_snapshot(step) for step in (5, 10, 15, 20)]
    weak, _ = _run(snapshots, tracker=_EvidenceTracker([_evidence(evidence_kind="PROGRESS_TIMEOUT")]))
    strong, _ = _run(snapshots, tracker=_EvidenceTracker([_evidence()]))
    assert weak[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE
    assert strong[0].deviation_status is DeviationStatus.CONFIRMED_DEVIATION


def test_late_strong_evidence_upgrades_once_without_changing_event_id() -> None:
    strong = _evidence(due=20, emitted=20, expires=30)
    tracker = _EvidenceTracker(evidence_by_step={20: (strong,), 25: (strong,)})
    triggers, _ = _run(
        [_initial_snapshot(0)] + [_dropped_snapshot(step) for step in (5, 10, 15, 20, 25)],
        steps=range(0, 26),
        tracker=tracker,
    )
    assert [item.deviation_status for item in triggers] == [
        DeviationStatus.ANOMALY_CANDIDATE,
        DeviationStatus.CONFIRMED_DEVIATION,
    ]
    assert len({item.deviation_event_id for item in triggers}) == 1
    assert len({item.event_origin_parent_sha256 for item in triggers}) == 1


def test_evidence_join_rejects_cross_object_expired_and_wrong_effect() -> None:
    evidence = (
        _evidence(object_id=BOOK_2, attempted_effect=AT_TARGET_2.pddl()),
        _evidence(expires=10),
        _evidence(attempted_effect=HOLDING_1.pddl()),
    )
    triggers, _ = _run(
        [_initial_snapshot(0)] + [_dropped_snapshot(step) for step in (5, 10, 15, 20)],
        tracker=_EvidenceTracker(evidence),
    )
    assert triggers[0].deviation_status is DeviationStatus.ANOMALY_CANDIDATE


def test_event_tracker_runs_every_step_while_snapshot_reader_is_sampled() -> None:
    tracker = _EvidenceTracker()
    _, observer = _run(
        [_initial_snapshot(0)] + [_dropped_snapshot(step) for step in (5, 10, 15, 20)],
        tracker=tracker,
    )
    assert tracker.observed_policy_steps == list(range(21))
    assert observer.metrics.snapshot_calls == 5


def test_tracker_and_snapshot_errors_are_fail_open_and_reset_confirmation() -> None:
    calls = 0

    def reader(observation):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("grounding")
        return _dropped_snapshot(calls * 5)

    contract = _monitor_contract()
    triggers: list = []
    observer = StableRecoveryObserver(
        plan_context=_shadow_plan_context(),
        monitor_contract=contract,
        snapshot_reader=reader,
        action_event_tracker=_EvidenceTracker(fail_steps={6}),
        on_trigger=triggers.append,
        interval_steps=5,
        confirmation_count=3,
    )
    for step in range(1, 21):
        observer(_step_context(step))
    assert observer.metrics.snapshot_errors == 1
    assert observer.metrics.event_tracker_errors == 1
    assert [item.policy_step for item in triggers] == [20]


def test_trigger_callback_error_is_contained_and_observation_is_copied() -> None:
    def fail(trigger):
        trigger.observation["agentview_image"][0, 0, 0] = 99
        raise RuntimeError("sink")

    observer, _ = _observer(
        [_dropped_snapshot(step) for step in (5, 10, 15)], callback=fail
    )
    contexts = [_step_context(step) for step in range(1, 16)]
    for context in contexts:
        observer(context)
    assert observer.metrics.trigger_callback_errors == 1
    assert contexts[-1].observation["agentview_image"][0, 0, 0] == 0


def test_fact_universe_drift_is_a_snapshot_error() -> None:
    drifted = _snapshot(
        5,
        true={AT_ABNORMAL, HANDEMPTY, OPEN},
        false={AT_SOURCE, AT_TARGET, HOLDING_1},
        universe=UNIVERSE | {Fact("open", ("other_access",))},
        version="drifted",
    )
    triggers, observer = _run(
        [_initial_snapshot(0), drifted, _dropped_snapshot(10), _dropped_snapshot(15), _dropped_snapshot(20)]
    )
    assert [item.policy_step for item in triggers] == [20]
    assert observer.metrics.snapshot_errors == 1


def test_certificate_reconciler_ignores_noise_and_is_sticky_stale() -> None:
    reconciler = ShadowCertificateReconciler(_shadow_plan_context(), "1" * 64)
    same = reconciler.reconcile(_initial_snapshot(0), _initial_snapshot(5, noise="pixels"))
    uncovered = reconciler.reconcile(_initial_snapshot(5), _dropped_snapshot(10))
    sticky = reconciler.reconcile(_dropped_snapshot(10), _dropped_snapshot(15, noise="more"))
    assert same.certificate_state is CertificateState.CURRENT
    assert uncovered.certificate_state is CertificateState.STALE
    assert sticky.certificate_state is CertificateState.STALE
    assert same.relevant_fact_sha256 != ""


def test_certificate_reconciler_accepts_a_covered_action_transition() -> None:
    reconciler = ShadowCertificateReconciler(_shadow_plan_context(), "1" * 64)
    result = reconciler.reconcile(_initial_snapshot(0), _goal_snapshot(True, 5))
    assert result.certificate_state is CertificateState.CURRENT


def test_certificate_reconciler_covers_nominal_macro_transport_then_stales_on_regression() -> None:
    reconciler = ShadowCertificateReconciler(_shadow_plan_context(), "1" * 64)
    held = _snapshot(
        5,
        true={HOLDING_1, OPEN},
        false={AT_SOURCE, AT_TARGET, AT_ABNORMAL, HANDEMPTY},
    )
    released = _goal_snapshot(True, 10)
    regressed = _snapshot(
        15,
        true={HANDEMPTY, OPEN},
        false={AT_SOURCE, AT_TARGET, AT_ABNORMAL, HOLDING_1},
    )

    entered = reconciler.reconcile(_initial_snapshot(0), held)
    completed = reconciler.reconcile(held, released)
    lost = reconciler.reconcile(released, regressed)

    assert entered.certificate_state is CertificateState.CURRENT
    assert completed.certificate_state is CertificateState.CURRENT
    assert lost.certificate_state is CertificateState.STALE


def test_certificate_reconciler_does_not_enter_transport_on_unknown_holding() -> None:
    reconciler = ShadowCertificateReconciler(_shadow_plan_context(), "1" * 64)
    ambiguous = _snapshot(
        5,
        true={OPEN},
        false={AT_SOURCE, AT_TARGET, AT_ABNORMAL, HANDEMPTY},
    )

    result = reconciler.reconcile(_initial_snapshot(0), ambiguous)

    assert ambiguous.truth(HOLDING_1) is TruthValue.UNKNOWN
    assert result.certificate_state is CertificateState.STALE


def test_certificate_reconciler_stales_on_inflight_wrong_location_release() -> None:
    reconciler = ShadowCertificateReconciler(_shadow_plan_context(), "1" * 64)
    held = _snapshot(
        5,
        true={HOLDING_1, OPEN},
        false={AT_SOURCE, AT_TARGET, AT_ABNORMAL, HANDEMPTY},
    )

    entered = reconciler.reconcile(_initial_snapshot(0), held)
    wrong_release = reconciler.reconcile(held, _dropped_snapshot(10))

    assert entered.certificate_state is CertificateState.CURRENT
    assert wrong_release.certificate_state is CertificateState.STALE


def test_shadow_plan_context_rejects_mixed_plan_artifacts() -> None:
    context = _shadow_plan_context()
    with pytest.raises(ValueError, match="certificate"):
        replace(context, certificate_hash="f" * 64)
    with pytest.raises(ValueError, match="agenda|plan"):
        replace(context, plan=())


def test_event_origin_ignores_epoch_and_pixel_noise_but_binds_semantics() -> None:
    first, _ = _run(
        [_dropped_snapshot(step, noise="a") for step in (5, 10, 15)],
        steps=range(1, 16),
    )
    second, _ = _run(
        [_dropped_snapshot(step + 100, noise="b") for step in (5, 10, 15)],
        steps=range(1, 16),
    )
    changed_goal_context = _shadow_plan_context(goal=AT_ABNORMAL)
    third, _ = _run(
        [_dropped_snapshot(step, noise="a") for step in (5, 10, 15)],
        steps=range(1, 16),
        plan_context=changed_goal_context,
    )
    assert first[0].event_origin_parent_sha256 == second[0].event_origin_parent_sha256
    assert first[0].deviation_event_id == second[0].deviation_event_id
    assert third == [] or first[0].event_origin_parent_sha256 != third[0].event_origin_parent_sha256


class _FeatureReader:
    def __init__(self, contract: MonitorEvidenceContract, rows: Mapping[int, tuple[ActionTransitionFeatures, ...]]) -> None:
        self.monitor_contract_sha256 = contract.contract_sha256
        self.tracker_version = contract.tracker_version
        self.rule_ids = tuple(rule.rule_id for rule in contract.action_event_rules)
        self.rows = rows

    def __call__(self, context: ShadowStepContext) -> tuple[ActionTransitionFeatures, ...]:
        return self.rows.get(context.policy_step, ())


def _feature(
    contract: MonitorEvidenceContract,
    rule: ActionEventRule,
    step: int,
    *,
    gripper: float | None,
    contact: int | None = 0,
    holding: TruthValue = TruthValue.FALSE,
    source: TruthValue = TruthValue.TRUE,
    destination: TruthValue = TruthValue.FALSE,
    abnormal: TruthValue = TruthValue.FALSE,
    abnormal_region_id: str | None = None,
    destination_distance: float | None = 1.0,
    abnormal_distance: float | None = None,
    correlation: float | None = 0.0,
    rule_id: str | None = None,
) -> ActionTransitionFeatures:
    return ActionTransitionFeatures.create(
        policy_step=step,
        monitor_contract_sha256=contract.contract_sha256,
        tracker_version=contract.tracker_version,
        rule_id=rule_id or rule.rule_id,
        object_id=rule.object_id,
        source_region=rule.source_region,
        destination_region=rule.destination_region,
        gripper_qpos=gripper,
        contact_count=contact,
        holding=holding,
        source_region_truth=source,
        destination_region_truth=destination,
        abnormal_region_truth=abnormal,
        source_region_distance=0.0,
        destination_region_distance=destination_distance,
        abnormal_region_id=abnormal_region_id,
        abnormal_region_distance=abnormal_distance,
        object_eef_motion_correlation=correlation,
    )


def _observe_rows(contract: MonitorEvidenceContract, rows: Mapping[int, tuple[ActionTransitionFeatures, ...]]) -> VersionedActionEventTracker:
    tracker = VersionedActionEventTracker(contract, _FeatureReader(contract, rows))
    for step in sorted(rows):
        tracker.observe(_step_context(step))
    return tracker


def test_concrete_tracker_emits_timeout_at_due_and_cancels_on_success() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
    )
    contract = _monitor_contract(rules=(rule,))
    rows = {
        0: (_feature(contract, rule, 0, gripper=0.8),),
        1: (_feature(contract, rule, 1, gripper=-0.8, contact=1, correlation=0.9),),
        2: (_feature(contract, rule, 2, gripper=-0.8),),
        3: (_feature(contract, rule, 3, gripper=-0.8),),
        4: (_feature(contract, rule, 4, gripper=-0.8),),
    }
    tracker = _observe_rows(contract, rows)
    active = tracker.active_for(
        object_id=BOOK_1,
        attempted_effect=HOLDING_1.pddl(),
        source_region=SOURCE,
        destination_region=None,
        policy_step=4,
    )
    assert [item.evidence_kind for item in active] == ["ATTEMPTED_EFFECT_TIMEOUT"]
    assert active[0].effect_due_policy_step == 4
    assert active[0].evidence_expires_policy_step == 9
    _validate_evidence_record(
        asdict(active[0]),
        _parse_contract(contract.canonical_json()),
        contract.event_detector_sha256,
        root_policy_step=4,
        root_object_ids=frozenset({BOOK_1, BOOK_2}),
    )

    successful_rows = dict(rows)
    successful_rows[3] = (_feature(contract, rule, 3, gripper=-0.8, holding=TruthValue.TRUE),)
    successful = _observe_rows(contract, successful_rows)
    assert successful.active_for(
        object_id=BOOK_1,
        attempted_effect=HOLDING_1.pddl(),
        source_region=SOURCE,
        destination_region=None,
        policy_step=4,
    ) == ()


def test_due_time_unknown_is_inconclusive_and_cannot_be_backdated() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
        due_after=2,
    )
    contract = _monitor_contract(rules=(rule,))
    rows = {
        0: (_feature(contract, rule, 0, gripper=0.8),),
        1: (_feature(contract, rule, 1, gripper=-0.8, contact=1, correlation=1.0),),
        2: (_feature(contract, rule, 2, gripper=-0.8),),
        3: (_feature(contract, rule, 3, gripper=-0.8, holding=TruthValue.UNKNOWN),),
        4: (_feature(contract, rule, 4, gripper=-0.8, holding=TruthValue.FALSE),),
    }
    tracker = _observe_rows(contract, rows)
    assert tracker.evidence_record_count == 0


def test_successful_manipulation_retains_bounded_abnormal_transfer_attribution() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
        attribution_ttl=4,
    )
    contract = _monitor_contract(rules=(rule,))
    rows = {
        0: (_feature(contract, rule, 0, gripper=0.8),),
        1: (_feature(contract, rule, 1, gripper=-0.8, contact=1, correlation=1.0),),
        2: (_feature(contract, rule, 2, gripper=-0.8, holding=TruthValue.TRUE),),
        3: (_feature(contract, rule, 3, gripper=-0.8, abnormal=TruthValue.TRUE, abnormal_region_id=ABNORMAL, abnormal_distance=0.01),),
    }
    tracker = _observe_rows(contract, rows)
    active = tracker.active_for(
        object_id=BOOK_1,
        attempted_effect=HOLDING_1.pddl(),
        source_region=SOURCE,
        destination_region=None,
        policy_step=3,
    )
    assert [item.evidence_kind for item in active] == [
        "ABNORMAL_TRANSFER_AFTER_MANIPULATION"
    ]
    assert active[0].effect_due_policy_step == 3


def test_abnormal_transfer_rejects_cross_object_unregistered_and_expired_attribution() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
        attribution_ttl=2,
    )
    contract = _monitor_contract(rules=(rule,))
    rows = {
        0: (_feature(contract, rule, 0, gripper=0.8),),
        1: (_feature(contract, rule, 1, gripper=-0.8, contact=1, correlation=1.0),),
        2: (_feature(contract, rule, 2, gripper=-0.8),),
        3: (_feature(contract, rule, 3, gripper=-0.8),),
        4: (_feature(contract, rule, 4, gripper=-0.8, abnormal=TruthValue.TRUE, abnormal_region_id="unknown", abnormal_distance=0.01),),
    }
    tracker = _observe_rows(contract, rows)
    assert all(
        item.evidence_kind != "ABNORMAL_TRANSFER_AFTER_MANIPULATION"
        for item in tracker.active_for(
            object_id=BOOK_1,
            attempted_effect=HOLDING_1.pddl(),
            source_region=SOURCE,
            destination_region=None,
            policy_step=4,
        )
    )


def test_tracker_requires_complete_identity_matched_consecutive_features() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
    )
    contract = _monitor_contract(rules=(rule,))
    mismatched = _FeatureReader(
        contract,
        {0: (_feature(contract, rule, 0, gripper=0.8, rule_id="other"),)},
    )
    tracker = VersionedActionEventTracker(contract, mismatched)
    with pytest.raises(ValueError, match="rule|identity"):
        tracker.observe(_step_context(0))

    missing = VersionedActionEventTracker(contract, _FeatureReader(contract, {0: ()}))
    with pytest.raises(ValueError, match="missing rule"):
        missing.observe(_step_context(0))


def test_missing_feature_gap_cannot_bridge_a_gripper_transition() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
    )
    contract = _monitor_contract(rules=(rule,))
    reader = _FeatureReader(
        contract,
        {
            0: (_feature(contract, rule, 0, gripper=0.8),),
            1: (),
            2: (_feature(contract, rule, 2, gripper=-0.8, contact=1, correlation=1.0),),
            3: (_feature(contract, rule, 3, gripper=0.8),),
            4: (_feature(contract, rule, 4, gripper=-0.8, contact=1, correlation=1.0),),
        },
    )
    tracker = VersionedActionEventTracker(contract, reader)
    tracker.observe(_step_context(0))
    with pytest.raises(ValueError, match="missing rule"):
        tracker.observe(_step_context(1))
    tracker.observe(_step_context(2))
    assert tracker.attempt_record_count == 0
    tracker.observe(_step_context(3))
    tracker.observe(_step_context(4))
    assert tracker.attempt_record_count == 1


def test_attempt_and_evidence_capacity_are_cumulative_and_diagnostic() -> None:
    rule = _rule(
        rule_id="grasp_book_1",
        attempt_kind="grasp",
        attempted_effect=HOLDING_1.pddl(),
        destination_region=None,
        due_after=1,
        evidence_ttl=1,
    )
    contract = _monitor_contract(
        rules=(rule,), max_attempt_records=2, max_evidence_records=1
    )
    rows = {
        0: (_feature(contract, rule, 0, gripper=0.8),),
        1: (_feature(contract, rule, 1, gripper=-0.8, contact=1, correlation=1.0),),
        2: (_feature(contract, rule, 2, gripper=-0.8),),
        3: (_feature(contract, rule, 3, gripper=0.8),),
        4: (_feature(contract, rule, 4, gripper=-0.8, contact=1, correlation=1.0),),
        5: (_feature(contract, rule, 5, gripper=-0.8),),
    }
    tracker = _observe_rows(contract, rows)
    assert tracker.attempt_record_count == 2
    assert tracker.evidence_record_count == 1
    assert tracker.overflow_count == 1
    assert tracker.active_for(
        object_id=BOOK_1,
        attempted_effect=HOLDING_1.pddl(),
        source_region=SOURCE,
        destination_region=None,
        policy_step=5,
    ) == ()


def test_fact_event_validation_shares_evidence_capacity_and_deduplicates() -> None:
    rule = _rule()
    contract = _monitor_contract(rules=(rule,), max_evidence_records=1)
    tracker = VersionedActionEventTracker(contract, _FeatureReader(contract, {}))
    evidence = _evidence(
        contract=contract,
        evidence_kind="GOAL_REGRESSION",
        due=5,
        emitted=5,
        expires=25,
    )
    assert tracker.record_fact_event(evidence) is True
    assert tracker.record_fact_event(evidence) is False
    other = _evidence(
        contract=contract,
        evidence_kind="PROGRESS_TIMEOUT",
        start=1,
        due=21,
        emitted=21,
        expires=41,
    )
    assert tracker.record_fact_event(other) is False
    assert tracker.evidence_record_count == 1
    assert tracker.overflow_count == 1


def test_monitor_contract_rejects_hash_drift_bad_windows_and_reserved_rule_ids() -> None:
    contract = _monitor_contract()
    with pytest.raises(ValueError, match="hash"):
        replace(contract, contract_sha256="0" * 64)
    with pytest.raises(ValueError, match="positive"):
        replace(contract, monitor_interval_steps=0)
    with pytest.raises(ValueError, match="reserved|rule"):
        replace(contract.action_event_rules[0], rule_id="__reserved__")
    with pytest.raises(ValueError, match="interval"):
        StableRecoveryObserver(
            plan_context=_shadow_plan_context(),
            monitor_contract=contract,
            snapshot_reader=lambda observation: _initial_snapshot(),
            action_event_tracker=_EvidenceTracker(),
            on_trigger=lambda trigger: None,
            interval_steps=4,
            confirmation_count=3,
        )


def test_versioned_config_loads_task_5_and_8_and_rejects_unknown_task() -> None:
    path = ROOT / "configs/logiv/r2m-monitor-evidence-v1.json"
    task5 = load_monitor_evidence_contract(path, task_id=5)
    task8 = load_monitor_evidence_contract(path, task_id=8)
    assert task5.task_id == 5
    assert task8.task_id == 8
    assert task5.contract_sha256 != task8.contract_sha256
    with pytest.raises(ValueError, match="task"):
        load_monitor_evidence_contract(path, task_id=1)
