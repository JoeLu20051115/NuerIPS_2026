from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.dag import CausalGraph
from pi05_libero_repro.logiv.domain import FixedDomain
from pi05_libero_repro.logiv.model import (
    ContextEnvelope,
    ContextPhase,
    Fact,
    FactSnapshot,
    GoalMode,
    TaskProblem,
    TruthValue,
    fact_universe_sha256,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.task5_terminal_recovery import (
    assess_task5_terminal,
    certify_task5_recovery,
    derive_recovery_policy_seed,
    load_task5_recovery_capability,
    verify_task5_recovery_commit,
)
from pi05_libero_repro.logiv.val import PlanCertificate, ValWrapper


ROOT = Path(__file__).parents[2]
CONTRACT = ROOT / "configs/logiv/task5-terminal-pi-recover-v1.json"
REAL_VAL = Path("/home/xingrui/.local/bin/Validate")
TARGET = Fact("at", ("black_book_1", "desk_caddy_1_back_contain_region"))
HOLDING = Fact("holding", ("black_book_1",))


def _audited_snapshot(
    *,
    epoch_id: int,
    universe: frozenset[Fact],
    true_facts: frozenset[Fact],
    false_facts: frozenset[Fact],
) -> FactSnapshot:
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
    payload_json = json.dumps(
        {
            "epoch_id": epoch_id,
            "observation_hash": "e" * 64,
            "values": values,
            "dominance_overrides": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    version = "task5-terminal-test-v1"
    return FactSnapshot(
        epoch_id=epoch_id,
        true_facts=true_facts,
        false_facts=false_facts,
        evidence_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        fact_universe=universe,
        fact_universe_version=version,
        fact_universe_sha256=fact_universe_sha256(version, universe),
        evidence_payload_json=payload_json,
    )


def _snapshot_with(
    snapshot: FactSnapshot,
    *,
    true_facts: frozenset[Fact] | None = None,
    false_facts: frozenset[Fact] | None = None,
) -> FactSnapshot:
    assert snapshot.fact_universe is not None
    return _audited_snapshot(
        epoch_id=snapshot.epoch_id,
        universe=snapshot.fact_universe,
        true_facts=snapshot.true_facts if true_facts is None else true_facts,
        false_facts=snapshot.false_facts if false_facts is None else false_facts,
    )


@pytest.fixture(scope="module")
def capability():
    return load_task5_recovery_capability(CONTRACT)


@pytest.fixture(scope="module")
def task5_state(capability):
    package = ScriptedProposalProvider().propose(task_id=5, epoch_id=7)
    problem = package.problem
    universe = frozenset(problem.initial_state | problem.initial_false)
    true_facts = frozenset(
        fact
        for fact in universe
        if fact not in {Fact("handempty"), HOLDING}
        and not (fact.predicate == "at" and fact.arguments[:1] == ("black_book_1",))
        and fact in problem.initial_state
    ) | {HOLDING}
    snapshot = _audited_snapshot(
        epoch_id=7,
        universe=universe,
        true_facts=frozenset(true_facts),
        false_facts=frozenset(universe - true_facts),
    )
    parent_context = ContextEnvelope(
        phase=ContextPhase.PREINSTALL_VAL,
        goal_mode=GoalMode.METADATA_ASSISTED,
        request_id="task5-parent",
        request_generation=0,
        episode_id="episode-5-36",
        goal_id="libero10-task-5-official-goal-v1",
        goal_epoch=0,
        epoch_id=7,
        graph_version=None,
        occurrence_id=None,
        attempt_id=None,
        certificate_hash=None,
        safety_epoch=None,
    )
    certificate = PlanCertificate(
        certificate_hash="b" * 64,
        component_hashes=(),
        context=parent_context,
        val_binary=str(REAL_VAL),
        val_binary_sha256="d" * 64,
        val_version="test-parent",
        wrapper_version="logiv-val-wrapper-v1",
        timeout_seconds=5.0,
        forbidden_retry_keys=(),
        retry_ledger_version=0,
    )
    graph = CausalGraph(
        graph_version="task5-graph-v1",
        graph_hash="a" * 64,
        source_epoch=7,
        certificate_hash=certificate.certificate_hash,
        nodes=(),
        edges=(),
        causal_links=(),
        canonical_agenda=(),
    )
    action = FixedDomain().ground(
        problem,
        "place-held-in",
        (
            "black_book_1",
            "desk_caddy_1_back_contain_region",
            "desk_caddy_1_access",
        ),
    )
    return {
        "capability": capability,
        "problem": problem,
        "snapshot": snapshot,
        "certificate": certificate,
        "graph": graph,
        "action": action,
    }


@pytest.fixture(scope="module")
def eligible_inputs(task5_state):
    return {
        "capability": task5_state["capability"],
        "task_id": 5,
        "episode_id": "episode-5-36",
        "base_success": False,
        "native_terminal_status": "EPISODE_FAIL",
        "base_policy_steps": 340,
        "snapshot": task5_state["snapshot"],
        "problem": task5_state["problem"],
        "graph": task5_state["graph"],
        "certificate": task5_state["certificate"],
        "monitor_contract_sha256": "c" * 64,
        "certificate_state": "CURRENT",
        "place_node_action": task5_state["action"],
        "place_node_status": "ACTIVE",
    }


@pytest.fixture(scope="module")
def eligible_assessment(eligible_inputs):
    result = assess_task5_terminal(**eligible_inputs)
    assert result.eligible
    return result


@pytest.fixture(scope="module")
def permit_inputs(task5_state, eligible_assessment):
    return {
        "capability": task5_state["capability"],
        "assessment": eligible_assessment,
        "snapshot": task5_state["snapshot"],
        "problem": task5_state["problem"],
        "graph": task5_state["graph"],
        "certificate": task5_state["certificate"],
        "candidate_plan": (task5_state["action"],),
        "val_wrapper": ValWrapper(REAL_VAL, timeout_seconds=5.0),
        "recovery_checkpoint_sha256": "f" * 64,
    }


@pytest.fixture(scope="module")
def granted_permit(permit_inputs):
    result = certify_task5_recovery(**permit_inputs)
    assert result.granted, result.reason
    return result


@pytest.fixture(scope="module")
def post_snapshot(task5_state):
    before = task5_state["snapshot"]
    true_facts = frozenset((before.true_facts - {HOLDING}) | {TARGET, Fact("handempty")})
    return _snapshot_with(
        before,
        true_facts=true_facts,
        false_facts=frozenset(before.fact_universe - true_facts),
    )


@pytest.fixture(scope="module")
def commit_inputs(task5_state, granted_permit, post_snapshot):
    handoff_goal = Fact(
        "accessible",
        ("desk_caddy_1_front_contain_region", "desk_caddy_1_access"),
    )
    return {
        "capability": task5_state["capability"],
        "permit": granted_permit,
        "post_snapshot": post_snapshot,
        "handoff_goal_facts": frozenset({handoff_goal}),
        "event_id": granted_permit.event_id,
        "permit_sha256": granted_permit.permit_sha256,
        "plan_sha256": granted_permit.plan_sha256,
        "recovery_checkpoint_sha256": granted_permit.recovery_checkpoint_sha256,
        "capability_sha256": granted_permit.capability_sha256,
        "base_actions": 340,
        "recovery_actions": 100,
        "native_evaluator_success": True,
    }


def test_contract_is_self_hashed_and_locked():
    capability = load_task5_recovery_capability(CONTRACT)
    assert capability.task_id == 5
    assert capability.prompt == "place the held book in the back compartment of the caddy"
    assert capability.max_recovery_actions == 180
    assert capability.max_combined_actions == 520
    assert capability.capability_sha256 == capability.recompute_sha256()


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_contract_rejects_missing_and_unknown_keys(tmp_path, mutation):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload.pop("prompt")
    else:
        payload["extra"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields mismatch"):
        load_task5_recovery_capability(path)


def test_contract_rejects_changed_content_without_new_hash(tmp_path):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["settling_steps"] = 11
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_task5_recovery_capability(path)


def test_recovery_seed_uses_independent_domain():
    first = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-1")
    second = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-2")
    assert 0 <= first < 2**32
    assert first != second
    payload = "LOGIV-recovery-policy-seed-v1:54804909:5:36:terminal-event-1"
    assert first == int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        ({"base_success": True}, "BASE_SUCCEEDED"),
        ({"task_id": 4}, "UNSUPPORTED_TASK"),
        ({"native_terminal_status": "EPISODE_SUCCESS"}, "NATIVE_TERMINAL_NOT_FAILED"),
        ({"problem": None}, "MISSING_GRAPH_OR_CERTIFICATE"),
        ({"graph": None}, "MISSING_GRAPH_OR_CERTIFICATE"),
        ({"certificate": None}, "MISSING_GRAPH_OR_CERTIFICATE"),
        ({"certificate_state": "STALE"}, "CERTIFICATE_NOT_CURRENT"),
        ({"monitor_contract_sha256": None}, "MONITOR_CONTRACT_HASH_INVALID"),
        ({"snapshot": None}, "STRICT_AUDITED_SNAPSHOT_REQUIRED"),
        ({"place_node_action": None}, "PLACE_NODE_ACTION_MISMATCH"),
        ({"place_node_status": "READY"}, "PLACE_NODE_NOT_ACTIVE"),
        ({"base_policy_steps": 520}, "ACTION_BUDGET_EXHAUSTED"),
    ],
)
def test_terminal_assessment_denies_without_side_effects(eligible_inputs, args, reason):
    result = assess_task5_terminal(**(eligible_inputs | args))
    assert result.eligible is False
    assert result.reason == reason
    assert result.option_action_cap == 0


def test_terminal_assessment_rejects_changed_graph_certificate_hash(eligible_inputs):
    graph = replace(eligible_inputs["graph"], certificate_hash="0" * 64)
    result = assess_task5_terminal(**(eligible_inputs | {"graph": graph}))
    assert result.reason == "GRAPH_CERTIFICATE_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("location_unknown", "EXACTLY_ONE_LOCATION_NOT_PROVEN"),
        ("holding_false", "HOLDING_NOT_EXPLICITLY_TRUE"),
        ("target_unknown", "TARGET_NOT_EXPLICITLY_FALSE"),
        ("invariant_false", "PROTECTED_INVARIANT_NOT_TRUE"),
    ],
)
def test_terminal_assessment_requires_explicit_signed_facts(
    eligible_inputs, capability, mutation, reason
):
    snapshot = eligible_inputs["snapshot"]
    true_facts = snapshot.true_facts
    false_facts = snapshot.false_facts
    if mutation == "location_unknown":
        false_facts = false_facts - {TARGET}
    elif mutation == "holding_false":
        wrong_location = Fact(
            "at", ("black_book_1", "desk_caddy_1_front_contain_region")
        )
        true_facts = true_facts - {HOLDING}
        true_facts = true_facts | {wrong_location}
        false_facts = (false_facts | {HOLDING}) - {wrong_location}
    elif mutation == "target_unknown":
        assert snapshot.fact_universe is not None
        universe = snapshot.fact_universe - {TARGET}
        changed = _audited_snapshot(
            epoch_id=snapshot.epoch_id,
            universe=universe,
            true_facts=frozenset(true_facts),
            false_facts=frozenset(false_facts - {TARGET}),
        )
    else:
        invariant = next(iter(capability.protected_invariants))
        true_facts = true_facts - {invariant}
        false_facts = false_facts | {invariant}
    if mutation != "target_unknown":
        changed = _snapshot_with(
            snapshot,
            true_facts=frozenset(true_facts),
            false_facts=frozenset(false_facts),
        )
    result = assess_task5_terminal(**(eligible_inputs | {"snapshot": changed}))
    assert result.reason == reason


def test_terminal_assessment_grants_only_remaining_budget(eligible_inputs):
    result = assess_task5_terminal(**(eligible_inputs | {"base_policy_steps": 400}))
    assert result.eligible is True
    assert result.reason == "ELIGIBLE"
    assert result.option_action_cap == 120
    assert result.event_id is not None
    assert result.protected_true_facts


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("zero", "PLAN_ACTION_COUNT_NOT_ONE"),
        ("multiple", "PLAN_ACTION_COUNT_NOT_ONE"),
        ("wrong_action", "PLAN_ACTION_MISMATCH"),
        ("signed_state", "SIGNED_STATE_INVALID"),
        ("val_error", "VAL_VALIDATION_ERROR"),
        ("graph_hash", "GRAPH_HASH_CHANGED"),
        ("capability_hash", "CAPABILITY_HASH_MISMATCH"),
        ("checkpoint_hash", "RECOVERY_CHECKPOINT_HASH_INVALID"),
    ],
)
def test_recovery_certification_fails_closed(permit_inputs, mutation, reason):
    changed = dict(permit_inputs)
    if mutation == "zero":
        changed["candidate_plan"] = ()
    elif mutation == "multiple":
        changed["candidate_plan"] *= 2
    elif mutation == "wrong_action":
        changed["candidate_plan"] = (
            replace(changed["candidate_plan"][0], arguments=("black_book_1",)),
        )
    elif mutation == "signed_state":
        changed["problem"] = replace(
            changed["problem"],
            goal=changed["problem"].goal | {Fact("closed", ("desk_caddy_1_access",))},
        )
    elif mutation == "val_error":
        changed["val_wrapper"] = ValWrapper("/definitely/missing/Validate", timeout_seconds=1)
    elif mutation == "graph_hash":
        changed["graph"] = replace(changed["graph"], graph_hash="0" * 64)
    elif mutation == "capability_hash":
        changed["capability"] = replace(changed["capability"], capability_sha256="0" * 64)
    else:
        changed["recovery_checkpoint_sha256"] = "not-a-hash"
    result = certify_task5_recovery(**changed)
    assert result.granted is False
    assert result.reason == reason
    assert len(result.permit_sha256) == 64


def test_recovery_certification_binds_exactly_one_val_certificate(granted_permit):
    assert granted_permit.granted is True
    assert granted_permit.reason == "GRANTED"
    assert len(granted_permit.plan) == 1
    assert granted_permit.plan[0].pddl() == (
        "(place-held-in black_book_1 "
        "desk_caddy_1_back_contain_region desk_caddy_1_access)"
    )
    assert granted_permit.plan_sha256 is not None
    assert granted_permit.certificate is not None
    assert granted_permit.certificate.context.phase is ContextPhase.RECOVERY_VAL


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("permit", "PERMIT_NOT_GRANTED"),
        ("event", "EVENT_ID_CHANGED"),
        ("permit_hash", "PERMIT_HASH_CHANGED"),
        ("plan_hash", "PLAN_HASH_CHANGED"),
        ("checkpoint_hash", "RECOVERY_CHECKPOINT_HASH_CHANGED"),
        ("capability_hash", "CAPABILITY_HASH_CHANGED"),
        ("snapshot", "STRICT_AUDITED_POST_SNAPSHOT_REQUIRED"),
        ("unknown", "REQUIRED_LITERAL_UNKNOWN"),
        ("target", "TARGET_NOT_EXPLICITLY_TRUE"),
        ("handoff", "HANDOFF_GOAL_NOT_TRUE"),
        ("invariant", "PROTECTED_INVARIANT_NOT_TRUE"),
        ("negative_count", "ACTION_COUNT_INVALID"),
        ("recovery_cap", "RECOVERY_ACTION_CAP_EXCEEDED"),
        ("combined_cap", "COMBINED_ACTION_BUDGET_EXCEEDED"),
        ("native", "NATIVE_EVALUATOR_FAILED"),
    ],
)
def test_recovery_commit_fails_closed(commit_inputs, mutation, reason):
    changed = dict(commit_inputs)
    snapshot = changed["post_snapshot"]
    if mutation == "permit":
        changed["permit"] = replace(changed["permit"], granted=False)
    elif mutation == "event":
        changed["event_id"] = "changed-event"
    elif mutation == "permit_hash":
        changed["permit_sha256"] = "0" * 64
    elif mutation == "plan_hash":
        changed["plan_sha256"] = "0" * 64
    elif mutation == "checkpoint_hash":
        changed["recovery_checkpoint_sha256"] = "0" * 64
    elif mutation == "capability_hash":
        changed["capability_sha256"] = "0" * 64
    elif mutation == "snapshot":
        changed["post_snapshot"] = FactSnapshot(
            epoch_id=8,
            true_facts=snapshot.true_facts,
            false_facts=snapshot.false_facts,
            evidence_hash="legacy",
        )
    elif mutation in {"unknown", "target", "handoff", "invariant"}:
        if mutation == "unknown":
            fact = next(iter(changed["handoff_goal_facts"]))
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts - {fact}
        elif mutation == "target":
            fact = TARGET
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts | {fact}
        elif mutation == "handoff":
            fact = next(iter(changed["handoff_goal_facts"]))
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts | {fact}
        else:
            fact = next(iter(changed["capability"].protected_invariants))
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts | {fact}
        changed["post_snapshot"] = _snapshot_with(
            snapshot,
            true_facts=frozenset(true_facts),
            false_facts=frozenset(false_facts),
        )
    elif mutation == "negative_count":
        changed["recovery_actions"] = -1
    elif mutation == "recovery_cap":
        changed["base_actions"] = 0
        changed["recovery_actions"] = changed["permit"].action_cap + 1
    elif mutation == "combined_cap":
        changed["base_actions"] = 500
        changed["recovery_actions"] = 21
    else:
        changed["native_evaluator_success"] = False
    result = verify_task5_recovery_commit(**changed)
    assert result.committed is False
    assert result.reason == reason
    assert len(result.commit_sha256) == 64


def test_recovery_commit_requires_joint_symbolic_and_native_success(commit_inputs):
    result = verify_task5_recovery_commit(**commit_inputs)
    assert result.committed is True
    assert result.reason == "COMMITTED"
    assert result.base_actions == 340
    assert result.recovery_actions == 100
    assert result.combined_actions == 440
