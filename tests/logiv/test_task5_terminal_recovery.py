from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pi05_libero_repro.logiv import task5_terminal_recovery as recovery_module
from pi05_libero_repro.logiv.dag import CausalGraph, GraphNode, NodeKind
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
TARGET = Fact("at", ("black_book_1", "desk_caddy_1_back_contain_region"))
HOLDING = Fact("holding", ("black_book_1",))
HANDOFF_GOAL = Fact(
    "accessible",
    ("desk_caddy_1_front_contain_region", "desk_caddy_1_access"),
)


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
def val_binary(tmp_path_factory):
    path = tmp_path_factory.mktemp("task5-val") / "Validate"
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-h" ]; then\n'
        '  echo "Version task5-portable-test"\n'
        "  exit 0\n"
        "fi\n"
        'echo "Plan valid"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.fixture(scope="module")
def task5_state(capability, val_binary):
    package = ScriptedProposalProvider().propose(task_id=5, epoch_id=7)
    problem = replace(package.problem, goal=package.problem.goal | {HANDOFF_GOAL})
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
        val_binary=str(val_binary),
        val_binary_sha256="d" * 64,
        val_version="test-parent",
        wrapper_version="logiv-val-wrapper-v1",
        timeout_seconds=5.0,
        forbidden_retry_keys=(),
        retry_ledger_version=0,
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
    place_node = GraphNode(
        node_id="task5-place-held-in",
        kind=NodeKind.ACTION,
        canonical_rank=0,
        action=action,
    )
    graph = CausalGraph(
        graph_version="task5-graph-v1",
        graph_hash="a" * 64,
        source_epoch=7,
        certificate_hash=certificate.certificate_hash,
        nodes=(place_node,),
        edges=(),
        causal_links=(),
        canonical_agenda=(place_node.node_id,),
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
def permit_inputs(task5_state, eligible_assessment, val_binary):
    return {
        "capability": task5_state["capability"],
        "assessment": eligible_assessment,
        "snapshot": task5_state["snapshot"],
        "problem": task5_state["problem"],
        "graph": task5_state["graph"],
        "certificate": task5_state["certificate"],
        "candidate_plan": (task5_state["action"],),
        "val_wrapper": ValWrapper(val_binary, timeout_seconds=5.0),
        "recovery_checkpoint_sha256": "f" * 64,
        "expected_assessment_sha256": eligible_assessment.assessment_sha256,
        "expected_base_policy_steps": eligible_assessment.base_policy_steps,
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
    return {
        "capability": task5_state["capability"],
        "permit": granted_permit,
        "post_snapshot": post_snapshot,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("capability_id", "changed"),
        ("task_id", 4),
        ("event_type", "CHANGED"),
        ("prompt_version", "changed"),
        ("prompt", "changed"),
        ("holding_fact", "(holding changed_book)"),
        ("target_fact", "(at black_book_1 changed_region)"),
        ("action", "(changed-action)"),
        ("active_node_statuses", ["READY"]),
        ("protected_invariants", ["(open desk_caddy_1_access)"]),
        ("effect_confirmation_observations", 4),
        ("settling_steps", 11),
        ("max_recovery_actions", 181),
        ("max_combined_actions", 521),
        ("capability_required_successes", 9),
        ("capability_total", 11),
    ],
)
def test_contract_rejects_changed_semantics_even_when_rehashed(
    tmp_path, field, value
):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload[field] = value
    body = dict(payload)
    body.pop("capability_sha256")
    raw = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["capability_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen semantic values mismatch"):
        load_task5_recovery_capability(path)


def test_recovery_seed_uses_independent_domain():
    first = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-1")
    second = derive_recovery_policy_seed(54804909, 5, 36, "terminal-event-2")
    assert 0 <= first < 2**32
    assert first != second
    payload = "LOGIV-recovery-policy-seed-v1:54804909:5:36:terminal-event-1"
    assert first == int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", 4),
        ("event_type", "CHANGED"),
        ("holding_fact", Fact("holding", ("changed_book",))),
        ("target_fact", Fact("at", ("black_book_1", "changed_region"))),
        ("action", "(changed-action)"),
        ("active_node_statuses", frozenset({"READY"})),
        ("protected_invariants", frozenset({Fact("open", ("desk_caddy_1_access",))})),
        ("effect_confirmation_observations", 4),
        ("settling_steps", 11),
        ("max_recovery_actions", 181),
        ("max_combined_actions", 521),
        ("capability_required_successes", 9),
        ("capability_total", 11),
    ],
)
@pytest.mark.parametrize("boundary", ["assess", "certify", "commit"])
def test_public_boundaries_reject_rehashed_nonfrozen_capability(
    eligible_inputs, permit_inputs, commit_inputs, field, value, boundary
):
    changed = replace(eligible_inputs["capability"], **{field: value})
    changed = replace(changed, capability_sha256=changed.recompute_sha256())
    if boundary == "assess":
        result = assess_task5_terminal(**(eligible_inputs | {"capability": changed}))
    elif boundary == "certify":
        result = certify_task5_recovery(**(permit_inputs | {"capability": changed}))
    else:
        result = verify_task5_recovery_commit(
            **(
                commit_inputs
                | {
                    "capability": changed,
                    "capability_sha256": changed.capability_sha256,
                }
            )
        )
    assert result.reason == "CAPABILITY_NOT_FROZEN"


@pytest.mark.parametrize("boundary", ["assess", "certify", "commit"])
def test_public_boundaries_reject_unhashable_capability(
    eligible_inputs, permit_inputs, commit_inputs, boundary
):
    changed = replace(
        eligible_inputs["capability"],
        protected_invariants=frozenset({"not-a-fact"}),
    )
    if boundary == "assess":
        result = assess_task5_terminal(**(eligible_inputs | {"capability": changed}))
    elif boundary == "certify":
        result = certify_task5_recovery(**(permit_inputs | {"capability": changed}))
    else:
        result = verify_task5_recovery_commit(
            **(commit_inputs | {"capability": changed})
        )
    assert result.reason == "CAPABILITY_NOT_FROZEN"


def test_assessment_denial_sanitizes_nonjson_capability_digest(eligible_inputs):
    capability = replace(
        eligible_inputs["capability"], capability_sha256=object()
    )
    first = assess_task5_terminal(
        **(eligible_inputs | {"capability": capability})
    )
    second = assess_task5_terminal(
        **(eligible_inputs | {"capability": capability})
    )
    assert first.eligible is False
    assert first.reason == "CAPABILITY_NOT_FROZEN"
    assert first.capability_sha256 == ""
    assert first.assessment_sha256 == second.assessment_sha256
    assert len(first.assessment_sha256) == 64


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
        ({"base_policy_steps": 0}, "BASE_POLICY_STEPS_INVALID"),
        ({"base_policy_steps": -1}, "BASE_POLICY_STEPS_INVALID"),
        ({"base_policy_steps": True}, "BASE_POLICY_STEPS_INVALID"),
        ({"base_policy_steps": 1.5}, "BASE_POLICY_STEPS_INVALID"),
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


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mismatched"])
def test_terminal_assessment_requires_exactly_one_matching_place_node(
    eligible_inputs, mutation
):
    graph = eligible_inputs["graph"]
    node = graph.nodes[0]
    if mutation == "missing":
        nodes = ()
    elif mutation == "duplicate":
        nodes = (node, replace(node, node_id="duplicate", canonical_rank=1))
    else:
        nodes = (replace(node, action=replace(node.action, arguments=("black_book_1",))),)
    result = assess_task5_terminal(
        **(eligible_inputs | {"graph": replace(graph, nodes=nodes)})
    )
    assert result.reason == "PLACE_NODE_GRAPH_MEMBERSHIP_INVALID"


@pytest.mark.parametrize("mutation", ["missing_field", "tampered_payload"])
def test_terminal_assessment_revalidates_snapshot_audit(eligible_inputs, mutation):
    snapshot = replace(eligible_inputs["snapshot"])
    if mutation == "missing_field":
        object.__setattr__(snapshot, "fact_universe_version", None)
    else:
        object.__setattr__(snapshot, "evidence_payload_json", "{}")
    result = assess_task5_terminal(**(eligible_inputs | {"snapshot": snapshot}))
    assert result.reason == "STRICT_AUDITED_SNAPSHOT_REQUIRED"


def test_terminal_assessment_rejects_snapshot_older_than_graph(eligible_inputs):
    snapshot = eligible_inputs["snapshot"]
    stale = _audited_snapshot(
        epoch_id=eligible_inputs["graph"].source_epoch - 1,
        universe=snapshot.fact_universe,
        true_facts=snapshot.true_facts,
        false_facts=snapshot.false_facts,
    )
    result = assess_task5_terminal(**(eligible_inputs | {"snapshot": stale}))
    assert result.reason == "SNAPSHOT_EPOCH_STALE"


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


def test_terminal_assessment_authenticates_all_permit_inputs(
    eligible_inputs, capability
):
    result = assess_task5_terminal(**eligible_inputs)
    assert hasattr(result, "assessment_sha256")
    assert len(result.assessment_sha256) == 64
    assert result.capability_sha256 == capability.capability_sha256
    assert result.base_policy_steps == eligible_inputs["base_policy_steps"]
    assert result.place_node_id == eligible_inputs["graph"].nodes[0].node_id
    assert result.assessment_sha256 == recovery_module._assessment_sha256(result)


def _resign_assessment(assessment, **changes):
    changed = replace(assessment, **changes, assessment_sha256="")
    return replace(
        changed,
        assessment_sha256=recovery_module._assessment_sha256(changed),
    )


def test_recovery_certification_requires_expected_terminal_provenance(permit_inputs):
    without_expected = {
        key: value
        for key, value in permit_inputs.items()
        if key not in {"expected_assessment_sha256", "expected_base_policy_steps"}
    }
    result = certify_task5_recovery(**without_expected)
    assert result.granted is False
    assert result.reason == "EXPECTED_ASSESSMENT_SHA256_INVALID"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("expected_assessment_sha256", "", "EXPECTED_ASSESSMENT_SHA256_INVALID"),
        ("expected_assessment_sha256", object(), "EXPECTED_ASSESSMENT_SHA256_INVALID"),
        ("expected_base_policy_steps", None, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
        ("expected_base_policy_steps", 0, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
        ("expected_base_policy_steps", -1, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
        ("expected_base_policy_steps", 520, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
        ("expected_base_policy_steps", True, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
        ("expected_base_policy_steps", 1.5, "EXPECTED_BASE_POLICY_STEPS_INVALID"),
    ],
)
def test_recovery_certification_rejects_invalid_expected_provenance(
    permit_inputs, field, value, reason
):
    result = certify_task5_recovery(**(permit_inputs | {field: value}))
    assert result.granted is False
    assert result.reason == reason


@pytest.mark.parametrize("mutation", ["base_and_cap", "event", "monitor"])
def test_recovery_certification_rejects_coordinated_resigning(
    permit_inputs, mutation
):
    assessment = permit_inputs["assessment"]
    if mutation == "base_and_cap":
        forged = _resign_assessment(
            assessment,
            base_policy_steps=1,
            option_action_cap=180,
        )
    elif mutation == "event":
        forged = _resign_assessment(assessment, event_id="1" * 64)
    else:
        forged = _resign_assessment(
            assessment, monitor_contract_sha256="2" * 64
        )
    result = certify_task5_recovery(
        **(permit_inputs | {"assessment": forged})
    )
    assert result.granted is False
    assert result.reason == "ASSESSMENT_PROVENANCE_MISMATCH"


def test_recovery_certification_requires_both_external_anchors(permit_inputs):
    assessment = permit_inputs["assessment"]
    forged = _resign_assessment(
        assessment,
        base_policy_steps=1,
        option_action_cap=180,
    )
    result = certify_task5_recovery(
        **(
            permit_inputs
            | {
                "assessment": forged,
                "expected_assessment_sha256": forged.assessment_sha256,
            }
        )
    )
    assert result.granted is False
    assert result.reason == "BASE_POLICY_STEPS_PROVENANCE_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("cap", "ASSESSMENT_ACTION_CAP_MISMATCH"),
        ("base_steps", "ASSESSMENT_ACTION_CAP_MISMATCH"),
        ("protected_empty", "ASSESSMENT_PROTECTED_FACTS_MISMATCH"),
        ("protected_reduced", "ASSESSMENT_PROTECTED_FACTS_MISMATCH"),
        ("protected_substituted", "ASSESSMENT_PROTECTED_FACTS_MISMATCH"),
        ("capability", "ASSESSMENT_CAPABILITY_MISMATCH"),
        ("place_node", "ASSESSMENT_PLACE_NODE_MISMATCH"),
        ("reason", "ASSESSMENT_STATE_INVALID"),
        ("event_type", "ASSESSMENT_EVENT_TYPE_MISMATCH"),
        ("event_id", "ASSESSMENT_EVENT_ID_INVALID"),
        ("monitor", "ASSESSMENT_MONITOR_HASH_INVALID"),
        ("digest", "ASSESSMENT_HASH_MISMATCH"),
    ],
)
def test_recovery_certification_rejects_forged_assessment(
    permit_inputs, mutation, reason
):
    assessment = permit_inputs["assessment"]
    if mutation == "cap":
        forged = _resign_assessment(
            assessment, option_action_cap=assessment.option_action_cap + 1
        )
    elif mutation == "base_steps":
        forged = _resign_assessment(
            assessment, base_policy_steps=assessment.base_policy_steps + 1
        )
    elif mutation == "protected_empty":
        forged = _resign_assessment(assessment, protected_true_facts=frozenset())
    elif mutation == "protected_reduced":
        forged = _resign_assessment(
            assessment, protected_true_facts=frozenset({HANDOFF_GOAL})
        )
    elif mutation == "protected_substituted":
        forged = _resign_assessment(
            assessment, protected_true_facts=frozenset({Fact("handempty")})
        )
    elif mutation == "capability":
        forged = _resign_assessment(assessment, capability_sha256="0" * 64)
    elif mutation == "place_node":
        forged = _resign_assessment(assessment, place_node_id="changed-node")
    elif mutation == "reason":
        forged = _resign_assessment(assessment, reason="CHANGED")
    elif mutation == "event_type":
        forged = _resign_assessment(assessment, event_type="CHANGED")
    elif mutation == "event_id":
        forged = _resign_assessment(assessment, event_id="changed")
    elif mutation == "monitor":
        forged = _resign_assessment(assessment, monitor_contract_sha256="changed")
    else:
        forged = replace(assessment, assessment_sha256="0" * 64)
    changed = permit_inputs | {
        "assessment": forged,
        "expected_assessment_sha256": forged.assessment_sha256,
    }
    if mutation == "base_steps":
        changed["expected_base_policy_steps"] = forged.base_policy_steps
    result = certify_task5_recovery(**changed)
    assert result.granted is False
    assert result.reason == reason


@pytest.mark.parametrize("mutation", ["nonfinite_base", "malformed_protected"])
def test_recovery_certification_rejects_unhashable_assessment_fields(
    permit_inputs, mutation
):
    assessment = permit_inputs["assessment"]
    if mutation == "nonfinite_base":
        forged = replace(assessment, base_policy_steps=float("nan"))
    else:
        forged = replace(assessment, protected_true_facts=frozenset({"not-a-fact"}))
    result = certify_task5_recovery(
        **(permit_inputs | {"assessment": forged})
    )
    assert result.granted is False
    assert result.reason == "ASSESSMENT_HASH_MISMATCH"


def test_recovery_certification_denial_sanitizes_nonjson_event_id(permit_inputs):
    assessment = replace(permit_inputs["assessment"], event_id=object())
    first = certify_task5_recovery(
        **(permit_inputs | {"assessment": assessment})
    )
    second = certify_task5_recovery(
        **(permit_inputs | {"assessment": assessment})
    )
    assert first.granted is False
    assert first.reason == "ASSESSMENT_HASH_MISMATCH"
    assert first.event_id == ""
    assert first.permit_sha256 == second.permit_sha256
    assert len(first.permit_sha256) == 64


def test_recovery_certification_denial_discards_nonjson_plan(permit_inputs):
    result = certify_task5_recovery(
        **(
            permit_inputs
            | {
                "candidate_plan": (object(),),
                "expected_assessment_sha256": None,
            }
        )
    )
    assert result.granted is False
    assert result.reason == "EXPECTED_ASSESSMENT_SHA256_INVALID"
    assert result.plan == ()
    assert len(result.permit_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("zero", "PLAN_ACTION_COUNT_NOT_ONE"),
        ("multiple", "PLAN_ACTION_COUNT_NOT_ONE"),
        ("wrong_action", "PLAN_ACTION_MISMATCH"),
        ("signed_state", "SIGNED_STATE_INVALID"),
        ("val_error", "VAL_VALIDATION_ERROR"),
        ("graph_hash", "GRAPH_HASH_CHANGED"),
        ("capability_hash", "CAPABILITY_NOT_FROZEN"),
        ("checkpoint_hash", "RECOVERY_CHECKPOINT_HASH_INVALID"),
        ("snapshot_audit", "STRICT_AUDITED_SNAPSHOT_REQUIRED"),
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
    elif mutation == "checkpoint_hash":
        changed["recovery_checkpoint_sha256"] = "not-a-hash"
    else:
        snapshot = replace(changed["snapshot"])
        object.__setattr__(snapshot, "evidence_payload_json", "{}")
        changed["snapshot"] = snapshot
    result = certify_task5_recovery(**changed)
    assert result.granted is False
    assert result.reason == reason
    assert len(result.permit_sha256) == 64


def test_recovery_certification_binds_exactly_one_val_certificate(
    granted_permit, eligible_assessment
):
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
    assert granted_permit.protected_true_facts == eligible_assessment.protected_true_facts
    assert granted_permit.assessment_sha256 == eligible_assessment.assessment_sha256


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
        ("snapshot_tampered", "STRICT_AUDITED_POST_SNAPSHOT_REQUIRED"),
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
    elif mutation == "snapshot_tampered":
        tampered = replace(snapshot)
        object.__setattr__(tampered, "evidence_payload_json", "{}")
        changed["post_snapshot"] = tampered
    elif mutation in {"unknown", "target", "handoff", "invariant"}:
        if mutation == "unknown":
            fact = HANDOFF_GOAL
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts - {fact}
        elif mutation == "target":
            fact = TARGET
            true_facts = snapshot.true_facts - {fact}
            false_facts = snapshot.false_facts | {fact}
        elif mutation == "handoff":
            fact = HANDOFF_GOAL
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


@pytest.mark.parametrize(
    "protected_true_facts",
    [
        frozenset(),
        frozenset({HANDOFF_GOAL}),
        frozenset({Fact("handempty")}),
    ],
)
def test_recovery_commit_rejects_forged_handoff_fact_sets(
    commit_inputs, protected_true_facts
):
    forged = replace(
        commit_inputs["permit"], protected_true_facts=protected_true_facts
    )
    result = verify_task5_recovery_commit(
        **(commit_inputs | {"permit": forged})
    )
    assert result.reason == "PERMIT_HASH_CHANGED"


def _resign_permit(permit, **changes):
    changed = replace(permit, **changes, permit_sha256="")
    return replace(changed, permit_sha256=changed.recompute_sha256())


@pytest.mark.parametrize(
    "mutation",
    ["cap", "protected", "plan", "checkpoint", "capability", "assessment"],
)
def test_recovery_commit_rejects_resigned_permit_mutations(
    commit_inputs, mutation
):
    permit = commit_inputs["permit"]
    if mutation == "cap":
        forged = _resign_permit(permit, action_cap=permit.action_cap + 1)
    elif mutation == "protected":
        forged = _resign_permit(permit, protected_true_facts=frozenset())
    elif mutation == "plan":
        forged = _resign_permit(permit, plan=(), plan_sha256=None)
    elif mutation == "checkpoint":
        forged = _resign_permit(permit, recovery_checkpoint_sha256="0" * 64)
    elif mutation == "capability":
        forged = _resign_permit(permit, capability_sha256="0" * 64)
    else:
        forged = _resign_permit(permit, assessment_sha256="0" * 64)
    result = verify_task5_recovery_commit(
        **(commit_inputs | {"permit": forged})
    )
    assert result.committed is False
    assert result.reason == "PERMIT_HASH_CHANGED"


@pytest.mark.parametrize("mutation", ["malformed_protected", "malformed_plan"])
def test_recovery_commit_rejects_unhashable_permit_fields(commit_inputs, mutation):
    permit = commit_inputs["permit"]
    if mutation == "malformed_protected":
        forged = replace(permit, protected_true_facts=frozenset({"not-a-fact"}))
    else:
        forged = replace(permit, plan=("not-an-action",))
    result = verify_task5_recovery_commit(
        **(commit_inputs | {"permit": forged})
    )
    assert result.committed is False
    assert result.reason == "PERMIT_HASH_CHANGED"


def test_recovery_commit_denial_sanitizes_nonjson_event_id(commit_inputs):
    first = verify_task5_recovery_commit(
        **(commit_inputs | {"event_id": object()})
    )
    second = verify_task5_recovery_commit(
        **(commit_inputs | {"event_id": object()})
    )
    assert first.committed is False
    assert first.reason == "EVENT_ID_CHANGED"
    assert first.commit_sha256 == second.commit_sha256
    assert len(first.commit_sha256) == 64


def test_recovery_commit_audit_denial_sanitizes_nonjson_digest(commit_inputs):
    snapshot = replace(commit_inputs["post_snapshot"])
    object.__setattr__(snapshot, "evidence_hash", object())
    first = verify_task5_recovery_commit(
        **(commit_inputs | {"post_snapshot": snapshot})
    )
    second = verify_task5_recovery_commit(
        **(commit_inputs | {"post_snapshot": snapshot})
    )
    assert first.committed is False
    assert first.reason == "STRICT_AUDITED_POST_SNAPSHOT_REQUIRED"
    assert first.commit_sha256 == second.commit_sha256
    assert len(first.commit_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_actions", True),
        ("recovery_actions", 1.5),
        ("base_actions", float("inf")),
        ("recovery_actions", float("nan")),
    ],
)
def test_recovery_commit_rejects_non_integer_action_counts(
    commit_inputs, field, value
):
    result = verify_task5_recovery_commit(**(commit_inputs | {field: value}))
    assert result.committed is False
    assert result.reason == "ACTION_COUNT_INVALID"
    assert len(result.commit_sha256) == 64


def test_recovery_commit_requires_joint_symbolic_and_native_success(commit_inputs):
    result = verify_task5_recovery_commit(**commit_inputs)
    assert result.committed is True
    assert result.reason == "COMMITTED"
    assert result.base_actions == 340
    assert result.recovery_actions == 100
    assert result.combined_actions == 440
