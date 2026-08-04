from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pi05_libero_repro.logiv.dag import SchemaOnlyCausalDagCompiler
from pi05_libero_repro.logiv.configuration import resolved_json_sha256
from pi05_libero_repro.logiv.evaluation import (
    EvaluationContract,
    EvaluationContractError,
    GlobalRepairOperator,
    MethodArm,
    certify_initial_package,
)
from pi05_libero_repro.logiv.model import GoalMode
from pi05_libero_repro.logiv.initial_proposal import (
    InitialProposalResult,
    InitialProposalStatus,
)
from pi05_libero_repro.logiv.shadow_runtime import (
    ShadowRuntime,
    ShadowRuntimeCounters,
)
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.repair import RepairBounds, RepairOperator, RetryPolicy
from pi05_libero_repro.logiv.val import ValWrapper
from scripts.eval_logiv_libero import (
    _allocate_episode_artifact_dir,
    _base_physical_attempts,
    _parser,
    _replace_episode_environment,
    _run_config,
    _shadow_artifact_payloads,
    _validate_shadow_options,
)
from pi05_libero_repro.protocol import EpisodeOutcome, ShadowFailureRecord
import scripts.eval_logiv_libero as evaluator_script


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")


def test_evaluator_accepts_run_scoped_proposal_and_coverage_configs() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "config-isolation",
            "--method-arm",
            "BASE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/config-isolation",
            "--proposal-config",
            "/tmp/proposals.json",
            "--coverage-manifest",
            "/tmp/coverage.json",
        ]
    )

    assert args.proposal_config == Path("/tmp/proposals.json")
    assert args.coverage_manifest == Path("/tmp/coverage.json")


def test_evaluator_parser_accepts_shadow_data_collection_arm() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "shadow",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--episode-indices",
            "0",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/shadow",
            "--collect-recovery-roots",
            "--recovery-root-split",
            "DEV",
        ]
    )

    assert args.collect_recovery_roots
    assert args.recovery_root_split == "DEV"
    assert MethodArm(args.method_arm) is MethodArm.SHADOW_LOGIV
    assert args.shadow_monitor_interval_steps == 5
    assert args.shadow_confirmations == 3


def test_run_config_hashes_resolved_extended_configs(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.json"
    coverage = tmp_path / "coverage.json"
    proposal.write_text(
        '{"extends":"' + str(Path("configs/logiv/libero10-scripted-proposals.json").resolve()) + '"}',
        encoding="utf-8",
    )
    coverage.write_text(
        '{"extends":"' + str(Path("configs/logiv/libero10-coverage.json").resolve()) + '"}',
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--run-id",
            "resolved-config-hash",
            "--method-arm",
            "BASE",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--port",
            "8010",
            "--output-dir",
            str(tmp_path / "output"),
            "--proposal-config",
            str(proposal),
            "--coverage-manifest",
            str(coverage),
        ]
    )

    config = _run_config(args, (0,), (0,))

    assert config["proposal_config_sha256"] == resolved_json_sha256(proposal)
    assert config["coverage_manifest_sha256"] == resolved_json_sha256(coverage)


def test_run_config_freezes_shadow_collection_contract() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "shadow-config",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--episode-indices",
            "0",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/shadow-config",
            "--collect-recovery-roots",
        ]
    )

    contracts = _validate_shadow_options(args, (8,))
    config = _run_config(args, (8,), (0,))

    assert contracts[8].task_id == 8
    assert config["collect_recovery_roots"] is True
    assert config["recovery_root_split"] == "DEV"
    assert config["shadow_monitor_interval_steps"] == 5
    assert config["shadow_confirmations"] == 3
    assert config["shadow_monitor_contract"] == str(args.shadow_monitor_contract)
    assert config["shadow_monitor_contract_sha256"] == contracts[8].contract_sha256
    assert config["shadow_monitor_contract_sha256_by_task"] == {
        "8": contracts[8].contract_sha256
    }


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--method-arm", "BASE"), "SHADOW_LOGIV"),
        (("--recovery-root-split", "TRAIN"), "DEV"),
    ],
)
def test_phase0_recovery_collection_rejects_unsafe_scope(extra, message) -> None:
    values = [
        "--run-id",
        "bad-shadow",
        "--method-arm",
        "SHADOW_LOGIV",
        "--goal-mode",
        "METADATA_ASSISTED",
        "--deviation-mode",
        "NOMINAL",
        "--oracle-grounding",
        "--development-only",
        "--port",
        "8010",
        "--output-dir",
        "/tmp/bad-shadow",
        "--collect-recovery-roots",
    ]
    option, value = extra
    if option in values:
        values[values.index(option) + 1] = value
    else:
        values.extend((option, value))
    args = _parser().parse_args(values)

    with pytest.raises(ValueError, match=message):
        _validate_shadow_options(args, (8,))


def test_shadow_monitor_cli_cannot_override_frozen_detector_semantics() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "bad-monitor",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--development-only",
            "--task-ids",
            "8",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/bad-monitor",
            "--shadow-monitor-interval-steps",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="interval"):
        _validate_shadow_options(args, (8,))


def test_phase0_collection_requires_development_mode() -> None:
    args = _parser().parse_args(
        [
            "--run-id",
            "bad-holdout",
            "--method-arm",
            "SHADOW_LOGIV",
            "--goal-mode",
            "METADATA_ASSISTED",
            "--deviation-mode",
            "NOMINAL",
            "--oracle-grounding",
            "--prompt-locked",
            "--task-ids",
            "8",
            "--port",
            "8010",
            "--output-dir",
            "/tmp/bad-holdout",
            "--collect-recovery-roots",
        ]
    )

    with pytest.raises(ValueError, match="development-only"):
        _validate_shadow_options(args, (8,))


def test_shadow_artifacts_render_step0_containment_as_not_attempted() -> None:
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=1,
        inference_requests=1,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[np.zeros(7)],
        shadow_calls=2,
        shadow_errors=1,
        shadow_failure_records=(
            ShadowFailureRecord(0, "INPUT_COPY", "MemoryError"),
        ),
        shadow_wall_seconds=0.25,
        shadow_parity_valid=True,
    )
    runtime = ShadowRuntime(None, None, None, ShadowRuntimeCounters())

    proposal, monitor, compute, accounting = _shadow_artifact_payloads(
        outcome, runtime
    )

    assert proposal["status"] == "NOT_ATTEMPTED"
    assert proposal["request_count"] == 0
    assert proposal["reason"] == "INPUT_COPY:MemoryError"
    assert accounting["initial_proposal_status"] == "NOT_ATTEMPTED"
    assert accounting["initial_proposal_reason_code"] == "INPUT_COPY:MemoryError"
    assert monitor["callback_errors"] == 1
    assert monitor["aggregate_errors"] == 1
    assert compute["base_policy_requests"] == 1


def test_shadow_error_aggregate_sums_each_sole_owner_once() -> None:
    metrics = SimpleNamespace(
        snapshot_calls=7,
        snapshot_errors=2,
        event_tracker_errors=3,
        evidence_overflows=4,
        trigger_callback_errors=5,
        anomaly_candidates=1,
        confirmed_deviations=1,
        stale_certificates=0,
    )
    runtime = ShadowRuntime(
        None,
        None,
        SimpleNamespace(metrics=metrics),
        ShadowRuntimeCounters(
            root_count=2,
            root_write_errors=6,
            proposal_callback_errors=7,
            provenance_errors=8,
            trace_errors=9,
        ),
    )
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=0,
        inference_requests=0,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[],
        shadow_errors=1,
    )

    _, monitor, _, accounting = _shadow_artifact_payloads(outcome, runtime)

    assert monitor["trace_errors"] == 9
    assert monitor["aggregate_errors"] == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
    assert accounting["shadow_monitor_errors"] == monitor["aggregate_errors"]


def test_shadow_graph_artifact_is_written_only_for_accepted_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = getattr(evaluator_script, "_write_shadow_graph_artifact", None)
    assert writer is not None
    graph = SimpleNamespace(graph_hash="g" * 64)
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=graph,
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
        state_trace=[{"policy_step": 0}],
    )
    writes = []
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: writes.append((path, payload)),
    )

    assert writer(tmp_path, runtime) is graph
    assert writes == [
        (
            tmp_path / "graph.json",
            {"graph": graph, "state_trace": [{"policy_step": 0}]},
        )
    ]

    runtime.initial_proposal = replace(
        runtime.initial_proposal,
        status=InitialProposalStatus.REJECTED,
        validation=None,
    )
    assert writer(tmp_path, runtime) is None
    assert len(writes) == 1


def test_shadow_graph_artifact_write_failure_is_contained_and_accounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = SimpleNamespace(graph_hash="g" * 64)
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=graph,
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert evaluator_script._write_shadow_graph_artifact(tmp_path, runtime) is None
    assert runtime.counters.trace_errors == 1
    outcome = EpisodeOutcome(
        success=True,
        done=True,
        check_success=True,
        steps=1,
        inference_requests=1,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[np.zeros(7)],
    )
    _, monitor, _, accounting = _shadow_artifact_payloads(outcome, runtime)
    assert monitor["trace_errors"] == 1
    assert accounting["shadow_monitor_errors"] == 1


def test_shadow_outcome_none_fallback_counts_contained_graph_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounting_for = getattr(evaluator_script, "_shadow_record_accounting", None)
    assert accounting_for is not None
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.ACCEPTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=object(),
            validation=SimpleNamespace(
                certified_episode=SimpleNamespace(
                    graph=SimpleNamespace(graph_hash="g" * 64),
                    certificate=SimpleNamespace(certificate_hash="c" * 64),
                )
            ),
            reason=None,
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    monkeypatch.setattr(
        evaluator_script,
        "_graph_json",
        lambda value, *, state_trace: {"graph": value, "state_trace": state_trace},
    )
    monkeypatch.setattr(
        evaluator_script,
        "_write_json",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert evaluator_script._write_shadow_graph_artifact(tmp_path, runtime) is None
    accounting = accounting_for(
        outcome=None,
        runtime=runtime,
        exception_text="RuntimeError: Base evaluator failed",
        base_policy_requests=0,
    )

    assert accounting["initial_proposal_status"] == "ACCEPTED"
    assert accounting["initial_proposal_requests"] == 1
    assert accounting["shadow_monitor_errors"] == 1


@pytest.mark.parametrize(
    ("runtime", "expected_status", "expected_requests", "expected_reason"),
    [
        (None, "NOT_ATTEMPTED", 0, "ROLLOUT:RuntimeError"),
        (
            ShadowRuntime(
                InitialProposalResult(
                    status=InitialProposalStatus.REJECTED,
                    provider="test-provider",
                    request_count=1,
                    elapsed_seconds=0.1,
                    package=None,
                    validation=None,
                    reason="ValueError: rejected",
                ),
                None,
                None,
                ShadowRuntimeCounters(trace_errors=2),
            ),
            "REJECTED",
            1,
            "ValueError",
        ),
    ],
)
def test_shadow_outcome_none_fallback_preserves_proposal_accounting(
    runtime: ShadowRuntime | None,
    expected_status: str,
    expected_requests: int,
    expected_reason: str,
) -> None:
    accounting_for = getattr(evaluator_script, "_shadow_record_accounting", None)
    assert accounting_for is not None

    accounting = accounting_for(
        outcome=None,
        runtime=runtime,
        exception_text="RuntimeError: Base evaluator failed",
        base_policy_requests=0,
    )

    assert accounting["initial_proposal_status"] == expected_status
    assert accounting["initial_proposal_requests"] == expected_requests
    assert accounting["initial_proposal_reason_code"] == expected_reason
    assert accounting["shadow_monitor_errors"] == (
        0 if runtime is None else runtime.counters.trace_errors
    )


def test_shadow_compute_seconds_separate_initial_proposal_from_monitor() -> None:
    runtime = ShadowRuntime(
        InitialProposalResult(
            status=InitialProposalStatus.REJECTED,
            provider="test-provider",
            request_count=1,
            elapsed_seconds=0.1,
            package=None,
            validation=None,
            reason="ValueError: rejected",
        ),
        None,
        None,
        ShadowRuntimeCounters(),
    )
    outcome = EpisodeOutcome(
        success=False,
        done=False,
        check_success=False,
        steps=0,
        inference_requests=0,
        first_frame=np.zeros((1, 1, 3)),
        replay_frames=[],
        actions=[],
        shadow_wall_seconds=0.25,
    )

    _, monitor, compute, accounting = _shadow_artifact_payloads(outcome, runtime)

    assert monitor["callback_seconds"] == 0.25
    assert compute["initial_proposal_seconds"] == 0.1
    assert compute["shadow_monitor_seconds"] == pytest.approx(0.15)
    assert accounting["shadow_monitor_seconds"] == pytest.approx(0.15)


def test_base_rollout_is_one_physical_attempt_not_one_attempt_per_control_step() -> None:
    assert _base_physical_attempts(444) == 1
    assert _base_physical_attempts(0) == 0
    with pytest.raises(ValueError, match="steps"):
        _base_physical_attempts(-1)


def test_each_episode_replaces_and_closes_the_previous_simulator_environment() -> None:
    class FakeEnv:
        def __init__(self, marker):
            self.marker = marker
            self.closed = False

        def close(self):
            self.closed = True

    created = []

    def factory(**kwargs):
        env = FakeEnv(kwargs)
        created.append(env)
        return env

    old = FakeEnv("old")
    first = _replace_episode_environment(
        old,
        factory=factory,
        bddl_file="task.bddl",
    )
    second = _replace_episode_environment(
        first,
        factory=factory,
        bddl_file="task.bddl",
    )

    assert old.closed is True
    assert first.closed is True
    assert second is created[1]
    assert second.marker == {
        "bddl_file_name": "task.bddl",
        "camera_heights": 256,
        "camera_widths": 256,
    }


def test_resume_preserves_orphan_artifacts_and_allocates_a_generation(tmp_path: Path) -> None:
    first = _allocate_episode_artifact_dir(tmp_path, task_id=8, episode_idx=3)
    (first / "partial.json").write_text("interrupted")

    resumed = _allocate_episode_artifact_dir(tmp_path, task_id=8, episode_idx=3)

    assert first.name == "episode_003"
    assert resumed.name == "episode_003_resume_001"
    assert (first / "partial.json").read_text() == "interrupted"
    assert (resumed / "resume.json").exists()


def test_initial_certification_preserves_task8_parallel_graph() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=5)
    wrapper = ValWrapper(REAL_VAL, timeout_seconds=5.0)
    bounds = RepairBounds(max_edits=3, max_candidates=1000, max_val_calls=20)

    certified = certify_initial_package(
        package,
        package.proposal.initial_snapshot,
        episode_id="test-task8-episode0",
        val_wrapper=wrapper,
        allowed_schemas=frozenset({"pick", "put-down", "place-on", "place-held-on"}),
        repair_bounds=bounds,
        decompose_macro_sources=frozenset({"kitchen_table_recovery_surface"}),
    )

    assert certified.graph.action_layer_width() == 2
    first, second = certified.graph.canonical_agenda
    assert certified.graph.edge(first, second) is None
    assert certified.graph.edge(second, first) is None
    assert certified.certificate.certificate_hash == certified.graph.certificate_hash
    assert certified.repair_operator.decompose_macro_sources == frozenset(
        {"kitchen_table_recovery_surface"}
    )


def test_schema_only_arm_never_reuses_full_certificate() -> None:
    package = ScriptedProposalProvider().propose(8, epoch_id=5)
    certified = certify_initial_package(
        package,
        package.proposal.initial_snapshot,
        episode_id="test-task8-episode0",
        val_wrapper=ValWrapper(REAL_VAL, timeout_seconds=5.0),
        allowed_schemas=frozenset({"pick", "put-down", "place-on", "place-held-on"}),
        repair_bounds=RepairBounds(max_edits=3, max_candidates=1000, max_val_calls=20),
    )
    schema_graph = SchemaOnlyCausalDagCompiler().compile(
        certified.problem,
        certified.plan,
        certified.occurrence_sidecar,
        certified.certificate.context,
    )
    assert schema_graph.certificate_hash != certified.certificate.certificate_hash
    assert schema_graph.action_layer_width() == 2


def test_global_repair_ablation_discards_causal_slice() -> None:
    class CapturingRepair:
        def __init__(self):
            self.slice = "unset"

        def repair(self, *args, causal_slice=None, **kwargs):
            self.slice = causal_slice
            return "result"

    wrapped = CapturingRepair()
    result = GlobalRepairOperator(wrapped).repair("problem", causal_slice="localized")
    assert result == "result"
    assert wrapped.slice is None


def test_evaluation_contract_requires_explicit_modes_oracle_and_locked_holdout() -> None:
    base = EvaluationContract(
        method_arm=MethodArm.FULL_LOGIV,
        goal_mode=GoalMode.METADATA_ASSISTED,
        deviation_mode="NOMINAL",
        oracle_grounding=True,
        development_only=True,
        prompt_locked=False,
        task_ids=(8,),
        episode_indices=(0, 1),
    )
    base.validate()

    with pytest.raises(EvaluationContractError, match="oracle grounding acknowledgement"):
        replace(base, oracle_grounding=False).validate()
    replace(base, method_arm=MethodArm.BASE, oracle_grounding=False).validate()
    with pytest.raises(EvaluationContractError, match="locked prompt"):
        replace(base, development_only=False, prompt_locked=False).validate()
    with pytest.raises(EvaluationContractError, match="not supported by the no-API provider"):
        replace(base, goal_mode=GoalMode.GOAL_PREDICTION).validate()
    with pytest.raises(EvaluationContractError, match="frozen 10-task manifest"):
        replace(base, task_ids=(10,)).validate()
