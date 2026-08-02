from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider
from pi05_libero_repro.logiv.repair import RepairBounds, RepairOperator, RetryPolicy
from pi05_libero_repro.logiv.val import ValWrapper
from scripts.eval_logiv_libero import (
    _allocate_episode_artifact_dir,
    _base_physical_attempts,
    _parser,
    _replace_episode_environment,
    _run_config,
)


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
