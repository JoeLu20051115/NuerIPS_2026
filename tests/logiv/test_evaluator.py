from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pi05_libero_repro.logiv.dag import SchemaOnlyCausalDagCompiler
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
)


REAL_VAL = Path("/home/xingrui/.local/bin/Validate")


def test_base_rollout_is_one_physical_attempt_not_one_attempt_per_control_step() -> None:
    assert _base_physical_attempts(444) == 1
    assert _base_physical_attempts(0) == 0
    with pytest.raises(ValueError, match="steps"):
        _base_physical_attempts(-1)


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
    )

    assert certified.graph.action_layer_width() == 2
    first, second = certified.graph.canonical_agenda
    assert certified.graph.edge(first, second) is None
    assert certified.graph.edge(second, first) is None
    assert certified.certificate.certificate_hash == certified.graph.certificate_hash


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
