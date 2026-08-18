from __future__ import annotations

from pathlib import Path

from pi05_libero_repro.logiv.libero_adapter import TaskBinding
from pi05_libero_repro.logiv.prompts import SubtaskPromptRenderer
from pi05_libero_repro.logiv.proposal import ScriptedProposalProvider


ROOT = Path(__file__).resolve().parents[2]
ORIGIN = ROOT / "configs/logiv/origin"


def test_origin_task_binding_loads_the_frozen_runtime_contract() -> None:
    binding = TaskBinding.from_manifest(ORIGIN / "coverage.json", task_id=5)

    assert binding.frozen is True
    assert "black_book_1" in binding.registered_objects
    assert "place-in" in binding.supported_action_schemas
    assert binding.resolve("black_book_1")[0] == "black_book_1"


def test_origin_prompt_renderer_uses_the_stable_profile() -> None:
    package = ScriptedProposalProvider().propose(task_id=8, epoch_id=0)
    renderer = SubtaskPromptRenderer()
    prompts = [
        renderer.render(candidate.action)
        for candidate in package.proposal.candidate_subtasks
    ]

    assert renderer.prompt_version == "logiv-origin"
    assert len(set(prompts)) == 2
    assert any("left" in prompt.lower() for prompt in prompts)
    assert any("right" in prompt.lower() for prompt in prompts)


def test_origin_task5_proposal_and_coverage_agree_on_macro_schema() -> None:
    package = ScriptedProposalProvider().propose(task_id=5, epoch_id=0)
    binding = TaskBinding.from_manifest(ORIGIN / "coverage.json", task_id=5)
    actions = tuple(
        candidate.action for candidate in package.proposal.candidate_subtasks
    )

    assert [action.schema for action in actions] == ["place-in"]
    assert all(action.schema in binding.supported_action_schemas for action in actions)
