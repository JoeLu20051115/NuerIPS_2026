from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "run_robotwin_logiv_10x10",
    ROOT / "scripts" / "run_robotwin_logiv_10x10.py",
)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_task_specific_stall_observations_override_global_default() -> None:
    config = {
        "base_stall_observations": 4,
        "base_stall_observations_by_task": {
            "handover_block": 1,
            "place_dual_shoes": 3,
        },
    }

    assert LAUNCHER._stall_observations(config, "handover_block") == 1
    assert LAUNCHER._stall_observations(config, "place_dual_shoes") == 3
    assert LAUNCHER._stall_observations(config, "move_can_pot") == 4


def test_stage_specific_thresholds_are_parsed_per_task() -> None:
    config = {
        "stage_stall_observations_by_task": {
            "handover_block": [4, 4, 1],
        }
    }

    assert LAUNCHER._stage_stall_observations(config, "handover_block") == [4, 4, 1]
    assert LAUNCHER._stage_stall_observations(config, "move_can_pot") is None


def test_task_specific_stall_observations_must_be_positive() -> None:
    config = {"base_stall_observations_by_task": {"handover_block": 0}}

    try:
        LAUNCHER._stall_observations(config, "handover_block")
    except ValueError as error:
        assert "handover_block" in str(error)
    else:
        raise AssertionError("zero stall observations must be rejected")


def test_task_specific_base_protection_is_converted_to_dispatches() -> None:
    config = {
        "action_chunk_steps": 50,
        "min_base_steps_by_task": {
            "turn_switch": 350,
            "stack_blocks_three": 650,
        },
    }

    assert LAUNCHER._min_base_dispatches(config, "turn_switch") == 7
    assert LAUNCHER._min_base_dispatches(config, "stack_blocks_three") == 13
    assert LAUNCHER._min_base_dispatches(config, "move_can_pot") == 0


def test_launcher_passes_distinct_repair_chunk_size() -> None:
    config = {"action_chunk_steps": 50, "repair_action_chunk_steps": 10}

    assert LAUNCHER._repair_action_chunk_steps(config) == 10


def test_launcher_passes_configured_vlm_image_detail() -> None:
    assert LAUNCHER._vlm_image_detail({"vlm_image_detail": "high"}) == "high"


def test_full_dag_control_is_explicitly_configured() -> None:
    assert LAUNCHER._dag_from_start({"dag_from_start": True}) is True
    assert LAUNCHER._dag_from_start({}) is False


def test_registered_dag_prompts_are_explicitly_configured() -> None:
    assert LAUNCHER._use_registered_dag_prompts(
        {"use_registered_dag_prompts": True}
    ) is True
    assert LAUNCHER._use_registered_dag_prompts({}) is False


def test_checkpoint_cfn_is_selected_only_for_configured_tasks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    cfn = checkpoint / "cfns" / "handover_block_cfn.pt"
    cfn.parent.mkdir(parents=True)
    cfn.touch()
    config = {
        "checkpoint": str(checkpoint),
        "repair_cfn_tasks": ["handover_block"],
    }

    assert LAUNCHER._repair_cfn_path(config, "handover_block") == cfn
    assert LAUNCHER._repair_cfn_path(config, "move_can_pot") is None


def test_original_repair_prompt_ablation_is_explicitly_configured() -> None:
    assert LAUNCHER._preserve_original_repair_prompt(
        {"preserve_original_repair_prompt": True}
    ) is True
    assert LAUNCHER._preserve_original_repair_prompt({}) is False


def test_policy_replan_steps_are_optional() -> None:
    assert LAUNCHER._policy_replan_steps({"policy_replan_steps": 10}) == 10
    assert LAUNCHER._policy_replan_steps({}) is None


def test_control_options_can_be_tuned_per_task() -> None:
    config = {
        "repair_action_chunk_steps": 10,
        "repair_action_chunk_steps_by_task": {"open_microwave": 50},
        "vlm_image_detail": "low",
        "vlm_image_detail_by_task": {"open_microwave": "high"},
        "dag_from_start": True,
        "dag_from_start_by_task": {"handover_block": False},
        "use_registered_dag_prompts": False,
        "use_registered_dag_prompts_by_task": {"turn_switch": True},
        "preserve_original_repair_prompt": False,
        "preserve_original_repair_prompt_by_task": {"place_dual_shoes": True},
        "policy_replan_steps": None,
        "policy_replan_steps_by_task": {"stack_blocks_three": 10},
    }

    assert LAUNCHER._repair_action_chunk_steps(config, "open_microwave") == 50
    assert LAUNCHER._repair_action_chunk_steps(config, "turn_switch") == 10
    assert LAUNCHER._vlm_image_detail(config, "open_microwave") == "high"
    assert LAUNCHER._vlm_image_detail(config, "turn_switch") == "low"
    assert LAUNCHER._dag_from_start(config, "handover_block") is False
    assert LAUNCHER._dag_from_start(config, "turn_switch") is True
    assert LAUNCHER._use_registered_dag_prompts(config, "turn_switch") is True
    assert LAUNCHER._preserve_original_repair_prompt(
        config, "place_dual_shoes"
    ) is True
    assert LAUNCHER._policy_replan_steps(config, "stack_blocks_three") == 10
    assert LAUNCHER._policy_replan_steps(config, "turn_switch") is None


def test_launcher_accepts_explicit_openai_key_alias_without_secret_files(
    monkeypatch,
) -> None:
    key = "sk-" + "a" * 48
    monkeypatch.setenv("OPENAI_API_KEY", "none")
    monkeypatch.setenv("OPENAI_KEY", key)

    assert LAUNCHER._api_key() == key


def test_launcher_prefers_standard_openai_api_key(monkeypatch) -> None:
    standard = "sk-" + "b" * 48
    alias = "sk-" + "c" * 48
    monkeypatch.setenv("OPENAI_API_KEY", standard)
    monkeypatch.setenv("OPENAI_KEY", alias)

    assert LAUNCHER._api_key() == standard


def test_explicit_gpu_allows_tasks_independent_of_worker_group(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def record_run(gpu, tasks, args):
        calls.append((gpu, tasks, args))
        return 0

    monkeypatch.setattr(LAUNCHER, "_run_worker", record_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_robotwin_logiv_10x10.py",
            "--worker",
            "0",
            "--gpu",
            "1",
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--output",
            str(tmp_path / "output"),
            "--tag",
            "test",
            "--taco",
            str(tmp_path / "taco"),
            "--logiv-root",
            str(tmp_path / "logiv"),
            "--python",
            sys.executable,
            "--tokenizer",
            str(tmp_path / "tokenizer.model"),
            "--val-binary",
            str(tmp_path / "Validate"),
            "--tasks",
            "place_dual_shoes",
            "turn_switch",
        ],
    )

    assert LAUNCHER.main() == 0
    assert len(calls) == 1
    gpu, tasks, _args = calls[0]
    assert gpu == 1
    assert tasks == ("place_dual_shoes", "turn_switch")
