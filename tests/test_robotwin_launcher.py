from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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
