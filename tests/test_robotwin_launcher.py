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


def test_task_specific_stall_observations_must_be_positive() -> None:
    config = {"base_stall_observations_by_task": {"handover_block": 0}}

    try:
        LAUNCHER._stall_observations(config, "handover_block")
    except ValueError as error:
        assert "handover_block" in str(error)
    else:
        raise AssertionError("zero stall observations must be rejected")
