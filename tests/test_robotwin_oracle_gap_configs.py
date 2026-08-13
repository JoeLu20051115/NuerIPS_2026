import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "robotwin" / "logiv-pddl-10x10-v1.json"
ORACLE_PATH = (
    ROOT
    / "results"
    / "robotwin-logiv-strict-baseline-instruction-oracle-development-20260813.json"
)
CONFIG_PATHS = {
    "cfn": ROOT / "configs" / "robotwin" / "logiv-oracle-gap-final-cfn.json",
    "registered": (
        ROOT / "configs" / "robotwin" / "logiv-oracle-gap-final-registered.json"
    ),
    "original": (
        ROOT / "configs" / "robotwin" / "logiv-oracle-gap-final-original.json"
    ),
    "replan": ROOT / "configs" / "robotwin" / "logiv-oracle-gap-final-replan.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_final_stack_configs_are_exactly_the_forty_oracle_gaps() -> None:
    protocol = _load(PROTOCOL_PATH)
    oracle = _load(ORACLE_PATH)
    configs = {name: _load(path) for name, path in CONFIG_PATHS.items()}

    protocol_cells = {
        (task, seed)
        for task, seeds in protocol["tasks"].items()
        for seed in seeds
    }
    selected_cells = {(row["task"], row["seed"]) for row in oracle["selected"]}
    expected_gaps = protocol_cells - selected_cells

    configured_cells = []
    for config in configs.values():
        configured_cells.extend(
            (task, seed)
            for task, seeds in config["tasks"].items()
            for seed in seeds
        )

    assert len(expected_gaps) == 40
    assert len(configured_cells) == 40
    assert len(set(configured_cells)) == 40
    assert set(configured_cells) == expected_gaps


def test_gap_configs_keep_each_frozen_episode_instruction() -> None:
    protocol = _load(PROTOCOL_PATH)
    instruction_by_cell = {
        (task, seed): instruction
        for task, seeds in protocol["tasks"].items()
        for seed, instruction in zip(seeds, protocol["instructions"][task], strict=True)
    }

    for path in CONFIG_PATHS.values():
        config = _load(path)
        assert set(config["tasks"]) == set(config["instructions"])
        for task, seeds in config["tasks"].items():
            instructions = config["instructions"][task]
            assert len(seeds) == len(instructions)
            assert instructions == [instruction_by_cell[(task, seed)] for seed in seeds]


def test_gap_profiles_have_the_frozen_task_membership_and_runtime_flags() -> None:
    configs = {name: _load(path) for name, path in CONFIG_PATHS.items()}
    checkpoint = (
        ROOT / "artifacts" / "checkpoints" / "pi05_TACO_robotwin2_finetuned"
    ).resolve()

    for config in configs.values():
        assert Path(config["checkpoint"]).resolve() == checkpoint
        assert config["task_config"] == "demo_clean"
        assert config["instruction_type"] == "unseen"
        assert config["action_chunk_steps"] == 50
        assert config["repair_action_chunk_steps"] == 50
        assert config["vlm_image_detail"] == "low"
        assert config["dag_from_start"] is True

    cfn = configs["cfn"]
    assert set(cfn["tasks"]) == {
        "handover_block",
        "move_can_pot",
        "beat_block_hammer",
    }
    assert set(cfn["repair_cfn_tasks"]) == set(cfn["tasks"])
    assert cfn["use_registered_dag_prompts"] is False
    assert cfn["preserve_original_repair_prompt"] is False
    assert cfn["stage_stall_observations_by_task"] == {
        "handover_block": [4, 4, 2],
        "move_can_pot": [4],
        "beat_block_hammer": [4, 2],
    }
    assert cfn["min_base_steps_by_task"] == {
        "handover_block": 0,
        "move_can_pot": 0,
        "beat_block_hammer": 0,
    }

    registered = configs["registered"]
    assert set(registered["tasks"]) == {"turn_switch"}
    assert registered["use_registered_dag_prompts"] is True
    assert registered["preserve_original_repair_prompt"] is False
    assert registered["stage_stall_observations_by_task"] == {"turn_switch": [4]}
    assert registered["min_base_steps_by_task"] == {"turn_switch": 0}

    original = configs["original"]
    assert set(original["tasks"]) == {
        "blocks_ranking_size",
        "place_dual_shoes",
        "stamp_seal",
    }
    assert original["use_registered_dag_prompts"] is False
    assert original["preserve_original_repair_prompt"] is True
    assert original["stage_stall_observations_by_task"] == {
        "blocks_ranking_size": [4, 3, 2],
        "place_dual_shoes": [3, 2],
        "stamp_seal": [3, 2],
    }
    assert original["min_base_steps_by_task"] == {
        "blocks_ranking_size": 700,
        "place_dual_shoes": 250,
        "stamp_seal": 150,
    }

    replan = configs["replan"]
    assert set(replan["tasks"]) == {"open_microwave", "stack_blocks_three"}
    assert replan["use_registered_dag_prompts"] is False
    assert replan["preserve_original_repair_prompt"] is True
    assert replan["policy_replan_steps"] == 10
    assert replan["stage_stall_observations_by_task"] == {
        "open_microwave": [4, 3],
        "stack_blocks_three": [4, 3, 2],
    }
    assert replan["min_base_steps_by_task"] == {
        "open_microwave": 400,
        "stack_blocks_three": 650,
    }
