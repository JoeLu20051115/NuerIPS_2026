import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robotwin/logiv-gpt4o-seed-scan-v1-template.json"
TASKS = {
    "handover_block",
    "open_microwave",
    "place_dual_shoes",
    "stamp_seal",
    "blocks_ranking_size",
    "move_can_pot",
    "turn_switch",
    "stack_blocks_three",
    "stack_bowls_three",
    "beat_block_hammer",
}


def test_seed_scan_template_covers_ten_tasks_and_task_specific_repairs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert set(config["task_names"]) == TASKS
    assert config["evidence_label"] == "development/seed-scan"
    assert config["dag_from_start"] is True
    assert config["vlm_image_detail"] == "low"
    assert set(config["repair_cfn_tasks"]) == {
        "handover_block",
        "move_can_pot",
        "beat_block_hammer",
    }
    assert config["use_registered_dag_prompts_by_task"] == {
        "turn_switch": True
    }
    assert config["policy_replan_steps_by_task"] == {
        "open_microwave": 10,
        "stack_blocks_three": 10,
    }
    for mapping_name in (
        "repair_action_chunk_steps_by_task",
        "min_base_steps_by_task",
        "base_stall_observations_by_task",
        "stage_stall_observations_by_task",
    ):
        assert set(config[mapping_name]) == TASKS
