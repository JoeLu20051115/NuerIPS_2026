import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/robotwin/logiv-gpt4o-63-vs-pi05-56.json"
RESULT = ROOT / "results/robotwin-logiv-63-vs-pi05-56.json"
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
CONTROL_KEYS = (
    "task_config",
    "instruction_type",
    "action_chunk_steps",
    "repair_action_chunk_steps",
    "repair_action_chunk_steps_by_task",
    "vlm_image_detail",
    "dag_from_start",
    "use_registered_dag_prompts",
    "use_registered_dag_prompts_by_task",
    "preserve_original_repair_prompt",
    "preserve_original_repair_prompt_by_task",
    "policy_replan_steps_by_task",
    "repair_cfn_tasks",
    "min_base_steps_by_task",
    "base_stall_observations",
    "base_stall_observations_by_task",
    "stage_stall_observations_by_task",
)


def _digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_manifest_is_the_exact_portable_ten_by_ten_protocol() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert set(manifest["tasks"]) == TASKS
    assert set(manifest["instructions"]) == TASKS
    assert all(len(seeds) == 10 for seeds in manifest["tasks"].values())
    assert all(len(set(seeds)) == 10 for seeds in manifest["tasks"].values())
    assert all(
        len(manifest["instructions"][task]) == len(seeds)
        for task, seeds in manifest["tasks"].items()
    )
    assert _digest(
        {key: manifest[key] for key in ("tasks", "instructions")}
    ) == "415bdcb12c81ff3a8cac40fe435214c6f877bf5c1c1477841efd8741c47ec60f"
    assert _digest(
        {key: manifest[key] for key in CONTROL_KEYS}
    ) == "ce959f62853bf2bd1c11aa8fcff0b4fac4ac249663405fe6b6e1c27a4221306f"
    assert manifest["evidence_label"] == "development/frozen-rerun"
    assert manifest["expected_successes"] == {"logiv": 63, "baseline": 56}
    assert "checkpoint" not in manifest
    assert "seed_selection" not in manifest


def test_compact_result_contains_counts_but_no_episode_evidence() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["evidence_label"] == "development/frozen-rerun"
    assert result["completed"] == 100
    assert result["successes"] == 63
    assert result["baseline_successes"] == 56
    assert result["positive_flips"] == 17
    assert result["negative_flips"] == 10
    assert set(result["per_task"]) == TASKS
    assert all(row["completed"] == 10 for row in result["per_task"].values())
    assert not {
        "selected",
        "records",
        "events",
        "val_certificates",
        "gpt4o_calls",
        "vlm_audit",
    } & set(result)
