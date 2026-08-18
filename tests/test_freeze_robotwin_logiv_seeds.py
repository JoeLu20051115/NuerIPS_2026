from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "freeze_robotwin_logiv_seeds",
    Path("scripts/freeze_robotwin_logiv_seeds.py").resolve(),
)
FREEZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FREEZER)


def _record(task: str, seed: int, *, success: bool, actions: int) -> dict:
    return {
        "task": task,
        "seed": seed,
        "success": success,
        "actions": actions,
        "gpt4o_requests": 2,
        "original_instruction": f"instruction-{task}-{seed}",
        "vlm_audit": [
            {
                "epoch": 0,
                "path": f"vlm_audit/{task}-seed{seed}-epoch-000.png",
                "camera_order": [
                    "current/head_camera",
                    "current/right_camera",
                    "current/left_camera",
                ],
                "sha256": "a" * 64,
            }
        ],
        "events": [{"val_valid": True}],
    }


def test_freezer_ranks_success_then_actions_and_freezes_ten_per_task() -> None:
    tasks = ("task_a", "task_b")
    records = []
    for task in tasks:
        records.extend(
            _record(
                task,
                seed,
                success=seed % 3 != 0,
                actions=200000 - seed,
            )
            for seed in range(100000, 100020)
        )
    template = {
        "task_names": list(tasks),
        "checkpoint": "/checkpoint",
        "action_chunk_steps": 50,
    }

    frozen = FREEZER.freeze_seed_manifest(template, records)

    assert frozen["evidence_label"] == "development/seed-selected"
    assert set(frozen["tasks"]) == set(tasks)
    for task in tasks:
        assert len(frozen["tasks"][task]) == 10
        assert len(set(frozen["tasks"][task])) == 10
        assert all(seed % 3 != 0 for seed in frozen["tasks"][task])
        assert frozen["instructions"][task] == [
            f"instruction-{task}-{seed}" for seed in frozen["tasks"][task]
        ]
        assert frozen["seed_selection"]["per_task"][task]["candidate_count"] == 20
        assert frozen["seed_selection"]["per_task"][task]["selected_count"] == 10


def test_freezer_rejects_short_duplicate_or_uncertified_candidate_pools() -> None:
    template = {"task_names": ["task_a"]}
    short = [_record("task_a", seed, success=True, actions=1) for seed in range(19)]
    with pytest.raises(ValueError, match="20..30"):
        FREEZER.freeze_seed_manifest(template, short)

    complete = [
        _record("task_a", seed, success=True, actions=1) for seed in range(20)
    ]
    with pytest.raises(ValueError, match="duplicate"):
        FREEZER.freeze_seed_manifest(template, complete + [complete[0]])

    uncertified = [dict(record) for record in complete]
    uncertified[0] = {**uncertified[0], "events": [{"val_valid": False}]}
    with pytest.raises(ValueError, match="VAL"):
        FREEZER.freeze_seed_manifest(template, uncertified)
