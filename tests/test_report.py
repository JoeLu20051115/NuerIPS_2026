from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pi05_libero_repro.records import EpisodeRecord
from pi05_libero_repro.report import build_report, render_markdown


def record(checkpoint: str, task_id: int, episode_idx: int, success: bool) -> EpisodeRecord:
    return EpisodeRecord(
        checkpoint=checkpoint,
        task_id=task_id,
        task_name=f"official task {task_id}",
        episode_idx=episode_idx,
        init_state_sha256=f"{task_id:x}" * 64,
        seed=7,
        success=success,
        valid=True,
        steps=10,
        inference_requests=2,
        wall_seconds=1.0,
        exception=None,
        first_frame_sha256=f"{episode_idx % 16:x}" * 64,
        action_min=-1.0,
        action_max=1.0,
        action_mean=0.0,
        done=success,
        check_success=success,
        video_path=f"videos/{checkpoint}-{task_id}-{episode_idx}.mp4",
    )


def records_for(checkpoint: str, successes: int) -> list[EpisodeRecord]:
    return [
        record(checkpoint, task_id, episode_idx, task_id * 50 + episode_idx < successes)
        for task_id in range(10)
        for episode_idx in range(50)
    ]


def test_public_rates_cardinality_and_display_order() -> None:
    report = build_report(records_for("full", 462) + records_for("early", 215))

    assert report["accepted"] is True
    assert report["checkpoints"]["full"]["rate"] == 0.924
    assert report["checkpoints"]["early"]["rate"] == 0.43
    assert all(row["full"]["trials"] == row["early"]["trials"] == 50 for row in report["tasks"])
    assert [row["task_id"] for row in report["tasks"]] == [1, 3, 2, 4, 8, 7, 9, 0, 5, 6]

    markdown = render_markdown(report)
    assert markdown.index("Cream cheese + Butter") < markdown.index("Black bowl")
    assert markdown.rindex("Chocolate pudding") > markdown.index("Book")
    assert "92.4%" in markdown and "43.0%" in markdown


@pytest.mark.parametrize("checkpoint,successes", [("full", 440), ("early", 245)])
def test_out_of_range_rate_is_rejected(checkpoint: str, successes: int) -> None:
    full_successes = successes if checkpoint == "full" else 462
    early_successes = successes if checkpoint == "early" else 215
    report = build_report(records_for("full", full_successes) + records_for("early", early_successes))

    assert report["accepted"] is False
    assert any(error["code"] == "rate_out_of_range" for error in report["errors"])


def test_cardinality_duplicate_invalid_and_predicate_errors() -> None:
    base = records_for("full", 462) + records_for("early", 215)
    cases = [
        (base[:-1], "count_mismatch"),
        (base + [base[0]], "duplicate_episode"),
        ([replace(base[0], valid=False)] + base[1:], "invalid_episode"),
        ([replace(base[0], check_success=False)] + base[1:], "predicate_mismatch"),
    ]

    for broken, expected_code in cases:
        report = build_report(broken)
        assert report["accepted"] is False
        assert any(error["code"] == expected_code for error in report["errors"])


def test_seed_episode_index_and_hash_integrity_are_required() -> None:
    base = records_for("full", 462) + records_for("early", 215)
    cases = [
        ([replace(base[0], seed=8)] + base[1:], "seed_mismatch"),
        ([replace(base[0], episode_idx=50)] + base[1:], "episode_index_mismatch"),
        ([replace(base[0], init_state_sha256="bad")] + base[1:], "invalid_hash"),
    ]

    for broken, expected_code in cases:
        report = build_report(broken)
        assert report["accepted"] is False
        assert any(error["code"] == expected_code for error in report["errors"])


def test_report_cli_writes_outputs_and_returns_acceptance(tmp_path: Path) -> None:
    full_path = tmp_path / "full.jsonl"
    early_path = tmp_path / "early.jsonl"
    full_path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in records_for("full", 462))
    )
    early_path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in records_for("early", 215))
    )
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    script = Path(__file__).parents[1] / "scripts" / "report_results.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full",
            str(full_path),
            "--early",
            str(early_path),
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(json_path.read_text())["accepted"] is True
    assert "Acceptance: **PASS**" in markdown_path.read_text()
