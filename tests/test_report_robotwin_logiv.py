from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "report_robotwin_logiv",
    Path("scripts/report_robotwin_logiv.py").resolve(),
)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def test_baseline_parser_derives_per_seed_success_from_cumulative_count(tmp_path) -> None:
    log = tmp_path / "task.log"
    log.write_text(
        "Success rate: \x1b[96m0/1\x1b[0m => 0.0%, current seed: "
        "\x1b[90m100001\x1b[0m\n"
        "Success rate: 1/2 => 50.0%, current seed: 100002\n"
        "Success rate: 1/3 => 33.3%, current seed: 100003\n",
        encoding="utf-8",
    )

    assert REPORT.parse_baseline_log(log) == {
        100001: False,
        100002: True,
        100003: False,
    }


def test_report_audits_fixed_pairs_and_counts_flips(tmp_path) -> None:
    config = {
        "checkpoint": "/checkpoint",
        "tasks": {"task_a": [100001, 100002]},
        "instructions": {"task_a": ["first", "second"]},
    }
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "task_a.log").write_text(
        "Success rate: 0/1 => 0.0%, current seed: 100001\n"
        "Success rate: 1/2 => 50.0%, current seed: 100002\n",
        encoding="utf-8",
    )
    events = tmp_path / "events" / "nested"
    events.mkdir(parents=True)
    records = [
        {
            "task": "task_a",
            "seed": 100001,
            "success": True,
            "original_instruction": "first",
            "gpt4o_requests": 3,
            "events": [{"val_valid": True}],
        },
        {
            "task": "task_a",
            "seed": 100002,
            "success": False,
            "original_instruction": "second",
            "gpt4o_requests": 2,
            "events": [{"val_valid": True}],
        },
    ]
    (events / "logiv_events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = REPORT.build_report(config, tmp_path / "events", baseline)

    assert report["completed"] == 2
    assert report["successes"] == 1
    assert report["positive_flips"] == 1
    assert report["negative_flips"] == 1
    assert report["errors"] == []
    assert report["evidence_label"] == "development/tuning"
    assert report["strict_protocol_complete"] is True
    assert "not an independent holdout" in REPORT.render_markdown(report)
