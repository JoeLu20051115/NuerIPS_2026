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
            "gpt4o_calls": [
                {
                    "purpose": "state_gate",
                    "model": "gpt-4o-2024-08-06",
                    "request_sha256": "c" * 64,
                    "response_sha256": "d" * 64,
                }
                for _ in range(3)
            ],
            "events": [
                {"val_valid": True, "facts": {"fact": "UNRESOLVED"}}
                for _ in range(3)
            ],
            "vlm_audit": [
                {
                    "path": f"vlm_audit/first-{epoch}.png",
                    "sha256": "a" * 64,
                    "camera_order": [
                        "current/head_camera",
                        "current/right_camera",
                        "current/left_camera",
                    ],
                }
                for epoch in range(3)
            ],
        },
        {
            "task": "task_a",
            "seed": 100002,
            "success": False,
            "original_instruction": "second",
            "gpt4o_requests": 2,
            "gpt4o_calls": [
                {
                    "purpose": "state_gate",
                    "model": "gpt-4o-2024-08-06",
                    "request_sha256": "e" * 64,
                    "response_sha256": "f" * 64,
                }
                for _ in range(2)
            ],
            "events": [
                {"val_valid": True, "facts": {"fact": "FALSE"}}
                for _ in range(2)
            ],
            "vlm_audit": [
                {
                    "path": f"vlm_audit/second-{epoch}.png",
                    "sha256": "b" * 64,
                    "camera_order": [
                        "current/head_camera",
                        "current/right_camera",
                        "current/left_camera",
                    ],
                }
                for epoch in range(2)
            ],
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
    assert report["evidence_label"] == "development/seed-selected"
    assert report["strict_protocol_complete"] is True
    assert "not an independent holdout" in REPORT.render_markdown(report)


def test_report_rejects_missing_camera_audit_or_nonternary_fact(tmp_path) -> None:
    config = {
        "tasks": {"task_a": [100001]},
        "instructions": {"task_a": ["instruction"]},
    }
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "task_a.log").write_text(
        "Success rate: 0/1 => 0.0%, current seed: 100001\n",
        encoding="utf-8",
    )
    events = tmp_path / "events"
    events.mkdir()
    (events / "logiv_events.jsonl").write_text(
        json.dumps(
            {
                "task": "task_a",
                "seed": 100001,
                "success": False,
                "original_instruction": "instruction",
                "gpt4o_requests": 1,
                "gpt4o_calls": [
                    {
                        "purpose": "repair",
                        "model": "not-gpt-4o",
                        "request_sha256": "bad",
                        "response_sha256": "bad",
                    }
                ],
                "events": [
                    {"val_valid": True, "facts": {"fact": "UNKNOWN"}}
                ],
                "vlm_audit": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = REPORT.build_report(config, events, baseline)

    assert any("camera audit" in error for error in report["errors"])
    assert any("non-ternary" in error for error in report["errors"])
    assert any("GPT-4o provenance" in error for error in report["errors"])


def test_seed_selected_report_can_audit_native_score_without_baseline(tmp_path) -> None:
    config = {
        "tasks": {"task_a": [100001]},
        "instructions": {"task_a": ["instruction"]},
    }
    events = tmp_path / "events"
    events.mkdir()
    record = {
        "task": "task_a",
        "seed": 100001,
        "success": True,
        "original_instruction": "instruction",
        "gpt4o_requests": 1,
        "gpt4o_calls": [
            {
                "purpose": "state_gate",
                "model": "gpt-4o-2024-08-06",
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
            }
        ],
        "events": [{"val_valid": True, "facts": {"fact": "FALSE"}}],
        "vlm_audit": [
            {
                "path": "vlm_audit/frame.png",
                "sha256": "c" * 64,
                "camera_order": [
                    "current/head_camera",
                    "current/right_camera",
                    "current/left_camera",
                ],
            }
        ],
    }
    (events / "logiv_events.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    report = REPORT.build_report(config, events, None)

    assert report["successes"] == 1
    assert report["completed"] == 1
    assert report["baseline_successes"] is None
    assert report["strict_protocol_complete"] is True
    assert "Baseline: **not run**" in REPORT.render_markdown(report)
