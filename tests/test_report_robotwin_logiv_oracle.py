from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "report_robotwin_logiv_oracle",
    Path("scripts/report_robotwin_logiv_oracle.py").resolve(),
)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def test_oracle_selects_one_monitored_success_per_seed(tmp_path) -> None:
    seeds = list(range(100001, 100011))
    instructions = ["first", "second", *[f"instruction-{i}" for i in range(8)]]
    config = {
        "tasks": {"task_a": seeds},
        "instructions": {"task_a": instructions},
    }
    events = tmp_path / "events"
    events.mkdir()
    records = [
        {
            "task": "task_a",
            "seed": 100001,
            "success": True,
            "original_instruction": "first",
            "gpt4o_requests": 2,
            "events": [
                {
                    "epoch": 0,
                    "control_mode": "DAG_EXECUTION",
                    "val_valid": True,
                    "certificate_sha256": "a" * 64,
                }
            ],
        },
        {
            "task": "task_a",
            "seed": 100001,
            "success": True,
            "original_instruction": "changed prompt",
            "gpt4o_requests": 2,
            "events": [
                {"control_mode": "DAG_EXECUTION", "val_valid": True}
            ],
        },
        {
            "task": "task_a",
            "seed": 100002,
            "success": True,
            "original_instruction": "selected historical instruction",
            "gpt4o_requests": 2,
            "events": [
                {
                    "epoch": 0,
                    "control_mode": "BASE_MONITORED",
                    "val_valid": True,
                    "certificate_sha256": "b" * 64,
                }
            ],
        },
    ]
    (events / "logiv_events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = REPORT.build_oracle_report(config, [events])

    assert report["expected"] == 10
    assert report["successes"] == 2
    assert report["success_rate"] == 0.2
    assert report["per_task"]["task_a"]["successes"] == 2
    assert report["selected"][0]["task"] == "task_a"
    assert report["selected"][0]["seed"] == 100001
    assert report["selected"][0]["source_event_file"].endswith(
        "logiv_events.jsonl"
    )
    assert report["selected"][0]["source_event_file"] == "logiv_events.jsonl"
    assert not report["selected"][0]["source_event_file"].startswith("/")
    assert report["selected"][0]["source_record_sha256"]
    selected = {(row["task"], row["seed"]): row for row in report["selected"]}
    assert selected[("task_a", 100002)]["instruction"] == (
        "selected historical instruction"
    )
    assert len(report["missing"]) == 8


def test_oracle_report_can_be_reaudited_from_embedded_records(tmp_path) -> None:
    seeds = list(range(100001, 100011))
    config = {
        "tasks": {"task_a": seeds},
        "instructions": {"task_a": [f"instruction-{i}" for i in range(10)]},
    }
    events = tmp_path / "events"
    events.mkdir()
    record = {
        "task": "task_a",
        "seed": 100001,
        "success": True,
        "original_instruction": "chosen",
        "gpt4o_requests": 1,
        "events": [
            {
                "epoch": 0,
                "control_mode": "BASE_MONITORED",
                "val_valid": True,
                "certificate_sha256": "c" * 64,
            }
        ],
    }
    (events / "logiv_events.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    report = REPORT.build_oracle_report(config, [events], embed_records=True)

    assert REPORT.audit_embedded_report(report) == []
    assert report["protocol"]["tasks"] == config["tasks"]
    assert report["protocol"]["canonical_config_sha256"] == (
        REPORT._canonical_sha256(config)
    )
    assert report["selected"][0]["embedded_record_sha256"]
    report["selected"][0]["record"]["events"][0]["val_valid"] = False
    assert "selected record is not compliant" in REPORT.audit_embedded_report(
        report
    )[0]


def test_embedded_events_can_be_replanned_and_revalidated_with_real_val(
    tmp_path,
) -> None:
    seeds = list(range(100001, 100011))
    config = {
        "checkpoint": "/checkpoint",
        "task_config": "demo_clean",
        "instruction_type": "unseen",
        "tasks": {"task_a": seeds},
        "instructions": {"task_a": [f"instruction-{i}" for i in range(10)]},
    }
    events = tmp_path / "events"
    events.mkdir()
    record = {
        "task": "task_a",
        "seed": 100001,
        "success": True,
        "original_instruction": "instruction-0",
        "gpt4o_requests": 1,
        "events": [
            {
                "epoch": 0,
                "control_mode": "DAG_EXECUTION",
                "facts": {"first-fact": "FALSE", "goal-fact": "UNKNOWN"},
                "val_valid": True,
                "certificate_sha256": "c" * 64,
            }
        ],
    }
    (events / "logiv_events.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    report = REPORT.build_oracle_report(config, [events], embed_records=True)

    report["val_revalidation"] = REPORT.revalidate_embedded_report(
        report, Path("artifacts/tools/val-ubuntu22/Validate").resolve()
    )

    audit = report["val_revalidation"]
    assert audit["event_occurrences"] == 1
    assert audit["valid_event_occurrences"] == 1
    assert audit["unique_states"] == 1
    assert audit["states"][0]["val_valid"] is True
    assert "(define (domain logiv-task_a)" in audit["states"][0]["domain_pddl"]
    assert "Plan valid" in audit["states"][0]["val_stdout"]
    assert "/tmp/" not in audit["states"][0]["val_stdout"]
    assert REPORT.audit_embedded_report(report) == []

    audit["valid_event_occurrences"] = 0
    assert "VAL revalidation count mismatch" in REPORT.audit_embedded_report(report)

    audit["valid_event_occurrences"] = 1
    audit["states"][0]["plan_pddl"] = "(tampered)\n"
    assert "VAL revalidation count mismatch" in REPORT.audit_embedded_report(report)


def test_oracle_can_require_the_frozen_baseline_instruction(tmp_path) -> None:
    seeds = list(range(100001, 100011))
    config = {
        "tasks": {"task_a": seeds},
        "instructions": {"task_a": [f"baseline-{seed}" for seed in seeds]},
    }
    events = tmp_path / "events"
    events.mkdir()
    common = {
        "task": "task_a",
        "seed": 100001,
        "success": True,
        "gpt4o_requests": 1,
        "events": [
            {
                "epoch": 0,
                "control_mode": "DAG_EXECUTION",
                "val_valid": True,
                "certificate_sha256": "e" * 64,
            }
        ],
    }
    records = [
        {**common, "original_instruction": "historical synonym", "actions": 10},
        {**common, "original_instruction": "baseline-100001", "actions": 20},
        {
            **common,
            "seed": 100002,
            "original_instruction": "historical synonym only",
        },
    ]
    (events / "logiv_events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    permissive = REPORT.build_oracle_report(config, [events])
    strict = REPORT.build_oracle_report(
        config,
        [events],
        embed_records=True,
        require_baseline_instruction=True,
    )

    assert permissive["successes"] == 2
    assert permissive["baseline_instruction_matches"] == 0
    assert strict["successes"] == 1
    assert strict["baseline_instruction_matches"] == 1
    assert strict["selected"][0]["instruction"] == "baseline-100001"
    assert strict["require_baseline_instruction"] is True
    assert REPORT.audit_embedded_report(strict) == []
    strict["selected"][0]["record"]["original_instruction"] = "changed"
    strict["selected"][0]["embedded_record_sha256"] = REPORT._canonical_sha256(
        strict["selected"][0]["record"]
    )
    errors = REPORT.audit_embedded_report(strict)
    assert any("mismatch" in error for error in errors)


def test_embedded_audit_rejects_duplicate_seed_and_inconsistent_rate(tmp_path) -> None:
    seeds = list(range(100001, 100011))
    config = {
        "tasks": {"task_a": seeds},
        "instructions": {"task_a": [f"instruction-{i}" for i in range(10)]},
    }
    events = tmp_path / "events" / "run-a"
    events.mkdir(parents=True)
    record = {
        "task": "task_a",
        "seed": 100001,
        "success": True,
        "original_instruction": "chosen",
        "gpt4o_requests": 1,
        "events": [
            {
                "epoch": 0,
                "control_mode": "DAG_EXECUTION",
                "val_valid": True,
                "certificate_sha256": "d" * 64,
            }
        ],
    }
    (events / "logiv_events.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    report = REPORT.build_oracle_report(
        config, [tmp_path / "events"], embed_records=True
    )
    assert report["selected"][0]["source_event_file"] == (
        "run-a/logiv_events.jsonl"
    )

    report["selected"].append(dict(report["selected"][0]))
    report["successes"] = 2
    report["success_rate"] = 0.9

    errors = REPORT.audit_embedded_report(report)
    assert "duplicate selected cell" in errors
    assert "success rate does not match selected records" in errors


def test_oracle_rejects_non_ten_seed_task_protocol() -> None:
    config = {
        "tasks": {"task_a": [100001, 100001]},
        "instructions": {"task_a": ["first", "first"]},
    }

    try:
        REPORT.build_oracle_report(config, [])
    except ValueError as error:
        assert "10 distinct seeds" in str(error)
    else:
        raise AssertionError("invalid seed protocol was accepted")


def test_compliance_requires_explicit_epoch_zero_and_logiv_control_mode() -> None:
    valid = {
        "success": True,
        "original_instruction": "instruction",
        "gpt4o_requests": 1,
        "events": [
            {
                "epoch": 0,
                "control_mode": "DAG_EXECUTION",
                "val_valid": True,
                "certificate_sha256": "f" * 64,
            }
        ],
    }

    assert REPORT._is_compliant_success(valid)
    assert not REPORT._is_compliant_success(
        {**valid, "events": [{**valid["events"][0], "epoch": 1}]}
    )
    assert not REPORT._is_compliant_success(
        {
            **valid,
            "events": [
                {
                    key: value
                    for key, value in valid["events"][0].items()
                    if key != "control_mode"
                }
            ],
        }
    )
