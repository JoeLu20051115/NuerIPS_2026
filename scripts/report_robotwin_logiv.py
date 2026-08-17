#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


SUCCESS_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TERNARY_VALUES = {"TRUE", "FALSE", "UNRESOLVED"}
CURRENT_CAMERAS = {
    "current/head_camera",
    "current/right_camera",
    "current/left_camera",
}


def parse_baseline_log(path: Path) -> dict[int, bool]:
    text = ANSI_RE.sub(
        "", path.read_text(encoding="utf-8", errors="replace")
    ).replace("\r", "\n")
    result: dict[int, bool] = {}
    previous = 0
    for successes, _total, seed in SUCCESS_RE.findall(text):
        cumulative = int(successes)
        result[int(seed)] = cumulative > previous
        previous = cumulative
    return result


def _event_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("logiv_events.jsonl")):
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def build_report(
    config: dict[str, Any], events_root: Path, baseline_logs: Path
) -> dict[str, Any]:
    expected = {
        (task, int(seed)): config["instructions"][task][index]
        for task, seeds in config["tasks"].items()
        for index, seed in enumerate(seeds)
    }
    baseline = {
        (task, seed): success
        for task in config["tasks"]
        for seed, success in parse_baseline_log(baseline_logs / f"{task}.log").items()
    }
    records = _event_records(events_root)
    actual: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        key = (str(record.get("task")), int(record.get("seed")))
        if key in actual:
            errors.append(f"duplicate LOGIV record: {key}")
            continue
        actual[key] = record
        if key not in expected:
            errors.append(f"unexpected LOGIV record: {key}")
            continue
        if record.get("original_instruction") != expected[key]:
            errors.append(f"instruction mismatch: {key}")
        if not record.get("events"):
            errors.append(f"missing monitor events: {key}")
        elif not all(event.get("val_valid") is True for event in record["events"]):
            errors.append(f"VAL-invalid plan: {key}")
        request_count = int(record.get("gpt4o_requests", 0))
        if request_count < 1:
            errors.append(f"missing GPT-4o observation: {key}")
        calls = record.get("gpt4o_calls")
        if not isinstance(calls, list) or len(calls) != request_count or any(
            not isinstance(call, dict)
            or call.get("purpose") != "state_gate"
            or not str(call.get("model", "")).startswith("gpt-4o")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(call.get("request_sha256", ""))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(call.get("response_sha256", ""))
            )
            for call in calls or []
        ):
            errors.append(f"invalid GPT-4o provenance: {key}")
        event_values = {
            value
            for event in record.get("events", [])
            if isinstance(event, dict)
            for value in event.get("facts", {}).values()
        }
        if not event_values or not event_values <= TERNARY_VALUES:
            errors.append(f"non-ternary State Gate fact: {key}")
        audits = record.get("vlm_audit")
        if not isinstance(audits, list) or len(audits) != request_count:
            errors.append(f"missing or incomplete camera audit: {key}")
        elif any(
            not isinstance(audit, dict)
            or not isinstance(audit.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("sha256", "")))
            or not CURRENT_CAMERAS <= set(audit.get("camera_order", []))
            for audit in audits
        ):
            errors.append(f"invalid camera audit: {key}")
    missing = sorted(expected.keys() - actual.keys())
    errors.extend(f"missing LOGIV record: {key}" for key in missing)
    if set(baseline) != set(expected):
        errors.append("baseline seed set does not match frozen protocol")

    paired = sorted(set(actual) & set(expected) & set(baseline))
    successes = sum(bool(actual[key].get("success")) for key in paired)
    positive = sum(not baseline[key] and actual[key].get("success") for key in paired)
    negative = sum(baseline[key] and not actual[key].get("success") for key in paired)
    by_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"completed": 0, "successes": 0, "baseline_successes": 0}
    )
    for key in paired:
        task, _seed = key
        by_task[task]["completed"] += 1
        by_task[task]["successes"] += int(bool(actual[key].get("success")))
        by_task[task]["baseline_successes"] += int(baseline[key])
    return {
        "evidence_label": "development/seed-selected",
        "expected": len(expected),
        "completed": len(paired),
        "strict_protocol_complete": len(paired) == len(expected) and not errors,
        "successes": successes,
        "success_rate": successes / len(paired) if paired else None,
        "baseline_successes": sum(baseline[key] for key in paired),
        "positive_flips": positive,
        "negative_flips": negative,
        "per_task": dict(by_task),
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rate = report["success_rate"]
    rate_text = "n/a" if rate is None else f"{100 * rate:.1f}%"
    lines = [
        "# RoboTwin 2.0 LOGIV + PDDL — 10 tasks × 10 episodes",
        "",
        "- Evidence label: **development/seed-selected (not an independent holdout)**",
        f"- Overall: **{report['successes']}/{report['completed']} = {rate_text}**",
        f"- Baseline on paired scenes: **{report['baseline_successes']}/{report['completed']}**",
        f"- Flips: **+{report['positive_flips']} / -{report['negative_flips']}**",
        f"- Protocol completeness: **{report['completed']}/{report['expected']}**",
        f"- Strict protocol audit: **{'PASS' if report['strict_protocol_complete'] else 'FAIL'}**",
        "",
        "| Task | LOGIV | Baseline |",
        "|---|---:|---:|",
    ]
    for task, row in report["per_task"].items():
        lines.append(
            f"| `{task}` | {row['successes']}/{row['completed']} | "
            f"{row['baseline_successes']}/{row['completed']} |"
        )
    if report["errors"]:
        lines.extend(("", "## Audit errors", ""))
        lines.extend(f"- {error}" for error in report["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--events-root", required=True, type=Path)
    parser.add_argument("--baseline-logs", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.config.read_text()), args.events_root, args.baseline_logs
    )
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
