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


def _event_records(
    root: Path,
) -> tuple[list[tuple[dict[str, Any], Path]], list[str]]:
    records: list[tuple[dict[str, Any], Path]] = []
    errors: list[str] = []
    for path in sorted(root.rglob("logiv_events.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(
                        f"malformed JSON event: {path}:{line_number}"
                    )
                    continue
                records.append((record, path.parent))
    return records, errors


def build_report(
    config: dict[str, Any], events_root: Path, baseline_logs: Path | None
) -> dict[str, Any]:
    expected = {
        (task, int(seed)): config["instructions"][task][index]
        for task, seeds in config["tasks"].items()
        for index, seed in enumerate(seeds)
    }
    baseline = (
        {
            (task, seed): success
            for task in config["tasks"]
            for seed, success in parse_baseline_log(
                baseline_logs / f"{task}.log"
            ).items()
        }
        if baseline_logs is not None
        else None
    )
    records, errors = _event_records(events_root)
    actual: dict[tuple[str, int], dict[str, Any]] = {}
    for record, _episode_root in records:
        try:
            task = record["task"]
            seed = int(record["seed"])
            if not isinstance(task, str) or not task:
                raise ValueError
            key = (task, seed)
        except (KeyError, TypeError, ValueError):
            errors.append(f"malformed LOGIV record: {record!r}")
            continue
        if key in actual:
            errors.append(f"duplicate LOGIV record: {key}")
            continue
        actual[key] = record
        if key not in expected:
            errors.append(f"unexpected LOGIV record: {key}")
            continue
        if record.get("original_instruction") != expected[key]:
            errors.append(f"instruction mismatch: {key}")
    missing = sorted(expected.keys() - actual.keys())
    errors.extend(f"missing LOGIV record: {key}" for key in missing)
    if baseline is not None and set(baseline) != set(expected):
        errors.append("baseline seed set does not match frozen protocol")

    paired = sorted(set(actual) & set(expected))
    if baseline is not None:
        paired = sorted(set(paired) & set(baseline))
    successes = sum(bool(actual[key].get("success")) for key in paired)
    positive = (
        sum(not baseline[key] and actual[key].get("success") for key in paired)
        if baseline is not None
        else None
    )
    negative = (
        sum(baseline[key] and not actual[key].get("success") for key in paired)
        if baseline is not None
        else None
    )
    by_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"completed": 0, "successes": 0, "baseline_successes": 0}
    )
    for key in paired:
        task, _seed = key
        by_task[task]["completed"] += 1
        by_task[task]["successes"] += int(bool(actual[key].get("success")))
        if baseline is not None:
            by_task[task]["baseline_successes"] += int(baseline[key])
    return {
        "evidence_label": config.get(
            "evidence_label", "development/frozen-rerun"
        ),
        "expected": len(expected),
        "completed": len(paired),
        "strict_protocol_complete": len(paired) == len(expected) and not errors,
        "successes": successes,
        "success_rate": successes / len(paired) if paired else None,
        "baseline_successes": (
            sum(baseline[key] for key in paired) if baseline is not None else None
        ),
        "positive_flips": positive,
        "negative_flips": negative,
        "per_task": dict(by_task),
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rate = report["success_rate"]
    rate_text = "n/a" if rate is None else f"{100 * rate:.1f}%"
    lines = [
        "# RoboTwin 2.0 LOGIV + PDDL — frozen rerun",
        "",
        f"- Evidence label: **{report['evidence_label']}**",
        f"- Overall: **{report['successes']}/{report['completed']} = {rate_text}**",
    ]
    if report["baseline_successes"] is None:
        lines.append("- Baseline: **not run**")
    else:
        lines.extend(
            (
                f"- Baseline on paired scenes: **{report['baseline_successes']}/{report['completed']}**",
                f"- Flips: **+{report['positive_flips']} / -{report['negative_flips']}**",
            )
        )
    lines.extend(
        (
            f"- Protocol completeness: **{report['completed']}/{report['expected']}**",
            f"- Frozen protocol check: **{'PASS' if report['strict_protocol_complete'] else 'FAIL'}**",
            "",
            "| Task | LOGIV | Baseline |",
            "|---|---:|---:|",
        )
    )
    for task, row in report["per_task"].items():
        baseline_text = (
            "n/a"
            if report["baseline_successes"] is None
            else f"{row['baseline_successes']}/{row['completed']}"
        )
        lines.append(
            f"| `{task}` | {row['successes']}/{row['completed']} | "
            f"{baseline_text} |"
        )
    if report["errors"]:
        lines.extend(("", "## Report errors", ""))
        lines.extend(f"- {error}" for error in report["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--events-root", required=True, type=Path)
    parser.add_argument("--baseline-logs", type=Path)
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
