#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from pi05_libero_repro.logiv.records import (
    LogivEpisodeRecord,
    load_episode_records,
    paired_task_stratified_bootstrap,
    validate_episode_records,
)
from pi05_libero_repro.records import load_records, wilson_interval


def _stats(records) -> dict:
    records = list(records)
    successes = sum(bool(item.success) for item in records)
    total = len(records)
    return {
        "successes": successes,
        "allocated": total,
        "rate": successes / total if total else None,
        "wilson_95": list(wilson_interval(successes, total)) if total else [None, None],
    }


def _baseline_proxy(records) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            task_id=item.task_id,
            episode_idx=item.episode_idx,
            success=item.success,
            init_state_sha256=item.init_state_sha256,
        )
        for item in records
    ]


def load_comparator_records(path: Path):
    """Load either the original π0.5 record schema or the LOGIV BASE schema."""

    with Path(path).open(encoding="utf-8") as stream:
        first = next((json.loads(line) for line in stream if line.strip()), None)
    if first is None:
        return []
    if "method_arm" in first:
        return load_episode_records(path)
    return load_records(path)


def build_report(
    records: list[LogivEpisodeRecord],
    *,
    baseline_records=(),
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    baseline_records = list(baseline_records)
    logiv_baseline = bool(baseline_records) and all(
        isinstance(item, LogivEpisodeRecord) for item in baseline_records
    )
    reported_records = records + (baseline_records if logiv_baseline else [])
    errors = validate_episode_records(reported_records)
    grouped = defaultdict(list)
    for record in reported_records:
        grouped[(record.goal_mode, record.deviation_mode, record.method_arm)].append(record)
    settings = []
    for (goal_mode, deviation_mode, arm), arm_records in sorted(grouped.items()):
        tasks = []
        for task_id in sorted({item.task_id for item in arm_records}):
            task_records = [item for item in arm_records if item.task_id == task_id]
            tasks.append({"task_id": task_id, **_stats(task_records)})
        settings.append(
            {
                "goal_mode": goal_mode,
                "deviation_mode": deviation_mode,
                "method_arm": arm,
                "overall": _stats(arm_records),
                "tasks": tasks,
                "failure_taxonomy": dict(
                    sorted(Counter(item.terminal_cause for item in arm_records if not item.success).items())
                ),
            }
        )

    comparisons = []
    for goal_mode, deviation_mode in sorted(
        {(item.goal_mode, item.deviation_mode) for item in reported_records}
    ):
        full = [
            item
            for item in reported_records
            if item.goal_mode == goal_mode
            and item.deviation_mode == deviation_mode
            and item.method_arm == "FULL_LOGIV"
        ]
        if not full:
            continue
        comparator_arms = sorted(
            {
                item.method_arm
                for item in reported_records
                if item.goal_mode == goal_mode
                and item.deviation_mode == deviation_mode
                and item.method_arm != "FULL_LOGIV"
            }
        )
        candidates = [
            (
                arm,
                [
                    item
                    for item in reported_records
                    if item.goal_mode == goal_mode
                    and item.deviation_mode == deviation_mode
                    and item.method_arm == arm
                ],
            )
            for arm in comparator_arms
        ]
        if baseline_records and not logiv_baseline and deviation_mode == "NOMINAL":
            candidates.append(("BASE", _baseline_proxy(baseline_records)))
        for arm, comparator in candidates:
            try:
                interval = paired_task_stratified_bootstrap(
                    full,
                    comparator,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
            except ValueError as error:
                comparisons.append(
                    {
                        "goal_mode": goal_mode,
                        "deviation_mode": deviation_mode,
                        "comparator": arm,
                        "error": str(error),
                    }
                )
            else:
                comparisons.append(
                    {
                        "goal_mode": goal_mode,
                        "deviation_mode": deviation_mode,
                        "comparator": arm,
                        **interval,
                    }
                )
    return {
        "schema_version": 1,
        "records": len(reported_records),
        "valid_records": sum(item.valid for item in reported_records),
        "errors": errors,
        "settings": settings,
        "paired_comparisons": comparisons,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# LOGIV LIBERO-10 Results",
        "",
        f"Allocated episode records: **{report['records']}**; valid executions: **{report['valid_records']}**.",
        "",
        "| Goal mode | Deviation | Method | Success | 95% Wilson interval |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for setting in report["settings"]:
        stats = setting["overall"]
        interval = stats["wilson_95"]
        rendered_interval = (
            f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval[0] is not None else "—"
        )
        lines.append(
            f"| {setting['goal_mode']} | {setting['deviation_mode']} | {setting['method_arm']} | "
            f"{stats['successes']}/{stats['allocated']} | {rendered_interval} |"
        )
        for task in setting["tasks"]:
            task_interval = task["wilson_95"]
            lines.append(
                f"| ↳ task {task['task_id']} |  |  | {task['successes']}/{task['allocated']} | "
                f"[{task_interval[0]:.3f}, {task_interval[1]:.3f}] |"
            )
    lines.extend(["", "## Paired task-stratified comparisons", ""])
    if not report["paired_comparisons"]:
        lines.append("No complete paired comparison is available yet.")
    for comparison in report["paired_comparisons"]:
        if "error" in comparison:
            lines.append(f"- Full − {comparison['comparator']}: unavailable ({comparison['error']}).")
        else:
            low, high = comparison["percentile_95"]
            lines.append(
                f"- Full − {comparison['comparator']}: {comparison['estimate']:+.3f} "
                f"(paired bootstrap 95% [{low:+.3f}, {high:+.3f}], "
                f"N={comparison['paired_episodes']})."
            )
    if report["errors"]:
        lines.extend(["", "## Audit errors", ""])
        lines.extend(f"- {item}" for item in report["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report LOGIV task-wise and paired results")
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", default=10_000, type=int)
    parser.add_argument("--bootstrap-seed", default=2026, type=int)
    args = parser.parse_args()
    baseline = load_comparator_records(args.baseline) if args.baseline else []
    report = build_report(
        load_episode_records(args.episodes),
        baseline_records=baseline,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(render_markdown(report))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
