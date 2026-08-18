#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from pi05_libero_repro.logiv.records import (
    LogivEpisodeRecord,
    load_episode_records,
    paired_task_stratified_bootstrap,
    validate_episode_records,
)


def _pair_key(record: LogivEpisodeRecord) -> tuple[int, int, int]:
    return record.seed, record.task_id, record.episode_idx


def build_online_report(
    base_records: Iterable[LogivEpisodeRecord],
    online_records: Iterable[LogivEpisodeRecord],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    base_records = list(base_records)
    online_records = list(online_records)
    errors = validate_episode_records(base_records + online_records)
    base = {_pair_key(record): record for record in base_records}
    online = {_pair_key(record): record for record in online_records}
    if len(base) != len(base_records) or len(online) != len(online_records):
        errors.append("duplicate seed/task/episode pairing key")
    if base.keys() != online.keys():
        missing_online = sorted(base.keys() - online.keys())
        missing_base = sorted(online.keys() - base.keys())
        errors.append(
            f"unpaired records: missing_online={missing_online}, missing_base={missing_base}"
        )

    paired_keys = sorted(base.keys() & online.keys())
    positive = negative = unchanged_success = unchanged_failure = 0
    intervention_positive = no_trigger_positive = 0
    no_trigger_pairs = 0
    no_trigger_outcome_matches = 0
    no_trigger_step_matches = 0
    no_trigger_request_matches = 0
    trigger_counts: Counter[str] = Counter()
    per_task: dict[int, dict[str, int]] = {}
    pair_rows = []
    for key in paired_keys:
        baseline = base[key]
        repaired = online[key]
        trigger = repaired.online_trigger_kind or "NO_TRIGGER"
        if baseline.init_state_sha256 != repaired.init_state_sha256:
            errors.append(f"initial-state hash mismatch: {key}")
        if baseline.first_frame_sha256 != repaired.first_frame_sha256:
            errors.append(f"first-frame hash mismatch: {key}")
        if baseline.success and repaired.success:
            unchanged_success += 1
            flip = 0
        elif not baseline.success and not repaired.success:
            unchanged_failure += 1
            flip = 0
        elif repaired.success:
            positive += 1
            flip = 1
            if (
                repaired.online_trigger_kind is not None
                and repaired.repair_steps > 0
            ):
                intervention_positive += 1
            else:
                no_trigger_positive += 1
        else:
            negative += 1
            flip = -1
        trigger_counts[trigger] += 1
        if repaired.online_trigger_kind is None:
            no_trigger_pairs += 1
            no_trigger_outcome_matches += int(
                baseline.success == repaired.success
            )
            no_trigger_step_matches += int(baseline.steps == repaired.steps)
            no_trigger_request_matches += int(
                baseline.inference_requests == repaired.inference_requests
            )
            if baseline.success != repaired.success:
                errors.append(f"no-trigger outcome mismatch: {key}")
            if baseline.steps != repaired.steps:
                errors.append(f"no-trigger step mismatch: {key}")
            if baseline.inference_requests != repaired.inference_requests:
                errors.append(f"no-trigger policy-request mismatch: {key}")
        task = per_task.setdefault(
            repaired.task_id,
            {"paired": 0, "base_success": 0, "online_success": 0,
             "positive_flips": 0, "intervention_positive_flips": 0,
             "negative_flips": 0},
        )
        task["paired"] += 1
        task["base_success"] += int(baseline.success)
        task["online_success"] += int(repaired.success)
        task["positive_flips"] += int(flip > 0)
        task["intervention_positive_flips"] += int(
            flip > 0
            and repaired.online_trigger_kind is not None
            and repaired.repair_steps > 0
        )
        task["negative_flips"] += int(flip < 0)
        pair_rows.append(
            {
                "seed": repaired.seed,
                "task_id": repaired.task_id,
                "episode_idx": repaired.episode_idx,
                "base_success": baseline.success,
                "online_success": repaired.success,
                "flip": flip,
                "attributable_recovery": (
                    flip > 0
                    and repaired.online_trigger_kind is not None
                    and repaired.repair_steps > 0
                ),
                "trigger": trigger,
                "trigger_step": repaired.online_trigger_step,
                "base_prefix_steps": repaired.base_prefix_steps,
                "repair_steps": repaired.repair_steps,
                "combined_actions": repaired.combined_actions,
            }
        )

    comparison = None
    if paired_keys:
        def comparison_proxy(record: LogivEpisodeRecord) -> SimpleNamespace:
            return SimpleNamespace(
                seed=record.seed,
                task_id=record.task_id,
                episode_idx=record.episode_idx,
                success=record.success,
                init_state_sha256=record.init_state_sha256,
                first_frame_sha256=record.first_frame_sha256,
                checkpoint=record.checkpoint,
            )

        comparison = paired_task_stratified_bootstrap(
            [comparison_proxy(online[key]) for key in paired_keys],
            [comparison_proxy(base[key]) for key in paired_keys],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    prior_failures = positive + unchanged_failure
    return {
        "schema_version": 1,
        "evidence_label": "development/tuning evidence",
        "paired_episodes": len(paired_keys),
        "base_successes": sum(base[key].success for key in paired_keys),
        "online_successes": sum(online[key].success for key in paired_keys),
        "flips": {
            "positive": positive,
            "negative": negative,
            "net": positive - negative,
            "intervention_positive": intervention_positive,
            "no_trigger_positive": no_trigger_positive,
            "unchanged_success": unchanged_success,
            "unchanged_failure": unchanged_failure,
        },
        "failure_first_feasibility": {
            "prior_base_failures": prior_failures,
            "recovered": intervention_positive,
            "recovery_rate": (
                intervention_positive / prior_failures if prior_failures else None
            ),
        },
        "no_trigger_parity": {
            "pairs": no_trigger_pairs,
            "outcome_matches": no_trigger_outcome_matches,
            "step_matches": no_trigger_step_matches,
            "inference_request_matches": no_trigger_request_matches,
        },
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "tasks": [
            {"task_id": task_id, **counts}
            for task_id, counts in sorted(per_task.items())
        ],
        "paired_bootstrap": comparison,
        "pairs": pair_rows,
        "errors": errors,
    }


def render_online_markdown(report: dict) -> str:
    flips = report["flips"]
    feasibility = report["failure_first_feasibility"]
    parity = report["no_trigger_parity"]
    lines = [
        "# LOGIV Online vs BASE",
        "",
        f"> This is **{report['evidence_label']}**, not independent holdout evidence.",
        "",
        f"Paired episodes: **{report['paired_episodes']}**. BASE: "
        f"**{report['base_successes']}**; LOGIV Online: "
        f"**{report['online_successes']}**.",
        "",
        f"Positive flips: **{flips['positive']}**; negative flips: "
        f"**{flips['negative']}**; net flips: **{flips['net']:+d}**.",
        "",
        "Intervention-attributable positive flips: "
        f"**{flips['intervention_positive']}**; no-trigger positive flips: "
        f"**{flips['no_trigger_positive']}**.",
        "",
        "Failure-first feasibility: "
        f"**{feasibility['recovered']}/{feasibility['prior_base_failures']}** "
        "prior BASE failures recovered.",
        "",
        "No-trigger parity: "
        f"outcome {parity['outcome_matches']}/{parity['pairs']}, "
        f"steps {parity['step_matches']}/{parity['pairs']}, policy requests "
        f"{parity['inference_request_matches']}/{parity['pairs']}.",
        "",
        "| Task | Paired | BASE | LOGIV | Positive | Attributable | Negative | Net |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| {task['task_id']} | {task['paired']} | {task['base_success']} | "
            f"{task['online_success']} | {task['positive_flips']} | "
            f"{task['intervention_positive_flips']} | "
            f"{task['negative_flips']} | "
            f"{task['positive_flips'] - task['negative_flips']:+d} |"
        )
    if report["paired_bootstrap"] is not None:
        comparison = report["paired_bootstrap"]
        low, high = comparison["percentile_95"]
        lines.extend(
            [
                "",
                f"Task-stratified paired delta: **{comparison['estimate']:+.3f}** "
                f"(bootstrap 95% [{low:+.3f}, {high:+.3f}]).",
            ]
        )
    if report["errors"]:
        lines.extend(["", "## Audit errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.append("")
    return "\n".join(lines)


def _load_many(paths: Iterable[Path]) -> list[LogivEpisodeRecord]:
    return [record for path in paths for record in load_episode_records(path)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired BASE/LOGIV Online report")
    parser.add_argument("--base", action="append", required=True, type=Path)
    parser.add_argument("--logiv", action="append", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", default=10_000, type=int)
    parser.add_argument("--bootstrap-seed", default=2026, type=int)
    args = parser.parse_args()
    report = build_online_report(
        _load_many(args.base),
        _load_many(args.logiv),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(render_online_markdown(report))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
