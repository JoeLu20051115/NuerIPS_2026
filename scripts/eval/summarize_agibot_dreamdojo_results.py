#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODE_ALIASES = {
    "task_token_only": "task_token_only",
    "description_only": "task_token_only",
    "dual_llm": "dual_llm",
    "llm_val": "val_llm",
    "val_llm": "val_llm",
}
DEFAULT_PRINT_ORDER = ["task_token_only", "dual_llm", "val_llm"]
DISPLAY_LABELS = {
    "task_token_only": "description_only",
    "dual_llm": "dual_llm",
    "val_llm": "val_llm",
}


def canonicalize_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key not in MODE_ALIASES:
        raise ValueError(f"Unsupported mode '{mode}'")
    return MODE_ALIASES[key]


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("mean_l2") is not None]
    if not rows:
        return {
            "num_episodes": 0,
            "mean_l2": None,
            "mean_task_progress": None,
            "success_rate": None,
            "rate_of_l2_lt_0_1": None,
        }
    total_steps = sum(int(r.get("num_steps", 0)) for r in rows)
    total_pass = sum(int(r.get("num_step_pass_l2_lt_0_1", 0)) for r in rows)
    return {
        "num_episodes": len(rows),
        "mean_l2": (
            float(sum(float(r["mean_l2"]) for r in valid) / len(valid))
            if valid
            else None
        ),
        "mean_task_progress": float(
            sum(float(r["task_progress"]) for r in rows) / len(rows)
        ),
        "success_rate": float(
            sum(1.0 if r.get("task_success") else 0.0 for r in rows) / len(rows)
        ),
        "rate_of_l2_lt_0_1": float(total_pass / total_steps) if total_steps else None,
    }


def format_metric(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.4f}%"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DreamDojo AgiBot comparison results into a compact split log."
    )
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help="Output log path. Defaults to <results-path stem>_summary.log",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset/split name to print in the summary header.",
    )
    args = parser.parse_args()

    payload = json.loads(args.results_path.read_text())
    raw_summary = payload.get("summary")
    if raw_summary:
        summary = {
            canonicalize_mode(mode): stats
            for mode, stats in raw_summary.items()
        }
    else:
        grouped: dict[str, list[dict]] = {}
        for row in payload.get("results", []):
            mode = canonicalize_mode(str(row["mode"]))
            grouped.setdefault(mode, []).append(row)
        summary = {mode: summarize(rows) for mode, rows in grouped.items()}

    dataset_name = (
        args.dataset_name
        or payload.get("meta", {}).get("dataset_name")
        or args.results_path.stem
    )
    output_log = (
        args.output_log
        if args.output_log is not None
        else args.results_path.with_name(f"{args.results_path.stem}_summary.log")
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{dataset_name} summary"]
    for mode in DEFAULT_PRINT_ORDER:
        if mode not in summary:
            continue
        stats = summary[mode]
        lines.append(
            f"{DISPLAY_LABELS[mode]}: "
            f"Mean L2={format_metric(stats.get('mean_l2'))} | "
            f"Mean Task Progress={format_metric(stats.get('mean_task_progress'))} | "
            f"Success Rate={format_metric(stats.get('success_rate'), pct=True)} | "
            f"Rate of L2 < 0.1={format_metric(stats.get('rate_of_l2_lt_0_1'), pct=True)}"
        )

    output_log.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {output_log}")


if __name__ == "__main__":
    main()
