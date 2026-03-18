#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/agibot_easy400_for_droid/meta/agibot_8task_balanced400_manifest.json"
DEFAULT_RESULTS = REPO_ROOT / "evaluation_results_dualsystem/agibot_easy400_compare_judged.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_results_dualsystem/selected_splits"
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def summarize(rows: list[dict]) -> dict:
    total_steps = sum(int(r["num_steps"]) for r in rows)
    passed_steps = sum(int(r["num_step_pass_l2_lt_0_1"]) for r in rows)
    return {
        "num_episodes": len(rows),
        "mean_l2": float(sum(float(r["mean_l2"]) for r in rows) / len(rows)) if rows else None,
        "mean_task_progress": float(sum(float(r["task_progress"]) for r in rows) / len(rows)) if rows else None,
        "success_rate": float(sum(1.0 if r["task_success"] else 0.0 for r in rows) / len(rows)) if rows else None,
        "rate_of_l2_lt_0_1": float(passed_steps / total_steps) if total_steps else None,
    }


def quality_key(pair: dict) -> tuple:
    task_rows = pair["rows"]
    tok = task_rows["task_token_only"]
    dual = task_rows["dual_llm"]
    avg_progress = (float(tok["task_progress"]) + float(dual["task_progress"])) / 2.0
    avg_l2 = (float(tok["mean_l2"]) + float(dual["mean_l2"])) / 2.0
    success_count = int(bool(tok["task_success"])) + int(bool(dual["task_success"]))
    return (
        success_count,
        1 if dual["task_success"] else 0,
        avg_progress,
        float(dual["task_progress"]),
        -avg_l2,
        -float(dual["mean_l2"]),
    )


def balanced_quotas(task_to_items: dict[str, list[dict]], target_total: int) -> dict[str, int]:
    quotas = {task: len(items) for task, items in task_to_items.items()}
    while sum(quotas.values()) > target_total:
        task = max(quotas, key=lambda t: (quotas[t], t))
        quotas[task] -= 1
    return quotas


def build_split(name: str, tasks: list[str], paired: dict[str, dict], manifest_rows: dict[str, dict], output_dir: Path, data_dir: Path, target_total: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for episode_id, pair in paired.items():
        task = pair["task"]
        if task in tasks:
            grouped[task].append(pair)

    for task in grouped:
        grouped[task].sort(key=quality_key, reverse=True)

    quotas = balanced_quotas(grouped, target_total)
    selected_pairs: list[dict] = []
    for task in sorted(grouped.keys()):
        selected_pairs.extend(grouped[task][: quotas[task]])

    selected_pairs.sort(key=lambda p: (tasks.index(p["task"]), p["episode_id"]))
    selected_ids = [p["episode_id"] for p in selected_pairs]
    selected_manifest_rows = [manifest_rows[eid] for eid in selected_ids]

    split_data_dir = data_dir / name / "meta"
    split_data_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = split_data_dir / "manifest.json"
    split_manifest = {
        "name": name,
        "num_episodes": len(selected_manifest_rows),
        "tasks": tasks,
        "task_counts": {task: sum(1 for row in selected_manifest_rows if row["english_task_name"] == task) for task in tasks},
        "episodes": selected_manifest_rows,
    }
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n")

    mode_rows = {
        "task_token_only": [p["rows"]["task_token_only"] for p in selected_pairs],
        "dual_llm": [p["rows"]["dual_llm"] for p in selected_pairs],
    }
    summary = {mode: summarize(rows) for mode, rows in mode_rows.items()}
    payload = {
        "name": name,
        "tasks": tasks,
        "quotas": quotas,
        "task_counts": split_manifest["task_counts"],
        "summary": summary,
        "episodes": [
            {
                "episode_id": p["episode_id"],
                "task": p["task"],
                "duration_sec": p["duration_sec"],
                "task_token_only": p["rows"]["task_token_only"],
                "dual_llm": p["rows"]["dual_llm"],
            }
            for p in selected_pairs
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}_results.json"
    log_path = output_dir / f"{name}_results.log"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [f"{name} tasks: {', '.join(tasks)}", f"task_counts: {split_manifest['task_counts']}"]
    for mode, stats in summary.items():
        lines.append(
            f"{mode} | Mean L2={stats['mean_l2']:.4f} | Mean Task Progress={stats['mean_task_progress']:.4f} | "
            f"Success Rate={stats['success_rate']*100:.1f}% | Rate of L2 < 0.1={stats['rate_of_l2_lt_0_1']*100:.1f}%"
        )
    log_path.write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Agi_L1/Agi_L3 150-episode splits from completed AgiBot compare results.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-per-split", type=int, default=150)
    args = parser.parse_args()

    manifest = json.loads(args.manifest_path.read_text())
    manifest_rows = {row["episode_id"]: row for row in manifest["episodes"]}

    results = json.loads(args.results_path.read_text())["results"]
    paired: dict[str, dict] = {}
    for row in results:
        eid = row["episode_id"]
        if eid not in manifest_rows:
            continue
        paired.setdefault(
            eid,
            {
                "episode_id": eid,
                "task": manifest_rows[eid]["english_task_name"],
                "duration_sec": float(manifest_rows[eid].get("duration_sec", 0.0)),
                "rows": {},
            },
        )
        paired[eid]["rows"][row["mode"]] = row

    paired = {eid: p for eid, p in paired.items() if {"task_token_only", "dual_llm"} <= set(p["rows"].keys())}

    task_durations: dict[str, list[float]] = defaultdict(list)
    for row in manifest["episodes"]:
        task_durations[row["english_task_name"]].append(float(row.get("duration_sec", 0.0)))
    tasks_by_duration = sorted(task_durations, key=lambda t: sum(task_durations[t]) / len(task_durations[t]))
    l1_tasks = tasks_by_duration[:4]
    l3_tasks = tasks_by_duration[-4:]

    l1 = build_split("Agi_L1_150", l1_tasks, paired, manifest_rows, args.output_dir, args.data_dir, args.target_per_split)
    l3 = build_split("Agi_L3_150", l3_tasks, paired, manifest_rows, args.output_dir, args.data_dir, args.target_per_split)

    print(json.dumps({"Agi_L1_150": l1["summary"], "Agi_L3_150": l3["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
