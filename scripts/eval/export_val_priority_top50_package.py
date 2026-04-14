#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHORTLIST_CSV = (
    REPO_ROOT
    / "analysis_outputs"
    / "val_priority_top50_per_level_final"
    / "val_priority_top50_per_level.csv"
)
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "data" / "robotwin_lingbot_eval_seed42_l123_300"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "robotwin_val_priority_top50x3_final_package"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation_results_dualsystem"
DEFAULT_META_EPISODES = DEFAULT_BENCHMARK_DIR / "meta" / "episodes.jsonl"
DEFAULT_META_TASKS = DEFAULT_BENCHMARK_DIR / "meta" / "tasks.jsonl"

SOURCE_RESULT_FILES = [
    "robotwin_lingbot_full_3way_gpu1_shard0of2.json",
    "robotwin_lingbot_full_3way_gpu2_shard1of2.json",
    "robotwin_lingbot_full_3way_tail_gpu1.json",
    "robotwin_lingbot_full_3way_tail_gpu2.json",
    "robotwin_lingbot_full_3way_tail_manifest.json",
]

SOURCE_SESSION_FILES = [
    "robotwin_lingbot_full_3way_parallel_sessions.txt",
    "robotwin_lingbot_full_3way_gpu1.session",
    "robotwin_lingbot_full_3way_gpu1.log",
    "robotwin_lingbot_full_3way_tail_gpu1.resume.session",
    "robotwin_lingbot_full_3way_tail_gpu2.resume.session",
]

CAMERA_DIRS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

MODES = ["task_token_only", "dual_llm", "llm_val"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the final VAL-priority top-50-per-level shortlist into a self-contained package."
    )
    parser.add_argument("--shortlist-csv", type=Path, default=DEFAULT_SHORTLIST_CSV)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before exporting.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict) and "results" in payload:
        return list(payload["results"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unsupported result payload at {path}")


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def flatten_sub_instructions(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return " || ".join(str(item) for item in value)
    return str(value)


def flatten_planner_meta(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def flatten_bool(value: Any) -> int:
    return 1 if bool(value) else 0


def build_final_result_index(results_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    orig1 = results_dir / "robotwin_lingbot_full_3way_gpu1_shard0of2.json"
    orig2 = results_dir / "robotwin_lingbot_full_3way_gpu2_shard1of2.json"
    tail1 = results_dir / "robotwin_lingbot_full_3way_tail_gpu1.json"
    tail2 = results_dir / "robotwin_lingbot_full_3way_tail_gpu2.json"
    manifest = results_dir / "robotwin_lingbot_full_3way_tail_manifest.json"

    manifest_payload = read_json(manifest)
    tail_eps = {int(x) for x in manifest_payload.get("gpu1_episode_indices", [])}
    tail_eps |= {int(x) for x in manifest_payload.get("gpu2_episode_indices", [])}

    result_index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in load_results(orig1) + load_results(orig2):
        episode_index = int(row["episode_index"])
        mode = str(row["mode"])
        if episode_index in tail_eps:
            continue
        result_index[(episode_index, mode)] = row
    for row in load_results(tail1) + load_results(tail2):
        episode_index = int(row["episode_index"])
        mode = str(row["mode"])
        result_index[(episode_index, mode)] = row
    return result_index


def mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_l2": sum(float(row["mean_l2"]) for row in rows) / len(rows),
        "task_progress": sum(float(row["task_progress"]) for row in rows) / len(rows),
        "success_rate": sum(1.0 if row["task_success"] else 0.0 for row in rows) / len(rows),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_selected_tables(selected_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_level = defaultdict(list)
    for row in selected_rows:
        per_level[row["level"]].append(row)

    mix_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for level in ["L1", "L2", "L3"]:
        rows = per_level[level]
        tier_counts = Counter(row["selection_tier"] for row in rows)
        mix_rows.append(
            {
                "level": level,
                "selected_count": len(rows),
                "strict_monotonic": tier_counts["strict_monotonic"],
                "val_top_only": tier_counts["val_top_only"],
                "dual_mid_only": tier_counts["dual_mid_only"],
                "other": tier_counts["other"],
                "avg_val_score": sum(float(row["score_llm_val"]) for row in rows) / len(rows),
                "avg_dual_score": sum(float(row["score_dual_llm"]) for row in rows) / len(rows),
                "avg_token_score": sum(float(row["score_task_token_only"]) for row in rows) / len(rows),
            }
        )

        for mode in MODES:
            metric_rows.append(
                {
                    "scope": level,
                    "mode": mode,
                    **mode_summary([row[f"result_{mode}"] for row in rows]),
                }
            )

    for mode in MODES:
        metric_rows.append(
            {
                "scope": "overall_selected_150",
                "mode": mode,
                **mode_summary([row[f"result_{mode}"] for row in selected_rows]),
            }
        )

    return mix_rows, metric_rows


def main() -> None:
    args = parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    shortlist_rows = list(csv.DictReader(args.shortlist_csv.open()))
    selected_episode_indices = [int(row["episode_index"]) for row in shortlist_rows]
    if len(shortlist_rows) != 150:
        raise RuntimeError(f"Expected 150 shortlist rows, found {len(shortlist_rows)}")

    episode_meta_path = args.benchmark_dir / "meta" / "episodes.jsonl"
    task_meta_path = args.benchmark_dir / "meta" / "tasks.jsonl"
    episode_meta_by_index = {
        int(row["episode_index"]): row for row in read_jsonl(episode_meta_path)
    }
    task_meta_rows = read_jsonl(task_meta_path)
    task_meta_by_name = {}
    for row in task_meta_rows:
        task_name = row.get("task_name") or row.get("canonical_task") or row.get("name")
        if task_name:
            task_meta_by_name[str(task_name)] = row

    final_result_index = build_final_result_index(args.results_dir)

    package_root = args.output_dir
    dataset_root = package_root / "dataset"
    logs_root = package_root / "logs"
    tables_root = package_root / "tables"

    for path in [dataset_root, logs_root, tables_root]:
        path.mkdir(parents=True, exist_ok=True)

    selected_rows_enriched: list[dict[str, Any]] = []
    selected_episode_meta_rows: list[dict[str, Any]] = []
    selected_tasks_seen: set[str] = set()

    for row in shortlist_rows:
        episode_index = int(row["episode_index"])
        episode_name = f"episode_{episode_index:06d}"
        episode_meta = dict(episode_meta_by_index[episode_index])
        selected_episode_meta_rows.append(episode_meta)
        selected_tasks_seen.add(str(episode_meta["task_name"]))

        parquet_src = args.benchmark_dir / "data" / "chunk-000" / f"{episode_name}.parquet"
        parquet_dst = dataset_root / "data" / "chunk-000" / f"{episode_name}.parquet"
        symlink_force(parquet_src, parquet_dst)

        for camera_dir in CAMERA_DIRS:
            video_src = args.benchmark_dir / "videos" / "chunk-000" / camera_dir / f"{episode_name}.mp4"
            video_dst = dataset_root / "videos" / "chunk-000" / camera_dir / f"{episode_name}.mp4"
            symlink_force(video_src, video_dst)

        result_rows = {}
        for mode in MODES:
            result_rows[mode] = dict(final_result_index[(episode_index, mode)])

        selected_rows_enriched.append(
            {
                **row,
                "episode_index": episode_index,
                "step_count": int(row["step_count"]),
                "strict_monotonic": row["strict_monotonic"] == "True",
                "val_top": row["val_top"] == "True",
                "dual_gt_token": row["dual_gt_token"] == "True",
                "penalty": float(row["penalty"]),
                "score_llm_val": float(row["score_llm_val"]),
                "score_dual_llm": float(row["score_dual_llm"]),
                "score_task_token_only": float(row["score_task_token_only"]),
                "gap_val_minus_dual": float(row["gap_val_minus_dual"]),
                "gap_dual_minus_token": float(row["gap_dual_minus_token"]),
                "episode_meta": episode_meta,
                "result_task_token_only": result_rows["task_token_only"],
                "result_dual_llm": result_rows["dual_llm"],
                "result_llm_val": result_rows["llm_val"],
            }
        )

    # Dataset metadata.
    selected_episode_meta_rows.sort(key=lambda row: int(row["episode_index"]))
    write_jsonl(dataset_root / "meta" / "episodes.jsonl", selected_episode_meta_rows)

    selected_task_rows = [task_meta_by_name[name] for name in sorted(selected_tasks_seen) if name in task_meta_by_name]
    if selected_task_rows:
        write_jsonl(dataset_root / "meta" / "tasks.jsonl", selected_task_rows)

    subset_info = {
        "name": "robotwin_val_priority_top50x3_final_package",
        "source_benchmark_dir": str(args.benchmark_dir),
        "selection_csv": str(args.shortlist_csv),
        "selection_policy": "Exact 50 per level, prioritizing val composite score, then monotonic val > dual > task_token_only when available.",
        "file_strategy": "symlink",
        "num_selected_episodes": len(selected_rows_enriched),
        "levels": {
            level: sum(1 for row in selected_rows_enriched if row["level"] == level) for level in ["L1", "L2", "L3"]
        },
        "cameras": CAMERA_DIRS,
    }
    write_json(dataset_root / "meta" / "subset_info.json", subset_info)

    # Logs / extracted results.
    long_rows_jsonl: list[dict[str, Any]] = []
    long_rows_csv: list[dict[str, Any]] = []
    wide_rows_csv: list[dict[str, Any]] = []

    for row in selected_rows_enriched:
        wide_rows_csv.append(
            {
                "level": row["level"],
                "episode_index": row["episode_index"],
                "task_name": row["task_name"],
                "step_count": row["step_count"],
                "selection_tier": row["selection_tier"],
                "strict_monotonic": int(row["strict_monotonic"]),
                "val_top": int(row["val_top"]),
                "dual_gt_token": int(row["dual_gt_token"]),
                "score_llm_val": row["score_llm_val"],
                "score_dual_llm": row["score_dual_llm"],
                "score_task_token_only": row["score_task_token_only"],
                "gap_val_minus_dual": row["gap_val_minus_dual"],
                "gap_dual_minus_token": row["gap_dual_minus_token"],
                "token_mean_l2": row["result_task_token_only"]["mean_l2"],
                "token_task_progress": row["result_task_token_only"]["task_progress"],
                "token_success": flatten_bool(row["result_task_token_only"]["task_success"]),
                "dual_mean_l2": row["result_dual_llm"]["mean_l2"],
                "dual_task_progress": row["result_dual_llm"]["task_progress"],
                "dual_success": flatten_bool(row["result_dual_llm"]["task_success"]),
                "val_mean_l2": row["result_llm_val"]["mean_l2"],
                "val_task_progress": row["result_llm_val"]["task_progress"],
                "val_success": flatten_bool(row["result_llm_val"]["task_success"]),
            }
        )

        for mode in MODES:
            result_row = dict(row[f"result_{mode}"])
            long_rows_jsonl.append(
                {
                    "level": row["level"],
                    "selection_tier": row["selection_tier"],
                    "strict_monotonic": row["strict_monotonic"],
                    "val_top": row["val_top"],
                    "dual_gt_token": row["dual_gt_token"],
                    "score_task_token_only": row["score_task_token_only"],
                    "score_dual_llm": row["score_dual_llm"],
                    "score_llm_val": row["score_llm_val"],
                    "gap_val_minus_dual": row["gap_val_minus_dual"],
                    "gap_dual_minus_token": row["gap_dual_minus_token"],
                    **result_row,
                }
            )

            planner_meta = result_row.get("planner_meta", {})
            long_rows_csv.append(
                {
                    "level": row["level"],
                    "episode_index": row["episode_index"],
                    "task_name": row["task_name"],
                    "step_count": row["step_count"],
                    "selection_tier": row["selection_tier"],
                    "mode": mode,
                    "composite_score_for_mode": row[f"score_{mode}"] if mode != "task_token_only" else row["score_task_token_only"],
                    "strict_monotonic": int(row["strict_monotonic"]),
                    "val_top": int(row["val_top"]),
                    "dual_gt_token": int(row["dual_gt_token"]),
                    "mean_l2": result_row["mean_l2"],
                    "task_progress": result_row["task_progress"],
                    "task_success": flatten_bool(result_row["task_success"]),
                    "rule_success": flatten_bool(result_row.get("rule_success", False)),
                    "num_chunks": result_row.get("num_chunks", ""),
                    "predicted_steps": result_row.get("predicted_steps", ""),
                    "compared_steps": result_row.get("compared_steps", ""),
                    "plan_time": result_row.get("plan_time", ""),
                    "planner_mode": planner_meta.get("planner_mode", ""),
                    "val_result": planner_meta.get("val_result", ""),
                    "num_sub_instructions": len(result_row.get("sub_instructions", [])),
                    "sub_instructions": flatten_sub_instructions(result_row.get("sub_instructions")),
                    "prompt_used": result_row.get("prompt_used", ""),
                    "planner_meta_json": flatten_planner_meta(planner_meta),
                    "judge_reason": result_row.get("judge_reason", ""),
                    "source_uid": result_row.get("source_uid", ""),
                }
            )

    write_jsonl(logs_root / "selected_results_long.jsonl", long_rows_jsonl)
    write_csv(
        logs_root / "selected_results_long.csv",
        long_rows_csv,
        [
            "level",
            "episode_index",
            "task_name",
            "step_count",
            "selection_tier",
            "mode",
            "composite_score_for_mode",
            "strict_monotonic",
            "val_top",
            "dual_gt_token",
            "mean_l2",
            "task_progress",
            "task_success",
            "rule_success",
            "num_chunks",
            "predicted_steps",
            "compared_steps",
            "plan_time",
            "planner_mode",
            "val_result",
            "num_sub_instructions",
            "sub_instructions",
            "prompt_used",
            "planner_meta_json",
            "judge_reason",
            "source_uid",
        ],
    )
    write_csv(
        logs_root / "selected_results_wide.csv",
        wide_rows_csv,
        [
            "level",
            "episode_index",
            "task_name",
            "step_count",
            "selection_tier",
            "strict_monotonic",
            "val_top",
            "dual_gt_token",
            "score_llm_val",
            "score_dual_llm",
            "score_task_token_only",
            "gap_val_minus_dual",
            "gap_dual_minus_token",
            "token_mean_l2",
            "token_task_progress",
            "token_success",
            "dual_mean_l2",
            "dual_task_progress",
            "dual_success",
            "val_mean_l2",
            "val_task_progress",
            "val_success",
        ],
    )

    # Source logs/results provenance via symlink.
    for filename in SOURCE_RESULT_FILES:
        src = args.results_dir / filename
        if src.exists():
            symlink_force(src, logs_root / "source_results" / filename)
    for filename in SOURCE_SESSION_FILES:
        src = args.results_dir / filename
        if src.exists():
            symlink_force(src, logs_root / "source_sessions" / filename)

    mix_rows, metric_rows = build_selected_tables(selected_rows_enriched)
    write_csv(
        tables_root / "selection_mix.csv",
        mix_rows,
        [
            "level",
            "selected_count",
            "strict_monotonic",
            "val_top_only",
            "dual_mid_only",
            "other",
            "avg_val_score",
            "avg_dual_score",
            "avg_token_score",
        ],
    )
    write_csv(
        tables_root / "selected_metrics.csv",
        metric_rows,
        ["scope", "mode", "count", "mean_l2", "task_progress", "success_rate"],
    )

    summary_md = package_root / "README.md"
    lines = []
    lines.append("# Robotwin VAL-Priority Top50x3 Package")
    lines.append("")
    lines.append("This package contains the exact 150-episode shortlist requested from the final completed 300-episode evaluation.")
    lines.append("")
    lines.append("Selection rule:")
    lines.append("- Primary objective: `llm_val` composite score should be as high as possible.")
    lines.append("- Preferred ordering: `llm_val > dual_llm > task_token_only`.")
    lines.append("- Because the final full-run results do not contain 50 strict monotonic wins per level, the remainder is filled with the closest best-effort samples while keeping the average per-level trend monotonic.")
    lines.append("")
    lines.append("Composite score:")
    lines.append("- `0.45 * success + 0.35 * task_progress + 0.20 * normalized_l2_quality`")
    lines.append("")
    lines.append("## Package Layout")
    lines.append("")
    lines.append("- `dataset/`: symlinked subset benchmark files for the 150 selected episodes.")
    lines.append("- `logs/`: extracted per-sample result rows plus raw session/result provenance symlinks.")
    lines.append("- `tables/`: shortlist mix and selected-subset metric tables.")
    lines.append("")
    lines.append("## Selection Mix")
    lines.append("")
    lines.append("| Level | Count | Strict `val>dual>token` | VAL-top-only | Dual-mid-only | Other | Avg VAL Score | Avg Dual Score | Avg Token Score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in mix_rows:
        lines.append(
            f"| {row['level']} | {row['selected_count']} | {row['strict_monotonic']} | {row['val_top_only']} | {row['dual_mid_only']} | {row['other']} | {row['avg_val_score']:.4f} | {row['avg_dual_score']:.4f} | {row['avg_token_score']:.4f} |"
        )
    lines.append("")
    lines.append("## Selected-Subset Metrics")
    lines.append("")
    lines.append("| Scope | Mode | Count | mean_l2 | task_progress | success_rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in metric_rows:
        lines.append(
            f"| {row['scope']} | {row['mode']} | {row['count']} | {row['mean_l2']:.4f} | {row['task_progress']:.4f} | {row['success_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Key Paths")
    lines.append("")
    lines.append(f"- Selection CSV: `{args.shortlist_csv}`")
    lines.append(f"- Exported shortlist CSV: `{logs_root / 'selected_results_wide.csv'}`")
    lines.append(f"- Exported long logs: `{logs_root / 'selected_results_long.csv'}`")
    lines.append(f"- Subset metadata: `{dataset_root / 'meta' / 'episodes.jsonl'}`")
    lines.append(f"- Subset info: `{dataset_root / 'meta' / 'subset_info.json'}`")
    summary_md.write_text("\n".join(lines) + "\n")

    export_manifest = {
        "shortlist_csv": str(args.shortlist_csv),
        "benchmark_dir": str(args.benchmark_dir),
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "num_selected_episodes": len(selected_rows_enriched),
        "num_selected_result_rows": len(long_rows_jsonl),
        "selection_mix": mix_rows,
        "selected_metrics": metric_rows,
    }
    write_json(package_root / "export_manifest.json", export_manifest)

    print(
        json.dumps(
            {
                "output_dir": str(package_root),
                "dataset_meta": str(dataset_root / "meta" / "episodes.jsonl"),
                "selected_results_long_csv": str(logs_root / "selected_results_long.csv"),
                "selected_results_wide_csv": str(logs_root / "selected_results_wide.csv"),
                "tables_readme": str(summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
