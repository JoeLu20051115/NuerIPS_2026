#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "data" / "egodex_dreamdojo_easy400" / "meta" / "manifest.json"
INFO_JSON = REPO_ROOT / "data" / "egodex_eval_official" / "EgoDex_Eval" / "meta" / "info.json"
OUT_DATA_ROOT = REPO_ROOT / "20260331" / "data"
OUT_ANALYSIS_ROOT = REPO_ROOT / "20260331" / "analysis"
SPLIT_ORDER = ["L1", "L2", "L3"]
TARGET_PER_SPLIT = 50

VERBS = ["pick", "place", "stack", "unstack", "insert", "remove", "open", "close", "add"]
STATEFUL_OPS = {"stack", "unstack", "insert", "remove", "open", "close", "add"}


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def tercile_bounds(n: int) -> list[tuple[int, int]]:
    base = n // 3
    rem = n % 3
    counts = [base, base, base + rem]
    out = []
    start = 0
    for count in counts:
        end = start + count
        out.append((start, end))
        start = end
    return out


def task_group_ops(task_group: str) -> list[str]:
    tokens = set(task_group.split("_"))
    ops = [verb for verb in VERBS if verb in tokens]
    if "pick" in tokens and "place" not in tokens:
        ops.append("place")
    if "place" in tokens and "pick" not in tokens:
        ops.append("pick")
    return sorted(set(ops))


def episode_record(ep: dict, fps: float) -> dict:
    ops = task_group_ops(str(ep["task_group"]))
    stateful_ops = sorted(set(ops) & STATEFUL_OPS)
    return {
        **ep,
        "duration_sec": float(ep["num_frames"]) / fps,
        "semantic_ops": ops,
        "semantic_step_proxy": len(ops) if ops else 1,
        "semantic_unique_ops": len(ops),
        "semantic_stateful_ops": stateful_ops,
        "semantic_stateful_count": len(stateful_ops),
        "semantic_complexity_score": (len(ops) if ops else 1) + 0.75 * len(stateful_ops),
    }


def choose_split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    sorted_rows = sorted(rows, key=lambda row: (row["duration_sec"], row["episode_id"]))
    splits: dict[str, list[dict]] = {}
    for label, (start, end) in zip(SPLIT_ORDER, tercile_bounds(len(sorted_rows))):
        bucket = sorted_rows[start:end]
        if label == "L1":
            chosen = sorted(
                bucket,
                key=lambda row: (
                    row["semantic_complexity_score"],
                    row["duration_sec"],
                    row["episode_id"],
                ),
            )[:TARGET_PER_SPLIT]
        elif label == "L2":
            ordered = sorted(
                bucket,
                key=lambda row: (
                    row["semantic_complexity_score"],
                    row["duration_sec"],
                    row["episode_id"],
                ),
            )
            center = max(0, len(ordered) // 2 - TARGET_PER_SPLIT // 2)
            chosen = ordered[center : center + TARGET_PER_SPLIT]
        else:
            chosen = sorted(
                bucket,
                key=lambda row: (
                    -row["semantic_complexity_score"],
                    -row["duration_sec"],
                    row["episode_id"],
                ),
            )[:TARGET_PER_SPLIT]
        chosen = sorted(chosen, key=lambda row: (row["duration_sec"], row["episode_id"]))
        splits[label] = chosen
    return splits


def split_summary(rows: list[dict]) -> dict[str, float]:
    return {
        "count": len(rows),
        "mean_duration_sec": float(sum(row["duration_sec"] for row in rows) / len(rows)),
        "min_duration_sec": float(min(row["duration_sec"] for row in rows)),
        "max_duration_sec": float(max(row["duration_sec"] for row in rows)),
        "mean_step_proxy": float(sum(row["semantic_step_proxy"] for row in rows) / len(rows)),
        "mean_unique_ops": float(sum(row["semantic_unique_ops"] for row in rows) / len(rows)),
        "stateful_rate": float(sum(row["semantic_stateful_count"] > 0 for row in rows) / len(rows)),
    }


def write_manifest(split: str, rows: list[dict]) -> Path:
    dataset_name = f"EgoDex_DreamDojo_ComplexityAligned_{split}_{len(rows)}"
    out_dir = OUT_DATA_ROOT / dataset_name / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    task_counts = Counter(row["task_group"] for row in rows)
    payload = {
        "name": dataset_name,
        "selection_rule": (
            "Start from egodex_dreamdojo_easy400. Convert num_frames to duration_sec at 20 FPS. "
            "Sort all episodes by duration and split into three duration terciles. "
            "Within each tercile, compute a semantic complexity score from task_group verbs: "
            "score = (# semantic ops) + 0.75 * (# stateful ops), where stateful ops are "
            "{stack, unstack, insert, remove, open, close, add}. "
            "Select 50 episodes per split: lowest-score rows for L1, middle window for L2, "
            "highest-score rows for L3; for L3, ties prefer longer durations."
        ),
        "source_manifest": relative_or_absolute(SOURCE_MANIFEST),
        "source_info": relative_or_absolute(INFO_JSON),
        "count": len(rows),
        "task_counts": dict(task_counts),
        "summary": split_summary(rows),
        "episodes": rows,
    }
    out_path = out_dir / "manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def write_analysis(splits: dict[str, list[dict]]) -> None:
    OUT_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_lines = [
        "# EgoDex Complexity-Aligned 50x3 Splits",
        "",
        "This is a curated analysis split, not an unbiased random sample. The rule is explicit and reproducible.",
        "",
        "| Split | N | Mean Time (s) | Mean Step Proxy | Mean Unique Ops | Stateful-op Rate | Task Mix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in SPLIT_ORDER:
        rows = splits[split]
        summary = split_summary(rows)
        task_counts = Counter(row["task_group"] for row in rows)
        mix = ", ".join(f"{task}:{count}" for task, count in task_counts.items())
        summary_rows.append(
            {
                "split": split,
                "n": len(rows),
                "mean_time_sec": summary["mean_duration_sec"],
                "mean_step_proxy": summary["mean_step_proxy"],
                "mean_unique_ops": summary["mean_unique_ops"],
                "stateful_rate": summary["stateful_rate"],
                "task_mix": mix,
            }
        )
        detail_lines.append(
            "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                split,
                len(rows),
                summary["mean_duration_sec"],
                summary["mean_step_proxy"],
                summary["mean_unique_ops"],
                summary["stateful_rate"],
                mix,
            )
        )

    summary_tsv = OUT_ANALYSIS_ROOT / "egodex_complexity_aligned_50x3_summary.tsv"
    summary_md = OUT_ANALYSIS_ROOT / "egodex_complexity_aligned_50x3_summary.md"
    pd_lines = [
        "split\tn\tmean_time_sec\tmean_step_proxy\tmean_unique_ops\tstateful_rate\ttask_mix"
    ]
    for row in summary_rows:
        pd_lines.append(
            "{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}".format(
                row["split"],
                row["n"],
                row["mean_time_sec"],
                row["mean_step_proxy"],
                row["mean_unique_ops"],
                row["stateful_rate"],
                row["task_mix"],
            )
        )
    summary_tsv.write_text("\n".join(pd_lines) + "\n", encoding="utf-8")
    summary_md.write_text("\n".join(detail_lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    info = json.loads(INFO_JSON.read_text())
    fps = float(info["fps"])
    rows = [episode_record(ep, fps) for ep in manifest["episodes"]]
    splits = choose_split_rows(rows)
    for split in SPLIT_ORDER:
        write_manifest(split, splits[split])
    write_analysis(splits)


if __name__ == "__main__":
    main()
