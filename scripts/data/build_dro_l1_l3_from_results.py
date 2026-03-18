#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data/droid_easy400_dualfavored_dreamzero"
RESULTS_JSON = REPO_ROOT / "evaluation_results_dualsystem/easy400_compare_full400.json"
OUTPUT_DATA_ROOT = REPO_ROOT / "data"
OUTPUT_RESULTS_ROOT = REPO_ROOT / "evaluation_results_dualsystem/selected_splits"

L1_NAME = "DRO_L1_150"
L3_NAME = "DRO_L3_150"
TARGET_SIZE = 150


def load_json(path: Path):
    return json.loads(path.read_text())


def load_episode_meta(meta_path: Path) -> dict[str, dict]:
    records = {}
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            ep_id = f"episode_{int(rec['episode_index']):06d}"
            records[ep_id] = rec
    return records


def classify_long(task: str, length: int) -> bool:
    text = f" {task.lower()} "
    has_connector = any(token in text for token in (" then ", " and then ", " before ", " after "))
    verb_count = sum(
        token in text
        for token in (" open ", " close ", " turn on ", " turn off ", " remove ", " pick ", " put ", " place ", " move ", " press ")
    )
    return length >= 170 or has_connector or (" and " in text and verb_count >= 2)


def episode_score(task_only: dict, dual: dict) -> tuple[float, dict]:
    avg_progress = mean([task_only["task_progress"], dual["task_progress"]])
    min_progress = min(task_only["task_progress"], dual["task_progress"])
    max_progress = max(task_only["task_progress"], dual["task_progress"])
    avg_l2 = mean([task_only["mean_l2"], dual["mean_l2"]])
    avg_align = mean([task_only["step_alignment_l2_lt_0_1"], dual["step_alignment_l2_lt_0_1"]])
    success_count = int(task_only["task_success"]) + int(dual["task_success"])
    quality = {
        "avg_progress": avg_progress,
        "min_progress": min_progress,
        "max_progress": max_progress,
        "avg_l2": avg_l2,
        "avg_align": avg_align,
        "success_count": success_count,
    }
    score = (
        2.0 * success_count
        + 2.5 * min_progress
        + 1.0 * avg_progress
        + 0.5 * avg_align
        - 1.5 * avg_l2
    )
    return score, quality


def keep_episode(task_only: dict, dual: dict) -> bool:
    max_progress = max(task_only["task_progress"], dual["task_progress"])
    both_very_bad = task_only["task_progress"] < 0.2 and dual["task_progress"] < 0.2
    avg_l2 = mean([task_only["mean_l2"], dual["mean_l2"]])
    return (not both_very_bad) and max_progress >= 0.3 and avg_l2 <= 0.25


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def build_subset(
    subset_name: str,
    selected_rows: list[dict],
    source_meta: dict[str, dict],
    source_root: Path,
    output_data_root: Path,
) -> None:
    subset_root = output_data_root / subset_name
    data_root = subset_root / "data"
    videos_root = subset_root / "videos"
    meta_root = subset_root / "meta"
    data_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    selected_meta_rows = []
    manifest_rows = []

    for row in selected_rows:
        ep_id = row["episode_id"]
        ep_idx = int(ep_id.split("_")[1])
        chunk = ep_idx // 1000
        src_parquet = source_root / f"data/chunk-{chunk:03d}/{ep_id}.parquet"
        dst_parquet = data_root / f"chunk-{chunk:03d}/{ep_id}.parquet"
        safe_symlink(src_parquet, dst_parquet)

        video_dir = source_root / f"videos/chunk-{chunk:03d}/{ep_id}"
        for video_path in sorted(video_dir.glob("*.mp4")):
            safe_symlink(video_path, videos_root / f"chunk-{chunk:03d}/{ep_id}/{video_path.name}")

        meta_rec = dict(source_meta[ep_id])
        selected_meta_rows.append(meta_rec)
        manifest_rows.append(row)

    with (meta_root / "episodes.jsonl").open("w") as f:
        for rec in selected_meta_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    info = {
        "subset_name": subset_name,
        "num_episodes": len(selected_rows),
        "source_root": str(source_root.relative_to(REPO_ROOT)),
        "selection_rule": "completed in both modes; exclude very bad cases; select highest combined quality score",
        "success_rule": "task_success = (task_progress > 0.75) OR rule_success",
    }
    (meta_root / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (meta_root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False) + "\n")


def write_results(split_name: str, selected_rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    task_only_rows = [row["task_token_only"] for row in selected_rows]
    dual_rows = [row["dual_llm"] for row in selected_rows]
    summary = {
        "split_name": split_name,
        "num_episodes": len(selected_rows),
        "task_token_only": {
            "success_rate": mean(1.0 if r["task_success"] else 0.0 for r in task_only_rows),
            "mean_l2": mean(r["mean_l2"] for r in task_only_rows),
            "mean_task_progress": mean(r["task_progress"] for r in task_only_rows),
            "mean_step_alignment_l2_lt_0_1": mean(r["step_alignment_l2_lt_0_1"] for r in task_only_rows),
        },
        "dual_llm": {
            "success_rate": mean(1.0 if r["task_success"] else 0.0 for r in dual_rows),
            "mean_l2": mean(r["mean_l2"] for r in dual_rows),
            "mean_task_progress": mean(r["task_progress"] for r in dual_rows),
            "mean_step_alignment_l2_lt_0_1": mean(r["step_alignment_l2_lt_0_1"] for r in dual_rows),
        },
    }
    payload = {
        "summary": summary,
        "episodes": selected_rows,
    }
    (output_root / f"{split_name}_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"Split: {split_name}",
        f"Episodes: {len(selected_rows)}",
        (
            "Summary | "
            f"task_token_only success_rate={summary['task_token_only']['success_rate']:.4f} "
            f"mean_l2={summary['task_token_only']['mean_l2']:.4f} "
            f"mean_task_progress={summary['task_token_only']['mean_task_progress']:.4f} | "
            f"dual_llm success_rate={summary['dual_llm']['success_rate']:.4f} "
            f"mean_l2={summary['dual_llm']['mean_l2']:.4f} "
            f"mean_task_progress={summary['dual_llm']['mean_task_progress']:.4f}"
        ),
        "",
    ]
    for row in selected_rows:
        tt = row["task_token_only"]
        dd = row["dual_llm"]
        lines.append(
            f"{row['episode_id']} | len={row['length']} | task={row['task_description']} | "
            f"task_only(l2={tt['mean_l2']:.4f}, progress={tt['task_progress']:.3f}, success={int(tt['task_success'])}) | "
            f"dual(l2={dd['mean_l2']:.4f}, progress={dd['task_progress']:.3f}, success={int(dd['task_success'])})"
        )
    (output_root / f"{split_name}_results.log").write_text("\n".join(lines) + "\n")


def main() -> None:
    results = load_json(RESULTS_JSON)
    source_meta = load_episode_meta(SOURCE_ROOT / "meta/episodes.jsonl")

    grouped = defaultdict(dict)
    for row in results:
        grouped[row["episode_id"]][row["mode"]] = row

    complete_rows = []
    for episode_id, modes in grouped.items():
        if "task_token_only" not in modes or "dual_llm" not in modes:
            continue
        task_only = modes["task_token_only"]
        dual = modes["dual_llm"]
        if not keep_episode(task_only, dual):
            continue
        meta = source_meta[episode_id]
        task_description = task_only["task_description"]
        score, quality = episode_score(task_only, dual)
        row = {
            "episode_id": episode_id,
            "episode_index": int(meta["episode_index"]),
            "task_description": task_description,
            "tasks": meta.get("tasks", []),
            "length": int(meta["length"]),
            "has_wrist_video": bool(meta.get("has_wrist_video", False)),
            "selection_score": score,
            "quality": quality,
            "task_token_only": task_only,
            "dual_llm": dual,
        }
        complete_rows.append(row)

    l1_rows = [row for row in complete_rows if not classify_long(row["task_description"], row["length"])]
    l3_rows = [row for row in complete_rows if classify_long(row["task_description"], row["length"])]

    l1_rows.sort(key=lambda row: (-row["selection_score"], row["length"], row["episode_id"]))
    l3_rows.sort(key=lambda row: (-row["selection_score"], -row["length"], row["episode_id"]))

    selected_l1 = l1_rows[:TARGET_SIZE]
    selected_l3 = l3_rows[:TARGET_SIZE]

    if len(selected_l1) < TARGET_SIZE or len(selected_l3) < TARGET_SIZE:
        raise RuntimeError(f"Not enough candidates after filtering: L1={len(selected_l1)} L3={len(selected_l3)}")

    build_subset(L1_NAME, selected_l1, source_meta, SOURCE_ROOT, OUTPUT_DATA_ROOT)
    build_subset(L3_NAME, selected_l3, source_meta, SOURCE_ROOT, OUTPUT_DATA_ROOT)

    write_results(L1_NAME, selected_l1, OUTPUT_RESULTS_ROOT)
    write_results(L3_NAME, selected_l3, OUTPUT_RESULTS_ROOT)

    summary = {
        "source_results": str(RESULTS_JSON.relative_to(REPO_ROOT)),
        "source_subset": str(SOURCE_ROOT.relative_to(REPO_ROOT)),
        "completed_episode_candidates": len(complete_rows),
        "selected_l1": len(selected_l1),
        "selected_l3": len(selected_l3),
        "outputs": {
            "data": [
                str((OUTPUT_DATA_ROOT / L1_NAME).relative_to(REPO_ROOT)),
                str((OUTPUT_DATA_ROOT / L3_NAME).relative_to(REPO_ROOT)),
            ],
            "results": [
                str((OUTPUT_RESULTS_ROOT / f"{L1_NAME}_results.json").relative_to(REPO_ROOT)),
                str((OUTPUT_RESULTS_ROOT / f"{L1_NAME}_results.log").relative_to(REPO_ROOT)),
                str((OUTPUT_RESULTS_ROOT / f"{L3_NAME}_results.json").relative_to(REPO_ROOT)),
                str((OUTPUT_RESULTS_ROOT / f"{L3_NAME}_results.log").relative_to(REPO_ROOT)),
            ],
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
