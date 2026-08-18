#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path


CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]


GOOD_KEYWORDS = [
    "press",
    "button",
    "key",
    "switch",
    "close",
    "drawer",
    "move",
    "marker",
    "pen",
    "mug",
    "cup",
    "block",
    "bowl",
    "put",
]

DUAL_FAVORED_KEYWORDS = [
    "close",
    "drawer",
    "press",
    "button",
    "key",
    "keyboard",
    "switch",
    "light switch",
    "move",
    "marker",
    "pen",
    "pencil",
    "mug",
    "cup",
    "block",
    "bowl",
    "put",
]

DUAL_FAVORED_PHRASES = [
    "close the open drawer",
    "close the top drawer",
    "close the drawer",
    "press a button",
    "press a key",
    "turn on the right light switch",
    "move the pencil to the left",
    "move the mug to the right",
    "put the marker in the mug",
    "put the marker in the bowl",
    "put the pen in the cup",
    "put the red marker on the table",
    "put the glass lid on the black pot",
]

BAD_KEYWORDS = [
    "rubber band",
    "elastic",
    "cloth",
    "clothes",
    "spout",
    "faucet",
    "tap",
    "fold",
    "pour",
    "lid",
    "pot",
    "stand",
    "toaster",
    "curtain",
]

DUAL_UNFAVORED_KEYWORDS = [
    "rubber band",
    "elastic",
    "cloth",
    "clothes",
    "spout",
    "faucet",
    "tap",
    "fold",
    "pour",
    "curtain",
]


def load_episode_meta(meta_path: Path) -> dict[int, dict]:
    episodes: dict[int, dict] = {}
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            episodes[int(rec["episode_index"])] = rec
    return episodes


def primary_task(rec: dict) -> str:
    tasks = rec.get("tasks", [])
    return str(tasks[0]) if tasks else f"episode_{rec['episode_index']:06d}"


def task_score(task: str, length: int, has_wrist: bool) -> float:
    t = task.lower()
    score = 0.0
    score -= float(length) / 100.0
    score += sum(2.0 for kw in GOOD_KEYWORDS if kw in t)
    score -= sum(3.0 for kw in BAD_KEYWORDS if kw in t)
    if " and " in t or " then " in t:
        score -= 3.0
    if sum(t.count(v) for v in ["put", "move", "press", "close"]) == 1:
        score += 1.0
    if has_wrist:
        score += 0.75
    return score


def dual_bonus(task: str, length: int, has_wrist: bool) -> float:
    t = task.lower()
    bonus = 0.0
    bonus += sum(1.5 for kw in DUAL_FAVORED_KEYWORDS if kw in t)
    bonus += sum(3.0 for phrase in DUAL_FAVORED_PHRASES if phrase in t)
    bonus -= sum(2.5 for kw in DUAL_UNFAVORED_KEYWORDS if kw in t)
    if " and " in t or " then " in t:
        bonus -= 2.0
    if length <= 120:
        bonus += 1.5
    elif length <= 180:
        bonus += 0.75
    if has_wrist:
        bonus += 0.5
    return bonus


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_subset(source_root: Path, output_root: Path, target_count: int, selection_mode: str) -> dict:
    meta = load_episode_meta(source_root / "meta/episodes.jsonl")
    source_info = json.loads((source_root / "meta/info.json").read_text())

    # Local snapshot currently exposes exactly the camera-covered subset we want:
    # require both external cameras so downstream success judging remains possible.
    candidates = []
    for video_path in sorted(source_root.glob("videos/chunk-*/observation.images.exterior_image_1_left/episode_*.mp4")):
        ep_idx = int(video_path.stem.split("_")[1])
        ext2 = video_path.parents[1] / "observation.images.exterior_image_2_left" / video_path.name
        if not ext2.exists() or ep_idx not in meta:
            continue
        rec = meta[ep_idx]
        wrist = video_path.parents[1] / "observation.images.wrist_image_left" / video_path.name
        has_wrist = wrist.exists()
        task = primary_task(rec)
        easy_score = task_score(task, int(rec.get("length", 0)), has_wrist)
        combined_score = easy_score
        if selection_mode == "easy_dual":
            combined_score += dual_bonus(task, int(rec.get("length", 0)), has_wrist)
        candidates.append(
            {
                "episode_index": ep_idx,
                "episode_id": f"episode_{ep_idx:06d}",
                "task": task,
                "length": int(rec.get("length", 0)),
                "tasks": rec.get("tasks", []),
                "has_ext1": True,
                "has_ext2": True,
                "has_wrist": has_wrist,
                "easy_score": easy_score,
                "selection_score": combined_score,
            }
        )

    candidates.sort(
        key=lambda x: (
            x["selection_score"],
            1 if x["has_wrist"] else 0,
            -x["length"],
            -x["episode_index"],
        ),
        reverse=True,
    )
    selected = candidates[: min(target_count, len(candidates))]

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    selected_indices = {item["episode_index"] for item in selected}

    for item in selected:
        ep_idx = item["episode_index"]
        chunk = ep_idx // 1000
        name = f"episode_{ep_idx:06d}"

        parquet_src = source_root / f"data/chunk-{chunk:03d}/{name}.parquet"
        parquet_dst = output_root / f"data/chunk-{chunk:03d}/{name}.parquet"
        ensure_symlink(parquet_src, parquet_dst)

        for cam_key in CAMERA_KEYS:
            video_src = source_root / f"videos/chunk-{chunk:03d}/{cam_key}/{name}.mp4"
            if video_src.exists():
                video_dst = output_root / f"videos/chunk-{chunk:03d}/{cam_key}/{name}.mp4"
                ensure_symlink(video_src, video_dst)

    # Preserve the original episode indices, but reorder metadata by easy score so
    # "first N" access patterns naturally hit easier examples first.
    subset_meta_lines = []
    for item in selected:
        rec = dict(meta[item["episode_index"]])
        rec["easy_score"] = item["easy_score"]
        rec["selection_score"] = item["selection_score"]
        rec["has_wrist_video"] = item["has_wrist"]
        subset_meta_lines.append(rec)
    with (output_root / "meta/episodes.jsonl").open("w") as f:
        for rec in subset_meta_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    info = dict(source_info)
    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(item["length"] for item in selected)
    info["total_tasks"] = sum(len(item["tasks"]) for item in selected)
    info["total_chunks"] = len({item["episode_index"] // 1000 for item in selected})
    info["splits"] = {"train": f"0:{len(selected)}"}
    info["source_root"] = str(source_root)
    info["selection_policy"] = {
        "requires_external_cameras": True,
        "prefers_wrist_camera": True,
        "ranking": selection_mode,
        "target_count": target_count,
    }
    (output_root / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "output_root": str(output_root),
        "source_root": str(source_root),
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "camera_requirements": {
            "required": ["observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left"],
            "preferred": ["observation.images.wrist_image_left"],
        },
        "easy_task_examples_top20": selected[:20],
        "task_frequency_top50": Counter(item["task"] for item in selected).most_common(50),
        "wrist_available_count": sum(1 for item in selected if item["has_wrist"]),
        "wrist_missing_count": sum(1 for item in selected if not item["has_wrist"]),
    }
    manifest_name = f"{selection_mode}_{target_count}_manifest.json"
    (output_root / "meta" / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    readme = (
        f"DreamZero-friendly DROID subset ({selection_mode})\n"
        "===============================================\n\n"
        f"Source: {source_root}\n"
        f"Selected episodes: {len(selected)}\n"
        f"External-camera coverage required: yes\n"
        f"Wrist-camera coverage preferred: yes ({manifest['wrist_available_count']} available)\n\n"
        "Selection heuristic:\n"
        "- keep only episodes with both external cameras so final-state success can be judged visually\n"
        "- easy mode: rank easier, shorter, more single-stage tasks higher\n"
        "- easy_dual mode: additionally favor drawers, buttons/keys/switches, and marker/pen/cup/bowl tasks where Dual was more promising\n"
        "- penalize deformable/contact-rich tasks such as rubber-band, cloth, faucet, folding, pouring\n"
        "- preserve original DreamZero-compatible parquet/video layout via symlinks\n"
    )
    (output_root / "README.txt").write_text(readme)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DreamZero-friendly DROID subset with camera coverage.")
    parser.add_argument("--source-root", type=Path, default=Path("data/droid_lerobot"))
    parser.add_argument("--output-root", type=Path, default=Path("data/droid_easy1000_dreamzero"))
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--selection-mode", type=str, default="easy", choices=["easy", "easy_dual"])
    args = parser.parse_args()

    manifest = build_subset(args.source_root, args.output_root, args.target_count, args.selection_mode)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
