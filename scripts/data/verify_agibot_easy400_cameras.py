#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid/extracted"
OUTPUT_PATH = REPO_ROOT / "data/agibot_easy400_for_droid/meta/camera_validation.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, default=EXTRACT_ROOT)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = []
    for episode_dir in sorted(args.extract_root.glob("*/*/*/*/*")):
        if not episode_dir.is_dir():
            continue
        head = episode_dir / "head_color.mp4"
        hand_right = episode_dir / "hand_right_color.mp4"
        hand_left = episode_dir / "hand_left_color.mp4"
        h5 = episode_dir / "aligned_joints.h5"
        data_info = episode_dir / "data_info.json"
        status = {
            "episode_dir": _display_path(episode_dir),
            "has_head": head.exists(),
            "has_hand_right": hand_right.exists(),
            "has_hand_left": hand_left.exists(),
            "has_aligned_joints": h5.exists(),
            "has_data_info": data_info.exists(),
        }
        status["has_all_required"] = all(
            [status["has_head"], status["has_hand_right"], status["has_hand_left"], status["has_aligned_joints"], status["has_data_info"]]
        )
        rows.append(status)

    summary = {
        "num_episode_dirs": len(rows),
        "num_all_required": sum(1 for r in rows if r["has_all_required"]),
        "num_missing_any": sum(1 for r in rows if not r["has_all_required"]),
        "episodes": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
