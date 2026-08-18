#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def iter_episodes(root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for episode_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            rows.append((task_dir.name, episode_dir.name, episode_dir))
    return rows


def ensure_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a first-N symlinked subset of a WorldModel prediction directory.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    episodes = iter_episodes(args.source_root)[: args.num_episodes]
    args.dest_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for idx, (task_id, episode_id, episode_dir) in enumerate(episodes, start=1):
        dest_episode_dir = args.dest_root / task_id / episode_id
        ensure_symlink(episode_dir, dest_episode_dir)
        manifest_rows.append(
            {
                "index": idx,
                "task_id": task_id,
                "episode_id": episode_id,
                "source": str(episode_dir),
                "dest": str(dest_episode_dir),
            }
        )

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_rows, f, indent=2)

    print(f"Created subset with {len(manifest_rows)} episodes at {args.dest_root}")


if __name__ == "__main__":
    main()
