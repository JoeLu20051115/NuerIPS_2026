#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data/agibot_easy400_for_droid/manifest.json"
DEFAULT_EXTRACT_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid/extracted"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "data/agibot_easy400_for_droid/raw_archives"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--max-task-ids", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest_path.read_text())
    selected = manifest["selected_task_ids"]
    if args.max_task_ids is not None:
        selected = selected[: args.max_task_ids]

    args.extract_root.mkdir(parents=True, exist_ok=True)
    done = []

    for row in selected:
        task_group = row["task_group"]
        for archive_rel in row["archives"]:
            archive_path = (REPO_ROOT / archive_rel).resolve()
            # Prefer the actual downloaded archive under raw_archives over the
            # original challenge_meta LFS pointer path.
            if "data/agibot_challenge_meta/" in archive_rel:
                normalized = archive_rel.replace("data/agibot_challenge_meta/", "", 1)
                downloaded_path = (args.archive_root / normalized).resolve()
                if downloaded_path.exists():
                    archive_path = downloaded_path
            archive_name = archive_path.stem
            target_dir = args.extract_root / task_group
            target_dir.mkdir(parents=True, exist_ok=True)
            marker = target_dir / f".{archive_name}.done"
            if marker.exists():
                print(f"skip {archive_name} (already extracted)")
                continue
            print(f"extract {archive_path} -> {target_dir}")
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=target_dir)
            marker.write_text("ok\n")
            done.append({"archive": _display_path(archive_path), "target_dir": _display_path(target_dir)})

    print(json.dumps({"extracted_count": len(done), "extract_root": _display_path(args.extract_root)}, indent=2))


if __name__ == "__main__":
    main()
