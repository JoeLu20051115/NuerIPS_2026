#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data/agibot_heldout1000_for_droid/manifest.json"
REPO_ID = "agibot-world/AgiBotWorldChallenge-2025"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-task-ids", type=int, default=None, help="Download only the first N selected task_ids for warmup/testing.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--local-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["selected_task_ids"]
    if args.max_task_ids is not None:
        selected = selected[: args.max_task_ids]

    allow_patterns = ["README.md"]
    for row in selected:
        for archive_path in row["archives"]:
            normalized = archive_path
            if normalized.startswith("data/agibot_challenge_meta/"):
                normalized = normalized[len("data/agibot_challenge_meta/") :]
            allow_patterns.append(normalized)
        task_group = row["task_group"]
        allow_patterns.extend(
            [
                f"Manipulation-SimData/{task_group}/task_train.json",
                f"Manipulation-SimData/{task_group}/task_train_1st_part.json",
                f"Manipulation-SimData/{task_group}/task_train_2nd_part.json",
            ]
        )

    local_dir = args.local_dir.resolve() if args.local_dir else manifest_path.parent / "raw_archives"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded selected AgiBot archives to {local_dir}")


if __name__ == "__main__":
    main()
