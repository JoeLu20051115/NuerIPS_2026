#!/usr/bin/env python3
"""Inject AgiBot stats into the DreamZero checkpoint metadata.json.

Usage:
    python scripts/eval/inject_agibot_stats_into_metadata.py \
        --stats-json /tmp/agibot_challenge_stats.json \
        --metadata-json checkpoints/DreamZero-DROID/experiment_cfg/metadata.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-json", type=Path, default=Path("/tmp/agibot_challenge_stats.json"))
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("checkpoints/DreamZero-DROID/experiment_cfg/metadata.json"),
    )
    args = parser.parse_args()

    with open(args.stats_json) as f:
        agibot_stats = json.load(f)

    with open(args.metadata_json) as f:
        metadata = json.load(f)

    if "agibot" in metadata:
        print("WARNING: 'agibot' key already exists in metadata.json — overwriting.")

    # Backup
    backup = args.metadata_json.with_suffix(".json.bak")
    shutil.copy(args.metadata_json, backup)
    print(f"Backed up original metadata to {backup}")

    metadata["agibot"] = agibot_stats

    with open(args.metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Injected 'agibot' entry into {args.metadata_json}")
    print(f"Keys now in metadata.json: {list(metadata.keys())}")


if __name__ == "__main__":
    main()
