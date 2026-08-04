#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pi05_libero_repro.logiv.recovery_splits import (
    RecoverySplitError,
    load_recovery_manifests,
    validate_recovery_splits,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively load recovery_root.json artifacts and reject split leakage."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="dataset directories containing recovery_root.json files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = validate_recovery_splits(
            load_recovery_manifests(args.directories)
        )
    except (OSError, RecoverySplitError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
