#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pi05_libero_repro.artifacts import verify_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a deterministic artifact manifest.")
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    if args.command == "create":
        write_manifest(args.root, args.manifest)
        return 0

    errors = verify_manifest(args.root, args.manifest)
    if errors:
        print(*errors, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
