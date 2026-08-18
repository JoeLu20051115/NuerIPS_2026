#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pi05_libero_repro.records import load_records
from pi05_libero_repro.report import build_report, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit full and early π₀.₅ LIBERO results")
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--early", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    report = build_report(load_records(args.full) + load_records(args.early))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(render_markdown(report))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
