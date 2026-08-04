#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pi05_libero_repro.logiv.recovery_splits import (
    DatasetRole,
    RecoverySplitError,
    append_role_allocation,
    load_recovery_manifests,
    make_role_allocation,
    preflight_role_allocation_outputs,
    preview_role_allocation_append,
    require_canonical_role_registry,
    write_role_allocation,
    write_run_registry_provenance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a complete pre-outcome role allocation and append it to the "
            "locked canonical registry with an expected-head compare-and-swap."
        )
    )
    parser.add_argument(
        "source_dataset",
        nargs="+",
        type=Path,
        help="immutable source dataset directories containing recovery_root.json",
    )
    parser.add_argument(
        "--complete-allocation",
        required=True,
        type=Path,
        help="JSON object mapping every source independence unit to one dataset role",
    )
    parser.add_argument("--allocation-version", required=True)
    parser.add_argument(
        "--canonical-registry", required=True, type=Path, help="frozen project registry path"
    )
    parser.add_argument("--registry-id", required=True)
    parser.add_argument(
        "--expected-head",
        required=True,
        help="expected current canonical registry head SHA-256",
    )
    parser.add_argument("--output-allocation", required=True, type=Path)
    parser.add_argument(
        "--run-json",
        required=True,
        type=Path,
        help="run.json written with allocation and new registry hashes before outcomes",
    )
    return parser


def _load_roles(path: Path) -> dict[str, DatasetRole]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoverySplitError("complete allocation file is unreadable") from error
    if not isinstance(payload, dict):
        raise RecoverySplitError("complete allocation must be a JSON object")
    try:
        return {unit: DatasetRole(role) for unit, role in payload.items()}
    except (TypeError, ValueError) as error:
        raise RecoverySplitError("complete allocation contains an invalid role") from error


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifests = load_recovery_manifests(args.source_dataset)
        registry = require_canonical_role_registry(
            args.canonical_registry, args.registry_id
        )
        if not registry.path.exists():
            if registry.genesis_head().head_sha256 != args.expected_head:
                raise RecoverySplitError(
                    "new canonical registry genesis does not match expected head"
                )
            registry.initialize()
        current = registry.read_head()
        allocation = make_role_allocation(
            manifests,
            roles=_load_roles(args.complete_allocation),
            allocation_version=args.allocation_version,
            registry_parent_head_sha256=args.expected_head,
        )
        predicted = preview_role_allocation_append(
            current, allocation, expected_head=args.expected_head
        )
        preflight_role_allocation_outputs(
            args.output_allocation,
            allocation,
            args.run_json,
            role_registry_head_sha256=predicted.head_sha256,
        )
        head = append_role_allocation(
            registry, allocation, expected_head=args.expected_head
        )
        if head != predicted:
            raise RecoverySplitError("committed registry head differs from preview")
        write_role_allocation(args.output_allocation, allocation)
        write_run_registry_provenance(
            args.run_json,
            source_dataset_sha256=allocation.source_dataset_sha256,
            role_allocation_sha256=allocation.allocation_sha256,
            role_registry_head_sha256=head.head_sha256,
        )
    except (OSError, RecoverySplitError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
