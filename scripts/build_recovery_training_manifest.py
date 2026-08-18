#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pi05_libero_repro.logiv.recovery_splits import (
    DatasetRole,
    RecoverySplitError,
    build_training_manifest,
    load_fresh_recovery_labels,
    load_recovery_manifests,
    load_role_allocation,
    require_canonical_role_registry,
    write_training_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an atomic, signed recovery training manifest; raw root directories "
            "are never trainer inputs."
        )
    )
    parser.add_argument(
        "source_dataset",
        nargs="+",
        type=Path,
        help="immutable source dataset directories containing recovery_root.json",
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[role.value for role in DatasetRole],
        help="required immutable dataset role",
    )
    parser.add_argument("--fresh-labels", required=True, type=Path)
    parser.add_argument("--source-dataset-sha256", required=True)
    parser.add_argument("--role-allocation", required=True, type=Path)
    parser.add_argument("--role-allocation-sha256", required=True)
    parser.add_argument("--role-registry", required=True, type=Path)
    parser.add_argument("--role-registry-id", required=True)
    parser.add_argument("--role-registry-head-sha256", required=True)
    parser.add_argument("--builder-version", required=True)
    parser.add_argument(
        "--output-manifest",
        required=True,
        type=Path,
        help="atomic output training manifest path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifests = load_recovery_manifests(args.source_dataset)
        labels = load_fresh_recovery_labels(args.fresh_labels)
        allocation = load_role_allocation(args.role_allocation)
        if allocation.allocation_sha256 != args.role_allocation_sha256:
            raise RecoverySplitError("required role allocation hash mismatch")
        registry = require_canonical_role_registry(
            args.role_registry, args.role_registry_id
        )
        head = registry.read_head()
        if head.head_sha256 != args.role_registry_head_sha256:
            raise RecoverySplitError("required canonical registry head hash mismatch")
        manifest = build_training_manifest(
            manifests,
            labels,
            role=DatasetRole(args.role),
            source_dataset_sha256=args.source_dataset_sha256,
            role_allocation=allocation,
            role_registry_head=head,
            builder_version=args.builder_version,
        )
        write_training_manifest(args.output_manifest, manifest)
    except (OSError, RecoverySplitError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
