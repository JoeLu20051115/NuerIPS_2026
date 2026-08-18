#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def _path_piece(key: object) -> str:
    for attribute in ("key", "idx", "name"):
        if hasattr(key, attribute):
            return str(getattr(key, attribute))
    return str(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot effective π₀.₅ config and parameter tree")
    parser.add_argument("--checkpoint-name", required=True, choices=("full", "early"))
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    openpi_dir = repo_root / "external_repos" / "openpi"
    sys.path.insert(0, str(openpi_dir / "src"))

    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.training import config as config_module

    revision = subprocess.check_output(["git", "-C", str(openpi_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != "650c5b0283a49c42784fb5055a0507da2c6d347d":
        raise RuntimeError(f"OpenPI revision mismatch: {revision}")

    train_config = config_module.get_config("pi05_libero")
    model_config = train_config.model
    effective = {
        "action_dim": model_config.action_dim,
        "action_expert_variant": model_config.action_expert_variant,
        "action_horizon": model_config.action_horizon,
        "discrete_state_input": model_config.discrete_state_input,
        "dtype": model_config.dtype,
        "max_token_len": model_config.max_token_len,
        "paligemma_variant": model_config.paligemma_variant,
        "pi05": model_config.pi05,
    }
    expected = {
        "action_dim": 32,
        "action_expert_variant": "gemma_300m",
        "action_horizon": 10,
        "discrete_state_input": False,
        "dtype": "bfloat16",
        "max_token_len": 200,
        "paligemma_variant": "gemma_2b",
        "pi05": True,
    }
    if effective != expected:
        raise RuntimeError(f"unexpected pi05_libero config: {effective}")

    params = model_module.restore_params(args.checkpoint_dir.resolve() / "params", dtype=jnp.bfloat16)
    leaves = []
    total_elements = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        shape = [int(value) for value in leaf.shape]
        elements = math.prod(shape)
        total_elements += elements
        leaves.append(
            {
                "dtype": str(leaf.dtype),
                "elements": elements,
                "path": "/".join(_path_piece(piece) for piece in path),
                "shape": shape,
            }
        )
    leaves.sort(key=lambda item: item["path"])
    manifest_hash = hashlib.sha256(args.checkpoint_manifest.read_bytes()).hexdigest()
    payload = {
        "checkpoint": args.checkpoint_name,
        "checkpoint_manifest_sha256": manifest_hash,
        "config": effective,
        "openpi_revision": revision,
        "parameter_leaf_count": len(leaves),
        "parameter_total_elements": total_elements,
        "parameters": leaves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
