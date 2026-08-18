#!/usr/bin/env python3
"""Compute AgiBot normalization statistics from competition training data.

Outputs an 'agibot' entry suitable for injection into
checkpoints/DreamZero-DROID/experiment_cfg/metadata.json.

Usage:
    python scripts/eval/compute_agibot_challenge_stats.py \
        --train-root data/agibot_challenge_2026/train \
        --out-json /tmp/agibot_stats.json \
        --max-episodes 5000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def load_episode_states(h5_path: Path) -> dict[str, np.ndarray] | None:
    try:
        with h5py.File(h5_path, "r") as f:
            joint = f["state/joint/position"][:]        # (T, 14)
            effector = f["state/effector/position"][:]   # (T, 2)
            head = f["state/head/position"][:]           # (T, 2)
            waist = f["state/waist/position"][:]         # (T, 2)
    except Exception:
        return None

    if joint.shape[1] < 14 or effector.shape[1] < 2:
        return None

    return {
        "left_arm_joint_position":  joint[:, :7].astype(np.float32),    # (T, 7)
        "right_arm_joint_position": joint[:, 7:14].astype(np.float32),  # (T, 7)
        "left_effector_position":   effector[:, 0:1].astype(np.float32),# (T, 1)
        "right_effector_position":  effector[:, 1:2].astype(np.float32),# (T, 1)
        "head_position":            head.astype(np.float32),             # (T, 2)
        "waist_pitch":              waist[:, 0:1].astype(np.float32),   # (T, 1)
        "waist_lift":               waist[:, 1:2].astype(np.float32),   # (T, 1)
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(arrays: list[np.ndarray]) -> dict:
    """Concatenate all timesteps and compute per-dim statistics."""
    data = np.concatenate(arrays, axis=0)  # (N, D)
    return {
        "min":  data.min(axis=0).tolist(),
        "max":  data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std":  data.std(axis=0).tolist(),
        "q01":  np.percentile(data, 1, axis=0).tolist(),
        "q99":  np.percentile(data, 99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AgiBot challenge normalization stats.")
    parser.add_argument("--train-root", type=Path, default=Path("data/agibot_challenge_2026/train"))
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/agibot_challenge_stats.json"))
    parser.add_argument("--max-episodes", type=int, default=5000, help="Max episodes to sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Collect all h5 paths
    h5_paths = sorted(args.train_root.glob("*/proprio_stats.h5"))
    print(f"Found {len(h5_paths)} training episodes under {args.train_root}")
    if len(h5_paths) == 0:
        raise FileNotFoundError(f"No proprio_stats.h5 found under {args.train_root}")

    if len(h5_paths) > args.max_episodes:
        h5_paths = random.sample(h5_paths, args.max_episodes)
        print(f"Sampled {args.max_episodes} episodes for stats computation.")

    # Accumulate per-field arrays
    field_names = [
        "left_arm_joint_position",
        "right_arm_joint_position",
        "left_effector_position",
        "right_effector_position",
        "head_position",
        "waist_pitch",
        "waist_lift",
    ]
    accum: dict[str, list[np.ndarray]] = {k: [] for k in field_names}

    loaded = 0
    for h5_path in h5_paths:
        states = load_episode_states(h5_path)
        if states is None:
            continue
        for k in field_names:
            accum[k].append(states[k])
        loaded += 1
        if loaded % 500 == 0:
            print(f"  Loaded {loaded}/{len(h5_paths)} episodes...")

    print(f"Successfully loaded {loaded} episodes.")

    # Compute stats
    state_stats: dict[str, dict] = {}
    for k in field_names:
        if not accum[k]:
            print(f"WARNING: no data for {k}, using zeros")
            continue
        state_stats[k] = compute_stats(accum[k])
        print(f"  {k}: shape {accum[k][0].shape[1:]}, "
              f"q01={[f'{v:.3f}' for v in state_stats[k]['q01']]}, "
              f"q99={[f'{v:.3f}' for v in state_stats[k]['q99']]}")

    # robot_velocity: not in competition data, use identity-range stats (zeros mean)
    robot_vel_dim = 2
    robot_velocity_stats = {
        "min":  [-2.0] * robot_vel_dim,
        "max":  [2.0] * robot_vel_dim,
        "mean": [0.0] * robot_vel_dim,
        "std":  [1.0] * robot_vel_dim,
        "q01":  [-1.0] * robot_vel_dim,
        "q99":  [1.0] * robot_vel_dim,
    }

    # Build metadata structure matching DatasetMetadata schema
    agibot_metadata = {
        "statistics": {
            "state": state_stats,
            "action": {
                **state_stats,          # same distribution for state and action
                "robot_velocity": robot_velocity_stats,
            },
        },
        "modalities": {
            "state": {
                "left_arm_joint_position":  {"absolute": True, "rotation_type": None, "shape": [7],  "continuous": True},
                "right_arm_joint_position": {"absolute": True, "rotation_type": None, "shape": [7],  "continuous": True},
                "left_effector_position":   {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "right_effector_position":  {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "head_position":            {"absolute": True, "rotation_type": None, "shape": [2],  "continuous": True},
                "waist_pitch":              {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "waist_lift":               {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
            },
            "action": {
                "left_arm_joint_position":  {"absolute": True, "rotation_type": None, "shape": [7],  "continuous": True},
                "right_arm_joint_position": {"absolute": True, "rotation_type": None, "shape": [7],  "continuous": True},
                "left_effector_position":   {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "right_effector_position":  {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "head_position":            {"absolute": True, "rotation_type": None, "shape": [2],  "continuous": True},
                "waist_pitch":              {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "waist_lift":               {"absolute": True, "rotation_type": None, "shape": [1],  "continuous": True},
                "robot_velocity":           {"absolute": True, "rotation_type": None, "shape": [2],  "continuous": True},
            },
            "video": {
                "top_head":   {"resolution": [320, 176], "channels": 3, "fps": 30.0},
                "hand_left":  {"resolution": [320, 176], "channels": 3, "fps": 30.0},
                "hand_right": {"resolution": [320, 176], "channels": 3, "fps": 30.0},
            },
        },
        "embodiment_tag": "agibot",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(agibot_metadata, f, indent=2)
    print(f"\nSaved AgiBot stats to {args.out_json}")
    print("Next: inject into metadata.json with inject_agibot_stats_into_metadata.py")


if __name__ == "__main__":
    main()
