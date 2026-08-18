#!/usr/bin/env python3
"""
Build a runnable LIBERO subset manifest for DreamZero experiments.

This script is intentionally conservative:
- it can skip tasks with zero init states
- it can optionally instantiate the environment to filter out broken tasks
- it expands a task subset into concrete (task_id, init_state_id) episodes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


def patch_torch_load_weights_only() -> None:
    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load


def parse_task_ids(task_ids: str, num_tasks: int) -> list[int]:
    if task_ids.strip():
        return [int(x) for x in task_ids.split(",") if x.strip()]
    return list(range(int(num_tasks)))


def task_is_runnable(task_suite, task_id: int, verify_env: bool) -> tuple[bool, str | None]:
    task = task_suite.get_task(task_id)
    try:
        init_states = task_suite.get_task_init_states(task_id)
    except Exception as exc:
        return False, f"init_state_load_failed:{type(exc).__name__}:{exc}"

    if len(init_states) == 0:
        return False, "no_init_states"

    if not verify_env:
        return True, None

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    try:
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=64,
            camera_widths=64,
            controller="JOINT_POSITION",
        )
        env.close()
        return True, None
    except Exception as exc:
        return False, f"env_init_failed:{type(exc).__name__}:{exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a runnable LIBERO subset manifest")
    parser.add_argument("--suite", type=str, default="libero_10", help="LIBERO suite name")
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Comma-separated task ids to consider; empty means use the first --num-tasks tasks",
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="How many leading tasks to consider when --task-ids is empty")
    parser.add_argument("--episodes-per-task", type=int, default=10, help="How many init states to include per kept task")
    parser.add_argument(
        "--verify-env",
        action="store_true",
        help="Instantiate each task environment once and drop tasks that fail to initialize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results_dualsystem/libero_subset_manifest.json",
        help="Output manifest path",
    )
    args = parser.parse_args()

    patch_torch_load_weights_only()

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_key = args.suite.lower()
    if suite_key in benchmark_dict:
        task_suite = benchmark_dict[suite_key]()
    elif hasattr(benchmark, "task_maps") and suite_key in benchmark.task_maps:
        task_suite = benchmark.Benchmark()
        task_suite.name = suite_key
        task_suite._make_benchmark()
    else:
        raise ValueError(f"Unknown LIBERO suite: {args.suite}")

    candidate_task_ids = parse_task_ids(args.task_ids, min(args.num_tasks, task_suite.n_tasks))
    kept_tasks: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    for task_id in candidate_task_ids:
        task = task_suite.get_task(task_id)
        runnable, reason = task_is_runnable(task_suite, task_id, args.verify_env)
        if not runnable:
            skipped_tasks.append(
                {
                    "task_id": int(task_id),
                    "language_instruction": task.language,
                    "skip_reason": reason,
                }
            )
            continue

        init_states = task_suite.get_task_init_states(task_id)
        capped_num_episodes = min(len(init_states), int(args.episodes_per_task))
        kept_tasks.append(
            {
                "task_id": int(task_id),
                "language_instruction": task.language,
                "available_init_states": int(len(init_states)),
                "selected_init_states": list(range(capped_num_episodes)),
            }
        )
        for init_state_id in range(capped_num_episodes):
            episodes.append(
                {
                    "task_id": int(task_id),
                    "init_state_id": int(init_state_id),
                    "language_instruction": task.language,
                }
            )

    summary = {
        "suite": args.suite,
        "candidate_task_ids": candidate_task_ids,
        "episodes_per_task": int(args.episodes_per_task),
        "verify_env": bool(args.verify_env),
        "num_kept_tasks": int(len(kept_tasks)),
        "num_skipped_tasks": int(len(skipped_tasks)),
        "num_episodes": int(len(episodes)),
        "kept_tasks": kept_tasks,
        "skipped_tasks": skipped_tasks,
        "episodes": episodes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
