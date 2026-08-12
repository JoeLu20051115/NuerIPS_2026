#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


GPU_TASKS = (
    ("handover_block", "stamp_seal", "turn_switch", "beat_block_hammer"),
    ("open_microwave", "blocks_ranking_size", "stack_blocks_three"),
    ("place_dual_shoes", "move_can_pot", "stack_bowls_three"),
)


def _stall_observations(config: dict, task: str) -> int:
    per_task = config.get("base_stall_observations_by_task", {})
    value = int(per_task.get(task, config.get("base_stall_observations", 4)))
    if value < 1:
        raise ValueError(f"invalid base stall observations for {task}: {value}")
    return value


def _stage_stall_observations(config: dict, task: str) -> list[int] | None:
    values = config.get("stage_stall_observations_by_task", {}).get(task)
    if values is None:
        return None
    parsed = [int(value) for value in values]
    if not parsed or any(value < 1 for value in parsed):
        raise ValueError(f"invalid stage stall observations for {task}: {parsed}")
    return parsed


def _min_base_dispatches(config: dict, task: str) -> int:
    steps = int(config.get("min_base_steps_by_task", {}).get(task, 0))
    chunk = int(config["action_chunk_steps"])
    if steps < 0:
        raise ValueError(f"invalid minimum base steps for {task}: {steps}")
    return (steps + chunk - 1) // chunk


def _repair_action_chunk_steps(config: dict) -> int:
    return int(config.get("repair_action_chunk_steps", config["action_chunk_steps"]))


def _vlm_image_detail(config: dict) -> str:
    return str(config.get("vlm_image_detail", "low"))


def _api_key() -> str:
    existing = os.environ.get("OPENAI_API_KEY", "")
    if existing.startswith("sk-") and len(existing) > 20:
        return existing
    path = Path.home() / ".cline/data/secrets.json"
    try:
        value = json.loads(path.read_text())["openRouterApiKey"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("no valid OpenAI API key source found") from error
    if not isinstance(value, str) or not value.startswith("sk-") or len(value) <= 20:
        raise RuntimeError("OpenAI API key source is invalid")
    return value


def _run_worker(gpu: int, tasks: tuple[str, ...], args: argparse.Namespace) -> int:
    config = json.loads(args.protocol.read_text())
    robotwin = args.taco / "third_party" / "Robotwin"
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = _api_key()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            ".",
            "../lerobot/src",
            str(args.taco),
            str(args.logiv_root / "src"),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        log = args.output / f"{task}.log"
        command = [
            str(args.python),
            "script/eval_lerobot_torch_pi05.py",
            "--config", "policy/pi05/deploy_policy.yml",
            "--overrides",
            "--policy_name", "pi05",
            "--task_name", task,
            "--task_config", config["task_config"],
            "--ckpt_setting", "unified_50tasks",
            "--seed", "0",
            "--tag", args.tag,
            "--instruction_type", config["instruction_type"],
            "--policy_path", config["checkpoint"],
            "--test_num", str(len(config["tasks"][task])),
            "--tokenizer_path", str(args.tokenizer),
            "--record_videos", "False",
            "--accepted_seeds", json.dumps(config["tasks"][task]),
            "--logiv_root", str(args.logiv_root),
            "--val_binary", str(args.val_binary),
            "--action_chunk_steps", str(config["action_chunk_steps"]),
            "--repair_action_chunk_steps", str(_repair_action_chunk_steps(config)),
            "--vlm_image_detail", _vlm_image_detail(config),
            "--max_gpt4o_retries", "2",
            "--base_stall_observations", str(_stall_observations(config, task)),
            "--min_base_dispatches", str(_min_base_dispatches(config, task)),
        ]
        instructions = config.get("instructions", {}).get(task)
        if instructions is not None:
            command.extend(["--accepted_instructions", json.dumps(instructions)])
        stage_stalls = _stage_stall_observations(config, task)
        if stage_stalls is not None:
            command.extend(["--stage_stall_observations", json.dumps(stage_stalls)])
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=robotwin,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=range(3), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--taco", type=Path, required=True)
    parser.add_argument("--logiv-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-binary", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    tasks = tuple(args.tasks) if args.tasks else GPU_TASKS[args.worker]
    unknown = set(tasks) - set(GPU_TASKS[args.worker])
    if unknown:
        parser.error(
            f"worker {args.worker} cannot run tasks: {', '.join(sorted(unknown))}"
        )
    return _run_worker(args.worker, tasks, args)


if __name__ == "__main__":
    sys.exit(main())
