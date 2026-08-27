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
ALL_TASKS = frozenset(task for tasks in GPU_TASKS for task in tasks)


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


def _task_option(config: dict, name: str, task: str | None, default=None):
    if task is not None and task in config.get(f"{name}_by_task", {}):
        return config[f"{name}_by_task"][task]
    return config.get(name, default)


def _repair_action_chunk_steps(config: dict, task: str | None = None) -> int:
    value = _task_option(config, "repair_action_chunk_steps", task)
    return int(config["action_chunk_steps"] if value is None else value)


def _vlm_image_detail(config: dict, task: str | None = None) -> str:
    return str(_task_option(config, "vlm_image_detail", task, "low"))


def _dag_from_start(config: dict, task: str | None = None) -> bool:
    return bool(_task_option(config, "dag_from_start", task, False))


def _use_registered_dag_prompts(config: dict, task: str | None = None) -> bool:
    return bool(_task_option(config, "use_registered_dag_prompts", task, False))


def _preserve_original_repair_prompt(
    config: dict, task: str | None = None
) -> bool:
    return bool(
        _task_option(config, "preserve_original_repair_prompt", task, False)
    )


def _policy_replan_steps(config: dict, task: str | None = None) -> int | None:
    value = _task_option(config, "policy_replan_steps", task)
    return None if value is None else int(value)


def _repair_cfn_path(
    config: dict, task: str, checkpoint: Path | None = None
) -> Path | None:
    if task not in config.get("repair_cfn_tasks", ()):
        return None
    root = checkpoint
    if root is None:
        root = Path(config["checkpoint"])
    return root / "cfns" / f"{task}_cfn.pt"


def _api_key() -> str:
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        value = os.environ.get(name, "")
        if value.startswith("sk-") and len(value) > 20:
            return value
    raise RuntimeError(
        "set a valid OPENAI_API_KEY (or OPENAI_KEY) for the direct OpenAI API"
    )


def _task_command(
    config: dict, task: str, args: argparse.Namespace
) -> list[str]:
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
        "--policy_path", str(args.checkpoint),
        "--test_num", str(len(config["tasks"][task])),
        "--tokenizer_path", str(args.tokenizer),
        "--record_videos", "False",
        "--accepted_seeds", json.dumps(config["tasks"][task]),
        "--logiv_root", str(args.logiv_root),
        "--val_binary", str(args.val_binary),
        "--action_chunk_steps", str(config["action_chunk_steps"]),
        "--repair_action_chunk_steps", str(_repair_action_chunk_steps(config, task)),
        "--vlm_image_detail", _vlm_image_detail(config, task),
        "--max_gpt4o_retries", "2",
        "--base_stall_observations", str(_stall_observations(config, task)),
        "--min_base_dispatches", str(_min_base_dispatches(config, task)),
        "--dag_from_start", str(_dag_from_start(config, task)),
        "--use_registered_dag_prompts",
        str(_use_registered_dag_prompts(config, task)),
        "--preserve_original_repair_prompt",
        str(_preserve_original_repair_prompt(config, task)),
    ]
    instructions = config.get("instructions", {}).get(task)
    if instructions is not None:
        command.extend(["--accepted_instructions", json.dumps(instructions)])
    stage_stalls = _stage_stall_observations(config, task)
    if stage_stalls is not None:
        command.extend(["--stage_stall_observations", json.dumps(stage_stalls)])
    repair_cfn = _repair_cfn_path(config, task, args.checkpoint)
    if repair_cfn is not None:
        command.extend(["--repair_cfn_path", str(repair_cfn)])
    policy_replan_steps = _policy_replan_steps(config, task)
    if policy_replan_steps is not None:
        command.extend(["--policy_replan_steps", str(policy_replan_steps)])
    return command


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
            str(args.taco / "cfn"),
            str(args.logiv_root / "src"),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        log = args.output / f"{task}.log"
        command = _task_command(config, task, args)
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
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--taco", type=Path, required=True)
    parser.add_argument("--logiv-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--val-binary", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    tasks = tuple(args.tasks) if args.tasks else GPU_TASKS[args.worker]
    unknown = set(tasks) - ALL_TASKS
    if unknown:
        parser.error(f"unknown tasks: {', '.join(sorted(unknown))}")
    gpu = args.worker if args.gpu is None else args.gpu
    return _run_worker(gpu, tasks, args)


if __name__ == "__main__":
    sys.exit(main())
