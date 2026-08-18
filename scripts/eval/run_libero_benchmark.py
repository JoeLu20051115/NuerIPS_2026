#!/usr/bin/env python3
"""
Run a small zero-shot DreamZero benchmark on LIBERO tasks.

This script connects to the existing DreamZero websocket policy server and
evaluates zero-shot rollouts on a subset of LIBERO tasks using the
JOINT_POSITION controller for compatibility with DreamZero's 8D action output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from eval_utils.policy_client import WebsocketClientPolicy


def patch_torch_load_weights_only() -> None:
    """LIBERO task init states require the pre-2.6 torch.load behavior."""
    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load


def build_request(obs: Dict[str, Any], instruction: str, session_id: str) -> Dict[str, Any]:
    """Convert LIBERO observation into the DreamZero websocket request format."""
    agentview = cv2.resize(obs["agentview_image"], (320, 180))
    wrist = cv2.resize(obs["robot0_eye_in_hand_image"], (320, 180))
    joint_position = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
    gripper_position = np.asarray(obs["robot0_gripper_qpos"][:1], dtype=np.float64)
    return {
        "observation/exterior_image_0_left": agentview,
        # Duplicate the single global view so the 3-camera DreamZero server
        # receives the expected keys.
        "observation/exterior_image_1_left": agentview,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_position,
        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
        "observation/gripper_position": gripper_position,
        "prompt": instruction,
        "session_id": session_id,
    }


class LLMPlanner:
    """Minimal LLM planner reused for LIBERO subtask prompting."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        cache_path: str | None = None,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.use_mock = not self.api_key
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                self.cache = {}

    def plan(self, task_description: str) -> Tuple[List[str], Dict[str, Any]]:
        cached = self.cache.get(task_description)
        if cached:
            return list(cached["sub_instructions"]), {
                "planner_mode": "cache",
                "api_called": False,
                "api_success": True,
                "error": None,
                "model": cached.get("model", self.model_name),
                "raw_response": cached.get("raw_response"),
            }
        if self.use_mock:
            return self._mock_plan(task_description), {
                "planner_mode": "mock",
                "api_called": False,
                "api_success": False,
                "error": "missing_openai_api_key",
                "model": self.model_name,
                "raw_response": None,
            }

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a robot task planner. Break a household manipulation task into "
                            "3-5 short stage-level sub-tasks that fully cover the whole task. "
                            "Prefer one complete sub-task per object interaction, not tiny motion primitives. "
                            "Every key object and final placement target mentioned in the task must appear in the plan. "
                            "Output one sub-task per line with no numbering."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task_description}\n"
                            "Robot: Franka arm with gripper\n"
                            "Available actions: move, grasp, place, open, close\n"
                            "Return 3-5 complete stage-level sub-tasks that preserve every important object and target."
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            raw_sub_instructions = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
            sub_instructions = compress_subtasks(task_description, raw_sub_instructions) or [task_description]
            if self.cache_path:
                self.cache[task_description] = {
                    "sub_instructions": sub_instructions,
                    "model": self.model_name,
                    "raw_response": content,
                }
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=2))
            return sub_instructions, {
                "planner_mode": "real_api",
                "api_called": True,
                "api_success": True,
                "error": None,
                "model": self.model_name,
                "raw_response": content,
            }
        except Exception as exc:
            return self._mock_plan(task_description), {
                "planner_mode": "fallback_mock",
                "api_called": True,
                "api_success": False,
                "error": str(exc),
                "model": self.model_name,
                "raw_response": None,
            }

    @staticmethod
    def _mock_plan(task: str) -> List[str]:
        task_lower = task.lower()
        if "put" in task_lower or "place" in task_lower:
            return ["move to object", "grasp object", "move to target", "place object", "open gripper"]
        if "turn on" in task_lower or "turn off" in task_lower:
            return ["move to switch", "grasp switch", "toggle switch", "release switch"]
        return [task]


def subtask_for_step(sub_instructions: List[str], step_idx: int, max_steps: int) -> str:
    """Assign a subtask by time along the rollout horizon."""
    if not sub_instructions:
        return ""
    denom = max(1, int(max_steps))
    sub_task_idx = min(int(step_idx * len(sub_instructions) / denom), len(sub_instructions) - 1)
    return sub_instructions[sub_task_idx]


def should_enable_system2(task_instruction: str) -> bool:
    """Conservative gate: only invoke System-2 for clearly multi-stage tasks."""
    text = task_instruction.lower().strip()
    connectors = [" and ", " then ", " after ", " before ", " while "]
    if any(tok in text for tok in connectors):
        return True
    if len(text.split()) >= 11:
        return True
    return False


def extract_coverage_terms(task_instruction: str) -> List[str]:
    """Extract key object / target phrases that the subtask set should preserve."""
    text = re.sub(r"\s+", " ", task_instruction.lower().strip(" ."))
    patterns = [
        r"put both the (?P<obj1>.+?) and the (?P<obj2>.+?) in the (?P<target>.+)",
        r"put both the (?P<obj1>.+?) and the (?P<obj2>.+?) into the (?P<target>.+)",
        r"put the (?P<obj1>.+?) in the (?P<target1>.+?) and close it",
        r"put the (?P<obj1>.+?) on the (?P<target1>.+?) and put the (?P<obj2>.+?) on the (?P<target2>.+)",
        r"pick up the (?P<obj1>.+?) and place it in the (?P<target>.+)",
        r"turn on the (?P<obj1>.+?) and put the (?P<obj2>.+?) on it",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return [value.strip() for value in match.groupdict().values() if value]

    phrases = re.findall(r"(?:the|both the)\s+([a-z0-9][a-z0-9\s-]*?)(?=\s+(?:and|in|into|on|to|of|with|$))", text)
    deduped: List[str] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if phrase and phrase not in deduped:
            deduped.append(phrase)
    return deduped


def compress_subtasks(task_instruction: str, sub_instructions: List[str], max_subtasks: int = 5) -> List[str]:
    """Compress verbose primitive plans into a small, coverage-preserving stage plan."""
    clean = [s.strip(" .") for s in sub_instructions if isinstance(s, str) and s.strip()]
    if not clean:
        return [task_instruction]

    compressed: List[str] = []
    i = 0
    while i < len(clean):
        current = clean[i]
        nxt = clean[i + 1] if i + 1 < len(clean) else None
        current_lower = current.lower()
        nxt_lower = nxt.lower() if nxt else ""

        if nxt and current_lower.startswith("move to ") and nxt_lower.startswith("grasp "):
            obj = nxt[6:].strip()
            compressed.append(f"pick up {obj}")
            i += 2
            continue
        if nxt and current_lower.startswith("move to ") and nxt_lower.startswith("place "):
            compressed.append(nxt)
            i += 2
            continue
        if nxt and current_lower.startswith("move to ") and nxt_lower.startswith(("open ", "close ", "toggle ", "release ")):
            compressed.append(nxt)
            i += 2
            continue

        compressed.append(current)
        i += 1

    normalized: List[str] = []
    for step in compressed:
        step = re.sub(r"\s+", " ", step.strip())
        if not normalized or normalized[-1].lower() != step.lower():
            normalized.append(step)

    if len(normalized) <= max_subtasks:
        return normalized

    coverage_terms = extract_coverage_terms(task_instruction)
    coverage_preserving = [
        step
        for step in normalized
        if any(term in step.lower() for term in coverage_terms)
    ]
    if len(coverage_preserving) >= 2:
        normalized = coverage_preserving

    while len(normalized) > max_subtasks:
        best_idx = None
        best_score = None
        for idx in range(len(normalized) - 1):
            merged = f"{normalized[idx]}; {normalized[idx + 1]}"
            merged_terms = sum(1 for term in coverage_terms if term in merged.lower())
            merged_penalty = len(merged.split())
            score = (-merged_terms, merged_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            normalized = normalized[:max_subtasks]
            break
        normalized = (
            normalized[:best_idx]
            + [f"{normalized[best_idx]}; {normalized[best_idx + 1]}"]
            + normalized[best_idx + 2 :]
        )

    return normalized


def validate_subtasks(task_instruction: str, sub_instructions: List[str]) -> bool:
    """Reject weak plans and safely fall back to the original task prompt."""
    clean = [s.strip() for s in sub_instructions if isinstance(s, str) and s.strip()]
    if not (3 <= len(clean) <= 5):
        return False
    if len(set(s.lower() for s in clean)) < len(clean) - 1:
        return False
    if any(len(s.split()) > 20 for s in clean):
        return False
    if clean == [task_instruction]:
        return False
    coverage_terms = extract_coverage_terms(task_instruction)
    if coverage_terms:
        joined = " ".join(clean).lower()
        if any(term not in joined for term in coverage_terms):
            return False
    return True


def compose_prompt(task_instruction: str, current_instruction: str, prompt_mode: str) -> str:
    if prompt_mode == "task":
        return task_instruction
    if prompt_mode == "task_token_only":
        return current_instruction
    if prompt_mode == "subtask":
        return current_instruction
    if prompt_mode == "hybrid":
        return f"{task_instruction}; {current_instruction}"
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def heuristic_plan(task_instruction: str) -> List[str]:
    text = task_instruction.strip()
    lower = text.lower()
    both_match = re.fullmatch(r"put both the (.+?) and the (.+?) in the (.+)", lower)
    if both_match:
        obj1, obj2, target = both_match.groups()
        return [
            f"pick up the {obj1}",
            f"place the {obj1} in the {target}",
            f"pick up the {obj2}",
            f"place the {obj2} in the {target}",
        ]
    if " and " in lower:
        parts = [p.strip(" .") for p in text.split(" and ") if p.strip()]
        if 2 <= len(parts) <= 5:
            return parts
    if " then " in lower:
        parts = [p.strip(" .") for p in text.split(" then ") if p.strip()]
        if 2 <= len(parts) <= 5:
            return parts
    if "put" in lower or "place" in lower:
        return ["move to object", "grasp object", "move to target", "place object", "open gripper"]
    if "turn on" in lower or "turn off" in lower:
        return ["move to switch", "grasp switch", "toggle switch", "release switch"]
    return ["move to object", "interact with object", "finish task"]


def rollout_single_task(
    task_suite,
    task_id: int,
    init_state_id: int,
    host: str,
    port: int,
    max_steps: int,
    open_loop_horizon: int,
    action_limit: float,
    prompt_mode: str,
    planner: LLMPlanner | None,
    force_system2: bool,
) -> Dict[str, Any]:
    """Run one zero-shot rollout for a single LIBERO task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    instruction = task.language
    task_token = f"task_{task_id + 1}"
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
        controller="JOINT_POSITION",
    )
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[init_state_id])

    client = WebsocketClientPolicy(host, port)
    session_id = f"libero_{uuid.uuid4()}"
    client.reset({"session_id": session_id})

    effective_prompt_mode = prompt_mode
    sub_instructions = [instruction]
    planner_meta: Dict[str, Any] = {
        "planner_mode": "disabled",
        "api_called": False,
        "api_success": False,
        "error": None,
        "model": None,
        "raw_response": None,
    }
    llm_plan_time = 0.0
    if prompt_mode in {"subtask", "hybrid"}:
        t_plan0 = time.perf_counter()
        if planner is None:
            raise RuntimeError("subtask prompt_mode requires a planner instance")
        if force_system2 or should_enable_system2(instruction):
            candidate_sub_instructions, planner_meta = planner.plan(instruction)
            if validate_subtasks(instruction, candidate_sub_instructions):
                sub_instructions = candidate_sub_instructions
            else:
                sub_instructions = heuristic_plan(instruction)
                planner_meta = {
                    "planner_mode": "heuristic_fallback",
                    "api_called": planner_meta.get("api_called", False),
                    "api_success": planner_meta.get("api_success", False),
                    "error": "invalid_or_low_quality_plan",
                    "model": planner_meta.get("model"),
                    "raw_response": planner_meta.get("raw_response"),
                }
        else:
            effective_prompt_mode = "task"
            planner_meta = {
                "planner_mode": "adaptive_skip",
                "api_called": False,
                "api_success": False,
                "error": "single_stage_task_skip_system2",
                "model": None,
                "raw_response": None,
            }
        llm_plan_time = time.perf_counter() - t_plan0

    action_chunk = None
    chunk_idx = 0
    current_instruction = instruction
    reward_sum = 0.0
    success = False
    action_norms = []
    step_logs = []
    t0 = time.perf_counter()

    try:
        for step_idx in range(max_steps):
            if effective_prompt_mode == "task_token_only":
                current_instruction = task_token
            if effective_prompt_mode in {"subtask", "hybrid"}:
                current_instruction = subtask_for_step(sub_instructions, step_idx, max_steps)
            if action_chunk is None or chunk_idx >= min(open_loop_horizon, len(action_chunk)):
                prompt = compose_prompt(instruction, current_instruction, effective_prompt_mode)
                action_chunk = client.infer(build_request(obs, prompt, session_id))
                chunk_idx = 0

            action = np.array(action_chunk[chunk_idx], dtype=np.float32, copy=True)
            chunk_idx += 1

            # Cap joint delta magnitudes for a conservative zero-shot rollout.
            action[:7] = np.clip(action[:7], -action_limit, action_limit)
            action[-1] = 1.0 if action[-1] > 0.5 else -1.0

            low, high = env.env.action_spec
            action = np.clip(action, low, high)
            action_norms.append(float(np.linalg.norm(action[:7])))

            obs, reward, done, _info = env.step(action)
            reward_sum += float(reward)
            current_success = bool(env.check_success())
            if step_idx % 20 == 0 or current_success:
                step_logs.append(
                    {
                        "step_idx": int(step_idx),
                        "reward": float(reward),
                        "success": current_success,
                        "joint_action_l2": float(np.linalg.norm(action[:7])),
                    }
                )

            if current_success:
                success = True
                break
            if done:
                break
    finally:
        env.close()

    elapsed = time.perf_counter() - t0
    return {
        "task_id": int(task_id),
        "task_name": task.name,
        "language_instruction": instruction,
        "init_state_id": int(init_state_id),
        "success": bool(success),
        "reward_sum": float(reward_sum),
        "steps_executed": int(step_idx + 1),
        "mean_joint_action_l2": float(np.mean(action_norms)) if action_norms else 0.0,
        "wall_clock_time_sec": float(elapsed),
        "prompt_mode": effective_prompt_mode,
        "requested_prompt_mode": prompt_mode,
        "task_token": task_token,
        "llm_plan_time_sec": float(llm_plan_time),
        "sub_instructions": sub_instructions,
        "planner_meta": planner_meta,
        "step_logs": step_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small DreamZero benchmark on LIBERO")
    parser.add_argument("--host", type=str, default="localhost", help="DreamZero policy server host")
    parser.add_argument("--port", type=int, default=8000, help="DreamZero policy server port")
    parser.add_argument("--suite", type=str, default="libero_10", help="LIBERO benchmark suite name")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to benchmark from the suite")
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Optional comma-separated task ids to run instead of the first N tasks",
    )
    parser.add_argument("--init-state-id", type=int, default=0, help="Initial state index per task")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum rollout steps per task")
    parser.add_argument("--open-loop-horizon", type=int, default=8, help="How many actions to consume from each DreamZero action chunk")
    parser.add_argument("--action-limit", type=float, default=1.0, help="Clamp absolute joint action values before stepping the environment")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["task", "task_token_only", "subtask", "hybrid"],
        default="task",
        help="Use raw task prompting, task-token-only prompting, subtask-only prompting, or task+subtask hybrid prompting",
    )
    parser.add_argument("--planner-temperature", type=float, default=0.0, help="LLM planner temperature for subtask mode")
    parser.add_argument("--planner-model", type=str, default="gpt-4o-mini", help="LLM planner model for subtask mode")
    parser.add_argument(
        "--force-system2",
        action="store_true",
        help="Force System-2 planning for every task when prompt-mode is subtask or hybrid",
    )
    parser.add_argument(
        "--plan-cache",
        type=str,
        default="evaluation_results_dualsystem/libero_plan_cache.json",
        help="Cache file for per-task LLM plans to avoid repeated API calls across reruns",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="How many init-state episodes to run per selected task",
    )
    parser.add_argument(
        "--skip-broken-tasks",
        action="store_true",
        help="Skip tasks / init states that fail to initialize instead of stopping the whole benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results_dualsystem/libero_10_first5_benchmark.json",
        help="Where to save the benchmark summary JSON",
    )
    args = parser.parse_args()

    patch_torch_load_weights_only()

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_key = str(args.suite).lower()
    if suite_key in benchmark_dict:
        task_suite = benchmark_dict[suite_key]()
    elif hasattr(benchmark, "task_maps") and suite_key in benchmark.task_maps:
        task_suite = benchmark.Benchmark()
        task_suite.name = suite_key
        task_suite._make_benchmark()
    else:
        raise ValueError(f"Unknown LIBERO suite: {args.suite}")
    if args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = list(range(min(int(args.num_tasks), int(task_suite.n_tasks))))
    num_tasks = len(task_ids)
    print(f"Running DreamZero on {args.suite}: tasks {task_ids}")

    planner = None
    if args.prompt_mode in {"subtask", "hybrid"}:
        planner = LLMPlanner(
            model_name=args.planner_model,
            temperature=args.planner_temperature,
            cache_path=args.plan_cache,
        )
        print(f"Dual-system subtask mode enabled with planner={args.planner_model}")

    results = []
    skipped = []
    episode_specs = []
    for task_id in task_ids:
        for init_state_id in range(max(1, int(args.episodes_per_task))):
            episode_specs.append((task_id, init_state_id))

    total_episodes = len(episode_specs)
    for idx, (task_id, init_state_id) in enumerate(episode_specs, start=1):
        print(f"[{idx}/{total_episodes}] task_id={task_id} init_state_id={init_state_id}")
        try:
            result = rollout_single_task(
                task_suite=task_suite,
                task_id=task_id,
                init_state_id=init_state_id,
                host=args.host,
                port=args.port,
                max_steps=args.max_steps,
                open_loop_horizon=args.open_loop_horizon,
                action_limit=args.action_limit,
                prompt_mode=args.prompt_mode,
                planner=planner,
                force_system2=bool(args.force_system2),
            )
        except Exception as exc:
            if args.skip_broken_tasks:
                skipped.append(
                    {
                        "task_id": int(task_id),
                        "init_state_id": int(init_state_id),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"  skipped | {type(exc).__name__}: {exc}")
                continue
            raise

        print(
            f"  success={result['success']} | reward_sum={result['reward_sum']:.3f} | "
            f"steps={result['steps_executed']} | mean_joint_action_l2={result['mean_joint_action_l2']:.3f}"
        )
        results.append(result)

    success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in results])) if results else 0.0
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": args.suite,
        "num_tasks": num_tasks,
        "task_ids": task_ids,
        "episodes_per_task": int(args.episodes_per_task),
        "num_requested_episodes": int(total_episodes),
        "num_completed_episodes": int(len(results)),
        "num_skipped_episodes": int(len(skipped)),
        "init_state_id": int(args.init_state_id),
        "max_steps": int(args.max_steps),
        "open_loop_horizon": int(args.open_loop_horizon),
        "action_limit": float(args.action_limit),
        "prompt_mode": args.prompt_mode,
        "planner_model": args.planner_model if args.prompt_mode == "subtask" else None,
        "host": args.host,
        "port": int(args.port),
        "success_rate": success_rate,
        "num_success": int(sum(1 for r in results if r["success"])),
        "mean_reward_sum": float(np.mean([r["reward_sum"] for r in results])) if results else 0.0,
        "mean_steps_executed": float(np.mean([r["steps_executed"] for r in results])) if results else 0.0,
        "mean_joint_action_l2": float(np.mean([r["mean_joint_action_l2"] for r in results])) if results else 0.0,
        "mean_llm_plan_time_sec": float(np.mean([r["llm_plan_time_sec"] for r in results])) if results else 0.0,
        "skipped": skipped,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LIBERO Benchmark Summary ===")
    print(f"Suite: {args.suite}")
    print(f"Tasks: {num_tasks}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Mean Reward Sum: {summary['mean_reward_sum']:.3f}")
    print(f"Mean Steps: {summary['mean_steps_executed']:.1f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
