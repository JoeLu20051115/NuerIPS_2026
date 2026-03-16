#!/usr/bin/env python3
"""
Print first-10 LIBERO offline mean errors in a concise format:
  - System1 task_token_only
  - Dual subtask (coverage-fix)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy
from groot.vla.common.utils import get_frames_by_timestamps


def extract_coverage_terms(task_instruction: str) -> List[str]:
    lower = task_instruction.lower().strip()
    phrases: List[str] = []
    both_match = re.fullmatch(r"put both the (.+?) and the (.+?) in the (.+)", lower)
    if both_match:
        obj1, obj2, target = both_match.groups()
        phrases.extend([obj1.strip(), obj2.strip(), target.strip()])
    normalized = []
    for phrase in phrases:
        phrase = re.sub(r"\s+", " ", phrase).strip(" .")
        if phrase and phrase not in normalized:
            normalized.append(phrase)
    return normalized


def compress_subtasks(task_instruction: str, sub_instructions: List[str], max_subtasks: int = 5) -> List[str]:
    normalized = [re.sub(r"\s+", " ", s).strip(" .") for s in sub_instructions if s and s.strip()]
    if len(normalized) <= max_subtasks:
        return normalized

    merged: List[str] = []
    idx = 0
    while idx < len(normalized):
        cur = normalized[idx]
        nxt = normalized[idx + 1] if idx + 1 < len(normalized) else None
        if nxt and ("move to" in cur.lower() or "reach" in cur.lower()) and any(
            token in nxt.lower() for token in ["grasp", "pick", "close gripper"]
        ):
            merged.append(f"{cur} and {nxt}")
            idx += 2
            continue
        if nxt and any(token in cur.lower() for token in ["move to", "carry", "move"]) and any(
            token in nxt.lower() for token in ["place", "put", "release", "open gripper"]
        ):
            merged.append(f"{cur} and {nxt}")
            idx += 2
            continue
        merged.append(cur)
        idx += 1

    normalized = merged
    while len(normalized) > max_subtasks:
        merged = []
        idx = 0
        while idx < len(normalized):
            cur = normalized[idx]
            nxt = normalized[idx + 1] if idx + 1 < len(normalized) else None
            if nxt is not None and len(normalized) - len(merged) > max_subtasks:
                merged.append(f"{cur} and {nxt}")
                idx += 2
            else:
                merged.append(cur)
                idx += 1
        normalized = merged[:max_subtasks]
    return normalized


def validate_subtasks(task_instruction: str, sub_instructions: List[str]) -> bool:
    clean = [s.strip() for s in sub_instructions if s.strip()]
    if not (1 <= len(clean) <= 5):
        return False
    coverage_terms = extract_coverage_terms(task_instruction)
    if coverage_terms:
        joined = " ".join(clean).lower()
        if any(term not in joined for term in coverage_terms):
            return False
    return True


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
    if "put" in lower or "place" in lower:
        return ["move to object", "grasp object", "move to target", "place object", "open gripper"]
    return ["move to object", "interact with object", "finish task"]


class LLMPlanner:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.use_mock = not self.api_key
        self.cache: Dict[str, List[str]] = {}

    def plan(self, task_description: str) -> List[str]:
        if task_description in self.cache:
            return self.cache[task_description]
        if self.use_mock:
            plan = heuristic_plan(task_description)
            self.cache[task_description] = plan
            return plan

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
                {"role": "user", "content": task_description},
            ],
        )
        text = response.choices[0].message.content or ""
        plan = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        plan = compress_subtasks(task_description, plan)
        self.cache[task_description] = plan
        return plan


def load_tasks_map(dataset_root: Path) -> Dict[int, str]:
    tasks = {}
    with open(dataset_root / "meta/tasks.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            tasks[int(rec["task_index"])] = rec["task"]
    return tasks


def load_episodes(dataset_root: Path) -> List[Dict[str, Any]]:
    episodes = []
    with open(dataset_root / "meta/episodes.jsonl", "r") as f:
        for line in f:
            episodes.append(json.loads(line))
    episodes.sort(key=lambda x: int(x["episode_index"]))
    return episodes


def load_episode_data(dataset_root: Path, episode_index: int):
    parquet_path = dataset_root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
    df = pq.read_table(parquet_path).to_pandas()
    timestamps = df["timestamp"].to_numpy()
    videos = {}
    for cam_key in [
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
        "observation.images.wrist_image_left",
    ]:
        video_path = dataset_root / f"videos/chunk-000/{cam_key}/episode_{episode_index:06d}.mp4"
        frames = get_frames_by_timestamps(str(video_path), timestamps, video_backend="ffmpeg")
        videos[cam_key] = frames
    return df, videos


def build_request(df, videos, step_idx: int, prompt: str, session_id: str) -> Dict[str, Any]:
    state = np.asarray(df["observation.state"].iloc[step_idx], dtype=np.float64)
    return {
        "observation/exterior_image_0_left": videos["observation.images.exterior_image_1_left"][step_idx],
        "observation/exterior_image_1_left": videos["observation.images.exterior_image_2_left"][step_idx],
        "observation/wrist_image_left": videos["observation.images.wrist_image_left"][step_idx],
        "observation/joint_position": state[7:14],
        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
        "observation/gripper_position": state[6:7],
        "prompt": prompt,
        "session_id": session_id,
    }


def gt_joint_action(df, step_idx: int) -> np.ndarray:
    action = np.asarray(df["action"].iloc[step_idx], dtype=np.float64)
    return action[14:21]


def sample_indices(ep_len: int, eval_steps: int) -> List[int]:
    if ep_len <= eval_steps:
        return list(range(ep_len))
    return np.linspace(0, ep_len - 1, eval_steps, dtype=int).tolist()


def subtask_for_step(sub_instructions: List[str], step_idx: int, ep_len: int) -> str:
    sub_task_idx = min(int(step_idx * len(sub_instructions) / ep_len), len(sub_instructions) - 1)
    return sub_instructions[sub_task_idx]


def eval_prompt_mode(
    client: WebsocketClientPolicy,
    planner: LLMPlanner,
    df,
    videos,
    episode_index: int,
    task_text: str,
    task_token: str,
    prompt_mode: str,
    eval_steps: int,
) -> Dict[str, Any]:
    ep_len = len(df)
    indices = sample_indices(ep_len, eval_steps)
    sub_instructions = [task_text]

    if prompt_mode == "dual_covfix":
        candidate = planner.plan(task_text)
        if validate_subtasks(task_text, candidate):
            sub_instructions = compress_subtasks(task_text, candidate)
        else:
            sub_instructions = heuristic_plan(task_text)

    step_errors = []
    for step_idx in indices:
        prompt = task_token if prompt_mode == "task_token_only" else subtask_for_step(sub_instructions, step_idx, ep_len)
        session_id = f"mean_{prompt_mode}_{episode_index}_{step_idx}_{uuid.uuid4().hex[:8]}"
        client.reset({"session_id": session_id})
        pred = client.infer(build_request(df, videos, step_idx, prompt, session_id))
        pred = np.asarray(pred)
        if pred.ndim == 2:
            pred = pred[0]
        else:
            pred = pred[0]
        step_errors.append(float(np.linalg.norm(pred[:7] - gt_joint_action(df, step_idx))))

    return {
        "step_indices": indices,
        "step_errors": step_errors,
        "mean_error": float(np.mean(step_errors)),
        "sub_instructions": sub_instructions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset-root", type=str, default="data/libero_subset_lerobot_100ep")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--planner-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results_dualsystem/libero_first10_tasktoken_vs_dual_covfix_mean.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    tasks_map = load_tasks_map(dataset_root)
    episodes = load_episodes(dataset_root)[: args.num_episodes]
    planner = LLMPlanner(args.planner_model, args.planner_temperature)
    client = WebsocketClientPolicy(args.host, args.port)

    results = {"task_token_only": [], "dual_covfix": []}

    for idx, ep in enumerate(episodes, start=1):
        episode_index = int(ep["episode_index"])
        df, videos = load_episode_data(dataset_root, episode_index)
        task_index = int(df["task_index"].iloc[0])
        task_text = tasks_map.get(task_index, ep["tasks"][0])
        task_token = f"task_{task_index}"

        task_res = eval_prompt_mode(client, planner, df, videos, episode_index, task_text, task_token, "task_token_only", args.eval_steps)
        dual_res = eval_prompt_mode(client, planner, df, videos, episode_index, task_text, task_token, "dual_covfix", args.eval_steps)
        results["task_token_only"].append({"episode_index": episode_index, "task": task_text, **task_res})
        results["dual_covfix"].append({"episode_index": episode_index, "task": task_text, **dual_res})

        print(
            f"[{idx:02d}/{len(episodes)}] episode_{episode_index:06d} | "
            f"task_token_only={task_res['mean_error']:.7f} | "
            f"dual_covfix={dual_res['mean_error']:.7f} | "
            f"task={task_text}"
        )

    system1_mean = float(np.mean([r["mean_error"] for r in results["task_token_only"]]))
    dual_covfix_mean = float(np.mean([r["mean_error"] for r in results["dual_covfix"]]))
    improvement = dual_covfix_mean - system1_mean

    output = {
        "config": vars(args),
        "results": results,
        "summary": {
            "system1_task_token_only": system1_mean,
            "dual_covfix": dual_covfix_mean,
            "improvement_vs_system1": improvement,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("\nSummary:")
    print(f"System1: {system1_mean:.7f}")
    print(f"新 Dual subtask (coverage-fix): {dual_covfix_mean:.7f}")
    print("所以现在：")
    if improvement < 0:
        print("新版 Dual subtask 已经超过了 System1")
    else:
        print("新版 Dual subtask 还没有超过 System1")
    print(f"相比 System1 改善了 {improvement:+.7f}")
    if results["dual_covfix"]:
        print("\n示例子任务：")
        for step in results["dual_covfix"][0]["sub_instructions"]:
            print(step)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
