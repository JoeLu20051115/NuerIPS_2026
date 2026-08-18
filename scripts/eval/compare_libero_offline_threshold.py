#!/usr/bin/env python3
"""
Compare offline LIBERO action errors between task-token-only System1 and
coverage-fixed Dual subtask prompting.

Episode-level rule:
  If any sampled step has joint-action L2 error > threshold, the episode fails.
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

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

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
    else:
        for token in [" in the ", " on the ", " into the ", " inside the "]:
            if token in lower:
                lhs, rhs = lower.split(token, 1)
                phrases.append(lhs.replace("put the ", "").replace("place the ", "").strip())
                phrases.append(rhs.strip(" ."))
                break
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


class LLMPlanner:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.use_mock = not self.api_key
        self.cache: Dict[str, List[str]] = {}

    def plan(self, task_description: str) -> Tuple[List[str], Dict[str, Any]]:
        if task_description in self.cache:
            return self.cache[task_description], {
                "planner_mode": "cache",
                "api_called": False,
                "api_success": True,
                "error": None,
                "model": self.model_name,
            }
        if self.use_mock:
            plan = heuristic_plan(task_description)
            self.cache[task_description] = plan
            return plan, {
                "planner_mode": "mock",
                "api_called": False,
                "api_success": False,
                "error": "missing_openai_api_key",
                "model": self.model_name,
            }

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
        return plan, {
            "planner_mode": "real_api",
            "api_called": True,
            "api_success": True,
            "error": None,
            "model": self.model_name,
        }


def subtask_for_step(sub_instructions: List[str], step_idx: int, ep_len: int) -> str:
    sub_task_idx = min(int(step_idx * len(sub_instructions) / ep_len), len(sub_instructions) - 1)
    return sub_instructions[sub_task_idx]


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


def build_request_from_dataset(df, videos, step_idx: int, prompt: str, session_id: str) -> Dict[str, Any]:
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


def eval_episode(
    client: WebsocketClientPolicy,
    planner: LLMPlanner,
    task_text: str,
    task_token: str,
    episode_index: int,
    df,
    videos,
    prompt_mode: str,
    eval_steps: int,
    threshold: float,
) -> Dict[str, Any]:
    ep_len = len(df)
    indices = sample_indices(ep_len, eval_steps)
    planner_meta: Dict[str, Any] = {"planner_mode": "disabled"}
    sub_instructions = [task_text]

    if prompt_mode == "dual_subtask_covfix":
        candidate_sub_instructions, planner_meta = planner.plan(task_text)
        if validate_subtasks(task_text, candidate_sub_instructions):
            sub_instructions = compress_subtasks(task_text, candidate_sub_instructions)
        else:
            sub_instructions = heuristic_plan(task_text)
            planner_meta = {
                "planner_mode": "heuristic_fallback",
                "api_called": planner_meta.get("api_called", False),
                "api_success": planner_meta.get("api_success", False),
                "error": "invalid_or_low_quality_plan",
                "model": planner_meta.get("model"),
            }

    step_errors: List[float] = []
    for step_idx in indices:
        if prompt_mode == "task_token_only":
            prompt = task_token
        else:
            prompt = subtask_for_step(sub_instructions, step_idx, ep_len)
        session_id = f"offline_{prompt_mode}_{episode_index}_{step_idx}_{uuid.uuid4().hex[:8]}"
        client.reset({"session_id": session_id})
        pred = client.infer(build_request_from_dataset(df, videos, step_idx, prompt, session_id))
        pred = np.asarray(pred[0], dtype=np.float64) if np.asarray(pred).ndim == 2 else np.asarray(pred, dtype=np.float64)[0]
        err = float(np.linalg.norm(pred[:7] - gt_joint_action(df, step_idx)))
        step_errors.append(err)

    max_error = max(step_errors) if step_errors else float("inf")
    step_success = [bool(err <= threshold) for err in step_errors]
    return {
        "episode_index": episode_index,
        "task": task_text,
        "prompt_mode": prompt_mode,
        "step_indices": indices,
        "step_errors": step_errors,
        "step_success": step_success,
        "num_step_pass": int(sum(step_success)),
        "num_steps": int(len(step_success)),
        "mean_error": float(np.mean(step_errors)) if step_errors else float("inf"),
        "max_error": float(max_error),
        "success": bool(max_error <= threshold),
        "threshold": threshold,
        "sub_instructions": sub_instructions,
        "planner_meta": planner_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset-root", type=str, default="data/libero_subset_lerobot_100ep")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--planner-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results_dualsystem/libero_first10_tasktoken_vs_dual_covfix_threshold0035.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    episodes = load_episodes(dataset_root)[: args.num_episodes]
    tasks_map = load_tasks_map(dataset_root)
    planner = LLMPlanner(model_name=args.planner_model, temperature=args.planner_temperature)
    client = WebsocketClientPolicy(args.host, args.port)

    all_results = {"task_token_only": [], "dual_subtask_covfix": []}

    for idx, ep in enumerate(episodes, start=1):
        episode_index = int(ep["episode_index"])
        df, videos = load_episode_data(dataset_root, episode_index)
        task_index = int(df["task_index"].iloc[0])
        task_text = tasks_map.get(task_index, ep["tasks"][0])
        task_token = f"task_{task_index}"

        task_token_result = eval_episode(
            client, planner, task_text, task_token, episode_index, df, videos,
            "task_token_only", args.eval_steps, args.threshold
        )
        dual_result = eval_episode(
            client, planner, task_text, task_token, episode_index, df, videos,
            "dual_subtask_covfix", args.eval_steps, args.threshold
        )
        all_results["task_token_only"].append(task_token_result)
        all_results["dual_subtask_covfix"].append(dual_result)

        task_token_steps = ", ".join(
            f"{e:.4f}{'✓' if ok else 'x'}"
            for e, ok in zip(task_token_result["step_errors"], task_token_result["step_success"])
        )
        dual_steps = ", ".join(
            f"{e:.4f}{'✓' if ok else 'x'}"
            for e, ok in zip(dual_result["step_errors"], dual_result["step_success"])
        )
        print(f"[{idx:02d}/{len(episodes)}] episode_{episode_index:06d}")
        print(
            f"  task_token_only: step_pass={task_token_result['num_step_pass']}/{task_token_result['num_steps']} | "
            f"steps=[{task_token_steps}]"
        )
        print(
            f"  dual_covfix    : step_pass={dual_result['num_step_pass']}/{dual_result['num_steps']} | "
            f"steps=[{dual_steps}]"
        )

    summary = {}
    for key, records in all_results.items():
        passes = sum(1 for r in records if r["success"])
        total_steps = sum(r["num_steps"] for r in records)
        total_step_pass = sum(r["num_step_pass"] for r in records)
        summary[key] = {
            "num_episodes": len(records),
            "num_episode_pass": passes,
            "episode_success_rate": passes / len(records) if records else 0.0,
            "num_step_pass": total_step_pass,
            "num_steps": total_steps,
            "step_success_rate": total_step_pass / total_steps if total_steps else 0.0,
            "mean_episode_error": float(np.mean([r["mean_error"] for r in records])) if records else float("inf"),
            "mean_max_step_error": float(np.mean([r["max_error"] for r in records])) if records else float("inf"),
        }

    output = {
        "config": vars(args),
        "summary": summary,
        "results": all_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("\nSummary:")
    for key, s in summary.items():
        print(
            f"  {key}: step_pass {s['num_step_pass']}/{s['num_steps']} "
            f"(step_success_rate={s['step_success_rate']:.3f}) | "
            f"episode_pass {s['num_episode_pass']}/{s['num_episodes']} "
            f"(episode_success_rate={s['episode_success_rate']:.3f}) | "
            f"mean_episode_error={s['mean_episode_error']:.4f} | "
            f"mean_max_step_error={s['mean_max_step_error']:.4f}"
        )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
