#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from diffusers.video_processor import VideoProcessor
from openai import OpenAI
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
LINGBOT_VA_ROOT = REPO_ROOT / "external_repos" / "lingbot-va"
if str(LINGBOT_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_VA_ROOT))
if str(REPO_ROOT / "scripts" / "eval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server
from llm_planner_val import LLMPlannerWithVAL
from run_dualsystem_evaluation import LLMPlanner


CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def create_chat_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
):
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, str):
            cleaned = obj.encode("utf-8", errors="replace").decode("utf-8")
            return "".join(c for c in cleaned if c >= " " or c in "\n\t")
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        return obj

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": _sanitize(messages),
    }
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_output_tokens
    else:
        kwargs["max_tokens"] = max_output_tokens
    return client.chat.completions.create(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-evaluate LingBot-VA on the RoboTwin benchmark with task_token_only / dual_llm / llm_val prompts."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_ROOT / "data/robotwin_lingbot_eval_seed42_l123_300",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints/lingbot-va-posttrain-robotwin",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_smoke.json",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=REPO_ROOT / "evaluation_results_dualsystem/robotwin_lingbot_smoke_artifacts",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="task_token_only,dual_llm,llm_val",
        help="Comma-separated list of modes to run.",
    )
    parser.add_argument(
        "--episodes-per-level",
        type=int,
        default=1,
        help="Balanced smoke setting: sample this many episodes from each of L1/L2/L3 before shuffling globally.",
    )
    parser.add_argument(
        "--max-step-count",
        type=int,
        default=320,
        help="Optional smoke cap. Episodes longer than this are skipped before sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", type=str, default="gpt-5.4")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument(
        "--planner-model",
        type=str,
        default="gpt-5.4",
        help="Planner model for dual_llm and llm_val.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previous artifact directories before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSON if present.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip saving generated rollout videos to reduce disk usage.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the sampled episode list into this many deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to run when num-shards > 1.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_selected_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(benchmark_dir / "meta" / "episodes.jsonl")


def sample_smoke_episodes(
    rows: list[dict[str, Any]],
    *,
    episodes_per_level: int,
    seed: int,
    max_step_count: int | None,
) -> list[dict[str, Any]]:
    if max_step_count is not None:
        rows = [row for row in rows if int(row["step_count"]) <= int(max_step_count)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["complexity_level"])].append(row)

    selected: list[dict[str, Any]] = []
    for level in ("L1", "L2", "L3"):
        pool = list(grouped[level])
        rng = random.Random(f"{seed}:{level}:smoke")
        rng.shuffle(pool)
        if len(pool) < episodes_per_level:
            raise RuntimeError(
                f"Not enough episodes in {level} after filtering: requested {episodes_per_level}, found {len(pool)}."
            )
        selected.extend(pool[:episodes_per_level])

    rng = random.Random(f"{seed}:global-smoke-order")
    rng.shuffle(selected)
    return selected


def parquet_path(benchmark_dir: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return benchmark_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(benchmark_dir: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return benchmark_dir / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def load_episode_arrays(benchmark_dir: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path(benchmark_dir, int(row["episode_index"])))
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    return state, action


def load_frame(video_file: Path, frame_index: int) -> np.ndarray:
    reader = imageio.get_reader(str(video_file))
    try:
        return np.asarray(reader.get_data(frame_index))
    finally:
        reader.close()


def load_final_frame(video_file: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_file))
    try:
        return np.asarray(reader.get_data(reader.count_frames() - 1))
    finally:
        reader.close()


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    new_pose = np.asarray(new_pose, dtype=np.float64)
    init_pose = np.asarray(init_pose, dtype=np.float64)
    new_rot = R.from_quat(normalize_quat(new_pose[3:7])[None])
    init_rot = R.from_quat(normalize_quat(init_pose[3:7])[None])
    out_rot = (init_rot * new_rot).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]], axis=0)


def add_init_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    out = np.concatenate([left_pose, right_pose], axis=0)
    out[3:7] = normalize_quat(out[3:7])
    out[11:15] = normalize_quat(out[11:15])
    return out.astype(np.float32)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("mean_l2") is not None]
    total_steps = sum(int(row.get("compared_steps", 0)) for row in valid)
    total_good_steps = sum(int(row.get("num_step_pass_l2_lt_0_1", 0)) for row in valid)
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([row["mean_l2"] for row in valid])) if valid else None,
        "mean_task_progress": float(np.mean([row["task_progress"] for row in rows])) if rows else None,
        "success_rate": float(np.mean([1.0 if row["task_success"] else 0.0 for row in rows])) if rows else None,
        "rate_of_l2_lt_0_1": float(total_good_steps / total_steps) if total_steps > 0 else None,
    }


def compose_prompt(task: str, sub_instructions: list[str]) -> str:
    clean_steps = [step.strip().rstrip(".") for step in sub_instructions if step.strip()]
    if not clean_steps:
        return task
    return f"Task: {task}\nPlan: " + " Then, ".join(clean_steps) + "."


def compose_weighted_stage_prompt(task: str, sub_instructions: list[str], active_idx: int) -> str:
    clean_steps = [step.strip().rstrip(".") for step in sub_instructions if step.strip()]
    if not clean_steps:
        return task
    active_idx = int(np.clip(active_idx, 0, len(clean_steps) - 1))
    parts = [f"Task: {task}", f"Overall plan: " + " Then, ".join(clean_steps) + "."]
    parts.append(f"Current stage ({active_idx + 1}/{len(clean_steps)}): {clean_steps[active_idx]}.")
    if active_idx > 0:
        parts.append("Completed before this: " + " Then, ".join(clean_steps[:active_idx]) + ".")
    if active_idx + 1 < len(clean_steps):
        parts.append("Remaining after this: " + " Then, ".join(clean_steps[active_idx + 1 :]) + ".")
    parts.append("Focus on the current stage while keeping the overall task goal consistent.")
    return "\n".join(parts)


def _normalize_step_windows(
    sub_instructions: list[str],
    planner_meta: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    clean_steps = [step.strip().rstrip(".") for step in sub_instructions if step and step.strip()]
    if not clean_steps:
        return [{"instruction": "", "start": 0.0, "end": 1.0}]

    planner_meta = planner_meta or {}
    structured_steps = planner_meta.get("steps") if isinstance(planner_meta, dict) else None
    start_fractions = planner_meta.get("start_fractions") if isinstance(planner_meta, dict) else None
    windows: list[dict[str, Any]] = []

    if isinstance(structured_steps, list) and structured_steps:
        for idx, raw_step in enumerate(structured_steps[: len(clean_steps)]):
            instruction = str(raw_step.get("instruction") or clean_steps[idx]).strip().rstrip(".")
            time_window = raw_step.get("time")
            if isinstance(time_window, (list, tuple)) and len(time_window) == 2:
                start = float(time_window[0])
                end = float(time_window[1])
            else:
                start = float(start_fractions[idx]) if isinstance(start_fractions, list) and idx < len(start_fractions) else None
                next_start = (
                    float(start_fractions[idx + 1])
                    if isinstance(start_fractions, list) and idx + 1 < len(start_fractions)
                    else None
                )
                start = 0.0 if start is None else start
                end = 1.0 if next_start is None else next_start
            windows.append({"instruction": instruction, "start": start, "end": end})
    elif isinstance(start_fractions, list) and len(start_fractions) == len(clean_steps):
        for idx, instruction in enumerate(clean_steps):
            start = float(start_fractions[idx])
            end = float(start_fractions[idx + 1]) if idx + 1 < len(start_fractions) else 1.0
            windows.append({"instruction": instruction, "start": start, "end": end})
    else:
        for idx, instruction in enumerate(clean_steps):
            start = idx / len(clean_steps)
            end = (idx + 1) / len(clean_steps)
            windows.append({"instruction": instruction, "start": start, "end": end})

    normalized: list[dict[str, Any]] = []
    prev_end = 0.0
    total = len(windows)
    for idx, item in enumerate(windows):
        start = float(item.get("start", prev_end))
        end = float(item.get("end", 1.0 if idx == total - 1 else start))
        start = float(np.clip(start, 0.0, 1.0))
        end = float(np.clip(end, 0.0, 1.0))
        start = max(prev_end, start)
        if idx == total - 1:
            end = 1.0
        elif end <= start:
            end = min(1.0, max(start + 1e-4, (idx + 1) / total))
        prev_end = end
        normalized.append(
            {
                "instruction": str(item.get("instruction", clean_steps[min(idx, len(clean_steps) - 1)])).strip().rstrip("."),
                "start": round(start, 4),
                "end": round(end, 4),
            }
        )

    normalized[0]["start"] = 0.0
    normalized[-1]["end"] = 1.0
    return normalized


class ChatCompletionJudge:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        import base64
        import io

        buf = io.BytesIO()
        imageio.imwrite(buf, frame, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def judge(
        self,
        task: str,
        initial_frame: np.ndarray,
        real_final_frame: np.ndarray,
        predicted_final_frame: np.ndarray,
    ) -> dict[str, Any]:
        prompt = (
            "You are judging robot task completion from visual evidence.\n"
            "You will see three images:\n"
            "1. initial real observation\n"
            "2. real final observation from the demonstration (reference goal state)\n"
            "3. predicted final frame from the model rollout\n\n"
            f"Task: {task}\n\n"
            "Return strict JSON only with schema:\n"
            "{\"task_progress\": number, \"rule_success\": boolean, \"reason\": string}\n\n"
            "Rules:\n"
            "- task_progress must be in [0,1].\n"
            "- rule_success means: the final state satisfies the task goal, yes or no.\n"
            "- Judge the model's predicted final frame against the task goal, using the real final frame only as a helpful reference for the intended successful end state.\n"
            "- Be slightly lenient to small pose offsets if the task intent is clearly satisfied.\n"
            "- 1.0 means the predicted final frame clearly satisfies the task.\n"
            "- 0.7-0.9 means the task is almost fully achieved and should count as success in a lenient setting.\n"
            "- 0.4-0.6 means partial completion or the key object reached the target region but final relation is not fully convincing.\n"
            "- 0.0-0.3 means the task goal is not achieved.\n"
            "- If the final state visibly satisfies the task goal, set rule_success=true even if task_progress is below 1.0.\n"
        )
        try:
            response = create_chat_completion(
                client=self.client,
                model=self.model,
                temperature=0.0,
                max_output_tokens=300,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful robot-evaluation judge. Output JSON only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Initial real observation:"},
                            {"type": "image_url", "image_url": {"url": self._frame_to_data_url(initial_frame)}},
                            {"type": "text", "text": "Real final observation from the demonstration:"},
                            {"type": "image_url", "image_url": {"url": self._frame_to_data_url(real_final_frame)}},
                            {"type": "text", "text": "Predicted final frame from the model rollout:"},
                            {"type": "image_url", "image_url": {"url": self._frame_to_data_url(predicted_final_frame)}},
                        ],
                    },
                ],
            )
        except Exception as exc:
            # Fail closed on judge API issues so a single bad request does not stop the full run.
            return {
                "task_progress": 0.0,
                "rule_success": False,
                "reason": f"judge_api_error: {type(exc).__name__}: {str(exc)[:200]}",
            }
        content = response.choices[0].message.content or "{}"
        try:
            start = content.find("{")
            end = content.rfind("}")
            payload = json.loads(content[start : end + 1] if start != -1 and end != -1 and end > start else content)
        except Exception:
            payload = {"task_progress": 0.0, "rule_success": False, "reason": f"non_json_response: {content[:200]}"}
        progress = float(np.clip(float(payload.get("task_progress", 0.0)), 0.0, 1.0))
        rule_success = bool(payload.get("rule_success", False))
        return {"task_progress": progress, "rule_success": rule_success, "reason": str(payload.get("reason", ""))}


class ChatCompletionLLMPlanner(LLMPlanner):
    def plan(self, task_description: str, episode_context: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        if self.use_mock:
            return self._mock_plan(task_description), {
                "planner_mode": "mock",
                "api_called": False,
                "api_success": False,
                "error": self.mock_reason,
                "model": self.model_name,
                "raw_response": None,
            }

        try:
            prompt = self._build_prompt(task_description, episode_context)
            messages = [
                {
                    "role": "system",
                    "content": "You are a robot task planning assistant. Break down complex tasks into 3-5 atomic sub-tasks that a robot can execute sequentially.",
                },
                {"role": "user", "content": prompt},
            ]
            client = OpenAI(api_key=self.api_key)
            response = create_chat_completion(
                client=client,
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=256,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            sub_instructions = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
            sub_instructions = sub_instructions[:5]
            if not sub_instructions:
                sub_instructions = [task_description]
            return sub_instructions, {
                "planner_mode": "real_api",
                "api_called": True,
                "api_success": True,
                "error": None,
                "model": self.model_name,
                "raw_response": content,
            }
        except Exception as exc:
            print(f"⚠️  LLM API error: {exc}, falling back to mock", flush=True)
            return self._mock_plan(task_description), {
                "planner_mode": "fallback_mock",
                "api_called": True,
                "api_success": False,
                "error": str(exc),
                "model": self.model_name,
                "raw_response": None,
            }


class ChatCompletionVALPlanner(LLMPlannerWithVAL):
    def _llm(self, user_prompt: str, max_tokens: int = 600) -> str:
        user_prompt = user_prompt.encode("utf-8", errors="replace").decode("utf-8")
        user_prompt = "".join(c for c in user_prompt if c >= " " or c in "\n\t")
        try:
            resp = create_chat_completion(
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a robot task planning assistant. Output ONLY the requested JSON format—no extra text."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[LLM error] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            return ""


class LingBotSmokeRunner:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        save_root: Path,
        judge_model: str,
        success_threshold: float,
        planner_model: str,
        save_video: bool,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_root = save_root
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.save_root / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = save_video
        self.judge = ChatCompletionJudge(judge_model)
        self.dual_planner = ChatCompletionLLMPlanner(
            model_name=planner_model,
            use_mock=False,
            temperature=0.0,
        )
        self.val_planner = ChatCompletionVALPlanner(model=planner_model, temperature=0.0)

        config = copy.deepcopy(VA_CONFIGS["robotwin_i2av"])
        config.wan22_pretrained_model_name_or_path = str(checkpoint_dir)
        config.save_root = str(save_root / "lingbot_server_outputs")
        config.rank = 0
        config.local_rank = 0
        config.world_size = 1
        self.model = VA_Server(config)
        self.model.video_processor = VideoProcessor(vae_scale_factor=1)
        self.success_threshold = float(success_threshold)

    def close(self) -> None:
        try:
            self.model.transformer.clear_cache(self.model.cache_name)
        except Exception:
            pass
        try:
            self.model.streaming_vae.clear_cache()
        except Exception:
            pass
        try:
            if self.model.streaming_vae_half:
                self.model.streaming_vae_half.clear_cache()
        except Exception:
            pass
        try:
            del self.model
        except Exception:
            pass
        torch.cuda.empty_cache()

    def _plan(self, mode: str, task: str) -> tuple[str, list[str], dict[str, Any], float]:
        if mode == "task_token_only":
            return task, [task], {"planner_mode": "disabled"}, 0.0
        t0 = time.perf_counter()
        if mode == "dual_llm":
            sub_instructions, planner_meta = self.dual_planner.plan(task, {})
        elif mode == "llm_val":
            sub_instructions, planner_meta = self.val_planner.plan(task, {})
        else:
            raise ValueError(f"Unknown mode: {mode}")
        prompt = compose_prompt(task, sub_instructions)
        return prompt, sub_instructions, planner_meta, time.perf_counter() - t0

    def _set_prompt_conditioning(self, prompt: str | None) -> None:
        if prompt is None:
            self.model.prompt_embeds = None
            self.model.negative_prompt_embeds = None
            return
        self.model.prompt_embeds, self.model.negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=self.model.job_config.guidance_scale > 1,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def _build_prompt_schedule(
        self,
        *,
        mode: str,
        task: str,
        base_prompt: str,
        sub_instructions: list[str],
        planner_meta: dict[str, Any],
        num_chunks: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if num_chunks <= 0:
            return [], {"strategy": "empty"}
        if mode != "llm_val":
            return [base_prompt] * num_chunks, {"strategy": "flat_prompt"}

        step_windows = _normalize_step_windows(sub_instructions, planner_meta)
        if len(step_windows) <= 1:
            return [base_prompt] * num_chunks, {"strategy": "flat_prompt_single_step", "step_windows": step_windows}

        chunk_active_step_ids: list[int] = []
        schedule: list[str] = []
        for chunk_id in range(num_chunks):
            progress = (chunk_id + 0.5) / num_chunks
            active_idx = len(step_windows) - 1
            for idx, window in enumerate(step_windows):
                if progress <= float(window["end"]) or idx == len(step_windows) - 1:
                    active_idx = idx
                    break
            chunk_active_step_ids.append(active_idx)
            schedule.append(compose_weighted_stage_prompt(task, sub_instructions, active_idx))

        return schedule, {
            "strategy": "weighted_chunk_focus",
            "step_windows": step_windows,
            "chunk_active_step_ids": chunk_active_step_ids,
        }

    def _write_input_images(self, frames: dict[str, np.ndarray]) -> str:
        tmpdir = tempfile.mkdtemp(prefix="lingbot_robotwin_smoke_")
        for key, frame in frames.items():
            imageio.imwrite(Path(tmpdir) / f"{key}.png", frame.astype(np.uint8))
        return tmpdir

    def _generate_episode(
        self,
        *,
        input_img_path: str,
        prompt: str,
        num_chunks: int,
        prompt_schedule: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.job_config.input_img_path = input_img_path
        self.model.job_config.num_chunks_to_infer = int(num_chunks)
        effective_schedule = list(prompt_schedule) if prompt_schedule else [prompt] * max(1, int(num_chunks))
        if len(effective_schedule) < num_chunks:
            effective_schedule.extend([effective_schedule[-1]] * (num_chunks - len(effective_schedule)))
        initial_prompt = effective_schedule[0] if effective_schedule else prompt
        self.model._reset(initial_prompt)
        exp_save_root = Path(self.model.exp_save_root)
        init_obs = self.model.load_init_obs()
        pred_action_lst: list[torch.Tensor] = []
        pred_latent_lst: list[torch.Tensor] | None = [] if self.save_video else None
        final_latent_cpu: torch.Tensor | None = None
        pred_action: np.ndarray | None = None
        pred_latent: torch.Tensor | None = None
        latents: torch.Tensor | None = None
        latents_mean: torch.Tensor | None = None
        latents_std: torch.Tensor | None = None
        video_tensor: torch.Tensor | None = None
        decoded_video: np.ndarray | None = None
        try:
            active_prompt = initial_prompt
            for chunk_id in range(num_chunks):
                chunk_prompt = effective_schedule[min(chunk_id, len(effective_schedule) - 1)]
                if chunk_prompt != active_prompt:
                    self._set_prompt_conditioning(chunk_prompt)
                    active_prompt = chunk_prompt
                actions, latents = self.model._infer(
                    init_obs, frame_st_id=(chunk_id * self.model.job_config.frame_chunk_size)
                )
                pred_action_lst.append(torch.from_numpy(actions.copy()))
                if self.save_video:
                    assert pred_latent_lst is not None
                    pred_latent_lst.append(latents.detach().cpu())
                else:
                    final_latent_cpu = latents[:, :, -1:, :, :].detach().cpu()
                del latents
                latents = None

            pred_action = torch.cat(pred_action_lst, dim=1).flatten(1).cpu().numpy()
            if self.save_video:
                assert pred_latent_lst is not None
                pred_latent = torch.cat(pred_latent_lst, dim=2)
            else:
                if final_latent_cpu is None:
                    raise RuntimeError("Missing final latent for no-save-video mode.")
                pred_latent = final_latent_cpu

            self.model.transformer.clear_cache(self.model.cache_name)
            self.model.streaming_vae.clear_cache()
            if self.model.streaming_vae_half:
                self.model.streaming_vae_half.clear_cache()
            torch.cuda.empty_cache()

            if self.model.enable_offload:
                self.model.vae = self.model.vae.to(self.model.device).to(self.model.dtype)

            with torch.inference_mode():
                latents = pred_latent.to(self.model.device)
                latents = latents.to(self.model.vae.dtype)
                latents_mean = (
                    torch.tensor(self.model.vae.config.latents_mean)
                    .view(1, self.model.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
                    1, self.model.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video_tensor = self.model.vae.decode(latents, return_dict=False)[0]
            decoded_video = self.model.video_processor.postprocess_video(
                video_tensor.detach(), output_type="np"
            )[0]
            return pred_action, np.asarray(decoded_video)
        finally:
            for attr in ("prompt_embeds", "negative_prompt_embeds", "init_latent"):
                if hasattr(self.model, attr):
                    setattr(self.model, attr, None)
            if hasattr(self.model.vae, "_feat_map"):
                self.model.vae._feat_map = None
            if hasattr(self.model.vae, "_conv_idx"):
                self.model.vae._conv_idx = 0
            if self.model.enable_offload:
                self.model.vae = self.model.vae.to("cpu")
            self.model.transformer.clear_cache(self.model.cache_name)
            self.model.streaming_vae.clear_cache()
            if self.model.streaming_vae_half:
                self.model.streaming_vae_half.clear_cache()
            del pred_action_lst
            del pred_latent_lst
            del final_latent_cpu
            del pred_latent
            del latents
            del latents_mean
            del latents_std
            del video_tensor
            del decoded_video
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(exp_save_root, ignore_errors=True)

    def run_episode(
        self,
        *,
        benchmark_dir: Path,
        row: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        task = str(row["tasks"][0])
        state, gt_action = load_episode_arrays(benchmark_dir, row)
        cam_high_path = video_path(benchmark_dir, episode_index, "observation.images.cam_high")
        cam_left_path = video_path(benchmark_dir, episode_index, "observation.images.cam_left_wrist")
        cam_right_path = video_path(benchmark_dir, episode_index, "observation.images.cam_right_wrist")

        first_frames = {
            "observation.images.cam_high": load_frame(cam_high_path, 0),
            "observation.images.cam_left_wrist": load_frame(cam_left_path, 0),
            "observation.images.cam_right_wrist": load_frame(cam_right_path, 0),
        }
        real_final_frame = load_final_frame(cam_high_path)

        prompt, sub_instructions, planner_meta, plan_time = self._plan(mode, task)
        planner_meta = dict(planner_meta)
        input_img_path = self._write_input_images(first_frames)
        try:
            num_chunks = max(1, int(math.ceil(max(1, len(gt_action) - 1) / 32.0)))
            prompt_schedule, execution_schedule = self._build_prompt_schedule(
                mode=mode,
                task=task,
                base_prompt=prompt,
                sub_instructions=sub_instructions,
                planner_meta=planner_meta,
                num_chunks=num_chunks,
            )
            planner_meta["execution_schedule"] = execution_schedule
            pred_action_raw, decoded_video = self._generate_episode(
                input_img_path=input_img_path,
                prompt=prompt,
                num_chunks=num_chunks,
                prompt_schedule=prompt_schedule,
            )
        finally:
            shutil.rmtree(input_img_path, ignore_errors=True)

        pred_action_raw = pred_action_raw.T.astype(np.float32)
        if pred_action_raw.shape[1] == 16:
            pred_action_abs = np.stack(
                [add_init_pose(step, state[0].astype(np.float64)) for step in pred_action_raw],
                axis=0,
            )
        else:
            raise RuntimeError(f"Unexpected predicted action shape: {pred_action_raw.shape}")

        compared_steps = min(len(gt_action), len(pred_action_abs))
        errors = np.linalg.norm(pred_action_abs[:compared_steps] - gt_action[:compared_steps], axis=1)

        pred_video_uint8 = decoded_video
        if pred_video_uint8.dtype != np.uint8:
            if pred_video_uint8.max() <= 1.0:
                pred_video_uint8 = np.clip(pred_video_uint8 * 255.0, 0, 255).astype(np.uint8)
            else:
                pred_video_uint8 = np.clip(pred_video_uint8, 0, 255).astype(np.uint8)
        pred_final_frame = np.asarray(pred_video_uint8[-1])

        out_video: Path | None = None
        if self.save_video:
            out_video = self.video_dir / f"episode_{episode_index:06d}_{mode}.mp4"
            imageio.mimsave(out_video, list(pred_video_uint8), fps=10)

        judged = self.judge.judge(task, first_frames["observation.images.cam_high"], real_final_frame, pred_final_frame)
        task_progress = float(judged["task_progress"])
        rule_success = bool(judged["rule_success"])
        task_success = bool((task_progress > self.success_threshold) or rule_success)

        return {
            "episode_index": episode_index,
            "source_uid": row["source_uid"],
            "task_name": row["task_name"],
            "task": task,
            "complexity_level": row["complexity_level"],
            "mode": mode,
            "prompt_used": prompt,
            "sub_instructions": sub_instructions,
            "planner_meta": planner_meta,
            "plan_time": plan_time,
            "num_chunks": num_chunks,
            "predicted_steps": int(len(pred_action_abs)),
            "compared_steps": int(compared_steps),
            "mean_l2": float(np.mean(errors)) if len(errors) else None,
            "num_step_pass_l2_lt_0_1": int(np.sum(errors < 0.1)) if len(errors) else 0,
            "step_alignment_l2_lt_0_1": float(np.mean(errors < 0.1)) if len(errors) else None,
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judged["reason"],
            "generated_video": str(out_video) if out_video is not None else None,
        }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    set_seed(args.seed)
    if torch.cuda.is_available():
        print(
            "CUDA mapping: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
            f"torch_device_count={torch.cuda.device_count()} | "
            f"current_device={torch.cuda.current_device()} | "
            f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}",
            flush=True,
        )

    if args.clean_output and args.save_root.exists():
        shutil.rmtree(args.save_root)
    args.save_root.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows = load_selected_episodes(args.benchmark_dir)
    smoke_rows = sample_smoke_episodes(
        rows,
        episodes_per_level=args.episodes_per_level,
        seed=args.seed,
        max_step_count=args.max_step_count,
    )
    if args.num_shards > 1:
        full_count = len(smoke_rows)
        smoke_rows = [row for idx, row in enumerate(smoke_rows) if idx % args.num_shards == args.shard_index]
        print(
            f"Sharding enabled: shard {args.shard_index}/{args.num_shards} "
            f"contains {len(smoke_rows)} episodes from {full_count} total.",
            flush=True,
        )

    all_results: list[dict[str, Any]] = []
    running: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    completed: set[tuple[int, str]] = set()
    if args.resume and args.output_json.exists():
        previous = json.loads(args.output_json.read_text())
        if previous.get("episodes"):
            smoke_rows = previous["episodes"]
        all_results = list(previous.get("results", []))
        for result in all_results:
            mode = str(result.get("mode"))
            if mode in running:
                running[mode].append(result)
            completed.add((int(result["episode_index"]), mode))
        print(
            f"Resuming from {len(completed)} completed episode-mode pairs "
            f"out of {len(smoke_rows) * len(modes)}.",
            flush=True,
        )

    runner = LingBotSmokeRunner(
        checkpoint_dir=args.checkpoint_dir,
        save_root=args.save_root,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
        planner_model=args.planner_model,
        save_video=not args.no_save_video,
    )

    try:
        total_jobs = len(smoke_rows) * len(modes)
        job_idx = len(completed)
        for row in smoke_rows:
            print(
                f"\n[episode {row['episode_index']}] {row['task_name']} "
                f"| level={row['complexity_level']} | steps={row['step_count']}",
                flush=True,
            )
            for mode in modes:
                if (int(row["episode_index"]), mode) in completed:
                    print(f"  [skip] mode={mode} already completed", flush=True)
                    continue
                job_idx += 1
                print(f"  [{job_idx}/{total_jobs}] mode={mode}", flush=True)
                result = runner.run_episode(benchmark_dir=args.benchmark_dir, row=row, mode=mode)
                running[mode].append(result)
                all_results.append(result)
                completed.add((int(row["episode_index"]), mode))
                cur = summarize(running[mode])
                l2_str = f"{result['mean_l2']:.4f}" if result["mean_l2"] is not None else "N/A"
                cur_l2_str = f"{cur['mean_l2']:.4f}" if cur["mean_l2"] is not None else "N/A"
                print(
                    "    "
                    f"mean_l2={l2_str} | "
                    f"task_progress={result['task_progress']:.3f} | "
                    f"success={'PASS' if result['task_success'] else 'FAIL'} | "
                    f"running_mean_l2={cur_l2_str} | "
                    f"running_mean_task_progress={cur['mean_task_progress']:.3f} | "
                    f"running_success_rate={cur['success_rate']:.3f}",
                    flush=True,
                )
                if mode != "task_token_only":
                    print(f"    sub_instructions={result['sub_instructions']}", flush=True)

                payload = {
                    "meta": {
                        "benchmark_dir": str(args.benchmark_dir),
                        "checkpoint_dir": str(args.checkpoint_dir),
                        "episodes_per_level": int(args.episodes_per_level),
                    "max_step_count": args.max_step_count,
                    "seed": int(args.seed),
                    "modes": modes,
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                },
                "episodes": smoke_rows,
                "results": all_results,
                "summary": {mode: summarize(rows_) for mode, rows_ in running.items()},
            }
                args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    finally:
        runner.close()

    final = {
        "meta": {
            "benchmark_dir": str(args.benchmark_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "episodes_per_level": int(args.episodes_per_level),
            "max_step_count": args.max_step_count,
            "seed": int(args.seed),
            "modes": modes,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "episodes": smoke_rows,
        "results": all_results,
        "summary": {mode: summarize(rows_) for mode, rows_ in running.items()},
    }
    args.output_json.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print("\n=== Summary ===", flush=True)
    print(json.dumps(final["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
