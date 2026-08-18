#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

from run_dualsystem_evaluation import (
    CAMERA_KEYS,
    ROBOARENA_KEY_MAP,
    DualSystemEvaluator,
    HAS_POLICY_CLIENT,
    HAS_VIDEO_LOADER,
    LLMPlanner,
    WebsocketClientPolicy,
    get_frames_by_timestamps,
)

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package is required for judged comparison eval") from exc


class FinalFrameJudge:
    def __init__(self, model: str) -> None:
        api_key = Path.home()
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        buf = io.BytesIO()
        imageio.imwrite(buf, frame, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def judge(self, task: str, initial_frame: np.ndarray, real_final_frame: np.ndarray, predicted_final_frame: np.ndarray) -> Dict:
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
            "- For drawers, switches, buttons: focus on the final controlled state.\n"
            "- For object placement: focus on whether the object is clearly at the intended destination/relation.\n"
            "- If the final state visibly satisfies the task goal, set rule_success=true even if task_progress is below 1.0.\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=300,
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


class CompareRunner:
    def __init__(
        self,
        dataset_root: Path,
        generated_video_dir: Path,
        host: str,
        port: int,
        judge_model: str,
        eval_steps: int,
        success_threshold: float,
    ) -> None:
        if not HAS_POLICY_CLIENT or not HAS_VIDEO_LOADER:
            raise RuntimeError("policy client and video loader are required")
        self.dataset_root = dataset_root
        self.generated_video_dir = generated_video_dir
        self.host = host
        self.port = port
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.planner = LLMPlanner(use_mock=False, temperature=0.0)
        self.dual_helper = DualSystemEvaluator(
            host=host,
            port=port,
            dataset_root=str(dataset_root),
            planner_mode="llm",
            prompt_mode="subtask",
            eval_steps=eval_steps,
        )
        self.dual_helper.policy_client = self.policy
        self.judge = FinalFrameJudge(judge_model)
        self.episode_task_map = self.dual_helper._load_episode_task_map()
        self.task_index_map = self.dual_helper._load_task_index_map()

    def _load_videos(self, episode_id: str) -> Dict[str, Optional[str]]:
        return self.dual_helper._load_episode_videos(episode_id)

    @staticmethod
    def _extract_first_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(0)
        reader.close()
        return frame

    @staticmethod
    def _extract_final_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(reader.count_frames() - 1)
        reader.close()
        return frame

    def _find_generated_video(self, session_id: str, existing_files: set[str]) -> Path:
        candidates = sorted(
            self.generated_video_dir.glob(f"*{session_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]

        # DROID server currently saves videos with a monotonically increasing
        # filename instead of the session_id, so fall back to "newest file that
        # appeared during this episode/mode run".
        fresh_candidates = sorted(
            [p for p in self.generated_video_dir.glob("*.mp4") if p.name not in existing_files],
            key=lambda p: p.stat().st_mtime,
        )
        if fresh_candidates:
            return fresh_candidates[-1]

        raise FileNotFoundError(f"No generated video found for session_id={session_id} in {self.generated_video_dir}")

    def _task_description(self, df, ep_idx: int) -> tuple[str, str]:
        task_description = self.episode_task_map.get(ep_idx, "")
        if not task_description and "task_index" in df.columns:
            task_idx = int(df["task_index"].iloc[0])
            task_description = self.task_index_map.get(task_idx, f"task_{task_idx}")
        if not task_description:
            task_description = f"task_{ep_idx}"
        task_token = ""
        if "task_index" in df.columns:
            task_token = f"task_{int(df['task_index'].iloc[0])}"
        return task_description, task_token

    def _mode_prompt(self, mode: str, task_description: str, task_token: str, current_instruction: str) -> str:
        if mode == "task_token_only":
            return task_token or task_description
        return current_instruction

    def _plan(self, mode: str, task_description: str, episode_id: str) -> tuple[List[str], Dict, float]:
        if mode == "task_token_only":
            return [task_description], {"planner_mode": "disabled"}, 0.0
        start = time.perf_counter()
        sub_instructions, planner_meta = self.planner.plan(task_description, {"episode_id": episode_id})
        return sub_instructions, planner_meta, time.perf_counter() - start

    def run_episode(self, episode_info: Dict, mode: str) -> Dict:
        episode_id = episode_info["episode_id"]
        ep_idx = int(episode_id.split("_")[1])
        chunk = ep_idx // 1000
        parquet_path = self.dataset_root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        df = pq.read_table(str(parquet_path)).to_pandas()
        timestamps = df["timestamp"].to_numpy()
        ep_len = len(df)
        task_description, task_token = self._task_description(df, ep_idx)
        videos = self._load_videos(episode_id)
        ref_video = next((videos[k] for k in CAMERA_KEYS if videos.get(k)), None)
        if ref_video is None:
            raise RuntimeError(f"Missing all camera videos for {episode_id}")

        sub_instructions, planner_meta, plan_time = self._plan(mode, task_description, episode_id)
        eval_indices = np.linspace(0, ep_len - 1, min(self.eval_steps, ep_len), dtype=int)
        action_errors: List[float] = []
        session_id = f"cmp_{mode}_{episode_id}_{uuid.uuid4().hex[:8]}"
        existing_videos = {p.name for p in self.generated_video_dir.glob("*.mp4")}
        self.policy.reset({"session_id": session_id})

        initial_frame = None
        for step_idx in eval_indices:
            sub_idx = min(int(step_idx * len(sub_instructions) / ep_len), len(sub_instructions) - 1)
            current_instruction = sub_instructions[sub_idx]
            obs = {}
            for cam_key, video_path in videos.items():
                robo_key = ROBOARENA_KEY_MAP[cam_key]
                if video_path:
                    frame = get_frames_by_timestamps(video_path, np.array([timestamps[step_idx]]), video_backend="ffmpeg")[0]
                    obs[robo_key] = frame
                    if initial_frame is None and video_path == ref_video:
                        initial_frame = frame
                else:
                    obs[robo_key] = np.zeros((180, 320, 3), dtype=np.uint8)
            state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
            obs["observation/joint_position"] = state[7:14].astype(np.float64)
            obs["observation/gripper_position"] = state[6:7].astype(np.float64)
            obs["prompt"] = self._mode_prompt(mode, task_description, task_token, current_instruction)
            obs["session_id"] = session_id

            result = self.policy.infer(obs)
            pred_action = result.get("action", None) if isinstance(result, dict) else result
            if pred_action is None:
                continue
            gt_action = np.array(df["action"].iloc[step_idx], dtype=np.float64)[14:21]
            pred_joint = pred_action[0, :7] if getattr(pred_action, "ndim", 1) == 2 else pred_action[:7]
            action_errors.append(float(np.sqrt(np.sum((pred_joint - gt_action) ** 2))))

        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        generated_video = self._find_generated_video(session_id, existing_videos)
        predicted_final = self._extract_final_frame(generated_video)
        real_final = self._extract_final_frame(Path(ref_video))
        judge = self.judge.judge(task_description, initial_frame, real_final, predicted_final)
        task_progress = judge["task_progress"]
        rule_success = bool(judge.get("rule_success", False))
        task_success = bool((task_progress > self.success_threshold) or rule_success)
        return {
            "episode_id": episode_id,
            "task_description": task_description,
            "mode": mode,
            "sub_instructions": sub_instructions,
            "planner_meta": planner_meta,
            "plan_time": plan_time,
            "mean_l2": float(np.mean(action_errors)) if action_errors else None,
            "step_alignment_l2_lt_0_1": float(np.mean([e < 0.1 for e in action_errors])) if action_errors else None,
            "num_step_pass_l2_lt_0_1": int(sum(e < 0.1 for e in action_errors)) if action_errors else 0,
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judge["reason"],
            "generated_video": str(generated_video),
        }


def load_first_n_episodes(meta_path: Path, n: int) -> List[Dict]:
    episodes = []
    with meta_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= n:
                break
            rec = json.loads(line)
            ep_idx = int(rec["episode_index"])
            episodes.append(
                {
                    "episode_id": f"episode_{ep_idx:06d}",
                    "task_description": rec.get("tasks", [""])[0] if rec.get("tasks") else "",
                }
            )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task_token_only vs dual-llm on judged offline rollout metrics.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/droid_easy400_dualfavored_dreamzero"))
    parser.add_argument("--generated-video-dir", type=Path, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--success-threshold", type=float, default=0.7)
    parser.add_argument("--log-json", type=Path, default=Path("evaluation_results_dualsystem/easy400_compare_first5.json"))
    args = parser.parse_args()

    episodes = load_first_n_episodes(args.dataset_root / "meta/episodes.jsonl", args.num_episodes)
    runner = CompareRunner(
        dataset_root=args.dataset_root,
        generated_video_dir=args.generated_video_dir,
        host=args.host,
        port=args.port,
        judge_model=args.judge_model,
        eval_steps=args.eval_steps,
        success_threshold=args.success_threshold,
    )

    all_results: List[Dict] = []
    running: Dict[str, List[Dict]] = {"task_token_only": [], "dual_llm": []}
    for idx, episode in enumerate(episodes, 1):
        print(f"\n[{idx}/{len(episodes)}] {episode['episode_id']} | task={episode['task_description']}")
        for mode in ["task_token_only", "dual_llm"]:
            mode_label = "dual_llm" if mode == "dual_llm" else "task_token_only"
            result = runner.run_episode(episode, mode)
            running[mode_label].append(result)
            all_results.append(result)
            success_rate = sum(1 for r in running[mode_label] if r["task_success"]) / len(running[mode_label])
            mean_l2 = np.mean([r["mean_l2"] for r in running[mode_label] if r["mean_l2"] is not None])
            mean_progress = np.mean([r["task_progress"] for r in running[mode_label]])
            print(
                f"  [{mode_label}] mean_l2={result['mean_l2']:.4f} | "
                f"task_progress={result['task_progress']:.3f} | "
                f"rule_success={'PASS' if result['rule_success'] else 'FAIL'} | "
                f"success={'PASS' if result['task_success'] else 'FAIL'} | "
                f"running_success_rate={success_rate:.3f} | "
                f"running_mean_l2={mean_l2:.4f} | "
                f"running_mean_progress={mean_progress:.3f}"
            )
            if mode_label == "dual_llm":
                print(f"    sub_instructions={result['sub_instructions']}")
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")

    summary = {}
    for mode, rows in running.items():
        summary[mode] = {
            "num_episodes": len(rows),
            "success_rate": float(np.mean([1.0 if r["task_success"] else 0.0 for r in rows])) if rows else 0.0,
            "mean_l2": float(np.mean([r["mean_l2"] for r in rows if r["mean_l2"] is not None])) if rows else None,
            "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])) if rows else None,
            "mean_step_alignment_l2_lt_0_1": float(np.mean([r["step_alignment_l2_lt_0_1"] for r in rows if r["step_alignment_l2_lt_0_1"] is not None])) if rows else None,
        }
    out = {"results": all_results, "summary": summary}
    args.log_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
