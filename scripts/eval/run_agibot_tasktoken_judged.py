#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import uuid
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("openai package is required for AgiBot judged eval") from exc

try:
    from eval_utils.policy_client import WebsocketClientPolicy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("policy_client is required") from exc


DEFAULT_MANIFEST = Path("data/agibot_easy400_for_droid/meta/agibot_easy400_dreamzero_manifest.json")
DEFAULT_GENERATED_VIDEO_DIR = Path("checkpoints/real_world_eval_gen_20260317_0/DreamZero-DROID")


class FinalFrameJudge:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
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
    ) -> dict:
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


class AgiBotTaskTokenOnlyRunner:
    def __init__(
        self,
        manifest_path: Path,
        generated_video_dir: Path,
        host: str,
        port: int,
        eval_steps: int,
        judge_model: str,
        success_threshold: float,
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text())
        self.episodes = self.manifest["episodes"]
        self.generated_video_dir = generated_video_dir
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.judge = FinalFrameJudge(judge_model)

    @staticmethod
    def _load_episode_metadata(row: dict) -> tuple[int, int]:
        episode_dir = Path(row["relative_episode_dir"])
        data_info_path = Path(row.get("data_info_path", episode_dir / "data_info.json"))
        if not data_info_path.exists():
            return 0, int(row.get("num_steps", 1)) - 1
        data = json.loads(data_info_path.read_text())
        action_cfg = data.get("label_info", {}).get("action_config", [])
        if not action_cfg:
            return 0, int(row.get("num_steps", 1)) - 1
        active_start = min(int(x.get("start_frame", 0)) for x in action_cfg)
        active_end = max(int(x.get("end_frame", int(row.get("num_steps", 1)) - 1)) for x in action_cfg)
        return active_start, active_end

    @staticmethod
    def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(frame_idx)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _extract_first_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(0)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _extract_final_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(reader.count_frames() - 1)
        finally:
            reader.close()
        return frame

    @staticmethod
    def _resize_frame(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))

    @staticmethod
    def _sample_indices(start: int, end: int, eval_steps: int) -> np.ndarray:
        start = int(max(0, start))
        end = int(max(start, end))
        return np.linspace(start, end, min(end - start + 1, eval_steps), dtype=int)

    def _build_obs_images(self, head_frame: np.ndarray, right_frame: np.ndarray) -> dict:
        head_frame = self._resize_frame(head_frame)
        right_frame = self._resize_frame(right_frame)
        return {
            "observation/exterior_image_0_left": head_frame,
            "observation/exterior_image_1_left": head_frame,
            "observation/wrist_image_left": right_frame,
        }

    def _find_generated_video(self, session_id: str, existing_files: set[str]) -> Path:
        candidates = sorted(
            self.generated_video_dir.glob(f"*{session_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
        fresh_candidates = sorted(
            [p for p in self.generated_video_dir.glob("*.mp4") if p.name not in existing_files],
            key=lambda p: p.stat().st_mtime,
        )
        if fresh_candidates:
            return fresh_candidates[-1]
        raise FileNotFoundError(f"No generated video found for session_id={session_id}")

    def run_episode(self, row: dict) -> dict:
        h5_path = Path(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            states = f["state/joint/position"][:]
            actions = f["action/joint/position"][:]
            right_gripper = f["state/right_effector/position"][:]

        head_path = Path(row["camera_paths"]["head"])
        right_path = Path(row["camera_paths"]["hand_right"])
        head_frames = int(row["camera_frames"]["head"])
        right_frames = int(row["camera_frames"]["hand_right"])

        active_start, active_end = self._load_episode_metadata(row)
        indices = self._sample_indices(active_start, active_end, self.eval_steps)
        task = row["english_task_name"] or row["task_group"]
        session_id = f"agibot_tasktoken_{row['episode_id']}_{uuid.uuid4().hex[:8]}"
        existing_videos = {p.name for p in self.generated_video_dir.glob('*.mp4')}
        action_errors: list[float] = []

        initial_frame = self._extract_first_frame(head_path)
        real_final_frame = self._extract_final_frame(head_path)
        self.policy.reset({"session_id": session_id})

        for step_idx in indices:
            head_idx = min(int(step_idx), head_frames - 1)
            right_idx = min(int(step_idx), right_frames - 1)
            obs = self._build_obs_images(
                self._load_frame(head_path, head_idx),
                self._load_frame(right_path, right_idx),
            )
            obs.update(
                {
                    "observation/joint_position": states[step_idx, 7:14].astype(np.float64),
                    "observation/gripper_position": right_gripper[step_idx].astype(np.float64),
                    "prompt": task,
                    "session_id": session_id,
                }
            )
            result = self.policy.infer(obs)
            pred_action = result.get("action", None) if isinstance(result, dict) else result
            if pred_action is None:
                continue
            pred_joint = pred_action[0, :7] if getattr(pred_action, "ndim", 1) == 2 else pred_action[:7]
            gt_joint = actions[step_idx, 7:14]
            action_errors.append(float(np.sqrt(np.sum((pred_joint - gt_joint) ** 2))))

        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        generated_video = self._find_generated_video(session_id, existing_videos)
        predicted_final_frame = self._extract_final_frame(generated_video)
        judged = self.judge.judge(task, initial_frame, real_final_frame, predicted_final_frame)
        task_progress = float(judged["task_progress"])
        rule_success = bool(judged["rule_success"])
        task_success = bool((task_progress > self.success_threshold) or rule_success)
        return {
            "episode_id": row["episode_id"],
            "task": task,
            "mean_l2": float(np.mean(action_errors)) if action_errors else None,
            "num_steps": len(action_errors),
            "num_step_pass_l2_lt_0_1": int(sum(err < 0.1 for err in action_errors)),
            "step_alignment_l2_lt_0_1": float(np.mean([err < 0.1 for err in action_errors])) if action_errors else None,
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judged["reason"],
            "generated_video": str(generated_video),
        }


def summarize(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "mean_l2": float(np.mean([r["mean_l2"] for r in rows if r["mean_l2"] is not None])) if rows else None,
        "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])) if rows else None,
        "success_rate": float(np.mean([1.0 if r["task_success"] else 0.0 for r in rows])) if rows else None,
        "rate_of_l2_lt_0_1": (
            float(sum(r["num_step_pass_l2_lt_0_1"] for r in rows) / sum(r["num_steps"] for r in rows))
            if rows and sum(r["num_steps"] for r in rows) > 0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgiBot task_token_only judged pilot.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generated-video-dir", type=Path, default=DEFAULT_GENERATED_VIDEO_DIR)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval-steps", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation_results_dualsystem/agibot_tasktoken_first10_judged.json"),
    )
    args = parser.parse_args()

    runner = AgiBotTaskTokenOnlyRunner(
        manifest_path=args.manifest_path,
        generated_video_dir=args.generated_video_dir,
        host=args.host,
        port=args.port,
        eval_steps=args.eval_steps,
        judge_model=args.judge_model,
        success_threshold=args.success_threshold,
    )

    rows: list[dict] = []
    for idx, row in enumerate(runner.episodes[: args.num_episodes], 1):
        print(f"\n[{idx}/{args.num_episodes}] {row['episode_id']} | task={row['english_task_name']}")
        result = runner.run_episode(row)
        rows.append(result)
        running = summarize(rows)
        print(
            "  "
            f"mean_l2={result['mean_l2']:.4f} | "
            f"task_progress={result['task_progress']:.3f} | "
            f"rule_success={'PASS' if result['rule_success'] else 'FAIL'} | "
            f"success={'PASS' if result['task_success'] else 'FAIL'} | "
            f"running_mean_l2={running['mean_l2']:.4f} | "
            f"running_mean_task_progress={running['mean_task_progress']:.3f} | "
            f"running_success_rate={running['success_rate']:.3f} | "
            f"running_rate_l2_lt_0_1={running['rate_of_l2_lt_0_1']:.3f}"
        )
        payload = {"results": rows, "summary": running}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    final = {"results": rows, "summary": summarize(rows)}
    args.output_json.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print("\n=== Summary ===")
    print(json.dumps(final["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
