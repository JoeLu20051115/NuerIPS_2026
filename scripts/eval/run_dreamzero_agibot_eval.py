#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    from eval_utils.policy_client import WebsocketClientPolicy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("policy_client is required") from exc


MANIFEST_PATH = Path("data/agibot_easy400_for_droid/meta/agibot_easy400_dreamzero_manifest.json")


class AgiBotDreamZeroEvaluator:
    def __init__(self, manifest_path: Path, host: str, port: int, eval_steps: int, camera_mode: str, prompt_mode: str) -> None:
        self.manifest = json.loads(manifest_path.read_text())
        self.episodes = self.manifest["episodes"]
        self.host = host
        self.port = port
        self.eval_steps = eval_steps
        self.camera_mode = camera_mode
        self.prompt_mode = prompt_mode
        self.policy = WebsocketClientPolicy(host=host, port=port)

    @staticmethod
    def _load_episode_metadata(row: dict) -> tuple[int, int, list[str]]:
        if "active_frame_start" in row and "active_frame_end" in row and "action_plan" in row:
            return int(row["active_frame_start"]), int(row["active_frame_end"]), list(row.get("action_plan") or [])
        episode_dir = Path(row["relative_episode_dir"])
        data_info_path = Path(row.get("data_info_path", episode_dir / "data_info.json"))
        if not data_info_path.exists():
            return 0, int(row.get("num_steps", 1)) - 1, []
        data = json.loads(data_info_path.read_text())
        action_cfg = data.get("label_info", {}).get("action_config", [])
        if not action_cfg:
            return 0, int(row.get("num_steps", 1)) - 1, []
        active_start = min(int(x.get("start_frame", 0)) for x in action_cfg)
        active_end = max(int(x.get("end_frame", int(row.get("num_steps", 1)) - 1)) for x in action_cfg)
        action_plan = [x.get("english_action_text", "").strip() for x in action_cfg if x.get("english_action_text")]
        return active_start, active_end, action_plan

    @staticmethod
    def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        try:
            frame = reader.get_data(frame_idx)
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

    def _prompt(self, row: dict, action_plan: list[str]) -> str:
        base = row["english_task_name"] or row["task_group"]
        if self.prompt_mode == "task_plus_plan" and action_plan:
            return base + ". Plan: " + " ".join(action_plan[:3])
        return base

    def _build_obs_images(self, head_frame: np.ndarray, left_frame: np.ndarray, right_frame: np.ndarray) -> dict:
        head_frame = self._resize_frame(head_frame)
        left_frame = self._resize_frame(left_frame)
        right_frame = self._resize_frame(right_frame)
        if self.camera_mode == "head_head_right":
            return {
                "observation/exterior_image_0_left": head_frame,
                "observation/exterior_image_1_left": head_frame,
                "observation/wrist_image_left": right_frame,
            }
        if self.camera_mode == "head_right_right":
            return {
                "observation/exterior_image_0_left": head_frame,
                "observation/exterior_image_1_left": right_frame,
                "observation/wrist_image_left": right_frame,
            }
        return {
            "observation/exterior_image_0_left": head_frame,
            "observation/exterior_image_1_left": left_frame,
            "observation/wrist_image_left": right_frame,
        }

    def run_episode(self, row: dict) -> dict:
        h5_path = Path(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            states = f["state/joint/position"][:]
            actions = f["action/joint/position"][:]
            right_gripper = f["state/right_effector/position"][:]
            steps = states.shape[0]

        head = Path(row["camera_paths"]["head"])
        hand_left = Path(row["camera_paths"]["hand_left"])
        hand_right = Path(row["camera_paths"]["hand_right"])
        head_frames = row["camera_frames"]["head"]
        left_frames = row["camera_frames"]["hand_left"]
        right_frames = row["camera_frames"]["hand_right"]

        active_start, active_end, action_plan = self._load_episode_metadata(row)
        indices = self._sample_indices(active_start, active_end, self.eval_steps)
        errors = []
        base_session = f"agibot_eval_{self.camera_mode}_{self.prompt_mode}_{row['episode_id']}"

        for step_idx in indices:
            head_idx = min(step_idx, head_frames - 1)
            left_idx = min(step_idx, left_frames - 1)
            right_idx = min(step_idx, right_frames - 1)
            obs = self._build_obs_images(
                self._load_frame(head, head_idx),
                self._load_frame(hand_left, left_idx),
                self._load_frame(hand_right, right_idx),
            )
            session_id = f"{base_session}_{int(step_idx)}"
            self.policy.reset({"session_id": session_id})
            obs.update(
                {
                "observation/joint_position": states[step_idx, 7:14].astype(np.float64),
                "observation/gripper_position": right_gripper[step_idx].astype(np.float64),
                "prompt": self._prompt(row, action_plan),
                "session_id": session_id,
                }
            )
            t0 = time.perf_counter()
            pred = self.policy.infer(obs)
            _ = time.perf_counter() - t0
            action = pred.get("action", None) if isinstance(pred, dict) else pred
            if action is None:
                continue
            pred_joint = action[0, :7] if getattr(action, "ndim", 1) == 2 else action[:7]
            gt_joint = actions[step_idx, 7:14]
            errors.append(float(np.sqrt(np.sum((pred_joint - gt_joint) ** 2))))

        return {
            "episode_id": row["episode_id"],
            "task": row["english_task_name"],
            "mean_l2": float(np.mean(errors)) if errors else None,
            "num_steps": len(errors),
            "camera_mode": self.camera_mode,
            "prompt_mode": self.prompt_mode,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval-steps", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--camera-mode", type=str, default="head_head_right", choices=["head_left_right", "head_head_right", "head_right_right"])
    parser.add_argument("--prompt-mode", type=str, default="task_only", choices=["task_only", "task_plus_plan"])
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_results_dualsystem/agibot_adapter_validation.json"))
    args = parser.parse_args()

    evaluator = AgiBotDreamZeroEvaluator(args.manifest_path, args.host, args.port, args.eval_steps, args.camera_mode, args.prompt_mode)
    rows = []
    for row in evaluator.episodes[: args.max_episodes]:
        print(f"running {row['episode_id']} | {row['english_task_name']}")
        rows.append(evaluator.run_episode(row))

    payload = {
        "manifest_path": str(args.manifest_path),
        "num_episodes": len(rows),
        "camera_mode": args.camera_mode,
        "prompt_mode": args.prompt_mode,
        "mean_l2": float(np.mean([r["mean_l2"] for r in rows if r["mean_l2"] is not None])) if rows else None,
        "episodes": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
