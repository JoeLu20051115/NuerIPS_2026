#!/usr/bin/env python3
"""
DROID L1/L3 evaluation: compare uniform vs LLM-predicted sub-instruction timing.

Three modes:
  task_token_only  - baseline (no sub-instructions)
  dual_llm         - LLM sub-instructions with UNIFORM temporal placement (existing)
  dual_llm_timed   - LLM sub-instructions with LLM-PREDICTED temporal placement (new)

Usage (per split):
  python run_droid_timed_compare.py \
      --dataset-root ../../data/final_data/DRO_L1_150 \
      --generated-video-dir /path/to/gen_video_dir \
      --log-json ../../evaluation_results_dualsystem/DRO_L1_timed.json

  python run_droid_timed_compare.py \
      --dataset-root ../../data/final_data/DRO_L3_150 \
      --generated-video-dir /path/to/gen_video_dir \
      --log-json ../../evaluation_results_dualsystem/DRO_L3_timed.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

# ── resolve sibling imports ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from run_dualsystem_evaluation import (
    CAMERA_KEYS,
    ROBOARENA_KEY_MAP,
    HAS_POLICY_CLIENT,
    HAS_VIDEO_LOADER,
    LLMPlanner,
    WebsocketClientPolicy,
    get_frames_by_timestamps,
)
from run_compare_tasktoken_dual_judged import FinalFrameJudge

try:
    from openai import OpenAI as _OpenAI
except ImportError as exc:
    raise RuntimeError("openai package required") from exc


# ── LLM planner with predicted timing ────────────────────────────────────────

class LLMPlannerWithTiming:
    """LLM planner that returns (sub_instructions, start_fractions).

    Asks the model to output lines in ``FRACTION|INSTRUCTION`` format, where
    FRACTION is the normalised episode position [0.0, 1.0) at which the
    sub-instruction should begin.  The first fraction is forced to 0.0 and
    fractions are required to be strictly increasing.
    """

    _SYSTEM = (
        "You are a robot task planning assistant. "
        "Output ONLY the requested format—no extra text."
    )

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.client = _OpenAI()
        self.model = model
        self.temperature = temperature

    def _prompt(self, task: str) -> str:
        return (
            f"Task: {task}\n\n"
            "Robot: Franka arm, indoor manipulation scene.\n\n"
            "Break this task into 3–5 atomic sub-instructions. "
            "For each sub-instruction, predict at what fraction of the total "
            "episode length it should BEGIN (0.0 = episode start, 1.0 = end).\n\n"
            "Rules:\n"
            "- First fraction must be 0.0.\n"
            "- Fractions must be strictly increasing.\n"
            "- Reflect natural task timing (approach phase is usually shorter "
            "than manipulation; don't spread everything evenly).\n"
            "- Use ALL sub-instructions (they collectively span the full episode).\n\n"
            "Output one line per sub-instruction, format:\n"
            "FRACTION|INSTRUCTION\n\n"
            "Example:\n"
            "0.0|reach toward the bottle\n"
            "0.2|grasp the bottle\n"
            "0.55|lift and carry to the shelf\n"
            "0.85|place bottle upright on shelf\n"
        )

    def plan_with_timing(
        self, task: str, episode_id: str = ""
    ) -> Tuple[List[str], List[float], Dict]:
        """Return (sub_instructions, start_fractions, meta)."""
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": self._prompt(task)},
                ],
            )
            raw = resp.choices[0].message.content or ""
            instrs, fracs = self._parse(raw, task)
            meta = {
                "planner_mode": "timed_api",
                "api_success": True,
                "model": self.model,
                "raw_response": raw,
                "error": None,
            }
        except Exception as exc:
            instrs, fracs = [task], [0.0]
            meta = {
                "planner_mode": "timed_fallback",
                "api_success": False,
                "model": self.model,
                "raw_response": None,
                "error": str(exc),
            }
        meta["plan_time"] = time.perf_counter() - t0
        return instrs, fracs, meta

    @staticmethod
    def _parse(
        content: str, fallback_task: str
    ) -> Tuple[List[str], List[float]]:
        instrs: List[str] = []
        fracs: List[float] = []
        for line in content.strip().splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            frac_str, _, instr = line.partition("|")
            instr = instr.strip()
            if not instr:
                continue
            try:
                frac = float(frac_str.strip())
            except ValueError:
                continue
            fracs.append(float(np.clip(frac, 0.0, 1.0)))
            instrs.append(instr)

        if not instrs:
            return [fallback_task], [0.0]

        # Enforce first fraction = 0.0
        fracs[0] = 0.0
        # Enforce strictly increasing (add small epsilon if needed)
        for i in range(1, len(fracs)):
            fracs[i] = max(fracs[i], fracs[i - 1] + 1e-3)
        # Cap to [0, 1)
        fracs = [min(f, 1.0 - 1e-6) for f in fracs]

        return instrs, fracs


def _sub_idx_by_timing(fracs: List[float], progress: float) -> int:
    """Return the index of the sub-instruction active at *progress* ∈ [0, 1]."""
    idx = 0
    for i, start in enumerate(fracs):
        if start <= progress:
            idx = i
        else:
            break
    return idx


# ── Compare runner (3 modes) ─────────────────────────────────────────────────

class TimedCompareRunner:
    """Runs task_token_only / dual_llm / dual_llm_timed on one episode."""

    def __init__(
        self,
        dataset_root: Path,
        generated_video_dir: Path,
        host: str,
        port: int,
        judge_model: str,
        eval_steps: int,
        success_threshold: float,
        modes: List[str],
    ) -> None:
        if not HAS_POLICY_CLIENT or not HAS_VIDEO_LOADER:
            raise RuntimeError("policy_client and video_loader are required")

        self.dataset_root = dataset_root
        self.generated_video_dir = generated_video_dir
        self.eval_steps = eval_steps
        self.success_threshold = success_threshold
        self.modes = modes

        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.uniform_planner = LLMPlanner(use_mock=False, temperature=0.0)
        self.timed_planner = LLMPlannerWithTiming(temperature=0.0)
        self.judge = FinalFrameJudge(judge_model)

        # Load episode/task metadata
        self._ep_task: Dict[int, str] = {}
        self._ep_token: Dict[int, str] = {}
        self._load_metadata()

    # ── metadata helpers ──────────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        ep_meta = self.dataset_root / "meta/episodes.jsonl"
        if ep_meta.exists():
            with ep_meta.open() as f:
                for line in f:
                    rec = json.loads(line)
                    idx = int(rec.get("episode_index", -1))
                    tasks = rec.get("tasks", [])
                    if idx >= 0 and tasks:
                        self._ep_task[idx] = str(tasks[0])

        task_meta = self.dataset_root / "meta/tasks.jsonl"
        if task_meta.exists():
            with task_meta.open() as f:
                for line in f:
                    rec = json.loads(line)
                    tidx = int(rec.get("task_index", -1))
                    txt = rec.get("task", "")
                    if tidx >= 0 and txt:
                        self._ep_token[tidx] = str(txt)

    def _task_description(self, df, ep_idx: int) -> Tuple[str, str]:
        task_desc = self._ep_task.get(ep_idx, "")
        if not task_desc and "task_index" in df.columns:
            tidx = int(df["task_index"].iloc[0])
            task_desc = self._ep_token.get(tidx, f"task_{tidx}")
        task_token = ""
        if "task_index" in df.columns:
            task_token = f"task_{int(df['task_index'].iloc[0])}"
        return task_desc or f"task_{ep_idx}", task_token

    def _video_paths(self, ep_idx: int) -> Dict[str, Optional[str]]:
        chunk = ep_idx // 1000
        out: Dict[str, Optional[str]] = {}
        for cam in CAMERA_KEYS:
            p = self.dataset_root / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
            out[cam] = str(p) if p.exists() else None
        return out

    # ── planning helpers ──────────────────────────────────────────────────────

    def _plan(
        self, mode: str, task_desc: str, ep_id: str
    ) -> Tuple[List[str], Optional[List[float]], Dict, float]:
        """Return (sub_instructions, start_fractions_or_None, meta, plan_time)."""
        if mode == "task_token_only":
            return [task_desc], None, {"planner_mode": "disabled"}, 0.0

        if mode == "dual_llm":
            t0 = time.perf_counter()
            instrs, meta = self.uniform_planner.plan(task_desc, {"episode_id": ep_id})
            return instrs, None, meta, time.perf_counter() - t0

        if mode == "dual_llm_timed":
            instrs, fracs, meta = self.timed_planner.plan_with_timing(task_desc, ep_id)
            return instrs, fracs, meta, meta.pop("plan_time", 0.0)

        raise ValueError(f"Unknown mode: {mode}")

    def _get_prompt(
        self,
        mode: str,
        task_desc: str,
        task_token: str,
        sub_instructions: List[str],
        start_fracs: Optional[List[float]],
        step_idx: int,
        ep_len: int,
    ) -> str:
        if mode == "task_token_only":
            return task_token or task_desc

        progress = step_idx / max(ep_len - 1, 1)

        if mode == "dual_llm":
            # Uniform mapping
            sub_idx = min(
                int(step_idx * len(sub_instructions) / ep_len),
                len(sub_instructions) - 1,
            )
        else:  # dual_llm_timed
            sub_idx = _sub_idx_by_timing(start_fracs or [0.0], progress)

        return sub_instructions[sub_idx]

    # ── video helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _last_frame(video_path: Path) -> np.ndarray:
        reader = imageio.get_reader(str(video_path))
        n = reader.count_frames()
        frame = reader.get_data(max(n - 1, 0))
        reader.close()
        return frame

    def _find_generated_video(
        self, session_id: str, existing: set
    ) -> Path:
        # Try session_id match first
        hits = sorted(
            self.generated_video_dir.glob(f"*{session_id}*.mp4"),
            key=lambda p: p.stat().st_mtime,
        )
        if hits:
            return hits[-1]
        # Fall back to newest new file
        fresh = sorted(
            [p for p in self.generated_video_dir.glob("*.mp4") if p.name not in existing],
            key=lambda p: p.stat().st_mtime,
        )
        if fresh:
            return fresh[-1]
        raise FileNotFoundError(
            f"No generated video for session_id={session_id} in {self.generated_video_dir}"
        )

    # ── episode runner ────────────────────────────────────────────────────────

    def run_episode(self, episode_info: Dict, mode: str) -> Dict:
        ep_id = episode_info["episode_id"]
        ep_idx = int(ep_id.split("_")[1])
        chunk = ep_idx // 1000
        parquet_path = (
            self.dataset_root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        )
        df = pq.read_table(str(parquet_path)).to_pandas()
        timestamps = df["timestamp"].to_numpy()
        ep_len = len(df)
        task_desc, task_token = self._task_description(df, ep_idx)

        videos = self._video_paths(ep_idx)
        ref_video = next(
            (videos[k] for k in CAMERA_KEYS if videos.get(k)), None
        )
        if ref_video is None:
            raise RuntimeError(f"No camera video for {ep_id}")

        sub_instrs, start_fracs, planner_meta, plan_time = self._plan(
            mode, task_desc, ep_id
        )

        eval_indices = np.linspace(
            0, ep_len - 1, min(self.eval_steps, ep_len), dtype=int
        )
        action_errors: List[float] = []
        session_id = f"timed_{mode}_{ep_id}_{uuid.uuid4().hex[:8]}"
        existing_vids = {p.name for p in self.generated_video_dir.glob("*.mp4")}
        self.policy.reset({"session_id": session_id})

        initial_frame: Optional[np.ndarray] = None

        for step_idx in eval_indices:
            prompt = self._get_prompt(
                mode, task_desc, task_token, sub_instrs, start_fracs, step_idx, ep_len
            )
            obs: Dict = {}
            for cam_key, vpath in videos.items():
                robo_key = ROBOARENA_KEY_MAP[cam_key]
                if vpath:
                    frame = get_frames_by_timestamps(
                        vpath,
                        np.array([timestamps[step_idx]]),
                        video_backend="ffmpeg",
                    )[0]
                    obs[robo_key] = frame
                    if initial_frame is None and vpath == ref_video:
                        initial_frame = frame
                else:
                    obs[robo_key] = np.zeros((180, 320, 3), dtype=np.uint8)

            state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
            obs["observation/joint_position"] = state[7:14].astype(np.float64)
            obs["observation/gripper_position"] = state[6:7].astype(np.float64)
            obs["prompt"] = prompt
            obs["session_id"] = session_id

            result = self.policy.infer(obs)
            pred_action = result.get("action") if isinstance(result, dict) else result
            if pred_action is None:
                continue

            gt_action = np.array(df["action"].iloc[step_idx], dtype=np.float64)[14:21]
            pred_joint = (
                pred_action[0, :7]
                if getattr(pred_action, "ndim", 1) == 2
                else pred_action[:7]
            )
            action_errors.append(float(np.sqrt(np.sum((pred_joint - gt_action) ** 2))))

        # Reset and wait for generated video
        self.policy.reset({"session_id": session_id})
        time.sleep(1.0)
        gen_video = self._find_generated_video(session_id, existing_vids)
        pred_final = self._last_frame(gen_video)
        real_final = self._last_frame(Path(ref_video))

        judge = self.judge.judge(task_desc, initial_frame, real_final, pred_final)
        task_progress = judge["task_progress"]
        rule_success = bool(judge.get("rule_success", False))
        task_success = bool(
            (task_progress > self.success_threshold) or rule_success
        )

        return {
            "episode_id": ep_id,
            "task_description": task_desc,
            "mode": mode,
            "sub_instructions": sub_instrs,
            "start_fractions": start_fracs,
            "planner_meta": planner_meta,
            "plan_time": plan_time,
            "mean_l2": float(np.mean(action_errors)) if action_errors else None,
            "step_alignment_l2_lt_0_1": (
                float(np.mean([e < 0.1 for e in action_errors])) if action_errors else None
            ),
            "num_step_pass_l2_lt_0_1": int(sum(e < 0.1 for e in action_errors)),
            "task_progress": task_progress,
            "rule_success": rule_success,
            "task_success": task_success,
            "judge_reason": judge["reason"],
            "generated_video": str(gen_video),
        }


# ── dataset loading ───────────────────────────────────────────────────────────

def load_episodes(meta_path: Path, n: int) -> List[Dict]:
    episodes: List[Dict] = []
    with meta_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= n:
                break
            rec = json.loads(line)
            ep_idx = int(rec["episode_index"])
            episodes.append(
                {
                    "episode_id": f"episode_{ep_idx:06d}",
                    "task_description": (
                        rec.get("tasks", [""])[0] if rec.get("tasks") else ""
                    ),
                }
            )
    return episodes


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DROID evaluation with LLM-predicted sub-instruction timing."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/final_data/DRO_L1_150"),
        help="Root of a DRO_L1_150 or DRO_L3_150 dataset.",
    )
    parser.add_argument("--generated-video-dir", type=Path, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=150)
    parser.add_argument("--success-threshold", type=float, default=0.7)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["task_token_only", "dual_llm", "dual_llm_timed"],
        choices=["task_token_only", "dual_llm", "dual_llm_timed"],
        help="Which modes to run (default: all 3).",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=Path("evaluation_results_dualsystem/DRO_timed_compare.json"),
    )
    args = parser.parse_args()

    episodes = load_episodes(
        args.dataset_root / "meta/episodes.jsonl", args.num_episodes
    )
    print(f"Loaded {len(episodes)} episodes from {args.dataset_root.name}")
    print(f"Running modes: {args.modes}")

    runner = TimedCompareRunner(
        dataset_root=args.dataset_root,
        generated_video_dir=args.generated_video_dir,
        host=args.host,
        port=args.port,
        judge_model=args.judge_model,
        eval_steps=args.eval_steps,
        success_threshold=args.success_threshold,
        modes=args.modes,
    )

    all_results: List[Dict] = []
    running: Dict[str, List[Dict]] = {m: [] for m in args.modes}

    for ep_no, episode in enumerate(episodes, 1):
        print(
            f"\n[{ep_no}/{len(episodes)}] {episode['episode_id']} | "
            f"task={episode['task_description'][:60]}"
        )
        for mode in args.modes:
            result = runner.run_episode(episode, mode)
            running[mode].append(result)
            all_results.append(result)

            sr = sum(r["task_success"] for r in running[mode]) / len(running[mode])
            valid_l2 = [r["mean_l2"] for r in running[mode] if r["mean_l2"] is not None]
            ml2 = float(np.mean(valid_l2)) if valid_l2 else float("nan")
            mp = float(np.mean([r["task_progress"] for r in running[mode]]))

            timing_str = ""
            if mode == "dual_llm_timed" and result.get("start_fractions"):
                fracs = [f"{f:.2f}" for f in result["start_fractions"]]
                timing_str = f" | fracs={fracs}"

            print(
                f"  [{mode}] l2={result['mean_l2'] or 0:.4f} | "
                f"progress={result['task_progress']:.3f} | "
                f"{'PASS' if result['task_success'] else 'FAIL'} | "
                f"run_sr={sr:.3f} | run_l2={ml2:.4f} | run_prog={mp:.3f}"
                + timing_str
            )
            if mode in ("dual_llm", "dual_llm_timed"):
                print(f"    sub_instructions={result['sub_instructions']}")

        # Save after every episode
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {"results": all_results, "running_summary": {
                    m: {
                        "n": len(v),
                        "success_rate": float(np.mean([r["task_success"] for r in v])) if v else 0.0,
                        "mean_l2": float(np.mean([r["mean_l2"] for r in v if r["mean_l2"] is not None])) if any(r["mean_l2"] is not None for r in v) else None,
                        "mean_task_progress": float(np.mean([r["task_progress"] for r in v])) if v else 0.0,
                    }
                    for m, v in running.items()
                }},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    # Final summary
    summary: Dict = {}
    for mode, rows in running.items():
        valid_l2 = [r["mean_l2"] for r in rows if r["mean_l2"] is not None]
        valid_align = [r["step_alignment_l2_lt_0_1"] for r in rows if r["step_alignment_l2_lt_0_1"] is not None]
        summary[mode] = {
            "num_episodes": len(rows),
            "success_rate": float(np.mean([r["task_success"] for r in rows])) if rows else 0.0,
            "mean_l2": float(np.mean(valid_l2)) if valid_l2 else None,
            "mean_task_progress": float(np.mean([r["task_progress"] for r in rows])) if rows else 0.0,
            "mean_step_alignment_l2_lt_0_1": float(np.mean(valid_align)) if valid_align else None,
        }

    out = {
        "dataset": str(args.dataset_root),
        "num_episodes": len(episodes),
        "results": all_results,
        "summary": summary,
    }
    args.log_json.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    print("\n=== Final Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
