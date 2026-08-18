#!/usr/bin/env python3
"""
Quick comparison: System1 (task_description) vs Dual (hybrid) on 5 episodes.
OXE_testdataset — real-time L2 error printing.
"""

import json, sys, time, os, re
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_utils.policy_client import WebsocketClientPolicy
from groot.vla.common.utils import get_frames_by_timestamps

# ── Config ──────────────────────────────────────────────────────────
DATASET_NAME = "OXE_testdataset"
HOST, PORT = "localhost", 8000
DATASET_ROOT = Path("data/droid_lerobot")
TEST_SET = "test_sets_final/L3_sequential_200.json"
NUM_EPISODES = 5
EVAL_STEPS = 10
ERROR_THRESHOLD = 0.15

CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]
ROBOARENA_KEY_MAP = {
    "observation.images.exterior_image_1_left": "observation/exterior_image_0_left",
    "observation.images.exterior_image_2_left": "observation/exterior_image_1_left",
    "observation.images.wrist_image_left": "observation/wrist_image_left",
}

# ── LLM Planner (for Dual mode) ────────────────────────────────────
def llm_plan(task_description: str) -> list[str]:
    """Call GPT-4o-mini to decompose task into sub-instructions."""
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            f"You are a robotic task planner. Break the following manipulation "
            f"task into 3-5 sequential sub-steps. Return ONLY a JSON list of strings.\n\n"
            f"Task: {task_description}\n\nSub-steps:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON list from response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return [task_description]
    except Exception as e:
        print(f"    [LLM Error] {e}, falling back to task_description")
        return [task_description]


# ── Shared inference logic ──────────────────────────────────────────
def load_episode(ep_info):
    ep_id = ep_info["episode_id"]
    ep_idx = int(ep_id.split("_")[1])
    chunk = ep_idx // 1000
    parquet_path = DATASET_ROOT / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    # Video paths
    video_paths = {}
    for cam_key in CAMERA_KEYS:
        vp = DATASET_ROOT / f"videos/chunk-{chunk:03d}/{cam_key}/episode_{ep_idx:06d}.mp4"
        video_paths[cam_key] = str(vp) if vp.exists() else None
    # Task description
    task_desc = ""
    if "language_instruction" in df.columns:
        task_desc = str(df["language_instruction"].iloc[0])
    task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    task_token = f"task_{task_idx}"
    # Load tasks from metadata
    if not task_desc:
        meta_path = DATASET_ROOT / "meta" / "episodes.jsonl"
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    e = json.loads(line)
                    if e["episode_index"] == ep_idx:
                        task_desc = e["tasks"][0] if e["tasks"] else ""
                        break
    if not task_desc:
        task_desc = f"task_{task_idx}"
    return df, video_paths, task_desc, task_token, task_idx


def run_episode_inference(client, df, video_paths, prompt_text, episode_id):
    """Run inference at eval_steps points and return per-step L2 errors."""
    timestamps = df["timestamp"].to_numpy()
    ep_len = len(df)
    eval_steps = min(EVAL_STEPS, ep_len)
    eval_indices = np.linspace(0, ep_len - 1, eval_steps, dtype=int)

    step_errors = []
    for step_idx in eval_indices:
        # Reset
        try:
            client.reset({"session_id": f"cmp_{episode_id}_{step_idx}"})
        except Exception:
            client = WebsocketClientPolicy(host=HOST, port=PORT)
            client.reset({"session_id": f"cmp_{episode_id}_{step_idx}"})

        obs = {}
        for cam_key, video_path in video_paths.items():
            rk = ROBOARENA_KEY_MAP[cam_key]
            if video_path:
                try:
                    ts = np.array([timestamps[step_idx]])
                    frame = get_frames_by_timestamps(video_path, ts, video_backend="ffmpeg")
                    obs[rk] = frame[0]
                except Exception:
                    obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)
            else:
                obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)

        state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
        obs["observation/joint_position"] = state[7:14].astype(np.float64)
        obs["observation/gripper_position"] = state[6:7].astype(np.float64)
        obs["prompt"] = prompt_text
        obs["session_id"] = f"cmp_{episode_id}_{step_idx}"

        try:
            action_result = client.infer(obs)
        except Exception as e:
            print(f"      [FAIL] step {step_idx}: {e}")
            step_errors.append(float("inf"))
            continue

        if isinstance(action_result, dict):
            pred_action = action_result.get("action")
            if pred_action is None:
                for v in action_result.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        pred_action = v
                        break
        elif isinstance(action_result, np.ndarray):
            pred_action = action_result
        else:
            step_errors.append(float("inf"))
            continue

        if pred_action is None:
            step_errors.append(float("inf"))
            continue

        gt_actions = np.array(df["action"].iloc[step_idx], dtype=np.float64)
        gt_joint_pos = gt_actions[14:21]

        if pred_action.ndim == 2:
            pred_joint = pred_action[0, :7]
        elif pred_action.ndim == 1:
            pred_joint = pred_action[:7]
        else:
            step_errors.append(float("inf"))
            continue

        l2 = float(np.sqrt(np.sum((pred_joint - gt_joint_pos) ** 2)))
        step_errors.append(l2)

    return step_errors, client


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  {DATASET_NAME} — System1 vs Dual 快速对比 ({NUM_EPISODES} episodes)")
    print("=" * 70)
    sys.stdout.flush()

    with open(TEST_SET) as f:
        all_eps = json.load(f)
    episodes = all_eps[:NUM_EPISODES]

    client = WebsocketClientPolicy(host=HOST, port=PORT)
    print(f"✓ 已连接推理服务器 {HOST}:{PORT}\n")

    sys1_results = []
    dual_results = []

    for i, ep_info in enumerate(episodes, 1):
        ep_id = ep_info["episode_id"]
        df, video_paths, task_desc, task_token, task_idx = load_episode(ep_info)
        ep_len = len(df)

        print(f"━━━ Episode {i}/{NUM_EPISODES}: {ep_id} (len={ep_len}) ━━━")
        print(f"  Task: {task_desc[:100]}")
        sys.stdout.flush()

        # ── System1: task_description prompt ──
        # System1 uses task_{idx} as prompt (this is how the model was trained)
        prompt_sys1 = task_token  # e.g. "task_0"
        print(f"\n  【System1】 prompt = \"{prompt_sys1}\"")
        sys.stdout.flush()

        t0 = time.perf_counter()
        errors_s1, client = run_episode_inference(client, df, video_paths, prompt_sys1, ep_id)
        t1 = time.perf_counter()

        valid_s1 = [e for e in errors_s1 if e != float("inf")]
        mean_s1 = np.mean(valid_s1) if valid_s1 else float("inf")
        sr_s1 = np.mean([1.0 if e < ERROR_THRESHOLD else 0.0 for e in valid_s1]) if valid_s1 else 0.0

        print(f"    Per-step L2: {['%.4f' % e for e in errors_s1]}")
        print(f"    Mean L2 = {mean_s1:.4f}  |  SR@{ERROR_THRESHOLD} = {sr_s1:.1%}  |  time = {t1-t0:.1f}s")
        sys.stdout.flush()
        sys1_results.append({"ep": ep_id, "task": task_desc, "mean_l2": mean_s1, "sr": sr_s1, "steps": errors_s1})

        # ── Dual: hybrid prompt (task_description; subtask) ──
        print(f"\n  【Dual-hybrid】 LLM planning...")
        sys.stdout.flush()

        t0_plan = time.perf_counter()
        sub_instructions = llm_plan(task_desc)
        t1_plan = time.perf_counter()
        print(f"    Sub-tasks ({t1_plan - t0_plan:.2f}s):")
        for si, inst in enumerate(sub_instructions, 1):
            print(f"      {si}. {inst}")
        sys.stdout.flush()

        # Run inference with hybrid prompts (change prompt per sub-task phase)
        timestamps = df["timestamp"].to_numpy()
        ep_len_d = len(df)
        eval_steps = min(EVAL_STEPS, ep_len_d)
        eval_indices = np.linspace(0, ep_len_d - 1, eval_steps, dtype=int)

        errors_dual = []
        for step_idx in eval_indices:
            sub_idx = int(step_idx * len(sub_instructions) / ep_len_d)
            sub_idx = min(sub_idx, len(sub_instructions) - 1)
            current_sub = sub_instructions[sub_idx]
            # hybrid prompt: "task_description; subtask"
            hybrid_prompt = f"{task_desc}; {current_sub}"

            try:
                client.reset({"session_id": f"dual_{ep_id}_{step_idx}"})
            except Exception:
                client = WebsocketClientPolicy(host=HOST, port=PORT)
                client.reset({"session_id": f"dual_{ep_id}_{step_idx}"})

            obs = {}
            for cam_key, video_path in video_paths.items():
                rk = ROBOARENA_KEY_MAP[cam_key]
                if video_path:
                    try:
                        ts = np.array([timestamps[step_idx]])
                        frame = get_frames_by_timestamps(video_path, ts, video_backend="ffmpeg")
                        obs[rk] = frame[0]
                    except Exception:
                        obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)
                else:
                    obs[rk] = np.zeros((180, 320, 3), dtype=np.uint8)

            state = np.array(df["observation.state"].iloc[step_idx], dtype=np.float64)
            obs["observation/joint_position"] = state[7:14].astype(np.float64)
            obs["observation/gripper_position"] = state[6:7].astype(np.float64)
            obs["prompt"] = hybrid_prompt
            obs["session_id"] = f"dual_{ep_id}_{step_idx}"

            try:
                action_result = client.infer(obs)
            except Exception as e:
                print(f"      [FAIL] step {step_idx}: {e}")
                errors_dual.append(float("inf"))
                continue

            if isinstance(action_result, dict):
                pred_action = action_result.get("action")
                if pred_action is None:
                    for v in action_result.values():
                        if isinstance(v, np.ndarray) and v.ndim >= 1:
                            pred_action = v
                            break
            elif isinstance(action_result, np.ndarray):
                pred_action = action_result
            else:
                errors_dual.append(float("inf"))
                continue

            if pred_action is None:
                errors_dual.append(float("inf"))
                continue

            gt_actions = np.array(df["action"].iloc[step_idx], dtype=np.float64)
            gt_joint_pos = gt_actions[14:21]
            if pred_action.ndim == 2:
                pred_joint = pred_action[0, :7]
            elif pred_action.ndim == 1:
                pred_joint = pred_action[:7]
            else:
                errors_dual.append(float("inf"))
                continue

            l2 = float(np.sqrt(np.sum((pred_joint - gt_joint_pos) ** 2)))
            errors_dual.append(l2)

        valid_dual = [e for e in errors_dual if e != float("inf")]
        mean_dual = np.mean(valid_dual) if valid_dual else float("inf")
        sr_dual = np.mean([1.0 if e < ERROR_THRESHOLD else 0.0 for e in valid_dual]) if valid_dual else 0.0

        print(f"    Per-step L2: {['%.4f' % e for e in errors_dual]}")
        print(f"    Mean L2 = {mean_dual:.4f}  |  SR@{ERROR_THRESHOLD} = {sr_dual:.1%}")
        sys.stdout.flush()
        dual_results.append({"ep": ep_id, "task": task_desc, "mean_l2": mean_dual, "sr": sr_dual, "steps": errors_dual})

        # ── Per-episode comparison ──
        diff = mean_dual - mean_s1
        winner = "Dual" if diff < 0 else ("System1" if diff > 0 else "Tie")
        print(f"\n  >>> ΔL2 = {diff:+.4f}  ({winner} wins)")
        print()
        sys.stdout.flush()

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"  {DATASET_NAME} — 总结 ({NUM_EPISODES} episodes)")
    print("=" * 70)
    print(f"{'Episode':<20} {'System1 L2':>12} {'Dual L2':>12} {'ΔL2':>10} {'Winner':>10}")
    print("-" * 70)
    for s1, dl in zip(sys1_results, dual_results):
        diff = dl["mean_l2"] - s1["mean_l2"]
        w = "Dual" if diff < 0 else ("Sys1" if diff > 0 else "Tie")
        print(f"{s1['ep']:<20} {s1['mean_l2']:>12.4f} {dl['mean_l2']:>12.4f} {diff:>+10.4f} {w:>10}")

    all_s1 = np.mean([r["mean_l2"] for r in sys1_results])
    all_dual = np.mean([r["mean_l2"] for r in dual_results])
    all_sr_s1 = np.mean([r["sr"] for r in sys1_results])
    all_sr_dual = np.mean([r["sr"] for r in dual_results])
    print("-" * 70)
    print(f"{'AVERAGE':<20} {all_s1:>12.4f} {all_dual:>12.4f} {all_dual-all_s1:>+10.4f}")
    print(f"{'SR@' + str(ERROR_THRESHOLD):<20} {all_sr_s1:>12.1%} {all_sr_dual:>12.1%}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
