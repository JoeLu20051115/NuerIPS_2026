#!/usr/bin/env python3
"""
Offline demo: run LLMPlannerWithVAL on the first N DROID episodes.
No policy server required — just shows how the new planner works.

Usage:
  python scripts/eval/demo_val_planner_droid.py [--num-episodes 10] [--out results/val_planner_droid_demo.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable from any CWD
sys.path.insert(0, str(Path(__file__).parent))
from llm_planner_val import LLMPlannerWithVAL


def load_episodes(meta_path: Path, n: int):
    episodes = []
    with meta_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            episodes.append({
                "episode_id": f"episode_{rec['episode_index']:06d}",
                "task": rec.get("tasks", [""])[0] if rec.get("tasks") else "",
            })
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/droid_easy400_dualfavored_dreamzero"),
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation_results_dualsystem/val_planner_droid_demo.json"),
    )
    parser.add_argument("--validate-path", type=str, default="Validate")
    args = parser.parse_args()

    meta_path = args.dataset_root / "meta/episodes.jsonl"
    episodes = load_episodes(meta_path, args.num_episodes)
    planner = LLMPlannerWithVAL(
        model=args.model,
        temperature=0.0,
        validate_path=args.validate_path,
    )

    all_results = []
    for i, ep in enumerate(episodes, 1):
        task = ep["task"]
        print(f"\n[{i}/{len(episodes)}] {ep['episode_id']} | Task: {task}")
        result = planner.plan_structured(task)

        # Pretty-print the plan
        print(f"  VAL result  : {result['val_result']}")
        if result["val_errors"]:
            print(f"  VAL errors  : {result['val_errors']}")
        if result["corrected"]:
            print(f"  Corrected   : YES")
        print(f"  Plan ({len(result['steps'])} steps):")
        for j, step in enumerate(result["steps"]):
            print(
                f"    [{j+1}] {step['time'][0]:.2f}-{step['time'][1]:.2f}  "
                f"{step['action']}({', '.join(step['args'])})  —  {step['instruction']}"
            )

        entry = {
            "episode_id": ep["episode_id"],
            "task": task,
            "steps": result["steps"],
            "sub_instructions": result["sub_instructions"],
            "start_fractions": result["start_fractions"],
            "val_result": result["val_result"],
            "val_errors": result["val_errors"],
            "corrected": result["corrected"],
            "plan_time": round(result["plan_time"], 3),
        }
        all_results.append(entry)

    # Summary
    print("\n=== VAL Summary ===")
    from collections import Counter
    val_counts = Counter(r["val_result"] for r in all_results)
    for k, v in val_counts.items():
        print(f"  {k}: {v}/{len(all_results)}")
    corrected_count = sum(1 for r in all_results if r["corrected"])
    print(f"  corrected: {corrected_count}/{len(all_results)}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
