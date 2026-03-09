#!/usr/bin/env python3
"""
Run remaining L3 dual-system evaluation episodes, skipping problematic ones.
"""

import json
import subprocess
import sys
from pathlib import Path
import time

def main():
    # L3 test episodes
    l3_episodes = [
        "episode_001395", "episode_001406", "episode_001407", "episode_001443",
        "episode_001466", "episode_001470", "episode_001499", "episode_001577",
        "episode_001614", "episode_001639", "episode_001669", "episode_001754",  # Skip #12
        "episode_001783", "episode_001798", "episode_001819", "episode_001827",
        "episode_001914", "episode_001938", "episode_001946", "episode_001992"
    ]
    
    # Episodes to skip (problematic)
    skip_episodes = {"episode_001754"}  # #12 卡死了
    
    # Episodes already completed
    results_dir = Path("/home/xingrui/lueq/NuerIPS_2026/evaluation_results_dualsystem")
    completed_episodes = set()
    if results_dir.exists():
        for d in results_dir.glob("dualsystem_mock_episode_*"):
            if (d / "evaluation_results.json").exists():
                # Extract episode ID from directory name
                parts = d.name.split("_")
                ep_id = parts[3]  # e.g., "001395"
                completed_episodes.add(f"episode_{ep_id}")
    
    # Filter: not completed, not skipped
    remaining = [ep for ep in l3_episodes 
                 if ep not in completed_episodes and ep not in skip_episodes]
    
    print(f"Already completed: {len(completed_episodes)} episodes")
    print(f"Skipping: {len(skip_episodes)} episodes")
    print(f"Remaining to run: {len(remaining)} episodes")
    print(f"Episodes: {remaining}\n")
    
    if not remaining:
        print("✓ All episodes completed or skipped!")
        return
    
    # Create a minimal test set for remaining episodes
    dataset_root = Path("/home/xingrui/lueq/NuerIPS_2026/data/droid_lerobot")
    test_set = []
    
    for ep in remaining:
        ep_num = ep.replace("episode_", "")
        chunk_num = str(int(ep_num) // 1000).zfill(3)
        path = f"data/droid_lerobot/data/chunk-{chunk_num}/{ep}.parquet"
        test_set.append({
            "episode_id": ep,
            "path": path
        })
    
    # Save test set
    test_set_file = Path("/tmp/l3_remaining_test_set.json")
    with open(test_set_file, 'w') as f:
        json.dump(test_set, f, indent=2)
    
    print(f"Created test set: {test_set_file}")
    print(f"Running {len(test_set)} episodes...\n")
    
    # Run evaluation
    cmd = [
        "conda", "run", "-n", "dreamzero",
        "python", "/home/xingrui/lueq/NuerIPS_2026/scripts/eval/run_dualsystem_evaluation.py",
        "--tier", "L3",
        "--mock-llm",
        "--host", "localhost",
        "--port", "8000",
        "--test-sets-dir", "/home/xingrui/lueq/NuerIPS_2026/test_sets_final"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
