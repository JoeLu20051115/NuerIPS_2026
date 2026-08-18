#!/usr/bin/env python3
"""检查是否真的调用了OpenAI API"""
import json
import os
from pathlib import Path

# 检查1: 环境变量
api_key = os.environ.get("OPENAI_API_KEY", "")
print(f"✓ API Key set: {bool(api_key)}")
print(f"  Key starts with: {api_key[:20]}...")

# 检查2: 是否有API日志
log_dir = Path("evaluation_results_dualsystem")

# 检查3: 查看结果文件中的sub_instructions
result_file = log_dir / "L1_dualsystem_realapi_full_summary.json"
if result_file.exists():
    with open(result_file) as f:
        data = json.load(f)
        
    print(f"\n✓ 检查L1结果文件...")
    print(f"  Episodes: {data['num_episodes']}")
    print(f"  Method: {data['method']}")
    
    # 查看前3个episode的sub_instructions
    for i, ep in enumerate(data['individual_results'][:3]):
        subs = ep.get('sub_instructions', [])
        print(f"\n  Episode {i+1}:")
        print(f"    sub_instructions: {subs}")
        print(f"    num_sub_tasks: {ep.get('num_sub_tasks')}")

# 检查4: _mock_plan()应该返回什么
print("\n✓ Mock plan应该返回的格式:")
print("  返回: ['move to object', 'grasp object', 'lift object', 'move to target location', 'place object']")
print("\n✗ 但实际sub_instructions:")
print("  返回: ['task_2627']")
print("\n✓ 这意味着什么？")
print("  1. task_description被设置为'task_2627'")
print("  2. LLM (或mock)直接返回了这个单一的任务")
print("  3. 这不像是真实的API规划！")

