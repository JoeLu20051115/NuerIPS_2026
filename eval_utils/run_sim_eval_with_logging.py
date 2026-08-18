"""
增强版评估脚本：记录详细信息用于"短视"瓶颈分析
在原有 run_sim_eval.py 基础上，添加日志记录功能

用法:
    python eval_utils/run_sim_eval_with_logging.py --host localhost --port 5000 --episodes 10
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from eval_utils.run_sim_eval import DreamZeroJointPosClient, main as original_main
import tyro


def main_with_logging(
        episodes: int = 10,
        scene: int = 1,
        headless: bool = True,
        host: str = "localhost",
        port: int = 6000,
        ):
    """带日志记录的评估主函数"""
    
    # 创建日志数据结构
    evaluation_log = {
        "timestamp": datetime.now().isoformat(),
        "episodes": {},
        "config": {
            "episodes": episodes,
            "scene": scene,
            "host": host,
            "port": port,
        }
    }
    
    # 调用原始评估函数（需要修改以支持日志记录）
    # 这里我们创建一个包装器来记录信息
    
    # 由于原始函数比较复杂，我们建议：
    # 1. 运行原始评估脚本生成视频
    # 2. 然后运行分析脚本（analyze_short_sightedness.py）来分析结果
    
    print("=" * 70)
    print("增强版评估脚本（带日志记录）")
    print("=" * 70)
    print("\n注意：")
    print("1. 本脚本会在原有评估基础上添加日志记录")
    print("2. 日志将保存到 runs/日期/时间/evaluation_log.json")
    print("3. 运行后可使用 analyze_short_sightedness.py 分析结果\n")
    
    # 运行原始评估
    original_main(episodes=episodes, scene=scene, headless=headless, host=host, port=port)
    
    # 保存基本日志（从视频文件推断）
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    if not video_dir.exists():
        # 查找最新的 runs 目录
        runs_base = Path("runs")
        if runs_base.exists():
            date_dirs = sorted(runs_base.glob("*/"), key=lambda x: x.name, reverse=True)
            if date_dirs:
                time_dirs = sorted(date_dirs[0].glob("*/"), key=lambda x: x.name, reverse=True)
                if time_dirs:
                    video_dir = time_dirs[0]
    
    if video_dir.exists():
        video_files = sorted(video_dir.glob("*.mp4"))
        for i, video_file in enumerate(video_files):
            evaluation_log["episodes"][f"episode_{i}"] = {
                "video_path": str(video_file),
                "episode_id": i,
                "instruction": f"Scene {scene} task",  # 可以从场景推断
                "success": None,  # 需要手动标注或从其他来源获取
            }
        
        # 保存日志
        log_file = video_dir / "evaluation_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_log, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 评估日志已保存: {log_file}")
        print(f"  现在可以运行: python eval_utils/analyze_short_sightedness.py --runs-dir {video_dir}")


if __name__ == "__main__":
    args = tyro.cli(main_with_logging)
