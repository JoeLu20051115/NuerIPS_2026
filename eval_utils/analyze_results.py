"""
评估结果分析脚本
用于计算准确率、生成可视化图表，并分析长短视时间范围

用法:
    python eval_utils/analyze_results.py --runs-dir runs/2024-01-01/12-00-00
    python eval_utils/analyze_results.py --runs-dir runs/ --action-horizon 24
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError as e:
    print(f"警告: 可视化库未安装 ({e})，将跳过图表生成")
    print("请安装: pip install numpy matplotlib seaborn")
    HAS_VISUALIZATION = False

if HAS_VISUALIZATION:
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def load_action_horizon_config() -> Dict:
    """从配置文件中加载 action_horizon 信息"""
    try:
        from groot.vla.utils.action_args_override_utils import update_action_horizon_configs
        from omegaconf import DictConfig
        
        # 尝试从配置文件读取默认值
        # 根据 README，默认 action_horizon=24
        default_horizon = 24
        return {
            "action_horizon": default_horizon,
            "num_frames_per_block": 2,  # 根据文档
            "control_frequency": 30,  # Hz，根据文档
            "chunk_duration": default_horizon / 30.0,  # 秒
        }
    except:
        # 如果无法导入，返回默认值
        return {
            "action_horizon": 24,
            "num_frames_per_block": 2,
            "control_frequency": 30,
            "chunk_duration": 24 / 30.0,
        }


def parse_episode_results(runs_dir: Path) -> Dict:
    """解析评估结果目录，提取episode信息"""
    results = {
        "episodes": [],
        "total_episodes": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "video_files": [],
    }
    
    if not runs_dir.exists():
        print(f"警告: 目录 {runs_dir} 不存在")
        return results
    
    # 查找所有视频文件
    video_files = list(runs_dir.glob("*.mp4"))
    results["video_files"] = [str(f) for f in video_files]
    results["total_episodes"] = len(video_files)
    
    # 尝试从文件名或目录结构推断episode信息
    for video_file in video_files:
        episode_info = {
            "episode_id": video_file.stem,
            "video_path": str(video_file),
            "success": None,  # 需要根据实际情况判断
        }
        results["episodes"].append(episode_info)
    
    # 如果有评估日志文件，尝试解析
    log_files = list(runs_dir.glob("*.json")) + list(runs_dir.glob("*.log"))
    for log_file in log_files:
        try:
            if log_file.suffix == ".json":
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                    # 根据实际日志格式解析
                    if "success" in log_data:
                        results["successful_episodes"] = log_data.get("success_count", 0)
                        results["failed_episodes"] = log_data.get("fail_count", 0)
        except:
            pass
    
    return results


def calculate_success_rate(results: Dict) -> Dict:
    """计算成功率"""
    total = results["total_episodes"]
    if total == 0:
        return {
            "success_rate": 0.0,
            "total_episodes": 0,
            "successful": 0,
            "failed": 0,
        }
    
    # 如果没有明确的成功/失败信息，假设需要手动标注或从其他来源获取
    successful = results.get("successful_episodes", 0)
    failed = results.get("failed_episodes", 0)
    
    # 如果成功+失败数不等于总数，使用估算
    if successful + failed != total and total > 0:
        # 这里可以添加更复杂的逻辑，比如从视频分析或日志推断
        # 暂时使用一个占位值
        print(f"注意: 无法确定所有episode的成功状态，总数: {total}")
    
    success_rate = (successful / total * 100) if total > 0 else 0.0
    
    return {
        "success_rate": success_rate,
        "total_episodes": total,
        "successful": successful,
        "failed": failed,
    }


def plot_success_rate_metrics(metrics: Dict, output_path: Path):
    """绘制成功率相关图表"""
    if not HAS_VISUALIZATION:
        print("跳过图表生成（缺少可视化库）")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 饼图：成功 vs 失败
    ax1 = axes[0]
    if metrics["total_episodes"] > 0:
        labels = ["成功", "失败"]
        sizes = [metrics["successful"], metrics["failed"]]
        colors = ["#2ecc71", "#e74c3c"]
        
        # 如果失败数为0，只显示成功
        if metrics["failed"] == 0:
            sizes = [metrics["successful"]]
            labels = ["成功"]
            colors = ["#2ecc71"]
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f"Episode 成功率分布\n(总计: {metrics['total_episodes']} episodes)", fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14)
        ax1.set_title("Episode 成功率分布", fontsize=12, fontweight='bold')
    
    # 柱状图：成功率
    ax2 = axes[1]
    if metrics["total_episodes"] > 0:
        categories = ["成功率"]
        values = [metrics["success_rate"]]
        colors_bar = ["#3498db"]
        
        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel("成功率 (%)", fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.set_title(f"总体成功率: {metrics['success_rate']:.2f}%", fontsize=12, fontweight='bold')
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=14)
        ax2.set_title("总体成功率", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "success_rate_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ 成功率图表已保存: {output_path / 'success_rate_metrics.png'}")


def plot_time_horizon_analysis(horizon_config: Dict, output_path: Path):
    """绘制时间范围分析图表"""
    if not HAS_VISUALIZATION:
        print("跳过图表生成（缺少可视化库）")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    action_horizon = horizon_config["action_horizon"]
    chunk_duration = horizon_config["chunk_duration"]
    control_freq = horizon_config["control_frequency"]
    
    # 1. Action Horizon 时间线
    ax1 = axes[0, 0]
    time_steps = np.arange(action_horizon)
    time_seconds = time_steps / control_freq
    
    ax1.plot(time_seconds, time_steps, 'o-', linewidth=2, markersize=6, color='#3498db')
    ax1.fill_between(time_seconds, 0, time_steps, alpha=0.3, color='#3498db')
    ax1.set_xlabel("时间 (秒)", fontsize=11)
    ax1.set_ylabel("Action Steps", fontsize=11)
    ax1.set_title(f"Action Horizon 时间范围\n(总时长: {chunk_duration:.2f}秒, {action_horizon}步)", 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 短视 vs 长视对比
    ax2 = axes[0, 1]
    short_horizon = action_horizon // 3  # 短视：前1/3
    long_horizon = action_horizon  # 长视：全部
    
    categories = ["短视范围", "长视范围"]
    horizons = [short_horizon, long_horizon]
    durations = [h / control_freq for h in horizons]
    colors = ["#e74c3c", "#2ecc71"]
    
    x = np.arange(len(categories))
    bars = ax2.bar(x, durations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("时间范围 (秒)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_title("短视 vs 长视时间范围对比", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, dur, hor) in enumerate(zip(bars, durations, horizons)):
        ax2.text(bar.get_x() + bar.get_width()/2., dur,
                f'{dur:.2f}s\n({hor}步)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 时间范围分布（按chunk）
    ax3 = axes[1, 0]
    num_chunks = 5  # 示例：显示5个chunk
    chunk_indices = np.arange(1, num_chunks + 1)
    chunk_times = [chunk_duration * i for i in chunk_indices]
    
    ax3.barh(chunk_indices, chunk_times, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel("累计时间 (秒)", fontsize=11)
    ax3.set_ylabel("Chunk 序号", fontsize=11)
    ax3.set_title(f"多 Chunk 累计时间范围\n(每个chunk: {chunk_duration:.2f}秒)", 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 控制频率与时间精度
    ax4 = axes[1, 1]
    time_precision = 1.0 / control_freq  # 每步的时间精度
    
    metrics_data = {
        "Action Horizon": action_horizon,
        "控制频率 (Hz)": control_freq,
        "Chunk时长 (s)": chunk_duration,
        "时间精度 (s)": time_precision,
    }
    
    y_pos = np.arange(len(metrics_data))
    values = list(metrics_data.values())
    labels = list(metrics_data.keys())
    
    bars = ax4.barh(y_pos, values, color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=10)
    ax4.set_xlabel("数值", fontsize=11)
    ax4.set_title("时间范围关键参数", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}' if val < 1 else f'{val:.1f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "time_horizon_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 时间范围分析图表已保存: {output_path / 'time_horizon_analysis.png'}")


def generate_summary_report(results: Dict, metrics: Dict, horizon_config: Dict, output_path: Path):
    """生成文本摘要报告"""
    report_path = output_path / "evaluation_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("DreamZero 评估结果报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【评估概览】\n")
        f.write("-" * 60 + "\n")
        f.write(f"总 Episode 数: {metrics['total_episodes']}\n")
        f.write(f"成功 Episode 数: {metrics['successful']}\n")
        f.write(f"失败 Episode 数: {metrics['failed']}\n")
        f.write(f"成功率: {metrics['success_rate']:.2f}%\n\n")
        
        f.write("【时间范围分析】\n")
        f.write("-" * 60 + "\n")
        f.write(f"Action Horizon: {horizon_config['action_horizon']} 步\n")
        f.write(f"控制频率: {horizon_config['control_frequency']} Hz\n")
        f.write(f"每个 Chunk 时长: {horizon_config['chunk_duration']:.3f} 秒\n")
        f.write(f"每帧时间: {1.0 / horizon_config['control_frequency']:.4f} 秒\n")
        f.write(f"短视范围 (前1/3): {horizon_config['action_horizon'] // 3} 步 "
                f"({horizon_config['action_horizon'] // 3 / horizon_config['control_frequency']:.3f} 秒)\n")
        f.write(f"长视范围 (全部): {horizon_config['action_horizon']} 步 "
                f"({horizon_config['chunk_duration']:.3f} 秒)\n\n")
        
        f.write("【输出文件】\n")
        f.write("-" * 60 + "\n")
        f.write(f"成功率图表: success_rate_metrics.png\n")
        f.write(f"时间范围分析: time_horizon_analysis.png\n")
        f.write(f"本报告: evaluation_report.txt\n")
        if results["video_files"]:
            f.write(f"\n视频文件 ({len(results['video_files'])} 个):\n")
            for i, video in enumerate(results["video_files"][:10], 1):  # 只显示前10个
                f.write(f"  {i}. {Path(video).name}\n")
            if len(results["video_files"]) > 10:
                f.write(f"  ... 还有 {len(results['video_files']) - 10} 个文件\n")
    
    print(f"✓ 评估报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 DreamZero 评估结果，生成准确率和可视化图表"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="评估结果目录路径 (例如: runs/2024-01-01/12-00-00)"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Action horizon 值 (默认从配置读取，通常为24)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认与runs-dir相同)"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"错误: 目录不存在: {runs_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DreamZero 评估结果分析")
    print("=" * 60)
    print(f"输入目录: {runs_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 1. 加载配置
    print("【步骤 1/4】加载配置...")
    horizon_config = load_action_horizon_config()
    if args.action_horizon:
        horizon_config["action_horizon"] = args.action_horizon
        horizon_config["chunk_duration"] = args.action_horizon / horizon_config["control_frequency"]
    print(f"  Action Horizon: {horizon_config['action_horizon']}")
    print(f"  控制频率: {horizon_config['control_frequency']} Hz")
    print(f"  Chunk 时长: {horizon_config['chunk_duration']:.3f} 秒\n")
    
    # 2. 解析结果
    print("【步骤 2/4】解析评估结果...")
    results = parse_episode_results(runs_dir)
    print(f"  找到 {results['total_episodes']} 个 episode")
    print(f"  视频文件: {len(results['video_files'])} 个\n")
    
    # 3. 计算指标
    print("【步骤 3/4】计算评估指标...")
    metrics = calculate_success_rate(results)
    print(f"  总 Episode 数: {metrics['total_episodes']}")
    print(f"  成功率: {metrics['success_rate']:.2f}%\n")
    
    # 4. 生成可视化
    print("【步骤 4/4】生成可视化图表...")
    if HAS_VISUALIZATION:
        plot_success_rate_metrics(metrics, output_dir)
        plot_time_horizon_analysis(horizon_config, output_dir)
    else:
        print("  跳过图表生成（缺少可视化库）")
    
    # 5. 生成报告
    print("\n生成摘要报告...")
    generate_summary_report(results, metrics, horizon_config, output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"  - success_rate_metrics.png (成功率图表)")
    print(f"  - time_horizon_analysis.png (时间范围分析)")
    print(f"  - evaluation_report.txt (文本报告)")


if __name__ == "__main__":
    main()
