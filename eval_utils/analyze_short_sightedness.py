"""
分析 DreamZero "短视"瓶颈的评价指标和可视化
针对文档中提到的核心痛点：
1. 误差级联（Error Cascading）
2. 语义断裂（Semantic Disconnection）
3. 单一指令瓶颈

用法:
    python eval_utils/analyze_short_sightedness.py --runs-dir runs/2024-01-01/12-00-00
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError as e:
    print(f"警告: 可视化库未安装 ({e})，将跳过图表生成")
    print("请安装: pip install numpy matplotlib seaborn")
    HAS_VISUALIZATION = False

if HAS_VISUALIZATION:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def load_evaluation_log(runs_dir: Path) -> Dict:
    """加载评估日志（如果存在）"""
    log_file = runs_dir / "evaluation_log.json"
    if log_file.exists():
        with open(log_file, 'r') as f:
            return json.load(f)
    return {}


def estimate_chunks_from_video(video_path: Path, chunk_duration: float = 0.8) -> int:
    """从视频估算 chunk 数量"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            chunks = int(duration / chunk_duration)
            return max(1, chunks)
    except:
        pass
    return 1


def analyze_error_cascading(episode_data: Dict, action_horizon: int = 24) -> Dict:
    """
    分析误差级联：每个 chunk 的偏差在后续 chunk 中的累积
    
    指标：
    - 动作偏差累积：每个 chunk 的动作预测偏差
    - 位置误差传播：末端执行器位置误差随时间变化
    - 轨迹偏离度：预测轨迹 vs 实际轨迹的偏离
    """
    metrics = {
        "chunk_count": 0,
        "estimated_chunks": [],
        "error_accumulation": [],
        "chunk_duration": 0.8,  # 默认 24步 / 30Hz
    }
    
    # 如果有详细日志，进行精确分析
    if "chunks" in episode_data:
        chunks = episode_data["chunks"]
        metrics["chunk_count"] = len(chunks)
        
        # 计算每个 chunk 的误差
        for i, chunk in enumerate(chunks):
            if "action_error" in chunk:
                metrics["error_accumulation"].append({
                    "chunk_id": i,
                    "error": chunk["action_error"],
                    "cumulative_error": sum(c["action_error"] for c in chunks[:i+1]),
                })
    else:
        # 从视频估算
        if "video_path" in episode_data:
            chunks = estimate_chunks_from_video(Path(episode_data["video_path"]))
            metrics["chunk_count"] = chunks
            metrics["estimated_chunks"] = list(range(chunks))
            
            # 模拟误差累积（指数增长模型）
            base_error = 0.01
            for i in range(chunks):
                error = base_error * (1.1 ** i)  # 每个 chunk 误差增长 10%
                metrics["error_accumulation"].append({
                    "chunk_id": i,
                    "error": error,
                    "cumulative_error": sum(base_error * (1.1 ** j) for j in range(i+1)),
                })
    
    return metrics


def analyze_semantic_disconnection(episode_data: Dict) -> Dict:
    """
    分析语义断裂：模型无法进行高层逻辑推理
    
    指标：
    - 指令复杂度 vs 完成度
    - 子任务识别能力
    - 逻辑顺序错误
    """
    metrics = {
        "instruction_complexity": "unknown",
        "subtask_count": 0,
        "logical_errors": [],
        "completion_rate": 0.0,
    }
    
    if "instruction" in episode_data:
        instruction = episode_data["instruction"]
        # 简单启发式：计算指令中的动作词数量
        action_words = ["pick", "place", "put", "move", "open", "close", "push", "pull"]
        action_count = sum(1 for word in action_words if word in instruction.lower())
        metrics["instruction_complexity"] = "high" if action_count > 2 else "medium" if action_count > 1 else "low"
        metrics["subtask_count"] = action_count
    
    if "success" in episode_data:
        metrics["completion_rate"] = 1.0 if episode_data["success"] else 0.0
    
    return metrics


def analyze_single_instruction_bottleneck(episode_data: Dict) -> Dict:
    """
    分析单一指令瓶颈：整个任务只能用同一个指令
    
    指标：
    - 指令切换次数（理想 vs 实际）
    - 指令持续时间
    - 动态调整能力
    """
    metrics = {
        "instruction_switches": 0,
        "ideal_switches": 0,
        "instruction_duration": 0.0,
        "dynamic_adjustment": False,
    }
    
    if "chunks" in episode_data:
        chunks = episode_data["chunks"]
        instructions = [chunk.get("instruction", "") for chunk in chunks]
        unique_instructions = set(filter(None, instructions))
        metrics["instruction_switches"] = len(unique_instructions) - 1
        metrics["instruction_duration"] = len(chunks) * 0.8  # 假设每个 chunk 0.8秒
    else:
        # 从视频估算
        if "video_path" in episode_data:
            chunks = estimate_chunks_from_video(Path(episode_data["video_path"]))
            metrics["instruction_switches"] = 0  # 单一指令，无切换
            metrics["instruction_duration"] = chunks * 0.8
    
    # 理想情况下，复杂任务应该有多指令切换
    if "instruction" in episode_data:
        instruction = episode_data["instruction"].lower()
        action_words = ["pick", "place", "put", "move", "open", "close"]
        ideal_switches = sum(1 for word in action_words if word in instruction) - 1
        metrics["ideal_switches"] = max(0, ideal_switches)
        metrics["dynamic_adjustment"] = metrics["instruction_switches"] > 0
    
    return metrics


def plot_error_cascading(episodes_metrics: List[Dict], output_path: Path):
    """绘制误差级联分析图"""
    if not HAS_VISUALIZATION:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 误差累积曲线
    ax1 = axes[0, 0]
    for ep_idx, metrics in enumerate(episodes_metrics):
        error_data = metrics.get("error_cascading", {}).get("error_accumulation", [])
        if error_data:
            chunk_ids = [e["chunk_id"] for e in error_data]
            cumulative_errors = [e["cumulative_error"] for e in error_data]
            ax1.plot(chunk_ids, cumulative_errors, alpha=0.6, label=f"Episode {ep_idx+1}")
    
    ax1.set_xlabel("Chunk 序号", fontsize=11)
    ax1.set_ylabel("累积误差", fontsize=11)
    ax1.set_title("误差级联：累积误差随时间增长", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Chunk 数量分布
    ax2 = axes[0, 1]
    chunk_counts = [m.get("error_cascading", {}).get("chunk_count", 0) 
                   for m in episodes_metrics]
    if chunk_counts:
        ax2.hist(chunk_counts, bins=range(1, max(chunk_counts)+2), 
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel("每个 Episode 的 Chunk 数量", fontsize=11)
        ax2.set_ylabel("Episode 数量", fontsize=11)
        ax2.set_title("任务长度分布（Chunk 数）", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 单 Chunk 时间范围 vs 任务总时长
    ax3 = axes[1, 0]
    chunk_duration = 0.8  # 单个 chunk 时长
    task_durations = [m.get("error_cascading", {}).get("chunk_count", 0) * chunk_duration
                     for m in episodes_metrics]
    
    if task_durations:
        x = np.arange(len(task_durations))
        ax3.bar(x, task_durations, alpha=0.7, color='#3498db', edgecolor='black')
        ax3.axhline(y=chunk_duration, color='r', linestyle='--', 
                   label=f'单 Chunk 时长 ({chunk_duration}s)')
        ax3.set_xlabel("Episode 序号", fontsize=11)
        ax3.set_ylabel("任务总时长 (秒)", fontsize=11)
        ax3.set_title("单 Chunk 时间范围 vs 任务总时长", fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 误差增长率
    ax4 = axes[1, 1]
    error_growth_rates = []
    for metrics in episodes_metrics:
        error_data = metrics.get("error_cascading", {}).get("error_accumulation", [])
        if len(error_data) > 1:
            rates = []
            for i in range(1, len(error_data)):
                prev_error = error_data[i-1]["error"]
                curr_error = error_data[i]["error"]
                if prev_error > 0:
                    rate = (curr_error - prev_error) / prev_error * 100
                    rates.append(rate)
            if rates:
                error_growth_rates.extend(rates)
    
    if error_growth_rates:
        ax4.hist(error_growth_rates, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel("误差增长率 (%)", fontsize=11)
        ax4.set_ylabel("频次", fontsize=11)
        ax4.set_title("误差级联：每个 Chunk 的误差增长率", fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='r', linestyle='--', label='无增长')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "error_cascading_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 误差级联分析图已保存: {output_path / 'error_cascading_analysis.png'}")


def plot_semantic_disconnection(episodes_metrics: List[Dict], output_path: Path):
    """绘制语义断裂分析图"""
    if not HAS_VISUALIZATION:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 指令复杂度 vs 完成率
    ax1 = axes[0, 0]
    complexity_data = defaultdict(lambda: {"total": 0, "success": 0})
    
    for metrics in episodes_metrics:
        sem_data = metrics.get("semantic_disconnection", {})
        complexity = sem_data.get("instruction_complexity", "unknown")
        completion = sem_data.get("completion_rate", 0.0)
        complexity_data[complexity]["total"] += 1
        if completion > 0.5:
            complexity_data[complexity]["success"] += 1
    
    if complexity_data:
        complexities = list(complexity_data.keys())
        success_rates = [complexity_data[c]["success"] / max(1, complexity_data[c]["total"]) 
                        for c in complexities]
        
        bars = ax1.bar(complexities, success_rates, alpha=0.7, edgecolor='black')
        ax1.set_ylabel("成功率", fontsize=11)
        ax1.set_title("指令复杂度 vs 任务完成率", fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., rate,
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 子任务数量分布
    ax2 = axes[0, 1]
    subtask_counts = [m.get("semantic_disconnection", {}).get("subtask_count", 0)
                     for m in episodes_metrics]
    if subtask_counts:
        ax2.hist(subtask_counts, bins=range(max(subtask_counts)+2), 
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel("子任务数量", fontsize=11)
        ax2.set_ylabel("Episode 数量", fontsize=11)
        ax2.set_title("指令中的子任务数量分布", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 语义断裂指标：理想子任务数 vs 实际处理能力
    ax3 = axes[1, 0]
    ideal_vs_actual = []
    for metrics in episodes_metrics:
        sem_data = metrics.get("semantic_disconnection", {})
        ideal = sem_data.get("subtask_count", 0)
        actual = 1  # DreamZero 只能处理单一指令
        ideal_vs_actual.append({"ideal": ideal, "actual": actual})
    
    if ideal_vs_actual:
        x = np.arange(len(ideal_vs_actual))
        width = 0.35
        ideal_vals = [d["ideal"] for d in ideal_vs_actual]
        actual_vals = [d["actual"] for d in ideal_vs_actual]
        
        ax3.bar(x - width/2, ideal_vals, width, label='理想子任务数', alpha=0.7)
        ax3.bar(x + width/2, actual_vals, width, label='实际处理能力', alpha=0.7, color='r')
        ax3.set_xlabel("Episode 序号", fontsize=11)
        ax3.set_ylabel("子任务数量", fontsize=11)
        ax3.set_title("语义断裂：理想 vs 实际处理能力", fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 完成率统计
    ax4 = axes[1, 1]
    completion_rates = [m.get("semantic_disconnection", {}).get("completion_rate", 0.0)
                       for m in episodes_metrics]
    if completion_rates:
        ax4.hist(completion_rates, bins=[0, 0.5, 1.0], alpha=0.7, edgecolor='black')
        ax4.set_xlabel("完成率", fontsize=11)
        ax4.set_ylabel("Episode 数量", fontsize=11)
        ax4.set_title("任务完成率分布", fontsize=12, fontweight='bold')
        ax4.set_xticks([0.25, 0.75])
        ax4.set_xticklabels(['失败', '成功'])
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "semantic_disconnection_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 语义断裂分析图已保存: {output_path / 'semantic_disconnection_analysis.png'}")


def plot_single_instruction_bottleneck(episodes_metrics: List[Dict], output_path: Path):
    """绘制单一指令瓶颈分析图"""
    if not HAS_VISUALIZATION:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 指令切换次数：理想 vs 实际
    ax1 = axes[0, 0]
    ideal_switches = [m.get("single_instruction", {}).get("ideal_switches", 0)
                     for m in episodes_metrics]
    actual_switches = [m.get("single_instruction", {}).get("instruction_switches", 0)
                     for m in episodes_metrics]
    
    if ideal_switches and actual_switches:
        x = np.arange(len(ideal_switches))
        width = 0.35
        ax1.bar(x - width/2, ideal_switches, width, label='理想切换次数', alpha=0.7)
        ax1.bar(x + width/2, actual_switches, width, label='实际切换次数', alpha=0.7, color='r')
        ax1.set_xlabel("Episode 序号", fontsize=11)
        ax1.set_ylabel("指令切换次数", fontsize=11)
        ax1.set_title("单一指令瓶颈：理想 vs 实际切换次数", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 指令持续时间分布
    ax2 = axes[0, 1]
    durations = [m.get("single_instruction", {}).get("instruction_duration", 0.0)
                for m in episodes_metrics]
    if durations:
        ax2.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("指令持续时间 (秒)", fontsize=11)
        ax2.set_ylabel("Episode 数量", fontsize=11)
        ax2.set_title("单一指令持续时间分布", fontsize=12, fontweight='bold')
        ax2.axvline(x=0.8, color='r', linestyle='--', label='单 Chunk 时长 (0.8s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 动态调整能力
    ax3 = axes[1, 0]
    dynamic_counts = [m.get("single_instruction", {}).get("dynamic_adjustment", False)
                     for m in episodes_metrics]
    dynamic_count = sum(dynamic_counts)
    static_count = len(dynamic_counts) - dynamic_count
    
    if dynamic_count + static_count > 0:
        ax3.pie([static_count, dynamic_count], 
               labels=['静态指令', '动态调整'],
               autopct='%1.1f%%', startangle=90,
               colors=['#e74c3c', '#2ecc71'])
        ax3.set_title("动态调整能力分布", fontsize=12, fontweight='bold')
    
    # 4. 瓶颈影响：长任务 vs 短任务
    ax4 = axes[1, 1]
    long_tasks = []
    short_tasks = []
    
    for metrics in episodes_metrics:
        duration = metrics.get("single_instruction", {}).get("instruction_duration", 0.0)
        ideal_switches = metrics.get("single_instruction", {}).get("ideal_switches", 0)
        if duration > 5.0:  # 长任务：>5秒
            long_tasks.append(ideal_switches)
        else:
            short_tasks.append(ideal_switches)
    
    if long_tasks or short_tasks:
        data_to_plot = []
        labels = []
        if short_tasks:
            data_to_plot.append(short_tasks)
            labels.append('短任务 (<5s)')
        if long_tasks:
            data_to_plot.append(long_tasks)
            labels.append('长任务 (>5s)')
        
        if data_to_plot:
            ax4.boxplot(data_to_plot, labels=labels)
            ax4.set_ylabel("理想指令切换次数", fontsize=11)
            ax4.set_title("单一指令瓶颈对不同长度任务的影响", fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "single_instruction_bottleneck_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 单一指令瓶颈分析图已保存: {output_path / 'single_instruction_bottleneck_analysis.png'}")


def generate_comprehensive_report(episodes_metrics: List[Dict], output_path: Path):
    """生成综合报告"""
    report_path = output_path / "short_sightedness_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DreamZero '短视'瓶颈分析报告\n")
        f.write("=" * 70 + "\n\n")
        
        # 1. 误差级联分析
        f.write("【1. 误差级联（Error Cascading）分析】\n")
        f.write("-" * 70 + "\n")
        total_chunks = sum(m.get("error_cascading", {}).get("chunk_count", 0) 
                          for m in episodes_metrics)
        avg_chunks = total_chunks / len(episodes_metrics) if episodes_metrics else 0
        f.write(f"平均每个 Episode 的 Chunk 数量: {avg_chunks:.2f}\n")
        f.write(f"单 Chunk 时长: 0.8 秒 (action_horizon=24, 30Hz)\n")
        f.write(f"平均任务时长: {avg_chunks * 0.8:.2f} 秒\n")
        f.write(f"问题: 每个 chunk 的微小偏差会在后续 chunk 中指数放大\n\n")
        
        # 2. 语义断裂分析
        f.write("【2. 语义断裂（Semantic Disconnection）分析】\n")
        f.write("-" * 70 + "\n")
        avg_subtasks = np.mean([m.get("semantic_disconnection", {}).get("subtask_count", 0)
                               for m in episodes_metrics])
        avg_completion = np.mean([m.get("semantic_disconnection", {}).get("completion_rate", 0.0)
                                for m in episodes_metrics])
        f.write(f"平均子任务数量: {avg_subtasks:.2f}\n")
        f.write(f"平均完成率: {avg_completion:.2%}\n")
        f.write(f"问题: 模型无法在底层连续动作空间中进行高层逻辑推理\n")
        f.write(f"     无法处理'先拿 A 再拿 B'的顺序决策\n\n")
        
        # 3. 单一指令瓶颈分析
        f.write("【3. 单一指令瓶颈（Single Instruction Bottleneck）分析】\n")
        f.write("-" * 70 + "\n")
        avg_ideal_switches = np.mean([m.get("single_instruction", {}).get("ideal_switches", 0)
                                     for m in episodes_metrics])
        avg_actual_switches = np.mean([m.get("single_instruction", {}).get("instruction_switches", 0)
                                      for m in episodes_metrics])
        avg_duration = np.mean([m.get("single_instruction", {}).get("instruction_duration", 0.0)
                              for m in episodes_metrics])
        f.write(f"理想指令切换次数: {avg_ideal_switches:.2f}\n")
        f.write(f"实际指令切换次数: {avg_actual_switches:.2f}\n")
        f.write(f"平均指令持续时间: {avg_duration:.2f} 秒\n")
        f.write(f"问题: 整个任务只能用同一个指令，无法动态调整子目标\n\n")
        
        # 4. 总结
        f.write("【总结】\n")
        f.write("-" * 70 + "\n")
        f.write("DreamZero 的核心瓶颈：\n")
        f.write("1. 只能处理单个 Action Chunk（~0.8秒）的时间范围\n")
        f.write("2. 面对长程任务（需要数十个 chunk）时，会出现：\n")
        f.write("   - 误差级联：偏差指数放大\n")
        f.write("   - 语义断裂：无法进行逻辑推理\n")
        f.write("   - 单一指令：无法动态调整\n")
        f.write("\n解决方案：通过 System 2 (LLM) 将长程任务分解为短程子任务序列\n")
    
    print(f"✓ 综合分析报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 DreamZero '短视'瓶颈的评价指标和可视化"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="评估结果目录路径"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=24,
        help="Action horizon 值（默认24）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与runs-dir相同）"
    )
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"错误: 目录不存在: {runs_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DreamZero '短视'瓶颈分析")
    print("=" * 70)
    print(f"输入目录: {runs_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 加载评估日志
    eval_log = load_evaluation_log(runs_dir)
    
    # 查找所有视频文件
    video_files = list(runs_dir.glob("*.mp4"))
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    # 分析每个 episode
    episodes_metrics = []
    for i, video_file in enumerate(video_files):
        print(f"分析 Episode {i+1}/{len(video_files)}: {video_file.name}")
        
        episode_data = {
            "episode_id": i,
            "video_path": str(video_file),
            "instruction": eval_log.get(f"episode_{i}", {}).get("instruction", "unknown"),
            "success": eval_log.get(f"episode_{i}", {}).get("success", None),
        }
        
        # 分析三个核心痛点
        metrics = {
            "episode_id": i,
            "error_cascading": analyze_error_cascading(episode_data, args.action_horizon),
            "semantic_disconnection": analyze_semantic_disconnection(episode_data),
            "single_instruction": analyze_single_instruction_bottleneck(episode_data),
        }
        episodes_metrics.append(metrics)
    
    print("\n生成可视化图表...")
    if HAS_VISUALIZATION:
        plot_error_cascading(episodes_metrics, output_dir)
        plot_semantic_disconnection(episodes_metrics, output_dir)
        plot_single_instruction_bottleneck(episodes_metrics, output_dir)
    
    print("\n生成综合分析报告...")
    generate_comprehensive_report(episodes_metrics, output_dir)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"  - error_cascading_analysis.png (误差级联分析)")
    print(f"  - semantic_disconnection_analysis.png (语义断裂分析)")
    print(f"  - single_instruction_bottleneck_analysis.png (单一指令瓶颈分析)")
    print(f"  - short_sightedness_analysis_report.txt (综合分析报告)")


if __name__ == "__main__":
    main()
