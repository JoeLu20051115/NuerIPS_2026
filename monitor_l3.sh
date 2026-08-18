#!/bin/bash
echo "╔════════════════════════════════════════════════════╗"
echo "║     L3评测实时监控（每分钟自动更新）              ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

while true; do
  clear
  echo "╔════════════════════════════════════════════════════╗"
  echo "║     L3评测实时监控（每分钟自动更新）              ║"
  echo "╚════════════════════════════════════════════════════╝"
  echo ""
  
  # 检查进程
  proc_count=$(ps aux | grep -c "dualsystem_realapi_full.*L3")
  if [ $proc_count -gt 1 ]; then
    echo "✓ 评测进程: 运行中"
  else
    echo "✗ 评测进程: 已停止"
  fi
  
  # 检查Episode完成数
  count=$(ls -d evaluation_results_dualsystem/dualsystem_realapi_full_L3_episode_* 2>/dev/null | wc -l)
  percent=$((count * 100 / 20))
  bar=$(printf '█%.0s' $(seq 1 $percent) | sed 's/.\{50\}/&\n/g' | head -1)
  echo ""
  echo "进度: $count/20 episodes"
  echo "进度条: [$bar$(printf ' %.0s' $(seq $((50-percent)) 50))] $percent%"
  
  # 检查汇总文件
  if [ -f "evaluation_results_dualsystem/L3_dualsystem_realapi_full_summary.json" ]; then
    echo ""
    echo "✅ L3评测已完成！"
    break
  fi
  
  # 显示时间戳
  echo ""
  echo "上次更新: $(date '+%H:%M:%S')"
  echo "下次更新: 1分钟后..."
  
  sleep 60
done
