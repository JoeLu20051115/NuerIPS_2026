#!/bin/bash
# Quick test with real OpenAI API - 3 episodes mini evaluation

# ======================================
# 🔑 在这里设置你的API key
# ======================================
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"  # ← 修改这里！
# ======================================

cd /home/xingrui/lueq/NuerIPS_2026

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 DreamZero Real API Quick Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "配置:"
echo "  • Episodes: 3 (mini test)"
echo "  • LLM: GPT-4o-mini (真实API)"
echo "  • 预计时间: 5-8分钟"
echo "  • 预计成本: ~$0.01-0.02"
echo ""

# Verify API key is set
if [ "$OPENAI_API_KEY" = "sk-YOUR-KEY-HERE" ]; then
    echo "❌ 错误: 请先在脚本中设置你的OPENAI_API_KEY！"
    echo "   编辑文件: run_realapi_quicktest.sh"
  echo "   修改第7行: export OPENAI_API_KEY=\"<your-api-key>\""
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 错误: OPENAI_API_KEY未设置"
    exit 1
fi

echo "✓ API Key已设置: ${OPENAI_API_KEY:0:8}...**********"
echo ""
echo "开始评测..."
echo ""

# Create custom test set file
cat > /tmp/mini_realapi_l1.json << 'TESTSET'
[
  {"episode_id": "episode_001004", "path": "data/droid_lerobot/data/chunk-001/episode_001004.parquet"},
  {"episode_id": "episode_001069", "path": "data/droid_lerobot/data/chunk-001/episode_001069.parquet"},
  {"episode_id": "episode_001096", "path": "data/droid_lerobot/data/chunk-001/episode_001096.parquet"}
]
TESTSET

# Run evaluation with real API (NO --mock-llm flag!)
conda run -n dreamzero python scripts/eval/run_dualsystem_evaluation.py \
  --tier L1 \
  --host localhost \
  --port 8000 \
  --test-sets-dir /tmp \
  --method dualsystem_realapi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 评测完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "结果位置: evaluation_results_dualsystem/"
