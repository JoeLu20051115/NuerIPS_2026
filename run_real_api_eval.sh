#!/bin/bash

if [ -z "$OPENAI_API_KEY" ]; then
  echo "❌ 错误: OPENAI_API_KEY 未设置。"
  echo "请先执行: export OPENAI_API_KEY=\"<your-api-key>\""
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ 环境变量验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 << 'PYEOF'
import os
api_key = os.environ.get("OPENAI_API_KEY", "")
print(f"API Key已设置: {bool(api_key)}")
if api_key:
  print(f"Key开头: {api_key[:8]}...********")
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ 清空旧结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rm -rf evaluation_results_dualsystem/dualsystem_realapi_full_episode_*
rm -f evaluation_results_dualsystem/L1_dualsystem_realapi_full_summary.json
rm -f evaluation_results_dualsystem/L3_dualsystem_realapi_full_summary.json
echo "✓ 已清空"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动L1评测（真实GPT-4o-mini API）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
conda run -n dreamzero python scripts/eval/run_dualsystem_evaluation.py \
  --tier L1 \
  --host localhost \
  --port 8000 \
  --test-sets-dir test_sets_final \
  2>&1
