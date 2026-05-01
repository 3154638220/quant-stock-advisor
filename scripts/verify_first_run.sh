#!/usr/bin/env bash
# 首跑烟测：固定两只标的（不依赖全市场 spot 接口），验证 DuckDB 增量与月度选股入口。
# 用法：在项目根目录；优先使用 conda 环境 quant-system（与 environment.yml 一致）：
#   cd "$(dirname "$0")/.." && bash scripts/verify_first_run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN=(python)
if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx '^quant-system$'; then
    RUN=(conda run -n quant-system --no-capture-output python)
  fi
fi

SYMS="${VERIFY_SYMBOLS:-600519,000001}"

"${RUN[@]}" scripts/fetch_only.py --symbols "${SYMS}"
"${RUN[@]}" scripts/run_monthly_selection_dataset.py --config config.yaml.backtest --dry-run

echo "OK: DuckDB 增量写入与月度选股 dataset 入口可用"
