#!/usr/bin/env bash
# 使用项目专用 Conda 环境 quant-system 执行命令（需已 conda env create -f environment.yml）。
# 若未安装 conda 或未创建该环境，则回退为当前 shell 的 python。
#
# 用法（在项目根目录）::
#   bash scripts/with_conda.sh python scripts/env_check.py
#   bash scripts/with_conda.sh python scripts/run_monthly_selection_dataset.py --config config.yaml.backtest
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx '^quant-system$'; then
    exec conda run -n quant-system --no-capture-output "$@"
  fi
fi

echo "with_conda: 未找到 conda 环境 quant-system，使用 PATH 中的解释器" >&2
exec "$@"
