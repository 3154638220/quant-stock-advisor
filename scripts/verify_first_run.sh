#!/usr/bin/env bash
# 首跑烟测：固定两只标的（不依赖全市场 spot 接口），验证 DuckDB 增量与推荐 CSV。
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
"${RUN[@]}" scripts/daily_run.py --symbols "${SYMS}" --skip-fetch --top-k 10

LATEST="$(ls -1t "${ROOT}/data/results"/recommend_*.csv 2>/dev/null | head -1)"
if [[ -z "${LATEST}" ]]; then
  echo "未找到 data/results/recommend_*.csv" >&2
  exit 1
fi
echo "OK: ${LATEST}"
wc -l "${LATEST}"
