#!/usr/bin/env bash
# 在项目根安装 pip 依赖、NVIDIA Jetson PyTorch wheel、以及可编辑安装 quant-system。
#
# Conda（推荐，环境名见 environment.yml）::
#   conda env create -f environment.yml    # 首次
#   conda activate quant-system
#   bash scripts/bootstrap_conda_env.sh
#
# 仅 venv（Python 须为 3.10.x，与 .python-version 一致）::
#   python3.10 -m venv .venv && source .venv/bin/activate
#   bash scripts/bootstrap_conda_env.sh
#
# 覆盖默认 wheel：export TORCH_WHEEL_URL='https://developer.download.nvidia.com/...whl'
#
# 需已激活目标环境（conda activate quant-system 或 source .venv/bin/activate）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${TORCH_WHEEL_URL:-}" ]]; then
  TORCH_WHEEL="${TORCH_WHEEL_URL}"
else
  WHEEL_FILE="${ROOT}/jetson/torch-wheel.url"
  if [[ ! -f "$WHEEL_FILE" ]]; then
    echo "未找到 ${WHEEL_FILE}；请设置环境变量 TORCH_WHEEL_URL 为 NVIDIA aarch64 wheel 的 URL" >&2
    exit 1
  fi
  TORCH_WHEEL="$(tr -d '[:space:]' < "$WHEEL_FILE")"
fi

python -m pip install -U pip
python -m pip install -r "${ROOT}/requirements-base.txt"
python -m pip install --no-cache-dir "${TORCH_WHEEL}"
python -m pip install -e "${ROOT}"

echo "PyTorch 校验:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

echo ""
echo "下一步：在项目根目录执行环境自检（须已 conda activate quant-system）："
echo "  python scripts/env_check.py"
