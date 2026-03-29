#!/usr/bin/env bash
# 在已启动的 PyTorch 容器内安装本项目依赖（不覆盖镜像自带 torch）。
# 适用基础镜像：dustynv/l4t-pytorch:r36.4.0（或 Dockerfile 中 ARG 指定的 L4T 标签），Python 3.10 与 pyproject 一致。
#
# 用法（项目根挂载为工作目录，例如 /workspace）::
#   bash scripts/install_ngc_container_deps.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# 勿升级 setuptools：避免移除 pkg_resources，导致镜像内 hypothesis/pytest 插件报错。
python3 -m pip install -U pip wheel
python3 -m pip install -r "${ROOT}/requirements-base.txt"
python3 -m pip install -e "${ROOT}[dev]"
echo "完成。可执行: python3 scripts/env_check.py"
