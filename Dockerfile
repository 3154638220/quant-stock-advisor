# Jetson / L4T 专用 PyTorch 容器（与 JetPack 6.x、L4T R36.4.x 线一致）。
# 本仓库目标设备示例：JetPack 6.2.1 / L4T R36.4.4（见 /etc/nv_tegra_release）。
#
# 说明：NGC 的 nvcr.io/nvidia/l4t-pytorch 长期以 r35.x 等标签为主；JetPack 6 / R36 常用
# jetson-containers 发布的 dustynv/l4t-pytorch（如 r36.4.0），与板端 JetPack 对齐。
# 若需其他 R36 变体可构建时覆盖：docker build --build-arg L4T_PYTORCH_TAG=r36.3.0-cu124 ...
ARG L4T_PYTORCH_TAG=r36.4.0
FROM dustynv/l4t-pytorch:${L4T_PYTORCH_TAG}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace

# 镜像已含 Jetson 版 PyTorch/CUDA；仅装业务依赖（勿用 requirements 覆盖 torch）
COPY requirements-base.txt pyproject.toml README.md ./
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests
# 勿升级 setuptools：镜像内 pytest/hypothesis 可能依赖 pkg_resources；setuptools≥82 已移除该模块。
# 使用 --index-url 覆盖基础镜像 pip.conf 中的 Jetson 镜像站（业务依赖走 PyPI；torch 仍来自基础镜像）。
RUN python3 -m pip install --index-url https://pypi.org/simple -U pip wheel \
    && python3 -m pip install --index-url https://pypi.org/simple -r requirements-base.txt \
    && python3 -m pip install --index-url https://pypi.org/simple -e ".[dev]"

# 默认配置（可被 bind-mount 覆盖）
COPY config.yaml ./

CMD ["bash"]
