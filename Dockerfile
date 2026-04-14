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
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_TRUSTED_HOST=pypi.org \
    PIP_EXTRA_INDEX_URL=
WORKDIR /workspace

# 镜像已含 Jetson 版 PyTorch/CUDA；仅装业务依赖（勿用 requirements 覆盖 torch）
COPY requirements-base.txt pyproject.toml README.md config.yaml.example ./
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests
# 构建阶段 DNS：若 `docker build` 报 pip 无法解析 pypi.org，见 README「Docker」或 `bash scripts/docker_build_jetson.sh`。
# 基础镜像将 PIP_INDEX_URL 指向 jetson.webredirect，且用户级 pip.conf 含 NGC extra-index-url；
# 与 PyPI 混用在部分网络下会触发 SSL 错误，故删除用户 pip 配置并仅用 PyPI（torch 仍来自基础镜像层）。
# 勿升级 setuptools：镜像内 pytest/hypothesis 可能依赖 pkg_resources；setuptools≥82 已移除该模块。
RUN rm -f /root/.pip/pip.conf /root/.config/pip/pip.conf \
    && printf '[global]\nindex-url = https://pypi.org/simple\n' > /usr/pip.conf \
    && cp /usr/pip.conf /etc/pip.conf \
    && cp /usr/pip.conf /etc/xdg/pip/pip.conf \
    && python3 -m pip install -U pip wheel \
    && python3 -m pip install -r requirements-base.txt \
    && python3 -m pip install -e ".[dev]"

# 配置文件建议在运行时通过 bind-mount 或 QUANT_CONFIG 注入；
# 仓库内仅保留 config.yaml.example 模板，不在镜像里硬编码 config.yaml。

CMD ["bash"]
