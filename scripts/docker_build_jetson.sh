#!/usr/bin/env bash
# 构建 Jetson / L4T 用镜像（仓库根 Dockerfile）。
# 默认使用宿主机网络，避免部分环境下 build 阶段 DNS 解析失败（Errno -3）。
#
# 用法（项目根目录）::
#   bash scripts/docker_build_jetson.sh
#   IMAGE_TAG=my-jetson-pytorch:latest bash scripts/docker_build_jetson.sh
#
# 环境变量::
#   IMAGE_TAG              镜像名:标签，默认 my-jetson-pytorch:latest
#   DOCKERFILE             默认 Dockerfile
#   DOCKER_BUILD_NETWORK   默认 host；若需桥接网络可设为 default（需宿主机 Docker 已配置 dns）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE_TAG="${IMAGE_TAG:-my-jetson-pytorch:latest}"
DOCKERFILE="${DOCKERFILE:-${ROOT}/Dockerfile}"
BUILD_NET="${DOCKER_BUILD_NETWORK:-host}"

exec docker build --network="$BUILD_NET" -f "$DOCKERFILE" -t "$IMAGE_TAG" "$ROOT"
