#!/usr/bin/env bash
# 在 L4T / Jetson 专用 PyTorch 容器（dustynv/l4t-pytorch，默认 r36.4.0）内构建与运行。
#
# 用法（项目根目录）::
#   bash scripts/docker_build_jetson.sh     # 仅构建镜像（推荐；默认 --network=host）
#   bash scripts/docker_run.sh              # 构建（同上网络策略）并可选进入/烟测
#   bash scripts/docker_run.sh --verify     # 构建后跑 verify_first_run（需网络）
#   bash scripts/docker_run.sh -- bash      # 进入交互 shell
#
# 环境变量::
#   IMAGE_NAME   默认 quant-system:l4t-r36.4
#   DOCKERFILE   默认项目根 Dockerfile
#
# GPU：需 nvidia-container-toolkit；Jetson 上若 --gpus 不可用，可试 --runtime=nvidia。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE_NAME="${IMAGE_NAME:-quant-system:l4t-r36.4}"
DOCKERFILE="${DOCKERFILE:-${ROOT}/Dockerfile}"

RUN_ARGS=()
if docker info 2>/dev/null | grep -q nvidia; then
  RUN_ARGS+=(--gpus all)
elif [[ -e /usr/bin/nvidia-container-runtime ]] || [[ -e /usr/bin/nvidia-docker ]]; then
  RUN_ARGS+=(--gpus all)
fi

# 部分环境默认 build 网络无 DNS，pip 会解析失败；可用 DOCKER_BUILD_NETWORK=default 强制桥接。
BUILD_NET="${DOCKER_BUILD_NETWORK:-host}"
docker build --network="$BUILD_NET" -f "$DOCKERFILE" -t "$IMAGE_NAME" "$ROOT"

VOL=( -v "${ROOT}:/workspace" -w /workspace )
ENV=( -e PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" )

if [[ "${1:-}" == "--verify" ]]; then
  shift || true
  exec docker run --rm -it "${RUN_ARGS[@]}" "${VOL[@]}" "${ENV[@]}" "$IMAGE_NAME" \
    bash -lc 'bash scripts/verify_first_run.sh'
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  exec docker run --rm -it "${RUN_ARGS[@]}" "${VOL[@]}" "${ENV[@]}" "$IMAGE_NAME" "$@"
fi

echo "镜像已构建: $IMAGE_NAME"
echo "交互进入: bash scripts/docker_run.sh -- bash"
echo "环境自检: docker run --rm ${RUN_ARGS[*]} -v ${ROOT}:/workspace -w /workspace $IMAGE_NAME python3 scripts/env_check.py"
echo "烟测:     bash scripts/docker_run.sh --verify"
