#!/usr/bin/env bash
# 月度量化管线一键执行：数据更新 → 数据准备 → 信号 → 模型 → 评估 → 报告
#
# 用法:
#   bash scripts/run_pipeline_all_steps.sh --config config.yaml
#   bash scripts/run_pipeline_all_steps.sh --config config.yaml --steps fetch,dataset,signal,eval
#   bash scripts/run_pipeline_all_steps.sh --config config.yaml --skip-fetch
#
# P2-8: 任意步骤失败立即退出，并通过 src/notify/ 发送告警（若配置了 webhook）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
CONFIG="config.yaml"
STEPS="all"
SKIP_FETCH=false
WEBHOOK_URL="${PIPELINE_WEBHOOK_URL:-}"
DRY_RUN=false
WAIT_FETCH="${WAIT_FOR_FETCH:-1}"

usage() {
  cat <<'HELP'
用法: run_pipeline_all_steps.sh [选项]

选项:
  --config PATH         YAML 配置文件路径（默认 config.yaml）
  --steps LIST          要执行的步骤，逗号分隔（默认 all）
                        可选: fetch,dataset,signal,model,eval,report,monitor
  --skip-fetch          跳过数据拉取步骤（已有最新数据时使用）
  --webhook URL         企业微信 Webhook URL（失败时发送告警；也可用环境变量 PIPELINE_WEBHOOK_URL）
  --dry-run             仅打印将执行的步骤，不实际运行
  --no-wait-fetch       不等待 DuckDB 锁释放
  -h, --help            显示此帮助

步骤说明:
  fetch     - 拉取市场+基本面+资金流+股东数据（fetch_only.py）
  dataset   - 构建月度数据集（run_monthly_selection_dataset.py）
  signal    - 生成多源信号（run_monthly_selection_multisource.py）
  model     - 训练模型 & walk-forward 评估（run_monthly_selection_baselines.py）
  eval      - M8 自然化行业约束评估（run_monthly_selection_m8_natural_industry_constraints.py）
  report    - 生成 M7 月度推荐报告（run_monthly_selection_report.py）
  monitor   - IC 衰减监控 + OOS 追踪

环境变量:
  PIPELINE_WEBHOOK_URL    企业微信 Webhook URL（与 --webhook 等效）
  WAIT_FOR_FETCH          是否等待 DuckDB 锁（默认 1）
HELP
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-config.yaml}"; shift 2 ;;
    --config=*)
      CONFIG="${1#*=}"; shift ;;
    --steps)
      STEPS="${2:-all}"; shift 2 ;;
    --steps=*)
      STEPS="${1#*=}"; shift ;;
    --skip-fetch)
      SKIP_FETCH=true; shift ;;
    --webhook)
      WEBHOOK_URL="${2:-}"; shift 2 ;;
    --webhook=*)
      WEBHOOK_URL="${1#*=}"; shift ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    --no-wait-fetch)
      WAIT_FETCH=0; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "未知参数: $1"; usage ;;
  esac
done

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

_step_echo() { echo; echo "=== [pipeline] $* ==="; echo; }

_notify_failure() {
  local step="$1" msg="$2"
  echo "[pipeline] 步骤失败: ${step} — ${msg}" >&2
  if [[ -n "${WEBHOOK_URL}" ]]; then
    python - "${WEBHOOK_URL}" "${step}" "${msg}" "$(date '+%Y-%m-%d %H:%M:%S')" <<'PY' 2>/dev/null || true
import sys, json, requests
url, step, msg, ts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
content = f"## 量化管线告警\n> 步骤: **{step}**\n> 时间: {ts}\n> 错误: {msg}\n\n请检查管线日志。"
payload = {"msgtype": "markdown", "markdown": {"content": content}}
try:
    requests.post(url, json=payload, timeout=10)
except Exception:
    pass
PY
  fi
}

_should_run() {
  local step="$1"
  [[ "$STEPS" == "all" ]] && return 0
  [[ ",${STEPS}," == *",${step},"* ]] && return 0
  return 1
}

_run_or_skip() {
  local step="$1" label="$2"; shift 2
  if ! _should_run "$step"; then
    echo "[pipeline] 跳过: ${label}"
    return 0
  fi
  _step_echo "${label}"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@" || { _notify_failure "$step" "命令失败: $*"; exit 1; }
}

_wait_duckdb() {
  if [[ "${WAIT_FETCH}" != "1" ]]; then return 0; fi
  echo "[pipeline] 等待 DuckDB 锁释放（最长约 6h）..."
  python - <<'PY'
import time, duckdb
from pathlib import Path
db = Path("data/market.duckdb")
for i in range(2160):
    try:
        con = duckdb.connect(str(db), read_only=True)
        con.close()
        print("[pipeline] DuckDB 已可只读连接。")
        break
    except Exception:
        if i % 6 == 0:
            print(f"[pipeline] 仍等待锁... ({i * 10}s)")
        time.sleep(10)
else:
    raise SystemExit("[pipeline] 错误：6h 内仍无法只读打开 DuckDB。")
PY
}

# ── 执行 ──────────────────────────────────────────────────────────────────────

echo "[pipeline] 配置: ${CONFIG}"
echo "[pipeline] 步骤: ${STEPS}"
echo "[pipeline] 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# Step 1: 数据拉取
if _should_run "fetch" && [[ "$SKIP_FETCH" != "true" ]]; then
  _run_or_skip "fetch" "Step 1/5: 数据拉取 (fetch_only)" \
    python scripts/fetch_only.py --config "${CONFIG}"

  # 等待基本面拉取完成（如果独立进程运行）
  FETCH_LOG="${ROOT}/data/logs/step2_fetch_fundamental_2000.log"
  if [[ -f "$FETCH_LOG" ]] && grep -q "基本面更新完成" "$FETCH_LOG" 2>/dev/null; then
    tail -3 "$FETCH_LOG" || true
  fi
fi

_wait_duckdb

# Step 2: 月度数据集构建
_run_or_skip "dataset" "Step 2/5: 月度数据集构建" \
  python scripts/run_monthly_selection_dataset.py --config "${CONFIG}"

# Step 3: 多源信号 + M5 baseline
_run_or_skip "signal" "Step 3/5: 多源信号" \
  python scripts/run_monthly_selection_multisource.py --config "${CONFIG}"

# Step 4: M8 自然化行业约束评估（当前主线模型）
_run_or_skip "eval" "Step 4/5: M8 自然化行业约束评估" \
  python scripts/run_monthly_selection_m8_natural_industry_constraints.py --config "${CONFIG}"

# Step 5: M7 月度推荐报告
_run_or_skip "report" "Step 5/5: M7 月度推荐报告" \
  python scripts/run_monthly_selection_report.py --config "${CONFIG}"

# Step 6: IC 衰减监控（独立步骤，失败不阻断）
if _should_run "monitor"; then
  _step_echo "Step 6/6: IC 衰减监控"
  python - "${CONFIG}" <<'PY' || echo "[pipeline] IC 监控异常（非阻断）"
import sys
from src.features.ic_monitor import ICMonitor
monitor = ICMonitor(store_path="data/logs/ic_monitor.json", db_path="data/market.duckdb")
try:
    alerts = monitor.check_decay_alerts(window=20, threshold=0.03)
finally:
    monitor.close()
if alerts:
    print(f"[pipeline] IC 衰减告警: {len(alerts)} 个因子")
    for a in alerts:
        print(f"  - {a}")
    raise SystemExit(1)
print("[pipeline] IC 衰减检查通过。")
PY
fi

echo
echo "[pipeline] ========================================"
echo "[pipeline] 全部步骤完成"
echo "[pipeline] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[pipeline] 配置: ${CONFIG}"
echo "[pipeline] ========================================"
