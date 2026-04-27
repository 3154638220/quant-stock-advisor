#!/usr/bin/env bash
# 顺序执行：Step1 网格 → Step3 universe 回测（最优 Top-K）→ Step4 F1 诊断
# 用法：在 Step2 fetch_fundamental 完成后执行；或本脚本会先等待 fetch PID 退出。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FETCH_LOG="${ROOT}/data/logs/step2_fetch_fundamental_2000.log"
WAIT_FETCH="${WAIT_FOR_FETCH:-1}"

# DuckDB 单写多读：以「能只读打开」为准，避免误等 shell 父 PID
if [[ "${WAIT_FETCH}" == "1" ]]; then
  echo "[pipeline] 等待 DuckDB 锁释放（fetch 完成后继续，最长约 6h）..."
  python - <<'PY'
import time
from pathlib import Path
import duckdb
db = Path("data/market.duckdb")
opened = False
# 2160 * 10s ≈ 6h
for i in range(2160):
    try:
        con = duckdb.connect(str(db), read_only=True)
        con.close()
        opened = True
        print("[pipeline] DuckDB 已可只读连接，继续。")
        break
    except Exception:
        if i % 6 == 0:
            print(f"[pipeline] 仍等待锁... ({i * 10}s)")
        time.sleep(10)
if not opened:
    raise SystemExit("[pipeline] 错误：6h 内仍无法只读打开 DuckDB，退出。")
PY
fi
if [[ -f "$FETCH_LOG" ]] && grep -q "基本面更新完成" "$FETCH_LOG" 2>/dev/null; then
  tail -3 "$FETCH_LOG" || true
fi

echo "=== Step1: Top-K 网格 (M, max_turnover=1.0) ==="
python scripts/run_backtest_eval.py --config config.yaml.backtest \
  --rebalance-rule M \
  --grid-search \
  --grid-topk-values 10,20,30,40,50 \
  --grid-max-turnover-values 1.0 \
  --grid-rebalance-rules M \
  --grid-search-out data/results/topk_grid_M_pipeline.csv \
  --json-report data/results/topk_grid_M_pipeline_main.json

echo "=== Step3: 解析最优 top_k 并生成 universe 配置 ==="
BEST_K="$(python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("data/results/topk_grid_M_pipeline.csv")
g = pd.read_csv(p)
g = g[g.get("sharpe_ratio", pd.Series(dtype=float)).notna()].copy()
if g.empty:
    print("20")  # 保守默认
else:
    g = g.sort_values("sharpe_ratio", ascending=False)
    print(int(g.iloc[0]["top_k"]))
PY
)"
echo "[pipeline] 网格最优 top_k=${BEST_K}"

python - <<PY
import yaml
from pathlib import Path
src = Path("config.yaml.backtest")
cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
cfg.setdefault("signals", {})["top_k"] = int("${BEST_K}")
cfg.setdefault("universe_filter", {})
cfg["universe_filter"]["enabled"] = True
cfg["universe_filter"]["min_amount_20d"] = 50_000_000
cfg["universe_filter"]["require_roe_ttm_positive"] = True
out = Path("data/results/config_universe_m24_pipeline.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")
print(f"[pipeline] 已写入 {out}")
PY

echo "=== Step3b: Universe 全样本回测 ==="
python scripts/run_backtest_eval.py \
  --config data/results/config_universe_m24_pipeline.yaml \
  --rebalance-rule M \
  --json-report data/results/universe_m24_topk_pipeline.json

echo "=== Step4: F1 + M2.4 IC 诊断（基本面已扩充后）==="
python scripts/diagnose_factor_ic.py --config config.yaml.backtest \
  --factors "ocf_to_net_profit,ocf_to_asset,gross_margin_delta,net_margin_stability,asset_turnover,lower_shadow_ratio,vol_to_turnover" \
  --f1-validate \
  --apply-universe-m2 \
  --out-json data/results/factor_ic_f1_pipeline.json \
  --out-csv data/results/factor_ic_f1_pipeline.csv

echo "[pipeline] 全部步骤完成。"
echo "  网格: data/results/topk_grid_M_pipeline.csv"
echo "  Universe 回测: data/results/universe_m24_topk_pipeline.json"
echo "  F1 诊断: data/results/factor_ic_f1_pipeline.json"
