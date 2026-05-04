#!/usr/bin/env python3
"""M8: 月度选股行业集中度约束与 lagged-regime 复核 —— 精简 CLI 入口。

核心算法已迁移至 src/pipeline/monthly_concentration.py 与 src/cli/monthly_concentration.py。
本文件仅保留 CLI 参数解析与 main() 编排。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.monthly_concentration import TOPK_PRESET_DEFAULT, TOPK_PRESETS
from src.cli.monthly_concentration import run_monthly_concentration_regime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M8 月度选股行业集中度与 lagged-regime 治理")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m8_concentration_regime")
    p.add_argument("--as-of-date", type=str, default="")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--topk-preset", type=str, default=TOPK_PRESET_DEFAULT, choices=tuple(TOPK_PRESETS.keys()))
    p.add_argument("--top-k", type=str, default="")
    p.add_argument("--cap-grid", type=str, default="")
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--max-fit-rows", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument("--model-n-jobs", type=int, default=0)
    p.add_argument("--min-state-history-months", type=int, default=24)
    p.add_argument("--families", type=str, default="industry_breadth,fund_flow,fundamental")
    p.add_argument("--skip-m6", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run_monthly_concentration_regime(
        config=args.config,
        dataset=args.dataset,
        duckdb_path=args.duckdb_path,
        output_prefix=args.output_prefix,
        as_of_date=args.as_of_date,
        results_dir=args.results_dir,
        topk_preset=args.topk_preset,
        top_k=args.top_k,
        cap_grid=args.cap_grid,
        candidate_pools=args.candidate_pools,
        bucket_count=args.bucket_count,
        min_train_months=args.min_train_months,
        min_train_rows=args.min_train_rows,
        max_fit_rows=args.max_fit_rows,
        cost_bps=args.cost_bps,
        random_seed=args.random_seed,
        availability_lag_days=args.availability_lag_days,
        model_n_jobs=args.model_n_jobs,
        min_state_history_months=args.min_state_history_months,
        families=args.families,
        skip_m6=args.skip_m6,
        root=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
