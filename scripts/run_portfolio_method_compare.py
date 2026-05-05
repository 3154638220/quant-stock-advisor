#!/usr/bin/env python3
"""组合方法对比 CLI（薄层）—— 等权 vs 风险平价 vs 最小方差 side-by-side。

Usage:
    python scripts/run_portfolio_method_compare.py --config config.yaml
    python scripts/run_portfolio_method_compare.py --signal-weights weights.parquet \\
        --returns returns.parquet --output reports/portfolio_compare.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def _build_cov_weights(
    symbols: list[str],
    daily_df: pd.DataFrame,
    scores: np.ndarray,
    method: str,
    cov_config: dict,
    asof_dates: list[pd.Timestamp],
    industry_col: Optional[str] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """为每个截面日期构建协方差驱动的组合权重。"""
    from src.portfolio.covariance import mean_cov_returns_from_daily_long
    from src.portfolio.optimizer import solve_weights_from_cov_method

    n_symbols = len(symbols)
    weight_records: list[dict] = []
    solver_diags: list[dict] = []

    for asof in asof_dates:
        try:
            mu, cov = mean_cov_returns_from_daily_long(
                daily_df,
                symbols,
                asof=asof,
                lookback_days=int(cov_config.get("cov_lookback_days", 60)),
                ridge=float(cov_config.get("cov_ridge", 1e-6)),
                shrinkage=str(cov_config.get("cov_shrinkage", "ledoit_wolf")),
                industry_col=industry_col,
            )
        except Exception:
            mu = np.zeros(n_symbols, dtype=np.float64)
            cov = np.eye(n_symbols, dtype=np.float64)

        if method == "mean_variance":
            w, diag = solve_weights_from_cov_method(
                "mean_variance",
                cov,
                mu=mu,
                risk_aversion=float(cov_config.get("risk_aversion", 1.0)),
            )
        elif method == "min_variance":
            w, diag = solve_weights_from_cov_method("min_variance", cov)
        elif method == "risk_parity":
            w, diag = solve_weights_from_cov_method("risk_parity", cov)
        else:
            w = np.ones(n_symbols) / n_symbols
            diag = {"method": method}

        rec = {"date": asof}
        for i, sym in enumerate(symbols):
            rec[sym] = float(w[i]) if i < len(w) else 0.0
        weight_records.append(rec)
        solver_diags.append(diag)

    weights_df = pd.DataFrame(weight_records).set_index("date")
    weights_df.index = pd.to_datetime(weights_df.index)
    return weights_df, solver_diags


def main():
    parser = argparse.ArgumentParser(description="组合方法对比分析")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--signal-weights", type=str, default=None, help="等权/score 权重宽表 (parquet)")
    parser.add_argument("--returns", type=str, default=None, help="资产日收益宽表 (parquet)")
    parser.add_argument("--daily-data", type=str, default=None, help="日线长表 (parquet, 用于协方差估计)")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 路径")
    parser.add_argument("--json", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 如果提供了预计算权重，直接加载
    weights_equal = None
    weights_score = None
    returns_df = None
    daily_df = None

    if args.signal_weights:
        wdf = pd.read_parquet(args.signal_weights)
        # 假设列名包含方法标识
        weights_equal = wdf

    if args.returns:
        returns_df = pd.read_parquet(args.returns)

    if args.daily_data:
        daily_df = pd.read_parquet(args.daily_data)

    # 如果缺少必要数据，尝试从数据库构建
    if weights_equal is None or returns_df is None:
        from src.data_fetcher.db_manager import DuckDBManager
        db_path = config.get("database", {}).get("path", "data/stock_data.duckdb")
        if not Path(db_path).exists():
            print(f"错误: 数据库不存在，且未提供 --signal-weights/--returns", file=sys.stderr)
            sys.exit(1)

        print("从数据库构建回测数据...")
        # 这里需要完整的 pipeline 数据获取；简化处理：提示用户先运行 M5 生成权重
        print("提示: 请先运行 run_monthly_selection_multisource.py 生成信号权重，"
              "然后通过 --signal-weights 和 --returns 传入", file=sys.stderr)
        sys.exit(1)

    # 确保权重表有日期索引
    if weights_equal.index.name != "date" and "date" in weights_equal.columns:
        weights_equal = weights_equal.set_index("date")
    weights_equal.index = pd.to_datetime(weights_equal.index)

    # 构建 score-weight（从同一信号表使用 score 列加权）
    if weights_score is None:
        # 如果权重表本身就是等权的，score_weight = equal_weight（使用相同数据）
        weights_score = weights_equal.copy()

    symbols = [c for c in weights_equal.columns if c not in ("date", "index")]
    port_cfg = config.get("portfolio", {})

    # 构建协方差驱动的权重（需要 daily_df）
    weights_risk_parity = None
    weights_min_variance = None
    solver_diags = None

    if daily_df is not None and len(symbols) > 1:
        asof_dates = [pd.Timestamp(d) for d in weights_equal.index]
        print(f"构建风险平价权重 ({len(asof_dates)} 个月)...")
        weights_risk_parity, diags_rp = _build_cov_weights(
            symbols, daily_df, None, "risk_parity", port_cfg, asof_dates,
        )
        print(f"构建最小方差权重...")
        weights_min_variance, diags_mv = _build_cov_weights(
            symbols, daily_df, None, "min_variance", port_cfg, asof_dates,
        )
        solver_diags = {
            "risk_parity": diags_rp,
            "min_variance": diags_mv,
        }

    # 运行对比
    from src.backtest.engine import BacktestConfig
    from src.analysis.portfolio_method_compare import (
        compare_portfolio_methods,
        portfolio_method_markdown,
    )

    # 对齐权重和收益的日期
    common_dates = weights_equal.index.intersection(returns_df.index)
    weights_equal = weights_equal.reindex(common_dates)
    weights_score = weights_score.reindex(common_dates)
    if weights_risk_parity is not None:
        weights_risk_parity = weights_risk_parity.reindex(common_dates)
    if weights_min_variance is not None:
        weights_min_variance = weights_min_variance.reindex(common_dates)

    report = compare_portfolio_methods(
        returns_df.reindex(common_dates),
        weights_equal,
        weights_score=weights_score,
        weights_risk_parity=weights_risk_parity,
        weights_min_variance=weights_min_variance,
        solver_diag=solver_diags,
    )

    md = portfolio_method_markdown(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"Markdown 报告已写入: {out_path}")
    else:
        print(md)

    if args.json:
        import json
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 报告已写入: {json_path}")


if __name__ == "__main__":
    main()
