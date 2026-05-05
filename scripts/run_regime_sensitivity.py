#!/usr/bin/env python3
"""Regime 参数敏感性网格分析 CLI（薄层）。

在 bull_return_threshold × bear_return_threshold 网格上评估 regime 分类边界
对因子权重的影响，输出 Markdown 报告供 M12 Promotion Package 使用。

Usage:
    python scripts/run_regime_sensitivity.py --config config.yaml
    python scripts/run_regime_sensitivity.py --benchmark 510300 --output reports/regime_sensitivity.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description="Regime 参数敏感性网格分析")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--benchmark", type=str, default=None, help="基准标的代码（默认从 config 读取）")
    parser.add_argument("--lookback-days", type=int, default=252 * 5, help="分析回看天数")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 路径")
    parser.add_argument("--json", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 连接数据库获取基准收益
    from src.data_fetcher.db_manager import DuckDBManager

    db_path = config.get("database", {}).get("path", "data/stock_data.duckdb")
    if not Path(db_path).exists():
        print(f"错误: 数据库不存在: {db_path}", file=sys.stderr)
        sys.exit(1)

    benchmark = args.benchmark or config.get("regime", {}).get("benchmark_symbol", "market_ew_proxy")

    from src.market.regime import get_benchmark_returns_from_db

    with DuckDBManager(db_path) as db:
        benchmark_series = get_benchmark_returns_from_db(
            db,
            benchmark,
            lookback_days=args.lookback_days,
        )

    if benchmark_series.empty:
        print("错误: 未能获取基准收益数据", file=sys.stderr)
        sys.exit(1)

    # 生成月度截面日期（每月最后一个交易日）
    benchmark_series.index = pd.to_datetime(benchmark_series.index)
    monthly_dates = benchmark_series.resample("ME").last().index.tolist()
    monthly_dates = [d for d in monthly_dates if d >= benchmark_series.index.min()]

    if len(monthly_dates) < 6:
        print(f"错误: 月度数据点不足 ({len(monthly_dates)} < 6)", file=sys.stderr)
        sys.exit(1)

    # 获取基准因子权重
    reg_cfg = config.get("regime", {})
    base_weights = reg_cfg.get("base_weights", {
        "momentum": 0.20,
        "reversal": 0.12,
        "short_reversal": 0.08,
        "realized_vol": 0.10,
        "atr": 0.05,
        "log_market_cap": 0.05,
        "recent_return": 0.05,
        "max_single_day_drop": 0.05,
        "bias_short": 0.05,
        "bias_long": 0.05,
    })

    from src.analysis.regime_sensitivity import regime_sensitivity_markdown, run_regime_sensitivity_grid

    report = run_regime_sensitivity_grid(
        benchmark_series,
        monthly_dates,
        base_weights,
    )

    md = regime_sensitivity_markdown(report)

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
