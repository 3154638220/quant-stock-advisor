#!/usr/bin/env python3
"""为 M8 Regime-Aware 混合打分 + indcap3 硬上限策略生成每月 Top20 选股 Markdown 报告。

输出目录：docs/reports/m8_indcap3_monthly/
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.reporting.m8_indcap3 import (
    CANDIDATE_POOL,
    MODEL_NAME,
    TOP_K,
    generate_monthly_reports,
    load_stock_names,
)

DEFAULT_HOLDINGS_CSV = ROOT / "data" / "results" / "monthly_selection_m8_concentration_regime_pitfix_2026_05_03_2026_05_03_topk_holdings.csv"
STOCK_NAMES_PATH = ROOT / "data" / "cache" / "a_share_stock_names.csv"
OUTPUT_DIR = ROOT / "docs" / "reports" / "m8_indcap3_monthly"


def run() -> None:
    parser = argparse.ArgumentParser(description="生成 M8 Regime-Aware + indcap3 月度 Top20 Markdown 报告")
    parser.add_argument("--holdings-csv", type=str, default=str(DEFAULT_HOLDINGS_CSV), help="M8 topk_holdings.csv 路径")
    parser.add_argument("--stock-names", type=str, default=str(STOCK_NAMES_PATH), help="股票名称映射 CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="策略模型名")
    parser.add_argument("--pool", type=str, default=CANDIDATE_POOL, help="候选池")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="TopK")
    args = parser.parse_args()

    holdings_csv = Path(args.holdings_csv)
    stock_names_path = Path(args.stock_names)
    output_dir = Path(args.output_dir)

    if not holdings_csv.exists():
        print(f"[ERROR] holdings 文件不存在: {holdings_csv}")
        return

    print(f"[m8-report] holdings: {holdings_csv}")
    print(f"[m8-report] 股票名称: {stock_names_path}")
    print(f"[m8-report] 输出目录: {output_dir}")
    print(f"[m8-report] 策略: {args.model} + {args.pool} + Top{args.top_k}")
    print()

    names = load_stock_names(stock_names_path)
    print(f"[m8-report] 股票名称映射: {len(names)} 条")
    print()

    generate_monthly_reports(
        holdings_csv=holdings_csv,
        names=names,
        output_dir=output_dir,
        model=args.model,
        pool=args.pool,
        top_k=args.top_k,
    )

    print("\n✅ 完成！")


if __name__ == "__main__":
    run()
