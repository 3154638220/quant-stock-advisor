#!/usr/bin/env python3
"""用 DuckDB 为 prepared factors parquet 定向回填资金流特征。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.refresh_fund_flow_cache import refresh_fund_flow_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="定向刷新 prepared factors cache 中的 fund_flow 特征")
    p.add_argument("--input-cache", required=True, help="输入 prepared factors parquet")
    p.add_argument("--output-cache", default="", help="输出 parquet；为空则覆盖输入")
    p.add_argument("--duckdb-path", default="data/market.duckdb", help="主库 DuckDB 路径")
    p.add_argument("--flow-table", default="a_share_fund_flow", help="资金流表名")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_cache = Path(args.input_cache).resolve()
    output_cache = Path(args.output_cache).resolve() if str(args.output_cache).strip() else input_cache
    duckdb_path = Path(args.duckdb_path).resolve()

    refresh_fund_flow_cache(
        input_cache=input_cache,
        output_cache=output_cache,
        duckdb_path=duckdb_path,
        flow_table=args.flow_table,
    )


if __name__ == "__main__":
    main()
