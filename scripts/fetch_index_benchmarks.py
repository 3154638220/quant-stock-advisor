#!/usr/bin/env python3
"""Fetch investable/index benchmark daily opens for benchmark suite.

The output CSV is intentionally narrow and matches
``scripts/run_monthly_benchmark_suite.py --index-csv``:

    trade_date, open, symbol

By default it fetches CSI 1000 and CSI 2000 from AkShare/Eastmoney.

（A3 迁移后：核心规格与标准化函数已迁入 src/data_fetcher/index_benchmarks.py，
本脚本仅保留 CLI + fetch/build 管道。）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.index_benchmarks import (
    DEFAULT_INDEX_SPECS,
    IndexFetchSpec,
    parse_index_specs,
    standardize_index_daily,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载 benchmark suite 指数基准 CSV")
    p.add_argument("--start-date", type=str, default="20210101")
    p.add_argument("--end-date", type=str, default="")
    p.add_argument("--output", type=str, default="data/cache/index_benchmarks.csv")
    p.add_argument(
        "--index",
        action="append",
        default=[],
        help=(
            "自定义指数，格式 name:output_symbol:akshare_symbol。"
            "例：csi1000:000852:sh000852。未提供时使用中证1000/2000。"
        ),
    )
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


# ── parse_index_specs / standardize_index_daily 已迁入 src/data_fetcher/index_benchmarks.py ──


def fetch_index_daily(spec: IndexFetchSpec, *, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    raw: Any = ak.stock_zh_index_daily_em(
        symbol=spec.akshare_symbol,
        start_date=start_date,
        end_date=end_date,
    )
    return standardize_index_daily(raw, spec)


def build_index_benchmark_csv(
    specs: tuple[IndexFetchSpec, ...],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = [fetch_index_daily(spec, start_date=start_date, end_date=end_date) for spec in specs]
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["trade_date", "open", "symbol", "name", "source_symbol"])
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    end_date = args.end_date.strip() or pd.Timestamp.now().strftime("%Y%m%d")
    specs = parse_index_specs(args.index)
    out = build_index_benchmark_csv(
        specs,
        start_date=args.start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if out.empty:
        print("[fetch-index-benchmarks] no rows fetched", file=sys.stderr)
        return 1
    output = _resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[fetch-index-benchmarks] rows={len(out)} symbols={out['symbol'].nunique()} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
