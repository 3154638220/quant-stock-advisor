#!/usr/bin/env python3
"""Fetch investable/index benchmark daily opens for benchmark suite.

The output CSV is intentionally narrow and matches
``scripts/run_monthly_benchmark_suite.py --index-csv``:

    trade_date, open, symbol

By default it fetches CSI 1000 and CSI 2000 from AkShare/Eastmoney.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class IndexFetchSpec:
    name: str
    output_symbol: str
    akshare_symbol: str


DEFAULT_INDEX_SPECS: tuple[IndexFetchSpec, ...] = (
    IndexFetchSpec("csi1000", "000852", "sh000852"),
    IndexFetchSpec("csi2000", "932000", "csi932000"),
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


def parse_index_specs(items: list[str]) -> tuple[IndexFetchSpec, ...]:
    if not items:
        return DEFAULT_INDEX_SPECS
    specs: list[IndexFetchSpec] = []
    for item in items:
        parts = [x.strip() for x in str(item).split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"--index 需要 name:output_symbol:akshare_symbol，收到: {item!r}")
        name, output_symbol, akshare_symbol = parts
        digits = "".join(ch for ch in output_symbol if ch.isdigit())
        if len(digits) != 6:
            raise ValueError(f"output_symbol 需要 6 位代码，收到: {output_symbol!r}")
        specs.append(IndexFetchSpec(name=name, output_symbol=digits, akshare_symbol=akshare_symbol))
    return tuple(specs)


def standardize_index_daily(raw: pd.DataFrame, spec: IndexFetchSpec) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["trade_date", "open", "symbol", "name", "source_symbol"])
    col_map = {
        "日期": "trade_date",
        "date": "trade_date",
        "datetime": "trade_date",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}).copy()
    if "trade_date" not in df.columns or "open" not in df.columns:
        raise ValueError(f"{spec.name} 指数日线缺少 trade_date/open 列: {list(raw.columns)}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    for col in ["close", "high", "low", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = spec.output_symbol.zfill(6)
    df["name"] = spec.name
    df["source_symbol"] = spec.akshare_symbol
    keep = ["trade_date", "open", "symbol", "name", "source_symbol"]
    keep.extend([c for c in ["close", "high", "low", "volume", "amount"] if c in df.columns])
    out = df[keep].dropna(subset=["trade_date", "open"])
    return out.drop_duplicates(["symbol", "trade_date"], keep="last").sort_values(["symbol", "trade_date"])


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
