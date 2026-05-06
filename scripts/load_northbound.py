#!/usr/bin/env python3
"""北向资金数据加载脚本。

从 AkShare (stock_hsgt_individual_em) 拉取个股北向持股历史数据，
写入 DuckDB a_share_northbound 表。

用法：
    # 加载单只（测试）
    python scripts/load_northbound.py --symbols 600519

    # 批量加载（从候选池采样）
    python scripts/load_northbound.py --pool U1_liquid_tradable --sample 100

    # 全量加载（注意限速，预计每只 ~1-2s，全市场需数小时）
    python scripts/load_northbound.py --all --max-workers 4

依赖：akshare >= 1.14.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="北向资金数据加载")
    p.add_argument("--db-path", type=str, default="data/market.duckdb")
    p.add_argument("--symbols", type=str, default="",
                   help="逗号分隔的股票代码列表")
    p.add_argument("--pool", type=str, default="",
                   help="从 monthly_long CSV 中读取候选池 symbols")
    p.add_argument("--pool-csv", type=str,
                   default="data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv")
    p.add_argument("--sample", type=int, default=0,
                   help="从候选池随机采样 N 只（0=全部）")
    p.add_argument("--all", action="store_true",
                   help="加载全市场（从 a_share_daily 获取所有 symbols）")
    p.add_argument("--max-workers", type=int, default=4,
                   help="并发 worker 数")
    p.add_argument("--delay", type=float, default=0.5,
                   help="请求间隔（秒）")
    p.add_argument("--dry-run", action="store_true",
                   help="打印将要拉取的 symbols，不实际执行")
    p.add_argument("--migrate", action="store_true",
                   help="仅运行 migration（建表）")
    return p.parse_args()


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]

    con = duckdb.connect(args.db_path, read_only=True)

    if args.pool:
        # Try to get symbols from pool_csv (per-stock monthly_long files have 'symbol')
        try:
            pool_df = pd.read_csv(args.pool_csv)
            if "symbol" in pool_df.columns and args.pool in pool_df["candidate_pool_version"].values:
                symbols = sorted(
                    pool_df[pool_df["candidate_pool_version"] == args.pool]["symbol"]
                    .dropna().astype(str).str.zfill(6).unique()
                )
            else:
                # Fallback: get from DuckDB using U1 liquid tradable filter
                symbols_df = con.execute(
                    "SELECT DISTINCT symbol FROM a_share_daily WHERE amount > 0 "
                    "ORDER BY symbol"
                ).df()
                symbols = sorted(symbols_df["symbol"].astype(str).str.zfill(6).tolist())
        except Exception:
            symbols_df = con.execute(
                "SELECT DISTINCT symbol FROM a_share_daily ORDER BY symbol"
            ).df()
            symbols = sorted(symbols_df["symbol"].astype(str).str.zfill(6).tolist())
    elif args.all:
        symbols_df = con.execute(
            "SELECT DISTINCT symbol FROM a_share_daily ORDER BY symbol"
        ).df()
        symbols = sorted(symbols_df["symbol"].astype(str).str.zfill(6).tolist())
    else:
        print("请指定 --symbols / --pool / --all 中的一个", file=sys.stderr)
        sys.exit(1)

    con.close()

    if args.sample and args.sample < len(symbols):
        import random
        random.seed(42)
        symbols = random.sample(symbols, args.sample)

    return symbols


def load_northbound_for_symbol(
    symbol: str,
    db_path: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> int:
    """为单个 symbol 拉取并写入北向数据。返回写入行数。"""
    try:
        from src.data_fetcher.northbound_client import fetch_northbound_individual
    except ImportError:
        # fallback: 直接调用 akshare
        import akshare as ak
        def fetch_northbound_individual(sym, **kw):
            return ak.stock_hsgt_individual_em(symbol=sym)

    for attempt in range(max_retries):
        try:
            raw = fetch_northbound_individual(symbol, max_retries=1)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    if raw is None or raw.empty:
        return 0

    df = raw.copy()
    # fetch_northbound_individual 已返回英文列名:
    # symbol, trade_date, close, pct_chg, hold_shares, hold_value, hold_pct_a,
    # net_buy_shares, net_buy_value, hold_value_chg
    col_rename = {
        "hold_value": "hold_amount",
        "hold_pct_a": "hold_ratio",
        "net_buy_value": "net_buy_amount",
    }
    for old, new in col_rename.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    keep = ["trade_date", "hold_amount", "hold_ratio", "net_buy_amount", "hold_shares"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["symbol"] = symbol.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"])
    if df.empty:
        return 0

    con = duckdb.connect(db_path)
    try:
        existing = con.execute(
            "SELECT trade_date FROM a_share_northbound WHERE symbol = ?", [symbol]
        ).fetchdf()
        if not existing.empty:
            existing_dates = set(pd.to_datetime(existing["trade_date"]).dt.normalize())
            df = df[~df["trade_date"].isin(existing_dates)]
        if df.empty:
            return 0
        con.executemany(
            """INSERT OR REPLACE INTO a_share_northbound
               (symbol, trade_date, hold_amount, hold_ratio, net_buy_amount, hold_shares)
               VALUES (?, ?, ?, ?, ?, ?)""",
            df[["symbol", "trade_date", "hold_amount", "hold_ratio",
                "net_buy_amount", "hold_shares"]].values.tolist(),
        )
        return len(df)
    finally:
        con.close()


def main() -> int:
    args = parse_args()

    # 迁移
    from src.data_fetcher.migrations import apply_migrations
    con = duckdb.connect(args.db_path)
    try:
        apply_migrations(con)
    finally:
        con.close()
    print(f"[northbound-loader] migration complete")

    if args.migrate:
        return 0

    symbols = resolve_symbols(args)
    print(f"[northbound-loader] {len(symbols)} symbols to load")

    if args.dry_run:
        print(f"[northbound-loader] dry-run, symbols: {symbols[:10]}...")
        return 0

    total_rows = 0
    ok = 0
    fail = 0
    for i, sym in enumerate(symbols):
        try:
            rows = load_northbound_for_symbol(
                sym, args.db_path, max_retries=3, retry_delay=args.delay,
            )
            total_rows += rows
            ok += 1
            if (i + 1) % 10 == 0 or i == len(symbols) - 1:
                print(f"[northbound-loader] {i+1}/{len(symbols)} symbols, "
                      f"{total_rows} rows, {fail} failed", flush=True)
        except Exception as e:
            fail += 1
            print(f"[northbound-loader] FAIL {sym}: {e}", file=sys.stderr)
        time.sleep(args.delay)

    print(f"[northbound-loader] done: {ok} ok, {fail} fail, {total_rows} rows total")
    return 1 if fail > len(symbols) * 0.5 else 0


if __name__ == "__main__":
    raise SystemExit(main())
