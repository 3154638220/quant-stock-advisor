#!/usr/bin/env python3
"""融资融券数据加载脚本。

从 AkShare (stock_margin_detail_sse + stock_margin_detail_szse) 拉取个股融资融券明细，
写入 DuckDB a_share_margin_trading 表。

用法:
    # 加载单日（测试）
    python scripts/load_margin_trading.py --date 20260429

    # 批量加载日期范围
    python scripts/load_margin_trading.py --start 20240101 --end 20260430

    # 全量加载（从配置起始日期起）
    python scripts/load_margin_trading.py --all

依赖：akshare >= 1.14.0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="融资融券数据加载")
    p.add_argument("--db-path", type=str, default="data/market.duckdb")
    p.add_argument("--date", type=str, default="",
                   help="单日加载 (YYYYMMDD)")
    p.add_argument("--start", type=str, default="",
                   help="起始日期 (YYYYMMDD)")
    p.add_argument("--end", type=str, default="",
                   help="结束日期 (YYYYMMDD)，默认今天")
    p.add_argument("--all", action="store_true",
                   help="从 2012-01-01 起全量加载")
    p.add_argument("--delay", type=float, default=1.0,
                   help="请求间隔（秒）")
    p.add_argument("--dry-run", action="store_true",
                   help="打印将要拉取的日期，不实际执行")
    p.add_argument("--migrate", action="store_true",
                   help="仅运行 migration（建表）")
    return p.parse_args()


def _to_date_str(d: str) -> str:
    """保持 YYYYMMDD 格式（AkShare API 不接受 YYYY-MM-DD）。"""
    return d.replace("-", "") if "-" in d else d


def _to_db_date(d: str) -> str:
    """将 YYYYMMDD 转为 YYYY-MM-DD（仅用于 DuckDB SQL）。"""
    d = _to_date_str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def trading_days_between(start: str, end: str, con) -> list[str]:
    """从 a_share_daily 获取区间内的交易日列表（返回 YYYYMMDD 格式）。"""
    rows = con.execute(
        "SELECT DISTINCT trade_date FROM a_share_daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        [_to_db_date(start), _to_db_date(end)],
    ).fetchall()
    # DuckDB returns datetime.date, convert to YYYYMMDD
    result = []
    for r in rows:
        d = r[0]
        if hasattr(d, 'strftime'):
            result.append(d.strftime('%Y%m%d'))
        else:
            result.append(_to_date_str(str(d)))
    return result


def load_margin_for_date(
    trade_date: str,
    db_path: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> int:
    """为单个交易日拉取并写入融资融券数据。返回写入行数。"""
    from src.data_fetcher.margin_trading_client import fetch_margin_detail

    df = None
    for attempt in range(max_retries):
        try:
            df = fetch_margin_detail(trade_date, max_retries=1)
            if df is not None and not df.empty:
                break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    else:
        print(f"[margin-loader] 空数据 {trade_date}", file=sys.stderr)
        return 0

    if df is None or df.empty:
        return 0

    # Standardise for DB
    keep_cols = ["symbol", "fin_balance", "fin_buy_amount", "short_volume", "short_sell_volume"]
    if "fin_repay_amount" in df.columns:
        keep_cols.append("fin_repay_amount")
    if "short_balance" in df.columns:
        keep_cols.append("short_balance")
    if "total_balance" in df.columns:
        keep_cols.append("total_balance")

    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df["trade_date"] = pd.to_datetime(trade_date, format="%Y%m%d")
    df = df.drop_duplicates(subset=["symbol", "trade_date"])
    if df.empty:
        return 0

    trade_date_ts = pd.Timestamp(trade_date)
    con = duckdb.connect(db_path)
    try:
        existing = con.execute(
            "SELECT symbol FROM a_share_margin_trading WHERE trade_date = ?",
            [trade_date_ts],
        ).fetchall()
        existing_symbols = set(str(r[0]) for r in existing)
        if existing_symbols:
            df = df[~df["symbol"].isin(existing_symbols)]
        if df.empty:
            return 0

        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        sql = f"INSERT INTO a_share_margin_trading ({col_names}) VALUES ({placeholders})"
        con.executemany(sql, df[cols].values.tolist())
        return len(df)
    finally:
        con.close()


def main() -> int:
    args = parse_args()

    con = duckdb.connect(args.db_path)
    try:
        from src.data_fetcher.migrations import apply_migrations
        apply_migrations(con)
    finally:
        con.close()
    print("[margin-loader] migration complete")

    if args.migrate:
        return 0

    # Resolve dates
    if args.date:
        dates = [args.date]
    elif args.start:
        end = args.end or datetime.today().strftime("%Y%m%d")
        con_r = duckdb.connect(args.db_path, read_only=True)
        dates = trading_days_between(args.start, end, con_r)
        con_r.close()
    elif args.all:
        con_r = duckdb.connect(args.db_path, read_only=True)
        dates = trading_days_between("2012-01-01", datetime.today().strftime("%Y%m%d"), con_r)
        con_r.close()
    else:
        print("请指定 --date / --start [--end] / --all", file=sys.stderr)
        return 1

    # Skip already-loaded dates (normalize both to YYYYMMDD)
    con_r = duckdb.connect(args.db_path, read_only=True)
    try:
        loaded = con_r.execute(
            "SELECT DISTINCT trade_date FROM a_share_margin_trading"
        ).fetchall()
        loaded_dates = set()
        for r in loaded:
            d = r[0]
            if hasattr(d, 'strftime'):
                loaded_dates.add(d.strftime('%Y%m%d'))
            else:
                loaded_dates.add(str(d).replace('-', '')[:8])
    except Exception:
        loaded_dates = set()
    finally:
        con_r.close()

    pending = [d for d in dates if d not in loaded_dates]
    print(f"[margin-loader] {len(dates)} total trading days, "
          f"{len(loaded_dates)} already loaded, {len(pending)} pending")

    if args.dry_run:
        print(f"[margin-loader] dry-run, first 10 pending: {pending[:10]}")
        return 0

    total_rows = 0
    ok = 0
    fail = 0
    for i, d in enumerate(pending):
        try:
            rows = load_margin_for_date(d, args.db_path, max_retries=3, retry_delay=args.delay)
            total_rows += rows
            ok += 1
            if (i + 1) % 5 == 0 or i == len(pending) - 1:
                print(f"[margin-loader] {i+1}/{len(pending)} days, "
                      f"{total_rows} rows, {fail} failed", flush=True)
        except Exception as e:
            fail += 1
            print(f"[margin-loader] FAIL {d}: {e}", file=sys.stderr)
            if fail > 10:
                print("[margin-loader] 失败过多，中止", file=sys.stderr)
                break
        time.sleep(args.delay)

    print(f"[margin-loader] done: {ok} ok, {fail} fail, {total_rows} rows total")
    return 0 if fail <= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
