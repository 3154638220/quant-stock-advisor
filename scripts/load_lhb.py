#!/usr/bin/env python3
"""M14: 龙虎榜历史日明细数据拉取。

数据源：新浪财经龙虎榜每日明细 (https://vip.stock.finance.sina.com.cn)
存储：DuckDB data/market.duckdb → a_share_lhb_daily

用法：
    python scripts/load_lhb.py                          # 增量：补齐最近 90 天
    python scripts/load_lhb.py --start 2021-01-01       # 全量回填
    python scripts/load_lhb.py --dry-run                # 预估拉取量
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market.duckdb"
SINA_URL = "https://vip.stock.finance.sina.com.cn/q/go.php/vInvestConsult/kind/lhb/index.phtml"
SLEEP_SEC = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M14 龙虎榜数据拉取")
    p.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认：90天前）")
    p.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认：今天）")
    p.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行")
    p.add_argument("--sleep", type=float, default=SLEEP_SEC, help="请求间隔秒数")
    return p.parse_args()


def get_trading_days(con: duckdb.DuckDBPyConnection, start: str, end: str) -> list[str]:
    """从 a_share_daily 获取交易日列表。"""
    days = con.execute(
        """
        SELECT DISTINCT trade_date
        FROM a_share_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """,
        [start, end],
    ).fetchall()
    return [d[0].strftime("%Y-%m-%d") if hasattr(d[0], "strftime") else str(d[0]) for d in days]


def get_existing_dates(con: duckdb.DuckDBPyConnection) -> set[str]:
    """获取已拉取的日期。"""
    try:
        rows = con.execute(
            "SELECT DISTINCT data_date FROM a_share_lhb_daily"
        ).fetchall()
        return {str(r[0]) for r in rows}
    except Exception:
        return set()


def fetch_lhb_daily(date_str: str) -> pd.DataFrame | None:
    """拉取单日龙虎榜明细。"""
    params = {"tradedate": date_str}
    resp = requests.get(SINA_URL, params=params, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, features="lxml")

    container = soup.find(name="div", attrs={"class": "list"})
    if container is None:
        return None

    tables = container.find_all(name="table", attrs={"class": "list_table"})
    if not tables:
        return None

    frames = []
    for table in tables:
        raw = pd.read_html(StringIO(table.prettify()), header=0, skiprows=1)
        if not raw or len(raw) == 0:
            continue
        temp_df = raw[0].copy()
        # 提取板块指标名称（表格上方标题行）
        try:
            header_df = pd.read_html(StringIO(table.prettify()))[0]
            indicator = header_df.iat[0, 0]
        except (IndexError, ValueError):
            indicator = ""
        temp_df["指标"] = indicator
        frames.append(temp_df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # 标准化股票代码为6位
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)

    # 删除"查看详情"列（链接）
    if "查看详情" in df.columns:
        del df["查看详情"]

    # 统一列名
    col_map = {
        "序号": "seq",
        "股票代码": "symbol",
        "股票名称": "stock_name",
        "收盘价": "close",
        "收盘价(元)": "close",
        "对应值": "change_pct",
        "对应值(%)": "change_pct",
        "成交量": "volume",
        "成交量(万股)": "volume_wan",
        "成交额": "amount",
        "成交额(万元)": "amount_wan",
        "指标": "lhb_reason",
    }
    df = df.rename(columns=col_map)

    # 保留统一后的列
    keep_cols = ["symbol", "stock_name", "close", "change_pct", "volume_wan", "amount_wan", "lhb_reason"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    # lhb_reason: NaN → 空字符串，避免 NOT NULL 约束冲突
    if "lhb_reason" in df.columns:
        df["lhb_reason"] = df["lhb_reason"].fillna("").astype(str)

    df["data_date"] = date_str

    # 类型转换
    for c in ["close", "change_pct", "volume_wan", "amount_wan"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def ensure_table(con: duckdb.DuckDBPyConnection):
    """创建 LHB 表（如不存在）。"""
    con.execute("""
        CREATE TABLE IF NOT EXISTS a_share_lhb_daily (
            symbol        VARCHAR NOT NULL,
            stock_name    VARCHAR,
            close         DOUBLE,
            change_pct    DOUBLE,
            volume_wan    DOUBLE,
            amount_wan    DOUBLE,
            lhb_reason    VARCHAR DEFAULT '',
            data_date     DATE NOT NULL,
        )
    """)


def main() -> int:
    args = parse_args()

    if not DB_PATH.exists():
        print(f"[ERROR] 数据库不存在: {DB_PATH}")
        return 1

    with duckdb.connect(str(DB_PATH)) as con:
        ensure_table(con)

        # 确定日期范围
        end_date = args.end or date.today().strftime("%Y-%m-%d")
        if args.start:
            start_date = args.start
        else:
            # 增量模式：最近 90 天除去已拉取的
            start_date = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")

        trading_days = get_trading_days(con, start_date, end_date)
        existing = get_existing_dates(con)
        todo = [d for d in trading_days if d not in existing]

        if args.dry_run:
            print(f"[DRY RUN] 交易日范围: {start_date} → {end_date}")
            print(f"  总交易日: {len(trading_days)}, 已拉取: {len(existing)}, 待拉取: {len(todo)}")
            if todo:
                print(f"  首日: {todo[0]}, 末日: {todo[-1]}")
            return 0

        if not todo:
            print(f"[LHB] 全部就绪: {len(trading_days)} 个交易日已覆盖 ({start_date} → {end_date})")
            return 0

        print(f"[LHB] 待拉取 {len(todo)} 个交易日 ({todo[0]} → {todo[-1]})", flush=True)

        total_rows = 0
        success = 0
        empty = 0
        failed = 0

        for i, d in enumerate(todo):
            try:
                df = fetch_lhb_daily(d)
                if df is None or len(df) == 0:
                    empty += 1
                else:
                    con.execute("DELETE FROM a_share_lhb_daily WHERE data_date = ?", [d])
                    con.register("_tmp", df)
                    con.execute("INSERT INTO a_share_lhb_daily SELECT * FROM _tmp")
                    con.unregister("_tmp")
                    total_rows += len(df)
                    success += 1

                if (i + 1) % 50 == 0 or i == len(todo) - 1:
                    msg = (f"  [{i+1}/{len(todo)}] {d}: {len(df) if df is not None else 0} rows  "
                           f"(ok={success}, empty={empty}, fail={failed})")
                    print(msg, flush=True)
            except Exception as e:
                failed += 1
                print(f"  [{i+1}/{len(todo)}] {d}: FAIL - {e}")
            time.sleep(args.sleep)

    print(f"[LHB] 完成: {success} 天成功, {total_rows} 行, {empty} 天无数据, {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
