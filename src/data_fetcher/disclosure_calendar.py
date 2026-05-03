"""P0-1: A 股财报实际披露日历工具。

提供披露日期表的创建、数据回填和查询接口。
优先使用实际公告日期判定 PIT 可得性，取代固定 availability_lag_days。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# DuckDB 披露日历建表 SQL
DISCLOSURE_CALENDAR_DDL = """
CREATE TABLE IF NOT EXISTS a_share_disclosure_date (
    symbol         VARCHAR,
    report_period  DATE,       -- 报告期，如 2025-09-30（三季报）
    disclosure_dt  DATE,        -- 实际公告日期
    report_type    VARCHAR,     -- annual / semi / q1 / q3
    PRIMARY KEY (symbol, report_period)
);
"""


def ensure_disclosure_calendar_table(db_path: Path | str) -> None:
    """在 DuckDB 中创建披露日历表（如不存在）。"""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(DISCLOSURE_CALENDAR_DDL)
    finally:
        con.close()


def load_disclosure_calendar(
    db_path: Path | str,
    *,
    table: str = "a_share_disclosure_date",
) -> pd.DataFrame:
    """从 DuckDB 加载披露日历。

    Returns
    -------
    DataFrame with columns: symbol, report_period, disclosure_dt, report_type
    若表不存在或为空，返回空 DataFrame。
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return pd.DataFrame(columns=["symbol", "report_period", "disclosure_dt", "report_type"])
        df = con.execute(f"SELECT * FROM {table}").df()
        return df
    finally:
        con.close()


def fetch_disclosure_dates_akshare() -> pd.DataFrame:
    """通过 akshare 获取 A 股财报实际披露日期。

    使用 akshare.stock_financial_report_disclosure_date 接口，
    返回所有历史财报的实际公告日期。

    Returns
    -------
    DataFrame with columns: symbol, report_period, disclosure_dt, report_type

    Notes
    -----
    需要 akshare >= 1.11.0。若 akshare 不可用，返回空 DataFrame。
    """
    try:
        import akshare as ak
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "akshare 不可用，无法获取实际披露日期。请安装: pip install akshare"
        )
        return pd.DataFrame(columns=["symbol", "report_period", "disclosure_dt", "report_type"])

    records: list[dict] = []
    report_type_map = {
        "年报": "annual",
        "半年报": "semi",
        "一季报": "q1",
        "三季报": "q3",
    }

    for year in range(2015, 2027):
        try:
            df = ak.stock_financial_report_disclosure_date(year=str(year))
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                symbol = str(row.get("stock_code", "")).strip().zfill(6)
                if len(symbol) != 6:
                    continue
                report_period_str = str(row.get("report_period", "")).strip()
                disclosure_str = str(row.get("disclosure_date", "")).strip()
                report_type_cn = str(row.get("report_type", "")).strip()

                if not report_period_str or not disclosure_str:
                    continue

                try:
                    rp = pd.Timestamp(report_period_str)
                    dd = pd.Timestamp(disclosure_str)
                except (ValueError, TypeError):
                    continue

                records.append({
                    "symbol": symbol,
                    "report_period": rp.date(),
                    "disclosure_dt": dd.date(),
                    "report_type": report_type_map.get(report_type_cn, "unknown"),
                })
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["symbol", "report_period", "disclosure_dt", "report_type"])

    out = pd.DataFrame(records)
    out = out.drop_duplicates(["symbol", "report_period"], keep="last")
    return out.sort_values(["symbol", "report_period"]).reset_index(drop=True)


def backfill_disclosure_calendar(
    db_path: Path | str,
    *,
    table: str = "a_share_disclosure_date",
) -> int:
    """回填披露日历到 DuckDB。

    从 akshare 获取历史披露日期并写入 DuckDB 表。

    Returns
    -------
    int : 写入的行数
    """
    ensure_disclosure_calendar_table(db_path)
    df = fetch_disclosure_dates_akshare()
    if df.empty:
        return 0

    con = duckdb.connect(str(db_path))
    try:
        # 使用 INSERT OR REPLACE 处理重复
        con.execute(f"DELETE FROM {table}")
        con.executemany(
            f"INSERT INTO {table} (symbol, report_period, disclosure_dt, report_type) "
            "VALUES (?, ?, ?, ?)",
            [(r.symbol, r.report_period, r.disclosure_dt, r.report_type)
             for r in df.itertuples(index=False)],
        )
    finally:
        con.close()

    return len(df)
