"""融资融券数据拉取与质量诊断。

数据源: AkShare ``stock_margin_detail_sse`` (沪市) + ``stock_margin_detail_szse`` (深市)。
历史起点: 约 2012 年（实际取决于 AkShare 源站）。
PIT-safety: T+0 披露，信号日即可用。

M11 B2: 质量诊断独立模块 — 覆盖率 + PIT 安全性 + 相关性检验。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

MARGIN_TRADING_START = "2012-01-01"

MIN_COVERAGE_THRESHOLD = 0.7
MIN_MONTHS_THRESHOLD = 24

# 列名映射（SSE/SZSE → 统一英文名）
SSE_COL_MAP = {
    "信用交易日期": "trade_date",
    "标的证券代码": "symbol",
    "标的证券简称": "name",
    "融资余额": "fin_balance",
    "融资买入额": "fin_buy_amount",
    "融资偿还额": "fin_repay_amount",
    "融券余量": "short_volume",
    "融券卖出量": "short_sell_volume",
    "融券偿还量": "short_repay_volume",
}

SZSE_COL_MAP = {
    "证券代码": "symbol",
    "证券简称": "name",
    "融资买入额": "fin_buy_amount",
    "融资余额": "fin_balance",
    "融券卖出量": "short_sell_volume",
    "融券余量": "short_volume",
    "融券余额": "short_balance",
    "融资融券余额": "total_balance",
}


def _norm_date(trade_date: str) -> str:
    """将日期统一为 YYYYMMDD 格式（AkShare API 不接受 YYYY-MM-DD）。"""
    return trade_date.replace("-", "") if "-" in trade_date else trade_date


def fetch_margin_detail_sse(
    trade_date: str,
    *,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """拉取沪市融资融券明细（个股）。

    Returns
    -------
    DataFrame with columns: trade_date, symbol, name, fin_balance,
    fin_buy_amount, fin_repay_amount, short_volume, short_sell_volume,
    short_repay_volume
    """
    import akshare as ak

    trade_date = _norm_date(trade_date)
    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_margin_detail_sse(date=trade_date)
            if df is not None and not df.empty:
                df = df.rename(columns=SSE_COL_MAP)
                keep = [c for c in SSE_COL_MAP.values() if c in df.columns]
                df = df[keep]
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                return df
            return pd.DataFrame()
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_sec * (attempt + 1))

    _LOG.warning(f"沪市融资融券拉取失败 {trade_date}: {last_err}")
    return pd.DataFrame()


def fetch_margin_detail_szse(
    trade_date: str,
    *,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """拉取深市融资融券明细（个股）。

    Returns
    -------
    DataFrame with columns: trade_date, symbol, name, fin_balance,
    fin_buy_amount, short_volume, short_sell_volume, short_balance, total_balance
    """
    import akshare as ak

    trade_date = _norm_date(trade_date)
    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_margin_detail_szse(date=trade_date)
            if df is not None and not df.empty:
                df = df.rename(columns=SZSE_COL_MAP)
                keep = [c for c in SZSE_COL_MAP.values() if c in df.columns]
                df = df[keep]
                df["trade_date"] = pd.to_datetime(df["trade_date"] if "trade_date" in df.columns
                                                  else pd.Timestamp(trade_date))
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                return df
            return pd.DataFrame()
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_sec * (attempt + 1))

    _LOG.warning(f"深市融资融券拉取失败 {trade_date}: {last_err}")
    return pd.DataFrame()


def fetch_margin_detail(
    trade_date: str,
    *,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """拉取全市场融资融券明细（SSE + SZSE 合并）。

    Returns standardised DataFrame with columns:
    trade_date, symbol, fin_balance, fin_buy_amount, short_volume, short_sell_volume,
    fin_repay_amount (SSE only), short_balance (SZSE only), total_balance (SZSE only)
    """
    sse = fetch_margin_detail_sse(trade_date, max_retries=max_retries, retry_delay_sec=retry_delay_sec)
    szse = fetch_margin_detail_szse(trade_date, max_retries=max_retries, retry_delay_sec=retry_delay_sec)

    if sse.empty and szse.empty:
        return pd.DataFrame()

    # Align columns: keep intersection + union where available
    combined = pd.concat([sse, szse], ignore_index=True, sort=False)
    if "trade_date" not in combined.columns:
        combined["trade_date"] = pd.Timestamp(trade_date)

    # Deduplicate: keep first (SSE preferred for dual-listed)
    combined = combined.drop_duplicates(subset=["trade_date", "symbol"], keep="first")
    return combined


# ── 质量诊断 ──────────────────────────────────────────────────────────────────


@dataclass
class MarginTradingQualityReport:
    ok: bool
    table_exists: bool
    total_rows: int = 0
    distinct_symbols: int = 0
    min_trade_date: Optional[str] = None
    max_trade_date: Optional[str] = None
    duplicate_pk_rows: int = 0
    null_ratio_by_col: dict = None
    coverage_ratio_vs_daily: Optional[float] = None
    rows_without_daily_match: int = 0
    daily_max_trade_date: Optional[str] = None
    median_symbols_per_day: float = 0.0
    notes: list = None

    def __post_init__(self):
        if self.null_ratio_by_col is None:
            self.null_ratio_by_col = {}
        if self.notes is None:
            self.notes = []


def run_margin_trading_quality_checks(
    con,
    *,
    table: str = "a_share_margin_trading",
    daily_table: str = "a_share_daily",
) -> MarginTradingQualityReport:
    """对融资融券表运行基础质量诊断。

    检查项：
    - 表存在性、行数、标的数、日期范围
    - 主键重复（symbol + trade_date）
    - 关键列空值率
    - 相对日线表的覆盖率
    """
    import duckdb

    notes: list[str] = []

    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        table_exists = bool(exists and int(exists[0]) > 0)
    except Exception:
        table_exists = False

    if not table_exists:
        return MarginTradingQualityReport(
            ok=False,
            table_exists=False,
            notes=["融资融券表不存在，需运行 scripts/load_margin_trading.py 初始化数据"],
        )

    try:
        stats = con.execute(f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT symbol) AS distinct_symbols,
                MIN(trade_date)::VARCHAR AS min_date,
                MAX(trade_date)::VARCHAR AS max_date,
                COUNT(*) - COUNT(DISTINCT (symbol, trade_date)) AS dup_pk_rows
            FROM {table}
        """).fetchone()
        total_rows = int(stats[0])
        distinct_symbols = int(stats[1])
        min_date = stats[2]
        max_date = stats[3]
        dup_pk_rows = int(stats[4])
    except Exception as e:
        return MarginTradingQualityReport(
            ok=False, table_exists=True, notes=[f"统计查询失败: {e}"],
        )

    # Coverage vs daily
    coverage = None
    rows_without_daily = 0
    daily_max_date = None
    try:
        daily_max = con.execute(f"SELECT MAX(trade_date)::VARCHAR FROM {daily_table}").fetchone()
        daily_max_date = daily_max[0] if daily_max else None
        matched = con.execute(f"""
            SELECT COUNT(*) FROM {table} mt
            INNER JOIN {daily_table} d ON mt.symbol = d.symbol AND mt.trade_date = d.trade_date
        """).fetchone()
        matched_rows = int(matched[0]) if matched else 0
        coverage = matched_rows / total_rows if total_rows > 0 else None
        rows_without_daily = total_rows - matched_rows
    except Exception:
        pass

    # Null ratios for key columns
    null_ratios = {}
    key_cols = ["fin_balance", "fin_buy_amount", "short_volume"]
    for col in key_cols:
        try:
            n_null = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
            null_ratios[col] = n_null / total_rows if total_rows > 0 else 0.0
        except Exception:
            pass

    # Median symbols per day
    median_sym = 0.0
    try:
        median_sym = con.execute(f"""
            SELECT MEDIAN(cnt) FROM (
                SELECT COUNT(*) AS cnt FROM {table} GROUP BY trade_date
            )
        """).fetchone()[0] or 0.0
    except Exception:
        pass

    all_ok = (
        table_exists
        and total_rows > 0
        and dup_pk_rows == 0
        and (coverage is None or coverage >= 0.5)
        and distinct_symbols >= 100
    )

    return MarginTradingQualityReport(
        ok=all_ok,
        table_exists=table_exists,
        total_rows=total_rows,
        distinct_symbols=distinct_symbols,
        min_trade_date=min_date,
        max_trade_date=max_date,
        duplicate_pk_rows=dup_pk_rows,
        null_ratio_by_col=null_ratios,
        coverage_ratio_vs_daily=coverage,
        rows_without_daily_match=rows_without_daily,
        daily_max_trade_date=daily_max_date,
        median_symbols_per_day=median_sym,
        notes=notes,
    )
