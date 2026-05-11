"""Fetch A-share event data from akshare (东方财富) and persist to DuckDB v17 tables.

Covers four event types:
  - 业绩预告 (earnings guidance):  stock_yjyg_em
  - 股份回购 (buyback):           stock_repurchase_em
  - 大股东增减持 (shareholder):    stock_ggcg_em
  - 限售解禁 (unlock):             stock_restricted_release_detail_em

All tables use PRIMARY KEY (symbol, announce_date [+ unlock_date]) for
idempotent upsert — re-running is safe.

Usage::

    # Full historical backfill (2021-01-01 through today)
    python scripts/fetch_events.py --start 2021-01-01

    # Single date
    python scripts/fetch_events.py --start 2026-05-01 --end 2026-05-01

    # Dry-run (print summary without writing)
    python scripts/fetch_events.py --start 2026-05-01 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DB_DEFAULT = "data/market.duckdb"

# ── v17 table names ──────────────────────────────────────────────────────
TABLE_EARNINGS = "a_share_event_earnings_guidance"
TABLE_BUYBACK = "a_share_event_buyback"
TABLE_REDUCTION = "a_share_event_reduction"
TABLE_UNLOCK = "a_share_event_unlock"


def _ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create v17 event tables if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS a_share_event_earnings_guidance (
            symbol VARCHAR NOT NULL,
            announce_date DATE NOT NULL,
            forecast_metric VARCHAR NOT NULL DEFAULT '',
            guidance_direction VARCHAR,
            guidance_change_ratio DOUBLE,
            forecast_value DOUBLE,
            prev_year_value DOUBLE,
            source VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, announce_date, forecast_metric)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS a_share_event_buyback (
            symbol VARCHAR NOT NULL,
            announce_date DATE NOT NULL,
            buyback_amount DOUBLE,
            market_cap DOUBLE,
            progress_status VARCHAR,
            source VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, announce_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS a_share_event_reduction (
            symbol VARCHAR NOT NULL,
            announce_date DATE NOT NULL,
            holder_name VARCHAR NOT NULL,
            change_direction VARCHAR,
            change_amount DOUBLE,
            change_ratio_total DOUBLE,
            change_ratio_circulating DOUBLE,
            post_change_total DOUBLE,
            post_change_ratio_total DOUBLE,
            post_change_circulating DOUBLE,
            post_change_ratio_circulating DOUBLE,
            change_start_date DATE,
            change_end_date DATE,
            source VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, announce_date, holder_name)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS a_share_event_unlock (
            symbol VARCHAR NOT NULL,
            announce_date DATE NOT NULL,
            unlock_date DATE NOT NULL,
            unlock_shares DOUBLE,
            unlock_market_value DOUBLE,
            lockup_type VARCHAR,
            source VARCHAR,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, announce_date, unlock_date)
        )
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Column mappings: akshare output → DB columns
# These mappings may need tuning after observing actual akshare output.
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_earnings_guidance(date_str: str) -> pd.DataFrame:
    """Fetch earnings guidance for a report period via akshare.

    ``stock_yjyg_em(date=...)`` returns guidance by report period (e.g. "20260331").
    """
    import akshare as ak

    try:
        raw = ak.stock_yjyg_em(date=date_str)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Earnings API returned no data for %s", date_str)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalize columns (东方财富 uses Chinese column names)
    # Actual API output: 股票代码, 公告日期, 预测指标, 业绩变动, 预测数值, 业绩变动幅度,
    #                    业绩变动原因, 预告类型, 上年同期值
    col_map = {
        "股票代码": "symbol",
        "公告日期": "announce_date",
        "预测指标": "forecast_metric",
        "预告类型": "guidance_direction",
        "业绩变动幅度": "guidance_change_ratio",
        "预测数值": "forecast_value",
        "上年同期值": "prev_year_value",
    }
    existing = set(raw.columns)
    rename = {k: v for k, v in col_map.items() if k in existing}
    df = raw.rename(columns=rename)

    wanted = [
        "symbol", "announce_date", "forecast_metric",
        "guidance_direction", "guidance_change_ratio",
        "forecast_value", "prev_year_value",
    ]
    result = pd.DataFrame()
    for col in wanted:
        if col in df.columns:
            result[col] = df[col]
        elif col == "report_period":
            result[col] = pd.NaT
        else:
            result[col] = pd.NA
    result["source"] = "akshare:stock_yjyg_em"
    return result


def _fetch_buyback() -> pd.DataFrame:
    """Fetch all share buyback announcements via akshare.

    ``stock_repurchase_em()`` returns all historical records (~4800 rows).
    """
    import akshare as ak

    try:
        raw = ak.stock_repurchase_em()
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Buyback API returned no data")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    col_map = {
        "股票代码": "symbol",
        "最新公告日期": "announce_date",
        "已回购金额": "buyback_amount",
        "计划回购金额区间-上限": "buyback_amount_planned",
        "实施进度": "progress_status",
    }
    existing = set(raw.columns)
    rename = {k: v for k, v in col_map.items() if k in existing}
    df = raw.rename(columns=rename)

    wanted = ["symbol", "announce_date", "buyback_amount", "market_cap", "progress_status"]
    result = pd.DataFrame()
    for col in wanted:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = pd.NA
    result["source"] = "akshare:stock_repurchase_em"
    return result


def _fetch_reduction(start: str, end: str) -> pd.DataFrame:
    """Fetch major shareholder holding changes via akshare.

    ``stock_ggcg_em(symbol="全部")`` returns all shareholder increase/decrease
    records with pagination.  We filter to the requested date range afterwards
    because the API does not accept date parameters.
    """
    import akshare as ak

    try:
        raw = ak.stock_ggcg_em(symbol="全部")
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Shareholder change API returned no data")
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    col_map = {
        "代码": "symbol",
        "公告日": "announce_date",
        "股东名称": "holder_name",
        "持股变动信息-增减": "change_direction",
        "持股变动信息-变动数量": "change_amount",
        "持股变动信息-占总股本比例": "change_ratio_total",
        "持股变动信息-占流通股比例": "change_ratio_circulating",
        "变动后持股情况-持股总数": "post_change_total",
        "变动后持股情况-占总股本比例": "post_change_ratio_total",
        "变动后持股情况-持流通股数": "post_change_circulating",
        "变动后持股情况-占流通股比例": "post_change_ratio_circulating",
        "变动开始日": "change_start_date",
        "变动截止日": "change_end_date",
    }
    existing = set(raw.columns)
    rename = {k: v for k, v in col_map.items() if k in existing}
    df = raw.rename(columns=rename)

    wanted = [
        "symbol", "announce_date", "holder_name",
        "change_direction", "change_amount",
        "change_ratio_total", "change_ratio_circulating",
        "post_change_total", "post_change_ratio_total",
        "post_change_circulating", "post_change_ratio_circulating",
        "change_start_date", "change_end_date",
    ]
    result = pd.DataFrame()
    for col in wanted:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = pd.NA

    # Date filter on announce_date
    result["announce_date"] = pd.to_datetime(result["announce_date"], errors="coerce")
    result = result[
        (result["announce_date"] >= pd.Timestamp(start)) &
        (result["announce_date"] <= pd.Timestamp(end))
    ]
    result["source"] = "akshare:stock_ggcg_em"
    return result


def _fetch_unlock(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch restricted share unlock schedule via akshare.

    ``stock_restricted_release_detail_em(start_date, end_date)`` returns unlock events.
    May return empty/None for date ranges with no unlock events.
    """
    import akshare as ak

    try:
        raw = ak.stock_restricted_release_detail_em(start_date=start_date, end_date=end_date)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("Unlock API returned no data for %s → %s", start_date, end_date)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    col_map = {
        "股票代码": "symbol",
        "解禁时间": "unlock_date",
        "实际解禁数量": "unlock_shares",
        "实际解禁市值": "unlock_market_value",
        "限售股类型": "lockup_type",
    }
    existing = set(raw.columns)
    rename = {k: v for k, v in col_map.items() if k in existing}
    df = raw.rename(columns=rename)

    wanted = ["symbol", "unlock_date", "unlock_shares", "unlock_market_value", "lockup_type"]
    result = pd.DataFrame()
    for col in wanted:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = pd.NA
    # API does not provide a separate announcement date; use unlock_date as proxy
    result["announce_date"] = result["unlock_date"]
    result["source"] = "akshare:stock_restricted_release_detail_em"
    # Reorder to match table schema: symbol, announce_date, unlock_date, unlock_shares, ...
    result = result[["symbol", "announce_date", "unlock_date", "unlock_shares", "unlock_market_value", "lockup_type", "source"]]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Upsert helpers
# ═══════════════════════════════════════════════════════════════════════════

def _upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> int:
    """Insert or ignore rows by PRIMARY KEY. Returns number of rows inserted."""
    if df.empty:
        return 0

    # Deduplicate on key columns to avoid PK violations from source data
    df = df.drop_duplicates(subset=[c for c in key_cols if c in df.columns], keep="first")

    # Use temp table + INSERT WHERE NOT EXISTS for idempotency
    tmp = f"_tmp_{table}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    con.register(tmp, df)
    try:
        join_on = " AND ".join(f"t.{c} = s.{c}" for c in key_cols)
        con.execute(f"""
            INSERT INTO {table}
            SELECT s.* FROM {tmp} s
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} t
                WHERE {join_on}
            )
        """)
        return con.execute(f"SELECT COUNT(*) FROM {tmp}").fetchone()[0]
    finally:
        con.unregister(tmp)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch A-share event data and persist to DuckDB")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="", help="End date, defaults to today")
    p.add_argument("--db", default=DB_DEFAULT, help=f"DuckDB path (default: {DB_DEFAULT})")
    p.add_argument("--dry-run", action="store_true", help="Fetch but do not write to DB")
    p.add_argument(
        "--tables",
        default="earnings,buyback,reduction,unlock",
        help="Comma-separated list of event types to fetch",
    )
    p.add_argument("--report-periods", default="", help="Semicolon-separated report periods for earnings (e.g. 20260331;20251231)")
    p.add_argument("--sleep-sec", type=float, default=1.0, help="Sleep between fetches to avoid rate limits")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    tables = set(t.strip() for t in args.tables.split(","))

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    if not args.dry_run:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        _ensure_tables(con)
    else:
        con = None

    results: dict[str, int] = {}
    import time as _time

    try:
        # ── Earnings guidance ──────────────────────────────────────────
        if "earnings" in tables:
            if args.report_periods:
                periods = [p.strip() for p in args.report_periods.split(";") if p.strip()]
            else:
                # Generate quarterly report periods in the date range
                periods = []
                cursor = date(start_date.year, ((start_date.month - 1) // 3) * 3 + 1, 1)
                while cursor <= end_date:
                    # Report period is end of quarter: 0331, 0630, 0930, 1231
                    q_end_month = cursor.month + 2
                    q_end_day = 31 if q_end_month in (3, 12) else 30
                    period_str = f"{cursor.year}{q_end_month:02d}{q_end_day}"
                    if period_str not in periods:
                        periods.append(period_str)
                    # Next quarter
                    cursor_month = cursor.month + 3
                    cursor_year = cursor.year + (cursor_month - 1) // 12
                    cursor_month = ((cursor_month - 1) % 12) + 1
                    cursor = date(cursor_year, cursor_month, 1)

            logger.info("Earnings guidance: %d report periods to fetch", len(periods))
            total = 0
            for i, period in enumerate(periods):
                logger.info("  [%d/%d] report_period=%s", i + 1, len(periods), period)
                df = _fetch_earnings_guidance(period)
                if df.empty:
                    logger.info("    → 0 rows")
                    continue
                df["fetched_at"] = datetime.now()
                df["forecast_metric"] = df["forecast_metric"].fillna("")
                if not args.dry_run and con is not None:
                    n = _upsert(con, TABLE_EARNINGS, df, ["symbol", "announce_date", "forecast_metric"])
                    logger.info("    → %d rows (upserted %d new)", len(df), n)
                else:
                    logger.info("    → %d rows (dry-run)", len(df))
                total += len(df)
                _time.sleep(args.sleep_sec)
            results["earnings"] = total

        # ── Buyback ────────────────────────────────────────────────────
        if "buyback" in tables:
            logger.info("Buyback: fetching full history...")
            df = _fetch_buyback()
            if df.empty:
                logger.info("  → 0 rows")
                results["buyback"] = 0
            else:
                # Filter to date range
                df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
                df = df[
                    (df["announce_date"] >= pd.Timestamp(start_date)) &
                    (df["announce_date"] <= pd.Timestamp(end_date))
                ]
                df["fetched_at"] = datetime.now()
                if not args.dry_run and con is not None:
                    n = _upsert(con, TABLE_BUYBACK, df, ["symbol", "announce_date"])
                    logger.info("  → %d rows in range, %d new", len(df), n)
                else:
                    logger.info("  → %d rows (dry-run)", len(df))
                results["buyback"] = len(df)

        # ── Reduction ──────────────────────────────────────────────────
        if "reduction" in tables:
            logger.info("Reduction: fetching from %s to %s...", start_date, end_date)
            df = _fetch_reduction(str(start_date), str(end_date))
            if df.empty:
                logger.info("  → 0 rows")
                results["reduction"] = 0
            else:
                df["fetched_at"] = datetime.now()
                if not args.dry_run and con is not None:
                    n = _upsert(con, TABLE_REDUCTION, df, ["symbol", "announce_date", "holder_name"])
                    logger.info("  → %d rows, %d new", len(df), n)
                else:
                    logger.info("  → %d rows (dry-run)", len(df))
                results["reduction"] = len(df)

        # ── Unlock ─────────────────────────────────────────────────────
        if "unlock" in tables:
            logger.info("Unlock: fetching from %s to %s...", start_date, end_date)
            # Fetch in monthly chunks to avoid large single requests
            total = 0
            cursor = start_date
            while cursor <= end_date:
                chunk_end = min(
                    date(cursor.year + (cursor.month) // 12, ((cursor.month) % 12) + 1, 1) - timedelta(days=1),
                    end_date,
                )
                logger.info(
                    "  chunk %s → %s",
                    cursor.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
                df = _fetch_unlock(
                    cursor.strftime("%Y%m%d"),
                    chunk_end.strftime("%Y%m%d"),
                )
                if not df.empty:
                    df["fetched_at"] = datetime.now()
                    if not args.dry_run and con is not None:
                        _upsert(con, TABLE_UNLOCK, df, ["symbol", "announce_date", "unlock_date"])
                    total += len(df)
                cursor = chunk_end + timedelta(days=1)
                _time.sleep(args.sleep_sec)
            logger.info("  → %d rows total", total)
            results["unlock"] = total

    finally:
        if con is not None:
            con.close()

    # ── Final summary ──────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Fetch complete:")
    for table, count in sorted(results.items()):
        logger.info("  %-30s %d rows", table, count)
    logger.info("  %-30s %s", "mode", "DRY-RUN" if args.dry_run else "persisted")
    logger.info("  %-30s %s", "db", db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
