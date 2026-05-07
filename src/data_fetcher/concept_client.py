"""概念板块数据拉取与质量诊断。

数据源: AkShare 同花顺（THS）concept APIs，东方财富（EM）概念成分股快照。
API:
- ``stock_board_concept_name_ths()`` → 概念名称与代码（375 个概念）
- ``stock_board_concept_index_ths(symbol, start_date, end_date)`` → 日线 OHLCV
- ``stock_board_concept_info_ths(symbol)`` → 概念详情（涨跌家数等）
- ``stock_board_concept_cons_em(symbol)`` → 当前成分股快照（M13-B）

历史起点: 各概念板块不同，最早约 2019 年。
PIT-safety: 板块日线 OHLCV 为 T 日收盘后可见；成分股仅按 snapshot_date
及滚动有效期使用，不回填到快照日前，避免历史回测前瞻。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

_LOG = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 3.0
DEFAULT_BATCH_DELAY = 1.5
DEFAULT_MEMBERSHIP_MAX_RETRIES = 3


def fetch_concept_list(
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> pd.DataFrame:
    """拉取同花顺全量概念板块名称与代码。

    Returns
    -------
    DataFrame with columns: concept_code, concept_name
    """
    import akshare as ak

    last_err = None
    for attempt in range(max_retries):
        try:
            raw = ak.stock_board_concept_name_ths()
            if raw is None or raw.empty:
                raise RuntimeError("empty concept list")
            raw = raw.rename(
                columns={
                    "code": "concept_code",
                    "name": "concept_name",
                }
            )
            raw["concept_code"] = raw["concept_code"].astype(str)
            return raw
        except Exception as e:
            last_err = e
            _LOG.warning("fetch_concept_list attempt %d/%d: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"fetch_concept_list failed after {max_retries} retries: {last_err}")


def fetch_concept_daily_history(
    concept_name: str,
    *,
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> pd.DataFrame:
    """拉取单个概念板块日线历史 OHLCV（同花顺指数）。

    Parameters
    ----------
    concept_name : str
        概念板块名称，如 ``"AI PC"``。
    start_date : str
        起始日期 YYYYMMDD。
    end_date : str | None
        截止日期 YYYYMMDD，None 表示今日。

    Returns
    -------
    DataFrame with columns: trade_date, open, close, high, low, pct_chg, volume, amount
    """
    import akshare as ak

    today = date.today().strftime("%Y%m%d")
    if end_date is None:
        end_date = today

    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_board_concept_index_ths(
                symbol=concept_name,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                _LOG.debug("%s: empty response", concept_name)
                return pd.DataFrame()

            df = df.rename(
                columns={
                    "日期": "trade_date",
                    "开盘价": "open",
                    "收盘价": "close",
                    "最高价": "high",
                    "最低价": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            keep_cols = ["trade_date", "open", "close", "high", "low", "volume", "amount"]
            df = df[[c for c in keep_cols if c in df.columns]]
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.dropna(subset=["trade_date"])
            for c in ["open", "close", "high", "low", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # 计算涨跌幅 (pct_chg): 用 close 计算日收益率
            df = df.sort_values("trade_date")
            df["prev_close"] = df["close"].shift(1)
            df["pct_chg"] = (df["close"] - df["prev_close"]) / df["prev_close"] * 100
            df = df.drop(columns=["prev_close"])
            return df

        except Exception as e:
            last_err = e
            _LOG.debug("%s attempt %d/%d: %s", concept_name, attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    _LOG.warning("%s: failed after %d retries: %s", concept_name, max_retries, last_err)
    return pd.DataFrame()


def backfill_all_concepts(
    conn,
    *,
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    concept_limit: Optional[int] = None,
    batch_delay: float = DEFAULT_BATCH_DELAY,
) -> dict:
    """拉取全量概念板块日线历史并写入 DuckDB。

    Returns
    -------
    dict with keys: concepts_total, concepts_success, concepts_empty, total_rows
    """
    import duckdb

    meta = fetch_concept_list()
    concepts = meta[["concept_code", "concept_name"]].copy()
    if concept_limit:
        concepts = concepts.head(concept_limit)

    _LOG.info("backfilling %d concepts (THS) from %s", len(concepts), start_date)

    success, empty_count, total_rows = 0, 0, 0
    for i, (_, row) in enumerate(concepts.iterrows()):
        name = row["concept_name"]
        code = str(row["concept_code"])

        if i > 0:
            time.sleep(batch_delay)

        df = fetch_concept_daily_history(name, start_date=start_date, end_date=end_date)
        if df.empty:
            empty_count += 1
            _LOG.info("[%d/%d] %s %s: empty", i + 1, len(concepts), code, name)
            continue

        df["concept_code"] = code
        conn.execute(
            """
            INSERT OR REPLACE INTO a_share_concept_daily
            (concept_code, trade_date, open, close, high, low, pct_chg, volume, amount)
            SELECT concept_code, trade_date, open, close, high, low, pct_chg, volume, amount
            FROM df
            """,
        )
        success += 1
        total_rows += len(df)
        _LOG.info("[%d/%d] %s %s: %d rows", i + 1, len(concepts), code, name, len(df))

    # 写入 meta 表
    meta_insert = meta[["concept_code", "concept_name"]].copy()
    meta_insert["first_seen_date"] = date.today()
    meta_insert["stock_count"] = None
    meta_insert["total_mv"] = None

    conn.execute("DELETE FROM a_share_concept_meta")
    conn.execute(
        """
        INSERT INTO a_share_concept_meta
        (concept_code, concept_name, stock_count, total_mv, first_seen_date)
        SELECT concept_code, concept_name, stock_count, total_mv, first_seen_date
        FROM meta_insert
        """,
    )

    return {
        "concepts_total": len(concepts),
        "concepts_success": success,
        "concepts_empty": empty_count,
        "total_rows": total_rows,
    }


# ── M13-B: 当前概念成分股快照 ─────────────────────────────────────────────

_SYMBOL_COL_ALIASES = ("代码", "证券代码", "股票代码", "symbol", "code")
_STOCK_NAME_COL_ALIASES = ("名称", "股票名称", "证券简称", "stock_name", "name")


def _first_existing_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    cols = {str(c): str(c) for c in df.columns}
    for alias in aliases:
        if alias in cols:
            return cols[alias]
    return None


def _normalize_membership_frame(
    raw: pd.DataFrame,
    *,
    concept_code: str,
    concept_name: str,
    snapshot_date: date,
    source: str,
) -> pd.DataFrame:
    """标准化 AkShare 概念成分股响应。"""
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol", "concept_code", "concept_name", "snapshot_date",
                "entry_date", "exit_date", "source",
            ]
        )

    symbol_col = _first_existing_col(raw, _SYMBOL_COL_ALIASES)
    if symbol_col is None:
        raise RuntimeError(f"membership response missing symbol column: {list(raw.columns)}")

    out = pd.DataFrame()
    out["symbol"] = (
        raw[symbol_col]
        .astype(str)
        .str.extract(r"(\d{1,6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )
    out = out[out["symbol"].str.fullmatch(r"\d{6}")].copy()
    out["concept_code"] = str(concept_code)
    out["concept_name"] = str(concept_name)
    out["snapshot_date"] = pd.to_datetime(snapshot_date).date()
    out["entry_date"] = pd.to_datetime(snapshot_date).date()
    out["exit_date"] = pd.NaT
    out["source"] = source

    stock_name_col = _first_existing_col(raw, _STOCK_NAME_COL_ALIASES)
    if stock_name_col is not None:
        out["stock_name"] = raw.loc[out.index, stock_name_col].astype(str)

    return out.drop_duplicates(["symbol", "concept_code", "snapshot_date"], keep="last")


def fetch_concept_members_em(
    concept_name: str,
    *,
    max_retries: int = DEFAULT_MEMBERSHIP_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> pd.DataFrame:
    """拉取东方财富概念当前成分股。"""
    import akshare as ak

    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_board_concept_cons_em(symbol=concept_name)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            last_err = e
            _LOG.warning(
                "fetch_concept_members_em %s attempt %d/%d: %s",
                concept_name, attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"fetch_concept_members_em({concept_name}) failed: {last_err}")


def fetch_concept_membership_snapshot(
    concepts: pd.DataFrame,
    *,
    snapshot_date: date | None = None,
    concept_limit: Optional[int] = None,
    batch_delay: float = DEFAULT_BATCH_DELAY,
) -> tuple[pd.DataFrame, dict]:
    """按概念列表拉取当前成分股快照。

    Parameters
    ----------
    concepts
        DataFrame with columns ``concept_code`` and ``concept_name``.

    Returns
    -------
    (membership, stats)
    """
    snap = snapshot_date or date.today()
    work = concepts[["concept_code", "concept_name"]].dropna().copy()
    if concept_limit:
        work = work.head(concept_limit)

    frames: list[pd.DataFrame] = []
    success, empty_count, failed = 0, 0, 0
    for i, (_, row) in enumerate(work.iterrows()):
        if i > 0:
            time.sleep(batch_delay)
        concept_code = str(row["concept_code"])
        concept_name = str(row["concept_name"])
        try:
            raw = fetch_concept_members_em(concept_name)
            norm = _normalize_membership_frame(
                raw,
                concept_code=concept_code,
                concept_name=concept_name,
                snapshot_date=snap,
                source="akshare.stock_board_concept_cons_em",
            )
            if norm.empty:
                empty_count += 1
            else:
                frames.append(norm)
                success += 1
            _LOG.info("[%d/%d] %s %s: %d members", i + 1, len(work), concept_code, concept_name, len(norm))
        except Exception as e:
            failed += 1
            _LOG.warning("[%d/%d] %s %s: failed: %s", i + 1, len(work), concept_code, concept_name, e)

    membership = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[
            "symbol", "concept_code", "concept_name", "snapshot_date",
            "entry_date", "exit_date", "source",
        ]
    )
    stats = {
        "concepts_total": int(len(work)),
        "concepts_success": int(success),
        "concepts_empty": int(empty_count),
        "concepts_failed": int(failed),
        "membership_rows": int(len(membership)),
        "membership_symbols": int(membership["symbol"].nunique()) if not membership.empty else 0,
    }
    return membership, stats


def backfill_concept_membership_snapshot(
    conn,
    *,
    snapshot_date: date | None = None,
    concept_limit: Optional[int] = None,
    batch_delay: float = DEFAULT_BATCH_DELAY,
) -> dict:
    """拉取并写入当前个股-概念成分股快照。"""
    meta = conn.execute(
        """
        SELECT concept_code, concept_name
        FROM a_share_concept_meta
        ORDER BY concept_code
        """
    ).df()
    if meta.empty:
        meta = fetch_concept_list()

    membership, stats = fetch_concept_membership_snapshot(
        meta,
        snapshot_date=snapshot_date,
        concept_limit=concept_limit,
        batch_delay=batch_delay,
    )
    if membership.empty:
        return stats

    snap = pd.to_datetime(membership["snapshot_date"].iloc[0]).date()
    conn.execute("DELETE FROM a_share_concept_membership WHERE snapshot_date = ?", [snap])
    conn.execute(
        """
        INSERT OR REPLACE INTO a_share_concept_membership
        (symbol, concept_code, snapshot_date, concept_name, entry_date, exit_date, source, fetched_at)
        SELECT symbol, concept_code, snapshot_date, concept_name, entry_date, exit_date, source, CURRENT_TIMESTAMP
        FROM membership
        """,
    )
    return stats


# ── 质量诊断 ────────────────────────────────────────────────────────────────

@dataclass
class ConceptQualityReport:
    concept_count: int
    daily_date_min: Optional[date]
    daily_date_max: Optional[date]
    daily_total_rows: int
    daily_concept_count: int
    pct_chg_null_rate: float
    coverage_pct: float
    membership_rows: int = 0
    membership_symbol_count: int = 0
    membership_concept_count: int = 0
    membership_snapshot_min: Optional[date] = None
    membership_snapshot_max: Optional[date] = None


def diagnose_concept_quality(conn) -> ConceptQualityReport:
    """对 a_share_concept_* 表进行质量诊断。"""
    meta_row = conn.execute("SELECT COUNT(*) FROM a_share_concept_meta").fetchone()
    concept_count = int(meta_row[0]) if meta_row else 0

    range_row = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM a_share_concept_daily"
    ).fetchone()
    daily_date_min = range_row[0] if range_row else None
    daily_date_max = range_row[1] if range_row else None
    daily_total_rows = int(range_row[2]) if range_row else 0

    dc_row = conn.execute(
        "SELECT COUNT(DISTINCT concept_code) FROM a_share_concept_daily"
    ).fetchone()
    daily_concept_count = int(dc_row[0]) if dc_row else 0

    null_row = conn.execute(
        "SELECT COUNT(*) FROM a_share_concept_daily WHERE pct_chg IS NULL"
    ).fetchone()
    null_count = int(null_row[0]) if null_row else 0
    pct_chg_null_rate = null_count / daily_total_rows if daily_total_rows > 0 else 1.0
    coverage_pct = daily_concept_count / concept_count * 100 if concept_count > 0 else 0.0

    membership_rows = 0
    membership_symbol_count = 0
    membership_concept_count = 0
    membership_snapshot_min = None
    membership_snapshot_max = None
    exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'a_share_concept_membership'"
    ).fetchone()
    if exists and int(exists[0]) > 0:
        m_row = conn.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT symbol),
                COUNT(DISTINCT concept_code),
                MIN(snapshot_date),
                MAX(snapshot_date)
            FROM a_share_concept_membership
            """
        ).fetchone()
        if m_row:
            membership_rows = int(m_row[0])
            membership_symbol_count = int(m_row[1])
            membership_concept_count = int(m_row[2])
            membership_snapshot_min = m_row[3]
            membership_snapshot_max = m_row[4]

    return ConceptQualityReport(
        concept_count=concept_count,
        daily_date_min=daily_date_min,
        daily_date_max=daily_date_max,
        daily_total_rows=daily_total_rows,
        daily_concept_count=daily_concept_count,
        pct_chg_null_rate=pct_chg_null_rate,
        coverage_pct=coverage_pct,
        membership_rows=membership_rows,
        membership_symbol_count=membership_symbol_count,
        membership_concept_count=membership_concept_count,
        membership_snapshot_min=membership_snapshot_min,
        membership_snapshot_max=membership_snapshot_max,
    )
