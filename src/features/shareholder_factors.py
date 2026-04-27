"""股东人数因子：基于报告期末股东户数做 PIT 接入。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS = 30


def attach_shareholder_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_shareholder",
    availability_lag_days: int = DEFAULT_SHAREHOLDER_AVAILABILITY_LAG_DAYS,
) -> pd.DataFrame:
    """
    将股东人数表按 PIT 方式接入到日频因子表。

    优先使用 ``notice_date`` 作为可用日；若旧表尚未存该列，
    则退化为 ``end_date + availability_lag_days`` 的保守近似。
    """
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    out = out.dropna(subset=["trade_date"]).copy()

    con = duckdb.connect(str(Path(db_path)), read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return _attach_empty_shareholder_columns(out)
        raw = con.execute(
            f"""
            SELECT *
            FROM {table_name}
            """
        ).df()
    finally:
        con.close()

    if raw.empty:
        return _attach_empty_shareholder_columns(out)

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["end_date"] = pd.to_datetime(raw["end_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    if "notice_date" in raw.columns:
        raw["notice_date"] = pd.to_datetime(raw["notice_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    else:
        raw["notice_date"] = pd.NaT
    raw["holder_count"] = pd.to_numeric(raw["holder_count"], errors="coerce")
    raw["holder_change"] = pd.to_numeric(raw["holder_change"], errors="coerce")
    raw = raw.dropna(subset=["end_date"]).copy()
    if raw.empty:
        return _attach_empty_shareholder_columns(out)

    raw["availability_date"] = raw["notice_date"]
    # A notice date before the reporting period end is not PIT-safe; use the
    # conservative lag fallback instead of trusting the source field.
    fallback_mask = raw["availability_date"].isna() | (raw["availability_date"] < raw["end_date"])
    raw.loc[fallback_mask, "availability_date"] = raw.loc[fallback_mask, "end_date"] + pd.to_timedelta(
        int(availability_lag_days),
        unit="D",
    )
    raw = raw.sort_values(["symbol", "availability_date", "end_date"], kind="mergesort")
    raw = raw.drop_duplicates(["symbol", "availability_date", "end_date"], keep="last")

    merged_chunks: list[pd.DataFrame] = []
    want_cols = ["symbol", "end_date", "availability_date", "holder_count", "holder_change"]
    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    for _, chunk in out.groupby(pd.Grouper(key="trade_date", freq="31D"), sort=True):
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
        chunk_end = pd.Timestamp(chunk["trade_date"].max())
        chunk_symbols = chunk["symbol"].astype(str).unique().tolist()
        raw_chunk = raw[
            (raw["availability_date"] <= chunk_end) & raw["symbol"].astype(str).isin(chunk_symbols)
        ][want_cols].copy()
        if raw_chunk.empty:
            merged = chunk.copy()
            for c in want_cols[1:]:
                if c not in merged.columns:
                    merged[c] = np.nan
        else:
            raw_chunk = raw_chunk.sort_values(["availability_date", "symbol"], kind="mergesort").reset_index(drop=True)
            merged = pd.merge_asof(
                chunk,
                raw_chunk,
                left_on="trade_date",
                right_on="availability_date",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
        merged_chunks.append(merged)

    if not merged_chunks:
        return _attach_empty_shareholder_columns(out)

    merged = pd.concat(merged_chunks, ignore_index=True)
    merged = merged.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)

    holder_count = pd.to_numeric(merged["holder_count"], errors="coerce")
    holder_change = pd.to_numeric(merged["holder_change"], errors="coerce")
    merged["holder_count_log"] = np.log(holder_count.where(holder_count > 0))
    merged["holder_count_change_pct"] = (
        merged.groupby("symbol", sort=False)["holder_count"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").pct_change(fill_method=None))
        .replace([np.inf, -np.inf], np.nan)
    )
    if holder_change.notna().any():
        merged["holder_change_rate"] = holder_change / holder_count.replace(0, np.nan)
    else:
        merged["holder_change_rate"] = merged["holder_count_change_pct"]
    merged["holder_change_rate"] = pd.to_numeric(merged["holder_change_rate"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    merged["holder_change_rate_z"] = _cross_sectional_z(merged, merged["holder_change_rate"])
    merged["holder_count_log_z"] = _cross_sectional_z(merged, merged["holder_count_log"])
    merged["holder_concentration_proxy"] = -merged["holder_count_log_z"]
    return merged.drop(columns=["end_date", "notice_date", "availability_date"], errors="ignore")


def _attach_empty_shareholder_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)
    for col in (
        "holder_count",
        "holder_change",
        "holder_count_log",
        "holder_count_change_pct",
        "holder_change_rate",
        "holder_change_rate_z",
        "holder_count_log_z",
        "holder_concentration_proxy",
    ):
        out[col] = np.nan
    return out


def _cross_sectional_z(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    grouped = pd.to_numeric(series, errors="coerce").groupby(df["trade_date"], sort=False)
    mu = grouped.transform("mean")
    sd = grouped.transform("std", ddof=1).replace(0, np.nan)
    return ((pd.to_numeric(series, errors="coerce") - mu) / sd).clip(-3.0, 3.0)
