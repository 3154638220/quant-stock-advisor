"""W4: 结构化公告事件因子（M16 前置探索）。

数据源（DuckDB）:
    - a_share_event_earnings_guidance
    - a_share_event_buyback
    - a_share_event_reduction
    - a_share_event_unlock

PIT 规则:
    - 信号日 t 仅使用 announce_date <= t 的公告
    - 解禁因子只看 future 30 天窗口（unlock_date in (t, t+30]）
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

EVENT_FACTOR_COLS: tuple[str, ...] = (
    "feature_event_earnings_guidance_direction",
    "feature_event_earnings_guidance_magnitude",
    "feature_event_earnings_surprise_ttm",
    "feature_event_buyback_amount_ratio",
    "feature_event_buyback_recent_30d",
    "feature_event_reduction_plan_flag",
    "feature_event_unlock_ratio_30d",
)

EVENT_FACTOR_DIRECTION: dict[str, int] = {
    "feature_event_earnings_guidance_direction": 1,
    "feature_event_earnings_guidance_magnitude": 1,
    "feature_event_earnings_surprise_ttm": 1,
    "feature_event_buyback_amount_ratio": 1,
    "feature_event_buyback_recent_30d": 1,
    "feature_event_reduction_plan_flag": -1,
    "feature_event_unlock_ratio_30d": -1,
}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and int(row[0]) > 0)


def _available_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    if not _table_exists(con, table):
        return set()
    info = con.execute(f"PRAGMA table_info('{table}')").df()
    return set(info["name"].astype(str))


def _pick_col(cols: set[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _norm_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d{1,6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )


def _map_guidance_direction(x: object) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        if v > 0:
            return 1.0
        if v < 0:
            return -1.0
        return 0.0
    txt = str(x).strip().lower()
    up_tokens = ("up", "raise", "positive", "预增", "扭亏", "续盈", "略增", "上调")
    down_tokens = ("down", "cut", "negative", "预减", "预亏", "首亏", "续亏", "略减", "下调")
    if any(tok in txt for tok in up_tokens):
        return 1.0
    if any(tok in txt for tok in down_tokens):
        return -1.0
    return 0.0


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _compute_guidance_features(
    con: duckdb.DuckDBPyConnection,
    sd: pd.Timestamp,
) -> pd.DataFrame:
    table = "a_share_event_earnings_guidance"
    cols = _available_columns(con, table)
    if not cols:
        return pd.DataFrame(columns=[
            "symbol",
            "feature_event_earnings_guidance_direction",
            "feature_event_earnings_guidance_magnitude",
            "feature_event_earnings_surprise_ttm",
        ])

    direction_col = _pick_col(cols, ("guidance_direction", "direction", "forecast_direction"))
    magnitude_col = _pick_col(cols, ("guidance_change_ratio", "change_ratio", "guidance_ratio"))
    np_min_col = _pick_col(cols, ("expected_net_profit_min", "net_profit_min", "forecast_profit_min"))
    np_max_col = _pick_col(cols, ("expected_net_profit_max", "net_profit_max", "forecast_profit_max"))
    prev_np_col = _pick_col(cols, ("prev_year_net_profit", "last_year_net_profit", "net_profit_prev_year"))

    select_cols = ["symbol", "announce_date"]
    for col in [direction_col, magnitude_col, np_min_col, np_max_col, prev_np_col]:
        if col and col not in select_cols:
            select_cols.append(col)

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {table}
        WHERE announce_date IS NOT NULL
          AND CAST(announce_date AS DATE) <= CAST(? AS DATE)
          AND CAST(announce_date AS DATE) >= CAST(? AS DATE)
    """
    hist_start = sd - pd.Timedelta(days=730)
    df = con.execute(query, [sd.strftime("%Y-%m-%d"), hist_start.strftime("%Y-%m-%d")]).df()
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "feature_event_earnings_guidance_direction",
            "feature_event_earnings_guidance_magnitude",
            "feature_event_earnings_surprise_ttm",
        ])

    df["symbol"] = _norm_symbol(df["symbol"])
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce").dt.normalize()
    if direction_col:
        df[direction_col] = df[direction_col].apply(_map_guidance_direction)
    if magnitude_col:
        df[magnitude_col] = pd.to_numeric(df[magnitude_col], errors="coerce")
    if np_min_col:
        df[np_min_col] = pd.to_numeric(df[np_min_col], errors="coerce")
    if np_max_col:
        df[np_max_col] = pd.to_numeric(df[np_max_col], errors="coerce")
    if prev_np_col:
        df[prev_np_col] = pd.to_numeric(df[prev_np_col], errors="coerce")

    if not magnitude_col and np_min_col and np_max_col and prev_np_col:
        mid = (df[np_min_col] + df[np_max_col]) / 2.0
        df["_derived_magnitude"] = _safe_div(mid, df[prev_np_col]) - 1.0
        magnitude_col = "_derived_magnitude"

    if not direction_col and magnitude_col:
        df["_derived_direction"] = np.sign(pd.to_numeric(df[magnitude_col], errors="coerce"))
        direction_col = "_derived_direction"

    df = df.sort_values(["symbol", "announce_date"], kind="mergesort")
    latest = df.groupby("symbol", as_index=False).tail(1).copy()

    out = latest[["symbol"]].copy()
    out["feature_event_earnings_guidance_direction"] = (
        pd.to_numeric(latest.get(direction_col), errors="coerce") if direction_col else np.nan
    )
    out["feature_event_earnings_guidance_magnitude"] = (
        pd.to_numeric(latest.get(magnitude_col), errors="coerce") if magnitude_col else np.nan
    )

    # surprise_ttm: 最新预告幅度相对历史 2 年分布的标准化偏离
    if magnitude_col:
        hist = df[["symbol", magnitude_col]].copy()

        def _zscore(g: pd.DataFrame) -> float:
            vals = pd.to_numeric(g[magnitude_col], errors="coerce").dropna()
            if len(vals) < 3:
                return np.nan
            latest_v = float(vals.iloc[-1])
            hist_v = vals.iloc[:-1]
            if len(hist_v) < 2:
                return np.nan
            mu = float(hist_v.mean())
            sigma = float(hist_v.std(ddof=1))
            if not np.isfinite(sigma) or sigma == 0:
                return 0.0
            return float((latest_v - mu) / sigma)

        surprise = (
            hist.groupby("symbol", as_index=False)
            .apply(_zscore, include_groups=False)
            .rename(columns={None: "feature_event_earnings_surprise_ttm"})
        )
        if "feature_event_earnings_surprise_ttm" not in surprise.columns:
            surprise.columns = ["symbol", "feature_event_earnings_surprise_ttm"]
        out = out.merge(surprise, on="symbol", how="left")
    else:
        out["feature_event_earnings_surprise_ttm"] = np.nan

    return out


def _compute_buyback_features(
    con: duckdb.DuckDBPyConnection,
    sd: pd.Timestamp,
) -> pd.DataFrame:
    table = "a_share_event_buyback"
    cols = _available_columns(con, table)
    if not cols:
        return pd.DataFrame(columns=[
            "symbol",
            "feature_event_buyback_amount_ratio",
            "feature_event_buyback_recent_30d",
        ])

    amount_col = _pick_col(cols, ("buyback_amount", "planned_amount", "amount"))
    mcap_col = _pick_col(cols, ("market_cap", "announce_market_cap", "total_market_cap"))
    if amount_col is None:
        return pd.DataFrame(columns=[
            "symbol",
            "feature_event_buyback_amount_ratio",
            "feature_event_buyback_recent_30d",
        ])

    select_cols = ["symbol", "announce_date", amount_col]
    if mcap_col:
        select_cols.append(mcap_col)
    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {table}
        WHERE announce_date IS NOT NULL
          AND CAST(announce_date AS DATE) <= CAST(? AS DATE)
          AND CAST(announce_date AS DATE) > CAST(? AS DATE)
    """
    win_start = sd - pd.Timedelta(days=30)
    df = con.execute(query, [sd.strftime("%Y-%m-%d"), win_start.strftime("%Y-%m-%d")]).df()
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "feature_event_buyback_amount_ratio",
            "feature_event_buyback_recent_30d",
        ])

    df["symbol"] = _norm_symbol(df["symbol"])
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    if mcap_col:
        df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
        df["_ratio"] = _safe_div(df[amount_col], df[mcap_col])
    else:
        df["_ratio"] = np.nan

    agg = df.groupby("symbol", as_index=False).agg(
        feature_event_buyback_amount_ratio=("_ratio", "sum"),
        _cnt=(amount_col, "count"),
    )
    agg["feature_event_buyback_recent_30d"] = (agg["_cnt"] > 0).astype(float)
    return agg[["symbol", "feature_event_buyback_amount_ratio", "feature_event_buyback_recent_30d"]]


def _compute_reduction_features(
    con: duckdb.DuckDBPyConnection,
    sd: pd.Timestamp,
) -> pd.DataFrame:
    table = "a_share_event_reduction"
    cols = _available_columns(con, table)
    if not cols:
        return pd.DataFrame(columns=["symbol", "feature_event_reduction_plan_flag"])

    query = f"""
        SELECT symbol, announce_date
        FROM {table}
        WHERE announce_date IS NOT NULL
          AND CAST(announce_date AS DATE) <= CAST(? AS DATE)
          AND CAST(announce_date AS DATE) > CAST(? AS DATE)
    """
    win_start = sd - pd.Timedelta(days=30)
    df = con.execute(query, [sd.strftime("%Y-%m-%d"), win_start.strftime("%Y-%m-%d")]).df()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_event_reduction_plan_flag"])

    df["symbol"] = _norm_symbol(df["symbol"])
    agg = df.groupby("symbol", as_index=False).size().rename(columns={"size": "_cnt"})
    agg["feature_event_reduction_plan_flag"] = (agg["_cnt"] > 0).astype(float)
    return agg[["symbol", "feature_event_reduction_plan_flag"]]


def _compute_unlock_features(
    con: duckdb.DuckDBPyConnection,
    sd: pd.Timestamp,
) -> pd.DataFrame:
    table = "a_share_event_unlock"
    cols = _available_columns(con, table)
    if not cols:
        return pd.DataFrame(columns=["symbol", "feature_event_unlock_ratio_30d"])

    unlock_col = _pick_col(cols, ("unlock_market_value", "unlock_amount", "unlock_value"))
    unlock_date_col = _pick_col(cols, ("unlock_date", "listing_date", "effective_date"))
    mcap_col = _pick_col(cols, ("market_cap", "announce_market_cap", "total_market_cap"))
    if unlock_col is None or unlock_date_col is None:
        return pd.DataFrame(columns=["symbol", "feature_event_unlock_ratio_30d"])

    select_cols = ["symbol", "announce_date", unlock_date_col, unlock_col]
    if mcap_col:
        select_cols.append(mcap_col)
    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {table}
        WHERE announce_date IS NOT NULL
          AND CAST(announce_date AS DATE) <= CAST(? AS DATE)
          AND CAST({unlock_date_col} AS DATE) > CAST(? AS DATE)
          AND CAST({unlock_date_col} AS DATE) <= CAST(? AS DATE)
    """
    win_end = sd + pd.Timedelta(days=30)
    df = con.execute(
        query,
        [sd.strftime("%Y-%m-%d"), sd.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")],
    ).df()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_event_unlock_ratio_30d"])

    df["symbol"] = _norm_symbol(df["symbol"])
    df[unlock_col] = pd.to_numeric(df[unlock_col], errors="coerce")
    if mcap_col:
        df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
        df["_ratio"] = _safe_div(df[unlock_col], df[mcap_col])
    else:
        df["_ratio"] = np.nan

    agg = df.groupby("symbol", as_index=False).agg(
        feature_event_unlock_ratio_30d=("_ratio", "sum")
    )
    return agg


def compute_event_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    min_coverage: float = 0.10,
) -> pd.DataFrame:
    """计算单个信号日的事件因子截面。"""
    sd = pd.Timestamp(signal_date).normalize()
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception:
        return pd.DataFrame()

    try:
        guid = _compute_guidance_features(con, sd)
        buyback = _compute_buyback_features(con, sd)
        reduction = _compute_reduction_features(con, sd)
        unlock = _compute_unlock_features(con, sd)
    finally:
        con.close()

    frames = [x for x in [guid, buyback, reduction, unlock] if not x.empty]
    if not frames:
        return pd.DataFrame()

    result = frames[0].copy()
    for f in frames[1:]:
        result = result.merge(f, on="symbol", how="outer")

    # 有事件但比率不可计算时，至少保留事件 flag
    if "feature_event_buyback_recent_30d" not in result.columns:
        result["feature_event_buyback_recent_30d"] = 0.0
    if "feature_event_reduction_plan_flag" not in result.columns:
        result["feature_event_reduction_plan_flag"] = 0.0

    result["trade_date"] = sd

    for col in EVENT_FACTOR_COLS:
        if col not in result.columns:
            result[col] = np.nan
        cov = result[col].notna().mean()
        if cov < min_coverage:
            result[col] = np.nan

    return result[["symbol", "trade_date"] + list(EVENT_FACTOR_COLS)]


def attach_event_factors(
    factors: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """按 signal_date 批量附加事件因子。"""
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    date_col = "trade_date" if "trade_date" in out.columns else "signal_date"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    unique_dates = sorted(out[date_col].dropna().unique())
    frames: list[pd.DataFrame] = []
    for dt in unique_dates:
        ef = compute_event_factors(db_path, signal_date=dt)
        if ef.empty:
            continue
        ef[date_col] = pd.Timestamp(dt)
        frames.append(ef)

    if not frames:
        for col in EVENT_FACTOR_COLS:
            out[col] = np.nan
        return out

    all_events = pd.concat(frames, ignore_index=True)
    all_events[date_col] = pd.to_datetime(all_events[date_col], errors="coerce").dt.normalize()
    return out.merge(
        all_events[["symbol", date_col] + list(EVENT_FACTOR_COLS)],
        on=["symbol", date_col],
        how="left",
    )
