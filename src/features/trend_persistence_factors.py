"""W7: 多空趋势持续性因子。

基于 a_share_daily 表的收盘价，用 EMA12/EMA26 近似复现东方财富
「多空趋势线」红绿状态，并在月度信号日构造状态持续性截面因子。
所有计算只使用 trade_date <= signal_date 的日线数据。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

TREND_PERSISTENCE_FACTOR_COLS: tuple[str, ...] = (
    "feature_trend_bull_state",
    "feature_trend_streak_days",
    "feature_trend_bull_ratio_20d",
    "feature_trend_bull_ratio_60d",
    "feature_trend_flip_days_ago",
    "feature_trend_ema_spread",
)

TREND_PERSISTENCE_FACTOR_DIRECTION: dict[str, int] = {
    "feature_trend_bull_state": 1,
    "feature_trend_streak_days": 1,
    "feature_trend_bull_ratio_20d": 1,
    "feature_trend_bull_ratio_60d": 1,
    "feature_trend_flip_days_ago": 1,
    "feature_trend_ema_spread": 1,
}


def compute_trend_persistence_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    table_name: str = "a_share_daily",
    min_history_days: int = 90,
    min_coverage: float = 0.30,
    fast_span: int = 12,
    slow_span: int = 26,
) -> pd.DataFrame:
    """从日线数据计算多空趋势持续性因子截面。

    Parameters
    ----------
    db_path : str
        DuckDB 数据库路径。
    signal_date : str or Timestamp
        信号日，只使用 trade_date <= signal_date 的数据。
    table_name : str
        日线表名。
    min_history_days : int
        每只股票最少需要的交易日数。
    min_coverage : float
        最低截面覆盖率阈值，低于阈值的列会置为 NaN。
    fast_span, slow_span : int
        EMA 快慢线参数，默认 12/26。

    Returns
    -------
    pd.DataFrame
        包含 symbol, trade_date (=signal_date) 及因子列。
    """
    sd = pd.Timestamp(signal_date).normalize()
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception:
        return pd.DataFrame()

    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return pd.DataFrame()

        lookback_start = sd - pd.Timedelta(days=260)
        raw = con.execute(
            f"""
            SELECT symbol, trade_date, close
            FROM {table_name}
            WHERE trade_date <= CAST(? AS DATE)
              AND trade_date >= CAST(? AS DATE)
            ORDER BY symbol, trade_date
            """,
            [sd.strftime("%Y-%m-%d"), lookback_start.strftime("%Y-%m-%d")],
        ).df()
    finally:
        con.close()

    if raw.empty:
        return pd.DataFrame()

    raw["symbol"] = raw["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna(subset=["symbol", "trade_date", "close"])
    raw = raw[raw["close"] > 0].sort_values(["symbol", "trade_date"], kind="mergesort")

    day_counts = raw.groupby("symbol").size()
    valid_symbols = day_counts[day_counts >= min_history_days].index
    raw = raw[raw["symbol"].isin(valid_symbols)].copy()
    if raw.empty:
        return pd.DataFrame()

    g = raw.groupby("symbol", sort=False)
    raw["_ema_fast"] = g["close"].transform(
        lambda s: s.ewm(span=fast_span, min_periods=fast_span, adjust=False).mean()
    )
    raw["_ema_slow"] = g["close"].transform(
        lambda s: s.ewm(span=slow_span, min_periods=slow_span, adjust=False).mean()
    )
    raw["_bull_state"] = (raw["_ema_fast"] > raw["_ema_slow"]).astype(float)
    raw.loc[raw[["_ema_fast", "_ema_slow"]].isna().any(axis=1), "_bull_state"] = np.nan

    raw["feature_trend_bull_state"] = raw["_bull_state"]
    raw["feature_trend_streak_days"] = _compute_signed_state_streak(raw, "_bull_state")
    raw["feature_trend_bull_ratio_20d"] = g["_bull_state"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    raw["feature_trend_bull_ratio_60d"] = g["_bull_state"].transform(
        lambda s: s.rolling(60, min_periods=30).mean()
    )
    raw["feature_trend_flip_days_ago"] = raw["feature_trend_streak_days"].abs() - 1.0
    raw.loc[raw["feature_trend_streak_days"].isna(), "feature_trend_flip_days_ago"] = np.nan
    raw["feature_trend_ema_spread"] = (raw["_ema_fast"] - raw["_ema_slow"]) / raw["close"].replace(0, np.nan)

    # 若 signal_date 不是交易日，取每只股票 signal_date 前最后一个可用交易日；
    # 输出 trade_date 仍规范化为 signal_date，便于月度截面合并。
    latest_idx = raw.groupby("symbol", sort=False)["trade_date"].idxmax()
    result = raw.loc[latest_idx, ["symbol", *TREND_PERSISTENCE_FACTOR_COLS]].copy()
    result.insert(1, "trade_date", sd)
    result = result.drop_duplicates(["symbol"])

    for col in TREND_PERSISTENCE_FACTOR_COLS:
        cov = result[col].notna().mean() if len(result) else 0.0
        if cov < min_coverage:
            result[col] = np.nan

    return result


def _compute_signed_state_streak(df: pd.DataFrame, state_col: str) -> pd.Series:
    """Return signed consecutive bull/bear state length per symbol."""
    streak = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby("symbol", sort=False):
        cur = 0
        last_state: float | None = None
        vals: list[float] = []
        for val in pd.to_numeric(grp[state_col], errors="coerce"):
            if pd.isna(val):
                cur = 0
                last_state = None
                vals.append(np.nan)
                continue
            state = 1.0 if val > 0 else 0.0
            cur = cur + 1 if last_state == state else 1
            last_state = state
            vals.append(float(cur if state > 0 else -cur))
        streak.loc[grp.index] = vals
    return streak


def attach_trend_persistence_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_daily",
) -> pd.DataFrame:
    """将多空趋势持续性因子附加到现有因子长表。"""
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    date_col = "trade_date" if "trade_date" in out.columns else "signal_date"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    unique_dates = sorted(out[date_col].dropna().unique())
    all_frames: list[pd.DataFrame] = []

    for dt in unique_dates:
        trend = compute_trend_persistence_factors(db_path, signal_date=dt, table_name=table_name)
        if trend.empty:
            continue
        trend[date_col] = pd.Timestamp(dt)
        all_frames.append(trend)

    if not all_frames:
        for col in TREND_PERSISTENCE_FACTOR_COLS:
            out[col] = np.nan
        return out

    trend_all = pd.concat(all_frames, ignore_index=True)
    trend_all[date_col] = pd.to_datetime(trend_all[date_col]).dt.normalize()

    out = out.merge(
        trend_all[["symbol", date_col] + list(TREND_PERSISTENCE_FACTOR_COLS)],
        on=["symbol", date_col],
        how="left",
    )
    return out
