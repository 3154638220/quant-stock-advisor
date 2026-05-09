"""W1 Phase 2: 短期反转 & 量能异常因子。

基于 a_share_daily 表的日线价量数据构造截面因子。
所有因子均为 PIT-safe（市场数据在 trade_date 即可用）。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

# ── 因子列表常量 ─────────────────────────────────────────────────────────

REVERSAL_VOLUME_FACTOR_COLS: tuple[str, ...] = (
    "feature_reversal_st_reversal_1m",
    "feature_reversal_st_reversal_1w",
    "feature_reversal_volume_spike",
    "feature_reversal_turnover_anomaly",
    "feature_reversal_pv_divergence",
)

REVERSAL_VOLUME_FACTOR_DIRECTION: dict[str, int] = {
    "feature_reversal_st_reversal_1m": -1,     # 短期反转 = 做空信号
    "feature_reversal_st_reversal_1w": -1,     # 周度反转 = 做空信号
    "feature_reversal_volume_spike": -1,       # 量能异常放大 = 情绪过热
    "feature_reversal_turnover_anomaly": -1,   # 换手率异常 = 散户拥挤
    "feature_reversal_pv_divergence": -1,      # 量价背离 = 负向信号
}


def compute_reversal_volume_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    table_name: str = "a_share_daily",
    min_history_days: int = 120,
    min_coverage: float = 0.30,
) -> pd.DataFrame:
    """从日线数据计算短期反转 & 量能异常因子截面。

    Parameters
    ----------
    db_path : str
        DuckDB 数据库路径。
    signal_date : str or Timestamp
        信号日（月末），只使用 trade_date <= signal_date 的数据。
    table_name : str
        日线表名。
    min_history_days : int
        每只股票最少需要的交易日数。
    min_coverage : float
        最低截面覆盖率阈值。

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

        # 回溯 300 个交易日确保 250 日窗口有足够数据
        lookback_start = sd - pd.Timedelta(days=400)
        raw = con.execute(
            f"""
            SELECT symbol, trade_date, close, amount, turnover, volume
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

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    for c in ["close", "amount", "turnover", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["close", "trade_date"])
    raw = raw.sort_values(["symbol", "trade_date"])

    # 过滤历史不足的股票
    day_counts = raw.groupby("symbol").size()
    valid_symbols = day_counts[day_counts >= min_history_days].index
    raw = raw[raw["symbol"].isin(valid_symbols)]

    if raw.empty:
        return pd.DataFrame()

    # 日收益 & 市场等权收益
    raw["_ret_1d"] = raw.groupby("symbol", sort=False)["close"].pct_change()
    mkt_ret = raw.groupby("trade_date")["_ret_1d"].mean().rename("_mkt_ret")
    raw = raw.merge(mkt_ret, on="trade_date", how="left")
    raw["_excess_1d"] = raw["_ret_1d"] - raw["_mkt_ret"]

    # 重新创建 groupby，确保包含所有新列
    g = raw.groupby("symbol", sort=False)

    # ── 1. st_reversal_1m: 近 21 个交易日超额收益（cumsum） ──
    raw["feature_reversal_st_reversal_1m"] = g["_excess_1d"].transform(
        lambda s: s.rolling(21, min_periods=10).sum()
    )

    # ── 2. st_reversal_1w: 近 5 个交易日超额收益 ──
    raw["feature_reversal_st_reversal_1w"] = g["_excess_1d"].transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )

    # ── 3. volume_spike: 近 5 日均成交额 / 近 60 日均成交额 ──
    avg_amount_5d = g["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    avg_amount_60d = g["amount"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    raw["feature_reversal_volume_spike"] = avg_amount_5d / avg_amount_60d.replace(0, np.nan)

    # ── 4. turnover_anomaly: turnover 相对自身 60 日均值的 z-score ──
    turnover_mean_60d = g["turnover"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    turnover_std_60d = g["turnover"].transform(lambda s: s.rolling(60, min_periods=30).std())
    raw["feature_reversal_turnover_anomaly"] = (
        (raw["turnover"] - turnover_mean_60d) / turnover_std_60d.replace(0, np.nan)
    )

    # ── 5. pv_divergence: 近 20 日 日收益与量变化的滚动相关性 ──
    raw["_vol_chg"] = g["volume"].pct_change()
    pv_div = _compute_rolling_correlation(
        raw, "_ret_1d", "_vol_chg", window=20, min_periods=10
    )
    # 量价背离定义：价涨量缩或价跌量增 → 取 -corr（正相关=健康，负相关=背离）
    raw["feature_reversal_pv_divergence"] = -pv_div

    # ── 提取信号日截面 ──
    result = (
        raw[raw["trade_date"] == sd][
            ["symbol", "trade_date"] + list(REVERSAL_VOLUME_FACTOR_COLS)
        ]
        .drop_duplicates(["symbol"])
        .copy()
    )

    # 覆盖率过滤
    for col in REVERSAL_VOLUME_FACTOR_COLS:
        if col in result.columns:
            cov = result[col].notna().mean()
            if cov < min_coverage:
                result[col] = np.nan

    return result


def _compute_rolling_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    window: int = 20,
    min_periods: int = 10,
) -> pd.Series:
    """Compute per-symbol rolling Pearson correlation between two columns.

    Uses groupby-apply to avoid pandas rolling.corr alignment issues
    with non-unique indices after merge operations.
    """
    def _grp_corr(grp: pd.DataFrame) -> pd.Series:
        a = grp[col_a]
        b = grp[col_b]
        ma = a.rolling(window, min_periods=min_periods).mean()
        mb = b.rolling(window, min_periods=min_periods).mean()
        cov = ((a - ma) * (b - mb)).rolling(window, min_periods=min_periods).mean()
        std_a = a.rolling(window, min_periods=min_periods).std()
        std_b = b.rolling(window, min_periods=min_periods).std()
        out = cov / (std_a * std_b).replace(0, np.nan)
        out.index = grp.index
        return out

    result = df.groupby("symbol", sort=False).apply(_grp_corr, include_groups=False)
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    return result.droplevel(0)


def attach_reversal_volume_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_daily",
) -> pd.DataFrame:
    """将短期反转 & 量能异常因子附加到现有因子长表。

    对每个 signal_date，逐月计算并合并。
    """
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    date_col = "trade_date" if "trade_date" in out.columns else "signal_date"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    unique_dates = sorted(out[date_col].dropna().unique())
    all_frames: list[pd.DataFrame] = []

    for dt in unique_dates:
        rv = compute_reversal_volume_factors(db_path, signal_date=dt, table_name=table_name)
        if rv.empty:
            continue
        rv[date_col] = pd.Timestamp(dt)
        all_frames.append(rv)

    if not all_frames:
        for col in REVERSAL_VOLUME_FACTOR_COLS:
            out[col] = np.nan
        return out

    rv_all = pd.concat(all_frames, ignore_index=True)
    rv_all[date_col] = pd.to_datetime(rv_all[date_col]).dt.normalize()

    out = out.merge(
        rv_all[["symbol", date_col] + list(REVERSAL_VOLUME_FACTOR_COLS)],
        on=["symbol", date_col],
        how="left",
    )
    return out
