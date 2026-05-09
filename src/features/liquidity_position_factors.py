"""W1 Phase 3: 流动性 & 价格位置扩展因子。

基于 a_share_daily 表的日线价量数据构造截面因子。
所有因子均为 PIT-safe（市场数据在 trade_date 即可用）。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

# ── 因子列表常量 ─────────────────────────────────────────────────────────

LIQUIDITY_POSITION_FACTOR_COLS: tuple[str, ...] = (
    "feature_liquidity_amihud",
    "feature_liquidity_high52w_ratio",
    "feature_liquidity_low52w_ratio",
    "feature_liquidity_price_range_width",
)

LIQUIDITY_POSITION_FACTOR_DIRECTION: dict[str, int] = {
    "feature_liquidity_amihud": -1,            # 高非流动性 = 负向（流动性差）
    "feature_liquidity_high52w_ratio": 1,      # 接近高点 = 动量延续
    "feature_liquidity_low52w_ratio": -1,      # 接近低点 = 负向信号
    "feature_liquidity_price_range_width": -1, # 宽幅震荡 = 不确定性高
}


def compute_liquidity_position_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    table_name: str = "a_share_daily",
    min_history_days: int = 120,
    min_coverage: float = 0.30,
) -> pd.DataFrame:
    """从日线数据计算流动性 & 价格位置因子截面。

    Parameters
    ----------
    db_path : str
        DuckDB 数据库路径。
    signal_date : str or Timestamp
        信号日（月末）。
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

        lookback_start = sd - pd.Timedelta(days=400)
        raw = con.execute(
            f"""
            SELECT symbol, trade_date, close, high, low, amount, volume
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
    for c in ["close", "high", "low", "amount", "volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["close", "trade_date"])
    raw = raw.sort_values(["symbol", "trade_date"])

    day_counts = raw.groupby("symbol").size()
    valid_symbols = day_counts[day_counts >= min_history_days].index
    raw = raw[raw["symbol"].isin(valid_symbols)]

    if raw.empty:
        return pd.DataFrame()

    g = raw.groupby("symbol", sort=False)
    window_250 = 250
    window_20 = 20

    # ── 1. amihud_illiquidity: 日均 |ret| / amount，过去 20 日均值 × 10^6 ──
    raw["_ret_1d"] = g["close"].pct_change().abs()
    raw["_daily_illiq"] = raw["_ret_1d"] / raw["amount"].replace(0, np.nan)
    raw["feature_liquidity_amihud"] = (
        g["_daily_illiq"]
        .transform(lambda s: s.rolling(window_20, min_periods=10).mean())
        * 1e6
    )

    # ── 2. high52w_ratio: 当前收盘价 / 250 日最高价 ──
    roll_high_250 = g["high"].transform(
        lambda s: s.rolling(window_250, min_periods=60).max()
    )
    raw["feature_liquidity_high52w_ratio"] = raw["close"] / roll_high_250.replace(0, np.nan)

    # ── 3. low52w_ratio: 当前收盘价 / 250 日最低价 ──
    roll_low_250 = g["low"].transform(
        lambda s: s.rolling(window_250, min_periods=60).min()
    )
    raw["feature_liquidity_low52w_ratio"] = raw["close"] / roll_low_250.replace(0, np.nan)

    # ── 4. price_range_width: (250日最高 - 250日最低) / 250日均价 ──
    roll_mean_250 = g["close"].transform(
        lambda s: s.rolling(window_250, min_periods=60).mean()
    )
    raw["feature_liquidity_price_range_width"] = (
        (roll_high_250 - roll_low_250) / roll_mean_250.replace(0, np.nan)
    )

    # ── 提取信号日截面 ──
    result = (
        raw[raw["trade_date"] == sd][
            ["symbol", "trade_date"] + list(LIQUIDITY_POSITION_FACTOR_COLS)
        ]
        .drop_duplicates(["symbol"])
        .copy()
    )

    for col in LIQUIDITY_POSITION_FACTOR_COLS:
        if col in result.columns:
            cov = result[col].notna().mean()
            if cov < min_coverage:
                result[col] = np.nan

    return result


def attach_liquidity_position_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_daily",
) -> pd.DataFrame:
    """将流动性 & 价格位置扩展因子附加到现有因子长表。"""
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    date_col = "trade_date" if "trade_date" in out.columns else "signal_date"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    unique_dates = sorted(out[date_col].dropna().unique())
    all_frames: list[pd.DataFrame] = []

    for dt in unique_dates:
        lp = compute_liquidity_position_factors(db_path, signal_date=dt, table_name=table_name)
        if lp.empty:
            continue
        lp[date_col] = pd.Timestamp(dt)
        all_frames.append(lp)

    if not all_frames:
        for col in LIQUIDITY_POSITION_FACTOR_COLS:
            out[col] = np.nan
        return out

    lp_all = pd.concat(all_frames, ignore_index=True)
    lp_all[date_col] = pd.to_datetime(lp_all[date_col]).dt.normalize()

    out = out.merge(
        lp_all[["symbol", date_col] + list(LIQUIDITY_POSITION_FACTOR_COLS)],
        on=["symbol", date_col],
        how="left",
    )
    return out
