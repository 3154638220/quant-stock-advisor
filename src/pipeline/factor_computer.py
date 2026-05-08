"""技术因子计算模块。

从回测执行管线中抽取的纯算法组件，只做计算不放编排逻辑：
- 技术指标计算（KDJ、ATR、RSI、动量等）
- z-score 辅助函数
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── 因子列表常量 ─────────────────────────────────────────────────────────

PREPARED_FACTORS_SCHEMA_VERSION = 2

PREPARED_FACTORS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "momentum",
    "momentum_12_1",
    "rsi",
    "realized_vol",
    "short_reversal",
    "turnover_roll_mean",
    "turnover_rel_pct_252",
    "vol_ret_corr",
    "vol_to_turnover",
    "volume_skew_log",
    "log_market_cap",
    "bias_short",
    "bias_long",
    "max_single_day_drop",
    "recent_return",
    "price_position",
    "weekly_kdj_k",
    "weekly_kdj_d",
    "weekly_kdj_j",
    "weekly_kdj_oversold",
    "weekly_kdj_oversold_depth",
    "weekly_kdj_rebound",
    "intraday_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_open_return",
    "overnight_gap",
    "tail_strength",
    "volume_price_trend",
    "intraday_range_skew",
    "limit_move_hits_5d",
    "proxy_main_inflow_pct_5d",
    "proxy_main_inflow_pct_10d",
    "proxy_main_inflow_pct_20d",
    "proxy_up_down_ratio_5d",
    "announcement_date",
    "pe_ttm",
    "pb",
    "ev_ebitda",
    "roe_ttm",
    "net_profit_yoy",
    "gross_margin_change",
    "debt_to_assets_change",
    "ocf_to_net_profit",
    "ocf_to_asset",
    "gross_margin_delta",
    "asset_turnover",
    "net_margin_stability",
    "northbound_net_inflow",
    "margin_buy_ratio",
    "main_inflow_z_5d",
    "super_inflow_z_5d",
    "flow_divergence_5d",
    "main_inflow_z_10d",
    "super_inflow_z_10d",
    "flow_divergence_10d",
    "main_inflow_z_20d",
    "super_inflow_z_20d",
    "flow_divergence_20d",
    "main_inflow_streak",
    "holder_count",
    "holder_change",
    "holder_count_log",
    "holder_count_change_pct",
    "holder_change_rate",
    "holder_change_rate_z",
    "holder_count_log_z",
    "holder_concentration_proxy",
    "_universe_eligible",
)


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def _zscore_clip(s: pd.Series, clip: float = 3.0) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return ((s - mu) / sd).clip(-clip, clip)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _weekly_kdj_completed_from_daily(
    trade_date: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    n: int = 9,
    initial_k: float = 50.0,
    initial_d: float = 50.0,
) -> pd.DataFrame:
    """按最近已完成周线对齐回日频，周内保持上一根已完成周线值。"""
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(trade_date, errors="coerce"),
            "close": pd.to_numeric(close, errors="coerce"),
            "high": pd.to_numeric(high, errors="coerce"),
            "low": pd.to_numeric(low, errors="coerce"),
        }
    ).sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["weekly_kdj_k", "weekly_kdj_d", "weekly_kdj_j",
                                      "weekly_kdj_oversold", "weekly_kdj_oversold_depth",
                                      "weekly_kdj_rebound"])

    week_key = df["trade_date"].dt.to_period("W-FRI")
    weekly = (
        df.assign(week_key=week_key)
        .groupby("week_key", sort=True)
        .agg(
            weekly_close=("close", "last"),
            weekly_high=("high", "max"),
            weekly_low=("low", "min"),
            week_last_trade_date=("trade_date", "max"),
        )
        .reset_index(drop=True)
    )
    roll_high = weekly["weekly_high"].rolling(n, min_periods=1).max()
    roll_low = weekly["weekly_low"].rolling(n, min_periods=1).min()
    denom = (roll_high - roll_low).replace(0, np.nan)
    rsv = (weekly["weekly_close"] - roll_low) / denom * 100.0
    rsv = rsv.fillna(50.0)

    k_vals: list[float] = []
    d_vals: list[float] = []
    j_vals: list[float] = []
    prev_k = float(initial_k)
    prev_d = float(initial_d)
    for value in rsv.to_numpy(dtype=np.float64):
        cur_k = prev_k * 2.0 / 3.0 + value / 3.0
        cur_d = prev_d * 2.0 / 3.0 + cur_k / 3.0
        cur_j = 3.0 * cur_k - 2.0 * cur_d
        k_vals.append(cur_k)
        d_vals.append(cur_d)
        j_vals.append(cur_j)
        prev_k = cur_k
        prev_d = cur_d
    weekly["weekly_kdj_k"] = k_vals
    weekly["weekly_kdj_d"] = d_vals
    weekly["weekly_kdj_j"] = j_vals
    weekly["weekly_kdj_oversold"] = (weekly["weekly_kdj_j"] <= -5.0).astype(float)
    weekly["weekly_kdj_oversold_depth"] = np.maximum(0.0, -5.0 - weekly["weekly_kdj_j"])
    weekly["weekly_kdj_rebound"] = (
        (weekly["weekly_kdj_j"].shift(1) <= -5.0)
        & (weekly["weekly_kdj_j"] > weekly["weekly_kdj_j"].shift(1))
    ).astype(float)
    weekly.loc[weekly.index[0], "weekly_kdj_rebound"] = np.nan

    merged = df[["trade_date"]].merge(
        weekly[["week_last_trade_date", "weekly_kdj_k", "weekly_kdj_d", "weekly_kdj_j",
                "weekly_kdj_oversold", "weekly_kdj_oversold_depth", "weekly_kdj_rebound"]],
        left_on="trade_date", right_on="week_last_trade_date", how="left",
    )
    weekly_cols = ["weekly_kdj_k", "weekly_kdj_d", "weekly_kdj_j",
                   "weekly_kdj_oversold", "weekly_kdj_oversold_depth", "weekly_kdj_rebound"]
    merged[weekly_cols] = merged[weekly_cols].ffill()
    intraweek_mask = merged["trade_date"] != merged["week_last_trade_date"]
    merged.loc[intraweek_mask, weekly_cols] = merged.loc[intraweek_mask, weekly_cols].shift(1)
    return merged[weekly_cols]


# ── 因子计算主函数 ───────────────────────────────────────────────────────

def compute_factors(daily_df: pd.DataFrame, min_hist_days: int = 130) -> pd.DataFrame:
    """从日线 OHLCV 计算全量技术因子。"""

    out = []
    for sym, g in daily_df.groupby("symbol", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        c = g["close"]
        o = g["open"]
        h = g["high"]
        lo = g["low"]
        v = g["volume"]
        t = g["turnover"]
        pcg = g["pct_chg"] / 100.0
        if c.notna().sum() < min_hist_days:
            continue

        momentum = c / c.shift(10) - 1.0
        momentum_12_1 = c.shift(21) / c.shift(252) - 1.0

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_l = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rsi = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))

        daily_ret = c.pct_change()
        realized_vol = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
        short_reversal = -(c / c.shift(5) - 1.0)
        atr = _wilder_atr(h, lo, c, period=14)
        turnover_roll_mean = t.rolling(20, min_periods=10).mean()
        turnover_5d_mean = t.rolling(5, min_periods=3).mean()
        turnover_rel_pct_252 = turnover_5d_mean.rolling(252, min_periods=40).rank(pct=True)
        vol_ret_corr = v.rolling(20, min_periods=10).corr(daily_ret).fillna(0.0)
        vol_to_turnover = np.log1p(v / t.replace(0, np.nan)).fillna(0.0)
        volume_skew_log = np.log(v.clip(lower=0) + 1.0).rolling(20, min_periods=10).skew()
        log_market_cap = np.log((c * v / (t / 100.0 + 1e-8)).clip(lower=1.0))

        ma20 = c.rolling(20, min_periods=10).mean()
        ma60 = c.rolling(60, min_periods=30).mean()
        bias_short = (c - ma20) / ma20.replace(0, np.nan)
        bias_long = (c - ma60) / ma60.replace(0, np.nan)
        max_single_day_drop = pcg.rolling(20, min_periods=5).min().abs()
        recent_return = c / c.shift(3) - 1.0

        hi250 = h.rolling(250, min_periods=60).max()
        lo250 = lo.rolling(250, min_periods=60).min()
        price_position = ((c - lo250) / (hi250 - lo250).replace(0, np.nan)).clip(0, 1)
        weekly_kdj = _weekly_kdj_completed_from_daily(g["trade_date"], c, h, lo)

        intraday_range = (h - lo) / c.replace(0, np.nan)
        candle_range = (h - lo).replace(0, np.nan)
        upper_shadow_ratio = (h - np.maximum(c, o)) / candle_range
        upper_shadow_ratio = upper_shadow_ratio.fillna(0).clip(0, 1)
        lower_shadow_ratio = (np.minimum(c, o) - lo) / candle_range
        lower_shadow_ratio = lower_shadow_ratio.fillna(0).clip(0, 1)
        close_open_return = (c - o) / o.replace(0, np.nan)
        overnight_gap = o / c.shift(1).replace(0, np.nan) - 1.0
        tail_strength = close_open_return.rolling(10, min_periods=5).mean()
        volume_price_trend = (np.log(v.clip(lower=0) + 1.0) * daily_ret).rolling(20, min_periods=10).sum()
        intraday_range_skew = intraday_range.rolling(20, min_periods=10).skew()

        if sym.startswith(("300", "688")):
            lim = 0.20
        elif sym.startswith(("8", "4")):
            lim = 0.30
        else:
            lim = 0.10
        limit_move_hits_5d = (pcg.abs() >= (lim - 0.005)).astype(float).rolling(5, min_periods=1).sum()

        amt = g.get("amount", v * c)
        amt = pd.to_numeric(amt, errors="coerce") if "amount" in g.columns else (v * c)

        close_position = (c - lo) / (h - lo).replace(0, np.nan)
        close_position = close_position.fillna(0.5).clip(0, 1)

        buy_pressure = amt * close_position
        sell_pressure = amt * (1 - close_position)
        net_buy_pressure = buy_pressure - sell_pressure
        total_pressure = buy_pressure + sell_pressure
        total_pressure = total_pressure.replace(0, np.nan)
        net_buy_pressure_pct = (net_buy_pressure / total_pressure * 100).fillna(0)

        net_buy_pressure_pct_5d = net_buy_pressure_pct.rolling(5, min_periods=3).mean()
        net_buy_pressure_pct_10d = net_buy_pressure_pct.rolling(10, min_periods=5).mean()
        net_buy_pressure_pct_20d = net_buy_pressure_pct.rolling(20, min_periods=10).mean()

        up_days = (pcg > 0).astype(float)
        down_days = (pcg < 0).astype(float)
        up_amt = (amt * up_days).rolling(5, min_periods=3).sum()
        down_amt = (amt * down_days).rolling(5, min_periods=3).sum()
        total_amt_5d = amt.rolling(5, min_periods=3).sum()
        total_amt_5d = total_amt_5d.replace(0, np.nan)
        up_down_ratio_5d = ((up_amt - down_amt) / total_amt_5d * 100).fillna(0)

        out.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "trade_date": g["trade_date"].values,
                    "momentum": momentum.values,
                    "momentum_12_1": momentum_12_1.values,
                    "rsi": rsi.values,
                    "atr": atr.values,
                    "realized_vol": realized_vol.values,
                    "short_reversal": short_reversal.values,
                    "turnover_roll_mean": turnover_roll_mean.values,
                    "turnover_rel_pct_252": turnover_rel_pct_252.values,
                    "vol_ret_corr": vol_ret_corr.values,
                    "vol_to_turnover": vol_to_turnover.values,
                    "volume_skew_log": volume_skew_log.values,
                    "log_market_cap": log_market_cap.values,
                    "bias_short": bias_short.values,
                    "bias_long": bias_long.values,
                    "max_single_day_drop": max_single_day_drop.values,
                    "recent_return": recent_return.values,
                    "price_position": price_position.values,
                    "weekly_kdj_k": weekly_kdj["weekly_kdj_k"].to_numpy(),
                    "weekly_kdj_d": weekly_kdj["weekly_kdj_d"].to_numpy(),
                    "weekly_kdj_j": weekly_kdj["weekly_kdj_j"].to_numpy(),
                    "weekly_kdj_oversold": weekly_kdj["weekly_kdj_oversold"].to_numpy(),
                    "weekly_kdj_oversold_depth": weekly_kdj["weekly_kdj_oversold_depth"].to_numpy(),
                    "weekly_kdj_rebound": weekly_kdj["weekly_kdj_rebound"].to_numpy(),
                    "intraday_range": intraday_range.values,
                    "upper_shadow_ratio": upper_shadow_ratio.values,
                    "lower_shadow_ratio": lower_shadow_ratio.values,
                    "close_open_return": close_open_return.values,
                    "overnight_gap": overnight_gap.values,
                    "tail_strength": tail_strength.values,
                    "volume_price_trend": volume_price_trend.values,
                    "intraday_range_skew": intraday_range_skew.values,
                    "limit_move_hits_5d": limit_move_hits_5d.values,
                    "proxy_main_inflow_pct_5d": net_buy_pressure_pct_5d.values,
                    "proxy_main_inflow_pct_10d": net_buy_pressure_pct_10d.values,
                    "proxy_main_inflow_pct_20d": net_buy_pressure_pct_20d.values,
                    "proxy_up_down_ratio_5d": up_down_ratio_5d.values,
                }
            )
        )
    if not out:
        raise RuntimeError("因子计算为空")
    return pd.concat(out, ignore_index=True)
