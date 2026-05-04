"""回测执行核心管线。

从 scripts/run_backtest_eval.py 提取核心算法逻辑：
- 因子计算
- 截面打分
- Top-K 权重构建
- Regime 动态调权
- 因子缓存管理
- 数据加载

不放 CLI 参数解析与格式化输出。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import duckdb
import numpy as np
import pandas as pd

from src.features.fund_flow_factors import attach_fund_flow
from src.features.fundamental_factors import pit_safe_fundamental_rows, preprocess_fundamental_cross_section
from src.features.ic_monitor import ICMonitor
from src.features.shareholder_factors import attach_shareholder_factors
from src.backtest.engine import build_open_to_open_returns as _build_open_to_open_returns
from src.market.regime import (
    MARKET_EW_PROXY,
    classify_regime,
    get_regime_weights,
    regime_config_from_mapping,
)
from src.models.rank_score import sort_key_for_dataframe
from src.portfolio.covariance import mean_cov_returns_from_daily_long
from src.portfolio.weights import build_portfolio_weights
from src.research.gates import apply_prefilter, attach_universe_filter

# ── 因子列表 ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


# ── 日线加载 ─────────────────────────────────────────────────────────────

def load_daily_from_duckdb(
    db_path: str, start: str, end: str, lookback_days: int
) -> pd.DataFrame:
    start_dt = (pd.Timestamp(start) - pd.offsets.BDay(lookback_days)).strftime("%Y-%m-%d")
    con = duckdb.connect(db_path, read_only=True)
    sql = f"""
        SELECT symbol, trade_date, open, close, high, low, volume, amount, turnover, pct_chg
        FROM a_share_daily
        WHERE trade_date >= '{start_dt}' AND trade_date <= '{end}'
        ORDER BY symbol, trade_date
    """
    df = con.execute(sql).df()
    con.close()
    if df.empty:
        raise RuntimeError("DuckDB 查询为空，请检查数据范围")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for c in ("open", "close", "high", "low", "volume", "turnover", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 因子计算 ─────────────────────────────────────────────────────────────

def _zscore_clip(s: pd.Series, clip: float = 3.0) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return ((s - mu) / sd).clip(-clip, clip)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.Series(np.nan, index=close.index, dtype=np.float64)
    prev_close = close.shift(1)
    tr.iloc[1:] = np.maximum.reduce(
        [
            (high.iloc[1:] - low.iloc[1:]).to_numpy(dtype=np.float64),
            (high.iloc[1:] - prev_close.iloc[1:]).abs().to_numpy(dtype=np.float64),
            (low.iloc[1:] - prev_close.iloc[1:]).abs().to_numpy(dtype=np.float64),
        ]
    )
    atr = pd.Series(np.nan, index=close.index, dtype=np.float64)
    if len(close) <= period:
        return atr
    init = tr.iloc[1 : period + 1].mean()
    atr.iloc[period] = init
    for i in range(period + 1, len(close)):
        prev_atr = atr.iloc[i - 1]
        cur_tr = tr.iloc[i]
        if np.isfinite(prev_atr) and np.isfinite(cur_tr):
            atr.iloc[i] = (prev_atr * (period - 1) + cur_tr) / period
    return atr


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


# ── PIT 基本面 attach ────────────────────────────────────────────────────

def _attach_pit_fundamentals(factors: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """按公告日 merge_asof，将基本面快照对齐到每个交易日（PIT）。"""
    out = factors.copy(deep=False)
    con = duckdb.connect(db_path, read_only=True)
    want_cols = [
        "symbol", "report_period", "announcement_date",
        "pe_ttm", "pb", "ev_ebitda", "roe_ttm", "net_profit_yoy",
        "gross_margin_change", "debt_to_assets_change", "ocf_to_net_profit",
        "ocf_to_asset", "gross_margin_delta", "asset_turnover",
        "net_margin_stability", "northbound_net_inflow", "margin_buy_ratio", "source",
    ]
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'a_share_fundamental'"
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return out
        info = con.execute("PRAGMA table_info('a_share_fundamental')").fetchall()
        have = {str(r[1]) for r in info}
        sel = [c for c in want_cols if c in have]
        fund = con.execute(f"SELECT {', '.join(sel)} FROM a_share_fundamental").df()
        for c in want_cols:
            if c not in fund.columns:
                fund[c] = "" if c == "source" else np.nan
    finally:
        con.close()
    if fund.empty:
        return out

    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["trade_date"])
    fund["symbol"] = fund["symbol"].astype(str).str.zfill(6)
    fund["announcement_date"] = (
        pd.to_datetime(fund["announcement_date"], errors="coerce").dt.normalize()
    )
    fund["report_period"] = pd.to_datetime(fund["report_period"], errors="coerce")
    fund = fund.dropna(subset=["announcement_date"])
    fund = fund[pit_safe_fundamental_rows(fund)].copy()
    if fund.empty:
        return out
    fund = fund.sort_values(["symbol", "announcement_date", "report_period"], na_position="last", kind="mergesort")
    fund = fund.drop_duplicates(["symbol", "announcement_date"], keep="last")
    fund = fund.drop(columns=["report_period", "source"], errors="ignore")

    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    fund = fund.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
    chunked: list[pd.DataFrame] = []
    for _, chunk in out.groupby(pd.Grouper(key="trade_date", freq="31D"), sort=True):
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
        chunk_end = pd.Timestamp(chunk["trade_date"].max())
        chunk_symbols = chunk["symbol"].astype(str).unique().tolist()
        fund_chunk = fund[
            (fund["announcement_date"] <= chunk_end)
            & fund["symbol"].astype(str).isin(chunk_symbols)
        ].copy()
        if fund_chunk.empty:
            merged = chunk.copy()
            for c in want_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
        else:
            fund_chunk = fund_chunk.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
            merged = pd.merge_asof(
                chunk, fund_chunk,
                left_on="trade_date", right_on="announcement_date",
                by="symbol", direction="backward", allow_exact_matches=True,
            )
        merged = preprocess_fundamental_cross_section(
            merged, date_col="trade_date", size_col="log_market_cap", neutralize=True,
        )
        chunked.append(merged)
    if not chunked:
        return out
    return pd.concat(chunked, ignore_index=True)


# ── 因子缓存 ─────────────────────────────────────────────────────────────

def _json_sanitize(obj: Any) -> Any:
    from src.reporting.markdown_report import json_sanitize as _js
    return _js(obj)


def _factor_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _prepared_factors_cache_expected_meta(
    *,
    start_date: str,
    end_date: str,
    lookback_days: int,
    min_hist_days: int,
    db_path: str,
    results_dir: str,
    universe_filter_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "lookback_days": int(lookback_days),
        "min_hist_days": int(min_hist_days),
        "db_path": str(db_path),
        "results_dir": str(results_dir),
        "universe_filter_cfg": _json_sanitize(universe_filter_cfg),
        "cache_format_version": 2,
        "prepared_factors_schema_version": PREPARED_FACTORS_SCHEMA_VERSION,
        "required_columns": list(PREPARED_FACTORS_REQUIRED_COLUMNS),
    }


def load_prepared_factors_cache(cache_path: Path, expected_meta: dict[str, Any]) -> pd.DataFrame | None:
    meta_path = _factor_cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if actual_meta != expected_meta:
        return None
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return None
    required_columns = {str(col) for col in expected_meta.get("required_columns", []) if str(col).strip()}
    if required_columns:
        cached_columns = {str(col) for col in cached.columns}
        if not required_columns.issubset(cached_columns):
            return None
    if "trade_date" in cached.columns:
        cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce")
    if "symbol" in cached.columns:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
    return cached


def write_prepared_factors_cache(cache_path: Path, factors: pd.DataFrame, meta: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(cache_path, index=False)
    _factor_cache_meta_path(cache_path).write_text(
        json.dumps(_json_sanitize(meta), ensure_ascii=False, indent=2), encoding="utf-8",
    )


# ── 因子准备管线 ─────────────────────────────────────────────────────────

def prepare_factors_for_backtest(
    daily_df: pd.DataFrame,
    *,
    min_hist_days: int,
    db_path: str,
    results_dir: Path,
    universe_filter_cfg: dict[str, Any],
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    cache_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, bool]:
    expected_meta = dict(cache_meta or {})
    if cache_path is not None and not refresh_cache:
        cached = load_prepared_factors_cache(cache_path, expected_meta)
        if cached is not None:
            return cached, True

    factors = compute_factors(daily_df, min_hist_days=min_hist_days)
    factors = _attach_pit_fundamentals(factors, db_path)
    factors = attach_fund_flow(factors, db_path)
    factors = attach_shareholder_factors(factors, db_path)
    factors = attach_universe_filter(
        factors, daily_df,
        enabled=bool(universe_filter_cfg.get("enabled", False)),
        min_amount_20d=float(universe_filter_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(universe_filter_cfg.get("require_roe_ttm_positive", True)),
    )
    if cache_path is not None:
        write_prepared_factors_cache(cache_path, factors, expected_meta)
    return factors, False


# ── 截面打分 ─────────────────────────────────────────────────────────────

def build_score(
    factors: pd.DataFrame,
    weights: Dict[str, float],
    *,
    weights_by_date: Dict[pd.Timestamp, Dict[str, float]] | None = None,
    universe_eligible_col: str | None = "_universe_eligible",
    sort_by: str = "composite_extended",
    tree_bundle_dir: str | None = None,
    tree_raw_features: Iterable[str] | None = None,
    tree_rsi_mode: str = "level",
) -> pd.DataFrame:
    mode = str(sort_by).lower().strip()
    fac_cols = [c for c in weights.keys() if c in factors.columns]
    rows = []
    for dt, g in factors.groupby("trade_date"):
        if universe_eligible_col and universe_eligible_col in g.columns:
            g = g.loc[g[universe_eligible_col].to_numpy(dtype=bool)]
        if mode == "xgboost":
            raw = [str(c) for c in (tree_raw_features or [])]
            if not raw:
                raise ValueError("sort_by=xgboost 需要 tree_raw_features")
            g = g.dropna(subset=raw, how="all").copy()
        else:
            g = g.dropna(subset=fac_cols, how="all").copy()
        if len(g) < 10:
            continue
        if mode == "xgboost":
            ranked = sort_key_for_dataframe(
                g, sort_by="xgboost",
                tree_bundle_dir=tree_bundle_dir,
                tree_raw_features=list(tree_raw_features or []),
                tree_rsi_mode=tree_rsi_mode,
            )
            rows.append(pd.DataFrame({
                "symbol": ranked["symbol"].values,
                "trade_date": dt,
                "score": pd.to_numeric(ranked["tree_score"], errors="coerce").to_numpy(dtype=np.float64),
            }))
            continue
        effective_weights = weights_by_date.get(pd.Timestamp(dt), weights) if weights_by_date else weights
        active_cols = []
        for fc in fac_cols:
            col = pd.to_numeric(g[fc], errors="coerce")
            m = col.notna() & np.isfinite(col)
            if m.sum() >= 5 and abs(float(effective_weights.get(fc, 0.0))) > 1e-15:
                active_cols.append(fc)
        if not active_cols:
            continue
        abs_sum = float(sum(abs(float(effective_weights.get(fc, 0.0))) for fc in active_cols))
        if abs_sum <= 1e-15:
            continue
        score = pd.Series(0.0, index=g.index)
        for fc in fac_cols:
            if fc not in active_cols:
                continue
            col = pd.to_numeric(g[fc], errors="coerce")
            m = col.notna() & np.isfinite(col)
            if m.sum() < 5:
                continue
            w = float(effective_weights.get(fc, 0.0)) / abs_sum
            score[m] += _zscore_clip(col[m]) * w
        rows.append(pd.DataFrame({"symbol": g["symbol"].values, "trade_date": dt, "score": score.values}))
    if not rows:
        raise RuntimeError("得分构建为空")
    return pd.concat(rows, ignore_index=True).dropna(subset=["score"])


def load_industry_map(industry_map_csv: str) -> Dict[str, str]:
    p = Path(industry_map_csv)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    tab = pd.read_csv(
        p,
        encoding="utf-8-sig",
        converters={
            "symbol": lambda v: str(v).strip(),
            "代码": lambda v: str(v).strip(),
            "industry": lambda v: str(v).strip(),
            "行业": lambda v: str(v).strip(),
        },
    )
    sym_col = "symbol" if "symbol" in tab.columns else ("代码" if "代码" in tab.columns else None)
    ind_col = "industry" if "industry" in tab.columns else ("行业" if "行业" in tab.columns else None)
    if sym_col is None or ind_col is None:
        return {}
    tab = tab[[sym_col, ind_col]].copy()
    tab[sym_col] = tab[sym_col].astype(str).str.zfill(6)
    tab[ind_col] = tab[ind_col].astype(str).str.strip()
    tab = tab[(tab[sym_col].str.len() == 6) & (tab[ind_col] != "")]
    tab = tab.drop_duplicates(subset=[sym_col], keep="last")
    return dict(zip(tab[sym_col], tab[ind_col]))


# ── Regime 动态调权 ──────────────────────────────────────────────────────

def build_market_ew_benchmark(
    daily_df: pd.DataFrame, start: str, end: str, min_days: int = 500
) -> pd.Series:
    """构建全市场等权日收益基准（close-to-close）。"""
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["close"] > 0)
    ].copy()
    sym_cnt = df.groupby("symbol")["trade_date"].count()
    good = sym_cnt[sym_cnt >= min_days].index
    if len(good) == 0:
        good = sym_cnt.index
    df = df[df["symbol"].isin(good)].sort_values(["symbol", "trade_date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    cs = df.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean()
    cs.index = pd.to_datetime(cs.index)
    return cs


def build_regime_weight_overrides(
    factors: pd.DataFrame,
    daily_df: pd.DataFrame,
    base_weights: Dict[str, float],
    benchmark_symbol: str,
    regime_cfg_raw: dict,
    *,
    market_ew_min_days: int = 130,
) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], pd.DataFrame]:
    cfg = regime_config_from_mapping(regime_cfg_raw)
    sym_key = str(benchmark_symbol).strip()
    if sym_key.lower() == MARKET_EW_PROXY.lower():
        fac_start = pd.to_datetime(factors["trade_date"]).min()
        fac_end = pd.to_datetime(factors["trade_date"]).max()
        bench_s = build_market_ew_benchmark(
            daily_df, str(fac_start.date()), str(fac_end.date()),
            min_days=int(market_ew_min_days),
        )
        if bench_s.empty:
            return {}, pd.DataFrame(columns=["trade_date", "regime", "short_return", "vol_ann"])
    else:
        bench = daily_df[daily_df["symbol"].astype(str).str.zfill(6) == sym_key.zfill(6)].copy()
        if bench.empty:
            return {}, pd.DataFrame(columns=["trade_date", "regime", "short_return", "vol_ann"])
        bench = bench.sort_values("trade_date")
        bench["ret"] = pd.to_numeric(bench["close"], errors="coerce").pct_change()
        bench = bench.dropna(subset=["ret"])
        bench_s = pd.Series(
            bench["ret"].to_numpy(dtype=np.float64), index=pd.to_datetime(bench["trade_date"])
        ).sort_index()

    overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for dt in sorted(pd.to_datetime(factors["trade_date"]).unique()):
        regime, result = classify_regime(bench_s, dt, cfg=cfg)
        overrides[pd.Timestamp(dt)] = get_regime_weights(base_weights, regime, cfg=cfg, regime_result=result)
        rows.append({
            "trade_date": pd.Timestamp(dt),
            "regime": regime,
            "short_return": float(result.short_return),
            "vol_ann": float(result.realized_vol_ann),
        })
    return overrides, pd.DataFrame(rows)


# ── Top-K 权重构建 ───────────────────────────────────────────────────────

def _rebalance_dates(all_dates: Iterable[pd.Timestamp], rule: str) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(list(all_dates)))
    if not dates:
        return []
    arr = np.array(dates, dtype="datetime64[ns]")
    freq = rule
    if str(rule).upper() == "M":
        freq = "ME"
    elif str(rule).upper() == "BM":
        freq = "2BME"
    anchors = pd.date_range(dates[0], dates[-1], freq=freq)
    out: list[pd.Timestamp] = []
    for a in anchors:
        pos = np.searchsorted(arr, np.datetime64(a), side="right") - 1
        if pos >= 0:
            out.append(pd.Timestamp(arr[pos]))
    return sorted(set(out))


def _pick_topk_with_industry_cap(
    day_df: pd.DataFrame,
    *,
    top_k: int,
    industry_map: Dict[str, str] | None,
    industry_cap_count: int | None,
) -> pd.DataFrame:
    ranked = day_df.nlargest(top_k * 5, "score")[["symbol", "score"]].copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.zfill(6)
    if not industry_map or not industry_cap_count or industry_cap_count <= 0:
        return ranked.nlargest(top_k, "score")

    cap = int(industry_cap_count)
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for _, r in ranked.sort_values("score", ascending=False).iterrows():
        sym = str(r["symbol"]).zfill(6)
        ind = industry_map.get(sym, "_UNKNOWN_")
        cur = counts.get(ind, 0)
        if ind == "_UNKNOWN_" or cur < cap:
            picked.append({"symbol": sym, "score": float(r["score"])})
            counts[ind] = cur + 1
        if len(picked) >= top_k:
            break
    return pd.DataFrame(picked).nlargest(top_k, "score")


def _select_topk_with_holding_buffer(
    day_df: pd.DataFrame,
    *,
    top_k: int,
    entry_top_k: int,
    hold_buffer_top_k: int,
    prev_holdings: set[str],
    industry_map: Dict[str, str] | None = None,
    industry_cap_count: int | None = None,
) -> pd.DataFrame:
    hold = day_df[day_df["symbol"].astype(str).isin(prev_holdings)].copy()
    non_hold = day_df[~day_df["symbol"].astype(str).isin(prev_holdings)].copy()

    result = _pick_topk_with_industry_cap(
        hold, top_k=min(hold_buffer_top_k, len(hold)),
        industry_map=industry_map, industry_cap_count=industry_cap_count,
    )
    remain = top_k - len(result)
    if remain > 0 and not non_hold.empty:
        non_picked = _pick_topk_with_industry_cap(
            non_hold, top_k=remain,
            industry_map=industry_map, industry_cap_count=industry_cap_count,
        )
        result = pd.concat([result, non_picked], ignore_index=True)
    return result.nlargest(top_k, "score")


def build_topk_weights(
    score_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    top_k: int,
    rebalance_rule: str,
    prefilter_cfg: dict,
    max_turnover: float,
    entry_top_k: int | None = None,
    hold_buffer_top_k: int | None = None,
    top_tier_count: int | None = None,
    top_tier_weight_share: float | None = None,
    industry_map: Dict[str, str] | None = None,
    industry_cap_count: int | None = None,
    portfolio_method: str = "equal_weight",
    cov_lookback_days: int = 60,
    cov_ridge: float = 1e-6,
    cov_shrinkage: str = "ledoit_wolf",
    cov_ewma_halflife: float = 20.0,
    risk_aversion: float = 1.0,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_df = score_df.copy()
    factor_df = factor_df.copy()
    score_df["trade_date"] = pd.to_datetime(score_df["trade_date"])
    factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])
    fac_today = factor_df[
        ["symbol", "trade_date", "turnover_roll_mean", "price_position", "limit_move_hits_5d"]
    ].copy()

    rd_list = _rebalance_dates(score_df["trade_date"].unique(), rebalance_rule)
    rows = []
    diag_rows: list[dict[str, Any]] = []
    prev_holdings: set[str] = set()
    entry_top_k = int(max(1, int(entry_top_k if entry_top_k is not None else top_k)))
    hold_buffer_top_k = int(max(top_k, int(hold_buffer_top_k if hold_buffer_top_k is not None else top_k)))

    pf_enabled = bool(prefilter_cfg.get("enabled", True))
    limit_move_max = int(prefilter_cfg.get("limit_move_max", 2))
    turnover_low_pct = float(prefilter_cfg.get("turnover_low_pct", 0.10))
    turnover_high_pct = float(prefilter_cfg.get("turnover_high_pct", 0.98))
    price_position_high_pct = float(prefilter_cfg.get("price_position_high_pct", 0.90))

    for rd in rd_list:
        day_s = score_df[score_df["trade_date"] == rd].copy()
        if day_s.empty:
            continue
        day_s = day_s.merge(
            fac_today[fac_today["trade_date"] == rd],
            on=["symbol", "trade_date"], how="left",
        )
        filtered = apply_prefilter(
            day_s, top_k,
            enabled=pf_enabled,
            limit_move_max=limit_move_max,
            turnover_low_pct=turnover_low_pct,
            turnover_high_pct=turnover_high_pct,
            price_position_high_pct=price_position_high_pct,
        )

        if hold_buffer_top_k > top_k:
            topk = _select_topk_with_holding_buffer(
                filtered, top_k=top_k, entry_top_k=entry_top_k,
                hold_buffer_top_k=hold_buffer_top_k,
                prev_holdings=prev_holdings,
                industry_map=industry_map, industry_cap_count=industry_cap_count,
            )
        else:
            topk = _pick_topk_with_industry_cap(
                filtered, top_k=top_k,
                industry_map=industry_map, industry_cap_count=industry_cap_count,
            )
        if topk.empty:
            continue

        if prev_holdings and max_turnover < 1.0 and len(topk) >= top_k:
            selected = topk["symbol"].astype(str).tolist()
            selected_set = set(selected)
            required_overlap = int(np.ceil(top_k * (1.0 - max_turnover)))
            overlap = len(selected_set & prev_holdings)
            if overlap < required_overlap:
                need = required_overlap - overlap
                cand_prev = filtered[filtered["symbol"].astype(str).isin(prev_holdings)]
                cand_prev = cand_prev[~cand_prev["symbol"].astype(str).isin(selected_set)]
                cand_prev = cand_prev.nlargest(need, "score")[["symbol", "score"]]
                if not cand_prev.empty:
                    drop_cnt = min(len(cand_prev), len(topk))
                    non_overlap = topk[~topk["symbol"].astype(str).isin(prev_holdings)]
                    if len(non_overlap) >= drop_cnt:
                        to_drop = non_overlap.nsmallest(drop_cnt, "score").index
                    else:
                        to_drop = topk.nsmallest(drop_cnt, "score").index
                    topk = topk.drop(index=to_drop)
                    topk = pd.concat([topk, cand_prev], ignore_index=True)
                    topk = topk.drop_duplicates(subset=["symbol"], keep="first").nlargest(top_k, "score")

        topk = topk.sort_values("score", ascending=False).reset_index(drop=True)
        pm = str(portfolio_method).lower().strip()

        # 权重分配
        if pm in ("", "equal", "equal_weight"):
            ww = np.ones(len(topk), dtype=np.float64) / float(len(topk))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": "equal_weight",
                "n_assets": int(len(topk)), "effective_n": float(len(topk)),
                "weight_std": float(np.std(ww)), "max_weight": float(np.max(ww)),
                "is_equal_like": True, "solver_success": True, "fallback_reason": "",
                "post_constraint_l1_shift": 0.0,
            })
        elif pm in ("tiered_equal_weight", "two_tier_equal_weight"):
            ww, weight_diag = build_portfolio_weights(
                topk, weight_method=pm, score_col="score",
                top_tier_count=top_tier_count, top_tier_weight_share=top_tier_weight_share,
                max_single_weight=1.0, max_industry_weight=None, industry_col=None,
                prev_weights_aligned=None, max_turnover=1.0, return_diagnostics=True,
            )
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": pm,
                "n_assets": int(len(topk)), "effective_n": final_w_diag.get("effective_n"),
                "weight_std": final_w_diag.get("weight_std"),
                "max_weight": final_w_diag.get("max_weight"),
                "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                "is_equal_like": final_w_diag.get("is_close_to_reference"),
                "solver_success": True, "fallback_reason": "",
                "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
            })
        else:
            syms_topk = topk["symbol"].astype(str).str.zfill(6).tolist()
            mu_arr, cov_mtx = mean_cov_returns_from_daily_long(
                daily_df, syms_topk, asof=rd,
                lookback_days=int(cov_lookback_days),
                ridge=float(cov_ridge),
                shrinkage=str(cov_shrinkage).lower(),
                ewma_halflife=float(cov_ewma_halflife),
            )
            method_map = {"risk_parity": "risk_parity", "min_variance": "min_variance", "mean_variance": "mean_variance"}
            m = method_map.get(pm, "equal")
            exp_ret = mu_arr if m == "mean_variance" else None
            ww, weight_diag = build_portfolio_weights(
                topk, weight_method=m, score_col="score",
                max_single_weight=1.0, max_industry_weight=None, industry_col=None,
                prev_weights_aligned=None, max_turnover=1.0,
                cov_matrix=cov_mtx if m != "equal" else None,
                expected_returns=exp_ret, risk_aversion=float(risk_aversion),
                turnover_cost_model=None, return_diagnostics=True,
            )
            opt_diag = dict(weight_diag.get("optimizer", {}))
            cov_diag = dict(opt_diag.get("covariance", {}))
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append({
                "trade_date": pd.Timestamp(rd), "portfolio_method": pm,
                "n_assets": int(len(topk)), "effective_n": final_w_diag.get("effective_n"),
                "weight_std": final_w_diag.get("weight_std"),
                "max_weight": final_w_diag.get("max_weight"),
                "diag_share": cov_diag.get("diag_share"),
                "condition_number": cov_diag.get("condition_number"),
                "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                "is_equal_like": final_w_diag.get("is_close_to_reference"),
                "solver_success": opt_diag.get("solver_success"),
                "fallback_reason": opt_diag.get("fallback_reason"),
                "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
            })

        for i, r in topk.iterrows():
            rows.append({"trade_date": rd, "symbol": str(r["symbol"]).zfill(6), "weight": float(ww[i])})
        prev_holdings = set(topk["symbol"].astype(str).tolist())

    if not rows:
        raise RuntimeError("未生成任何调仓权重")
    w_long = pd.DataFrame(rows)
    w_wide = w_long.pivot(index="trade_date", columns="symbol", values="weight").fillna(0.0)
    w_wide.index = pd.to_datetime(w_wide.index)
    if not return_details:
        return w_wide
    diag_detail = pd.DataFrame(diag_rows).sort_values("trade_date").reset_index(drop=True)

    def _summarize_diag(detail: pd.DataFrame, *, method: str) -> dict[str, Any]:
        if detail.empty:
            return {"portfolio_method": str(method), "n_rebalances": 0}

        def _sm(col: str) -> float | None:
            if col not in detail.columns:
                return None
            vals = pd.to_numeric(detail[col], errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else None

        def _smed(col: str) -> float | None:
            if col not in detail.columns:
                return None
            vals = pd.to_numeric(detail[col], errors="coerce").dropna()
            return float(vals.median()) if not vals.empty else None

        return {
            "portfolio_method": str(method),
            "n_rebalances": int(len(detail)),
            "mean_weight_std": _sm("weight_std"),
            "median_effective_n": _smed("effective_n"),
            "mean_diag_share": _sm("diag_share"),
            "median_condition_number": _smed("condition_number"),
            "mean_l1_diff_vs_equal": _sm("l1_diff_vs_equal"),
            "equal_like_ratio": (
                float(pd.to_numeric(detail["is_equal_like"], errors="coerce").fillna(0.0).mean())
                if "is_equal_like" in detail.columns else None
            ),
            "solver_success_ratio": (
                float(pd.to_numeric(detail["solver_success"], errors="coerce").fillna(0.0).mean())
                if "solver_success" in detail.columns else None
            ),
        }

    diag_summary = _summarize_diag(diag_detail, method=portfolio_method)
    return w_wide, diag_detail, diag_summary


# ── 权重归一化 ───────────────────────────────────────────────────────────

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """归一化 composite_extended 权重：零权重因子剔除后按绝对值归一。"""
    cleaned = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    s = sum(abs(v) for v in cleaned.values())
    if s <= 0:
        raise ValueError("composite_extended 权重和为 0")
    return {k: v / s for k, v in cleaned.items()}


# ── P1 因子策略 ──────────────────────────────────────────────────────────

def load_factor_ic_summary(ic_report_path: str) -> pd.DataFrame:
    """从 CSV 或 JSON 加载因子 IC 汇总表。"""
    import json as _json
    from pathlib import Path as _Path

    p = _Path(ic_report_path).expanduser()
    if not p.is_absolute():
        return pd.DataFrame()
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".csv":
        tab = pd.read_csv(p, encoding="utf-8-sig")
    elif p.suffix.lower() == ".json":
        payload = _json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("summary"), list):
            tab = pd.DataFrame(payload["summary"])
        else:
            tab = pd.DataFrame(payload)
    else:
        return pd.DataFrame()
    need = {"factor", "horizon_key", "ic_mean"}
    if not need.issubset(set(tab.columns)):
        return pd.DataFrame()
    tab["factor"] = tab["factor"].astype(str)
    tab["horizon_key"] = tab["horizon_key"].astype(str)
    tab["ic_mean"] = pd.to_numeric(tab["ic_mean"], errors="coerce")
    return tab.dropna(subset=["factor", "horizon_key", "ic_mean"]).copy()


def _normalize_clip(weights: Dict[str, float], clip_abs_weight: float) -> Dict[str, float]:
    if not weights:
        return {}
    c = float(max(1e-6, clip_abs_weight))
    clipped = {k: float(np.clip(v, -c, c)) for k, v in weights.items()}
    s = float(sum(abs(v) for v in clipped.values()))
    if s <= 1e-12:
        return {}
    return {k: v / s for k, v in clipped.items()}


def _icir_from_history(ic_hist: pd.Series, half_life: float) -> float:
    ser = pd.to_numeric(ic_hist, errors="coerce").dropna()
    if ser.empty:
        return float("nan")
    n = len(ser)
    decay = float(np.exp(np.log(0.5) / max(float(half_life), 1.0)))
    w = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
    w = w / np.sum(w)
    x = ser.to_numpy(dtype=np.float64)
    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    sd = float(np.sqrt(max(var, 0.0)))
    if sd < 1e-12:
        return float("nan")
    return mu / sd


def _rolling_icir_series(
    ic_ser: pd.Series,
    *,
    window: int,
    min_obs: int,
    half_life: float,
) -> pd.Series:
    values = pd.to_numeric(ic_ser, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    history: list[float] = []
    for idx, val in enumerate(values):
        if np.isfinite(val):
            history.append(float(val))
        if len(history) < int(min_obs):
            continue
        hist = pd.Series(history[-int(window):], dtype=np.float64)
        icir = _icir_from_history(hist, half_life=half_life)
        if np.isfinite(icir):
            out[idx] = float(icir)
    return pd.Series(out, index=ic_ser.index, dtype=np.float64)


def build_weights_by_date(
    ic_df: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
    half_life: float,
    clip_abs_weight: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if ic_df.empty:
        return out
    tab = ic_df.copy()
    tab["trade_date"] = pd.to_datetime(tab["trade_date"]).dt.normalize()
    tab["factor"] = tab["factor"].astype(str)
    wide = (
        tab.pivot_table(index="trade_date", columns="factor", values="ic", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    if wide.empty:
        return out
    icir_wide = pd.DataFrame(index=wide.index)
    for fac in wide.columns:
        icir_wide[str(fac)] = _rolling_icir_series(
            wide[fac],
            window=window,
            min_obs=min_obs,
            half_life=half_life,
        )
    for dt, row in icir_wide.iterrows():
        raw = {str(fac): float(val) for fac, val in row.items() if np.isfinite(float(val))}
        normed = _normalize_clip(raw, clip_abs_weight=clip_abs_weight)
        if normed:
            out[pd.Timestamp(dt).strftime("%Y-%m-%d")] = normed
    return out


def apply_p1_factor_policy(
    base_weights: Dict[str, float],
    ic_summary: pd.DataFrame,
    *,
    remove_if_t1_and_t21_negative: bool = True,
    zero_if_abs_t1_below: float = 0.0,
    flip_if_t1_negative_and_t21_above: float = 0.005,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """按 P1 IC 规则过滤/归零/反转因子权重。"""
    if not base_weights or ic_summary.empty:
        return dict(base_weights), pd.DataFrame(columns=["factor", "action", "ic_mean_t1", "ic_mean_t21"])

    piv = (
        ic_summary.pivot_table(index="factor", columns="horizon_key", values="ic_mean", aggfunc="first")
        .rename_axis(None, axis=1)
        .reset_index()
    )
    if "tplus1_open_1d" not in piv.columns:
        piv["tplus1_open_1d"] = np.nan
    if "close_21d" not in piv.columns:
        piv["close_21d"] = np.nan

    piv = piv.set_index("factor")
    updated = dict(base_weights)
    rows: list[dict[str, Any]] = []
    for fac, cur_w_raw in base_weights.items():
        row = piv.loc[fac] if fac in piv.index else None
        t1 = float(row["tplus1_open_1d"]) if row is not None and pd.notna(row["tplus1_open_1d"]) else float("nan")
        t21 = float(row["close_21d"]) if row is not None and pd.notna(row["close_21d"]) else float("nan")
        action = "keep"
        cur_w = float(cur_w_raw)
        if not np.isfinite(t1) and not np.isfinite(t21):
            updated[fac] = 0.0
            action = "remove"
        elif (
            bool(remove_if_t1_and_t21_negative)
            and np.isfinite(t1)
            and np.isfinite(t21)
            and t1 < 0.0
            and t21 < 0.0
        ):
            updated[fac] = 0.0
            action = "remove"
        elif np.isfinite(t1) and abs(t1) < float(zero_if_abs_t1_below):
            updated[fac] = 0.0
            action = "zero"
        elif (
            np.isfinite(t1)
            and np.isfinite(t21)
            and t1 < 0.0
            and t21 > float(flip_if_t1_negative_and_t21_above)
            and abs(cur_w) > 1e-12
        ):
            updated[fac] = -cur_w
            action = "flip"
        rows.append(
            {
                "factor": fac,
                "action": action,
                "ic_mean_t1": t1,
                "ic_mean_t21": t21,
            }
        )
    try:
        normalized = normalize_weights(updated)
    except ValueError:
        normalized = dict(base_weights)
    return normalized, pd.DataFrame(rows).sort_values(["action", "factor"]).reset_index(drop=True)


# ── IC 动态权重 ──────────────────────────────────────────────────────────

def load_ic_weights_by_date(ic_weights_json: str) -> Dict[pd.Timestamp, Dict[str, float]]:
    """从 JSON 文件加载按日期的 IC 动态权重。"""
    import json as _json
    from pathlib import Path as _Path

    p = _Path(ic_weights_json).expanduser()
    if not p.exists():
        return {}
    try:
        payload = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("weights_by_date")
    if not isinstance(rows, dict):
        return {}
    out: Dict[pd.Timestamp, Dict[str, float]] = {}
    for k, v in rows.items():
        if not isinstance(v, dict):
            continue
        dt = pd.to_datetime(k, errors="coerce")
        if pd.isna(dt):
            continue
        out[pd.Timestamp(dt).normalize()] = {str(f): float(w) for f, w in v.items()}
    return out


def build_ic_weights_from_monitor(
    monitor_path: str,
    *,
    window: int,
    min_obs: int,
    half_life: float,
    clip_abs_weight: float,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """从 IC monitor 日志构建按日期的动态权重。"""
    from pathlib import Path as _Path

    p = _Path(monitor_path).expanduser()
    if not p.exists():
        return {}
    mon = ICMonitor(p)
    ic_df = mon.load_dataframe()
    if ic_df.empty:
        return {}
    raw_out = build_weights_by_date(
        ic_df,
        window=window,
        min_obs=min_obs,
        half_life=half_life,
        clip_abs_weight=clip_abs_weight,
    )
    return {
        pd.Timestamp(dt).normalize(): {str(f): float(w) for f, w in weights.items()}
        for dt, weights in raw_out.items()
    }


# ── 基准 & 收益矩阵 ─────────────────────────────────────────────────────

def build_market_ew_open_to_open_benchmark(
    daily_df: pd.DataFrame,
    start: str,
    end: str,
    min_days: int = 500,
) -> pd.Series:
    """构建全市场等权 open-to-open 日收益基准。"""
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (pd.to_numeric(daily_df["open"], errors="coerce") > 0)
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    sym_cnt = df.groupby("symbol")["trade_date"].count()
    good = sym_cnt[sym_cnt >= min_days].index
    if len(good) == 0:
        good = sym_cnt.index
    open_returns = _build_open_to_open_returns(df[df["symbol"].isin(good)], zero_if_limit_up_open=False)
    cs = open_returns.mean(axis=1, skipna=True).dropna()
    cs.index = pd.to_datetime(cs.index)
    return cs.astype(np.float64)


def build_symbol_benchmark(daily_df: pd.DataFrame, symbol: str, start: str, end: str) -> pd.Series:
    """构建单标的 close-to-close 日收益基准。"""
    sym = str(symbol).zfill(6)
    df = daily_df[
        (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["symbol"] == sym)
        & (daily_df["close"] > 0)
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float64)
    df = df.sort_values("trade_date")
    ret = pd.to_numeric(df["close"], errors="coerce").pct_change()
    ret.index = pd.to_datetime(df["trade_date"])
    return ret.dropna().astype(np.float64)


def build_asset_returns(daily_df: pd.DataFrame, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """从日线构建标的 close-to-close 收益矩阵。"""
    syms = set(str(s).zfill(6) for s in symbols)
    d = daily_df[
        daily_df["symbol"].isin(syms)
        & (daily_df["trade_date"] >= pd.Timestamp(start))
        & (daily_df["trade_date"] <= pd.Timestamp(end))
        & (daily_df["close"] > 0)
    ].copy()
    d = d.sort_values(["symbol", "trade_date"])
    d["ret"] = d.groupby("symbol")["close"].pct_change()
    d = d.dropna(subset=["ret"])
    returns = d.pivot(index="trade_date", columns="symbol", values="ret").sort_index()
    returns.index = pd.to_datetime(returns.index)
    return returns.fillna(0.0)
