"""
A 股量化回测评估脚本（改进版）
================================

改进点（对应 docs/backtest_report.md）：
1) 因子方向纠偏：从 config.yaml 读取 composite_extended 权重，支持反转方向。
2) 预过滤：限制近 5 日涨跌停次数、极端换手、绝对高位。
3) 低换手约束：通过持仓重叠率约束等权 Top-K 的半 L1 换手。
4) 回测执行：统一 close_to_close 口径，输出无成本/含成本对比。
5) 样本外验证：滚动窗口 + 时间切片 Walk-Forward。
6) 默认排序键约定：生产与评估默认使用 composite_extended。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import duckdb
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.update_ic_weights import build_weights_by_date
from scripts.research_identity import build_full_backtest_research_identity, canonical_research_config, slugify_token
from src.features.fund_flow_factors import attach_fund_flow
from src.features.fundamental_factors import preprocess_fundamental_cross_section
from src.features.ic_monitor import ICMonitor
from src.features.shareholder_factors import attach_shareholder_factors
from src.backtest.engine import BacktestConfig, build_open_to_open_returns, run_backtest
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.transaction_costs import transaction_cost_params_from_mapping
from src.backtest.walk_forward import (
    compare_full_vs_slices,
    contiguous_time_splits,
    rolling_walk_forward_windows,
    summarize_oos_excess_returns,
    walk_forward_backtest,
)
from src.market.regime import (
    MARKET_EW_PROXY,
    classify_regime,
    get_regime_weights,
    regime_config_from_mapping,
)
from src.models.experiment import append_backtest_result
from src.models.artifacts import load_bundle_metadata
from src.models.rank_score import sort_key_for_dataframe
from src.portfolio.covariance import mean_cov_returns_from_daily_long
from src.portfolio.weights import build_portfolio_weights

DEFAULT_CONFIG: dict = {
    "paths": {
        "duckdb_path": "data/market.duckdb",
        "asof_trade_date": "",
    },
    "transaction_costs": {
        "commission_buy_bps": 2.5,
        "commission_sell_bps": 2.5,
        "slippage_bps_per_side": 4.5,
        "stamp_duty_sell_bps": 5.0,
    },
    "signals": {
        "top_k": 10,
        "composite_extended": {
            "momentum_12_1": 0.10,
            "momentum": -0.15,
            "rsi": -0.15,
            "realized_vol": 0.03,
            "turnover_roll_mean": 0.05,
            "turnover_rel_pct_252": -0.04,
            "vol_ret_corr": 0.05,
            "vol_to_turnover": 0.02,
            "bias_short": -0.12,
            "bias_long": -0.06,
            "max_single_day_drop": 0.06,
            "recent_return": -0.15,
            "price_position": -0.04,
            "intraday_range": -0.04,
            "upper_shadow_ratio": -0.05,
            "lower_shadow_ratio": 0.04,
            "close_open_return": 0.04,
            "tail_strength": 0.05,
            "pe_ttm": -0.04,
            "pb": -0.04,
            "roe_ttm": 0.04,
            "net_profit_yoy": 0.03,
            "ocf_to_net_profit": 0.03,
        },
        "p1_factor_filter": {
            "enabled": False,
            "ic_report_path": "data/results/factor_ic_report.csv",
            "remove_if_t1_and_t21_negative": True,
            "zero_if_abs_t1_below": 0.0,
            "flip_if_t1_negative_and_t21_above": 0.005,
        },
        "ic_weighting": {
            "enabled": False,
            "weights_path": "data/cache/ic_weights.json",
            "monitor_path": "data/logs/ic_monitor.json",
            "window": 60,
            "min_obs": 20,
            "half_life": 20.0,
            "clip_abs_weight": 0.25,
        },
    },
    "prefilter": {
        "enabled": True,
        "limit_move_max": 2,
        "turnover_low_pct": 0.10,
        "turnover_high_pct": 0.98,
        "price_position_high_pct": 0.90,
    },
    "portfolio": {
        "max_turnover": 0.3,
        "industry_cap_count": 5,
    },
    "backtest": {
        "execution_mode": "tplus1_open",
        "execution_lag": 1,
        "eval_rebalance_rule": "M",
        "limit_up_mode": "redistribute",
        "vwap_slippage_bps_per_side": 3.0,
        "vwap_impact_bps": 8.0,
    },
}


def fmt_pct(v: float, d: int = 2) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v * 100:+.{d}f}%"


def fmt_num(v: float, d: int = 3) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v:+.{d}f}"


def _safe_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _merge_signals(base_sig: dict, override_sig: dict) -> dict:
    """合并 signals：composite_extended 由配置文件整表覆盖，不与内置默认逐键合并。"""
    out = dict(base_sig)
    for k, v in override_sig.items():
        if k == "composite_extended" and isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "signals" and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_signals(out[k], v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str = "") -> tuple[dict, str]:
    candidate_paths: list[Path] = []
    if str(config_path).strip():
        p = Path(str(config_path).strip())
        candidate_paths.append(p if p.is_absolute() else PROJECT_ROOT / p)
    else:
        candidate_paths.extend([PROJECT_ROOT / "config.yaml.backtest", PROJECT_ROOT / "config.yaml"])

    for cfg_path in candidate_paths:
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULT_CONFIG, loaded), str(cfg_path)

    searched = ", ".join(str(p) for p in candidate_paths)
    print(f"[提示] 未找到配置文件（{searched}），使用脚本内置默认配置。")
    return dict(DEFAULT_CONFIG), "builtin_defaults"


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    s = sum(abs(v) for v in cleaned.values())
    if s <= 0:
        raise ValueError("composite_extended 权重和为 0")
    return {k: v / s for k, v in cleaned.items()}


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
        return pd.DataFrame(columns=["weekly_kdj_k", "weekly_kdj_d", "weekly_kdj_j"])

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
        weekly[
            [
                "week_last_trade_date",
                "weekly_kdj_k",
                "weekly_kdj_d",
                "weekly_kdj_j",
                "weekly_kdj_oversold",
                "weekly_kdj_oversold_depth",
                "weekly_kdj_rebound",
            ]
        ],
        left_on="trade_date",
        right_on="week_last_trade_date",
        how="left",
    )
    weekly_cols = [
        "weekly_kdj_k",
        "weekly_kdj_d",
        "weekly_kdj_j",
        "weekly_kdj_oversold",
        "weekly_kdj_oversold_depth",
        "weekly_kdj_rebound",
    ]
    merged[weekly_cols] = merged[weekly_cols].ffill()
    intraweek_mask = merged["trade_date"] != merged["week_last_trade_date"]
    merged.loc[intraweek_mask, weekly_cols] = merged.loc[intraweek_mask, weekly_cols].shift(1)
    return merged[weekly_cols]


def _resolve_optional_path(path_like: str) -> Path | None:
    p_raw = str(path_like or "").strip()
    if not p_raw:
        return None
    p = Path(p_raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_factor_ic_summary(ic_report_path: str) -> pd.DataFrame:
    p = _resolve_optional_path(ic_report_path)
    if p is None or not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".csv":
        tab = pd.read_csv(p, encoding="utf-8-sig")
    elif p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
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


def apply_p1_factor_policy(
    base_weights: Dict[str, float],
    ic_summary: pd.DataFrame,
    *,
    remove_if_t1_and_t21_negative: bool = True,
    zero_if_abs_t1_below: float = 0.0,
    flip_if_t1_negative_and_t21_above: float = 0.005,
) -> tuple[Dict[str, float], pd.DataFrame]:
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


def load_ic_weights_by_date(ic_weights_json: str) -> Dict[pd.Timestamp, Dict[str, float]]:
    p = Path(ic_weights_json)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
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
    p = Path(monitor_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
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


def load_daily_from_duckdb(db_path: str, start: str, end: str, lookback_days: int) -> pd.DataFrame:
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


def compute_factors(daily_df: pd.DataFrame, min_hist_days: int = 130) -> pd.DataFrame:
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
        # P1-A：12-1 动量（排除最近一个月），降低短期反转噪声。
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
        # 使用 pandas rolling().rank(pct=True) 向量化计算百分位排名（pandas 1.2+）
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
        if "amount" not in g.columns:
            amt = v * c
        else:
            amt = pd.to_numeric(g["amount"], errors="coerce")

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


def _factor_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.meta.json")


PREPARED_FACTORS_SCHEMA_VERSION = 20260424
PREPARED_FACTORS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "momentum",
    "momentum_12_1",
    "rsi",
    "atr",
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
    "llm_sentiment_z",
    "_universe_eligible",
)


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
    except Exception:  # noqa: BLE001
        return None
    if actual_meta != expected_meta:
        return None
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:  # noqa: BLE001
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
        json.dumps(_json_sanitize(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _tree_score_auto_flipped_from_metrics(metrics: dict[str, Any]) -> bool:
    for key in ("val_rank_ic", "train_rank_ic"):
        val = _safe_float_or_nan(metrics.get(key))
        if np.isfinite(val):
            return bool(val < 0)
    return False


def _tree_bundle_report_meta(
    *,
    sort_by: str,
    bundle_dir: str,
    tree_feature_group: str,
    requested_features: list[str],
) -> dict[str, Any]:
    if str(sort_by).lower().strip() != "xgboost":
        return {}
    out: dict[str, Any] = {
        "bundle_dir": str(bundle_dir or ""),
        "feature_group": str(tree_feature_group or ""),
        "requested_features": list(requested_features),
        "tree_score_auto_flipped": None,
        "label_spec": {},
    }
    if not str(bundle_dir or "").strip():
        return out
    try:
        meta = load_bundle_metadata(bundle_dir)
    except Exception as exc:  # noqa: BLE001
        out["bundle_meta_error"] = str(exc)
        return out
    params = dict(getattr(meta, "params", {}) or {})
    metrics = dict(getattr(meta, "metrics", {}) or {})
    label_spec = dict(params.get("label_spec") or {})
    out.update(
        {
            "model_version": getattr(meta, "model_version", ""),
            "feature_version": getattr(meta, "feature_version", ""),
            "created_at": getattr(meta, "created_at", ""),
            "research_topic": params.get("research_topic", ""),
            "research_config_id": params.get("research_config_id", ""),
            "research_group": params.get("research_group", ""),
            "bundle_label": params.get("bundle_label", ""),
            "label_spec": label_spec,
            "metrics": {
                "train_rank_ic": metrics.get("train_rank_ic"),
                "val_rank_ic": metrics.get("val_rank_ic"),
                "train_mse": metrics.get("train_mse"),
                "val_mse": metrics.get("val_mse"),
            },
            "tree_score_auto_flipped": _tree_score_auto_flipped_from_metrics(metrics),
        }
    )
    if not out["feature_group"]:
        out["feature_group"] = str(params.get("research_group") or "")
    return _json_sanitize(out)


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
    factors = _attach_llm_sentiment(factors, results_dir)
    factors = attach_universe_filter(
        factors,
        daily_df,
        enabled=bool(universe_filter_cfg.get("enabled", False)),
        min_amount_20d=float(universe_filter_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(universe_filter_cfg.get("require_roe_ttm_positive", True)),
    )
    if cache_path is not None:
        write_prepared_factors_cache(cache_path, factors, expected_meta)
    return factors, False


def _attach_pit_fundamentals(factors: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """按公告日 merge_asof，将基本面快照对齐到每个交易日（PIT）。"""
    out = factors.copy(deep=False)
    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'a_share_fundamental'
            """
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return out
        info = con.execute("PRAGMA table_info('a_share_fundamental')").fetchall()
        have = {str(r[1]) for r in info}
        want_cols = [
            "symbol",
            "report_period",
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
        ]
        sel = [c for c in want_cols if c in have]
        fund = con.execute(f"SELECT {', '.join(sel)} FROM a_share_fundamental").df()
        for c in want_cols:
            if c not in fund.columns:
                fund[c] = np.nan
    finally:
        con.close()
    if fund.empty:
        return out

    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    out = out.dropna(subset=["trade_date"])
    fund["symbol"] = fund["symbol"].astype(str).str.zfill(6)
    fund["announcement_date"] = (
        pd.to_datetime(fund["announcement_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    )
    fund["report_period"] = pd.to_datetime(fund["report_period"], errors="coerce").astype("datetime64[ns]")
    fund = fund.dropna(subset=["announcement_date"])
    if fund.empty:
        return out
    # 同一 (symbol, announcement_date) 可能对应多份报告期；保留 report_period 最新的一条，避免 merge_asof 右侧乱序
    fund = fund.sort_values(
        ["symbol", "announcement_date", "report_period"],
        na_position="last",
        kind="mergesort",
    )
    fund = fund.drop_duplicates(["symbol", "announcement_date"], keep="last")
    fund = fund.drop(columns=["report_period"], errors="ignore")
    # `merge_asof + preprocess_fundamental_cross_section` 在全量 700w+ 行上峰值内存过高，
    # 改为按日期块分批处理，保持同日截面处理语义不变，同时显著降低峰值内存。
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
            (fund["announcement_date"] <= chunk_end) & fund["symbol"].astype(str).isin(chunk_symbols)
        ].copy()
        if fund_chunk.empty:
            merged = chunk.copy()
            for c in want_cols:
                if c not in merged.columns:
                    merged[c] = np.nan
        else:
            fund_chunk = fund_chunk.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
            merged = pd.merge_asof(
                chunk,
                fund_chunk,
                left_on="trade_date",
                right_on="announcement_date",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
        merged = preprocess_fundamental_cross_section(
            merged,
            date_col="trade_date",
            size_col="log_market_cap",
            neutralize=True,
        )
        chunked.append(merged)
    if not chunked:
        return out
    return pd.concat(chunked, ignore_index=True)


def _attach_llm_sentiment(factors: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """
    读取 ``llm_attention_*.csv`` 历史结果，构造按日截面的 ``llm_sentiment_z``。
    历史不足时保留空列，回测自动降级。
    """
    out = factors.copy()
    files = sorted(results_dir.glob("llm_attention_*.csv"))
    if not files:
        out["llm_sentiment_z"] = np.nan
        return out

    rows: list[pd.DataFrame] = []
    for p in files:
        ds = p.stem.replace("llm_attention_", "")
        dt = pd.to_datetime(ds, errors="coerce")
        if pd.isna(dt):
            continue
        try:
            tab = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            continue
        if tab.empty:
            continue
        sym_col = "symbol" if "symbol" in tab.columns else ("代码" if "代码" in tab.columns else None)
        if sym_col is None:
            continue
        sig_col = "significance" if "significance" in tab.columns else None
        rank_col = "attention_rank_change" if "attention_rank_change" in tab.columns else None
        score = pd.Series(0.0, index=tab.index, dtype=float)
        if sig_col is not None:
            score = score + pd.to_numeric(tab[sig_col], errors="coerce").fillna(0.0)
        if rank_col is not None:
            score = score + pd.to_numeric(tab[rank_col], errors="coerce").fillna(0.0) * 0.5
        sd = float(score.std(ddof=0))
        z = pd.Series(0.0, index=score.index, dtype=float) if abs(sd) < 1e-12 else (score - float(score.mean())) / sd
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": pd.Timestamp(dt).normalize(),
                    "symbol": tab[sym_col].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6),
                    "llm_sentiment_z": z.astype(float),
                }
            )
        )
    if not rows:
        out["llm_sentiment_z"] = np.nan
        return out

    llm = pd.concat(rows, ignore_index=True)
    llm = llm[(llm["symbol"].str.len() == 6)].drop_duplicates(["trade_date", "symbol"], keep="last")
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out = out.merge(llm, on=["trade_date", "symbol"], how="left")
    return out


def attach_universe_filter(
    factors: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    enabled: bool,
    min_amount_20d: float,
    require_roe_ttm_positive: bool,
) -> pd.DataFrame:
    """M2.4：流动性（20 日日均成交额）+ 盈利（ROE_TTM>0）universe，写入 _universe_eligible。"""
    out = factors.copy()
    if not enabled:
        out["_universe_eligible"] = True
        return out
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    amt = pd.to_numeric(d["amount"], errors="coerce")
    d["_amt20"] = amt.groupby(d["symbol"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    liq = d[["symbol", "trade_date", "_amt20"]].drop_duplicates(["symbol", "trade_date"], keep="last")
    out = out.merge(liq, on=["symbol", "trade_date"], how="left")
    roe = pd.to_numeric(out["roe_ttm"], errors="coerce") if "roe_ttm" in out.columns else pd.Series(np.nan, index=out.index)
    ok = pd.Series(True, index=out.index)
    if float(min_amount_20d) > 0.0:
        ok &= out["_amt20"].fillna(0.0) >= float(min_amount_20d)
    if bool(require_roe_ttm_positive):
        ok &= roe.fillna(-np.inf) > 0.0
    out["_universe_eligible"] = ok.to_numpy(dtype=bool)
    return out.drop(columns=["_amt20"])


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
                g,
                sort_by="xgboost",
                tree_bundle_dir=tree_bundle_dir,
                tree_raw_features=list(tree_raw_features or []),
                tree_rsi_mode=tree_rsi_mode,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": ranked["symbol"].values,
                        "trade_date": dt,
                        "score": pd.to_numeric(ranked["tree_score"], errors="coerce").to_numpy(dtype=np.float64),
                    }
                )
            )
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


def resolve_industry_cap_and_map(
    industry_cap_raw: int,
    industry_map_csv: str,
) -> tuple[int, Dict[str, str], str]:
    """
    解析行业约束参数：
    - cap <= 0：关闭行业约束；
    - cap > 0 且映射存在：启用；
    - cap > 0 但映射缺失：静默降级为关闭（P2-B）。
    """
    industry_cap_count = int(max(0, int(industry_cap_raw)))
    if industry_cap_count <= 0:
        return 0, {}, "disabled_by_config"
    industry_map = load_industry_map(industry_map_csv)
    if not industry_map:
        return 0, {}, "disabled_missing_map"
    return industry_cap_count, industry_map, "enabled"


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

    # 行业约束采用硬约束：若约束后不足 top_k，则允许持仓数小于 top_k。
    # 这样可确保「每行业持仓 <= cap」不被后续补齐逻辑破坏。
    return pd.DataFrame(picked).nlargest(top_k, "score")


def _select_topk_with_holding_buffer(
    day_df: pd.DataFrame,
    *,
    top_k: int,
    entry_top_k: int,
    hold_buffer_top_k: int,
    prev_holdings: set[str],
    industry_map: Dict[str, str] | None,
    industry_cap_count: int | None,
) -> pd.DataFrame:
    """
    进入/退出缓冲带：
    - 新买入只从 ``entry_top_k`` 内挑选；
    - 旧持仓只要仍在 ``hold_buffer_top_k`` 内即可继续保留；
    - 最终持仓数仍固定为 ``top_k``。
    """
    ranked = day_df.sort_values("score", ascending=False).reset_index(drop=True)
    base_buy = _pick_topk_with_industry_cap(
        ranked,
        top_k=max(1, int(entry_top_k)),
        industry_map=industry_map,
        industry_cap_count=industry_cap_count,
    ).sort_values("score", ascending=False)
    if not prev_holdings or int(hold_buffer_top_k) <= int(top_k):
        return base_buy.nlargest(top_k, "score")

    buffer_pool = _pick_topk_with_industry_cap(
        ranked,
        top_k=max(int(top_k), int(hold_buffer_top_k)),
        industry_map=industry_map,
        industry_cap_count=industry_cap_count,
    ).sort_values("score", ascending=False)
    keep = buffer_pool[buffer_pool["symbol"].astype(str).isin(prev_holdings)].copy()
    keep = keep.sort_values("score", ascending=False)

    prioritized = []
    if not keep.empty:
        prioritized.append(keep.assign(_priority=0))
    if not base_buy.empty:
        prioritized.append(base_buy.assign(_priority=1))
    if not buffer_pool.empty:
        prioritized.append(buffer_pool.assign(_priority=2))
    if not prioritized:
        return pd.DataFrame(columns=["symbol", "score"])

    selected = pd.concat(prioritized, ignore_index=True)
    selected["symbol"] = selected["symbol"].astype(str).str.zfill(6)
    selected = selected.drop_duplicates(subset=["symbol"], keep="first")
    selected = selected.sort_values(["_priority", "score"], ascending=[True, False]).head(top_k)
    return selected.drop(columns=["_priority"], errors="ignore").sort_values("score", ascending=False).reset_index(drop=True)


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
            daily_df,
            str(fac_start.date()),
            str(fac_end.date()),
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
        bench_s = pd.Series(bench["ret"].to_numpy(dtype=np.float64), index=pd.to_datetime(bench["trade_date"]))
        bench_s = bench_s.sort_index()

    overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for dt in sorted(pd.to_datetime(factors["trade_date"]).unique()):
        regime, result = classify_regime(bench_s, dt, cfg=cfg)
        overrides[pd.Timestamp(dt)] = get_regime_weights(base_weights, regime, cfg=cfg, regime_result=result)
        rows.append(
            {
                "trade_date": pd.Timestamp(dt),
                "regime": regime,
                "short_return": float(result.short_return),
                "vol_ann": float(result.realized_vol_ann),
            }
        )
    return overrides, pd.DataFrame(rows)


def _rebalance_dates(all_dates: Iterable[pd.Timestamp], rule: str) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(list(all_dates)))
    if not dates:
        return []
    arr = np.array(dates, dtype="datetime64[ns]")
    freq = "ME" if str(rule).upper() == "M" else rule
    anchors = pd.date_range(dates[0], dates[-1], freq=freq)
    out: list[pd.Timestamp] = []
    for a in anchors:
        pos = np.searchsorted(arr, np.datetime64(a), side="right") - 1
        if pos >= 0:
            out.append(pd.Timestamp(arr[pos]))
    return sorted(set(out))


def _summarize_portfolio_diagnostics(detail: pd.DataFrame, *, method: str) -> dict[str, Any]:
    """按调仓日明细聚合组合优化诊断摘要。"""
    if detail.empty:
        return {"portfolio_method": str(method), "n_rebalances": 0}

    def _safe_mean(col: str) -> float | None:
        if col not in detail.columns:
            return None
        vals = pd.to_numeric(detail[col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None

    def _safe_median(col: str) -> float | None:
        if col not in detail.columns:
            return None
        vals = pd.to_numeric(detail[col], errors="coerce").dropna()
        return float(vals.median()) if not vals.empty else None

    fallback_counts = (
        detail["fallback_reason"]
        .fillna("")
        .astype(str)
        .replace("", "_none")
        .value_counts()
        .to_dict()
        if "fallback_reason" in detail.columns
        else {}
    )
    return {
        "portfolio_method": str(method),
        "n_rebalances": int(len(detail)),
        "mean_weight_std": _safe_mean("weight_std"),
        "median_effective_n": _safe_median("effective_n"),
        "mean_diag_share": _safe_mean("diag_share"),
        "median_condition_number": _safe_median("condition_number"),
        "mean_l1_diff_vs_equal": _safe_mean("l1_diff_vs_equal"),
        "max_l1_diff_vs_equal": (
            float(pd.to_numeric(detail["l1_diff_vs_equal"], errors="coerce").max())
            if "l1_diff_vs_equal" in detail.columns and not detail.empty
            else None
        ),
        "equal_like_ratio": (
            float(pd.to_numeric(detail["is_equal_like"], errors="coerce").fillna(0.0).mean())
            if "is_equal_like" in detail.columns
            else None
        ),
        "solver_success_ratio": (
            float(pd.to_numeric(detail["solver_success"], errors="coerce").fillna(0.0).mean())
            if "solver_success" in detail.columns
            else None
        ),
        "fallback_counts": fallback_counts,
    }


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
            on=["symbol", "trade_date"],
            how="left",
        )
        filtered = day_s
        if pf_enabled and len(day_s) >= top_k:
            lm = pd.to_numeric(filtered["limit_move_hits_5d"], errors="coerce").fillna(0.0)
            filtered = filtered[lm <= float(limit_move_max)]

            to = pd.to_numeric(filtered["turnover_roll_mean"], errors="coerce")
            if to.notna().sum() >= max(50, top_k):
                lo = to.quantile(turnover_low_pct)
                hi = to.quantile(turnover_high_pct)
                filtered = filtered[(to >= lo) & (to <= hi) | to.isna()]

            pp = pd.to_numeric(filtered["price_position"], errors="coerce")
            filtered = filtered[(pp <= price_position_high_pct) | pp.isna()]
            if len(filtered) < top_k:
                filtered = day_s

        if hold_buffer_top_k > top_k:
            topk = _select_topk_with_holding_buffer(
                filtered,
                top_k=top_k,
                entry_top_k=entry_top_k,
                hold_buffer_top_k=hold_buffer_top_k,
                prev_holdings=prev_holdings,
                industry_map=industry_map,
                industry_cap_count=industry_cap_count,
            )
        else:
            topk = _pick_topk_with_industry_cap(
                filtered,
                top_k=top_k,
                industry_map=industry_map,
                industry_cap_count=industry_cap_count,
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
        if pm in ("", "equal", "equal_weight"):
            ww = np.ones(len(topk), dtype=np.float64) / float(len(topk))
            diag_rows.append(
                {
                    "trade_date": pd.Timestamp(rd),
                    "portfolio_method": "equal_weight",
                    "n_assets": int(len(topk)),
                    "effective_n": float(len(topk)),
                    "weight_std": float(np.std(ww)),
                    "max_weight": float(np.max(ww)),
                    "diag_share": None,
                    "condition_number": None,
                    "l1_diff_vs_equal": 0.0,
                    "max_abs_diff_vs_equal": 0.0,
                    "is_equal_like": True,
                    "solver_success": True,
                    "solver_status": 0,
                    "solver_iterations": 0,
                    "fallback_reason": "",
                    "post_constraint_l1_shift": 0.0,
                }
            )
        elif pm in ("tiered_equal_weight", "two_tier_equal_weight"):
            ww, weight_diag = build_portfolio_weights(
                topk,
                weight_method=pm,
                score_col="score",
                top_tier_count=top_tier_count,
                top_tier_weight_share=top_tier_weight_share,
                max_single_weight=1.0,
                max_industry_weight=None,
                industry_col=None,
                prev_weights_aligned=None,
                max_turnover=1.0,
                return_diagnostics=True,
            )
            opt_diag = dict(weight_diag.get("optimizer", {}))
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append(
                {
                    "trade_date": pd.Timestamp(rd),
                    "portfolio_method": pm,
                    "n_assets": int(len(topk)),
                    "effective_n": final_w_diag.get("effective_n"),
                    "weight_std": final_w_diag.get("weight_std"),
                    "max_weight": final_w_diag.get("max_weight"),
                    "diag_share": None,
                    "condition_number": None,
                    "mean_abs_offdiag": None,
                    "mean_correlation": None,
                    "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                    "max_abs_diff_vs_equal": final_w_diag.get("max_abs_diff_vs_reference"),
                    "is_equal_like": final_w_diag.get("is_close_to_reference"),
                    "solver_success": True,
                    "solver_status": 0,
                    "solver_iterations": 0,
                    "fallback_reason": "",
                    "risk_contribution_std": None,
                    "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
                    "top_tier_count": opt_diag.get("top_tier_count"),
                    "top_tier_weight_share": opt_diag.get("top_tier_weight_share"),
                }
            )
        else:
            syms_topk = topk["symbol"].astype(str).str.zfill(6).tolist()
            mu_arr, cov_mtx = mean_cov_returns_from_daily_long(
                daily_df,
                syms_topk,
                asof=rd,
                lookback_days=int(cov_lookback_days),
                ridge=float(cov_ridge),
                shrinkage=str(cov_shrinkage).lower(),  # type: ignore[arg-type]
                ewma_halflife=float(cov_ewma_halflife),
            )
            method_map = {
                "risk_parity": "risk_parity",
                "min_variance": "min_variance",
                "mean_variance": "mean_variance",
            }
            m = method_map.get(pm, "equal")
            exp_ret = mu_arr if m == "mean_variance" else None
            ww, weight_diag = build_portfolio_weights(
                topk,
                weight_method=m,
                score_col="score",
                max_single_weight=1.0,
                max_industry_weight=None,
                industry_col=None,
                prev_weights_aligned=None,
                max_turnover=1.0,
                cov_matrix=cov_mtx if m != "equal" else None,
                expected_returns=exp_ret,
                risk_aversion=float(risk_aversion),
                turnover_cost_model=None,
                return_diagnostics=True,
            )
            opt_diag = dict(weight_diag.get("optimizer", {}))
            cov_diag = dict(opt_diag.get("covariance", {}))
            final_w_diag = dict(weight_diag.get("post_constraints", {}))
            diag_rows.append(
                {
                    "trade_date": pd.Timestamp(rd),
                    "portfolio_method": pm,
                    "n_assets": int(len(topk)),
                    "effective_n": final_w_diag.get("effective_n"),
                    "weight_std": final_w_diag.get("weight_std"),
                    "max_weight": final_w_diag.get("max_weight"),
                    "diag_share": cov_diag.get("diag_share"),
                    "condition_number": cov_diag.get("condition_number"),
                    "mean_abs_offdiag": cov_diag.get("mean_abs_offdiag"),
                    "mean_correlation": cov_diag.get("mean_correlation"),
                    "l1_diff_vs_equal": final_w_diag.get("l1_diff_vs_reference"),
                    "max_abs_diff_vs_equal": final_w_diag.get("max_abs_diff_vs_reference"),
                    "is_equal_like": final_w_diag.get("is_close_to_reference"),
                    "solver_success": opt_diag.get("solver_success"),
                    "solver_status": opt_diag.get("solver_status"),
                    "solver_iterations": opt_diag.get("solver_iterations"),
                    "fallback_reason": opt_diag.get("fallback_reason"),
                    "risk_contribution_std": opt_diag.get("risk_contribution_std"),
                    "post_constraint_l1_shift": weight_diag.get("post_constraint_l1_shift"),
                }
            )
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
    diag_summary = _summarize_portfolio_diagnostics(diag_detail, method=portfolio_method)
    return w_wide, diag_detail, diag_summary


def build_asset_returns(daily_df: pd.DataFrame, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
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


def build_market_ew_benchmark(daily_df: pd.DataFrame, start: str, end: str, min_days: int = 500) -> pd.Series:
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


def build_symbol_benchmark(daily_df: pd.DataFrame, symbol: str, start: str, end: str) -> pd.Series:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A 股回测评估（改进版）")
    p.add_argument(
        "--config",
        default="",
        help="配置文件路径；默认按 config.yaml.backtest -> config.yaml -> 内置默认 顺序查找",
    )
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期（默认取 config.paths.asof_trade_date）")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="标的最少历史交易日")
    p.add_argument(
        "--rebalance-rule",
        default="",
        help="覆盖 config.backtest.eval_rebalance_rule（如 M、2M、3M）；为空则用配置",
    )
    p.add_argument(
        "--json-report",
        default="",
        help="将回测摘要写入 JSON 文件（供文档/CI 使用）",
    )
    p.add_argument(
        "--no-regime",
        action="store_true",
        help="关闭 regime 动态调权，并跳过对照实验",
    )
    p.add_argument(
        "--industry-cap-count",
        type=int,
        default=None,
        help="行业持仓只数上限（默认读取 config.portfolio.industry_cap_count；0 表示不启用）",
    )
    p.add_argument(
        "--industry-map-csv",
        default="data/cache/industry_map.csv",
        help="symbol->industry 映射文件（CSV，需含 symbol/industry 列）",
    )
    p.add_argument(
        "--portfolio-method",
        default="",
        choices=(
            "",
            "equal_weight",
            "tiered_equal_weight",
            "two_tier_equal_weight",
            "risk_parity",
            "min_variance",
            "mean_variance",
        ),
        help="组合权重方法；为空则读取 config.portfolio.weight_method（兼容 equal）",
    )
    p.add_argument("--top-k", type=int, default=None, help="覆盖 config.signals.top_k")
    p.add_argument(
        "--sort-by",
        default="",
        choices=("", "composite_extended", "xgboost"),
        help="覆盖 config.signals.sort_by；当前回测入口支持 composite_extended 或 xgboost",
    )
    p.add_argument(
        "--tree-bundle-dir",
        default="",
        help="sort_by=xgboost 时覆盖 config.signals.tree_model.bundle_dir",
    )
    p.add_argument(
        "--tree-features",
        default="",
        help="sort_by=xgboost 时覆盖 config.signals.tree_model.features（逗号分隔）",
    )
    p.add_argument(
        "--tree-rsi-mode",
        default="",
        help="sort_by=xgboost 时覆盖 config.signals.tree_model.rsi_mode",
    )
    p.add_argument(
        "--tree-feature-group",
        default="",
        help="sort_by=xgboost 时记录树模型特征分组（如 G0/G1），仅用于结果追踪",
    )
    p.add_argument("--research-topic", default="", help="覆盖结果文件 research_topic")
    p.add_argument("--research-config-id", default="", help="覆盖结果文件 research_config_id")
    p.add_argument("--output-stem", default="", help="覆盖结果文件 output_stem")
    p.add_argument(
        "--canonical-config",
        default="",
        help="记录 canonical research config 快照（如 v3_market_ew_full_backtest）",
    )
    p.add_argument("--max-turnover", type=float, default=None, help="覆盖 config.portfolio.max_turnover")
    p.add_argument(
        "--entry-top-k",
        type=int,
        default=None,
        help="进入持仓时只允许从前 N 名买入；为空则读取 config.portfolio.entry_top_k",
    )
    p.add_argument(
        "--hold-buffer-top-k",
        type=int,
        default=None,
        help="旧持仓只要仍留在前 N 名就允许继续持有；为空则读取 config.portfolio.hold_buffer_top_k",
    )
    p.add_argument(
        "--top-tier-count",
        type=int,
        default=None,
        help="tiered_equal_weight 下前层持仓数；为空则读取 config.portfolio.top_tier_count",
    )
    p.add_argument(
        "--top-tier-weight-share",
        type=float,
        default=None,
        help="tiered_equal_weight 下前层权重占比；为空则读取 config.portfolio.top_tier_weight_share",
    )
    p.add_argument(
        "--ic-weights-json",
        default="",
        help="P2-A 历史动态权重 JSON（update_ic_weights.py 输出）",
    )
    p.add_argument(
        "--ic-monitor-path",
        default="",
        help="当未提供 --ic-weights-json 时，可从 ic_monitor.json 在线构建历史权重",
    )
    p.add_argument("--ic-window", type=int, default=None, help="IC 动态权重窗口")
    p.add_argument("--ic-min-obs", type=int, default=None, help="IC 动态权重最小样本")
    p.add_argument("--ic-half-life", type=float, default=None, help="IC 动态权重半衰期")
    p.add_argument("--ic-clip-abs-weight", type=float, default=None, help="IC 动态权重裁剪上限")
    p.add_argument(
        "--prepared-factors-cache",
        default="",
        help="prepared factors parquet 缓存路径；命中后跳过 compute_factors/PIT/LLM/universe 预处理",
    )
    p.add_argument(
        "--refresh-prepared-factors-cache",
        action="store_true",
        help="忽略已有 prepared factors 缓存并强制重建",
    )
    p.add_argument(
        "--prepare-factors-only",
        action="store_true",
        help="只构建并写出 prepared factors cache，然后退出；需配合 --prepared-factors-cache 使用",
    )
    p.add_argument("--grid-search", action="store_true", help="执行 Top-K/换手/调仓频率网格搜索")
    p.add_argument(
        "--grid-search-out",
        default="data/results/backtest_grid_search.csv",
        help="网格搜索结果输出 CSV",
    )
    p.add_argument("--grid-topk-values", default="10,20,30,40", help="网格 top_k 列表")
    p.add_argument("--grid-max-turnover-values", default="0.3,0.4,0.5", help="网格 max_turnover 列表")
    p.add_argument("--grid-rebalance-rules", default="M,2M,3M", help="网格调仓规则列表")
    p.add_argument("--wf-train-window", type=int, default=252, help="滚动 WF 训练窗交易日")
    p.add_argument("--wf-test-window", type=int, default=63, help="滚动 WF 测试窗交易日")
    p.add_argument(
        "--wf-step-window",
        type=int,
        default=0,
        help="滚动 WF 步长交易日（<=0 时默认等于 wf-test-window）",
    )
    p.add_argument("--wf-slice-splits", type=int, default=5, help="时间切片 WF 折数")
    p.add_argument("--wf-slice-min-train-days", type=int, default=252, help="时间切片每折最少训练天数")
    p.add_argument(
        "--wf-slice-fixed-window",
        action="store_true",
        help="时间切片 WF 使用固定训练窗（默认扩展窗）",
    )
    p.add_argument(
        "--execution-lag",
        type=int,
        default=None,
        help="close_to_close/vwap：相对信号的额外执行滞后天数（默认读 config.backtest.execution_lag 或 1；0=旧版）",
    )
    p.add_argument(
        "--cov-lookback-days",
        type=int,
        default=None,
        help="风险平价/均值方差协方差回看天数；默认读 config.portfolio.cov_lookback_days",
    )
    p.add_argument(
        "--factor-ic-report",
        default="",
        help="P1-B 因子 IC 汇总（diagnose_factor_ic.py 输出 csv/json）；启用后会按规则做删减/翻转",
    )
    p.add_argument(
        "--disable-p1-factor-filter",
        action="store_true",
        help="禁用 P1-B 基于 IC 的因子删减/翻转",
    )
    return p.parse_args()


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    cfg, config_source = load_config(args.config)

    end_date = args.end or str(cfg.get("paths", {}).get("asof_trade_date", ""))
    if not end_date:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = args.start

    db_path = str(PROJECT_ROOT / cfg["paths"]["duckdb_path"])
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    results_dir = PROJECT_ROOT / str(cfg.get("paths", {}).get("results_dir", "data/results"))

    signals = cfg.get("signals", {})
    backtest_cfg = cfg.get("backtest", {})
    prefilter_cfg = cfg.get("prefilter", {})
    portfolio_cfg = cfg.get("portfolio", {})
    regime_cfg_raw = cfg.get("regime", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    tree_sig = signals.get("tree_model", {}) or {}
    ce_weights = normalize_weights(signals.get("composite_extended", {}))
    p1_cfg = signals.get("p1_factor_filter", {}) or {}
    ic_cfg = signals.get("ic_weighting", {}) or {}
    top_k = int(signals.get("top_k", 10))
    if args.top_k is not None:
        top_k = int(args.top_k)
    sort_by = str(args.sort_by or signals.get("sort_by", "composite_extended")).lower().strip()
    if sort_by not in ("composite_extended", "xgboost"):
        raise SystemExit(f"run_backtest_eval 当前仅支持 sort_by=composite_extended|xgboost，收到 {sort_by!r}")
    tree_bundle_dir = str(args.tree_bundle_dir or tree_sig.get("bundle_dir", "")).strip()
    tree_raw_features = [
        str(x).strip()
        for x in (
            args.tree_features.split(",")
            if args.tree_features.strip()
            else (tree_sig.get("features", []) or [])
        )
        if str(x).strip()
    ]
    tree_rsi_mode = str(args.tree_rsi_mode or tree_sig.get("rsi_mode", "level")).strip().lower() or "level"
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    if str(args.rebalance_rule).strip():
        rebalance_rule = str(args.rebalance_rule).strip()
    max_turnover = float(portfolio_cfg.get("max_turnover", 1.0))
    if args.max_turnover is not None:
        max_turnover = float(args.max_turnover)
    entry_top_k = portfolio_cfg.get("entry_top_k")
    hold_buffer_top_k = portfolio_cfg.get("hold_buffer_top_k")
    top_tier_count = portfolio_cfg.get("top_tier_count")
    top_tier_weight_share = portfolio_cfg.get("top_tier_weight_share")
    if args.entry_top_k is not None:
        entry_top_k = int(args.entry_top_k)
    if args.hold_buffer_top_k is not None:
        hold_buffer_top_k = int(args.hold_buffer_top_k)
    if args.top_tier_count is not None:
        top_tier_count = int(args.top_tier_count)
    if args.top_tier_weight_share is not None:
        top_tier_weight_share = float(args.top_tier_weight_share)
    portfolio_method_cfg = str(portfolio_cfg.get("weight_method", "equal_weight")).lower().strip()
    portfolio_method = str(args.portfolio_method or portfolio_method_cfg or "equal_weight").lower().strip()
    if portfolio_method == "score":
        portfolio_method = "equal_weight"
    cov_lookback_days = int(portfolio_cfg.get("cov_lookback_days", 60))
    if args.cov_lookback_days is not None:
        cov_lookback_days = int(args.cov_lookback_days)
    cov_ridge = float(portfolio_cfg.get("cov_ridge", 1e-6))
    cov_shrinkage = str(portfolio_cfg.get("cov_shrinkage", "ledoit_wolf")).lower()
    cov_ewma_halflife = float(portfolio_cfg.get("cov_ewma_halflife", 20.0))
    risk_aversion = float(portfolio_cfg.get("risk_aversion", 1.0))
    regime_enabled = bool(regime_cfg_raw.get("enabled", True)) and not bool(args.no_regime)
    if sort_by != "composite_extended" and regime_enabled:
        print(f"[提示] sort_by={sort_by} 当前不接 regime 动态调权，本次已自动关闭。")
        regime_enabled = False
    benchmark_symbol = str(risk_cfg.get("benchmark_symbol", "510300"))
    industry_cap_default = int(portfolio_cfg.get("industry_cap_count", 5))
    industry_cap_raw = industry_cap_default if args.industry_cap_count is None else int(args.industry_cap_count)
    industry_cap_count, industry_map, industry_cap_status = resolve_industry_cap_and_map(
        industry_cap_raw,
        args.industry_map_csv,
    )
    wf_train_window = int(max(20, args.wf_train_window))
    wf_test_window = int(max(5, args.wf_test_window))
    wf_step_window = int(args.wf_step_window) if int(args.wf_step_window) > 0 else wf_test_window
    wf_slice_splits = int(max(2, args.wf_slice_splits))
    wf_slice_min_train_days = int(max(20, args.wf_slice_min_train_days))
    wf_slice_expanding = not bool(args.wf_slice_fixed_window)
    execution_lag = int(backtest_cfg.get("execution_lag", 1))
    if args.execution_lag is not None:
        execution_lag = int(args.execution_lag)
    execution_lag = max(0, execution_lag)
    p1_filter_enabled = bool(p1_cfg.get("enabled", False)) and not bool(args.disable_p1_factor_filter)
    p1_ic_report_path = str(args.factor_ic_report or p1_cfg.get("ic_report_path", "")).strip()
    p1_remove_both_neg = bool(p1_cfg.get("remove_if_t1_and_t21_negative", True))
    p1_zero_abs_t1 = float(p1_cfg.get("zero_if_abs_t1_below", 0.0))
    p1_flip_t21 = float(p1_cfg.get("flip_if_t1_negative_and_t21_above", 0.005))
    ic_weighting_enabled = bool(args.ic_weights_json or args.ic_monitor_path or ic_cfg.get("enabled", False))
    ic_weights_json_path = str(args.ic_weights_json or ic_cfg.get("weights_path", "")).strip()
    ic_monitor_path = str(args.ic_monitor_path or ic_cfg.get("monitor_path", "")).strip()
    ic_window = int(args.ic_window) if args.ic_window is not None else int(ic_cfg.get("window", 60))
    ic_min_obs = int(args.ic_min_obs) if args.ic_min_obs is not None else int(ic_cfg.get("min_obs", 20))
    ic_half_life = (
        float(args.ic_half_life) if args.ic_half_life is not None else float(ic_cfg.get("half_life", 20.0))
    )
    ic_clip_abs_weight = (
        float(args.ic_clip_abs_weight)
        if args.ic_clip_abs_weight is not None
        else float(ic_cfg.get("clip_abs_weight", 0.25))
    )
    if p1_filter_enabled and p1_ic_report_path:
        ic_summary = load_factor_ic_summary(p1_ic_report_path)
        if ic_summary.empty:
            print(f"[P1] 因子过滤已启用，但 IC 报告为空或不可读：{p1_ic_report_path}；保留静态权重。")
        else:
            ce_weights, p1_actions = apply_p1_factor_policy(
                ce_weights,
                ic_summary,
                remove_if_t1_and_t21_negative=p1_remove_both_neg,
                zero_if_abs_t1_below=p1_zero_abs_t1,
                flip_if_t1_negative_and_t21_above=p1_flip_t21,
            )
            if not p1_actions.empty:
                vc = p1_actions["action"].value_counts().to_dict()
                print(
                    "[P1] 因子 IC 规则已应用："
                    f" keep={int(vc.get('keep', 0))}"
                    f", zero={int(vc.get('zero', 0))}"
                    f", flip={int(vc.get('flip', 0))}"
                    f", remove={int(vc.get('remove', 0))}"
                )

    print("=" * 70)
    print("A 股回测评估（改进版）")
    print(
        f"区间: {start_date} ~ {end_date} | Top-{top_k} | 调仓: {rebalance_rule} | "
        f"max_turnover={max_turnover:.2f} | execution_mode={str(backtest_cfg.get('execution_mode', 'close_to_close'))}"
        f" | execution_lag={execution_lag}"
    )
    print(
        f"排序键约定: {sort_by} | regime={'on' if regime_enabled else 'off'}"
        f" | 行业上限(只数)={industry_cap_count if industry_cap_count > 0 else 'off'}"
        f" | 组合方法={portfolio_method}"
        f" | cov_lookback_days={cov_lookback_days}"
        f" | p1_factor_filter={'on' if p1_filter_enabled else 'off'}"
        f" | ic_weighting={'on' if ic_weighting_enabled else 'off'}"
    )
    if industry_cap_status == "disabled_missing_map":
        print(
            f"[P2] 行业约束已请求但未生效：缺少行业映射文件 {args.industry_map_csv}；"
            "本次回测已自动降级为关闭。"
        )
    print(
        "WF参数: "
        f"rolling(train={wf_train_window}, test={wf_test_window}, step={wf_step_window}) | "
        f"slices(n={wf_slice_splits}, min_train={wf_slice_min_train_days}, "
        f"expanding={'on' if wf_slice_expanding else 'off'})"
    )
    print("=" * 70)

    print("[1/7] 读取日线数据...")
    daily_df = load_daily_from_duckdb(db_path, start_date, end_date, args.lookback_days)
    print(f"  日线: {len(daily_df):,} 行, 标的: {daily_df['symbol'].nunique():,}")

    print("[2/7] 计算因子...")
    uf = cfg.get("universe_filter", {}) or {}
    prepared_factors_cache = _resolve_optional_path(args.prepared_factors_cache)
    if bool(args.prepare_factors_only) and prepared_factors_cache is None:
        raise SystemExit("--prepare-factors-only 需要配合 --prepared-factors-cache 使用")
    prepared_factors_cache_meta = _prepared_factors_cache_expected_meta(
        start_date=start_date,
        end_date=end_date,
        lookback_days=int(args.lookback_days),
        min_hist_days=int(args.min_hist_days),
        db_path=db_path,
        results_dir=str(results_dir),
        universe_filter_cfg=uf,
    )
    tree_feature_group = str(args.tree_feature_group or "").strip()
    identity_selector_parts: dict[str, Any] = {}
    if sort_by == "xgboost":
        identity_selector_parts["tree_group"] = tree_feature_group or "unknown"
    research_identity = build_full_backtest_research_identity(
        topic=args.research_topic or "full_backtest",
        output_prefix=args.output_stem or "full_backtest",
        sort_by=sort_by,
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
        portfolio_method=portfolio_method,
        execution_mode=str(backtest_cfg.get("execution_mode", "close_to_close")),
        prefilter_enabled=bool(prefilter_cfg.get("enabled", False)),
        universe_filter_enabled=bool(uf.get("enabled", False)),
        benchmark_symbol=MARKET_EW_PROXY,
        start_date=start_date,
        end_date=end_date,
        selector_parts=identity_selector_parts,
    )
    if str(args.research_config_id).strip():
        research_identity["research_config_id"] = slugify_token(args.research_config_id)
    if str(args.output_stem).strip() and str(args.research_config_id).strip():
        research_identity["output_stem"] = slugify_token(args.output_stem)
    canonical_config_snapshot: dict[str, Any] = {}
    if str(args.canonical_config).strip():
        canonical_config_snapshot = canonical_research_config(args.canonical_config)
    tree_report_meta = _tree_bundle_report_meta(
        sort_by=sort_by,
        bundle_dir=tree_bundle_dir,
        tree_feature_group=tree_feature_group,
        requested_features=tree_raw_features,
    )
    factors, factors_cache_hit = prepare_factors_for_backtest(
        daily_df,
        min_hist_days=int(args.min_hist_days),
        db_path=db_path,
        results_dir=results_dir,
        universe_filter_cfg=uf,
        cache_path=prepared_factors_cache,
        refresh_cache=bool(args.refresh_prepared_factors_cache),
        cache_meta=prepared_factors_cache_meta,
    )
    if prepared_factors_cache is not None:
        cache_state = "hit" if factors_cache_hit else "rebuilt"
        print(f"  prepared_factors_cache: {cache_state} -> {prepared_factors_cache}")
    print(f"  因子长表: {len(factors):,} 行")
    if bool(uf.get("enabled", False)):
        ne = int(factors["_universe_eligible"].sum())
        print(f"  M2.4 universe 过滤: 合格截面行 {ne:,} / {len(factors):,}")
    if bool(args.prepare_factors_only):
        print("-" * 70)
        print("prepared factors cache 已完成，按参数要求提前退出。")
        print("完成。")
        return

    print("[3/7] 截面打分...")
    regime_overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    regime_trace = pd.DataFrame(columns=["trade_date", "regime", "short_return", "vol_ann"])
    if regime_enabled:
        regime_overrides, regime_trace = build_regime_weight_overrides(
            factors,
            daily_df,
            ce_weights,
            benchmark_symbol=benchmark_symbol,
            regime_cfg_raw=regime_cfg_raw,
            market_ew_min_days=int(args.min_hist_days),
        )
    ic_overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    if ic_weighting_enabled and ic_weights_json_path:
        ic_overrides = load_ic_weights_by_date(ic_weights_json_path)
    elif ic_weighting_enabled and ic_monitor_path:
        ic_overrides = build_ic_weights_from_monitor(
            ic_monitor_path,
            window=ic_window,
            min_obs=ic_min_obs,
            half_life=ic_half_life,
            clip_abs_weight=ic_clip_abs_weight,
        )
    if ic_overrides:
        print(f"  P2 IC 动态权重: 已加载 {len(ic_overrides)} 个日期快照")
    merged_overrides: Dict[pd.Timestamp, Dict[str, float]] = {}
    base_dates = set(regime_overrides.keys()) | set(ic_overrides.keys())
    for dt in sorted(base_dates):
        merged = dict(ce_weights)
        if dt in regime_overrides:
            merged.update(regime_overrides[dt])
        if dt in ic_overrides:
            merged.update(ic_overrides[dt])
        merged_overrides[pd.Timestamp(dt)] = merged
    if sort_by == "xgboost":
        if not tree_bundle_dir:
            raise SystemExit("sort_by=xgboost 需要 --tree-bundle-dir 或 config.signals.tree_model.bundle_dir")
        if not tree_raw_features:
            raise SystemExit("sort_by=xgboost 需要 --tree-features 或 config.signals.tree_model.features")
    score_df = build_score(
        factors,
        ce_weights,
        weights_by_date=merged_overrides if merged_overrides else (regime_overrides if regime_enabled else None),
        sort_by=sort_by,
        tree_bundle_dir=tree_bundle_dir or None,
        tree_raw_features=tree_raw_features,
        tree_rsi_mode=tree_rsi_mode,
    )
    print(f"  得分行数: {len(score_df):,}")

    print("[4/7] 构建权重...")
    weights, portfolio_diag_detail, portfolio_diag_summary = build_topk_weights(
        score_df=score_df,
        factor_df=factors,
        daily_df=daily_df,
        top_k=top_k,
        rebalance_rule=rebalance_rule,
        prefilter_cfg=prefilter_cfg,
        max_turnover=max_turnover,
        entry_top_k=entry_top_k,
        hold_buffer_top_k=hold_buffer_top_k,
        top_tier_count=top_tier_count,
        top_tier_weight_share=top_tier_weight_share,
        industry_map=industry_map,
        industry_cap_count=industry_cap_count,
        portfolio_method=portfolio_method,
        cov_lookback_days=cov_lookback_days,
        cov_ridge=cov_ridge,
        cov_shrinkage=cov_shrinkage,
        cov_ewma_halflife=cov_ewma_halflife,
        risk_aversion=risk_aversion,
        return_details=True,
    )
    weights = weights[weights.index >= pd.Timestamp(start_date)]
    print(f"  调仓日: {len(weights)} | 标的列: {weights.shape[1]}")
    weights_base: pd.DataFrame | None = None
    if regime_enabled and sort_by == "composite_extended":
        score_base = build_score(factors, ce_weights, weights_by_date=None, sort_by=sort_by)
        weights_base = build_topk_weights(
            score_df=score_base,
            factor_df=factors,
            daily_df=daily_df,
            top_k=top_k,
            rebalance_rule=rebalance_rule,
            prefilter_cfg=prefilter_cfg,
            max_turnover=max_turnover,
            entry_top_k=entry_top_k,
            hold_buffer_top_k=hold_buffer_top_k,
            top_tier_count=top_tier_count,
            top_tier_weight_share=top_tier_weight_share,
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=portfolio_method,
            cov_lookback_days=cov_lookback_days,
            cov_ridge=cov_ridge,
            cov_shrinkage=cov_shrinkage,
            cov_ewma_halflife=cov_ewma_halflife,
            risk_aversion=risk_aversion,
        )
        weights_base = weights_base[weights_base.index >= pd.Timestamp(start_date)]

    print("[5/7] 回测执行...")
    execution_mode = str(backtest_cfg.get("execution_mode", "close_to_close")).lower().strip()
    if execution_mode not in ("close_to_close", "tplus1_open", "vwap"):
        print(f"[警告] 未识别 execution_mode={execution_mode}，回退为 close_to_close")
        execution_mode = "close_to_close"

    symbol_universe = set(str(c).zfill(6) for c in weights.columns)
    if weights_base is not None and not weights_base.empty:
        symbol_universe.update(str(c).zfill(6) for c in weights_base.columns)
    target_cols = sorted(symbol_universe)

    if execution_mode == "tplus1_open":
        open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
        open_returns = open_returns.sort_index()
        open_returns.index = pd.to_datetime(open_returns.index)
        open_returns = open_returns.reindex(columns=target_cols).fillna(0.0)
        asset_returns = open_returns[
            (open_returns.index >= pd.Timestamp(start_date)) & (open_returns.index <= pd.Timestamp(end_date))
        ]
    else:
        asset_returns = build_asset_returns(daily_df, target_cols, start_date, end_date)
    weights = weights.reindex(columns=target_cols, fill_value=0.0)
    if weights_base is not None and not weights_base.empty:
        weights_base = weights_base.reindex(columns=target_cols, fill_value=0.0)
    first_reb = weights.index.min()
    if weights_base is not None and not weights_base.empty:
        first_reb = min(first_reb, weights_base.index.min())
    asset_returns = asset_returns[asset_returns.index >= first_reb]
    if not asset_returns.empty and not weights.empty and weights.index.min() > asset_returns.index.min():
        seed = weights.iloc[[0]].copy()
        seed.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights = pd.concat([seed, weights], axis=0)
        weights = weights[~weights.index.duplicated(keep="last")].sort_index()
    if (
        weights_base is not None
        and not weights_base.empty
        and not asset_returns.empty
        and weights_base.index.min() > asset_returns.index.min()
    ):
        seed_base = weights_base.iloc[[0]].copy()
        seed_base.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights_base = pd.concat([seed_base, weights_base], axis=0)
        weights_base = weights_base[~weights_base.index.duplicated(keep="last")].sort_index()

    bt_no_cost = BacktestConfig(
        cost_params=None,
        execution_mode=execution_mode,
        execution_lag=execution_lag,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
        vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
        vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
    )
    bt_cost = BacktestConfig(
        cost_params=costs,
        execution_mode=execution_mode,
        execution_lag=execution_lag,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
        vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
        vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
    )
    res_nc = run_backtest(asset_returns, weights, config=bt_no_cost)
    res_wc = run_backtest(asset_returns, weights, config=bt_cost)
    regime_control: dict[str, Any] | None = None
    if weights_base is not None and not weights_base.empty:
        res_wc_base = run_backtest(asset_returns, weights_base, config=bt_cost)
        regime_control = {
            "enabled_run": res_wc.panel.to_dict(),
            "disabled_run": res_wc_base.panel.to_dict(),
            "delta": {
                "annualized_return": float(res_wc.panel.annualized_return - res_wc_base.panel.annualized_return),
                "sharpe_ratio": float(res_wc.panel.sharpe_ratio - res_wc_base.panel.sharpe_ratio),
                "max_drawdown": float(res_wc.panel.max_drawdown - res_wc_base.panel.max_drawdown),
            },
        }

    print("[6/7] 基准与分年统计...")
    n_trade_days = int(asset_returns.index.nunique())
    adaptive_min_days = max(60, int(0.35 * max(n_trade_days, 1)))
    benchmark_series: dict[str, pd.Series] = {
        "hs300_510300": build_symbol_benchmark(daily_df, "510300", start_date, end_date),
        "csi500_510500": build_symbol_benchmark(daily_df, "510500", start_date, end_date),
        "market_ew": build_market_ew_benchmark(daily_df, start_date, end_date, min_days=adaptive_min_days),
    }
    benchmark_panels: dict[str, dict] = {}
    excess_panels: dict[str, dict] = {}
    benchmark_yearly: dict[str, pd.Series] = {}
    strat_all = res_wc.daily_returns.sort_index()
    for name, bench_s in benchmark_series.items():
        common = strat_all.index.intersection(bench_s.index)
        strat = strat_all.reindex(common).fillna(0.0)
        bench = bench_s.reindex(common).fillna(0.0)
        if len(common) > 0:
            bench_panel = compute_performance_panel(bench.to_numpy(), periods_per_year=252.0)
            excess_panel = compute_performance_panel((strat - bench).to_numpy(), periods_per_year=252.0)
            benchmark_yearly[name] = bench.groupby(bench.index.year).apply(lambda r: (1 + r).prod() - 1)
        else:
            bench_panel = compute_performance_panel(np.array([], dtype=np.float64), periods_per_year=252.0)
            excess_panel = compute_performance_panel(np.array([], dtype=np.float64), periods_per_year=252.0)
            benchmark_yearly[name] = pd.Series(dtype=np.float64)
        benchmark_panels[name] = bench_panel.to_dict()
        excess_panels[name] = excess_panel.to_dict()

    print("[7/7] Walk-Forward OOS...")
    rolling = rolling_walk_forward_windows(
        asset_returns.index,
        train_days=wf_train_window,
        test_days=wf_test_window,
        step_days=wf_step_window,
    )
    _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cost, use_test_only=True)
    slices = contiguous_time_splits(
        asset_returns.index,
        n_splits=wf_slice_splits,
        min_train_days=wf_slice_min_train_days,
        expanding_window=wf_slice_expanding,
    )
    _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cost, use_test_only=True)
    overfit_cmp = compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}
    market_benchmark = benchmark_series.get("market_ew", pd.Series(dtype=np.float64))
    rolling_excess = summarize_oos_excess_returns(res_wc.daily_returns, market_benchmark, rolling)
    slice_excess = summarize_oos_excess_returns(res_wc.daily_returns, market_benchmark, slices)
    wf_agg["median_ann_excess_vs_market"] = float(rolling_excess.get("median_ann_excess_return", np.nan))
    sp_agg["median_ann_excess_vs_market"] = float(slice_excess.get("median_ann_excess_return", np.nan))

    print("\n" + "-" * 70)
    print("全样本绩效")
    print(
        f"无成本: 年化 {fmt_pct(res_nc.panel.annualized_return)} | 夏普 {fmt_num(res_nc.panel.sharpe_ratio)} | "
        f"最大回撤 {fmt_pct(res_nc.panel.max_drawdown)} | 半L1换手 {fmt_pct(res_nc.panel.turnover_mean)}"
    )
    print(
        f"含成本: 年化 {fmt_pct(res_wc.panel.annualized_return)} | 夏普 {fmt_num(res_wc.panel.sharpe_ratio)} | "
        f"最大回撤 {fmt_pct(res_wc.panel.max_drawdown)} | 半L1换手 {fmt_pct(res_wc.panel.turnover_mean)}"
    )
    if portfolio_method not in ("", "equal", "equal_weight"):
        print(
            "组合优化诊断: "
            f"mean L1(diff vs equal)={fmt_num(_safe_float_or_nan(portfolio_diag_summary.get('mean_l1_diff_vs_equal')))} | "
            f"equal-like ratio={fmt_pct(_safe_float_or_nan(portfolio_diag_summary.get('equal_like_ratio')))} | "
            f"median cond={fmt_num(_safe_float_or_nan(portfolio_diag_summary.get('median_condition_number')))}"
        )
    for name in ("hs300_510300", "csi500_510500", "market_ew"):
        bp = benchmark_panels.get(name, {})
        ep = excess_panels.get(name, {})
        print(
            f"{name}: 年化 {fmt_pct(float(bp.get('annualized_return', np.nan)))} | "
            f"夏普 {fmt_num(float(bp.get('sharpe_ratio', np.nan)))} | "
            f"最大回撤 {fmt_pct(float(bp.get('max_drawdown', np.nan)))}"
        )
        print(
            f"策略超额(含成本) vs {name}: 年化 {fmt_pct(float(ep.get('annualized_return', np.nan)))} | "
            f"夏普 {fmt_num(float(ep.get('sharpe_ratio', np.nan)))} | "
            f"最大回撤 {fmt_pct(float(ep.get('max_drawdown', np.nan)))}"
        )

    yearly = res_wc.daily_returns.groupby(res_wc.daily_returns.index.year).apply(lambda r: (1 + r).prod() - 1)
    yearly_mkt = benchmark_yearly.get("market_ew", pd.Series(dtype=float))
    print("\n年度表现（含成本 vs 全市场等权）")
    print(f"{'年份':>6} {'策略':>12} {'市场':>12} {'超额':>12}")
    for y in yearly.index:
        mv = yearly_mkt.get(y, np.nan)
        ev = yearly[y] - mv if np.isfinite(mv) else np.nan
        print(f"{y:>6} {fmt_pct(yearly[y]):>12} {fmt_pct(mv):>12} {fmt_pct(ev):>12}")

    print("\nWalk-Forward（滚动窗口）")
    if wf_detail.empty:
        print("  无有效折次")
    else:
        print(
            f"  折数 {len(wf_detail)} | OOS 年化均值 {fmt_pct(wf_agg.get('annualized_return_agg', np.nan))} | "
            f"OOS 夏普均值 {fmt_num(wf_agg.get('sharpe_ratio_agg', np.nan))}"
        )
        print(
            f"  OOS 年化分位: p25 {fmt_pct(wf_agg.get('p25_ann_return', np.nan))} | "
            f"中位数 {fmt_pct(wf_agg.get('median_ann_return', np.nan))} | "
            f"p75 {fmt_pct(wf_agg.get('p75_ann_return', np.nan))}"
        )

    print("Walk-Forward（时间切片）")
    if sp_detail.empty:
        print("  无有效折次")
    else:
        print(
            f"  折数 {len(sp_detail)} | OOS 年化均值 {fmt_pct(sp_agg.get('annualized_return_agg', np.nan))} | "
            f"OOS 夏普均值 {fmt_num(sp_agg.get('sharpe_ratio_agg', np.nan))}"
        )
        print(
            f"  OOS 年化分位: p25 {fmt_pct(sp_agg.get('p25_ann_return', np.nan))} | "
            f"中位数 {fmt_pct(sp_agg.get('median_ann_return', np.nan))} | "
            f"p75 {fmt_pct(sp_agg.get('p75_ann_return', np.nan))}"
        )
        d_sharpe = overfit_cmp.get("delta_sharpe_ratio", np.nan)
        d_ann = overfit_cmp.get("delta_annualized_return", np.nan)
        print(f"  全样本-切片差值: 年化 {fmt_pct(d_ann)} | 夏普 {fmt_num(d_sharpe)}")

    if regime_control is not None:
        delta = regime_control.get("delta", {})
        print("\nRegime 对照（含成本，全样本）")
        print(
            f"  动态调权-关闭调权: 年化 {fmt_pct(float(delta.get('annualized_return', np.nan)))} | "
            f"夏普 {fmt_num(float(delta.get('sharpe_ratio', np.nan)))} | "
            f"最大回撤 {fmt_pct(float(delta.get('max_drawdown', np.nan)))}"
        )

    grid_df = pd.DataFrame()
    if args.grid_search:
        print("\n网格搜索（Top-K / max_turnover / rebalance_rule）...")
        topk_vals = [int(x) for x in str(args.grid_topk_values).split(",") if str(x).strip()]
        mt_vals = [float(x) for x in str(args.grid_max_turnover_values).split(",") if str(x).strip()]
        rb_vals = [str(x).strip() for x in str(args.grid_rebalance_rules).split(",") if str(x).strip()]
        rows: list[dict[str, Any]] = []
        for tk in topk_vals:
            for mt in mt_vals:
                for rb in rb_vals:
                    try:
                        gw = build_topk_weights(
                            score_df=score_df,
                            factor_df=factors,
                            daily_df=daily_df,
                            top_k=int(tk),
                            rebalance_rule=str(rb),
                            prefilter_cfg=prefilter_cfg,
                            max_turnover=float(mt),
                            entry_top_k=entry_top_k,
                            hold_buffer_top_k=hold_buffer_top_k,
                            top_tier_count=top_tier_count,
                            top_tier_weight_share=top_tier_weight_share,
                            industry_map=industry_map,
                            industry_cap_count=industry_cap_count,
                            portfolio_method=portfolio_method,
                            cov_lookback_days=cov_lookback_days,
                            cov_ridge=cov_ridge,
                            cov_shrinkage=cov_shrinkage,
                            cov_ewma_halflife=cov_ewma_halflife,
                            risk_aversion=risk_aversion,
                        )
                        gw = gw.reindex(columns=target_cols, fill_value=0.0)
                        if not asset_returns.empty and gw.index.min() > asset_returns.index.min():
                            seed_gw = gw.iloc[[0]].copy()
                            seed_gw.index = pd.DatetimeIndex([asset_returns.index.min()])
                            gw = pd.concat([seed_gw, gw], axis=0)
                            gw = gw[~gw.index.duplicated(keep="last")].sort_index()
                        gr = run_backtest(asset_returns, gw, config=bt_cost)
                        rows.append(
                            {
                                "top_k": int(tk),
                                "max_turnover": float(mt),
                                "rebalance_rule": str(rb),
                                "portfolio_method": portfolio_method,
                                "annualized_return": float(gr.panel.annualized_return),
                                "sharpe_ratio": float(gr.panel.sharpe_ratio),
                                "calmar_ratio": float(gr.panel.calmar_ratio),
                                "max_drawdown": float(gr.panel.max_drawdown),
                                "turnover_mean": float(gr.panel.turnover_mean),
                            }
                        )
                    except Exception as e_grid:  # noqa: BLE001
                        rows.append(
                            {
                                "top_k": int(tk),
                                "max_turnover": float(mt),
                                "rebalance_rule": str(rb),
                                "portfolio_method": portfolio_method,
                                "error": str(e_grid),
                            }
                        )
        if rows:
            grid_df = pd.DataFrame(rows)
            valid = grid_df[grid_df["sharpe_ratio"].notna()].copy() if "sharpe_ratio" in grid_df.columns else pd.DataFrame()
            if not valid.empty:
                valid = valid.sort_values(["sharpe_ratio", "calmar_ratio"], ascending=False)
                best = valid.iloc[0]
                print(
                    "  最优参数: "
                    f"top_k={int(best['top_k'])}, max_turnover={float(best['max_turnover']):.2f}, "
                    f"rebalance_rule={best['rebalance_rule']} | 夏普={fmt_num(float(best['sharpe_ratio']))}, "
                    f"Calmar={fmt_num(float(best['calmar_ratio']))}"
                )
            out_csv = Path(args.grid_search_out)
            if not out_csv.is_absolute():
                out_csv = PROJECT_ROOT / out_csv
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"  网格结果已写入: {out_csv}")

    # P4-C：每次回测运行统一记录实验参数与关键指标（JSONL + CSV）。
    try:
        experiments_dir = cfg.get("paths", {}).get("experiments_dir", "data/experiments")
        if not Path(experiments_dir).is_absolute():
            experiments_dir = PROJECT_ROOT / str(experiments_dir)
        exp_metrics: Dict[str, Any] = {
            "annualized_return_with_cost": float(res_wc.panel.annualized_return),
            "sharpe_with_cost": float(res_wc.panel.sharpe_ratio),
            "max_drawdown_with_cost": float(res_wc.panel.max_drawdown),
            "turnover_mean_with_cost": float(res_wc.panel.turnover_mean),
            "annualized_return_no_cost": float(res_nc.panel.annualized_return),
            "sharpe_no_cost": float(res_nc.panel.sharpe_ratio),
            "max_drawdown_no_cost": float(res_nc.panel.max_drawdown),
            "wf_rolling_ann_mean": float(wf_agg.get("annualized_return_agg", np.nan)),
            "wf_rolling_ann_median": float(wf_agg.get("median_ann_return", np.nan)),
            "wf_rolling_ann_p25": float(wf_agg.get("p25_ann_return", np.nan)),
            "wf_rolling_ann_p75": float(wf_agg.get("p75_ann_return", np.nan)),
            "wf_slices_ann_mean": float(sp_agg.get("annualized_return_agg", np.nan)),
            "wf_slices_ann_median": float(sp_agg.get("median_ann_return", np.nan)),
            "wf_slices_ann_p25": float(sp_agg.get("p25_ann_return", np.nan)),
            "wf_slices_ann_p75": float(sp_agg.get("p75_ann_return", np.nan)),
            "excess_market_ew_ann": float(excess_panels.get("market_ew", {}).get("annualized_return", np.nan)),
            "excess_market_ew_sharpe": float(excess_panels.get("market_ew", {}).get("sharpe_ratio", np.nan)),
            "portfolio_diag_mean_l1_diff_vs_equal": float(
                portfolio_diag_summary.get("mean_l1_diff_vs_equal", np.nan)
            ),
            "portfolio_diag_equal_like_ratio": float(portfolio_diag_summary.get("equal_like_ratio", np.nan)),
        }
        exp_params: Dict[str, Any] = {
            "result_type": research_identity["result_type"],
            "research_topic": research_identity["research_topic"],
            "research_config_id": research_identity["research_config_id"],
            "output_stem": research_identity["output_stem"],
            "canonical_config": str(args.canonical_config or ""),
            "canonical_config_snapshot": canonical_config_snapshot,
            "start": start_date,
            "end": end_date,
            "sort_by": sort_by,
            "top_k": top_k,
            "rebalance_rule": rebalance_rule,
            "max_turnover": max_turnover,
            "entry_top_k": entry_top_k,
            "hold_buffer_top_k": hold_buffer_top_k,
            "top_tier_count": top_tier_count,
            "top_tier_weight_share": top_tier_weight_share,
            "portfolio_method": portfolio_method,
            "execution_mode": execution_mode,
            "execution_lag": execution_lag,
            "regime_enabled": regime_enabled,
            "industry_cap_count": industry_cap_count,
            "industry_cap_requested": industry_cap_raw,
            "industry_cap_status": industry_cap_status,
            "industry_map_csv": str(args.industry_map_csv),
            "ic_weighting_enabled": ic_weighting_enabled,
            "ic_weights_json": ic_weights_json_path,
            "ic_monitor_path": ic_monitor_path,
            "ic_window": ic_window,
            "ic_min_obs": ic_min_obs,
            "ic_half_life": ic_half_life,
            "ic_clip_abs_weight": ic_clip_abs_weight,
            "prepared_factors_cache": str(prepared_factors_cache) if prepared_factors_cache is not None else "",
            "refresh_prepared_factors_cache": bool(args.refresh_prepared_factors_cache),
            "prepared_factors_cache_hit": bool(factors_cache_hit),
            "prepared_factors_cache_meta": prepared_factors_cache_meta,
            "wf_train_window": wf_train_window,
            "wf_test_window": wf_test_window,
            "wf_step_window": wf_step_window,
            "wf_slice_splits": wf_slice_splits,
            "wf_slice_min_train_days": wf_slice_min_train_days,
            "wf_slice_expanding_window": wf_slice_expanding,
            "prefilter": prefilter_cfg,
            "tree_bundle_dir": tree_bundle_dir,
            "tree_features": tree_raw_features,
            "tree_feature_group": tree_feature_group,
            "tree_report_meta": tree_report_meta,
            "tree_rsi_mode": tree_rsi_mode,
            "p1_factor_filter_enabled": p1_filter_enabled,
            "p1_factor_ic_report": p1_ic_report_path,
            "p1_zero_if_abs_t1_below": p1_zero_abs_t1,
            "p1_flip_if_t1_negative_and_t21_above": p1_flip_t21,
        }
        rec_paths = append_backtest_result(
            base_dir=experiments_dir,
            params=_json_sanitize(exp_params),
            metrics=_json_sanitize(exp_metrics),
            duration_sec=float(time.perf_counter() - t0),
            bundle_dir=args.json_report or "run_backtest_eval",
            extra={"config_source": config_source},
        )
        print(f"实验记录已写入: {rec_paths['jsonl']} | {rec_paths['csv']}")
    except Exception as e_exp:  # noqa: BLE001
        print(f"[提示] 写入 experiments 失败（不影响主流程）: {e_exp}")

    if args.json_report:
        cost_drag_ann = float(res_wc.panel.annualized_return) - float(res_nc.panel.annualized_return)
        yearly_rows = []
        for y in yearly.index:
            mv = yearly_mkt.get(y, np.nan)
            yearly_rows.append(
                {
                    "year": int(y),
                    "strategy": float(yearly[y]),
                    "market": float(mv) if np.isfinite(mv) else None,
                    "excess": float(yearly[y] - mv) if np.isfinite(mv) else None,
                }
            )
        report = {
            "generated_at": date.today().isoformat(),
            "result_type": research_identity["result_type"],
            "research_topic": research_identity["research_topic"],
            "research_config_id": research_identity["research_config_id"],
            "output_stem": research_identity["output_stem"],
            "config_source": config_source,
            "canonical_config": str(args.canonical_config or ""),
            "canonical_config_snapshot": canonical_config_snapshot,
            "parameters": {
                "start": start_date,
                "end": end_date,
                "sort_by": sort_by,
                "lookback_days": args.lookback_days,
                "min_hist_days": args.min_hist_days,
                "top_k": top_k,
                "rebalance_rule": rebalance_rule,
                "max_turnover": max_turnover,
                "entry_top_k": entry_top_k,
                "hold_buffer_top_k": hold_buffer_top_k,
                "top_tier_count": top_tier_count,
                "top_tier_weight_share": top_tier_weight_share,
                "portfolio_method": portfolio_method,
                "cov_lookback_days": cov_lookback_days,
                "cov_ridge": cov_ridge,
                "cov_shrinkage": cov_shrinkage,
                "cov_ewma_halflife": cov_ewma_halflife,
                "risk_aversion": risk_aversion,
                "execution_mode": execution_mode,
                "execution_lag": execution_lag,
                "regime_enabled": regime_enabled,
                "benchmark_symbol": benchmark_symbol,
                "industry_cap_count": industry_cap_count,
                "industry_cap_requested": industry_cap_raw,
                "industry_cap_status": industry_cap_status,
                "industry_map_csv": str(args.industry_map_csv),
                "ic_weighting_enabled": ic_weighting_enabled,
                "ic_weights_json": ic_weights_json_path,
                "ic_monitor_path": ic_monitor_path,
                "ic_window": ic_window,
                "ic_min_obs": ic_min_obs,
                "ic_half_life": ic_half_life,
                "ic_clip_abs_weight": ic_clip_abs_weight,
                "prepared_factors_cache": str(prepared_factors_cache) if prepared_factors_cache is not None else "",
                "refresh_prepared_factors_cache": bool(args.refresh_prepared_factors_cache),
                "prepared_factors_cache_hit": bool(factors_cache_hit),
                "prepared_factors_cache_meta": prepared_factors_cache_meta,
                "tree_bundle_dir": tree_bundle_dir,
                "tree_features": tree_raw_features,
                "tree_feature_group": tree_feature_group,
                "tree_rsi_mode": tree_rsi_mode,
                "composite_extended_weights": ce_weights,
                "prefilter": prefilter_cfg,
                "p1_factor_filter_enabled": p1_filter_enabled,
                "p1_factor_ic_report": p1_ic_report_path,
                "p1_zero_if_abs_t1_below": p1_zero_abs_t1,
                "p1_flip_if_t1_negative_and_t21_above": p1_flip_t21,
                "benchmark_min_history_days": adaptive_min_days,
                "wf_train_window": wf_train_window,
                "wf_test_window": wf_test_window,
                "wf_step_window": wf_step_window,
                "wf_slice_splits": wf_slice_splits,
                "wf_slice_min_train_days": wf_slice_min_train_days,
                "wf_slice_expanding_window": wf_slice_expanding,
            },
            "meta": {
                "n_trading_days": int(res_wc.panel.n_periods),
                "n_rebalances": int(res_wc.meta.get("n_rebalances", 0)),
                "n_weight_symbols": int(weights.shape[1]),
                "industry_cap_status": industry_cap_status,
                "portfolio_diagnostics_summary": _json_sanitize(portfolio_diag_summary),
                "prepared_factors_cache": {
                    "path": str(prepared_factors_cache) if prepared_factors_cache is not None else "",
                    "hit": bool(factors_cache_hit),
                    "meta": prepared_factors_cache_meta,
                    "schema_version": prepared_factors_cache_meta.get("prepared_factors_schema_version"),
                    "cache_format_version": prepared_factors_cache_meta.get("cache_format_version"),
                },
                "tree_model": tree_report_meta,
            },
            "full_sample": {
                "no_cost": res_nc.panel.to_dict(),
                "with_cost": res_wc.panel.to_dict(),
                "benchmarks": benchmark_panels,
                "excess_vs_benchmarks": excess_panels,
                "market_ew": benchmark_panels.get("market_ew", {}),
                "excess_vs_market": excess_panels.get("market_ew", {}),
                "cost_drag_annualized": cost_drag_ann,
            },
            "yearly": yearly_rows,
            "walk_forward_rolling": {
                "detail": _json_sanitize(wf_detail.to_dict(orient="records"))
                if not wf_detail.empty
                else [],
                "agg": _json_sanitize(dict(wf_agg)),
                "excess_vs_market": _json_sanitize(rolling_excess),
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records"))
                if not sp_detail.empty
                else [],
                "agg": _json_sanitize(dict(sp_agg)),
                "excess_vs_market": _json_sanitize(slice_excess),
                "full_vs_slices": _json_sanitize(overfit_cmp),
            },
            "regime": {
                "enabled": regime_enabled,
                "trace": _json_sanitize(regime_trace.to_dict(orient="records"))
                if not regime_trace.empty
                else [],
                "control": _json_sanitize(regime_control) if regime_control is not None else None,
            },
            "portfolio_diagnostics": {
                "summary": _json_sanitize(portfolio_diag_summary),
                "detail": _json_sanitize(portfolio_diag_detail.to_dict(orient="records"))
                if not portfolio_diag_detail.empty
                else [],
            },
            "grid_search": _json_sanitize(grid_df.to_dict(orient="records")) if not grid_df.empty else [],
        }
        out_path = Path(args.json_report).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_sanitize(report), f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print("完成。")


if __name__ == "__main__":
    main()
