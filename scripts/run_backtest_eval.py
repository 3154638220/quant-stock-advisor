"""
A 股量化回测评估脚本（改进版）
================================

改进点（对应 docs/backtest_report.md）：
1) 因子方向纠偏：从 config.yaml 读取 composite_extended 权重，支持反转方向。
2) 预过滤：限制近 5 日涨跌停次数、极端换手、绝对高位。
3) 低换手约束：通过持仓重叠率约束等权 Top-K 的半 L1 换手。
4) 回测执行：统一 close_to_close 口径，输出无成本/含成本对比。
5) 样本外验证：滚动窗口 + 时间切片 Walk-Forward。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable

import duckdb
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, build_open_to_open_returns, run_backtest
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.transaction_costs import transaction_cost_params_from_mapping
from src.backtest.walk_forward import (
    compare_full_vs_slices,
    contiguous_time_splits,
    rolling_walk_forward_windows,
    walk_forward_backtest,
)

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
        "top_k": 50,
        "composite_extended": {
            "momentum": -0.15,
            "rsi": -0.15,
            "realized_vol": 0.03,
            "turnover_roll_mean": 0.05,
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
        "max_turnover": 0.5,
    },
    "backtest": {
        "execution_mode": "close_to_close",
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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        print("[提示] 未找到 config.yaml，使用脚本内置默认配置。")
        return dict(DEFAULT_CONFIG)
    with open(cfg_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    s = sum(abs(v) for v in cleaned.values())
    if s <= 0:
        raise ValueError("composite_extended 权重和为 0")
    return {k: v / s for k, v in cleaned.items()}


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

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_l = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rsi = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))

        daily_ret = c.pct_change()
        realized_vol = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
        turnover_roll_mean = t.rolling(20, min_periods=10).mean()
        vol_ret_corr = v.rolling(20, min_periods=10).corr(daily_ret).fillna(0.0)
        vol_to_turnover = np.log1p(v / t.replace(0, np.nan)).fillna(0.0)

        ma20 = c.rolling(20, min_periods=10).mean()
        ma60 = c.rolling(60, min_periods=30).mean()
        bias_short = (c - ma20) / ma20.replace(0, np.nan)
        bias_long = (c - ma60) / ma60.replace(0, np.nan)
        max_single_day_drop = pcg.rolling(20, min_periods=5).min().abs()
        recent_return = c / c.shift(3) - 1.0

        hi250 = h.rolling(250, min_periods=60).max()
        lo250 = lo.rolling(250, min_periods=60).min()
        price_position = ((c - lo250) / (hi250 - lo250).replace(0, np.nan)).clip(0, 1)

        intraday_range = (h - lo) / c.replace(0, np.nan)
        candle_range = (h - lo).replace(0, np.nan)
        upper_shadow_ratio = (h - np.maximum(c, o)) / candle_range
        upper_shadow_ratio = upper_shadow_ratio.fillna(0).clip(0, 1)
        lower_shadow_ratio = (np.minimum(c, o) - lo) / candle_range
        lower_shadow_ratio = lower_shadow_ratio.fillna(0).clip(0, 1)
        close_open_return = (c - o) / o.replace(0, np.nan)
        tail_strength = close_open_return.rolling(10, min_periods=5).mean()

        if sym.startswith(("300", "688")):
            lim = 0.20
        elif sym.startswith(("8", "4")):
            lim = 0.30
        else:
            lim = 0.10
        limit_move_hits_5d = (pcg.abs() >= (lim - 0.005)).astype(float).rolling(5, min_periods=1).sum()

        out.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "trade_date": g["trade_date"].values,
                    "momentum": momentum.values,
                    "rsi": rsi.values,
                    "realized_vol": realized_vol.values,
                    "turnover_roll_mean": turnover_roll_mean.values,
                    "vol_ret_corr": vol_ret_corr.values,
                    "vol_to_turnover": vol_to_turnover.values,
                    "bias_short": bias_short.values,
                    "bias_long": bias_long.values,
                    "max_single_day_drop": max_single_day_drop.values,
                    "recent_return": recent_return.values,
                    "price_position": price_position.values,
                    "intraday_range": intraday_range.values,
                    "upper_shadow_ratio": upper_shadow_ratio.values,
                    "lower_shadow_ratio": lower_shadow_ratio.values,
                    "close_open_return": close_open_return.values,
                    "tail_strength": tail_strength.values,
                    "limit_move_hits_5d": limit_move_hits_5d.values,
                }
            )
        )
    if not out:
        raise RuntimeError("因子计算为空")
    return pd.concat(out, ignore_index=True)


def build_score(factors: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    fac_cols = [c for c in weights.keys() if c in factors.columns]
    rows = []
    for dt, g in factors.groupby("trade_date"):
        g = g.dropna(subset=fac_cols, how="all").copy()
        if len(g) < 10:
            continue
        score = pd.Series(0.0, index=g.index)
        for fc in fac_cols:
            col = pd.to_numeric(g[fc], errors="coerce")
            m = col.notna() & np.isfinite(col)
            if m.sum() < 5:
                continue
            score[m] += _zscore_clip(col[m]) * weights[fc]
        rows.append(pd.DataFrame({"symbol": g["symbol"].values, "trade_date": dt, "score": score.values}))
    if not rows:
        raise RuntimeError("得分构建为空")
    return pd.concat(rows, ignore_index=True).dropna(subset=["score"])


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


def build_topk_weights(
    score_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    top_k: int,
    rebalance_rule: str,
    prefilter_cfg: dict,
    max_turnover: float,
) -> pd.DataFrame:
    score_df = score_df.copy()
    factor_df = factor_df.copy()
    score_df["trade_date"] = pd.to_datetime(score_df["trade_date"])
    factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])
    fac_today = factor_df[
        ["symbol", "trade_date", "turnover_roll_mean", "price_position", "limit_move_hits_5d"]
    ].copy()

    rd_list = _rebalance_dates(score_df["trade_date"].unique(), rebalance_rule)
    rows = []
    prev_holdings: set[str] = set()

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

        topk = filtered.nlargest(top_k, "score")[["symbol", "score"]].copy()
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

        w = 1.0 / len(topk)
        for _, r in topk.iterrows():
            rows.append({"trade_date": rd, "symbol": str(r["symbol"]).zfill(6), "weight": w})
        prev_holdings = set(topk["symbol"].astype(str).tolist())

    if not rows:
        raise RuntimeError("未生成任何调仓权重")
    w_long = pd.DataFrame(rows)
    w_wide = w_long.pivot(index="trade_date", columns="symbol", values="weight").fillna(0.0)
    w_wide.index = pd.to_datetime(w_wide.index)
    return w_wide


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
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期（默认取 config.paths.asof_trade_date）")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="标的最少历史交易日")
    p.add_argument(
        "--json-report",
        default="",
        help="将回测摘要写入 JSON 文件（供文档/CI 使用）",
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
    args = parse_args()
    cfg = load_config()
    cfg_path = PROJECT_ROOT / "config.yaml"
    config_source = str(cfg_path) if cfg_path.exists() else "builtin_defaults"

    end_date = args.end or str(cfg.get("paths", {}).get("asof_trade_date", ""))
    if not end_date:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = args.start

    db_path = str(PROJECT_ROOT / cfg["paths"]["duckdb_path"])
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))

    signals = cfg.get("signals", {})
    backtest_cfg = cfg.get("backtest", {})
    prefilter_cfg = cfg.get("prefilter", {})
    portfolio_cfg = cfg.get("portfolio", {})
    ce_weights = normalize_weights(signals.get("composite_extended", {}))
    top_k = int(signals.get("top_k", 50))
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    max_turnover = float(portfolio_cfg.get("max_turnover", 1.0))

    print("=" * 70)
    print("A 股回测评估（改进版）")
    print(
        f"区间: {start_date} ~ {end_date} | Top-{top_k} | 调仓: {rebalance_rule} | "
        f"max_turnover={max_turnover:.2f} | execution_mode={str(backtest_cfg.get('execution_mode', 'close_to_close'))}"
    )
    print("=" * 70)

    print("[1/7] 读取日线数据...")
    daily_df = load_daily_from_duckdb(db_path, start_date, end_date, args.lookback_days)
    print(f"  日线: {len(daily_df):,} 行, 标的: {daily_df['symbol'].nunique():,}")

    print("[2/7] 计算因子...")
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  因子长表: {len(factors):,} 行")

    print("[3/7] 截面打分...")
    score_df = build_score(factors, ce_weights)
    print(f"  得分行数: {len(score_df):,}")

    print("[4/7] 构建权重...")
    weights = build_topk_weights(
        score_df=score_df,
        factor_df=factors,
        top_k=top_k,
        rebalance_rule=rebalance_rule,
        prefilter_cfg=prefilter_cfg,
        max_turnover=max_turnover,
    )
    weights = weights[weights.index >= pd.Timestamp(start_date)]
    print(f"  调仓日: {len(weights)} | 标的列: {weights.shape[1]}")

    print("[5/7] 回测执行...")
    execution_mode = str(backtest_cfg.get("execution_mode", "close_to_close")).lower().strip()
    if execution_mode not in ("close_to_close", "tplus1_open", "vwap"):
        print(f"[警告] 未识别 execution_mode={execution_mode}，回退为 close_to_close")
        execution_mode = "close_to_close"

    if execution_mode == "tplus1_open":
        open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
        open_returns = open_returns.sort_index()
        open_returns.index = pd.to_datetime(open_returns.index)
        open_returns = open_returns.reindex(columns=weights.columns).fillna(0.0)
        asset_returns = open_returns[
            (open_returns.index >= pd.Timestamp(start_date)) & (open_returns.index <= pd.Timestamp(end_date))
        ]
    else:
        asset_returns = build_asset_returns(daily_df, weights.columns, start_date, end_date)
    first_reb = weights.index.min()
    asset_returns = asset_returns[asset_returns.index >= first_reb]
    if not asset_returns.empty and not weights.empty and weights.index.min() > asset_returns.index.min():
        seed = weights.iloc[[0]].copy()
        seed.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights = pd.concat([seed, weights], axis=0)
        weights = weights[~weights.index.duplicated(keep="last")].sort_index()

    bt_no_cost = BacktestConfig(
        cost_params=None,
        execution_mode=execution_mode,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
        vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
        vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
    )
    bt_cost = BacktestConfig(
        cost_params=costs,
        execution_mode=execution_mode,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "idle")),
        vwap_slippage_bps_per_side=float(backtest_cfg.get("vwap_slippage_bps_per_side", 3.0)),
        vwap_impact_bps=float(backtest_cfg.get("vwap_impact_bps", 8.0)),
    )
    res_nc = run_backtest(asset_returns, weights, config=bt_no_cost)
    res_wc = run_backtest(asset_returns, weights, config=bt_cost)

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
    rolling = rolling_walk_forward_windows(asset_returns.index, train_days=252, test_days=63, step_days=63)
    _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cost, use_test_only=True)
    slices = contiguous_time_splits(asset_returns.index, n_splits=5, min_train_days=252)
    _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cost, use_test_only=True)
    overfit_cmp = compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}

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

    print("Walk-Forward（时间切片）")
    if sp_detail.empty:
        print("  无有效折次")
    else:
        print(
            f"  折数 {len(sp_detail)} | OOS 年化均值 {fmt_pct(sp_agg.get('annualized_return_agg', np.nan))} | "
            f"OOS 夏普均值 {fmt_num(sp_agg.get('sharpe_ratio_agg', np.nan))}"
        )
        d_sharpe = overfit_cmp.get("delta_sharpe_ratio", np.nan)
        d_ann = overfit_cmp.get("delta_annualized_return", np.nan)
        print(f"  全样本-切片差值: 年化 {fmt_pct(d_ann)} | 夏普 {fmt_num(d_sharpe)}")

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
            "config_source": config_source,
            "parameters": {
                "start": start_date,
                "end": end_date,
                "lookback_days": args.lookback_days,
                "min_hist_days": args.min_hist_days,
                "top_k": top_k,
                "rebalance_rule": rebalance_rule,
                "max_turnover": max_turnover,
                "execution_mode": execution_mode,
                "composite_extended_weights": ce_weights,
                "prefilter": prefilter_cfg,
                "benchmark_min_history_days": adaptive_min_days,
            },
            "meta": {
                "n_trading_days": int(res_wc.panel.n_periods),
                "n_rebalances": int(res_wc.meta.get("n_rebalances", 0)),
                "n_weight_symbols": int(weights.shape[1]),
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
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records"))
                if not sp_detail.empty
                else [],
                "agg": _json_sanitize(dict(sp_agg)),
                "full_vs_slices": _json_sanitize(overfit_cmp),
            },
        }
        out_path = Path(args.json_report).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_sanitize(report), f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print("完成。")


if __name__ == "__main__":
    main()
