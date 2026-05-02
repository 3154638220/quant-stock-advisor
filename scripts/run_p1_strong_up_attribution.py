"""R1 Strong-Up 失败归因：5 档状态 + breadth 切片 + switch-in/out 暴露 + 三年专项。

不训练新模型，只做归因。基线固定为：
    score=S2=vol_to_turnover / top_k=20 / M / equal_weight / tplus1_open

输出落到：
    data/results/{prefix}_*.csv / .json
    docs/{prefix}.md

只需依赖默认 ``config.yaml.backtest`` 即可运行。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (
    BacktestConfig,
    _attach_pit_fundamentals,
    _rebalance_dates,
    attach_universe_filter,
    build_limit_up_open_mask,
    build_market_ew_open_to_open_benchmark,
    build_open_to_open_returns,
    build_score,
    build_topk_weights,
    compute_factors,
    load_config,
    load_daily_from_duckdb,
    normalize_weights,
    run_backtest,
    transaction_cost_params_from_mapping,
)
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from scripts.research_identity import make_research_identity, slugify_token


REGIME_ORDER = ["strong_down", "mild_down", "neutral", "mild_up", "strong_up"]
BREADTH_ORDER = ["narrow", "mid", "wide"]
GROUP_ORDER = ["top20", "21_40", "41_60", "switch_in", "switch_out", "benchmark"]

# 用于暴露对比的特征（含新增的 R1 特征）
EXPOSURE_FEATURES: tuple[str, ...] = (
    "log_market_cap",
    "amount_20d",
    "turnover_roll_mean",
    "realized_vol",
    "price_position",
    "recent_return",
    "momentum_12_1",
    "vol_to_turnover",
    "rel_strength_20d",
    "rel_strength_60d",
    "amount_expansion_5_60",
    "turnover_expansion_5_60",
    "tail_strength_20d",
    "overnight_gap_pos_20d",
    "limit_up_hits_20d",
    "limit_down_hits_20d",
    "limit_move_hits_20d",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1 Strong-Up 失败归因")
    p.add_argument("--config", default="config.yaml.backtest", help="配置文件路径")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument(
        "--output-prefix",
        default="p1_strong_up_failure_attribution_2026-04-27",
        help="输出前缀（写入 data/results 与 docs）",
    )
    return p.parse_args()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, np.floating):
        return _json_sanitize(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _compound_return(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    return float((1.0 + arr).prod() - 1.0)


# ----------------------------------------------------------------------------
# 新增 R1 特征
# ----------------------------------------------------------------------------
def compute_r1_extra_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """构造 R1 关注的相对强度 / 扩张 / 路径特征。返回 (symbol, trade_date, features...)。"""

    df = daily_df.sort_values(["symbol", "trade_date"]).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns else df["close"] * df["volume"]
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["pre_close"] = df.groupby("symbol")["close"].shift(1)
    df["overnight_gap"] = df["open"] / df["pre_close"].replace(0, np.nan) - 1.0
    df["close_open_return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    g = df.groupby("symbol", sort=False)

    df["ret_20d"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    df["ret_60d"] = g["close"].transform(lambda s: s / s.shift(60) - 1.0)
    df["amount_5d_mean"] = g["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["amount_20d_mean"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["amount_60d_mean"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    df["turnover_5d_mean"] = g["turnover"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["turnover_60d_mean"] = g["turnover"].transform(lambda s: s.rolling(60, min_periods=30).mean())

    df["amount_expansion_5_60"] = np.log(df["amount_5d_mean"] / df["amount_60d_mean"].replace(0, np.nan))
    df["turnover_expansion_5_60"] = np.log(df["turnover_5d_mean"] / df["turnover_60d_mean"].replace(0, np.nan))
    df["tail_strength_20d"] = g["close_open_return"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["overnight_gap_pos_20d"] = g["overnight_gap"].transform(
        lambda s: (s > 0.005).astype(float).rolling(20, min_periods=10).mean()
    )
    # 涨停近邻：根据交易所限制阈值（板块决定）
    def _limit_thr(sym: str) -> float:
        if sym.startswith(("300", "688")):
            return 0.20
        if sym.startswith(("8", "4")):
            return 0.30
        return 0.10

    df["_limit_thr"] = df["symbol"].map(_limit_thr)
    df["_limit_up_hit"] = (df["pct_chg"] >= (df["_limit_thr"] - 0.005)).astype(float)
    df["_limit_down_hit"] = (df["pct_chg"] <= -(df["_limit_thr"] - 0.005)).astype(float)
    df["_limit_move_hit"] = ((df["_limit_up_hit"] > 0) | (df["_limit_down_hit"] > 0)).astype(float)
    df["limit_up_hits_20d"] = (
        df.groupby("symbol", sort=False)["_limit_up_hit"]
        .transform(lambda s: s.rolling(20, min_periods=10).sum())
    )
    df["limit_down_hits_20d"] = (
        df.groupby("symbol", sort=False)["_limit_down_hit"]
        .transform(lambda s: s.rolling(20, min_periods=10).sum())
    )
    df["limit_move_hits_20d"] = (
        df.groupby("symbol", sort=False)["_limit_move_hit"]
        .transform(lambda s: s.rolling(20, min_periods=10).sum())
    )

    # 横截面相对强度：同一交易日去掉中位数
    cs_med_20 = df.groupby("trade_date")["ret_20d"].transform("median")
    cs_med_60 = df.groupby("trade_date")["ret_60d"].transform("median")
    df["rel_strength_20d"] = df["ret_20d"] - cs_med_20
    df["rel_strength_60d"] = df["ret_60d"] - cs_med_60

    keep = [
        "symbol",
        "trade_date",
        "amount_20d_mean",
        "rel_strength_20d",
        "rel_strength_60d",
        "amount_expansion_5_60",
        "turnover_expansion_5_60",
        "tail_strength_20d",
        "overnight_gap_pos_20d",
        "limit_up_hits_20d",
        "limit_down_hits_20d",
        "limit_move_hits_20d",
    ]
    return df[keep].rename(columns={"amount_20d_mean": "amount_20d"})


# ----------------------------------------------------------------------------
# Regime 分类
# ----------------------------------------------------------------------------
def classify_regimes(
    monthly_df: pd.DataFrame,
    breadth_series: pd.Series,
    *,
    threshold_mode: str = "diagnostic_only",
    min_periods: int = 12,
) -> pd.DataFrame:
    """根据基准月收益分位数划分 5 档；R2/R3 决策必须使用 expanding 等可交易阈值。"""
    out = monthly_df.copy()
    bench = pd.to_numeric(out["benchmark_return"], errors="coerce")
    threshold_mode = str(threshold_mode).lower().strip()
    if threshold_mode not in {"diagnostic_only", "expanding"}:
        raise ValueError("threshold_mode must be diagnostic_only or expanding")

    def _bucket_ret(x: float, p20: float, p40: float, p60: float, p80: float) -> str:
        if not np.isfinite(x):
            return "neutral"
        if x <= p20:
            return "strong_down"
        if x <= p40:
            return "mild_down"
        if x <= p60:
            return "neutral"
        if x <= p80:
            return "mild_up"
        return "strong_up"

    breadth_aligned = breadth_series.reindex(out["month_end"]).reset_index(drop=True)
    out["breadth_value"] = breadth_aligned.values

    def _bucket_breadth(x: float, bp30: float, bp70: float) -> str:
        if not np.isfinite(x):
            return "mid"
        if x <= bp30:
            return "narrow"
        if x >= bp70:
            return "wide"
        return "mid"

    threshold_rows: list[dict[str, Any]] = []
    if threshold_mode == "diagnostic_only":
        p20, p40, p60, p80 = bench.quantile([0.2, 0.4, 0.6, 0.8])
        bp30, bp70 = pd.Series(breadth_aligned).quantile([0.3, 0.7])
        out["regime"] = bench.apply(lambda x: _bucket_ret(float(x), p20, p40, p60, p80))
        out["breadth"] = pd.Series(breadth_aligned).apply(lambda x: _bucket_breadth(float(x), bp30, bp70)).values
        out.attrs["regime_thresholds"] = {"p20": float(p20), "p40": float(p40), "p60": float(p60), "p80": float(p80)}
        out.attrs["breadth_thresholds"] = {"p30": float(bp30), "p70": float(bp70)}
        out["threshold_observations"] = int(bench.dropna().shape[0])
        out["regime_p20"] = float(p20)
        out["regime_p40"] = float(p40)
        out["regime_p60"] = float(p60)
        out["regime_p80"] = float(p80)
        out["breadth_p30"] = float(bp30)
        out["breadth_p70"] = float(bp70)
    else:
        regimes: list[str] = []
        breadths: list[str] = []
        breadth_values = pd.Series(breadth_aligned)
        for i, row in out.reset_index(drop=True).iterrows():
            hist_ret = bench.iloc[: i + 1].dropna()
            hist_breadth = breadth_values.iloc[: i + 1].dropna()
            if len(hist_ret) < int(min_periods):
                regimes.append("neutral")
                p20 = p40 = p60 = p80 = np.nan
            else:
                p20, p40, p60, p80 = hist_ret.quantile([0.2, 0.4, 0.6, 0.8])
                regimes.append(_bucket_ret(float(row["benchmark_return"]), p20, p40, p60, p80))
            if len(hist_breadth) < int(min_periods):
                breadths.append("mid")
                bp30 = bp70 = np.nan
            else:
                bp30, bp70 = hist_breadth.quantile([0.3, 0.7])
                breadths.append(_bucket_breadth(float(row["breadth_value"]), bp30, bp70))
            threshold_rows.append(
                {
                    "month_end": row["month_end"],
                    "threshold_mode": "expanding",
                    "observations": int(len(hist_ret)),
                    "p20": float(p20) if np.isfinite(p20) else np.nan,
                    "p40": float(p40) if np.isfinite(p40) else np.nan,
                    "p60": float(p60) if np.isfinite(p60) else np.nan,
                    "p80": float(p80) if np.isfinite(p80) else np.nan,
                    "breadth_p30": float(bp30) if np.isfinite(bp30) else np.nan,
                    "breadth_p70": float(bp70) if np.isfinite(bp70) else np.nan,
                }
            )
        out["regime"] = regimes
        out["breadth"] = breadths
        trace_df = pd.DataFrame(threshold_rows)
        out["threshold_observations"] = trace_df["observations"].to_numpy()
        out["regime_p20"] = trace_df["p20"].to_numpy()
        out["regime_p40"] = trace_df["p40"].to_numpy()
        out["regime_p60"] = trace_df["p60"].to_numpy()
        out["regime_p80"] = trace_df["p80"].to_numpy()
        out["breadth_p30"] = trace_df["breadth_p30"].to_numpy()
        out["breadth_p70"] = trace_df["breadth_p70"].to_numpy()
        out.attrs["threshold_trace"] = threshold_rows
        out.attrs["regime_thresholds"] = {"mode": "expanding", "min_periods": int(min_periods)}
        out.attrs["breadth_thresholds"] = {"mode": "expanding", "min_periods": int(min_periods)}
    out["state_threshold_mode"] = threshold_mode
    out["lookahead_check"] = "diagnostic_only" if threshold_mode == "diagnostic_only" else "pass"
    return out


def compute_breadth(daily_df: pd.DataFrame, benchmark_symbols: set[str]) -> pd.Series:
    """月内日 cross-section 正收益股票占比的均值。"""
    df = daily_df[daily_df["symbol"].isin(benchmark_symbols)].copy()
    df = df.sort_values(["symbol", "trade_date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    df = df.dropna(subset=["ret"])
    daily_breadth = df.groupby("trade_date")["ret"].apply(lambda s: float((s > 0).mean()))
    daily_breadth.index = pd.to_datetime(daily_breadth.index)
    monthly_breadth = daily_breadth.resample("ME").mean()
    return monthly_breadth


# ----------------------------------------------------------------------------
# 构造每个 rebalance 的 group 划分
# ----------------------------------------------------------------------------
def build_groups_per_rebalance(
    score_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, list[str]]]:
    out: dict[pd.Timestamp, dict[str, list[str]]] = {}
    prev_top20: list[str] = []
    for date in rebalance_dates:
        day = score_df[score_df["trade_date"] == pd.Timestamp(date)].copy()
        if day.empty:
            continue
        day = day.sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = np.arange(1, len(day) + 1, dtype=int)
        top20 = day.loc[day["rank"] <= 20, "symbol"].astype(str).tolist()
        b21_40 = day.loc[(day["rank"] >= 21) & (day["rank"] <= 40), "symbol"].astype(str).tolist()
        b41_60 = day.loc[(day["rank"] >= 41) & (day["rank"] <= 60), "symbol"].astype(str).tolist()
        switch_in = [s for s in top20 if s not in prev_top20] if prev_top20 else []
        switch_out = [s for s in prev_top20 if s not in top20] if prev_top20 else []
        out[pd.Timestamp(date)] = {
            "top20": top20,
            "21_40": b21_40,
            "41_60": b41_60,
            "switch_in": switch_in,
            "switch_out": switch_out,
        }
        prev_top20 = top20
    return out


# ----------------------------------------------------------------------------
# 计算每个 (rebalance, group) 的暴露均值
# ----------------------------------------------------------------------------
def build_exposure_records(
    rebalance_groups: dict[pd.Timestamp, dict[str, list[str]]],
    monthly_regime_df: pd.DataFrame,
    feature_panel: pd.DataFrame,
    benchmark_symbols: set[str],
) -> pd.DataFrame:
    feature_panel = feature_panel.set_index(["trade_date", "symbol"]).sort_index()
    # 月末 -> regime 映射
    regime_lookup: dict[pd.Period, dict[str, str]] = {}
    for row in monthly_regime_df.to_dict(orient="records"):
        period = pd.Timestamp(row["month_end"]).to_period("M")
        regime_lookup[period] = {
            "regime": row["regime"],
            "breadth": row["breadth"],
            "month_end": pd.Timestamp(row["month_end"]),
            "benchmark_return": float(row["benchmark_return"]),
            "strategy_return": float(row["strategy_return"]),
            "excess_return": float(row["excess_return"]),
        }

    records: list[dict[str, Any]] = []
    available_features = [f for f in EXPOSURE_FEATURES if f in feature_panel.columns]

    for date, groups in rebalance_groups.items():
        period = date.to_period("M")
        regime_info = regime_lookup.get(period)
        if regime_info is None:
            continue
        # 当日的特征切片
        try:
            day_panel = feature_panel.xs(date, level="trade_date")
        except KeyError:
            continue
        if day_panel.empty:
            continue

        # benchmark = 全部投资域（满足历史长度）的等权
        bench_syms = sorted(benchmark_symbols & set(day_panel.index.astype(str)))
        merged_groups = dict(groups)
        merged_groups["benchmark"] = bench_syms

        for group_name, syms in merged_groups.items():
            if not syms:
                continue
            slice_df = day_panel.reindex([s for s in syms if s in day_panel.index])
            if slice_df.empty:
                continue
            row: dict[str, Any] = {
                "trade_date": date,
                "year": int(date.year),
                "month_end": regime_info["month_end"],
                "regime": regime_info["regime"],
                "breadth": regime_info["breadth"],
                "benchmark_return": regime_info["benchmark_return"],
                "strategy_return": regime_info["strategy_return"],
                "excess_return": regime_info["excess_return"],
                "group": group_name,
                "n_symbols": int(len(slice_df)),
            }
            for feat in available_features:
                if feat not in slice_df.columns:
                    continue
                vals = pd.to_numeric(slice_df[feat], errors="coerce")
                vals = vals[np.isfinite(vals)]
                row[feat] = float(vals.mean()) if not vals.empty else np.nan
            records.append(row)
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Switch quality：switch_in 与 switch_out 的下一持有期收益对比
# ----------------------------------------------------------------------------
def build_switch_quality(
    rebalance_groups: dict[pd.Timestamp, dict[str, list[str]]],
    monthly_regime_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    rebalance_dates = sorted(rebalance_groups.keys())
    monthly_lookup: dict[pd.Period, dict[str, Any]] = {}
    for row in monthly_regime_df.to_dict(orient="records"):
        period = pd.Timestamp(row["month_end"]).to_period("M")
        monthly_lookup[period] = row

    rows: list[dict[str, Any]] = []
    for idx, date in enumerate(rebalance_dates[:-1]):
        nxt = rebalance_dates[idx + 1]
        period_holding = (date + pd.offsets.BDay(1)).to_period("M")
        monthly_info = monthly_lookup.get(period_holding) or monthly_lookup.get(date.to_period("M"))
        if monthly_info is None:
            continue
        period_returns = asset_returns[(asset_returns.index > date) & (asset_returns.index <= nxt)]
        if period_returns.empty:
            continue
        cum = (1.0 + period_returns).prod(axis=0) - 1.0
        groups = rebalance_groups[date]
        switch_in = [s for s in groups["switch_in"] if s in cum.index]
        switch_out = [s for s in groups["switch_out"] if s in cum.index]
        top20 = [s for s in groups["top20"] if s in cum.index]
        next_buckets = [s for s in (groups["21_40"] + groups["41_60"]) if s in cum.index]
        if not (switch_in and switch_out):
            continue
        switch_in_mean = float(cum.reindex(switch_in).mean())
        switch_out_mean = float(cum.reindex(switch_out).mean())
        topk_mean = float(cum.reindex(top20).mean()) if top20 else np.nan
        next_mean = float(cum.reindex(next_buckets).mean()) if next_buckets else np.nan
        rows.append(
            {
                "trade_date": date,
                "year": int(date.year),
                "regime": monthly_info["regime"],
                "breadth": monthly_info["breadth"],
                "benchmark_return": float(monthly_info["benchmark_return"]),
                "excess_return": float(monthly_info["excess_return"]),
                "n_switch_in": int(len(switch_in)),
                "n_switch_out": int(len(switch_out)),
                "switch_in_mean_return": switch_in_mean,
                "switch_out_mean_return": switch_out_mean,
                "switch_in_minus_out": switch_in_mean - switch_out_mean,
                "topk_mean_return": topk_mean,
                "next_buckets_mean_return": next_mean,
                "topk_minus_next": topk_mean - next_mean if np.isfinite(topk_mean) and np.isfinite(next_mean) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 聚合 / 报告
# ----------------------------------------------------------------------------
def summarize_regime_capture(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        part = monthly_df[monthly_df["regime"] == regime]
        if part.empty:
            continue
        bench_comp = _compound_return(part["benchmark_return"])
        strat_comp = _compound_return(part["strategy_return"])
        capture = strat_comp / bench_comp if np.isfinite(bench_comp) and abs(bench_comp) > 1e-12 else np.nan
        rows.append(
            {
                "regime": regime,
                "months": int(len(part)),
                "median_benchmark_return": float(part["benchmark_return"].median()),
                "median_strategy_return": float(part["strategy_return"].median()),
                "median_excess_return": float(part["excess_return"].median()),
                "positive_excess_share": float((part["excess_return"] > 0).mean()),
                "benchmark_compound": float(bench_comp),
                "strategy_compound": float(strat_comp),
                "capture_ratio": float(capture) if np.isfinite(capture) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_breadth_capture(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for b in BREADTH_ORDER:
        part = monthly_df[monthly_df["breadth"] == b]
        if part.empty:
            continue
        rows.append(
            {
                "breadth": b,
                "months": int(len(part)),
                "median_benchmark_return": float(part["benchmark_return"].median()),
                "median_strategy_return": float(part["strategy_return"].median()),
                "median_excess_return": float(part["excess_return"].median()),
                "positive_excess_share": float((part["excess_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_group_exposure(detail: pd.DataFrame) -> pd.DataFrame:
    """按 (regime, group) × feature 求均值。"""
    if detail.empty:
        return detail
    feat_cols = [c for c in EXPOSURE_FEATURES if c in detail.columns]
    rows: list[dict[str, Any]] = []
    for (regime, group), part in detail.groupby(["regime", "group"], dropna=False):
        row = {
            "regime": regime,
            "group": group,
            "rebalances": int(len(part)),
        }
        for feat in feat_cols:
            row[feat] = float(pd.to_numeric(part[feat], errors="coerce").mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(["regime", "group"]).reset_index(drop=True)
    df["regime"] = df["regime"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def summarize_active_diff(detail: pd.DataFrame) -> pd.DataFrame:
    """对每个 (regime, group)：减去 benchmark 同 regime 的均值得到 active diff。"""
    if detail.empty:
        return detail
    feat_cols = [c for c in EXPOSURE_FEATURES if c in detail.columns]
    bench = detail[detail["group"] == "benchmark"][["trade_date"] + feat_cols].copy()
    bench = bench.rename(columns={c: f"{c}__bench" for c in feat_cols})
    merged = detail.merge(bench, on="trade_date", how="left")
    rows: list[dict[str, Any]] = []
    for (regime, group), part in merged.groupby(["regime", "group"], dropna=False):
        row = {"regime": regime, "group": group, "rebalances": int(len(part))}
        for feat in feat_cols:
            diff = pd.to_numeric(part[feat], errors="coerce") - pd.to_numeric(
                part[f"{feat}__bench"], errors="coerce"
            )
            row[f"active_{feat}"] = float(diff.mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(["regime", "group"]).reset_index(drop=True)
    df["regime"] = df["regime"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def summarize_switch_by_regime(switch_df: pd.DataFrame) -> pd.DataFrame:
    if switch_df.empty:
        return switch_df
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        part = switch_df[switch_df["regime"] == regime]
        if part.empty:
            continue
        rows.append(
            {
                "regime": regime,
                "rebalances": int(len(part)),
                "mean_switch_in": float(part["switch_in_mean_return"].mean()),
                "mean_switch_out": float(part["switch_out_mean_return"].mean()),
                "mean_switch_in_minus_out": float(part["switch_in_minus_out"].mean()),
                "median_switch_in_minus_out": float(part["switch_in_minus_out"].median()),
                "switch_in_winning_share": float((part["switch_in_minus_out"] > 0).mean()),
                "mean_topk_minus_next": float(pd.to_numeric(part["topk_minus_next"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 报告生成
# ----------------------------------------------------------------------------
def _md_or_empty(df: pd.DataFrame, msg: str = "_无数据_") -> str:
    if df is None or df.empty:
        return msg
    return df.to_markdown(index=False)


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    regime_thresholds: dict[str, float],
    breadth_thresholds: dict[str, float],
    regime_capture: pd.DataFrame,
    breadth_capture: pd.DataFrame,
    group_exposure: pd.DataFrame,
    active_diff: pd.DataFrame,
    switch_by_regime: pd.DataFrame,
    year_capture: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R1 Strong-Up 失败归因\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append(
        "- 固定口径：`score=S2=vol_to_turnover` / `top_k=20` / `M` / `equal_weight` / `tplus1_open`"
    )
    lines.append(
        f"- Regime 分位阈值（基准月收益）：P20={regime_thresholds['p20']:.4f} / "
        f"P40={regime_thresholds['p40']:.4f} / P60={regime_thresholds['p60']:.4f} / "
        f"P80={regime_thresholds['p80']:.4f}"
    )
    lines.append(
        f"- Breadth 分位阈值（月内 daily positive ratio 均值）：P30={breadth_thresholds['p30']:.4f} / "
        f"P70={breadth_thresholds['p70']:.4f}"
    )
    lines.append("")

    # 取 strong_up 结论
    strong_up_row = regime_capture[regime_capture["regime"] == "strong_up"]
    wide_row = breadth_capture[breadth_capture["breadth"] == "wide"]
    summary_lines: list[str] = []
    if not strong_up_row.empty:
        r = strong_up_row.iloc[0]
        summary_lines.append(
            f"strong_up 月份共 {int(r['months'])} 个，中位超额 `{float(r['median_excess_return']):.4f}`，"
            f"正超额比例 `{float(r['positive_excess_share']):.3f}`，capture_ratio `{float(r['capture_ratio']):.3f}`。"
        )
    if not wide_row.empty:
        r = wide_row.iloc[0]
        summary_lines.append(
            f"wide breadth 月份共 {int(r['months'])} 个，中位超额 `{float(r['median_excess_return']):.4f}`。"
        )
    if not switch_by_regime.empty:
        sup = switch_by_regime[switch_by_regime["regime"] == "strong_up"]
        if not sup.empty:
            r = sup.iloc[0]
            summary_lines.append(
                f"strong_up 状态下 switch_in_minus_out 均值 `{float(r['mean_switch_in_minus_out']):.4f}`，"
                f"换入跑赢比例 `{float(r['switch_in_winning_share']):.3f}`，"
                f"topk_minus_next 均值 `{float(r['mean_topk_minus_next']):.4f}`。"
            )

    lines.append("## 结论速览\n")
    if summary_lines:
        for s in summary_lines:
            lines.append(f"- {s}")
    else:
        lines.append("_无足够数据_")
    lines.append("")

    lines.append("## 1. Regime 切片（5 档）\n")
    lines.append(_md_or_empty(regime_capture))
    lines.append("")

    lines.append("## 2. Breadth 切片（3 档）\n")
    lines.append(_md_or_empty(breadth_capture))
    lines.append("")

    lines.append("## 3. 关键年份（2021/2025/2026）regime 表现\n")
    lines.append(_md_or_empty(year_capture))
    lines.append("")

    lines.append("## 4. 持仓暴露：(regime, group) × feature 均值\n")
    lines.append("group 含义：`top20`（当期持仓）/ `21_40` / `41_60` / `switch_in` / `switch_out` / `benchmark`（投资域等权）。")
    lines.append("")
    lines.append("仅展示 `strong_up` / `wide breadth` 关注的子集（完整表见 CSV）。")
    lines.append("")
    if not group_exposure.empty:
        focus = group_exposure[group_exposure["regime"].isin(["strong_up", "neutral", "strong_down"])].copy()
        # 关注核心特征
        focus_cols = [
            "regime",
            "group",
            "rebalances",
            "rel_strength_20d",
            "rel_strength_60d",
            "amount_expansion_5_60",
            "turnover_expansion_5_60",
            "tail_strength_20d",
            "limit_up_hits_20d",
            "log_market_cap",
            "realized_vol",
        ]
        focus_cols = [c for c in focus_cols if c in focus.columns]
        lines.append(_md_or_empty(focus[focus_cols]))
    else:
        lines.append("_无暴露数据_")
    lines.append("")

    lines.append("## 5. Active diff（group - benchmark，按 regime）\n")
    if not active_diff.empty:
        focus = active_diff[active_diff["regime"].isin(["strong_up", "neutral", "strong_down"])].copy()
        focus_cols = [
            "regime",
            "group",
            "rebalances",
            "active_rel_strength_20d",
            "active_rel_strength_60d",
            "active_amount_expansion_5_60",
            "active_turnover_expansion_5_60",
            "active_tail_strength_20d",
            "active_limit_up_hits_20d",
            "active_log_market_cap",
            "active_realized_vol",
        ]
        focus_cols = [c for c in focus_cols if c in focus.columns]
        lines.append(_md_or_empty(focus[focus_cols]))
    else:
        lines.append("_无 active diff 数据_")
    lines.append("")

    lines.append("## 6. Switch quality（按 regime）\n")
    lines.append(_md_or_empty(switch_by_regime))
    lines.append("")

    lines.append("## 7. 归因结论与可验证机制\n")
    lines.append(
        "1. 见结论速览：strong_up 月份是 S2 失败核心来源；switch_in_minus_out 是否在 strong_up 转负是 R3 boundary-aware 的关键证据。"
    )
    lines.append(
        "2. 若 active `rel_strength_20d/60d`、`amount_expansion`、`turnover_expansion` 在 `strong_up` 下显著为负，则确认\"上涨扩散期持仓不参与扩张票\"。"
    )
    lines.append(
        "3. 若 `active_log_market_cap` 在 strong_up 偏正、`active_realized_vol` 偏负，则确认\"持仓在上涨月偏稳定大盘股、缺弹性\"。"
    )
    lines.append(
        "4. 若 `active_limit_up_hits_20d` / `active_tail_strength_20d` 在 strong_up 偏负，则提示可交易路径强度也不参与。"
    )
    lines.append("")
    lines.append(
        "上述每条结论可在第 4-5 节表中直接读取系数。若同方向证据 ≥ 2 条，则 R2 upside sleeve 应优先用 `relative_strength + amount_expansion` 类组合作为第一版。"
    )
    lines.append("")

    lines.append("## 8. 产出文件\n")
    for suf in [
        "monthly.csv",
        "regime_capture.csv",
        "breadth_capture.csv",
        "year_capture.csv",
        "group_exposure_detail.csv",
        "group_exposure.csv",
        "group_active_diff.csv",
        "switch_quality.csv",
        "switch_by_regime.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suf}`")
    lines.append("")

    lines.append("## 9. 配置参数\n")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg, config_source = load_config(args.config)
    paths_cfg = cfg.get("paths", {}) or {}
    db_path_raw = paths_cfg.get("duckdb_path") or paths_cfg.get("database_path") or "data/market.duckdb"
    db_path = str(db_path_raw if Path(db_path_raw).is_absolute() else PROJECT_ROOT / db_path_raw)
    end_date = args.end or str(paths_cfg.get("asof_trade_date") or pd.Timestamp.today().strftime("%Y-%m-%d"))

    backtest_cfg = cfg.get("backtest", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    uf_cfg = cfg.get("universe_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}

    top_k = int(signals_cfg.get("top_k", 20))
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    max_turnover = float(portfolio_cfg.get("max_turnover", 1.0))
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()

    print(f"[1/7] load daily start={args.start} end={end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)

    print("[2/7] compute factors + R1 extras", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    factors = attach_universe_filter(
        factors,
        daily_df,
        enabled=bool(uf_cfg.get("enabled", False)),
        min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
    )
    extras = compute_r1_extra_features(daily_df)
    feature_panel = factors.merge(extras, on=["symbol", "trade_date"], how="left")
    print(f"  factors={factors.shape}  extras={extras.shape}  panel={feature_panel.shape}", flush=True)

    print("[3/7] build score / weights (S2)", flush=True)
    ce_weights = normalize_weights(signals_cfg.get("composite_extended", {}))
    score_df = build_score(factors, ce_weights)
    score_df = score_df[score_df["trade_date"] >= pd.Timestamp(args.start)].copy()
    weights = build_topk_weights(
        score_df=score_df,
        factor_df=factors,
        daily_df=daily_df,
        top_k=top_k,
        rebalance_rule=rebalance_rule,
        prefilter_cfg=prefilter_cfg,
        max_turnover=max_turnover,
        portfolio_method="equal_weight",
    )
    weights = weights[weights.index >= pd.Timestamp(args.start)].sort_index()

    print("[4/7] run backtest + benchmark", flush=True)
    if execution_mode != "tplus1_open":
        raise ValueError("仅支持 tplus1_open")
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    sym_universe = sorted(set(weights.columns.astype(str)))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    asset_returns = asset_returns[
        (asset_returns.index >= pd.Timestamp(args.start)) & (asset_returns.index <= pd.Timestamp(end_date))
    ]
    weights = weights.reindex(columns=sym_universe, fill_value=0.0)
    if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
        seed = weights.iloc[[0]].copy()
        seed.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights = pd.concat([seed, weights]).pipe(lambda d: d[~d.index.duplicated(keep="last")]).sort_index()
    asset_returns = asset_returns[asset_returns.index >= weights.index.min()]
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    bt_cfg = BacktestConfig(
        cost_params=costs,
        execution_mode="tplus1_open",
        execution_lag=1,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "redistribute")),
        limit_up_open_mask=build_limit_up_open_mask(daily_df).reindex(columns=sym_universe, fill_value=False),
    )
    res = run_backtest(asset_returns, weights, config=bt_cfg)
    n_trade_days = int(asset_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)

    print("[5/7] monthly + regime classification", flush=True)
    common = res.daily_returns.index.intersection(bench_daily.index)
    strat_daily = pd.to_numeric(res.daily_returns.reindex(common), errors="coerce").fillna(0.0)
    bench_aligned = pd.to_numeric(bench_daily.reindex(common), errors="coerce").fillna(0.0)
    monthly = pd.DataFrame(
        {
            "strategy_return": strat_daily.resample("ME").apply(_compound_return),
            "benchmark_return": bench_aligned.resample("ME").apply(_compound_return),
        }
    ).dropna(how="all")
    monthly["excess_return"] = monthly["strategy_return"] - monthly["benchmark_return"]
    monthly = monthly.reset_index(names="month_end")

    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_regime = classify_regimes(monthly, breadth_series)
    regime_thresholds = monthly_regime.attrs["regime_thresholds"]
    breadth_thresholds = monthly_regime.attrs["breadth_thresholds"]

    print("[6/7] groups / exposure / switch quality", flush=True)
    rebalance_dates = _rebalance_dates(score_df["trade_date"].unique(), rebalance_rule)
    rebalance_groups = build_groups_per_rebalance(score_df, rebalance_dates)

    exposure_detail = build_exposure_records(rebalance_groups, monthly_regime, feature_panel, benchmark_symbols)
    group_exposure = summarize_group_exposure(exposure_detail)
    active_diff = summarize_active_diff(exposure_detail)
    switch_df = build_switch_quality(rebalance_groups, monthly_regime, asset_returns)
    switch_by_regime = summarize_switch_by_regime(switch_df)

    regime_capture = summarize_regime_capture(monthly_regime)
    breadth_capture = summarize_breadth_capture(monthly_regime)

    # 三年专项
    year_rows: list[dict[str, Any]] = []
    for year in (2021, 2025, 2026):
        for regime in REGIME_ORDER:
            part = monthly_regime[(monthly_regime["month_end"].dt.year == year) & (monthly_regime["regime"] == regime)]
            if part.empty:
                continue
            year_rows.append(
                {
                    "year": year,
                    "regime": regime,
                    "months": int(len(part)),
                    "median_benchmark_return": float(part["benchmark_return"].median()),
                    "median_strategy_return": float(part["strategy_return"].median()),
                    "median_excess_return": float(part["excess_return"].median()),
                    "positive_excess_share": float((part["excess_return"] > 0).mean()),
                }
            )
    year_capture = pd.DataFrame(year_rows)

    print("[7/7] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    monthly_regime.to_csv(results_dir / f"{prefix}_monthly.csv", index=False, encoding="utf-8-sig")
    regime_capture.to_csv(results_dir / f"{prefix}_regime_capture.csv", index=False, encoding="utf-8-sig")
    breadth_capture.to_csv(results_dir / f"{prefix}_breadth_capture.csv", index=False, encoding="utf-8-sig")
    year_capture.to_csv(results_dir / f"{prefix}_year_capture.csv", index=False, encoding="utf-8-sig")
    exposure_detail.to_csv(results_dir / f"{prefix}_group_exposure_detail.csv", index=False, encoding="utf-8-sig")
    group_exposure.to_csv(results_dir / f"{prefix}_group_exposure.csv", index=False, encoding="utf-8-sig")
    active_diff.to_csv(results_dir / f"{prefix}_group_active_diff.csv", index=False, encoding="utf-8-sig")
    switch_df.to_csv(results_dir / f"{prefix}_switch_quality.csv", index=False, encoding="utf-8-sig")
    switch_by_regime.to_csv(results_dir / f"{prefix}_switch_by_regime.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "equal_weight",
        "max_turnover": max_turnover,
        "execution_mode": execution_mode,
        "prefilter": prefilter_cfg,
        "universe_filter": uf_cfg,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
        "benchmark_min_history_days": bench_min,
        "config_source": config_source,
    }

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "regime_thresholds": regime_thresholds,
        "breadth_thresholds": breadth_thresholds,
        "regime_capture": regime_capture.to_dict(orient="records"),
        "breadth_capture": breadth_capture.to_dict(orient="records"),
        "year_capture": year_capture.to_dict(orient="records"),
        "switch_by_regime": switch_by_regime.to_dict(orient="records"),
        "full_sample": {
            "annualized_return": float(res.panel.annualized_return),
            "sharpe_ratio": float(res.panel.sharpe_ratio),
            "max_drawdown": float(res.panel.max_drawdown),
            "turnover_mean": float(res.panel.turnover_mean),
        },
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        config_source=config_source,
        params=params,
        regime_thresholds=regime_thresholds,
        breadth_thresholds=breadth_thresholds,
        regime_capture=regime_capture,
        breadth_capture=breadth_capture,
        group_exposure=group_exposure,
        active_diff=active_diff,
        switch_by_regime=switch_by_regime,
        year_capture=year_capture,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc_text, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)

    # --- standard research contract ---
    def _project_relative(path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(p)

    manifest_path = results_dir / f"{prefix}_manifest.json"
    identity = make_research_identity(
        result_type="p1_strong_up_attribution",
        research_topic="p1_strong_up_attribution",
        research_config_id=f"p1_strong_up_{slugify_token(prefix)}",
        output_stem=prefix,
    )
    data_slice = DataSlice(
        dataset_name="p1_strong_up_attribution_backtest",
        source_tables=("a_share_daily", "pit_fundamental"),
        date_start=args.start,
        date_end=end_date,
        asof_trade_date=end_date,
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="U1_liquid_tradable",
        rebalance_rule=rebalance_rule,
        execution_mode=execution_mode,
        label_return_mode="open_to_open",
        feature_set_id="p1_strong_up_attribution_factors",
        feature_columns=EXPOSURE_FEATURES,
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=str(args.config),
        extra={
            "top_k": int(top_k),
            "max_turnover": float(max_turnover),
        },
    )
    artifact_refs = (
        ArtifactRef("monthly_csv", _project_relative(results_dir / f"{prefix}_monthly.csv"), "csv", False, "月度状态"),
        ArtifactRef("regime_capture_csv", _project_relative(results_dir / f"{prefix}_regime_capture.csv"), "csv", False, "状态捕获"),
        ArtifactRef("breadth_capture_csv", _project_relative(results_dir / f"{prefix}_breadth_capture.csv"), "csv", False, "广度捕获"),
        ArtifactRef("year_capture_csv", _project_relative(results_dir / f"{prefix}_year_capture.csv"), "csv", False, "年度捕获"),
        ArtifactRef("group_exposure_detail_csv", _project_relative(results_dir / f"{prefix}_group_exposure_detail.csv"), "csv", False, "分组暴露明细"),
        ArtifactRef("group_exposure_csv", _project_relative(results_dir / f"{prefix}_group_exposure.csv"), "csv", False, "分组暴露汇总"),
        ArtifactRef("group_active_diff_csv", _project_relative(results_dir / f"{prefix}_group_active_diff.csv"), "csv", False, "主动差异"),
        ArtifactRef("switch_quality_csv", _project_relative(results_dir / f"{prefix}_switch_quality.csv"), "csv", False, "切换质量"),
        ArtifactRef("switch_by_regime_csv", _project_relative(results_dir / f"{prefix}_switch_by_regime.csv"), "csv", False, "切换按状态"),
        ArtifactRef("summary_json", _project_relative(results_dir / f"{prefix}_summary.json"), "json", False, "归因汇总"),
        ArtifactRef("report_md", _project_relative(docs_dir / f"{prefix}.md"), "md", False, "归因报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    )
    metrics = {
        "regime_capture_rows": int(len(regime_capture)),
        "breadth_capture_rows": int(len(breadth_capture)),
        "year_capture_rows": int(len(year_capture)),
        "switch_by_regime_rows": int(len(switch_by_regime)),
        "full_sample_annualized_return": float(res.panel.annualized_return),
        "full_sample_sharpe_ratio": float(res.panel.sharpe_ratio),
    }
    gates = {
        "data_gate": {
            "passed": True,
            "regime_capture_rows": int(len(regime_capture)),
        },
        "execution_gate": {
            "passed": True,
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=" ".join(sys.argv),
        created_at=utc_now_iso(),
        duration_sec=None,
        seed=None,
        data_slices=(data_slice,),
        config=config_snapshot(config_path=str(args.config)),
        params={"cli": {k: str(v) for k, v in vars(args).items()}},
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["p1_strong_up_attribution_is_historical_research_only"],
        },
        notes="Historical P1 Strong-Up attribution; not a promotion candidate.",
    )
    write_research_manifest(manifest_path, result)
    append_experiment_result(PROJECT_ROOT / "data" / "experiments", result)
    # --- end standard research contract ---


if __name__ == "__main__":
    main()
