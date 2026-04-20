"""V3 基准差距归因：月度超额、市场参与能力、持仓暴露、排名覆盖。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
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
    build_market_ew_benchmark,
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


FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("log_market_cap", "log_market_cap"),
    ("amount_20d", "amount_20d"),
    ("turnover_roll_mean", "turnover_roll_mean"),
    ("realized_vol", "realized_vol"),
    ("price_position", "price_position"),
    ("recent_return", "recent_return"),
    ("momentum_12_1", "momentum_12_1"),
    ("vol_to_turnover", "vol_to_turnover"),
)

RANK_BUCKET_ORDER = ["01_20", "21_40", "41_60", "61_100", "101_plus"]


@dataclass(frozen=True)
class ThresholdCase:
    trade_date: pd.Timestamp
    previous_rank: int
    current_rank: int
    score: float
    forward_return: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 V3 基准差距归因诊断")
    p.add_argument("--config", default="config.yaml.backtest.r7_s2_prefilter_off_universe_on", help="配置快照")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空则取今天")
    p.add_argument(
        "--output-prefix",
        default="benchmark_gap_diagnostics_2026-04-20_v3",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    return p.parse_args()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return _json_sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _compound_return(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    return float((1.0 + arr).prod() - 1.0)


def summarize_monthly_excess(strategy_daily: pd.Series, benchmark_daily: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = strategy_daily.index.intersection(benchmark_daily.index)
    strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0).sort_index()
    bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0).sort_index()
    monthly = pd.DataFrame(
        {
            "strategy_return": strat.resample("ME").apply(_compound_return),
            "benchmark_return": bench.resample("ME").apply(_compound_return),
        }
    ).dropna(how="all")
    if monthly.empty:
        return monthly.reset_index(names="month_end"), {}
    monthly["excess_return"] = monthly["strategy_return"] - monthly["benchmark_return"]
    monthly["year"] = monthly.index.year.astype(int)
    monthly["month"] = monthly.index.month.astype(int)
    monthly["benchmark_up_month"] = monthly["benchmark_return"] > 0
    monthly["strategy_up_month"] = monthly["strategy_return"] > 0

    neg = monthly.loc[monthly["excess_return"] < 0, "excess_return"]
    neg_abs = neg.abs()
    worst3_share = float(neg_abs.nlargest(min(3, len(neg_abs))).sum() / neg_abs.sum()) if neg_abs.sum() > 0 else np.nan
    summary = {
        "months": int(len(monthly)),
        "negative_excess_months": int((monthly["excess_return"] < 0).sum()),
        "negative_excess_ratio": float((monthly["excess_return"] < 0).mean()),
        "median_monthly_excess": float(monthly["excess_return"].median()),
        "mean_monthly_excess": float(monthly["excess_return"].mean()),
        "worst_monthly_excess": float(monthly["excess_return"].min()),
        "best_monthly_excess": float(monthly["excess_return"].max()),
        "worst_3_negative_share": worst3_share,
        "mild_negative_share": (
            float(((monthly["excess_return"] < 0) & (monthly["excess_return"] > -0.05)).sum() / max(len(neg), 1))
            if len(neg) > 0
            else np.nan
        ),
        "severe_negative_share": (
            float((monthly["excess_return"] <= -0.05).sum() / max(len(neg), 1)) if len(neg) > 0 else np.nan
        ),
    }
    return monthly.reset_index(names="month_end"), summary


def summarize_market_capture(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for label, mask in (
        ("benchmark_up", monthly_df["benchmark_return"] > 0),
        ("benchmark_down", monthly_df["benchmark_return"] < 0),
    ):
        part = monthly_df.loc[mask].copy()
        if part.empty:
            continue
        strat_comp = _compound_return(part["strategy_return"])
        bench_comp = _compound_return(part["benchmark_return"])
        capture_ratio = float(strat_comp / bench_comp) if np.isfinite(bench_comp) and abs(bench_comp) > 1e-12 else np.nan
        rows.append(
            {
                "regime": label,
                "months": int(len(part)),
                "benchmark_compound_return": bench_comp,
                "strategy_compound_return": strat_comp,
                "capture_ratio": capture_ratio,
                "median_benchmark_return": float(part["benchmark_return"].median()),
                "median_strategy_return": float(part["strategy_return"].median()),
                "median_excess_return": float(part["excess_return"].median()),
                "positive_strategy_share": float((part["strategy_return"] > 0).mean()),
                "positive_excess_share": float((part["excess_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_exposure_frame(daily_df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    d = daily_df[["symbol", "trade_date", "amount"]].copy()
    d = d.sort_values(["symbol", "trade_date"])
    d["amount_20d"] = (
        pd.to_numeric(d["amount"], errors="coerce").groupby(d["symbol"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
    )
    amt = d[["symbol", "trade_date", "amount_20d"]].drop_duplicates(["symbol", "trade_date"], keep="last")
    need_cols = ["symbol", "trade_date"] + [name for name, _ in FEATURE_SPECS if name in factors.columns]
    out = factors[need_cols].copy()
    out = out.merge(amt, on=["symbol", "trade_date"], how="left")
    return out


def summarize_exposures(
    weights: pd.DataFrame,
    exposure_df: pd.DataFrame,
    benchmark_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weights.empty or exposure_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for trade_date in pd.to_datetime(weights.index):
        w = weights.loc[trade_date]
        w = pd.to_numeric(w, errors="coerce")
        w = w[w > 0]
        if w.empty:
            continue
        day = exposure_df[exposure_df["trade_date"] == pd.Timestamp(trade_date)].copy()
        if day.empty:
            continue
        hold = day[day["symbol"].isin(w.index.astype(str))].copy()
        bench = day[day["symbol"].isin(benchmark_symbols)].copy()
        if hold.empty or bench.empty:
            continue
        hold["weight"] = hold["symbol"].map(w.to_dict()).astype(float)
        hold["weight"] = hold["weight"] / hold["weight"].sum()
        for feature, label in FEATURE_SPECS:
            if feature not in day.columns:
                continue
            hold_vals = pd.to_numeric(hold[feature], errors="coerce")
            hold_mask = hold_vals.notna() & np.isfinite(hold_vals)
            bench_vals = pd.to_numeric(bench[feature], errors="coerce")
            bench_mask = bench_vals.notna() & np.isfinite(bench_vals)
            if hold_mask.sum() == 0 or bench_mask.sum() == 0:
                continue
            hold_mean = float(np.average(hold_vals[hold_mask], weights=hold.loc[hold_mask, "weight"]))
            bench_mean = float(bench_vals[bench_mask].mean())
            bench_std = float(bench_vals[bench_mask].std(ddof=0))
            active_z = float((hold_mean - bench_mean) / bench_std) if bench_std > 1e-12 else np.nan
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "feature": label,
                    "holdings_weighted_mean": hold_mean,
                    "benchmark_equal_mean": bench_mean,
                    "active_diff": float(hold_mean - bench_mean),
                    "active_zscore": active_z,
                    "n_holdings": int(hold_mask.sum()),
                    "n_benchmark": int(bench_mask.sum()),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("feature", dropna=False)
        .agg(
            observations=("trade_date", "count"),
            mean_active_diff=("active_diff", "mean"),
            median_active_diff=("active_diff", "median"),
            mean_active_zscore=("active_zscore", "mean"),
            median_active_zscore=("active_zscore", "median"),
        )
        .reset_index()
        .sort_values("feature")
        .reset_index(drop=True)
    )
    return detail, summary


def _assign_rank_bucket(rank: int) -> str:
    if rank <= 20:
        return "01_20"
    if rank <= 40:
        return "21_40"
    if rank <= 60:
        return "41_60"
    if rank <= 100:
        return "61_100"
    return "101_plus"


def build_rank_coverage_tables(
    score_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    rebalance_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rebalance_dates = _rebalance_dates(score_df["trade_date"].unique(), rebalance_rule)
    if len(rebalance_dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    bucket_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    threshold_cases: list[ThresholdCase] = []
    prev_rank_map: dict[str, int] = {}

    for idx, trade_date in enumerate(rebalance_dates[:-1]):
        next_date = pd.Timestamp(rebalance_dates[idx + 1])
        day = score_df[score_df["trade_date"] == pd.Timestamp(trade_date)].copy()
        if day.empty:
            continue
        day = day.sort_values("score", ascending=False).reset_index(drop=True)
        day["rank"] = np.arange(1, len(day) + 1, dtype=int)
        period = asset_returns[(asset_returns.index > pd.Timestamp(trade_date)) & (asset_returns.index <= next_date)]
        if period.empty:
            continue
        fwd = (1.0 + period).prod(axis=0) - 1.0
        merged = day.merge(
            fwd.rename("forward_return"),
            left_on="symbol",
            right_index=True,
            how="left",
        )
        merged["forward_return"] = pd.to_numeric(merged["forward_return"], errors="coerce")
        merged = merged.dropna(subset=["forward_return"]).copy()
        if merged.empty:
            continue
        merged["year"] = pd.Timestamp(trade_date).year
        merged["bucket"] = merged["rank"].map(_assign_rank_bucket)
        detail_rows.append(merged[["trade_date", "year", "symbol", "rank", "score", "bucket", "forward_return"]])

        for bucket, grp in merged.groupby("bucket", dropna=False):
            bucket_rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "year": int(pd.Timestamp(trade_date).year),
                    "bucket": str(bucket),
                    "n_symbols": int(len(grp)),
                    "mean_forward_return": float(grp["forward_return"].mean()),
                    "median_forward_return": float(grp["forward_return"].median()),
                    "positive_share": float((grp["forward_return"] > 0).mean()),
                }
            )

        score_20 = merged.loc[merged["rank"] == 20, "score"]
        score_21 = merged.loc[merged["rank"] == 21, "score"]
        top20 = merged.loc[merged["rank"] <= 20, "symbol"].astype(str).tolist()
        top40 = merged.loc[merged["rank"] <= 40, "symbol"].astype(str).tolist()
        cur_rank_map = dict(zip(merged["symbol"].astype(str), merged["rank"]))
        carried_to_21_40 = [
            sym
            for sym, prev_rank in prev_rank_map.items()
            if prev_rank <= 20 and 21 <= cur_rank_map.get(sym, 10**9) <= 40
        ]
        carried_df = merged[merged["symbol"].astype(str).isin(carried_to_21_40)].copy()
        if not carried_df.empty:
            for row in carried_df.itertuples(index=False):
                threshold_cases.append(
                    ThresholdCase(
                        trade_date=pd.Timestamp(trade_date),
                        previous_rank=int(prev_rank_map[str(row.symbol)]),
                        current_rank=int(row.rank),
                        score=float(row.score),
                        forward_return=float(row.forward_return),
                    )
                )
        threshold_rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "year": int(pd.Timestamp(trade_date).year),
                "score_gap_20_21": (
                    float(score_20.iloc[0] - score_21.iloc[0]) if not score_20.empty and not score_21.empty else np.nan
                ),
                "top20_overlap_with_prev": float(len(set(top20) & set(prev_rank_map.keys())) / 20.0) if prev_rank_map else np.nan,
                "prev_top20_now_21_40_count": int(len(carried_to_21_40)),
                "prev_top20_now_21_40_mean_forward_return": (
                    float(carried_df["forward_return"].mean()) if not carried_df.empty else np.nan
                ),
                "prev_top20_now_21_40_positive_share": (
                    float((carried_df["forward_return"] > 0).mean()) if not carried_df.empty else np.nan
                ),
                "bucket_01_20_mean_forward_return": (
                    float(merged.loc[merged["bucket"] == "01_20", "forward_return"].mean())
                    if (merged["bucket"] == "01_20").any()
                    else np.nan
                ),
                "bucket_21_40_mean_forward_return": (
                    float(merged.loc[merged["bucket"] == "21_40", "forward_return"].mean())
                    if (merged["bucket"] == "21_40").any()
                    else np.nan
                ),
                "bucket_21_40_positive_share": (
                    float((merged.loc[merged["bucket"] == "21_40", "forward_return"] > 0).mean())
                    if (merged["bucket"] == "21_40").any()
                    else np.nan
                ),
                "top40_count": int(len(top40)),
            }
        )
        prev_rank_map = cur_rank_map

    bucket_detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if bucket_rows:
        bucket_summary = (
            pd.DataFrame(bucket_rows)
            .groupby("bucket", dropna=False)
            .agg(
                rebalances=("trade_date", "count"),
                mean_forward_return=("mean_forward_return", "mean"),
                median_forward_return=("median_forward_return", "median"),
                mean_positive_share=("positive_share", "mean"),
            )
            .reset_index()
        )
        bucket_summary["bucket"] = pd.Categorical(bucket_summary["bucket"], categories=RANK_BUCKET_ORDER, ordered=True)
        bucket_summary = bucket_summary.sort_values("bucket").reset_index(drop=True)
        bucket_summary["bucket"] = bucket_summary["bucket"].astype(str)
    else:
        bucket_summary = pd.DataFrame()
    threshold_df = pd.DataFrame(threshold_rows)
    summary = {
        "rebalances": int(len(threshold_df)),
        "median_score_gap_20_21": (
            float(pd.to_numeric(threshold_df["score_gap_20_21"], errors="coerce").median())
            if not threshold_df.empty
            else np.nan
        ),
        "median_prev_top20_now_21_40_count": (
            float(pd.to_numeric(threshold_df["prev_top20_now_21_40_count"], errors="coerce").median())
            if not threshold_df.empty
            else np.nan
        ),
        "mean_bucket_21_40_forward_return": (
            float(pd.to_numeric(threshold_df["bucket_21_40_mean_forward_return"], errors="coerce").mean())
            if not threshold_df.empty
            else np.nan
        ),
        "mean_bucket_01_20_forward_return": (
            float(pd.to_numeric(threshold_df["bucket_01_20_mean_forward_return"], errors="coerce").mean())
            if not threshold_df.empty
            else np.nan
        ),
        "buffer_case_count": int(len(threshold_cases)),
        "buffer_case_positive_share": (
            float(np.mean([case.forward_return > 0 for case in threshold_cases])) if threshold_cases else np.nan
        ),
        "buffer_case_mean_forward_return": (
            float(np.mean([case.forward_return for case in threshold_cases])) if threshold_cases else np.nan
        ),
    }
    return bucket_detail, bucket_summary, threshold_df, summary


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    monthly_summary: dict[str, Any],
    capture_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    rank_bucket_summary: pd.DataFrame,
    threshold_df: pd.DataFrame,
    industry_note: str,
    output_prefix: str,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    capture_md = capture_df.to_markdown(index=False) if not capture_df.empty else "_无可用月份数据_"
    yearly_md = yearly_df.to_markdown(index=False) if not yearly_df.empty else "_无年度数据_"
    exposure_md = exposure_summary.to_markdown(index=False) if not exposure_summary.empty else "_无暴露数据_"
    rank_md = rank_bucket_summary.to_markdown(index=False) if not rank_bucket_summary.empty else "_无覆盖数据_"
    threshold_cols = [
        "trade_date",
        "score_gap_20_21",
        "prev_top20_now_21_40_count",
        "prev_top20_now_21_40_mean_forward_return",
        "bucket_01_20_mean_forward_return",
        "bucket_21_40_mean_forward_return",
    ]
    threshold_view = threshold_df[threshold_cols].tail(10) if not threshold_df.empty else pd.DataFrame()
    threshold_md = threshold_view.to_markdown(index=False) if not threshold_view.empty else "_无阈值敏感度数据_"

    broad_or_concentrated = (
        "广泛小幅落后"
        if float(monthly_summary.get("worst_3_negative_share", np.nan) or np.nan) < 0.5
        else "少数极端月份主导"
    )
    capture_rows = {str(row["regime"]): row for row in capture_df.to_dict(orient="records")} if not capture_df.empty else {}
    up_row = capture_rows.get("benchmark_up", {})
    down_row = capture_rows.get("benchmark_down", {})
    exposure_rows = (
        {str(row["feature"]): row for row in exposure_summary.to_dict(orient="records")}
        if not exposure_summary.empty
        else {}
    )

    size_diff = exposure_rows.get("log_market_cap", {}).get("mean_active_zscore", np.nan)
    liquidity_diff = exposure_rows.get("amount_20d", {}).get("mean_active_zscore", np.nan)
    vol_diff = exposure_rows.get("realized_vol", {}).get("mean_active_zscore", np.nan)
    price_pos_diff = exposure_rows.get("price_position", {}).get("mean_active_zscore", np.nan)
    recent_return_diff = exposure_rows.get("recent_return", {}).get("mean_active_zscore", np.nan)

    if np.isfinite(size_diff) and np.isfinite(liquidity_diff):
        if size_diff > 0.5 and liquidity_diff > 0.5:
            exposure_takeaway = "当前持仓并不偏小盘或低流动性，反而整体偏向更大市值、成交额更高的股票。"
        elif size_diff < -0.5 and liquidity_diff < -0.5:
            exposure_takeaway = "当前持仓明显偏向更小市值、低流动性的股票，交易摩擦与 beta 覆盖不足需要一起警惕。"
        else:
            exposure_takeaway = "当前持仓在市值与流动性上没有出现单边极端偏移，弱势更可能来自排序目标与持有表达。"
    else:
        exposure_takeaway = "市值与流动性暴露证据不足，需谨慎解释持仓画像。"

    if np.isfinite(vol_diff) and np.isfinite(price_pos_diff) and np.isfinite(recent_return_diff):
        if vol_diff < -0.5 and price_pos_diff > 0 and recent_return_diff < 0:
            exposure_takeaway += " 同时，它偏低波动、价格位置较高但近期相对动量偏弱，更像在挑选稳定票，而不是追逐上涨扩散期的最强弹性。"
        elif vol_diff > 0.5 and recent_return_diff > 0.2:
            exposure_takeaway += " 同时，它偏高波动且近期强势，更接近进攻型暴露。"

    bucket_01_20 = rank_bucket_summary.loc[rank_bucket_summary["bucket"] == "01_20"]
    bucket_21_40 = rank_bucket_summary.loc[rank_bucket_summary["bucket"] == "21_40"]
    top20_mean = float(bucket_01_20["mean_forward_return"].iloc[0]) if not bucket_01_20.empty else np.nan
    top21_40_mean = float(bucket_21_40["mean_forward_return"].iloc[0]) if not bucket_21_40.empty else np.nan
    top21_40_median = float(bucket_21_40["median_forward_return"].iloc[0]) if not bucket_21_40.empty else np.nan
    threshold_mean = (
        float(pd.to_numeric(threshold_df["prev_top20_now_21_40_mean_forward_return"], errors="coerce").mean())
        if not threshold_df.empty
        else np.nan
    )
    if np.isfinite(top20_mean) and np.isfinite(top21_40_mean):
        if top21_40_mean >= top20_mean * 0.9:
            rank_takeaway = (
                f"`21-40` 桶的平均前向收益 `{top21_40_mean:.2%}` 已经非常接近 `01-20` 的 `{top20_mean:.2%}`，"
                "说明硬切 Top-20 会丢掉一部分仍具备收益能力的候选。"
            )
        else:
            rank_takeaway = (
                f"`21-40` 桶平均前向收益 `{top21_40_mean:.2%}` 明显低于 `01-20` 的 `{top20_mean:.2%}`，"
                "说明简单扩宽覆盖会稀释信号。"
            )
        if np.isfinite(top21_40_median) and top21_40_median < 0:
            rank_takeaway += f" 但它的中位前向收益仍为 `{top21_40_median:.2%}`，提示这更适合做缓冲持有，而不是直接线性扩仓。"
        if np.isfinite(threshold_mean) and threshold_mean > 0:
            rank_takeaway += f" 前一期 Top-20 掉到 `21-40` 的股票，平均后续收益仍有 `{threshold_mean:.2%}`，支持缓冲带思路。"
    else:
        rank_takeaway = "排名覆盖证据不足，暂时无法判断 Top-20 硬切是否误伤边界候选。"

    year_takeaway = (
        f"`2021/2025/2026` 的主要落后更符合“上涨月份参与不足”的模式：上涨月中位超额 `{up_row.get('median_excess_return', np.nan):.2%}`，"
        f"仅有 `{up_row.get('positive_excess_share', np.nan):.1%}` 的上涨月跑赢基准；"
        f"而下跌月中位超额 `{down_row.get('median_excess_return', np.nan):.2%}`，"
        f"有 `{down_row.get('positive_excess_share', np.nan):.1%}` 的下跌月跑赢基准。"
        if up_row and down_row
        else "缺少上涨/下跌月份拆解，暂时无法解释年度落后机制。"
    )
    return f"""# V3 基准差距归因

- 生成时间：`{generated_at}`
- 配置快照：`{config_source}`
- 固定口径：`score=S2=vol_to_turnover` / `top_k={params["top_k"]}` / `{params["rebalance_rule"]}` / `equal_weight` / `tplus1_open`
- 基准口径：`benchmark_symbol={params["benchmark_symbol"]}` / `benchmark_min_history_days={params["benchmark_min_history_days"]}`

## 结论摘要

1. 月度超额更接近“**{broad_or_concentrated}**”，负超额月份占比约 `{monthly_summary.get("negative_excess_ratio", np.nan):.1%}`，最差单月超额约 `{monthly_summary.get("worst_monthly_excess", np.nan):.2%}`。
2. {year_takeaway}
3. {exposure_takeaway}
4. {rank_takeaway}

## 年度超额

{yearly_md}

## 市场参与能力

{capture_md}

## 持仓暴露 vs 基准

{exposure_md}

行业分布：{industry_note}

## 排名覆盖

{rank_md}

## 进入 / 退出阈值近况

{threshold_md}

## 本轮产物

- `data/results/{output_prefix}_summary.json`
- `data/results/{output_prefix}_monthly.csv`
- `data/results/{output_prefix}_capture.csv`
- `data/results/{output_prefix}_exposure_summary.csv`
- `data/results/{output_prefix}_rank_bucket_summary.csv`
- `data/results/{output_prefix}_threshold.csv`
"""


def main() -> None:
    args = parse_args()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    cfg, config_source = load_config(args.config)
    signals = cfg.get("signals", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    uf_cfg = cfg.get("universe_filter", {}) or {}

    db_path = str(PROJECT_ROOT / str(cfg["paths"]["duckdb_path"]))
    top_k = int(signals.get("top_k", 20))
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    max_turnover = float(portfolio_cfg.get("max_turnover", 0.3))
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()

    print(f"[1/6] load daily data: start={args.start} end={end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)

    print("[2/6] compute factors + PIT + universe", flush=True)
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
    print(f"  factors={factors.shape}", flush=True)

    print("[3/6] rebuild V3 score/weights", flush=True)
    ce_weights = normalize_weights(signals.get("composite_extended", {}))
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
    print(f"  rebalances={len(weights)} symbols={weights.shape[1]}", flush=True)

    print("[4/6] run backtest + benchmark align", flush=True)
    if execution_mode != "tplus1_open":
        raise ValueError(f"本脚本当前仅支持 tplus1_open，收到: {execution_mode}")
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    symbol_universe = sorted(set(weights.columns.astype(str)))
    asset_returns = open_returns.reindex(columns=symbol_universe).fillna(0.0)
    asset_returns = asset_returns[(asset_returns.index >= pd.Timestamp(args.start)) & (asset_returns.index <= pd.Timestamp(end_date))]
    weights = weights.reindex(columns=symbol_universe, fill_value=0.0)
    if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
        seed = weights.iloc[[0]].copy()
        seed.index = pd.DatetimeIndex([asset_returns.index.min()])
        weights = pd.concat([seed, weights], axis=0)
        weights = weights[~weights.index.duplicated(keep="last")].sort_index()
    first_reb = weights.index.min()
    asset_returns = asset_returns[asset_returns.index >= first_reb]

    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    bt_cfg = BacktestConfig(
        cost_params=costs,
        execution_mode="tplus1_open",
        execution_lag=1,
        limit_up_mode=str(backtest_cfg.get("limit_up_mode", "redistribute")),
    )
    res = run_backtest(asset_returns, weights, config=bt_cfg)
    n_trade_days = int(asset_returns.index.nunique())
    benchmark_min_history_days = max(60, int(0.35 * max(n_trade_days, 1)))
    benchmark_market = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_history_days)

    print("[5/6] aggregate monthly/capture/exposure/coverage", flush=True)
    monthly_df, monthly_summary = summarize_monthly_excess(res.daily_returns, benchmark_market)
    capture_df = summarize_market_capture(monthly_df)
    yearly_df = (
        monthly_df.groupby("year", dropna=False)[["strategy_return", "benchmark_return"]]
        .apply(lambda frame: pd.Series(
            {
                "strategy_return": _compound_return(frame["strategy_return"]),
                "benchmark_return": _compound_return(frame["benchmark_return"]),
            }
        ))
        .reset_index()
    )
    if not yearly_df.empty:
        yearly_df["excess_return"] = yearly_df["strategy_return"] - yearly_df["benchmark_return"]

    exposure_df = _prepare_exposure_frame(daily_df, factors)
    symbol_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(symbol_counts[symbol_counts >= benchmark_min_history_days].index.astype(str))
    exposure_detail_df, exposure_summary_df = summarize_exposures(weights, exposure_df, benchmark_symbols)

    rank_detail_df, rank_bucket_summary_df, threshold_df, rank_summary = build_rank_coverage_tables(
        score_df=score_df,
        asset_returns=asset_returns,
        rebalance_rule=rebalance_rule,
    )

    print("[6/6] write outputs", flush=True)
    monthly_path = results_dir / f"{output_prefix}_monthly.csv"
    capture_path = results_dir / f"{output_prefix}_capture.csv"
    exposure_detail_path = results_dir / f"{output_prefix}_exposure_detail.csv"
    exposure_summary_path = results_dir / f"{output_prefix}_exposure_summary.csv"
    rank_detail_path = results_dir / f"{output_prefix}_rank_detail.csv"
    rank_summary_path = results_dir / f"{output_prefix}_rank_bucket_summary.csv"
    threshold_path = results_dir / f"{output_prefix}_threshold.csv"
    yearly_path = results_dir / f"{output_prefix}_yearly.csv"
    summary_path = results_dir / f"{output_prefix}_summary.json"
    doc_path = docs_dir / f"{output_prefix}.md"

    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    capture_df.to_csv(capture_path, index=False, encoding="utf-8-sig")
    exposure_detail_df.to_csv(exposure_detail_path, index=False, encoding="utf-8-sig")
    exposure_summary_df.to_csv(exposure_summary_path, index=False, encoding="utf-8-sig")
    rank_detail_df.to_csv(rank_detail_path, index=False, encoding="utf-8-sig")
    rank_bucket_summary_df.to_csv(rank_summary_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    industry_note = "缺少 `data/cache/industry_map.csv`，本轮未能输出行业分布对比。"
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": {
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
            "benchmark_min_history_days": benchmark_min_history_days,
        },
        "full_sample": {
            "annualized_return": float(res.panel.annualized_return),
            "sharpe_ratio": float(res.panel.sharpe_ratio),
            "max_drawdown": float(res.panel.max_drawdown),
            "turnover_mean": float(res.panel.turnover_mean),
        },
        "monthly_summary": monthly_summary,
        "capture_summary": _json_sanitize(capture_df.to_dict(orient="records")),
        "rank_summary": _json_sanitize(rank_summary),
        "industry_note": industry_note,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        config_source=config_source,
        params=summary["parameters"],
        monthly_summary=monthly_summary,
        capture_df=capture_df,
        yearly_df=yearly_df,
        exposure_summary=exposure_summary_df,
        rank_bucket_summary=rank_bucket_summary_df,
        threshold_df=threshold_df,
        industry_note=industry_note,
        output_prefix=output_prefix,
    )
    doc_path.write_text(doc_text, encoding="utf-8")

    print(f"  summary -> {summary_path}", flush=True)
    print(f"  doc     -> {doc_path}", flush=True)


if __name__ == "__main__":
    main()
