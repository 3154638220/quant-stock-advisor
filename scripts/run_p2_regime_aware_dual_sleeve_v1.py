"""R2 Day 6-7：Regime-aware dual sleeve v1。

固定口径延续 Day 3-5：
    defensive_sleeve = S2 = vol_to_turnover
    upside_sleeve = UPSIDE_C = limit_up_hits_20d + tail_strength_20d
    top_k=20 / M / equal_weight / max_turnover=1.0 / tplus1_open

关键约束：
    1. 只测试 plan 指定的 3 组权重；
    2. 状态权重使用上一已完成月份的 market regime / breadth，避免用当月收益事后调权；
    3. promotion 仍以 daily_bt_like_proxy_annualized_excess_vs_market 为准。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (  # noqa: E402
    attach_universe_filter,
    build_limit_up_open_mask,
    build_market_ew_open_to_open_benchmark,
    build_open_to_open_returns,
    load_config,
    load_daily_from_duckdb,
    transaction_cost_params_from_mapping,
)
from scripts.run_p1_strong_up_attribution import (  # noqa: E402
    REGIME_ORDER,
    _compound_return,
    _json_sanitize,
    build_groups_per_rebalance,
    build_switch_quality,
    classify_regimes,
    compute_breadth,
    compute_r1_extra_features,
    summarize_breadth_capture,
    summarize_regime_capture,
    summarize_switch_by_regime,
)
from scripts.run_p2_upside_sleeve_v1 import (  # noqa: E402
    GATE_FULL_BACKTEST,
    GATE_REJECT,
    _add_cs_rank,
    _accept_summary,
    _gate_decision,
)
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.models.xtree.p1_workflow import (  # noqa: E402
    build_tree_score_weight_matrix,
    summarize_tree_daily_backtest_like_proxy,
)


SLEEVE_RULES: list[dict[str, Any]] = [
    {
        "id": "BASELINE_S2",
        "label": "S2 vol_to_turnover (defensive baseline)",
        "strong_or_wide_upside": 0.0,
        "mild_or_mid_upside": 0.0,
        "neutral_upside": 0.0,
    },
    {
        "id": "DUAL_V1_80_20_TRIGGER_ONLY",
        "label": "lagged strong_up/wide: defensive 80% + upside 20%; otherwise defensive 100%",
        "strong_or_wide_upside": 0.20,
        "mild_or_mid_upside": 0.0,
        "neutral_upside": 0.0,
    },
    {
        "id": "DUAL_V2_60_40_MILD_85_15",
        "label": "lagged strong_up/wide: 60/40; mild_up/mid: 85/15; neutral/down/narrow defensive",
        "strong_or_wide_upside": 0.40,
        "mild_or_mid_upside": 0.15,
        "neutral_upside": 0.0,
    },
    {
        "id": "DUAL_V3_40_60_MILD_70_30_NEUTRAL_90_10",
        "label": "lagged strong_up/wide: 40/60; mild_up/mid: 70/30; neutral 90/10; down/narrow defensive",
        "strong_or_wide_upside": 0.60,
        "mild_or_mid_upside": 0.30,
        "neutral_upside": 0.10,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2 Day 6-7 regime-aware dual sleeve v1")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default="p2_regime_aware_dual_sleeve_v1_2026-04-28")
    return p.parse_args()


def _filter_universe(panel: pd.DataFrame) -> pd.DataFrame:
    if "_universe_eligible" not in panel.columns:
        return panel
    return panel[panel["_universe_eligible"].astype(bool)].copy()


def _long_with_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "candidate_id", cid)
    return out


def _attach_pit_roe_ttm(factors: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """轻量 PIT 对齐，仅为 universe_filter 提供 roe_ttm。"""
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
            out["roe_ttm"] = np.nan
            return out
        info = con.execute("PRAGMA table_info('a_share_fundamental')").fetchall()
        have = {str(r[1]) for r in info}
        if not {"symbol", "announcement_date", "roe_ttm"}.issubset(have):
            out["roe_ttm"] = np.nan
            return out
        fund = con.execute(
            """
            SELECT symbol, announcement_date, report_period, roe_ttm
            FROM a_share_fundamental
            WHERE announcement_date IS NOT NULL
            """
        ).df()
    finally:
        con.close()
    if fund.empty:
        out["roe_ttm"] = np.nan
        return out

    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    out = out.dropna(subset=["trade_date"])
    fund["symbol"] = fund["symbol"].astype(str).str.zfill(6)
    fund["announcement_date"] = (
        pd.to_datetime(fund["announcement_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    )
    fund["report_period"] = pd.to_datetime(fund.get("report_period"), errors="coerce").astype("datetime64[ns]")
    fund = fund.dropna(subset=["announcement_date"])
    if fund.empty:
        out["roe_ttm"] = np.nan
        return out
    fund = fund.sort_values(["symbol", "announcement_date", "report_period"], kind="mergesort")
    fund = fund.drop_duplicates(["symbol", "announcement_date"], keep="last")
    fund = fund[["symbol", "announcement_date", "roe_ttm"]]

    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    fund = fund.sort_values(["announcement_date", "symbol"], kind="mergesort").reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
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
            merged["roe_ttm"] = np.nan
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
            merged = merged.drop(columns=["announcement_date"], errors="ignore")
        chunks.append(merged)
    if not chunks:
        out["roe_ttm"] = np.nan
        return out
    return pd.concat(chunks, ignore_index=True)


def _compute_minimal_defensive_factors(daily_df: pd.DataFrame, min_hist_days: int) -> pd.DataFrame:
    """Day 6-7 只需要 S2，避免调用完整 compute_factors。"""
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    d["symbol"] = d["symbol"].astype(str).str.zfill(6)
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.normalize()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d["turnover"] = pd.to_numeric(d["turnover"], errors="coerce")
    hist = d.groupby("symbol", sort=False)["close"].transform("count")
    d = d[hist >= int(min_hist_days)].copy()
    d["vol_to_turnover"] = np.log1p(d["volume"] / d["turnover"].replace(0, np.nan)).fillna(0.0)
    return d[["symbol", "trade_date", "vol_to_turnover"]]


def build_sleeve_scores(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    for col in ("vol_to_turnover", "limit_up_hits_20d", "tail_strength_20d"):
        if col not in df.columns:
            raise KeyError(f"feature panel 缺少列: {col}")
        df = _add_cs_rank(df, col, f"_rk_{col}")
    df["score__defensive"] = df["_rk_vol_to_turnover"]
    upside_parts = df[["_rk_limit_up_hits_20d", "_rk_tail_strength_20d"]].to_numpy(dtype=float)
    df["score__upside_c"] = np.where(np.isnan(upside_parts).any(axis=1), np.nan, upside_parts.sum(axis=1))
    return df


def _monthly_benchmark_frame(bench_daily: pd.Series, breadth_series: pd.Series) -> pd.DataFrame:
    bench = bench_daily.copy()
    bench.index = pd.to_datetime(bench.index).normalize()
    monthly = pd.DataFrame({"benchmark_return": bench.resample("ME").apply(_compound_return)})
    monthly["strategy_return"] = 0.0
    monthly["excess_return"] = 0.0 - monthly["benchmark_return"]
    monthly = monthly.dropna(subset=["benchmark_return"]).reset_index(names="month_end")
    return classify_regimes(monthly, breadth_series, threshold_mode="expanding")


def _lagged_state_by_rebalance(rebalance_dates: pd.DatetimeIndex, monthly_state: pd.DataFrame) -> pd.DataFrame:
    states = monthly_state[
        [
            "month_end",
            "regime",
            "breadth",
            "benchmark_return",
            "breadth_value",
            "state_threshold_mode",
            "lookahead_check",
            "threshold_observations",
            "regime_p20",
            "regime_p40",
            "regime_p60",
            "regime_p80",
            "breadth_p30",
            "breadth_p70",
        ]
    ].copy()
    states["month_end"] = pd.to_datetime(states["month_end"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    for rd in pd.DatetimeIndex(rebalance_dates).sort_values():
        prev = states[states["month_end"] < pd.Timestamp(rd).normalize()]
        if prev.empty:
            regime = "neutral"
            breadth = "mid"
            source_month = pd.NaT
            bench_ret = np.nan
            breadth_value = np.nan
            threshold_mode = "expanding"
            lookahead_check = "pass"
            threshold_observations = 0
            regime_p20 = regime_p40 = regime_p60 = regime_p80 = np.nan
            breadth_p30 = breadth_p70 = np.nan
        else:
            r = prev.iloc[-1]
            regime = str(r["regime"])
            breadth = str(r["breadth"])
            source_month = pd.Timestamp(r["month_end"])
            bench_ret = float(r["benchmark_return"])
            breadth_value = float(r["breadth_value"])
            threshold_mode = str(r["state_threshold_mode"])
            lookahead_check = str(r["lookahead_check"])
            threshold_observations = int(r["threshold_observations"])
            regime_p20 = float(r["regime_p20"])
            regime_p40 = float(r["regime_p40"])
            regime_p60 = float(r["regime_p60"])
            regime_p80 = float(r["regime_p80"])
            breadth_p30 = float(r["breadth_p30"])
            breadth_p70 = float(r["breadth_p70"])
        rows.append(
            {
                "rebalance_date": pd.Timestamp(rd).normalize(),
                "lagged_state_month": source_month,
                "state_threshold_mode": threshold_mode,
                "state_threshold_source": f"{threshold_mode}_through_lagged_state_month",
                "state_threshold_observations": threshold_observations,
                "state_lag": "previous_completed_month",
                "state_feature_available_date": source_month,
                "lookahead_check": lookahead_check,
                "lagged_regime": regime,
                "lagged_breadth": breadth,
                "lagged_benchmark_return": bench_ret,
                "lagged_breadth_value": breadth_value,
                "regime_p20": regime_p20,
                "regime_p40": regime_p40,
                "regime_p60": regime_p60,
                "regime_p80": regime_p80,
                "breadth_p30": breadth_p30,
                "breadth_p70": breadth_p70,
            }
        )
    return pd.DataFrame(rows)


def _upside_weight(rule: dict[str, Any], regime: str, breadth: str) -> float:
    if regime in {"strong_down", "mild_down"} or breadth == "narrow":
        return 0.0
    if regime == "strong_up" or breadth == "wide":
        return float(rule["strong_or_wide_upside"])
    if regime == "mild_up" or breadth == "mid":
        return float(rule["mild_or_mid_upside"])
    return float(rule["neutral_upside"])


def _blend_weights(
    defensive_weights: pd.DataFrame,
    upside_weights: pd.DataFrame,
    state_by_rebalance: pd.DataFrame,
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(defensive_weights.columns) | set(upside_weights.columns))
    idx = defensive_weights.index.union(upside_weights.index).sort_values()
    dw = defensive_weights.reindex(index=idx, columns=cols, fill_value=0.0)
    uw = upside_weights.reindex(index=idx, columns=cols, fill_value=0.0)
    state_map = state_by_rebalance.set_index("rebalance_date")
    rows: list[pd.Series] = []
    diag_rows: list[dict[str, Any]] = []
    prev = pd.Series(0.0, index=cols, dtype=float)
    for rd in idx:
        if pd.Timestamp(rd) in state_map.index:
            st = state_map.loc[pd.Timestamp(rd)]
            regime = str(st["lagged_regime"])
            breadth = str(st["lagged_breadth"])
        else:
            regime = "neutral"
            breadth = "mid"
        up_w = _upside_weight(rule, regime, breadth)
        def_w = 1.0 - up_w
        blended = def_w * dw.loc[rd] + up_w * uw.loc[rd]
        total = float(blended.sum())
        if total > 0:
            blended = blended / total
        blended.name = pd.Timestamp(rd)
        turnover = 0.5 * float(np.abs(blended - prev).sum())
        rows.append(blended)
        diag_rows.append(
            {
                "trade_date": pd.Timestamp(rd),
                "candidate_id": rule["id"],
                "lagged_regime": regime,
                "lagged_breadth": breadth,
                "defensive_weight": def_w,
                "upside_weight": up_w,
                "turnover_half_l1": turnover,
                "n_defensive_names": int((dw.loc[rd] > 0).sum()),
                "n_upside_names": int((uw.loc[rd] > 0).sum()),
                "n_combined_names": int((blended > 0).sum()),
            }
        )
        prev = blended
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def _detail_from_weights(
    *,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    cost_params: Any,
    scenario: str,
    limit_up_open_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_cols = [
        "period",
        "trade_date",
        "strategy_return",
        "benchmark_return",
        "scenario",
        "excess_return",
        "benchmark_up",
        "strategy_up",
        "beat_benchmark",
        "turnover_half_l1",
    ]
    if weights.empty:
        return pd.DataFrame(columns=out_cols), {"n_rebalances": 0, "avg_turnover_half_l1": float("nan")}
    cols = sorted(set(weights.columns) | set(asset_returns.columns))
    ar = asset_returns.reindex(columns=cols).fillna(0.0).sort_index()
    ws = weights.reindex(columns=cols, fill_value=0.0).sort_index()
    start = pd.Timestamp(ws.index.min()).normalize()
    end = pd.Timestamp(ar.index.max()).normalize()
    ar = ar[(ar.index >= start) & (ar.index <= end)]
    bt = BacktestConfig(
        cost_params=cost_params,
        execution_mode="tplus1_open",
        execution_lag=1,
        limit_up_mode="redistribute",
        limit_up_open_mask=limit_up_open_mask.reindex(columns=cols, fill_value=False)
        if limit_up_open_mask is not None
        else None,
    )
    res = run_backtest(ar, ws, config=bt)
    bench = bench_daily.copy().sort_index()
    bench.index = pd.to_datetime(bench.index).normalize()
    bench = bench[(bench.index >= start) & (bench.index <= end)].astype(np.float64)
    common = res.daily_returns.index.intersection(bench.index).sort_values()
    if common.empty:
        return pd.DataFrame(columns=out_cols), {"n_rebalances": int(len(ws)), "avg_turnover_half_l1": float("nan")}
    detail = pd.DataFrame(
        {
            "period": pd.DatetimeIndex(common).strftime("%Y-%m-%d"),
            "trade_date": common,
            "strategy_return": res.daily_returns.reindex(common).fillna(0.0).to_numpy(dtype=float),
            "benchmark_return": bench.reindex(common).fillna(0.0).to_numpy(dtype=float),
            "scenario": scenario,
        }
    )
    detail["excess_return"] = detail["strategy_return"] - detail["benchmark_return"]
    detail["benchmark_up"] = detail["benchmark_return"] > 0.0
    detail["strategy_up"] = detail["strategy_return"] > 0.0
    detail["beat_benchmark"] = detail["excess_return"] > 0.0
    detail = detail.merge(
        res.rebalance_turnover.rename("turnover_half_l1").reset_index().rename(columns={"index": "trade_date"}),
        on="trade_date",
        how="left",
    )
    meta = {
        "n_rebalances": int(len(ws)),
        "avg_turnover_half_l1": float(pd.to_numeric(res.rebalance_turnover.dropna(), errors="coerce").mean()),
        "daily_periods_per_year": 252.0,
        "backtest_like_strategy_annualized_return_engine": float(res.panel.annualized_return),
        "backtest_like_strategy_max_drawdown_engine": float(res.panel.max_drawdown),
        "limit_up_detection": str(res.meta.get("limit_up_detection", "")),
        "buy_fail_event_count": int(res.meta.get("buy_fail_event_count", 0)),
        "buy_fail_total_weight": float(res.meta.get("buy_fail_total_weight", 0.0)),
        "buy_fail_redistributed_weight": float(res.meta.get("buy_fail_redistributed_weight", 0.0)),
        "buy_fail_idle_weight": float(res.meta.get("buy_fail_idle_weight", 0.0)),
        "buy_fail_diagnostic": res.meta.get("buy_fail_diagnostic", []),
    }
    return detail, meta


def _monthly_with_regime(detail: pd.DataFrame, breadth_series: pd.Series) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            columns=["month_end", "strategy_return", "benchmark_return", "excess_return", "regime", "breadth"]
        )
    d = detail.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
    d = d.set_index("trade_date").sort_index()
    monthly = pd.DataFrame(
        {
            "strategy_return": d["strategy_return"].resample("ME").apply(_compound_return),
            "benchmark_return": d["benchmark_return"].resample("ME").apply(_compound_return),
        }
    )
    monthly["excess_return"] = monthly["strategy_return"] - monthly["benchmark_return"]
    monthly = monthly.dropna(how="all").reset_index(names="month_end")
    return classify_regimes(monthly, breadth_series)


def _score_for_switch_groups(panel: pd.DataFrame, state_by_rebalance: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    state_map = state_by_rebalance.set_index("rebalance_date")
    df = panel[["symbol", "trade_date", "score__defensive", "score__upside_c"]].copy()
    scores = []
    for rd, part in df.groupby("trade_date", sort=False):
        rd_ts = pd.Timestamp(rd).normalize()
        if rd_ts in state_map.index:
            st = state_map.loc[rd_ts]
            up_w = _upside_weight(rule, str(st["lagged_regime"]), str(st["lagged_breadth"]))
        else:
            up_w = 0.0
        scores.append((1.0 - up_w) * part["score__defensive"] + up_w * part["score__upside_c"])
    df["score"] = pd.concat(scores).sort_index()
    return df[["symbol", "trade_date", "score"]].dropna(subset=["score"])


def _run_one_rule(
    *,
    rule: dict[str, Any],
    panel: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    upside_weights: pd.DataFrame,
    state_by_rebalance: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    breadth_series: pd.Series,
    cost_params: Any,
    limit_up_open_mask: pd.DataFrame | None,
) -> dict[str, Any]:
    if rule["id"] == "BASELINE_S2":
        weights = defensive_weights.copy()
        diag = pd.DataFrame(
            {
                "trade_date": weights.index,
                "candidate_id": rule["id"],
                "lagged_regime": "baseline",
                "lagged_breadth": "baseline",
                "defensive_weight": 1.0,
                "upside_weight": 0.0,
                "turnover_half_l1": np.nan,
                "n_combined_names": (weights > 0).sum(axis=1).to_numpy(dtype=int),
            }
        )
    else:
        weights, diag = _blend_weights(defensive_weights, upside_weights, state_by_rebalance, rule)
    detail, meta = _detail_from_weights(
        weights=weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        cost_params=cost_params,
        scenario=rule["id"],
        limit_up_open_mask=limit_up_open_mask,
    )
    summary = summarize_tree_daily_backtest_like_proxy(detail)
    monthly = _monthly_with_regime(detail, breadth_series)
    regime_capture = summarize_regime_capture(monthly)
    breadth_capture = summarize_breadth_capture(monthly)
    score_df = _score_for_switch_groups(panel, state_by_rebalance, rule)
    rebalance_actual = sorted(pd.to_datetime(weights.index).normalize().unique().tolist())
    groups = build_groups_per_rebalance(score_df, rebalance_actual)
    switch_df = build_switch_quality(groups, monthly, asset_returns)
    switch_by_regime = summarize_switch_by_regime(switch_df)
    year_rows: list[dict[str, Any]] = []
    for year in (2021, 2025, 2026):
        for regime in REGIME_ORDER:
            part = monthly[(monthly["month_end"].dt.year == year) & (monthly["regime"] == regime)]
            if part.empty:
                continue
            year_rows.append(
                {
                    "year": year,
                    "regime": regime,
                    "months": int(len(part)),
                    "median_excess_return": float(part["excess_return"].median()),
                    "positive_excess_share": float((part["excess_return"] > 0).mean()),
                }
            )
    return {
        "candidate": rule,
        "summary": summary,
        "meta": meta,
        "monthly": monthly,
        "regime_capture": regime_capture,
        "breadth_capture": breadth_capture,
        "year_capture": pd.DataFrame(year_rows),
        "switch_by_regime": switch_by_regime,
        "switch_detail": switch_df,
        "state_diag": diag,
    }


def _build_leaderboard(results: list[dict[str, Any]], baseline_id: str) -> pd.DataFrame:
    base = next(r for r in results if r["candidate"]["id"] == baseline_id)

    def _regime_row(rc: pd.DataFrame, regime: str) -> dict[str, float]:
        sub = rc[rc["regime"] == regime] if not rc.empty else pd.DataFrame()
        if sub.empty:
            return {"median_excess": np.nan, "positive_share": np.nan, "capture": np.nan}
        r = sub.iloc[0]
        return {
            "median_excess": float(r["median_excess_return"]),
            "positive_share": float(r["positive_excess_share"]),
            "capture": float(r.get("capture_ratio", np.nan)),
        }

    def _switch_strong_up(sw: pd.DataFrame) -> dict[str, float]:
        sub = sw[sw["regime"] == "strong_up"] if not sw.empty else pd.DataFrame()
        if sub.empty:
            return {"strong_up_switch_in_minus_out": np.nan, "strong_up_topk_minus_next": np.nan}
        r = sub.iloc[0]
        return {
            "strong_up_switch_in_minus_out": float(r["mean_switch_in_minus_out"]),
            "strong_up_topk_minus_next": float(r["mean_topk_minus_next"]),
        }

    base_su = _regime_row(base["regime_capture"], "strong_up")
    base_sd = _regime_row(base["regime_capture"], "strong_down")
    base_proxy = float(base["summary"].get("annualized_excess_vs_market", np.nan))
    base_turnover = float(base["meta"].get("avg_turnover_half_l1", np.nan))
    rows: list[dict[str, Any]] = []
    for r in results:
        proxy = float(r["summary"].get("annualized_excess_vs_market", np.nan))
        su = _regime_row(r["regime_capture"], "strong_up")
        sd = _regime_row(r["regime_capture"], "strong_down")
        sw = _switch_strong_up(r["switch_by_regime"])
        avg_turnover = float(r["meta"].get("avg_turnover_half_l1", np.nan))
        rows.append(
            {
                "candidate_id": r["candidate"]["id"],
                "label": r["candidate"]["label"],
                "daily_proxy_annualized_excess_vs_market": proxy,
                "delta_vs_baseline_proxy": proxy - base_proxy,
                "gate_decision": _gate_decision(proxy),
                "strong_up_median_excess": su["median_excess"],
                "delta_vs_baseline_strong_up_median_excess": su["median_excess"] - base_su["median_excess"],
                "strong_up_positive_share": su["positive_share"],
                "delta_vs_baseline_strong_up_positive_share": su["positive_share"] - base_su["positive_share"],
                "strong_up_capture": su["capture"],
                "strong_down_median_excess": sd["median_excess"],
                "delta_vs_baseline_strong_down_median_excess": sd["median_excess"] - base_sd["median_excess"],
                "strong_up_switch_in_minus_out": sw["strong_up_switch_in_minus_out"],
                "strong_up_topk_minus_next": sw["strong_up_topk_minus_next"],
                "avg_turnover_half_l1": avg_turnover,
                "delta_vs_baseline_turnover": avg_turnover - base_turnover,
                "n_periods": int(r["summary"].get("n_periods", 0)),
                "n_rebalances": int(r["meta"].get("n_rebalances", 0)),
                "primary_result_type": "daily_bt_like_proxy",
                "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
                "p1_experiment_mode": "daily_proxy_first",
                "legacy_proxy_decision_role": "diagnostic_only",
            }
        )
    out = pd.DataFrame(rows)
    out["_sort"] = out["candidate_id"].apply(lambda x: 0 if x == baseline_id else 1)
    out["_proxy_sort"] = -pd.to_numeric(out["daily_proxy_annualized_excess_vs_market"], errors="coerce")
    return out.sort_values(["_sort", "_proxy_sort"]).drop(columns=["_sort", "_proxy_sort"]).reset_index(drop=True)


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    leaderboard: pd.DataFrame,
    accept_map: dict[str, dict[str, Any]],
    regime_long: pd.DataFrame,
    breadth_long: pd.DataFrame,
    year_long: pd.DataFrame,
    switch_long: pd.DataFrame,
    state_diag_long: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2 Regime-Aware Dual Sleeve v1 (Day 6-7)\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append("- defensive sleeve：`S2 = vol_to_turnover`")
    lines.append("- upside sleeve：`UPSIDE_C = limit_up_hits_20d + tail_strength_20d`")
    lines.append("- 状态输入：上一已完成月份的 `regime / breadth`，避免当月事后调权")
    lines.append("- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open`")
    lines.append("- `primary_decision_metric`: `daily_bt_like_proxy_annualized_excess_vs_market`")
    lines.append("- Gate：`<0%`→reject / `0%~+3%`→gray_zone / `>=+3%`→full_backtest_candidate")
    lines.append("")
    lines.append("## 1. Leaderboard\n")
    cols_show = [
        "candidate_id",
        "label",
        "daily_proxy_annualized_excess_vs_market",
        "delta_vs_baseline_proxy",
        "gate_decision",
        "strong_up_median_excess",
        "delta_vs_baseline_strong_up_median_excess",
        "strong_up_positive_share",
        "delta_vs_baseline_strong_up_positive_share",
        "strong_down_median_excess",
        "delta_vs_baseline_strong_down_median_excess",
        "strong_up_switch_in_minus_out",
        "avg_turnover_half_l1",
        "delta_vs_baseline_turnover",
        "n_rebalances",
    ]
    lines.append(leaderboard[[c for c in cols_show if c in leaderboard.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 2. R2 验收（对 BASELINE 的相对改善）\n")
    accept_rows = []
    for cid, info in accept_map.items():
        row = {"candidate_id": cid, "status": info["status"]}
        row.update(info.get("checks", {}))
        accept_rows.append(row)
    lines.append(pd.DataFrame(accept_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 状态权重触发统计\n")
    if state_diag_long.empty:
        lines.append("_无状态诊断_")
    else:
        trigger = (
            state_diag_long.groupby(["candidate_id", "upside_weight"], dropna=False)
            .agg(rebalances=("trade_date", "count"), avg_combined_names=("n_combined_names", "mean"))
            .reset_index()
            .sort_values(["candidate_id", "upside_weight"])
        )
        lines.append(trigger.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 4. Regime 切片（candidate × regime）\n")
    lines.append(regime_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 5. Breadth 切片（candidate × breadth）\n")
    lines.append(breadth_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 6. 关键年份 strong_up（2021/2025/2026）\n")
    sub = year_long[year_long["regime"] == "strong_up"] if not year_long.empty else pd.DataFrame()
    lines.append("_strong_up 无样本_" if sub.empty else sub.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 7. Switch quality（candidate × regime）\n")
    lines.append("_无 switch 样本_" if switch_long.empty else switch_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 8. 结论 / 下一步\n")
    base_id = "BASELINE_S2"
    passed = [cid for cid, info in accept_map.items() if cid != base_id and info["status"] == "pass"]
    candidates = leaderboard[leaderboard["candidate_id"] != base_id].copy()
    non_negative = candidates[candidates["daily_proxy_annualized_excess_vs_market"] >= 0]["candidate_id"].tolist()
    if passed:
        lines.append(f"- 通过 R2 验收的双袖套候选：{', '.join(passed)}。")
    else:
        lines.append("- **没有双袖套候选同时满足 R2 验收四条**。")
    if non_negative:
        lines.append(f"- daily proxy 非负候选：{', '.join(non_negative)}；其中 `>= +3%` 才允许补正式 full backtest。")
    else:
        lines.append("- 三组双袖套 daily proxy 仍低于 0%，按 R0 不补正式 full backtest。")
    lines.append("- 若保守 80/20 仍无法改善 strong_up 且 proxy 为负，下一步应回到候选层重做 upside 输入，而不是继续加大 sleeve 权重。")
    lines.append("")
    lines.append("## 9. 产出文件\n")
    for suf in [
        "leaderboard.csv",
        "regime_long.csv",
        "breadth_long.csv",
        "year_long.csv",
        "switch_long.csv",
        "state_diag_long.csv",
        "monthly_long.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suf}`")
    lines.append("")
    lines.append("## 10. 配置参数\n")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    return "\n".join(lines)


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

    print(f"[1/7] load daily {args.start}->{end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)

    print("[2/7] compute factors + lightweight PIT roe + R1 extras + universe filter", flush=True)
    print("  [2a] compute minimal S2 factors", flush=True)
    factors = _compute_minimal_defensive_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  [2b] attach lightweight PIT roe: factors={factors.shape}", flush=True)
    factors = _attach_pit_roe_ttm(factors, db_path)
    print("  [2c] attach universe filter", flush=True)
    factors = attach_universe_filter(
        factors,
        daily_df,
        enabled=bool(uf_cfg.get("enabled", False)),
        min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
    )
    print("  [2d] compute R1 extras", flush=True)
    extras = compute_r1_extra_features(daily_df)
    print("  [2e] merge panel + build scores", flush=True)
    panel = factors.merge(extras, on=["symbol", "trade_date"], how="left")
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start)].copy()
    panel = _filter_universe(panel)
    panel = build_sleeve_scores(panel)
    print(f"  panel(after filter)={panel.shape}", flush=True)

    print("[3/7] precompute open-to-open returns + market_ew benchmark + monthly states", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    limit_up_open_mask = build_limit_up_open_mask(daily_df).sort_index()
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_state = _monthly_benchmark_frame(bench_daily, breadth_series)

    print("[4/7] build defensive/upside sleeve weight matrices", flush=True)
    score_base = panel[["symbol", "trade_date", "score__defensive"]].dropna().copy()
    score_upside = panel[["symbol", "trade_date", "score__upside_c"]].dropna().copy()
    defensive_weights, defensive_diag = build_tree_score_weight_matrix(
        score_base,
        score_col="score__defensive",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    upside_weights, upside_diag = build_tree_score_weight_matrix(
        score_upside,
        score_col="score__upside_c",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    state_by_rebalance = _lagged_state_by_rebalance(defensive_weights.index, monthly_state)

    print("[5/7] run daily proxy for baseline + 3 dual-sleeve rules", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    results = []
    for rule in SLEEVE_RULES:
        print(f"  -> {rule['id']}", flush=True)
        results.append(
            _run_one_rule(
                rule=rule,
                panel=panel,
                defensive_weights=defensive_weights,
                upside_weights=upside_weights,
                state_by_rebalance=state_by_rebalance,
                asset_returns=asset_returns,
                bench_daily=bench_daily,
                breadth_series=breadth_series,
                cost_params=cost_params,
                limit_up_open_mask=limit_up_open_mask,
            )
        )

    print("[6/7] aggregate diagnostics", flush=True)
    leaderboard = _build_leaderboard(results, baseline_id="BASELINE_S2")
    base_row = leaderboard[leaderboard["candidate_id"] == "BASELINE_S2"].iloc[0]
    accept_map = {row["candidate_id"]: _accept_summary(row, base_row) for _, row in leaderboard.iterrows()}
    regime_long = pd.concat([_long_with_id(r["regime_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    breadth_long = pd.concat([_long_with_id(r["breadth_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    year_long = pd.concat([_long_with_id(r["year_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    switch_long = pd.concat([_long_with_id(r["switch_by_regime"], r["candidate"]["id"]) for r in results], ignore_index=True)
    monthly_long = pd.concat([_long_with_id(r["monthly"], r["candidate"]["id"]) for r in results], ignore_index=True)
    state_diag_long = pd.concat([r["state_diag"] for r in results], ignore_index=True)

    print("[7/7] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    leaderboard.to_csv(results_dir / f"{prefix}_leaderboard.csv", index=False, encoding="utf-8-sig")
    regime_long.to_csv(results_dir / f"{prefix}_regime_long.csv", index=False, encoding="utf-8-sig")
    breadth_long.to_csv(results_dir / f"{prefix}_breadth_long.csv", index=False, encoding="utf-8-sig")
    year_long.to_csv(results_dir / f"{prefix}_year_long.csv", index=False, encoding="utf-8-sig")
    switch_long.to_csv(results_dir / f"{prefix}_switch_long.csv", index=False, encoding="utf-8-sig")
    monthly_long.to_csv(results_dir / f"{prefix}_monthly_long.csv", index=False, encoding="utf-8-sig")
    state_diag_long.to_csv(results_dir / f"{prefix}_state_diag_long.csv", index=False, encoding="utf-8-sig")
    state_by_rebalance.to_csv(results_dir / f"{prefix}_lagged_state_by_rebalance.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "dual_sleeve_blended_equal_weight",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "state_lag": "previous_completed_month",
        "defensive_sleeve": "vol_to_turnover",
        "upside_sleeve": "limit_up_hits_20d + tail_strength_20d",
        "prefilter": prefilter_cfg,
        "universe_filter": uf_cfg,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
        "benchmark_min_history_days": bench_min,
        "config_source": config_source,
        "p1_experiment_mode": "daily_proxy_first",
        "legacy_proxy_decision_role": "diagnostic_only",
        "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
        "gate_thresholds": {"reject": GATE_REJECT, "full_backtest": GATE_FULL_BACKTEST},
        "defensive_weight_diag_rows": int(len(defensive_diag)),
        "upside_weight_diag_rows": int(len(upside_diag)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "rules": SLEEVE_RULES,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "accept": accept_map,
        "monthly_state_thresholds": {
            "regime": monthly_state.attrs.get("regime_thresholds"),
            "breadth": monthly_state.attrs.get("breadth_thresholds"),
            "trace": monthly_state.attrs.get("threshold_trace"),
        },
        "candidates": [
            {
                "id": r["candidate"]["id"],
                "label": r["candidate"]["label"],
                "summary": r["summary"],
                "meta": r["meta"],
            }
            for r in results
        ],
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        config_source=config_source,
        params=params,
        leaderboard=leaderboard,
        accept_map=accept_map,
        regime_long=regime_long,
        breadth_long=breadth_long,
        year_long=year_long,
        switch_long=switch_long,
        state_diag_long=state_diag_long,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc_text, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)


if __name__ == "__main__":
    main()
