#!/usr/bin/env python3
"""M8: 月度选股行业集中度约束与 lagged-regime 复核。

本脚本消费 M2 canonical dataset，复用 M5/M6 的 walk-forward score 生成逻辑，
在选择层评估 unconstrained 与 industry names cap，并构造一个只使用信号日
可见状态的 regime-aware fixed policy 候选。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import slugify_token
from scripts.run_monthly_selection_baselines import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    POOL_RULES,
    _format_markdown_table,
    _json_sanitize,
    _rank_pct_score,
    model_n_jobs_token,
    normalize_model_n_jobs,
    build_quantile_spread,
    build_rank_ic,
    build_realized_market_states,
    load_baseline_dataset,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
    summarize_industry_exposure,
    summarize_regime_slice,
    summarize_year_slice,
    valid_pool_frame,
)
from scripts.run_monthly_selection_ltr import (
    M6RunConfig,
    build_m6_feature_spec,
    build_walk_forward_ltr_scores,
    summarize_ltr_feature_importance,
)
from scripts.run_monthly_selection_multisource import (
    M5RunConfig,
    attach_enabled_families,
    build_all_m5_scores,
    build_feature_specs,
    summarize_feature_coverage_by_spec,
    summarize_feature_importance as summarize_m5_feature_importance,
)
from src.settings import load_config, resolve_config_path


STABLE_M5_ELASTICNET = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess"
STABLE_M5_EXTRATREES = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess"
WATCHLIST_M6_RANK = "M6_xgboost_rank_ndcg"
WATCHLIST_M6_TOP20 = "M6_top20_calibrated"
M8_POLICY_MODEL = "M8_regime_aware_fixed_policy"


@dataclass(frozen=True)
class M8RunConfig:
    top_ks: tuple[int, ...] = (20, 30, 50)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    min_state_history_months: int = 24
    cap_grid: dict[int, tuple[int, ...]] | None = None
    model_n_jobs: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M8 行业集中度与 lagged-regime 治理")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m8_concentration_regime")
    p.add_argument("--as-of-date", type=str, default="", help="输出文件日期；默认使用当前日期。")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--cap-grid", type=str, default="20:3,4,5;30:4,5,6,8;50:6,8,10")
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--max-fit-rows", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument(
        "--model-n-jobs",
        type=int,
        default=0,
        help="模型训练线程数；0 表示使用全部 CPU 核心，1 保持旧的单线程行为。",
    )
    p.add_argument("--min-state-history-months", type=int, default=24)
    p.add_argument("--families", type=str, default="industry_breadth,fund_flow,fundamental")
    p.add_argument("--skip-m6", action="store_true", help="仅运行 M5 稳定底座和行业 cap，跳过 M6 watchlist。")
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_cap_grid(raw: str, top_ks: list[int]) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for chunk in str(raw).split(";"):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            raise ValueError(f"cap-grid 片段缺少 ':': {chunk}")
        k_raw, vals_raw = chunk.split(":", 1)
        k = int(k_raw.strip())
        vals = tuple(sorted({int(x.strip()) for x in vals_raw.split(",") if x.strip()}))
        if vals:
            out[k] = vals
    for k in top_ks:
        out.setdefault(k, tuple())
    return out


def _industry_threshold(top_k: int) -> float:
    if int(top_k) <= 20:
        return 0.30
    if int(top_k) <= 30:
        return 0.35
    return 0.35


def select_with_industry_cap(part: pd.DataFrame, *, k: int, max_industry_names: int) -> pd.DataFrame:
    """Greedy score selection with an optional max names per industry cap."""
    if part.empty or int(k) <= 0:
        return part.iloc[0:0].copy()
    ordered = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
    if int(max_industry_names) <= 0:
        return ordered.head(int(k)).copy()
    counts: dict[str, int] = {}
    selected_idx: list[Any] = []
    for idx, row in ordered.iterrows():
        industry = str(row.get("industry_level1") if pd.notna(row.get("industry_level1")) else "_UNKNOWN_")
        if counts.get(industry, 0) >= int(max_industry_names):
            continue
        selected_idx.append(idx)
        counts[industry] = counts.get(industry, 0) + 1
        if len(selected_idx) >= int(k):
            break
    return ordered.loc[selected_idx].copy()


def _weighted_turnover(prev: set[str] | None, cur: set[str]) -> float:
    if prev is None:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


def build_constrained_monthly(
    scores: pd.DataFrame,
    *,
    top_ks: list[int],
    cap_grid: dict[int, tuple[int, ...]],
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    prev_by_key: dict[tuple[str, str, int, int], set[str]] = {}
    ordered = scores.sort_values(
        ["candidate_pool_version", "model", "signal_date", "score", "symbol"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    for (pool, base_model, signal_date), part in ordered.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        part = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
        candidate_width = int(part["symbol"].nunique())
        market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
        pool_ret = float(pd.to_numeric(part[LABEL_COL], errors="coerce").mean())
        model_type = str(part["model_type"].dropna().iloc[0]) if part["model_type"].notna().any() else ""
        for k in top_ks:
            caps = [0, *[int(x) for x in cap_grid.get(int(k), tuple())]]
            for cap in caps:
                selected = select_with_industry_cap(part, k=int(k), max_industry_names=int(cap))
                if selected.empty:
                    continue
                remaining = part.drop(index=selected.index, errors="ignore")
                nextk = select_with_industry_cap(remaining, k=int(k), max_industry_names=int(cap))
                variant_model = f"{base_model}__{'uncapped' if int(cap) <= 0 else f'indcap{int(cap)}'}"
                cur_symbols = set(selected["symbol"].astype(str))
                turnover = _weighted_turnover(prev_by_key.get((pool, variant_model, int(k), int(cap))), cur_symbols)
                prev_by_key[(pool, variant_model, int(k), int(cap))] = cur_symbols
                top_ret = float(pd.to_numeric(selected[LABEL_COL], errors="coerce").mean())
                next_ret = float(pd.to_numeric(nextk[LABEL_COL], errors="coerce").mean()) if not nextk.empty else np.nan
                top_ind_excess = (
                    float(pd.to_numeric(selected[INDUSTRY_EXCESS_COL], errors="coerce").mean())
                    if INDUSTRY_EXCESS_COL in selected.columns
                    else np.nan
                )
                h = selected.copy()
                h["base_model"] = base_model
                h["model"] = variant_model
                h["base_model_type"] = model_type
                h["model_type"] = "unconstrained_selection" if int(cap) <= 0 else "industry_cap_selection"
                h["top_k"] = int(k)
                h["selected_rank"] = np.arange(1, len(h) + 1)
                h["selection_policy"] = "uncapped" if int(cap) <= 0 else "industry_names_cap"
                h["max_industry_names"] = int(cap)
                keep_cols = [
                    "signal_date",
                    "candidate_pool_version",
                    "base_model",
                    "model",
                    "base_model_type",
                    "model_type",
                    "selection_policy",
                    "max_industry_names",
                    "top_k",
                    "selected_rank",
                    "symbol",
                    "score",
                    LABEL_COL,
                    "industry_level1",
                    "risk_flags",
                ]
                holdings.append(h[[c for c in keep_cols if c in h.columns]].copy())
                rows.append(
                    {
                        "signal_date": signal_date,
                        "candidate_pool_version": pool,
                        "base_model": base_model,
                        "model": variant_model,
                        "base_model_type": model_type,
                        "model_type": "unconstrained_selection" if int(cap) <= 0 else "industry_cap_selection",
                        "selection_policy": "uncapped" if int(cap) <= 0 else "industry_names_cap",
                        "max_industry_names": int(cap),
                        "top_k": int(k),
                        "candidate_pool_width": candidate_width,
                        "selected_count": int(len(selected)),
                        "topk_return": top_ret,
                        "market_ew_return": market_ret,
                        "topk_excess_vs_market": float(top_ret - market_ret) if np.isfinite(market_ret) else np.nan,
                        "topk_industry_neutral_excess": top_ind_excess,
                        "candidate_pool_mean_return": pool_ret,
                        "topk_minus_pool_mean": float(top_ret - pool_ret),
                        "nextk_return": next_ret,
                        "topk_minus_nextk": float(top_ret - next_ret) if np.isfinite(next_ret) else np.nan,
                        "turnover_half_l1": turnover,
                        "cost_bps": float(cost_bps),
                        "cost_drag": float(turnover * cost_bps / 10000.0) if np.isfinite(turnover) else np.nan,
                        "topk_excess_after_cost": float(top_ret - market_ret - turnover * cost_bps / 10000.0)
                        if np.isfinite(market_ret) and np.isfinite(turnover)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()


def summarize_industry_concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    h = holdings.copy()
    h["industry_level1"] = h["industry_level1"].fillna("_UNKNOWN_").astype(str)
    counts = (
        h.groupby(
            [
                "signal_date",
                "candidate_pool_version",
                "base_model",
                "model",
                "top_k",
                "selection_policy",
                "max_industry_names",
                "industry_level1",
            ],
            sort=True,
        )["symbol"]
        .nunique()
        .rename("industry_symbol_count")
        .reset_index()
    )
    total = (
        counts.groupby(
            [
                "signal_date",
                "candidate_pool_version",
                "base_model",
                "model",
                "top_k",
                "selection_policy",
                "max_industry_names",
            ],
            sort=True,
        )
        .agg(
            selected_count=("industry_symbol_count", "sum"),
            industry_count=("industry_level1", "nunique"),
            max_industry_count=("industry_symbol_count", "max"),
        )
        .reset_index()
    )
    total["max_industry_share"] = total["max_industry_count"] / total["selected_count"].replace(0, np.nan)
    total["concentration_threshold"] = total["top_k"].map(_industry_threshold)
    total["concentration_pass"] = total["max_industry_share"] <= total["concentration_threshold"]
    return total


def _summarize_concentration_for_leaderboard(industry_concentration: pd.DataFrame) -> pd.DataFrame:
    if industry_concentration.empty:
        return pd.DataFrame()
    return (
        industry_concentration.groupby(
            ["candidate_pool_version", "model", "top_k", "selection_policy", "max_industry_names"], sort=True
        )
        .agg(
            max_industry_share_mean=("max_industry_share", "mean"),
            max_industry_share_max=("max_industry_share", "max"),
            industry_count_mean=("industry_count", "mean"),
            concentration_pass_rate=("concentration_pass", "mean"),
        )
        .reset_index()
    )


def build_constrained_leaderboard(
    monthly: pd.DataFrame,
    rank_ic: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    regime_slice: pd.DataFrame,
    industry_concentration: pd.DataFrame,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    agg = (
        monthly.groupby(
            [
                "candidate_pool_version",
                "base_model",
                "model",
                "base_model_type",
                "model_type",
                "selection_policy",
                "max_industry_names",
                "top_k",
            ],
            sort=True,
        )
        .agg(
            months=("signal_date", "nunique"),
            mean_topk_return=("topk_return", "mean"),
            median_topk_return=("topk_return", "median"),
            topk_excess_mean=("topk_excess_vs_market", "mean"),
            topk_excess_median=("topk_excess_vs_market", "median"),
            topk_hit_rate=("topk_excess_vs_market", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            topk_minus_nextk_mean=("topk_minus_nextk", "mean"),
            industry_neutral_topk_excess_mean=("topk_industry_neutral_excess", "mean"),
            turnover_mean=("turnover_half_l1", "mean"),
            cost_drag_mean=("cost_drag", "mean"),
            topk_excess_after_cost_mean=("topk_excess_after_cost", "mean"),
        )
        .reset_index()
    )
    agg["topk_excess_annualized"] = (1.0 + agg["topk_excess_mean"]).pow(12) - 1.0

    if not rank_ic.empty:
        ic = (
            rank_ic.groupby(["candidate_pool_version", "model"], sort=True)
            .agg(rank_ic_mean=("rank_ic", "mean"), rank_ic_std=("rank_ic", "std"), rank_ic_months=("rank_ic", "count"))
            .reset_index()
            .rename(columns={"model": "base_model"})
        )
        ic["rank_ic_ir"] = ic["rank_ic_mean"] / ic["rank_ic_std"].replace(0.0, np.nan)
        agg = agg.merge(ic, on=["candidate_pool_version", "base_model"], how="left")

    if not quantile_spread.empty:
        qs = (
            quantile_spread.groupby(["candidate_pool_version", "model"], sort=True)
            .agg(quantile_top_minus_bottom_mean=("top_minus_bottom_return", "mean"))
            .reset_index()
            .rename(columns={"model": "base_model"})
        )
        agg = agg.merge(qs, on=["candidate_pool_version", "base_model"], how="left")

    if not regime_slice.empty:
        rs = regime_slice.pivot_table(
            index=["candidate_pool_version", "model", "top_k"],
            columns="realized_market_state",
            values="median_topk_excess",
            aggfunc="first",
        ).reset_index()
        rs.columns = [
            f"{c}_median_excess" if c in {"strong_up", "strong_down", "neutral"} else c for c in rs.columns
        ]
        agg = agg.merge(rs, on=["candidate_pool_version", "model", "top_k"], how="left")

    conc = _summarize_concentration_for_leaderboard(industry_concentration)
    if not conc.empty:
        agg = agg.merge(
            conc,
            on=["candidate_pool_version", "model", "top_k", "selection_policy", "max_industry_names"],
            how="left",
        )

    return agg.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "max_industry_share_mean", "rank_ic_mean"],
        ascending=[True, True, False, True, False],
    )


def build_lagged_state_frame(dataset: pd.DataFrame, *, min_history_months: int = 24) -> pd.DataFrame:
    base = dataset[dataset[LABEL_COL].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=["signal_date", "lagged_regime", "breadth_state"])
    monthly = (
        base.groupby("signal_date", sort=True)
        .agg(
            market_ew_return=(MARKET_COL, "first"),
            breadth_positive_ret20=("feature_ret_20d", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    monthly["signal_date"] = pd.to_datetime(monthly["signal_date"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    for i, row in monthly.iterrows():
        hist = monthly.iloc[:i].copy()
        lagged_regime = "neutral"
        breadth_state = "normal"
        if len(hist) >= int(min_history_months):
            prev = hist.iloc[-1]
            mvals = pd.to_numeric(hist["market_ew_return"], errors="coerce")
            lo = float(mvals.quantile(0.20))
            hi = float(mvals.quantile(0.80))
            prev_ret = float(prev["market_ew_return"])
            if np.isfinite(prev_ret) and np.isfinite(lo) and prev_ret <= lo:
                lagged_regime = "strong_down"
            elif np.isfinite(prev_ret) and np.isfinite(hi) and prev_ret >= hi:
                lagged_regime = "strong_up"
            bvals = pd.to_numeric(hist["breadth_positive_ret20"], errors="coerce")
            blo = float(bvals.quantile(0.20))
            bhi = float(bvals.quantile(0.80))
            cur_breadth = float(row["breadth_positive_ret20"])
            if np.isfinite(cur_breadth) and np.isfinite(blo) and cur_breadth <= blo:
                breadth_state = "narrow"
            elif np.isfinite(cur_breadth) and np.isfinite(bhi) and cur_breadth >= bhi:
                breadth_state = "wide"
        rows.append(
            {
                "signal_date": row["signal_date"],
                "market_ew_return": row["market_ew_return"],
                "breadth_positive_ret20": row["breadth_positive_ret20"],
                "lagged_regime": lagged_regime,
                "breadth_state": breadth_state,
                "state_history_months": int(len(hist)),
            }
        )
    return pd.DataFrame(rows)


def build_regime_policy_scores(scores: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    wanted = {
        STABLE_M5_ELASTICNET: "_elasticnet_score",
        STABLE_M5_EXTRATREES: "_extratrees_score",
        WATCHLIST_M6_RANK: "_rank_score",
        WATCHLIST_M6_TOP20: "_top20_score",
    }
    base_cols = [
        "signal_date",
        "candidate_pool_version",
        "symbol",
        LABEL_COL,
        EXCESS_COL,
        INDUSTRY_EXCESS_COL,
        MARKET_COL,
        "industry_level1",
        "industry_level2",
        "log_market_cap",
        "risk_flags",
        "is_buyable_tplus1_open",
    ]
    frames: list[pd.DataFrame] = []
    for model, score_col in wanted.items():
        part = scores[scores["model"] == model].copy()
        if part.empty:
            continue
        cols = [c for c in base_cols if c in part.columns]
        part = part[cols + ["score"]].rename(columns={"score": score_col})
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    for frame in frames[1:]:
        score_cols = [c for c in frame.columns if c.startswith("_") and c.endswith("_score")]
        merged = merged.merge(frame[keys + score_cols], on=keys, how="outer")
    merged = merged.merge(states[["signal_date", "lagged_regime", "breadth_state"]], on="signal_date", how="left")
    merged["lagged_regime"] = merged["lagged_regime"].fillna("neutral")
    merged["breadth_state"] = merged["breadth_state"].fillna("normal")
    for col in wanted.values():
        if col not in merged.columns:
            merged[col] = np.nan
    stable_mean = merged[["_elasticnet_score", "_extratrees_score"]].mean(axis=1)
    for col in wanted.values():
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(stable_mean).fillna(0.5)

    score = pd.Series(0.5 * merged["_elasticnet_score"] + 0.5 * merged["_extratrees_score"], index=merged.index)
    down = merged["lagged_regime"].eq("strong_down")
    up_or_wide = merged["lagged_regime"].eq("strong_up") | merged["breadth_state"].eq("wide")
    narrow = merged["breadth_state"].eq("narrow") & ~up_or_wide
    score.loc[down] = 0.80 * merged.loc[down, "_elasticnet_score"] + 0.20 * merged.loc[down, "_extratrees_score"]
    score.loc[up_or_wide] = (
        0.60 * merged.loc[up_or_wide, "_extratrees_score"]
        + 0.25 * merged.loc[up_or_wide, "_rank_score"]
        + 0.15 * merged.loc[up_or_wide, "_top20_score"]
    )
    score.loc[narrow] = 0.55 * merged.loc[narrow, "_elasticnet_score"] + 0.45 * merged.loc[narrow, "_extratrees_score"]

    out = merged[[c for c in base_cols if c in merged.columns] + ["lagged_regime", "breadth_state"]].copy()
    out["model"] = M8_POLICY_MODEL
    out["model_type"] = "lagged_regime_fixed_score_blend"
    out["score"] = score.groupby([out["signal_date"], out["candidate_pool_version"]], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first",
        ascending=False,
    )
    return out


def _concentration_gate_threshold(top_k: int) -> float:
    return 0.30 if int(top_k) <= 20 else 0.35


def build_gate_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    baseline = leaderboard[
        (leaderboard["selection_policy"] == "uncapped")
        & (leaderboard["base_model"].isin([STABLE_M5_ELASTICNET, STABLE_M5_EXTRATREES]))
    ].copy()
    baseline = (
        baseline.groupby(["candidate_pool_version", "top_k"], sort=True)
        .agg(
            m5_stable_after_cost_baseline=("topk_excess_after_cost_mean", "max"),
            m5_stable_rank_ic_baseline=("rank_ic_mean", "max"),
        )
        .reset_index()
    )
    out = leaderboard.merge(baseline, on=["candidate_pool_version", "top_k"], how="left")
    out["baseline_delta_after_cost"] = out["topk_excess_after_cost_mean"] - out["m5_stable_after_cost_baseline"]
    out["baseline_gate"] = out["baseline_delta_after_cost"] >= -0.003
    out["rank_gate"] = pd.to_numeric(out.get("rank_ic_mean"), errors="coerce") > 0
    out["spread_gate"] = pd.to_numeric(out.get("topk_minus_nextk_mean"), errors="coerce") > 0
    out["year_regime_gate"] = (
        pd.to_numeric(out.get("strong_down_median_excess"), errors="coerce").fillna(0.0) > -0.03
    ) & (pd.to_numeric(out.get("strong_up_median_excess"), errors="coerce").fillna(0.0) > -0.03)
    # M8 gate is monthly, not only average: every evaluated month should satisfy
    # the Top-K concentration threshold before this gate is considered passed.
    out["concentration_gate"] = pd.to_numeric(out.get("concentration_pass_rate"), errors="coerce").fillna(0.0) >= 0.999
    out["cost_gate_10bps"] = pd.to_numeric(out.get("topk_excess_after_cost_mean"), errors="coerce") > 0
    gate_cols = [
        "candidate_pool_version",
        "base_model",
        "model",
        "selection_policy",
        "max_industry_names",
        "top_k",
        "topk_excess_after_cost_mean",
        "m5_stable_after_cost_baseline",
        "baseline_delta_after_cost",
        "rank_ic_mean",
        "topk_minus_nextk_mean",
        "max_industry_share_mean",
        "concentration_pass_rate",
        "industry_count_mean",
        "baseline_gate",
        "rank_gate",
        "spread_gate",
        "year_regime_gate",
        "concentration_gate",
        "cost_gate_10bps",
    ]
    out["m8_gate_pass"] = out[
        ["baseline_gate", "rank_gate", "spread_gate", "year_regime_gate", "concentration_gate", "cost_gate_10bps"]
    ].all(axis=1)
    return out[[c for c in gate_cols if c in out.columns] + ["m8_gate_pass"]].sort_values(
        ["top_k", "candidate_pool_version", "m8_gate_pass", "topk_excess_after_cost_mean"],
        ascending=[True, True, False, False],
    )


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    cfg: M8RunConfig,
    dataset_path: Path,
    db_path: Path,
    output_stem: str,
    config_source: str,
    research_config_id: str,
    enabled_families: list[str],
    include_m6: bool,
) -> dict[str, Any]:
    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m8_concentration_regime",
        "research_topic": "monthly_selection_m8_concentration_regime",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "duckdb_path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": list(cfg.candidate_pools),
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in cfg.candidate_pools},
        "top_ks": list(cfg.top_ks),
        "cap_grid": {str(k): list(v) for k, v in (cfg.cap_grid or {}).items()},
        "bucket_count": int(cfg.bucket_count),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_families": ["price_volume", *enabled_families],
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + industry-neutral excess",
        "pit_policy": "walk-forward ML uses only past signal months; M8 regime policy uses lagged realized market state and signal-date breadth with expanding historical thresholds",
        "cv_policy": "walk_forward_by_signal_month",
        "hyperparameter_policy": "fixed conservative defaults inherited from M5/M6; M8 selection cap grid is predeclared in plan",
        "concentration_policy": "greedy monthly Top-K selection with max names per industry; uncapped rows retained as control",
        "regime_policy": "strong_down=80% ElasticNet/20% ExtraTrees; strong_up or wide breadth=60% ExtraTrees/25% rank_ndcg/15% top20; neutral=50/50 stable M5",
        "include_m6": bool(include_m6),
        "max_fit_rows": int(cfg.max_fit_rows),
        "model_n_jobs": int(normalize_model_n_jobs(cfg.model_n_jobs)),
        "random_seed": int(cfg.random_seed),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "base_models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }


def build_doc(
    *,
    quality: dict[str, Any],
    leaderboard: pd.DataFrame,
    industry_concentration: pd.DataFrame,
    gate: pd.DataFrame,
    year_slice: pd.DataFrame,
    regime_slice: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    leader_view = leaderboard.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "max_industry_share_mean"],
        ascending=[True, True, False, True],
    ).head(60)
    gate_view = gate.sort_values(["top_k", "candidate_pool_version", "m8_gate_pass", "topk_excess_after_cost_mean"], ascending=[True, True, False, False]).head(60)
    conc_view = (
        industry_concentration.groupby(["candidate_pool_version", "model", "top_k", "selection_policy", "max_industry_names"], sort=True)
        .agg(
            mean_max_industry_share=("max_industry_share", "mean"),
            max_max_industry_share=("max_industry_share", "max"),
            mean_industry_count=("industry_count", "mean"),
            pass_rate=("concentration_pass", "mean"),
        )
        .reset_index()
        .sort_values(["top_k", "candidate_pool_version", "mean_max_industry_share"])
        .head(50)
        if not industry_concentration.empty
        else pd.DataFrame()
    )
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M8 Concentration + Regime

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m8_concentration_regime`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 训练/评估：M5/M6 walk-forward score；M8 在选择层做行业 cap 和 lagged-regime fixed policy。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`
- 行业集中度策略：`{quality.get('concentration_policy', '')}`
- Regime 策略：`{quality.get('regime_policy', '')}`

## Leaderboard

{_format_markdown_table(leader_view, max_rows=60)}

## Gate

{_format_markdown_table(gate_view, max_rows=60)}

## Industry Concentration

{_format_markdown_table(conc_view, max_rows=50)}

## Year Slice

{_format_markdown_table(year_slice.sort_values(["top_k", "candidate_pool_version", "model", "year"]).head(60), max_rows=60)}

## Realized Market State Slice

{_format_markdown_table(regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(60), max_rows=60)}

## 本轮结论

- 本轮新增的是选择层行业只数上限，而不是继续堆模型；unconstrained 行作为对照保留。
- `M8_regime_aware_fixed_policy` 只使用 lagged realized market state 和信号日可见 breadth，权重固定，不按测试月收益回填。
- `gate.csv` 是进入后续 M9/M10 的研究证据表；本脚本不写 promoted registry，也不改生产配置。
- 若候选通过集中度但收益、Top-K vs next-K 或 regime 切片失败，结论应是继续研究，不 promotion。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    args = parse_args()
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    db_path_raw = args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb")
    db_path = _resolve_project_path(db_path_raw)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    as_of = args.as_of_date.strip() or pd.Timestamp.now().strftime("%Y-%m-%d")
    docs_dir = ROOT / "docs" / "reports" / as_of[:7]
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    cap_grid = parse_cap_grid(args.cap_grid, top_ks)
    enabled_families = _parse_str_list(args.families)
    cfg = M8RunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        min_state_history_months=int(args.min_state_history_months),
        cap_grid=cap_grid,
        model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{as_of}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_capgrid_{slugify_token(args.cap_grid)}"
        f"_maxfit_{int(args.max_fit_rows)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )
    print(f"[monthly-m8] research_config_id={research_config_id}", flush=True)

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    m5_cfg = M5RunConfig(
        top_ks=cfg.top_ks,
        candidate_pools=cfg.candidate_pools,
        bucket_count=cfg.bucket_count,
        min_train_months=cfg.min_train_months,
        min_train_rows=cfg.min_train_rows,
        max_fit_rows=cfg.max_fit_rows,
        cost_bps=cfg.cost_bps,
        random_seed=cfg.random_seed,
        include_xgboost=False,
        availability_lag_days=cfg.availability_lag_days,
        ml_models=("elasticnet", "extratrees"),
        model_n_jobs=cfg.model_n_jobs,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    m5_specs = build_feature_specs(enabled_families)
    m5_spec = m5_specs[-1:]
    feature_coverage = summarize_feature_coverage_by_spec(dataset, m5_spec)
    m5_scores, m5_importance_raw = build_all_m5_scores(dataset, m5_spec, m5_cfg)
    score_frames = [m5_scores] if not m5_scores.empty else []
    importance_frames = [summarize_m5_feature_importance(m5_importance_raw)] if not m5_importance_raw.empty else []

    include_m6 = not bool(args.skip_m6)
    if include_m6:
        m6_cfg = M6RunConfig(
            top_ks=cfg.top_ks,
            candidate_pools=cfg.candidate_pools,
            bucket_count=cfg.bucket_count,
            min_train_months=cfg.min_train_months,
            min_train_rows=cfg.min_train_rows,
            max_fit_rows=cfg.max_fit_rows,
            cost_bps=cfg.cost_bps,
            random_seed=cfg.random_seed,
            availability_lag_days=cfg.availability_lag_days,
            relevance_grades=5,
            model_n_jobs=cfg.model_n_jobs,
            ltr_models=("xgboost_rank_ndcg", "top20_calibrated"),
        )
        m6_spec = build_m6_feature_spec(enabled_families)
        m6_scores, m6_importance_raw = build_walk_forward_ltr_scores(dataset, m6_spec, m6_cfg)
        if not m6_scores.empty:
            score_frames.append(m6_scores)
        if not m6_importance_raw.empty:
            importance_frames.append(summarize_ltr_feature_importance(m6_importance_raw))

    base_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if base_scores.empty:
        warnings.warn("M8 未生成任何 score；请检查训练窗、候选池或特征覆盖。", RuntimeWarning)
    lagged_states = build_lagged_state_frame(dataset, min_history_months=cfg.min_state_history_months)
    policy_scores = build_regime_policy_scores(base_scores, lagged_states)
    if not policy_scores.empty:
        base_scores = pd.concat([base_scores, policy_scores], ignore_index=True, sort=False)

    rank_ic = build_rank_ic(base_scores)
    quantile_spread = build_quantile_spread(base_scores, bucket_count=cfg.bucket_count)
    monthly_long, topk_holdings = build_constrained_monthly(
        base_scores,
        top_ks=top_ks,
        cap_grid=cap_grid,
        cost_bps=cfg.cost_bps,
    )
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    industry_concentration = summarize_industry_concentration(topk_holdings)
    leaderboard = build_constrained_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice, industry_concentration)
    gate = build_gate_table(leaderboard)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame()
    quality = build_quality_payload(
        dataset=dataset,
        scores=base_scores,
        cfg=cfg,
        dataset_path=dataset_path,
        db_path=db_path,
        output_stem=output_stem,
        config_source=config_source,
        research_config_id=research_config_id,
        enabled_families=enabled_families,
        include_m6=include_m6,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "industry_concentration": results_dir / f"{output_stem}_industry_concentration.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "gate": results_dir / f"{output_stem}_gate.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "lagged_states": results_dir / f"{output_stem}_lagged_states.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    industry_concentration.to_csv(paths_out["industry_concentration"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    gate.to_csv(paths_out["gate"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    lagged_states.to_csv(paths_out["lagged_states"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)

    summary_payload = {
        "quality": quality,
        "top_gate_pass": gate[gate["m8_gate_pass"]].head(20).to_dict(orient="records") if not gate.empty else [],
        "top_models_by_topk": leaderboard.groupby("top_k", as_index=False).head(10).to_dict(orient="records")
        if not leaderboard.empty
        else [],
    }
    paths_out["summary_json"].write_text(
        json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = [
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        **quality,
        "artifacts": [*artifact_paths, str(paths_out["doc"].relative_to(ROOT))],
    }
    paths_out["manifest"].write_text(
        json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths_out["doc"].write_text(
        build_doc(
            quality=quality,
            leaderboard=leaderboard,
            industry_concentration=industry_concentration,
            gate=gate,
            year_slice=year_slice,
            regime_slice=regime_slice,
            artifacts=[*artifact_paths, str(paths_out["manifest"].relative_to(ROOT))],
        ),
        encoding="utf-8",
    )

    print(f"[monthly-m8] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}", flush=True)
    print(f"[monthly-m8] leaderboard={paths_out['leaderboard']}", flush=True)
    print(f"[monthly-m8] gate={paths_out['gate']}", flush=True)
    print(f"[monthly-m8] doc={paths_out['doc']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
