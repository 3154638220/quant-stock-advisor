"""M8: naturalized industry constraints — 核心算法逻辑。

从 ``scripts/run_monthly_selection_m8_natural_industry_constraints.py``
提取的评分构建、软约束选择、月度回测拼接和 gate 评估函数，
供 M8 薄层脚本调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research_identity import slugify_token
from scripts.run_monthly_selection_multisource import (
    M5RunConfig,
    build_all_m5_scores,
    build_feature_specs,
)
from src.pipeline.monthly_baselines import (
    _rank_pct_score,
    build_rank_ic,
    build_quantile_spread,
    summarize_regime_slice,
    summarize_year_slice,
)
from src.pipeline.monthly_concentration import summarize_industry_concentration
from src.research.gates import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
)

# ── 常量 ──────────────────────────────────────────────────────────────────────

BASE_M5_ELASTICNET = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess"
BASE_M5_EXTRATREES = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess"
SOFT_OPTIMIZER_CANDIDATE_FLOOR = 200
SOFT_OPTIMIZER_CANDIDATE_MULTIPLIER = 6


# ── 配置 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelVariant:
    name: str
    description: str
    market_weight: float
    industry_neutral_weight: float


LABEL_VARIANTS: tuple[LabelVariant, ...] = (
    LabelVariant("market_excess", "train on market-relative forward return", 1.0, 0.0),
    LabelVariant("industry_neutral_excess", "train on same-industry neutral forward return", 0.0, 1.0),
    LabelVariant("blended_excess_50_50", "fixed 50/50 market and industry-neutral blend", 0.5, 0.5),
)


@dataclass(frozen=True)
class M8NaturalRunConfig:
    top_ks: tuple[int, ...] = (20, 30)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    model_n_jobs: int = 0
    soft_gammas: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.80)
    hardcap_tolerance: float = 0.005


# ── 标签变体 ──────────────────────────────────────────────────────────────────


def _target_for_variant(dataset: pd.DataFrame, variant: LabelVariant) -> pd.Series:
    market = pd.to_numeric(dataset.get(EXCESS_COL), errors="coerce")
    industry = pd.to_numeric(dataset.get(INDUSTRY_EXCESS_COL), errors="coerce")
    target = variant.market_weight * market + variant.industry_neutral_weight * industry
    if target.notna().sum() == 0:
        target = market
    return target


def _restore_eval_columns(scores: pd.DataFrame, original_dataset: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    eval_cols = [
        LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL,
        "industry_level1", "industry_level2", "log_market_cap", "risk_flags",
        "is_buyable_tplus1_open",
    ]
    original = original_dataset[[c for c in keys + eval_cols if c in original_dataset.columns]].drop_duplicates(keys)
    out = scores.drop(columns=[c for c in eval_cols if c in scores.columns], errors="ignore")
    out = out.merge(original, on=keys, how="left")
    return out


def build_label_variant_scores(
    dataset: pd.DataFrame,
    *,
    cfg: M8NaturalRunConfig,
    enabled_families: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_feature_specs(enabled_families)[-1:]
    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for variant in LABEL_VARIANTS:
        train_dataset = dataset.copy()
        train_dataset[EXCESS_COL] = _target_for_variant(train_dataset, variant)
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
        scores, importance = build_all_m5_scores(train_dataset, specs, m5_cfg)
        if not scores.empty:
            scores = _restore_eval_columns(scores, dataset)
            scores["base_model"] = scores["model"]
            scores["label_variant"] = variant.name
            scores["label_policy"] = variant.description
            scores["model"] = scores["model"].astype(str) + "__label_" + variant.name
            scores["model_type"] = scores["model_type"].astype(str) + "_label_variant"
            scores["score_family"] = "label_compare"
            score_frames.append(scores)
        if not importance.empty:
            importance = importance.copy()
            importance["label_variant"] = variant.name
            importance_frames.append(importance)
    score_out = pd.concat(score_frames, ignore_index=True, sort=False) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True, sort=False) if importance_frames else pd.DataFrame()
    return score_out, imp_out


# ── 评分分解 ──────────────────────────────────────────────────────────────────


def _rank_within_group(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=values.index)
    return x.rank(method="first", pct=True, ascending=True)


def build_score_decomposition_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_cols = ["signal_date", "candidate_pool_version", "model"]
    for _, part in scores.groupby(group_cols, sort=True):
        part = part.copy()
        part["industry_level1"] = part["industry_level1"].fillna("_UNKNOWN_").astype(str)
        part["_score_num"] = pd.to_numeric(part["score"], errors="coerce")
        part["_score_global_rank"] = _rank_pct_score(part["_score_num"])
        part["_score_within_industry"] = part.groupby("industry_level1")["_score_num"].transform(_rank_within_group)
        industry_mean = part.groupby("industry_level1")["_score_num"].transform("mean")
        part["_score_industry_allocation"] = _rank_pct_score(industry_mean)
        residual = part["_score_num"] - industry_mean + part["_score_num"].mean()
        part["_score_sector_residual"] = _rank_pct_score(residual)
        variants = {
            "within_industry_alpha": part["_score_within_industry"],
            "industry_allocation": part["_score_industry_allocation"],
            "within70_industry30": 0.70 * part["_score_within_industry"] + 0.30 * part["_score_industry_allocation"],
            "sector_residual": part["_score_sector_residual"],
        }
        for name, variant_score in variants.items():
            out = part.drop(columns=[c for c in part.columns if c.startswith("_score_")], errors="ignore").copy()
            out["source_model"] = out["model"]
            out["model"] = out["model"].astype(str) + "__" + name
            out["model_type"] = "natural_score_decomposition"
            out["score_family"] = "score_decomposition"
            out["decomposition_component"] = name
            out["score"] = _rank_pct_score(variant_score)
            out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
                method="first", ascending=False,
            )
            frames.append(out)
        summary_rows.append({
            "signal_date": part["signal_date"].iloc[0],
            "candidate_pool_version": part["candidate_pool_version"].iloc[0],
            "source_model": part["model"].iloc[0],
            "industry_count": int(part["industry_level1"].nunique()),
            "score_industry_mean_std": float(part.groupby("industry_level1")["_score_num"].mean().std()),
            "score_industry_mean_range": float(
                part.groupby("industry_level1")["_score_num"].mean().max()
                - part.groupby("industry_level1")["_score_num"].mean().min()
            ),
        })
    out_scores = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out_scores, pd.DataFrame(summary_rows)


# ── 软惩罚评分 ────────────────────────────────────────────────────────────────


def build_soft_penalty_scores(scores: pd.DataFrame, *, gammas: list[float]) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for _, part in scores.groupby(["signal_date", "candidate_pool_version", "model"], sort=True):
        part = part.copy()
        part["industry_level1"] = part["industry_level1"].fillna("_UNKNOWN_").astype(str)
        score = pd.to_numeric(part["score"], errors="coerce")
        industry_mean = part.groupby("industry_level1")["score"].transform("mean")
        industry_rank = _rank_pct_score(industry_mean)
        universe_share = part.groupby("industry_level1")["symbol"].transform("count") / max(len(part), 1)
        active_crowding_proxy = industry_rank - _rank_pct_score(universe_share)
        for gamma in gammas:
            adjusted = score - float(gamma) * active_crowding_proxy
            out = part.copy()
            token = slugify_token(f"{gamma:.4g}")
            out["source_model"] = out["model"]
            out["model"] = out["model"].astype(str) + "__soft_sector_penalty_gamma" + token
            out["model_type"] = "soft_concentration_penalty_score"
            out["score_family"] = "penalty_frontier"
            out["soft_penalty"] = "relative_to_universe_industry_score"
            out["soft_gamma"] = float(gamma)
            out["score"] = _rank_pct_score(adjusted)
            out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
                method="first", ascending=False,
            )
            frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


# ── 软约束行业风险预算 ────────────────────────────────────────────────────────


def _weighted_turnover(prev: set[str] | None, cur: set[str]) -> float:
    if prev is None:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


def select_soft_industry_risk(part: pd.DataFrame, *, k: int, gamma: float) -> pd.DataFrame:
    """软约束行业风险预算选股。

    每次迭代选择 adjusted_score = score - gamma * projected_industry_share 最高的标的，
    动态调整行业集中度，避免单一行业占比过高。
    """
    if part.empty or int(k) <= 0:
        return part.iloc[0:0].copy()
    ordered = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
    candidate_limit = max(SOFT_OPTIMIZER_CANDIDATE_FLOOR, SOFT_OPTIMIZER_CANDIDATE_MULTIPLIER * int(k))
    ordered = ordered.head(min(len(ordered), candidate_limit)).copy()
    ordered["industry_level1"] = ordered["industry_level1"].fillna("_UNKNOWN_").astype(str)
    selected_idx: list[Any] = []
    counts: dict[str, int] = {}
    remaining = ordered.copy()
    while len(selected_idx) < int(k) and not remaining.empty:
        temp = remaining.copy()
        projected_count = temp["industry_level1"].map(lambda x: counts.get(str(x), 0) + 1).astype(float)
        projected_share = projected_count / float(max(int(k), 1))
        risk_penalty = float(gamma) * projected_share
        temp["_adjusted_score"] = pd.to_numeric(temp["score"], errors="coerce") - risk_penalty
        temp = temp.sort_values(["_adjusted_score", "score", "symbol"], ascending=[False, False, True], kind="mergesort")
        picked_idx = temp.index[0]
        selected_idx.append(picked_idx)
        industry = str(ordered.loc[picked_idx, "industry_level1"])
        counts[industry] = counts.get(industry, 0) + 1
        remaining = remaining.drop(index=picked_idx)
    out = ordered.loc[selected_idx].copy()
    if not out.empty:
        projected_count = out["industry_level1"].map(lambda x: counts.get(str(x), 0)).astype(float)
        out["adjusted_score"] = pd.to_numeric(out["score"], errors="coerce") - float(gamma) * projected_count / float(max(int(k), 1))
    return out


# ── 月度回测拼接 ──────────────────────────────────────────────────────────────


def build_monthly_from_scores(
    scores: pd.DataFrame,
    *,
    top_ks: list[int],
    cost_bps: float,
    selection_policy: str,
    soft_gamma: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从评分构建月度回测表（每月选 top-k，计算换手和超额）。"""
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    prev_by_key: dict[tuple[str, str, int, str], set[str]] = {}
    ordered = scores.sort_values(
        ["candidate_pool_version", "model", "signal_date", "score", "symbol"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    for (pool, model, signal_date), part in ordered.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        part = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
        candidate_width = int(part["symbol"].nunique())
        market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
        pool_ret = float(pd.to_numeric(part[LABEL_COL], errors="coerce").mean())
        model_type = str(part["model_type"].dropna().iloc[0]) if part["model_type"].notna().any() else ""
        source_model = (
            str(part["source_model"].dropna().iloc[0])
            if "source_model" in part.columns and part["source_model"].notna().any()
            else str(model)
        )
        label_variant = (
            str(part["label_variant"].dropna().iloc[0])
            if "label_variant" in part and part["label_variant"].notna().any()
            else ""
        )
        score_family = (
            str(part["score_family"].dropna().iloc[0])
            if "score_family" in part and part["score_family"].notna().any()
            else ""
        )
        for k in top_ks:
            if selection_policy == "soft_industry_risk_budget":
                selected = select_soft_industry_risk(part, k=int(k), gamma=float(soft_gamma or 0.0))
                remaining = part.drop(index=selected.index, errors="ignore")
                nextk = select_soft_industry_risk(remaining, k=int(k), gamma=float(soft_gamma or 0.0))
            else:
                selected = part.head(int(k)).copy()
                nextk = part.iloc[int(k):2 * int(k)].copy()
            if selected.empty:
                continue
            variant_model = model
            cur_symbols = set(selected["symbol"].astype(str))
            turnover = _weighted_turnover(
                prev_by_key.get((pool, variant_model, int(k), selection_policy)), cur_symbols
            )
            prev_by_key[(pool, variant_model, int(k), selection_policy)] = cur_symbols
            top_ret = float(pd.to_numeric(selected[LABEL_COL], errors="coerce").mean())
            next_ret = float(pd.to_numeric(nextk[LABEL_COL], errors="coerce").mean()) if not nextk.empty else np.nan
            top_ind_excess = (
                float(pd.to_numeric(selected[INDUSTRY_EXCESS_COL], errors="coerce").mean())
                if INDUSTRY_EXCESS_COL in selected.columns
                else np.nan
            )
            h = selected.copy()
            h["base_model"] = source_model
            h["model"] = variant_model
            h["model_type"] = model_type
            h["top_k"] = int(k)
            h["selected_rank"] = np.arange(1, len(h) + 1)
            h["selection_policy"] = selection_policy
            h["max_industry_names"] = 0
            h["soft_gamma"] = np.nan if soft_gamma is None else float(soft_gamma)
            keep_cols = [
                "signal_date", "candidate_pool_version", "base_model", "model", "model_type",
                "selection_policy", "max_industry_names", "soft_gamma", "score_family",
                "label_variant", "decomposition_component", "top_k", "selected_rank",
                "symbol", "score", "adjusted_score", LABEL_COL, "industry_level1", "risk_flags",
            ]
            holdings.append(h[[c for c in keep_cols if c in h.columns]].copy())
            rows.append({
                "signal_date": signal_date,
                "candidate_pool_version": pool,
                "model": variant_model,
                "source_model": source_model,
                "model_type": model_type,
                "selection_policy": selection_policy,
                "score_family": score_family,
                "label_variant": label_variant,
                "soft_gamma": np.nan if soft_gamma is None else float(soft_gamma),
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
            })
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()


# ── Optimizer 变体 ────────────────────────────────────────────────────────────


def build_soft_optimizer_scores(
    scores: pd.DataFrame, *, gammas: list[float]
) -> list[tuple[float, pd.DataFrame]]:
    out: list[tuple[float, pd.DataFrame]] = []
    for gamma in gammas:
        part = scores.copy()
        token = slugify_token(f"{gamma:.4g}")
        part["source_model"] = part["model"]
        part["model"] = part["model"].astype(str) + "__soft_risk_budget_gamma" + token
        part["model_type"] = "soft_industry_risk_budget_optimizer"
        part["score_family"] = "optimizer_compare"
        part["soft_gamma"] = float(gamma)
        out.append((float(gamma), part))
    return out


def copy_source_metric_for_optimizer(metric: pd.DataFrame, optimizer_monthly: pd.DataFrame) -> pd.DataFrame:
    if metric.empty or optimizer_monthly.empty or "source_model" not in optimizer_monthly.columns:
        return pd.DataFrame()
    maps = optimizer_monthly[["candidate_pool_version", "model", "source_model"]].drop_duplicates()
    mapped = metric.merge(maps, left_on=["candidate_pool_version", "model"], right_on=["candidate_pool_version", "source_model"], how="inner")
    mapped = mapped.drop(columns=["model_x", "source_model"], errors="ignore").rename(columns={"model_y": "model"})
    return mapped


# ── Leaderboard 与 Gate ───────────────────────────────────────────────────────


def _concentration_summary(industry_concentration: pd.DataFrame) -> pd.DataFrame:
    if industry_concentration.empty:
        return pd.DataFrame()
    return (
        industry_concentration.groupby(
            ["candidate_pool_version", "model", "top_k", "selection_policy"], sort=True
        )
        .agg(
            max_industry_share_mean=("max_industry_share", "mean"),
            max_industry_share_max=("max_industry_share", "max"),
            industry_count_mean=("industry_count", "mean"),
            concentration_pass_rate=("concentration_pass", "mean"),
        )
        .reset_index()
    )


def build_leaderboard(
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
            ["candidate_pool_version", "model", "model_type", "selection_policy",
             "score_family", "label_variant", "soft_gamma", "top_k"],
            dropna=False, sort=True,
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
        )
        ic["rank_ic_ir"] = ic["rank_ic_mean"] / ic["rank_ic_std"].replace(0.0, np.nan)
        agg = agg.merge(ic, on=["candidate_pool_version", "model"], how="left")
    if not quantile_spread.empty:
        qs = (
            quantile_spread.groupby(["candidate_pool_version", "model"], sort=True)
            .agg(quantile_top_minus_bottom_mean=("top_minus_bottom_return", "mean"))
            .reset_index()
        )
        agg = agg.merge(qs, on=["candidate_pool_version", "model"], how="left")
    if not regime_slice.empty:
        rs = regime_slice.pivot_table(
            index=["candidate_pool_version", "model", "top_k"],
            columns="realized_market_state", values="median_topk_excess", aggfunc="first",
        ).reset_index()
        rs.columns = [f"{c}_median_excess" if c in {"strong_up", "strong_down", "neutral"} else c for c in rs.columns]
        agg = agg.merge(rs, on=["candidate_pool_version", "model", "top_k"], how="left")
    conc = _concentration_summary(industry_concentration)
    if not conc.empty:
        agg = agg.merge(conc, on=["candidate_pool_version", "model", "top_k", "selection_policy"], how="left")
    return agg.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "max_industry_share_mean", "rank_ic_mean"],
        ascending=[True, True, False, True, False],
    )


def load_hardcap_baseline(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    hard = df[df.get("selection_policy", "").eq("industry_names_cap")].copy()
    if hard.empty:
        return pd.DataFrame()
    out = (
        hard.groupby(["candidate_pool_version", "top_k"], sort=True)
        .agg(
            hardcap_after_cost_baseline=("topk_excess_after_cost_mean", "max"),
            hardcap_rank_ic_baseline=("rank_ic_mean", "max"),
            hardcap_source_model=("model", "first"),
        )
        .reset_index()
    )
    return out


def build_gate_table(
    leaderboard: pd.DataFrame, hardcap_baseline: pd.DataFrame, *, tolerance: float
) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    out = leaderboard.copy()
    if not hardcap_baseline.empty:
        out = out.merge(hardcap_baseline, on=["candidate_pool_version", "top_k"], how="left")
    else:
        out["hardcap_after_cost_baseline"] = np.nan
        out["hardcap_rank_ic_baseline"] = np.nan
        out["hardcap_source_model"] = ""
    out["hardcap_delta_after_cost"] = (
        pd.to_numeric(out["topk_excess_after_cost_mean"], errors="coerce")
        - pd.to_numeric(out["hardcap_after_cost_baseline"], errors="coerce")
    )
    out["natural_policy_gate"] = out["selection_policy"].ne("industry_names_cap")
    out["hardcap_closeness_gate"] = out["hardcap_delta_after_cost"].fillna(0.0) >= -float(tolerance)
    rank_ic = pd.to_numeric(
        out["rank_ic_mean"] if "rank_ic_mean" in out else pd.Series(np.nan, index=out.index), errors="coerce"
    )
    spread = pd.to_numeric(
        out["topk_minus_nextk_mean"] if "topk_minus_nextk_mean" in out else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    concentration_pass = pd.to_numeric(
        out["concentration_pass_rate"] if "concentration_pass_rate" in out else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    strong_down = pd.to_numeric(
        out["strong_down_median_excess"] if "strong_down_median_excess" in out else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    strong_up = pd.to_numeric(
        out["strong_up_median_excess"] if "strong_up_median_excess" in out else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    out["rank_gate"] = rank_ic > 0
    out["spread_gate"] = spread > 0
    out["concentration_gate"] = concentration_pass.fillna(0.0) >= 0.999
    out["year_regime_gate"] = (strong_down.fillna(0.0) > -0.03) & (strong_up.fillna(0.0) > -0.03)
    out["cost_gate_10bps"] = pd.to_numeric(out["topk_excess_after_cost_mean"], errors="coerce") > 0
    out["fixed_parameter_gate"] = True
    out["m8_natural_gate_pass"] = (
        out["natural_policy_gate"]
        & out["hardcap_closeness_gate"]
        & out["rank_gate"]
        & out["spread_gate"]
        & out["concentration_gate"]
        & out["year_regime_gate"]
        & out["cost_gate_10bps"]
        & out["fixed_parameter_gate"]
    )
    return out


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    cfg: M8NaturalRunConfig,
    dataset_path: "Path",
    db_path: "Path",
    output_stem: str,
    config_source: str,
    research_config_id: str,
    enabled_families: list[str],
    hardcap_path: "Path | None",
) -> dict[str, Any]:
    from src.pipeline.monthly_baselines import normalize_model_n_jobs, valid_pool_frame
    from src.research.gates import POOL_RULES

    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m8_natural_industry_constraints",
        "research_topic": "monthly_selection_m8_natural_industry_constraints",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(dataset_path.parents[2]))
        if hasattr(dataset_path, "relative_to") else str(dataset_path),
        "duckdb_path": str(db_path),
        "hardcap_leaderboard_path": str(hardcap_path or ""),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": list(cfg.candidate_pools),
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in cfg.candidate_pools},
        "top_ks": list(cfg.top_ks),
        "bucket_count": int(cfg.bucket_count),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_families": ["price_volume", *enabled_families],
        "label_variants": [v.name for v in LABEL_VARIANTS],
        "soft_gammas": list(cfg.soft_gammas),
        "hardcap_tolerance": float(cfg.hardcap_tolerance),
        "pit_policy": "walk-forward ML uses only past signal months",
        "cv_policy": "walk_forward_by_signal_month",
        "naturalization_policy": "label target variants + within-industry score decomposition + soft sector penalty + continuous industry risk-budget selection",
        "promotion_policy": "research only; this script never writes promoted registry or production config",
        "max_fit_rows": int(cfg.max_fit_rows),
        "model_n_jobs": int(normalize_model_n_jobs(cfg.model_n_jobs)),
        "random_seed": int(cfg.random_seed),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
    }


def build_m8_natural_doc(
    *,
    quality: dict[str, Any],
    leaderboard: pd.DataFrame,
    label_compare: pd.DataFrame,
    penalty_frontier: pd.DataFrame,
    score_decomposition: pd.DataFrame,
    optimizer_compare: pd.DataFrame,
    gate: pd.DataFrame,
    year_slice: pd.DataFrame,
    regime_slice: pd.DataFrame,
    artifacts: list[str],
) -> str:
    from src.reporting.markdown_report import format_markdown_table as _fmt

    def _view(df: pd.DataFrame, *, rows: int = 60) -> str:
        return _fmt(df.head(rows), max_rows=rows) if not df.empty else "_无数据_"

    generated_at = pd.Timestamp.utcnow().isoformat()
    pass_count = int(gate["m8_natural_gate_pass"].sum()) if not gate.empty and "m8_natural_gate_pass" in gate else 0
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M8 Natural Industry Constraints

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m8_natural_industry_constraints`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- hard-cap 对照：`{quality.get('hardcap_leaderboard_path', '')}`
- 训练/评估：M5 ElasticNet/ExtraTrees walk-forward；M8 自然化只使用固定先验或预声明参数网格。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`
- 自然化策略：`{quality.get('naturalization_policy', '')}`
- M8 natural gate 通过候选数：`{pass_count}`

## Gate

{_view(gate.sort_values(["top_k", "candidate_pool_version", "m8_natural_gate_pass", "topk_excess_after_cost_mean"], ascending=[True, True, False, False]), rows=80)}

## Leaderboard

{_view(leaderboard.sort_values(["top_k", "candidate_pool_version", "topk_excess_after_cost_mean"], ascending=[True, True, False]), rows=80)}

## Label Compare

{_view(label_compare.sort_values(["top_k", "candidate_pool_version", "topk_excess_after_cost_mean"], ascending=[True, True, False]), rows=60)}

## Penalty Frontier

{_view(penalty_frontier.sort_values(["top_k", "candidate_pool_version", "soft_gamma", "topk_excess_after_cost_mean"], ascending=[True, True, True, False]), rows=60)}

## Score Decomposition

{_view(score_decomposition.sort_values(["candidate_pool_version", "source_model", "signal_date"]).head(80), rows=80)}

## Optimizer Compare

{_view(optimizer_compare.sort_values(["top_k", "candidate_pool_version", "soft_gamma", "topk_excess_after_cost_mean"], ascending=[True, True, True, False]), rows=60)}

## Year Slice

{_view(year_slice.sort_values(["top_k", "candidate_pool_version", "model", "year"]), rows=60)}

## Realized Market State Slice

{_view(regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]), rows=60)}

## 本轮结论

- 这轮把 hard cap 保留为 stress baseline，但主候选来自标签、分数、软惩罚或连续风险预算，不再用"每行业最多 N 只"作为通过条件。
- `industry_neutral_excess` 与 `blended_excess_50_50` 是固定定义；soft gamma 是预声明网格，不按测试月收益回填。
- `m8_natural_gate_pass=True` 才表示 M8 自然化满足计划 gate；否则结论是自然化证据不足，不 promotion。
- 本脚本不写 `configs/promoted/promoted_registry.json`，也不改生产配置。

## 本轮产物

{artifact_lines}
"""
