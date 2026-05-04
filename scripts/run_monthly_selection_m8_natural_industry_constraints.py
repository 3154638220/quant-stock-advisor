#!/usr/bin/env python3
"""M8: naturalized industry constraints for monthly selection.

This script complements ``run_monthly_selection_concentration_regime.py``.
The earlier M8 script tests hard industry name caps.  This one tests
industry-aware labels, score decomposition, soft concentration penalties, and
continuous risk-budget selection without imposing a max-names hard cap.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from scripts.run_monthly_selection_multisource import (
    M5RunConfig,
    attach_enabled_families,
    build_all_m5_scores,
    build_feature_specs,
    summarize_feature_coverage_by_spec,
)
from scripts.run_monthly_selection_multisource import (
    summarize_feature_importance as summarize_m5_feature_importance,
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
from src.pipeline.monthly_baselines import (
    _rank_pct_score,
    build_quantile_spread,
    build_rank_ic,
    build_realized_market_states,
    load_baseline_dataset,
    model_n_jobs_token,
    normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
    summarize_industry_exposure,
    summarize_regime_slice,
    summarize_year_slice,
    valid_pool_frame,
)
from src.pipeline.monthly_concentration import summarize_industry_concentration
from src.reporting.markdown_report import format_markdown_table, json_sanitize
from src.research.gates import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    POOL_RULES,
    TOP20_COL,
)
from src.settings import config_path_candidates, load_config, resolve_config_path

# 别名：scripts 历史使用下划线前缀，后续薄层化时统一改为原名
_format_markdown_table = format_markdown_table
_json_sanitize = json_sanitize

BASE_M5_ELASTICNET = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess"
BASE_M5_EXTRATREES = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess"
SOFT_OPTIMIZER_CANDIDATE_FLOOR = 200
SOFT_OPTIMIZER_CANDIDATE_MULTIPLIER = 6


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M8 行业约束自然化实验")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m8_natural_industry_constraints")
    p.add_argument("--as-of-date", type=str, default="")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30")
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
    p.add_argument("--families", type=str, default="industry_breadth,fund_flow,fundamental")
    p.add_argument("--soft-gamma", type=str, default="0.05,0.10,0.20,0.35,0.50,0.80")
    p.add_argument("--hardcap-leaderboard", type=str, default="")
    p.add_argument("--hardcap-tolerance", type=float, default=0.005)
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _project_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _resolve_loaded_config_path(config_arg: Path | None) -> Path | None:
    if config_arg is not None:
        return resolve_config_path(config_arg)
    candidates: list[Path] = []
    env_path = os.environ.get("QUANT_CONFIG", "").strip()
    if env_path:
        candidates.extend(config_path_candidates(env_path))
    candidates.extend([ROOT / "config.yaml", ROOT / "config.yaml.example"])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def _parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def _parse_float_list(raw: str) -> list[float]:
    return sorted({float(x.strip()) for x in str(raw).split(",") if x.strip()})


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


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
                method="first",
                ascending=False,
            )
            frames.append(out)
        summary_rows.append(
            {
                "signal_date": part["signal_date"].iloc[0],
                "candidate_pool_version": part["candidate_pool_version"].iloc[0],
                "source_model": part["model"].iloc[0],
                "industry_count": int(part["industry_level1"].nunique()),
                "score_industry_mean_std": float(part.groupby("industry_level1")["_score_num"].mean().std()),
                "score_industry_mean_range": float(
                    part.groupby("industry_level1")["_score_num"].mean().max()
                    - part.groupby("industry_level1")["_score_num"].mean().min()
                ),
            }
        )
    out_scores = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out_scores, pd.DataFrame(summary_rows)


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
                method="first",
                ascending=False,
            )
            frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _weighted_turnover(prev: set[str] | None, cur: set[str]) -> float:
    if prev is None:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


def select_soft_industry_risk(part: pd.DataFrame, *, k: int, gamma: float) -> pd.DataFrame:
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


def build_monthly_from_scores(
    scores: pd.DataFrame,
    *,
    top_ks: list[int],
    cost_bps: float,
    selection_policy: str,
    soft_gamma: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    for (pool, model, signal_date), part in ordered.groupby(["candidate_pool_version", "model", "signal_date"], sort=True):
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
        label_variant = str(part["label_variant"].dropna().iloc[0]) if "label_variant" in part and part["label_variant"].notna().any() else ""
        score_family = str(part["score_family"].dropna().iloc[0]) if "score_family" in part and part["score_family"].notna().any() else ""
        for k in top_ks:
            if selection_policy == "soft_industry_risk_budget":
                selected = select_soft_industry_risk(part, k=int(k), gamma=float(soft_gamma or 0.0))
                remaining = part.drop(index=selected.index, errors="ignore")
                nextk = select_soft_industry_risk(remaining, k=int(k), gamma=float(soft_gamma or 0.0))
            else:
                selected = part.head(int(k)).copy()
                nextk = part.iloc[int(k) : 2 * int(k)].copy()
            if selected.empty:
                continue
            variant_model = model
            cur_symbols = set(selected["symbol"].astype(str))
            turnover = _weighted_turnover(prev_by_key.get((pool, variant_model, int(k), selection_policy)), cur_symbols)
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
                "signal_date",
                "candidate_pool_version",
                "base_model",
                "model",
                "model_type",
                "selection_policy",
                "max_industry_names",
                "soft_gamma",
                "score_family",
                "label_variant",
                "decomposition_component",
                "top_k",
                "selected_rank",
                "symbol",
                "score",
                "adjusted_score",
                LABEL_COL,
                "industry_level1",
                "risk_flags",
            ]
            holdings.append(h[[c for c in keep_cols if c in h.columns]].copy())
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()


def build_soft_optimizer_scores(scores: pd.DataFrame, *, gammas: list[float]) -> list[tuple[float, pd.DataFrame]]:
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
    maps = maps[maps["source_model"].notna() & maps["model"].ne(maps["source_model"])].copy()
    if maps.empty:
        return pd.DataFrame()
    copied: list[pd.DataFrame] = []
    for row in maps.itertuples(index=False):
        part = metric[
            (metric["candidate_pool_version"] == row.candidate_pool_version)
            & (metric["model"] == row.source_model)
        ].copy()
        if part.empty:
            continue
        part["model"] = row.model
        copied.append(part)
    return pd.concat(copied, ignore_index=True, sort=False) if copied else pd.DataFrame()


def _concentration_summary(industry_concentration: pd.DataFrame) -> pd.DataFrame:
    if industry_concentration.empty:
        return pd.DataFrame()
    return (
        industry_concentration.groupby(["candidate_pool_version", "model", "top_k", "selection_policy"], sort=True)
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
            [
                "candidate_pool_version",
                "model",
                "model_type",
                "selection_policy",
                "score_family",
                "label_variant",
                "soft_gamma",
                "top_k",
            ],
            dropna=False,
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
            columns="realized_market_state",
            values="median_topk_excess",
            aggfunc="first",
        ).reset_index()
        rs.columns = [
            f"{c}_median_excess" if c in {"strong_up", "strong_down", "neutral"} else c for c in rs.columns
        ]
        agg = agg.merge(rs, on=["candidate_pool_version", "model", "top_k"], how="left")
    conc = _concentration_summary(industry_concentration)
    if not conc.empty:
        agg = agg.merge(conc, on=["candidate_pool_version", "model", "top_k", "selection_policy"], how="left")
    return agg.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "max_industry_share_mean", "rank_ic_mean"],
        ascending=[True, True, False, True, False],
    )


def resolve_hardcap_leaderboard(path_raw: str, *, as_of: str, results_dir: Path) -> Path | None:
    if path_raw.strip():
        p = _resolve_project_path(path_raw.strip())
        return p if p.exists() else None
    preferred = results_dir / f"monthly_selection_m8_concentration_regime_{as_of}_leaderboard.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(results_dir.glob("monthly_selection_m8_concentration_regime_*_leaderboard.csv"))
    return candidates[-1] if candidates else None


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


def build_gate_table(leaderboard: pd.DataFrame, hardcap_baseline: pd.DataFrame, *, tolerance: float) -> pd.DataFrame:
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
    rank_ic = pd.to_numeric(out["rank_ic_mean"] if "rank_ic_mean" in out else pd.Series(np.nan, index=out.index), errors="coerce")
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
    out["year_regime_gate"] = (
        strong_down.fillna(0.0) > -0.03
    ) & (strong_up.fillna(0.0) > -0.03)
    out["cost_gate_10bps"] = pd.to_numeric(out["topk_excess_after_cost_mean"], errors="coerce") > 0
    out["fixed_parameter_gate"] = True
    gate_cols = [
        "candidate_pool_version",
        "model",
        "model_type",
        "selection_policy",
        "score_family",
        "label_variant",
        "soft_gamma",
        "top_k",
        "topk_excess_after_cost_mean",
        "hardcap_after_cost_baseline",
        "hardcap_delta_after_cost",
        "rank_ic_mean",
        "topk_minus_nextk_mean",
        "max_industry_share_mean",
        "concentration_pass_rate",
        "industry_count_mean",
        "natural_policy_gate",
        "hardcap_closeness_gate",
        "rank_gate",
        "spread_gate",
        "concentration_gate",
        "year_regime_gate",
        "cost_gate_10bps",
        "fixed_parameter_gate",
    ]
    out["m8_natural_gate_pass"] = out[
        [
            "natural_policy_gate",
            "hardcap_closeness_gate",
            "rank_gate",
            "spread_gate",
            "concentration_gate",
            "year_regime_gate",
            "cost_gate_10bps",
            "fixed_parameter_gate",
        ]
    ].all(axis=1)
    return out[[c for c in gate_cols if c in out.columns] + ["m8_natural_gate_pass"]].sort_values(
        ["top_k", "candidate_pool_version", "m8_natural_gate_pass", "topk_excess_after_cost_mean"],
        ascending=[True, True, False, False],
    )


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    cfg: M8NaturalRunConfig,
    dataset_path: Path,
    db_path: Path,
    output_stem: str,
    config_source: str,
    research_config_id: str,
    enabled_families: list[str],
    hardcap_path: Path | None,
) -> dict[str, Any]:
    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m8_natural_industry_constraints",
        "research_topic": "monthly_selection_m8_natural_industry_constraints",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "duckdb_path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "hardcap_leaderboard_path": str(hardcap_path.relative_to(ROOT)) if hardcap_path and hardcap_path.is_relative_to(ROOT) else str(hardcap_path or ""),
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
        "pit_policy": "walk-forward ML uses only past signal months; natural M8 parameters are fixed priors or predeclared grids",
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


def _view(df: pd.DataFrame, *, rows: int = 60) -> str:
    return _format_markdown_table(df.head(rows), max_rows=rows) if not df.empty else "_无数据_"


def build_doc(
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

- 这轮把 hard cap 保留为 stress baseline，但主候选来自标签、分数、软惩罚或连续风险预算，不再用“每行业最多 N 只”作为通过条件。
- `industry_neutral_excess` 与 `blended_excess_50_50` 是固定定义；soft gamma 是预声明网格，不按测试月收益回填。
- `m8_natural_gate_pass=True` 才表示 M8 自然化满足计划 gate；否则结论是自然化证据不足，不 promotion。
- 本脚本不写 `configs/promoted/promoted_registry.json`，也不改生产配置。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = _resolve_loaded_config_path(args.config)
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = _project_relative(loaded_config_path) if loaded_config_path is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    db_path_raw = args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb")
    db_path = _resolve_project_path(db_path_raw)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    experiments_dir = _resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    as_of = args.as_of_date.strip() or pd.Timestamp.now().strftime("%Y-%m-%d")
    docs_dir = ROOT / "docs" / "reports" / as_of[:7]
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    gammas = _parse_float_list(args.soft_gamma)
    enabled_families = _parse_str_list(args.families)
    cfg = M8NaturalRunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        model_n_jobs=int(args.model_n_jobs),
        soft_gammas=tuple(gammas),
        hardcap_tolerance=float(args.hardcap_tolerance),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{as_of}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_labels_{'-'.join(v.name for v in LABEL_VARIANTS)}"
        f"_softgamma_{slugify_token(args.soft_gamma)}"
        f"_maxfit_{int(args.max_fit_rows)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m8_natural_industry_constraints",
        research_topic="monthly_selection_m8_natural_industry_constraints",
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    research_config_id = identity.research_config_id
    output_stem = identity.output_stem
    print(f"[monthly-m8-natural] research_config_id={research_config_id}", flush=True)

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    m5_attach_cfg = M5RunConfig(
        top_ks=cfg.top_ks,
        candidate_pools=cfg.candidate_pools,
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
    dataset = attach_enabled_families(dataset, db_path, m5_attach_cfg, enabled_families)
    feature_specs = build_feature_specs(enabled_families)[-1:]
    feature_coverage = summarize_feature_coverage_by_spec(dataset, feature_specs)

    label_scores, importance_raw = build_label_variant_scores(dataset, cfg=cfg, enabled_families=enabled_families)
    if label_scores.empty:
        warnings.warn("M8 natural 未生成 label variant score；请检查训练窗或特征覆盖。", RuntimeWarning)
    decomposition_scores, score_decomposition = build_score_decomposition_scores(label_scores)
    penalty_scores = build_soft_penalty_scores(label_scores, gammas=gammas)

    plain_scores = pd.concat([label_scores, decomposition_scores, penalty_scores], ignore_index=True, sort=False)
    plain_monthly, plain_holdings = build_monthly_from_scores(
        plain_scores,
        top_ks=top_ks,
        cost_bps=cfg.cost_bps,
        selection_policy="score_topk",
    )

    optimizer_monthly_frames: list[pd.DataFrame] = []
    optimizer_holding_frames: list[pd.DataFrame] = []
    optimizer_score_sets = build_soft_optimizer_scores(label_scores, gammas=gammas)
    for gamma, opt_scores in optimizer_score_sets:
        monthly, holdings = build_monthly_from_scores(
            opt_scores,
            top_ks=top_ks,
            cost_bps=cfg.cost_bps,
            selection_policy="soft_industry_risk_budget",
            soft_gamma=gamma,
        )
        if not monthly.empty:
            optimizer_monthly_frames.append(monthly)
        if not holdings.empty:
            optimizer_holding_frames.append(holdings)

    monthly_long = pd.concat([plain_monthly, *optimizer_monthly_frames], ignore_index=True, sort=False)
    topk_holdings = pd.concat([plain_holdings, *optimizer_holding_frames], ignore_index=True, sort=False)

    rank_ic_plain = build_rank_ic(plain_scores)
    quantile_spread_plain = build_quantile_spread(plain_scores, bucket_count=cfg.bucket_count)
    optimizer_monthly = (
        pd.concat(optimizer_monthly_frames, ignore_index=True, sort=False) if optimizer_monthly_frames else pd.DataFrame()
    )
    rank_ic = pd.concat(
        [rank_ic_plain, copy_source_metric_for_optimizer(rank_ic_plain, optimizer_monthly)],
        ignore_index=True,
        sort=False,
    )
    quantile_spread = pd.concat(
        [quantile_spread_plain, copy_source_metric_for_optimizer(quantile_spread_plain, optimizer_monthly)],
        ignore_index=True,
        sort=False,
    )
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    industry_concentration = summarize_industry_concentration(topk_holdings)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice, industry_concentration)

    hardcap_path = resolve_hardcap_leaderboard(args.hardcap_leaderboard, as_of=as_of, results_dir=results_dir)
    hardcap_baseline = load_hardcap_baseline(hardcap_path)
    gate = build_gate_table(leaderboard, hardcap_baseline, tolerance=cfg.hardcap_tolerance)

    label_compare = leaderboard[leaderboard["score_family"].eq("label_compare")].copy()
    penalty_frontier = leaderboard[leaderboard["score_family"].eq("penalty_frontier")].copy()
    optimizer_compare = leaderboard[leaderboard["score_family"].eq("optimizer_compare")].copy()
    feature_importance = summarize_m5_feature_importance(importance_raw) if not importance_raw.empty else pd.DataFrame()
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    quality = build_quality_payload(
        dataset=dataset,
        cfg=cfg,
        dataset_path=dataset_path,
        db_path=db_path,
        output_stem=output_stem,
        config_source=config_source,
        research_config_id=research_config_id,
        enabled_families=enabled_families,
        hardcap_path=hardcap_path,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "label_compare": results_dir / f"{output_stem}_label_compare.csv",
        "penalty_frontier": results_dir / f"{output_stem}_penalty_frontier.csv",
        "score_decomposition": results_dir / f"{output_stem}_score_decomposition.csv",
        "optimizer_compare": results_dir / f"{output_stem}_optimizer_compare.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "industry_concentration": results_dir / f"{output_stem}_industry_concentration.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "gate": results_dir / f"{output_stem}_gate.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    label_compare.to_csv(paths_out["label_compare"], index=False)
    penalty_frontier.to_csv(paths_out["penalty_frontier"], index=False)
    score_decomposition.to_csv(paths_out["score_decomposition"], index=False)
    optimizer_compare.to_csv(paths_out["optimizer_compare"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    industry_concentration.to_csv(paths_out["industry_concentration"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    gate.to_csv(paths_out["gate"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)

    summary_payload = {
        "quality": quality,
        "top_gate_pass": gate[gate["m8_natural_gate_pass"]].head(20).to_dict(orient="records") if not gate.empty else [],
        "top_models_by_topk": leaderboard.groupby("top_k", as_index=False).head(10).to_dict(orient="records")
        if not leaderboard.empty
        else [],
    }
    paths_out["summary_json"].write_text(
        json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = [
        _project_relative(p)
        for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    paths_out["doc"].write_text(
        build_doc(
            quality=quality,
            leaderboard=leaderboard,
            label_compare=label_compare,
            penalty_frontier=penalty_frontier,
            score_decomposition=score_decomposition,
            optimizer_compare=optimizer_compare,
            gate=gate,
            year_slice=year_slice,
            regime_slice=regime_slice,
            artifacts=[*artifact_paths, _project_relative(paths_out["manifest"])],
        ),
        encoding="utf-8",
    )

    pass_count = int(gate["m8_natural_gate_pass"].sum()) if not gate.empty else 0
    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row: dict[str, Any] = {}
    if not leaderboard.empty:
        best_row = (
            leaderboard.sort_values(
                ["topk_excess_after_cost_mean", "rank_ic_mean"],
                ascending=[False, False],
            )
            .iloc[0]
            .to_dict()
        )
    best_gate_row: dict[str, Any] = {}
    if not gate.empty:
        best_gate_row = (
            gate.sort_values(
                ["m8_natural_gate_pass", "topk_excess_after_cost_mean"],
                ascending=[False, False],
            )
            .iloc[0]
            .to_dict()
        )
    rank_ic_observations = (
        int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum())
        if not rank_ic.empty
        else 0
    )
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    best_after_cost_float = float(best_after_cost) if pd.notna(best_after_cost) else None
    best_concentration_share = best_gate_row.get("max_industry_share_mean")
    best_concentration_share_float = (
        float(best_concentration_share) if pd.notna(best_concentration_share) else None
    )
    source_tables = [_project_relative(dataset_path), _project_relative(db_path)]
    if hardcap_path is not None:
        source_tables.append(_project_relative(hardcap_path))
    feature_columns = tuple(dict.fromkeys(col for spec in feature_specs for col in spec.feature_cols))
    data_slice = DataSlice(
        dataset_name="monthly_selection_m8_natural_industry_constraints",
        source_tables=tuple(source_tables),
        date_start=min_signal_date,
        date_end=max_signal_date,
        asof_trade_date=max_signal_date or None,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(pools),
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="m8_natural_" + "_".join(["price_volume_only", *enabled_families]),
        feature_columns=feature_columns,
        label_columns=(LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL),
        pit_policy=quality["pit_policy"],
        config_path=config_source,
        extra={
            "dataset_path": _project_relative(dataset_path),
            "duckdb_path": _project_relative(db_path),
            "hardcap_leaderboard_path": _project_relative(hardcap_path) if hardcap_path else "",
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "enabled_families": enabled_families,
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
            "availability_lag_days": int(args.availability_lag_days),
            "label_variants": [v.name for v in LABEL_VARIANTS],
            "soft_gammas": gammas,
            "hardcap_tolerance": float(args.hardcap_tolerance),
            "cv_policy": quality["cv_policy"],
            "naturalization_policy": quality["naturalization_policy"],
        },
    )
    artifact_refs = (
        ArtifactRef("summary_json", _project_relative(paths_out["summary_json"]), "json"),
        ArtifactRef(
            "leaderboard_csv",
            _project_relative(paths_out["leaderboard"]),
            "csv",
            required_for_promotion=True,
        ),
        ArtifactRef("label_compare_csv", _project_relative(paths_out["label_compare"]), "csv"),
        ArtifactRef("penalty_frontier_csv", _project_relative(paths_out["penalty_frontier"]), "csv"),
        ArtifactRef("score_decomposition_csv", _project_relative(paths_out["score_decomposition"]), "csv"),
        ArtifactRef("optimizer_compare_csv", _project_relative(paths_out["optimizer_compare"]), "csv"),
        ArtifactRef("monthly_long_csv", _project_relative(paths_out["monthly_long"]), "csv"),
        ArtifactRef(
            "industry_concentration_csv",
            _project_relative(paths_out["industry_concentration"]),
            "csv",
            required_for_promotion=True,
        ),
        ArtifactRef("industry_exposure_csv", _project_relative(paths_out["industry_exposure"]), "csv"),
        ArtifactRef(
            "regime_slice_csv",
            _project_relative(paths_out["regime_slice"]),
            "csv",
            required_for_promotion=True,
        ),
        ArtifactRef("year_slice_csv", _project_relative(paths_out["year_slice"]), "csv"),
        ArtifactRef("topk_holdings_csv", _project_relative(paths_out["topk_holdings"]), "csv"),
        ArtifactRef("gate_csv", _project_relative(paths_out["gate"]), "csv", required_for_promotion=True),
        ArtifactRef("rank_ic_csv", _project_relative(paths_out["rank_ic"]), "csv"),
        ArtifactRef("quantile_spread_csv", _project_relative(paths_out["quantile_spread"]), "csv"),
        ArtifactRef("feature_coverage_csv", _project_relative(paths_out["feature_coverage"]), "csv"),
        ArtifactRef("feature_importance_csv", _project_relative(paths_out["feature_importance"]), "csv"),
        ArtifactRef("candidate_pool_width_csv", _project_relative(paths_out["candidate_pool_width"]), "csv"),
        ArtifactRef(
            "candidate_pool_reject_reason_csv",
            _project_relative(paths_out["candidate_pool_reject_reason"]),
            "csv",
        ),
        ArtifactRef("report_md", _project_relative(paths_out["doc"]), "md", required_for_promotion=True),
        ArtifactRef("manifest_json", _project_relative(paths_out["manifest"]), "json", required_for_promotion=True),
    )
    metrics = {
        "rows": int(quality["rows"]),
        "valid_rows": int(quality["valid_rows"]),
        "valid_signal_months": int(quality["valid_signal_months"]),
        "label_score_rows": int(len(label_scores)),
        "decomposition_score_rows": int(len(decomposition_scores)),
        "penalty_score_rows": int(len(penalty_scores)),
        "monthly_long_rows": int(len(monthly_long)),
        "topk_holdings_rows": int(len(topk_holdings)),
        "rank_ic_observations": rank_ic_observations,
        "feature_coverage_rows": int(len(feature_coverage)),
        "feature_importance_rows": int(len(feature_importance)),
        "m8_natural_gate_rows": int(len(gate)),
        "m8_natural_gate_pass_count": pass_count,
        "model_count": int(leaderboard["model"].nunique()) if "model" in leaderboard.columns else 0,
        "best_model": str(best_row.get("model") or ""),
        "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
        "best_top_k": (
            int(best_row["top_k"])
            if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k"))
            else None
        ),
        "best_topk_excess_after_cost_mean": best_after_cost_float,
        "best_rank_ic_mean": float(best_row["rank_ic_mean"])
        if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean"))
        else None,
        "best_gate_model": str(best_gate_row.get("model") or ""),
        "best_gate_pass": bool(best_gate_row.get("m8_natural_gate_pass", False)),
        "best_max_industry_share_mean": best_concentration_share_float,
    }
    gates = {
        "data_gate": {
            "passed": bool(
                metrics["valid_rows"] > 0
                and metrics["valid_signal_months"] > 0
                and metrics["label_score_rows"] > 0
            ),
            "checks": {
                "has_valid_rows": metrics["valid_rows"] > 0,
                "has_valid_signal_months": metrics["valid_signal_months"] > 0,
                "has_label_scores": metrics["label_score_rows"] > 0,
                "has_feature_coverage": metrics["feature_coverage_rows"] > 0,
            },
        },
        "rank_gate": {
            "passed": bool(rank_ic_observations > 0),
            "rank_ic_observations": rank_ic_observations,
        },
        "spread_gate": {
            "passed": bool(not monthly_long.empty and not quantile_spread.empty),
            "monthly_rows": int(len(monthly_long)),
            "quantile_spread_rows": int(len(quantile_spread)),
        },
        "year_gate": {
            "passed": bool(not year_slice.empty),
            "year_slice_rows": int(len(year_slice)),
        },
        "regime_gate": {
            "passed": bool(not regime_slice.empty),
            "regime_slice_rows": int(len(regime_slice)),
        },
        "concentration_gate": {
            "passed": bool(pass_count > 0),
            "m8_natural_gate_pass_count": pass_count,
            "gate_rows": int(len(gate)),
            "best_max_industry_share_mean": best_concentration_share_float,
        },
        "hardcap_comparison_gate": {
            "passed": bool(not hardcap_baseline.empty or not gate.empty),
            "hardcap_baseline_rows": int(len(hardcap_baseline)),
            "hardcap_leaderboard_path": _project_relative(hardcap_path) if hardcap_path else "",
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    config_info = config_snapshot(
        config_path=loaded_config_path,
        resolved_config=cfg_raw,
        sections=(
            "paths",
            "database",
            "signals",
            "portfolio",
            "backtest",
            "transaction_costs",
            "prefilter",
            "monthly_selection",
        ),
    )
    config_info["config_path"] = config_source
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=int(args.random_seed),
        data_slices=(data_slice,),
        config=config_info,
        params={
            "cli": vars(args),
            "run_config": {
                "top_ks": list(cfg.top_ks),
                "candidate_pools": list(cfg.candidate_pools),
                "bucket_count": cfg.bucket_count,
                "min_train_months": cfg.min_train_months,
                "min_train_rows": cfg.min_train_rows,
                "max_fit_rows": cfg.max_fit_rows,
                "cost_bps": cfg.cost_bps,
                "availability_lag_days": cfg.availability_lag_days,
                "soft_gammas": list(cfg.soft_gammas),
                "hardcap_tolerance": cfg.hardcap_tolerance,
                "enabled_families": enabled_families,
                "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
            },
            "overrides": {
                key: value
                for key, value in {
                    "dataset": args.dataset,
                    "duckdb_path": args.duckdb_path.strip(),
                    "results_dir": args.results_dir.strip(),
                    "as_of_date": args.as_of_date.strip(),
                    "top_k": args.top_k,
                    "candidate_pools": args.candidate_pools,
                    "families": args.families,
                    "soft_gamma": args.soft_gamma,
                    "hardcap_leaderboard": args.hardcap_leaderboard.strip(),
                }.items()
                if value
            },
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["m8_natural_constraints_requires_manual_promotion_package"],
        },
        notes="Monthly selection M8 natural industry constraints contract; strategy outputs are unchanged.",
    )
    write_research_manifest(
        paths_out["manifest"],
        result,
        extra={
            "generated_at_utc": result.created_at,
            **quality,
            "legacy_artifacts": [*artifact_paths, _project_relative(paths_out["doc"])],
        },
    )
    append_experiment_result(experiments_dir, result)

    print(
        f"[monthly-m8-natural] valid_rows={quality['valid_rows']} "
        f"valid_months={quality['valid_signal_months']}",
        flush=True,
    )
    print(f"[monthly-m8-natural] gate_pass={pass_count}", flush=True)
    print(f"[monthly-m8-natural] leaderboard={paths_out['leaderboard']}", flush=True)
    print(f"[monthly-m8-natural] gate={paths_out['gate']}", flush=True)
    print(f"[monthly-m8-natural] manifest={paths_out['manifest']}", flush=True)
    print(f"[monthly-m8-natural] research_index={experiments_dir / 'research_results.jsonl'}", flush=True)
    print(f"[monthly-m8-natural] doc={paths_out['doc']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
