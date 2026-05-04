"""月度选股研究报告生成。

从 scripts/run_monthly_selection_report.py 提取核心报告逻辑：
- 推荐信号日选择
- full-fit 报告打分
- 推荐表构建
- M9 数据完整性检查
- 特征覆盖策略

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.monthly_ltr import (
    _tag_importance,
    _train_predict_xgboost_ranker,
)
from src.pipeline.monthly_multisource import (
    _cap_fit_rows,
)


@dataclass(frozen=True)
class M7RunConfig:
    top_ks: tuple[int, ...] = (20, 30)
    report_top_k: int = 20
    candidate_pools: tuple[str, ...] = ("U2_risk_sane",)
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    relevance_grades: int = 5
    model_name: str = "M6_xgboost_rank_ndcg"
    min_core_feature_coverage: float = 0.30
    model_n_jobs: int = 0


def select_report_signal_date(
    dataset: pd.DataFrame,
    *,
    candidate_pools: tuple[str, ...],
    requested: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    df = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if requested:
        target = pd.Timestamp(requested).normalize()
        part = df[df["signal_date"] == target]
        if part.empty:
            raise ValueError(f"报告信号日不存在于 dataset: {target.date()}")
        if not part["candidate_pool_pass"].astype(bool).any():
            raise ValueError(f"报告信号日无 candidate_pool_pass 标的: {target.date()}")
        passed = part[part["candidate_pool_pass"].astype(bool)]
        if "next_trade_date" in passed.columns and not passed["next_trade_date"].notna().all():
            raise ValueError(f"报告信号日存在 candidate_pool_pass 但缺少 next_trade_date: {target.date()}")
        return target
    eligible = df[df["candidate_pool_pass"].astype(bool)].copy()
    if "next_trade_date" in eligible.columns:
        eligible = eligible[eligible["next_trade_date"].notna()].copy()
    if eligible.empty:
        raise ValueError("没有可用于 M7 推荐的 candidate_pool_pass 信号月。")
    return pd.Timestamp(eligible["signal_date"].max()).normalize()


def build_full_fit_report_scores(
    dataset: pd.DataFrame,
    spec: Any,
    cfg: M7RunConfig,
    *,
    report_signal_date: pd.Timestamp,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_feature_cols = list(feature_cols) if feature_cols is not None else [c for c in spec.feature_cols if c in dataset.columns]
    active_feature_cols = [c for c in active_feature_cols if c in dataset.columns]
    if not active_feature_cols:
        return pd.DataFrame(), pd.DataFrame()
    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for pool in cfg.candidate_pools:
        pool_df = dataset[dataset["candidate_pool_version"] == pool].copy()
        train = pool_df[
            pool_df["candidate_pool_pass"].astype(bool)
            & (pool_df["signal_date"] < report_signal_date)
            & pool_df["label_forward_1m_o2o_return"].notna()
        ].copy()
        test = pool_df[
            pool_df["candidate_pool_pass"].astype(bool) & (pool_df["signal_date"] == report_signal_date)
        ].copy()
        if test.empty:
            continue
        if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows:
            continue
        train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
        scores, imp = _train_predict_xgboost_ranker(
            model_name=cfg.model_name, objective="rank:ndcg",
            train=train_fit, test=test, feature_cols=active_feature_cols,
            random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
            model_n_jobs=cfg.model_n_jobs,
        )
        if scores is not None and not scores.empty:
            scores = scores.copy()
            scores["feature_spec"] = spec.name
            scores["feature_families"] = ",".join(spec.families)
            scores["score_percentile"] = pd.to_numeric(scores["score"], errors="coerce")
            score_frames.append(scores)
        if imp is not None and not imp.empty:
            importance_frames.append(_tag_importance(imp, spec, pool, report_signal_date))

    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def summarize_report_feature_coverage(
    dataset: pd.DataFrame, spec: Any, *, candidate_pools: tuple[str, ...],
) -> pd.DataFrame:
    base = dataset[dataset["candidate_pool_version"].isin(candidate_pools)].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for pool, pool_df in base.groupby("candidate_pool_version", sort=True):
        pool_pass_part = pool_df["candidate_pool_pass"].astype(bool)
        for col in spec.feature_cols:
            raw_col = col[:-2] if col.endswith("_z") else col
            vals = pd.to_numeric(pool_df[raw_col], errors="coerce") if raw_col in pool_df.columns else pd.Series(np.nan, index=pool_df.index)
            rows.append({
                "candidate_pool_version": pool, "feature_spec": spec.name,
                "families": ",".join(spec.families), "feature": col, "raw_feature": raw_col,
                "rows": int(len(pool_df)), "candidate_pool_pass_rows": int(pool_pass_part.sum()),
                "non_null": int(vals.notna().sum()),
                "coverage_ratio": float(vals.notna().mean()) if len(pool_df) else np.nan,
                "candidate_pool_pass_coverage_ratio": float(vals.loc[pool_pass_part].notna().mean()) if pool_pass_part.any() else np.nan,
                "first_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].min().date()) if vals.notna().any() else "",
                "last_signal_date": str(pool_df.loc[vals.notna(), "signal_date"].max().date()) if vals.notna().any() else "",
            })
    return pd.DataFrame(rows)


def apply_m9_feature_coverage_policy(
    dataset: pd.DataFrame, spec: Any, feature_coverage: pd.DataFrame,
    *, candidate_pools: tuple[str, ...], min_core_coverage: float = 0.30,
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    active: list[str] = []
    coverage = feature_coverage.copy()
    if coverage.empty:
        active = [c for c in spec.feature_cols if c in dataset.columns]
        return active, pd.DataFrame()
    coverage["candidate_pool_pass_coverage_ratio"] = pd.to_numeric(
        coverage.get("candidate_pool_pass_coverage_ratio"), errors="coerce",
    )
    for feature in spec.feature_cols:
        if feature not in dataset.columns:
            rows.append({"feature": feature, "candidate_pool_pass_coverage_ratio": np.nan,
                         "m9_feature_policy": "missing_from_dataset", "active_feature": ""})
            continue
        raw_feature = feature[:-2] if feature.endswith("_z") else feature
        part = coverage[
            (coverage["feature"] == feature)
            & (coverage["candidate_pool_version"].isin(candidate_pools) if "candidate_pool_version" in coverage.columns else True)
        ].copy()
        cov = float(part["candidate_pool_pass_coverage_ratio"].min()) if not part.empty else np.nan
        missing_flag = f"is_missing_{raw_feature}"
        if np.isfinite(cov) and cov < float(min_core_coverage):
            if missing_flag in dataset.columns:
                active.append(missing_flag)
                policy = "missing_flag_only_low_coverage"
                active_feature = missing_flag
            else:
                policy = "dropped_low_coverage_no_missing_flag"
                active_feature = ""
        else:
            active.append(feature)
            policy = "core_feature"
            active_feature = feature
        rows.append({"feature": feature, "raw_feature": raw_feature,
                     "candidate_pool_pass_coverage_ratio": cov,
                     "m9_feature_policy": policy, "active_feature": active_feature})
    return list(dict.fromkeys(active)), pd.DataFrame(rows)


def apply_industry_cap(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    max_industry_share: float = 0.30,
    industry_col: str = "industry",
    score_col: str = "score",
) -> pd.DataFrame:
    """贪心选取：按 score 降序遍历，当某行业已达上限时跳过。

    保证最终推荐集大小 <= top_k。
    max_industry_share = 0.30 表示 Top20 中单行业最多 6 只。
    """
    if ranked.empty:
        return ranked
    cap = max(1, int(np.floor(max_industry_share * top_k)))
    industry_counts: dict[str, int] = {}
    selected = []
    for _, row in ranked.sort_values(score_col, ascending=False).iterrows():
        ind = str(row.get(industry_col, "") or "")
        if industry_counts.get(ind, 0) >= cap:
            continue
        selected.append(row)
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)
