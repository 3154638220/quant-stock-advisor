"""月度选股 Learning-to-Rank 管线。

从 scripts/run_monthly_selection_ltr.py 提取核心算法逻辑：
- LTR 训练/预测 (XGBoostRanker, Top20 calibrator, ensemble)
- Walk-forward query-group 打分
- 特征重要性汇总

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.monthly_baselines import (
    EXCESS_COL,
    LABEL_COL,
    TOP20_COL,
    _rank_pct_score,
    _score_base_columns,
    normalize_model_n_jobs,
    valid_pool_frame,
)
from src.pipeline.monthly_multisource import (
    FeatureSpec,
    M5RunConfig,
    _cap_fit_rows,
    attach_enabled_families,
    build_feature_specs,
)


@dataclass(frozen=True)
class M6RunConfig:
    top_ks: tuple[int, ...] = (20, 30, 50)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    relevance_grades: int = 5
    model_n_jobs: int = 0
    ltr_models: tuple[str, ...] = (
        "xgboost_rank_ndcg",
        "xgboost_rank_pairwise",
        "top20_calibrated",
        "ranker_top20_ensemble",
    )


def build_m6_feature_spec(enabled_families: list[str]) -> FeatureSpec:
    specs = build_feature_specs(enabled_families)
    spec = specs[-1]
    return FeatureSpec(
        name="m6_core_" + "_".join(spec.families),
        families=spec.families,
        feature_cols=spec.feature_cols,
    )


def build_ltr_relevance(
    train: pd.DataFrame,
    *,
    label_col: str = EXCESS_COL,
    date_col: str = "signal_date",
    grades: int = 5,
) -> pd.Series:
    vals = pd.to_numeric(train.get(label_col), errors="coerce")
    if vals.notna().sum() == 0:
        vals = pd.to_numeric(train[LABEL_COL], errors="coerce")
    pct = vals.groupby(train[date_col], sort=False).rank(method="average", pct=True)
    rel = np.ceil(pct.fillna(0.0) * max(int(grades), 2)).astype(int) - 1
    return rel.clip(lower=0, upper=max(int(grades), 2) - 1).astype(int)


def _prepare_ranker_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _sort_for_query_groups(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)


def _query_group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("signal_date", sort=False).size().to_numpy(dtype=np.uint32)


def _train_predict_xgboost_ranker(
    *,
    model_name: str,
    objective: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    relevance_grades: int,
    model_n_jobs: int = 1,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    try:
        from xgboost import XGBRanker
    except Exception as exc:
        warnings.warn(f"跳过 XGBoost ranker，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    train_sorted = _sort_for_query_groups(train)
    test_sorted = _sort_for_query_groups(test)
    y = build_ltr_relevance(train_sorted, grades=relevance_grades)
    if y.nunique(dropna=True) < 2:
        return None, pd.DataFrame()

    model = XGBRanker(
        n_estimators=120, max_depth=3, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.85,
        objective=objective, eval_metric="ndcg@20",
        random_state=random_seed, n_jobs=normalize_model_n_jobs(model_n_jobs),
        tree_method="hist",
    )
    try:
        model.fit(_prepare_ranker_matrix(train_sorted, feature_cols), y,
                  group=_query_group_sizes(train_sorted), verbose=False)
        raw_pred = pd.Series(model.predict(_prepare_ranker_matrix(test_sorted, feature_cols)), index=test_sorted.index)
    except Exception as exc:
        warnings.warn(f"{model_name} 训练失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    base = _score_base_columns(test_sorted)
    out = base.copy()
    out["model"] = model_name
    out["model_type"] = "xgboost_ranker"
    out["score"] = raw_pred.groupby(base["signal_date"], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False,
    )
    importance = pd.DataFrame({
        "model": model_name, "feature": feature_cols,
        "importance": model.feature_importances_, "signed_weight": np.nan,
    })
    return out, importance


def _train_predict_top20_calibrated(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    model_n_jobs: int = 1,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        warnings.warn(f"跳过 XGBoost top20 classifier，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    y = pd.to_numeric(train.get(TOP20_COL), errors="coerce")
    if y.notna().sum() == 0:
        pct = train.groupby("signal_date")[LABEL_COL].rank(method="first", pct=True, ascending=False)
        y = (pct <= 0.20).astype(int)
    y = y.fillna(0).clip(0, 1).astype(int)
    if y.nunique(dropna=True) < 2:
        return None, pd.DataFrame()
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    model = XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=max(neg / max(pos, 1.0), 1.0),
        random_state=random_seed, n_jobs=normalize_model_n_jobs(model_n_jobs),
        tree_method="hist",
    )
    try:
        model.fit(_prepare_ranker_matrix(train, feature_cols), y, verbose=False)
        raw_pred = pd.Series(model.predict_proba(_prepare_ranker_matrix(test, feature_cols))[:, 1], index=test.index)
    except Exception as exc:
        warnings.warn(f"M6_top20_calibrated 训练失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    base = _score_base_columns(test)
    out = base.copy()
    out["model"] = "M6_top20_calibrated"
    out["model_type"] = "top_bucket_classifier_rank_calibrated"
    out["score"] = raw_pred.groupby(base["signal_date"], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False,
    )
    importance = pd.DataFrame({
        "model": "M6_top20_calibrated", "feature": feature_cols,
        "importance": model.feature_importances_, "signed_weight": np.nan,
    })
    return out, importance


def _build_ranker_top20_ensemble(current_scores: list[pd.DataFrame]) -> pd.DataFrame:
    ranker = next((x for x in current_scores if (not x.empty and x["model"].iloc[0] == "M6_xgboost_rank_ndcg")), None)
    top20 = next((x for x in current_scores if (not x.empty and x["model"].iloc[0] == "M6_top20_calibrated")), None)
    if ranker is None or top20 is None:
        return pd.DataFrame()
    cols = ["signal_date", "candidate_pool_version", "symbol"]
    merged = ranker[cols + ["score"]].rename(columns={"score": "_ranker_score"}).merge(
        top20[cols + ["score"]].rename(columns={"score": "_top20_score"}), on=cols, how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    base = ranker.drop(columns=["score", "rank", "model", "model_type"], errors="ignore").merge(
        merged[cols + ["_ranker_score", "_top20_score"]], on=cols, how="inner",
    )
    out = base.copy()
    out["model"] = "M6_ranker_top20_ensemble"
    out["model_type"] = "ranker_classifier_ensemble"
    out["score"] = 0.60 * out["_ranker_score"] + 0.40 * out["_top20_score"]
    out = out.drop(columns=["_ranker_score", "_top20_score"], errors="ignore")
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False,
    )
    return out


def _tag_importance(imp: pd.DataFrame, spec: FeatureSpec, pool: str, test_month: Any) -> pd.DataFrame:
    out = imp.copy()
    out["feature_spec"] = spec.name
    out["feature_families"] = ",".join(spec.families)
    out["candidate_pool_version"] = pool
    out["test_signal_date"] = pd.Timestamp(test_month)
    return out


def build_walk_forward_ltr_scores(
    dataset: pd.DataFrame, spec: FeatureSpec, cfg: M6RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_pool_frame(dataset)
    feature_cols = [c for c in spec.feature_cols if c in valid.columns]
    if valid.empty or not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    requested = set(cfg.ltr_models)

    for pool, pool_df in valid.groupby("candidate_pool_version", sort=True):
        months = sorted(pd.to_datetime(pool_df["signal_date"]).dropna().unique())
        for test_month in months:
            train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
            train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
            current_scores: list[pd.DataFrame] = []
            if "xgboost_rank_ndcg" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_ndcg", objective="rank:ndcg",
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "xgboost_rank_pairwise" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_pairwise", objective="rank:pairwise",
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "top20_calibrated" in requested or "ranker_top20_ensemble" in requested:
                scores, imp = _train_predict_top20_calibrated(
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "ranker_top20_ensemble" in requested:
                ensemble = _build_ranker_top20_ensemble(current_scores)
                if not ensemble.empty:
                    current_scores.append(ensemble)
            for scores in current_scores:
                scores = scores.copy()
                scores["feature_spec"] = spec.name
                scores["feature_families"] = ",".join(spec.families)
                score_frames.append(scores)

    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def summarize_ltr_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    out = importance.groupby(
        ["feature_spec", "feature_families", "candidate_pool_version", "model", "feature"], sort=True,
    ).agg(
        importance=("importance", "mean"),
        signed_weight=("signed_weight", "mean"),
        observations=("test_signal_date", "nunique"),
    ).reset_index()
    return out.sort_values(["candidate_pool_version", "model", "importance"], ascending=[True, True, False])
