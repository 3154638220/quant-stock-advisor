#!/usr/bin/env python3
"""M6: 月度选股 learning-to-rank 主模型。

本脚本消费 M2 canonical dataset，并复用 M5 的多源特征接入与 M4/M5
评价口径，训练真正按月度 query 分组的截面排序模型。
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
from scripts.run_monthly_selection_baselines import (
    EXCESS_COL,
    LABEL_COL,
    POOL_RULES,
    TOP20_COL,
    _format_markdown_table,
    _json_sanitize,
    _rank_pct_score,
    _score_base_columns,
    build_leaderboard,
    build_monthly_long,
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
from scripts.run_monthly_selection_multisource import (
    FUNDAMENTAL_RAW_FEATURES,
    FeatureSpec,
    M5RunConfig,
    _cap_fit_rows,
    attach_enabled_families,
    build_feature_specs,
    industry_neutral_zscore,
    summarize_feature_coverage_by_spec,
)
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    file_sha256,
    stable_hash,
    utc_now_iso,
    write_research_manifest,
)
from src.settings import config_path_candidates, load_config, resolve_config_path

# ═══════════════════════════════════════════════════════════════════════════
# 以下核心管线函数已提取到 src.pipeline.monthly_ltr：
#   build_m6_feature_spec, build_ltr_relevance, build_walk_forward_ltr_scores,
#   summarize_ltr_feature_importance, _train_predict_xgboost_ranker, 等。
# 本脚本保留本地副本仅为向后兼容；后续维护请直接修改 monthly_ltr 中的版本。
# ═══════════════════════════════════════════════════════════════════════════
from src.pipeline.monthly_ltr import (  # noqa: F401
    M6RunConfig,
    build_ltr_relevance,
    build_m6_feature_spec,
    build_walk_forward_ltr_scores,
    summarize_ltr_feature_importance,
    _build_ranker_top20_ensemble,
    _tag_importance,
    _train_predict_top20_calibrated,
    _train_predict_xgboost_ranker,
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
    availability_lag_days: int = 45
    relevance_grades: int = 5
    model_n_jobs: int = 0
    # P1-1: XGBoost 超参数时序感知调优
    hpo_enabled: bool = False
    hpo_n_trials: int = 30
    hpo_cv_folds: int = 3
    # P1-2: Walk-forward 窗口配置
    window_type: str = "expanding"  # "rolling" | "expanding"
    halflife_months: float = 36.0
    ltr_models: tuple[str, ...] = (
        "xgboost_rank_ndcg",
        "xgboost_rank_pairwise",
        "top20_calibrated",
        "ranker_top20_ensemble",
        # P2-3: Stacking 集成（需 OOF 收集，默认关闭以保持向后兼容）
        # "stacking_ensemble",
    )
    # P2-3: OOF meta-learner 配置
    stacking_enabled: bool = False
    stacking_meta_learner: str = "logistic"
    stacking_meta_C: float = 0.1
    # P0-1: 对基本面因子使用行业内 z-score 中性化（默认 False，保持向后兼容）
    use_industry_neutral_zscore: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M6 learning-to-rank 主模型")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m6_ltr")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument(
        "--max-fit-rows",
        type=int,
        default=0,
        help="每个 walk-forward 训练窗的确定性抽样上限；0 表示使用全部训练行。",
    )
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument("--relevance-grades", type=int, default=5)
    p.add_argument(
        "--model-n-jobs",
        type=int,
        default=0,
        help="模型训练线程数；0 表示使用全部 CPU 核心，1 保持旧的单线程行为。",
    )
    p.add_argument(
        "--families",
        type=str,
        default="industry_breadth,fund_flow,fundamental",
        help="M6 主输入特征家族。默认采用 M5 收敛出的 price_volume + industry_breadth + fund_flow + fundamental。",
    )
    p.add_argument(
        "--ltr-models",
        type=str,
        default="xgboost_rank_ndcg,xgboost_rank_pairwise,top20_calibrated,ranker_top20_ensemble",
        help="可选 xgboost_rank_ndcg,xgboost_rank_pairwise,top20_calibrated,ranker_top20_ensemble。",
    )
    p.add_argument(
        "--use-industry-neutral-zscore",
        action="store_true",
        default=False,
        help="P0-1: 对基本面因子使用行业内 z-score 中性化（生成 _ind_z 列替代 _z）。",
    )
    # H2: 换仓频率参数化
    p.add_argument(
        "--rebalance-rule", type=str, default="",
        choices=["", "W", "M", "BM", "Q", "W-FRI"],
        help="换仓频率（空字符串=从 dataset 自动检测，默认 M）",
    )
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


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def build_m6_feature_spec(
    enabled_families: list[str],
    *,
    use_industry_neutral_zscore: bool = False,
) -> FeatureSpec:
    """构建 M6 LTR 特征规格。P0-1: 支持行业内 z-score 中性化。"""
    specs = build_feature_specs(
        enabled_families,
        use_industry_neutral_zscore=use_industry_neutral_zscore,
    )
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
    """Convert each monthly cross-section into integer ranking relevance labels."""
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
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        warnings.warn(f"跳过 XGBoost ranker，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    train_sorted = _sort_for_query_groups(train)
    test_sorted = _sort_for_query_groups(test)
    y = build_ltr_relevance(train_sorted, grades=relevance_grades)
    if y.nunique(dropna=True) < 2:
        return None, pd.DataFrame()

    model = XGBRanker(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        objective=objective,
        eval_metric="ndcg@20",
        random_state=random_seed,
        n_jobs=normalize_model_n_jobs(model_n_jobs),
        tree_method="hist",
    )
    try:
        model.fit(
            _prepare_ranker_matrix(train_sorted, feature_cols),
            y,
            group=_query_group_sizes(train_sorted),
            verbose=False,
        )
        raw_pred = pd.Series(model.predict(_prepare_ranker_matrix(test_sorted, feature_cols)), index=test_sorted.index)
    except Exception as exc:  # pragma: no cover - defensive for optional model failures
        warnings.warn(f"{model_name} 训练失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    base = _score_base_columns(test_sorted)
    out = base.copy()
    out["model"] = model_name
    out["model_type"] = "xgboost_ranker"
    out["score"] = raw_pred.groupby(base["signal_date"], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first",
        ascending=False,
    )
    importance = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_cols,
            "importance": model.feature_importances_,
            "signed_weight": np.nan,
        }
    )
    return out, importance


def _train_predict_top20_calibrated(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    model_n_jobs: int = 1,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    x_train = _prepare_ranker_matrix(train, feature_cols)
    x_test = _prepare_ranker_matrix(test, feature_cols)
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        warnings.warn(
            f"XGBoost top20 classifier 不可用，使用 sklearn LogisticRegression 兜底: {exc}",
            RuntimeWarning,
        )
        XGBClassifier = None  # type: ignore[assignment]

    y = pd.to_numeric(train.get(TOP20_COL), errors="coerce")
    if y.notna().sum() == 0:
        pct = train.groupby("signal_date")[LABEL_COL].rank(method="first", pct=True, ascending=False)
        y = (pct <= 0.20).astype(int)
    y = y.fillna(0).clip(0, 1).astype(int)
    if y.nunique(dropna=True) < 2:
        return None, pd.DataFrame()
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    try:
        if XGBClassifier is not None:
            model = XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.045,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=max(neg / max(pos, 1.0), 1.0),
                random_state=random_seed,
                n_jobs=normalize_model_n_jobs(model_n_jobs),
                tree_method="hist",
            )
            model.fit(x_train, y, verbose=False)
            raw_pred = pd.Series(model.predict_proba(x_test)[:, 1], index=test.index)
            importance_values = model.feature_importances_
            signed_weights = np.full(len(feature_cols), np.nan)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    solver="liblinear",
                    random_state=random_seed,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            )
            model.fit(x_train, y)
            raw_pred = pd.Series(model.predict_proba(x_test)[:, 1], index=test.index)
            coef = model.named_steps["logisticregression"].coef_[0]
            importance_values = np.abs(coef)
            signed_weights = coef
    except Exception as exc:  # pragma: no cover - defensive for optional model failures
        warnings.warn(f"M6_top20_calibrated 训练失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    base = _score_base_columns(test)
    out = base.copy()
    out["model"] = "M6_top20_calibrated"
    out["model_type"] = "top_bucket_classifier_rank_calibrated"
    out["score"] = raw_pred.groupby(base["signal_date"], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first",
        ascending=False,
    )
    importance = pd.DataFrame(
        {
            "model": "M6_top20_calibrated",
            "feature": feature_cols,
            "importance": importance_values,
            "signed_weight": signed_weights,
        }
    )
    return out, importance


def _build_ranker_top20_ensemble(current_scores: list[pd.DataFrame]) -> pd.DataFrame:
    ranker = next((x for x in current_scores if (not x.empty and x["model"].iloc[0] == "M6_xgboost_rank_ndcg")), None)
    top20 = next((x for x in current_scores if (not x.empty and x["model"].iloc[0] == "M6_top20_calibrated")), None)
    if ranker is None or top20 is None:
        return pd.DataFrame()
    cols = ["signal_date", "candidate_pool_version", "symbol"]
    merged = ranker[cols + ["score"]].rename(columns={"score": "_ranker_score"}).merge(
        top20[cols + ["score"]].rename(columns={"score": "_top20_score"}),
        on=cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    base = ranker.drop(columns=["score", "rank", "model", "model_type"], errors="ignore").merge(
        merged[cols + ["_ranker_score", "_top20_score"]],
        on=cols,
        how="inner",
    )
    out = base.copy()
    out["model"] = "M6_ranker_top20_ensemble"
    out["model_type"] = "ranker_classifier_ensemble"
    out["score"] = 0.60 * out["_ranker_score"] + 0.40 * out["_top20_score"]
    out = out.drop(columns=["_ranker_score", "_top20_score"], errors="ignore")
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first",
        ascending=False,
    )
    return out


def build_walk_forward_ltr_scores(
    dataset: pd.DataFrame,
    spec: FeatureSpec,
    cfg: M6RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_pool_frame(dataset)

    # P0-1: 行业内 z-score 中性化（生成 _ind_z 列）
    if getattr(cfg, 'use_industry_neutral_zscore', False) and "industry_level1" in valid.columns:
        fundamental_z_cols = [
            f"{c}_z" for c in FUNDAMENTAL_RAW_FEATURES
            if f"{c}_z" in valid.columns
        ]
        if fundamental_z_cols:
            valid = industry_neutral_zscore(valid, fundamental_z_cols)

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
            print(
                "[monthly-m6] "
                f"pool={pool} test_month={pd.Timestamp(test_month).date()} "
                f"train_months={train['signal_date'].nunique()} train_rows={len(train)} test_rows={len(test)}",
                flush=True,
            )
            train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
            current_scores: list[pd.DataFrame] = []
            if "xgboost_rank_ndcg" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_ndcg",
                    objective="rank:ndcg",
                    train=train_fit,
                    test=test,
                    feature_cols=feature_cols,
                    random_seed=cfg.random_seed,
                    relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "xgboost_rank_pairwise" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_pairwise",
                    objective="rank:pairwise",
                    train=train_fit,
                    test=test,
                    feature_cols=feature_cols,
                    random_seed=cfg.random_seed,
                    relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "top20_calibrated" in requested or "ranker_top20_ensemble" in requested:
                scores, imp = _train_predict_top20_calibrated(
                    train=train_fit,
                    test=test,
                    feature_cols=feature_cols,
                    random_seed=cfg.random_seed,
                    model_n_jobs=cfg.model_n_jobs,
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


def _tag_importance(imp: pd.DataFrame, spec: FeatureSpec, pool: str, test_month: Any) -> pd.DataFrame:
    out = imp.copy()
    out["feature_spec"] = spec.name
    out["feature_families"] = ",".join(spec.families)
    out["candidate_pool_version"] = pool
    out["test_signal_date"] = pd.Timestamp(test_month)
    return out


def summarize_ltr_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    out = (
        importance.groupby(["feature_spec", "feature_families", "candidate_pool_version", "model", "feature"], sort=True)
        .agg(
            importance=("importance", "mean"),
            signed_weight=("signed_weight", "mean"),
            observations=("test_signal_date", "nunique"),
        )
        .reset_index()
    )
    return out.sort_values(["candidate_pool_version", "model", "importance"], ascending=[True, True, False])


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    spec: FeatureSpec,
    cfg: M6RunConfig,
    dataset_path: Path,
    db_path: Path,
    output_stem: str,
    config_source: str,
    research_config_id: str,
) -> dict[str, Any]:
    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m6_ltr",
        "research_topic": "monthly_selection_m6_ltr",
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "duckdb_path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": list(cfg.candidate_pools),
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in cfg.candidate_pools},
        "top_ks": list(cfg.top_ks),
        "bucket_count": int(cfg.bucket_count),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_spec": {"name": spec.name, "families": list(spec.families), "feature_count": len(spec.feature_cols)},
        "label_spec": "monthly query relevance from forward_1m_excess_vs_market; top20 classifier uses future_top_20pct",
        "pit_policy": "features are signal-date-or-earlier; fundamental uses announcement_date <= signal_date; shareholder is off by default; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month",
        "hyperparameter_policy": "fixed conservative defaults; no random CV and no future-month tuning",
        "ltr_models": list(cfg.ltr_models),
        "relevance_grades": int(cfg.relevance_grades),
        "max_fit_rows": int(cfg.max_fit_rows),
        "model_n_jobs": int(normalize_model_n_jobs(cfg.model_n_jobs)),
        "random_seed": int(cfg.random_seed),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }


def build_doc(
    *,
    quality: dict[str, Any],
    leaderboard: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    year_slice: pd.DataFrame,
    regime_slice: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    leader_view = leaderboard.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, True, False, False],
    )
    cov_view = feature_coverage.head(80).copy()
    year_view = year_slice.sort_values(["candidate_pool_version", "model", "top_k", "year"]).head(40)
    regime_view = regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(40)
    best_u1_top20 = pd.DataFrame()
    if not leaderboard.empty:
        best_u1_top20 = leaderboard[
            (leaderboard["candidate_pool_version"] == "U1_liquid_tradable") & (leaderboard["top_k"] == 20)
        ].head(5)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M6 Learning-to-Rank

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m6_ltr`
- 研究主题：`{quality.get('research_topic', '')}`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 数据库：`{quality.get('duckdb_path', '')}`
- 训练/评估：按 signal_date walk-forward；每个测试月只用历史月份训练。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`
- 单窗训练行上限：`{quality.get('max_fit_rows', 0)}`（`0` 表示不抽样）

## Leaderboard

{_format_markdown_table(leader_view, max_rows=40)}

## U1 Top20 Leading Models

{_format_markdown_table(best_u1_top20, max_rows=5)}

## Feature Coverage

{_format_markdown_table(cov_view, max_rows=80)}

## Year Slice

{_format_markdown_table(year_view, max_rows=40)}

## Realized Market State Slice

{_format_markdown_table(regime_view, max_rows=40)}

## 口径

- M6 默认输入沿用 M5 收敛方向：`price_volume + industry_breadth + fund_flow + fundamental`，暂不把 shareholder 作为主输入。
- `M6_xgboost_rank_ndcg` 与 `M6_xgboost_rank_pairwise` 使用每个 signal_date 作为 query group，标签为同月未来 market-relative excess 的分级 relevance。
- `M6_top20_calibrated` 使用 future top20 bucket 分类概率，并在每个测试月内转换为截面分位 score。
- `M6_ranker_top20_ensemble` 固定使用 `0.60 * rank_ndcg_percentile + 0.40 * top20_percentile`，不使用未来月调权。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- 本脚本只生成研究候选与诊断产物，不写入 promoted registry，不生成交易指令。

## 本轮结论

- 本轮新增：M6 learning-to-rank runner，覆盖 XGBoost Lambda/NDCG 排序、pairwise 排序、top-bucket rank calibration 与固定 ensemble。
- Gate 仍看 Rank IC、Top-K after-cost 超额、Top-K vs next-K、分桶 spread、年度/状态稳定性和行业暴露；oracle overlap 不作为主评价。
- 若 strong-up 或关键年份切片仍不稳，下一步优先做 regime-aware calibration，而不是把模型直接提升为推荐候选。

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
    docs_dir = ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    enabled_families = _parse_str_list(args.families)
    ltr_models = tuple(_parse_str_list(args.ltr_models))
    cfg = M6RunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        availability_lag_days=int(args.availability_lag_days),
        relevance_grades=int(args.relevance_grades),
        model_n_jobs=int(args.model_n_jobs),
        ltr_models=ltr_models,
        use_industry_neutral_zscore=bool(args.use_industry_neutral_zscore),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_models_{'-'.join(slugify_token(x) for x in ltr_models)}"
        f"_grades_{int(args.relevance_grades)}"
        f"_maxfit_{int(args.max_fit_rows)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m6_ltr",
        research_topic="monthly_selection_m6_ltr",
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    research_config_id = identity.research_config_id
    output_stem = identity.output_stem

    print(f"[monthly-m6] research_config_id={research_config_id}")
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
        availability_lag_days=cfg.availability_lag_days,
        model_n_jobs=cfg.model_n_jobs,
        use_industry_neutral_zscore=cfg.use_industry_neutral_zscore,
    )
    dataset = attach_enabled_families(dataset, db_path, m5_cfg, enabled_families)
    spec = build_m6_feature_spec(
        enabled_families,
        use_industry_neutral_zscore=cfg.use_industry_neutral_zscore,
    )
    feature_coverage = summarize_feature_coverage_by_spec(dataset, [spec])
    scores, raw_importance = build_walk_forward_ltr_scores(dataset, spec, cfg)
    if scores.empty:
        warnings.warn("M6 未生成任何 score；请检查训练窗、候选池或特征覆盖。", RuntimeWarning)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_ltr_feature_importance(raw_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)
    quality = build_quality_payload(
        dataset=dataset,
        scores=scores,
        spec=spec,
        cfg=cfg,
        dataset_path=dataset_path,
        db_path=db_path,
        output_stem=output_stem,
        config_source=config_source,
        research_config_id=research_config_id,
    )

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }

    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = {
        "quality": quality,
        "top_models_by_topk": leaderboard.sort_values(
            ["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"],
            ascending=[True, False, False],
        )
        .groupby("top_k", as_index=False)
        .head(5)
        .to_dict(orient="records")
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
            feature_coverage=feature_coverage,
            year_slice=year_slice,
            regime_slice=regime_slice,
            artifacts=[*artifact_paths, _project_relative(paths_out["manifest"])],
        ),
        encoding="utf-8",
    )

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
    rank_ic_observations = int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    best_after_cost_float = float(best_after_cost) if pd.notna(best_after_cost) else None
    # H2: 换仓频率——CLI 显式指定优先，否则从 dataset 自动检测
    if args.rebalance_rule:
        rebalance_rule = args.rebalance_rule
    elif not dataset.empty and "rebalance_rule" in dataset.columns:
        rebalance_rule = str(dataset["rebalance_rule"].iloc[0]).strip().upper() or "M"
    else:
        rebalance_rule = "M"
    data_slice = DataSlice(
        dataset_name="monthly_selection_m6_ltr",
        source_tables=(_project_relative(dataset_path), _project_relative(db_path)),
        date_start=min_signal_date,
        date_end=max_signal_date,
        asof_trade_date=max_signal_date or None,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(pools),
        rebalance_rule=rebalance_rule,
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=spec.name,
        feature_columns=tuple(spec.feature_cols),
        label_columns=(LABEL_COL, EXCESS_COL, TOP20_COL),
        pit_policy=quality["pit_policy"],
        config_path=config_source,
        extra={
            "dataset_path": _project_relative(dataset_path),
            "duckdb_path": _project_relative(db_path),
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "enabled_families": enabled_families,
            "feature_spec": quality["feature_spec"],
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
            "availability_lag_days": int(args.availability_lag_days),
            "relevance_grades": int(args.relevance_grades),
            "cv_policy": quality["cv_policy"],
        },
    )
    artifact_refs = (
        ArtifactRef("summary_json", _project_relative(paths_out["summary_json"]), "json"),
        ArtifactRef("leaderboard_csv", _project_relative(paths_out["leaderboard"]), "csv"),
        ArtifactRef("monthly_long_csv", _project_relative(paths_out["monthly_long"]), "csv"),
        ArtifactRef("rank_ic_csv", _project_relative(paths_out["rank_ic"]), "csv"),
        ArtifactRef("quantile_spread_csv", _project_relative(paths_out["quantile_spread"]), "csv"),
        ArtifactRef("feature_coverage_csv", _project_relative(paths_out["feature_coverage"]), "csv"),
        ArtifactRef("feature_importance_csv", _project_relative(paths_out["feature_importance"]), "csv"),
        ArtifactRef("topk_holdings_csv", _project_relative(paths_out["topk_holdings"]), "csv"),
        ArtifactRef("industry_exposure_csv", _project_relative(paths_out["industry_exposure"]), "csv"),
        ArtifactRef("candidate_pool_width_csv", _project_relative(paths_out["candidate_pool_width"]), "csv"),
        ArtifactRef(
            "candidate_pool_reject_reason_csv",
            _project_relative(paths_out["candidate_pool_reject_reason"]),
            "csv",
        ),
        ArtifactRef("year_slice_csv", _project_relative(paths_out["year_slice"]), "csv"),
        ArtifactRef("regime_slice_csv", _project_relative(paths_out["regime_slice"]), "csv"),
        ArtifactRef("market_states_csv", _project_relative(paths_out["market_states"]), "csv"),
        ArtifactRef("report_md", _project_relative(paths_out["doc"]), "md"),
        ArtifactRef("manifest_json", _project_relative(paths_out["manifest"]), "json"),
    )
    metrics = {
        "rows": int(quality["rows"]),
        "valid_rows": int(quality["valid_rows"]),
        "valid_signal_months": int(quality["valid_signal_months"]),
        "score_rows": int(len(scores)),
        "rank_ic_observations": rank_ic_observations,
        "monthly_long_rows": int(len(monthly_long)),
        "topk_holdings_rows": int(len(topk_holdings)),
        "feature_coverage_rows": int(len(feature_coverage)),
        "model_count": int(len(quality["models"])),
        "best_model": str(best_row.get("model") or ""),
        "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
        "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
        "best_topk_excess_after_cost_mean": best_after_cost_float,
        "best_rank_ic_mean": float(best_row["rank_ic_mean"])
        if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean"))
        else None,
    }
    gates = {
        "data_gate": {
            "passed": bool(metrics["valid_rows"] > 0 and metrics["valid_signal_months"] > 0),
            "checks": {
                "has_valid_rows": metrics["valid_rows"] > 0,
                "has_valid_signal_months": metrics["valid_signal_months"] > 0,
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
        "baseline_gate": {
            "passed": bool(best_after_cost_float is not None and best_after_cost_float > 0.0),
            "best_topk_excess_after_cost_mean": best_after_cost_float,
        },
        "year_gate": {
            "passed": bool(not year_slice.empty),
            "year_slice_rows": int(len(year_slice)),
        },
        "regime_gate": {
            "passed": bool(not regime_slice.empty),
            "regime_slice_rows": int(len(regime_slice)),
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
                "relevance_grades": cfg.relevance_grades,
                "ltr_models": list(cfg.ltr_models),
                "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
                "use_industry_neutral_zscore": cfg.use_industry_neutral_zscore,
            },
            "feature_file_hash": file_sha256(dataset_path),
            "runtime_config_hash": stable_hash({
                "dataset": str(dataset_path),
                "max_fit_rows": cfg.max_fit_rows,
                "min_train_months": cfg.min_train_months,
                "candidate_pools": list(cfg.candidate_pools),
                "ltr_models": list(cfg.ltr_models),
                "cost_bps": cfg.cost_bps,
                "random_seed": cfg.random_seed,
                "availability_lag_days": cfg.availability_lag_days,
                "use_industry_neutral_zscore": cfg.use_industry_neutral_zscore,
            }),
            "overrides": {
                key: value
                for key, value in {
                    "dataset": args.dataset,
                    "duckdb_path": args.duckdb_path.strip(),
                    "results_dir": args.results_dir.strip(),
                    "top_k": args.top_k,
                    "candidate_pools": args.candidate_pools,
                    "families": args.families,
                    "ltr_models": args.ltr_models,
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
            "blocking_reasons": ["m6_ltr_research_only_not_promotion_candidate"],
        },
        notes="Monthly selection M6 LTR contract; ranking outputs are unchanged.",
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

    print(f"[monthly-m6] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-m6] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-m6] manifest={paths_out['manifest']}")
    print(f"[monthly-m6] research_index={experiments_dir / 'research_results.jsonl'}")
    print(f"[monthly-m6] doc={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
