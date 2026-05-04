"""月度选股 Learning-to-Rank 管线。

从 scripts/run_monthly_selection_ltr.py 提取核心算法逻辑：
- LTR 训练/预测 (XGBoostRanker, Top20 calibrator, ensemble)
- Walk-forward query-group 打分
- 特征重要性汇总
- P2-3: OOF meta-learner Stacking 集成

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
    FUNDAMENTAL_RAW_FEATURES,
    FeatureSpec,
    M5RunConfig,
    _cap_fit_rows,
    attach_enabled_families,
    build_feature_specs,
    industry_neutral_zscore,
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
    availability_lag_days: int = 45  # P0-1: 兜底延迟提升至 45 天（旧值 30 天过于激进）
    relevance_grades: int = 5
    model_n_jobs: int = 0
    # P1-1: XGBoost 超参数时序感知调优
    hpo_enabled: bool = False
    hpo_n_trials: int = 30
    hpo_cv_folds: int = 3
    # P1-2: Walk-forward 窗口配置
    window_type: str = "expanding"  # "rolling" | "expanding"
    halflife_months: float = 36.0   # 扩张窗口样本半衰期（月），0 表示等权
    ltr_models: tuple[str, ...] = (
        "xgboost_rank_ndcg",
        # "xgboost_rank_pairwise",  # DEPRECATED P1-4: IC 不稳定，移入 ablation
        # "top20_calibrated",       # DEPRECATED P1-4: after-cost excess 为负
        # "ranker_top20_ensemble",  # DEPRECATED P1-4: 同上
        # "lightgbm_rank_ndcg",     # H1: LightGBM Ranker (per docs/plan-05-04.md)
        # "ranker_ensemble",        # H1: XGBoost+LightGBM 简单平均集成
        # P2-3: Stacking 集成（需 OOF 收集，默认关闭以保持向后兼容）
        # "stacking_ensemble",
    )
    # P2-3: OOF meta-learner 配置
    stacking_enabled: bool = False            # 是否启用 OOF Stacking 集成
    stacking_meta_learner: str = "logistic"   # "logistic"（LogisticRegression）
    stacking_meta_C: float = 0.1              # 正则化强度（越小越强）
    # P0-1: 对基本面因子使用行业内 z-score 中性化（默认 False，保持向后兼容）
    use_industry_neutral_zscore: bool = False


def build_m6_feature_spec(
    enabled_families: list[str],
    *,
    use_industry_neutral_zscore: bool = False,
) -> FeatureSpec:
    """构建 M6 LTR 特征规格。

    P0-1: 当 use_industry_neutral_zscore=True 时，基本面因子使用 _ind_z
    （行业内 z-score）替代 _z（全截面 z-score），消除行业间分布差异。
    """
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
    hpo_params: dict | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
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

    # P1-1: 优先使用 HPO 超参数，否则使用默认值
    params = hpo_params or {}
    model = XGBRanker(
        n_estimators=params.get("n_estimators", 120),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.045),
        subsample=params.get("subsample", 0.85),
        colsample_bytree=params.get("colsample_bytree", 0.85),
        min_child_weight=params.get("min_child_weight", 1),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective=objective, eval_metric="ndcg@20",
        random_state=random_seed, n_jobs=normalize_model_n_jobs(model_n_jobs),
        tree_method="hist",
    )
    try:
        # XGBRanker 与 group 参数一起使用时，sample_weight 必须是 per-group 而非 per-sample，
        # 因此不在 fit() 中传递 per-sample 权重。半衰期衰减通过 _cap_fit_rows 间接实现。
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


def _train_predict_lightgbm_ranker(
    *,
    model_name: str,
    objective: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    relevance_grades: int,
    model_n_jobs: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """H1: LightGBM Ranker 训练/预测（per docs/plan-05-04.md）。

    作为 XGBoost Ranker 的替代模型，提供模型多样性。
    使用 lambdarank 目标函数，与 XGBoost 的 rank:ndcg 对应。

    Parameters 与 ``_train_predict_xgboost_ranker`` 一致。
    """
    try:
        import lightgbm as lgb
    except Exception as exc:
        warnings.warn(f"跳过 LightGBM ranker，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    train_sorted = _sort_for_query_groups(train)
    test_sorted = _sort_for_query_groups(test)

    y = build_ltr_relevance(train_sorted, grades=relevance_grades)
    if y.nunique(dropna=True) < 2:
        return None, pd.DataFrame()

    group_sizes = _query_group_sizes(train_sorted)
    group_test = _query_group_sizes(test_sorted)

    X_train = _prepare_ranker_matrix(train_sorted, feature_cols)
    X_test = _prepare_ranker_matrix(test_sorted, feature_cols)

    n_jobs = normalize_model_n_jobs(model_n_jobs)

    # LightGBM 默认参数（与 XGBoost 保持类似的保守配置）
    params = {
        "objective": objective,          # "lambdarank"
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "boosting_type": "gbdt",
        "num_leaves": kwargs.get("num_leaves", 31),
        "max_depth": kwargs.get("max_depth", 5),
        "learning_rate": kwargs.get("learning_rate", 0.045),
        "feature_fraction": kwargs.get("feature_fraction", 0.85),
        "bagging_fraction": kwargs.get("bagging_fraction", 0.85),
        "bagging_freq": 1,
        "min_child_samples": kwargs.get("min_child_samples", 20),
        "lambda_l1": kwargs.get("lambda_l1", 0.0),
        "lambda_l2": kwargs.get("lambda_l2", 1.0),
        "seed": random_seed,
        "num_threads": n_jobs if n_jobs > 0 else -1,
        "verbosity": -1,
        "label_gain": list(range(int(relevance_grades))),
    }

    try:
        train_data = lgb.Dataset(
            X_train, label=y, group=group_sizes,
            params={"label_gain": list(range(int(relevance_grades)))},
        )
        model = lgb.train(
            params,
            train_data,
            num_boost_round=kwargs.get("num_boost_round", 120),
            valid_sets=None,
        )
        raw_pred = pd.Series(
            model.predict(X_test, group=group_test),
            index=test_sorted.index,
        )
    except Exception as exc:
        warnings.warn(f"{model_name} 训练失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    base = _score_base_columns(test_sorted)
    out = base.copy()
    out["model"] = model_name
    out["model_type"] = "lightgbm_ranker"
    out["score"] = raw_pred.groupby(base["signal_date"], sort=False).transform(_rank_pct_score)
    out["rank"] = out.groupby(
        ["signal_date", "candidate_pool_version", "model"], sort=False
    )["score"].rank(method="first", ascending=False)

    # LightGBM feature importance (gain-based)
    importance_values = np.array(model.feature_importance(importance_type="gain"))
    importance = pd.DataFrame({
        "model": model_name,
        "feature": feature_cols,
        "importance": importance_values,
        "signed_weight": np.nan,
    })
    return out, importance


def _build_ranker_ensemble(
    score_frames: list[pd.DataFrame],
    *,
    model_name: str = "M6_ranker_ensemble",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """H1: 简单平均集成多个 Ranker 的百分位分数。

    与 ``_build_ranker_top20_ensemble`` 不同，此函数对所有基于排序的模型
    （XGBoost Ranker、LightGBM Ranker 等）做等权或加权平均，不依赖 classifier。

    Parameters
    ----------
    score_frames
        各模型的打分 DataFrame 列表。每个 DataFrame 需有 model, score, rank 列。
    model_name
        集成模型名称。
    weights
        各模型权重，key 为 model 名称。None 表示等权。
    """
    if len(score_frames) < 2:
        return pd.DataFrame()

    cols = ["signal_date", "candidate_pool_version", "symbol"]
    merged = None
    active_models: list[str] = []
    for sf in score_frames:
        if sf is None or sf.empty:
            continue
        mname = sf["model"].iloc[0]
        sub = sf[cols + ["score"]].rename(columns={"score": f"_score_{mname}"})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=cols, how="inner")
        active_models.append(mname)

    if merged is None or merged.empty or len(active_models) < 2:
        return pd.DataFrame()

    # 计算加权平均分数
    w = weights or {m: 1.0 / len(active_models) for m in active_models}
    total_w = sum(w.get(m, 0.0) for m in active_models)
    if total_w <= 0:
        return pd.DataFrame()

    merged["score"] = 0.0
    for m in active_models:
        col = f"_score_{m}"
        if col in merged.columns:
            merged["score"] += merged[col] * (w.get(m, 0.0) / total_w)

    # Drop intermediate columns
    merge_cols = [f"_score_{m}" for m in active_models]
    merged = merged.drop(columns=merge_cols, errors="ignore")

    merged["model"] = model_name
    merged["model_type"] = "ranker_ensemble"
    merged["rank"] = merged.groupby(
        ["signal_date", "candidate_pool_version", "model"], sort=False
    )["score"].rank(method="first", ascending=False)

    return merged


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
    except Exception as exc:
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
                n_estimators=120, max_depth=3, learning_rate=0.045,
                subsample=0.85, colsample_bytree=0.85,
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=max(neg / max(pos, 1.0), 1.0),
                random_state=random_seed, n_jobs=normalize_model_n_jobs(model_n_jobs),
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
        "importance": importance_values, "signed_weight": signed_weights,
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


# ── P2-3: OOF Meta-Learner Stacking 集成 ─────────────────────────────────

def _collect_stacking_oof_frame(
    oof_records: list[dict],
    *,
    label_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将 walk-forward 中收集的 OOF 记录转换为 meta-learner 训练/特征 DataFrame。

    每行包含一个 (signal_date, symbol) 的基模型 OOF 分数和标签。
    """
    if not oof_records:
        return pd.DataFrame()
    oof = pd.DataFrame(oof_records)
    # pivot: 每个模型一列
    oof_pivot = oof.pivot_table(
        index=["signal_date", "symbol", "candidate_pool_version"],
        columns="oof_model",
        values="oof_score",
        aggfunc="first",
    ).reset_index()
    # 展平列名
    oof_pivot.columns = [str(c) for c in oof_pivot.columns]
    oof_pivot = oof_pivot.rename(columns=lambda c: f"oof_{c}" if c not in ("signal_date", "symbol", "candidate_pool_version") else c)

    # 附加标签
    if label_df is not None and not label_df.empty:
        label_cols = ["signal_date", "symbol", "candidate_pool_version", TOP20_COL, LABEL_COL, EXCESS_COL]
        available = [c for c in label_cols if c in label_df.columns]
        if available:
            oof_pivot = oof_pivot.merge(
                label_df[available].drop_duplicates(["signal_date", "symbol", "candidate_pool_version"]),
                on=["signal_date", "symbol", "candidate_pool_version"],
                how="left",
            )
    return oof_pivot


def _train_stacking_meta_learner(
    oof_frame: pd.DataFrame,
    *,
    oof_cols: list[str],
    label_col: str = TOP20_COL,
    meta_learner: str = "logistic",
    meta_C: float = 0.1,
    random_seed: int = 42,
) -> object | None:
    """P2-3: 以 LogisticRegression 为 meta-learner，基模型 OOF 分数为特征，Top20 标签为目标。

    Returns:
        训练好的 meta-learner，或 None（训练失败/数据不足时）。
    """
    if oof_frame.empty or not oof_cols:
        return None
    available = [c for c in oof_cols if c in oof_frame.columns]
    if len(available) < 2:
        return None

    X = oof_frame[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y_raw = pd.to_numeric(oof_frame.get(label_col), errors="coerce")
    if y_raw.isna().all():
        # 回退到 EXCESS_COL 的 top20 分位标签
        excess = pd.to_numeric(oof_frame.get(EXCESS_COL), errors="coerce")
        if excess.notna().any():
            threshold = excess.quantile(0.80)
            y_raw = (excess >= threshold).astype(int)
        else:
            return None
    y = y_raw.fillna(0).clip(0, 1).astype(int).to_numpy(dtype=np.int32)
    if np.unique(y).size < 2 or len(y) < 20:
        return None

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None

    model = LogisticRegression(
        C=float(meta_C),
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=int(random_seed),
        class_weight="balanced",
    )
    try:
        model.fit(X, y)
    except Exception:
        return None
    return model


def _predict_stacking_ensemble(
    test_scores: dict[str, pd.DataFrame],
    meta_learner: object,
    oof_cols: list[str],
    *,
    model_name: str = "M6_stacking_ensemble",
) -> pd.DataFrame:
    """P2-3: 使用训练好的 meta-learner 对测试集基模型分数做 Stacking 预测。"""
    if not test_scores or meta_learner is None:
        return pd.DataFrame()

    # 合并各基模型的测试集分数
    merged = None
    cols_key = ["signal_date", "symbol", "candidate_pool_version"]
    for mname, scores_df in test_scores.items():
        if scores_df is None or scores_df.empty:
            continue
        sub = scores_df[cols_key + ["score"]].rename(columns={"score": f"oof_{mname}"})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=cols_key, how="inner")

    if merged is None or merged.empty:
        return pd.DataFrame()

    available = [c for c in oof_cols if c in merged.columns]
    if len(available) < 2:
        return pd.DataFrame()

    X = merged[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    try:
        proba = meta_learner.predict_proba(X)[:, 1]
    except Exception:
        return pd.DataFrame()

    base = merged[cols_key].copy()
    base["model"] = model_name
    base["model_type"] = "stacking_ensemble"
    base["score"] = proba
    base["score"] = base.groupby("signal_date", sort=False)["score"].transform(_rank_pct_score)
    base["rank"] = base.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False,
    )
    return base


def _append_oof(oof_records: list[dict], scores: pd.DataFrame, model_name: str) -> None:
    """P2-3: 将单折 OOF 预测追加到记录列表。"""
    for _, row in scores.iterrows():
        oof_records.append({
            "signal_date": row.get("signal_date"),
            "symbol": row.get("symbol"),
            "candidate_pool_version": row.get("candidate_pool_version"),
            "oof_model": model_name,
            "oof_score": row.get("score"),
        })


def _build_stacking_from_existing_scores(
    score_frames: list[pd.DataFrame],
    meta_learner: object,
    oof_cols: list[str],
    spec: FeatureSpec,
    cfg: M6RunConfig,
) -> pd.DataFrame:
    """P2-3: 对已有基模型分数应用 meta-learner 生成 Stacking 集成。

    按信号日分组，将同一天的基模型分数输入 meta-learner 得到集成预测。
    """
    if not score_frames:
        return pd.DataFrame()

    all_scores = pd.concat(score_frames, ignore_index=True)
    model_names = all_scores["model"].unique()
    # 只取基模型（非 ensemble 类型）
    base_models = [m for m in model_names if m in ("M6_xgboost_rank_ndcg", "M6_xgboost_rank_pairwise", "M6_lightgbm_rank_ndcg", "M6_top20_calibrated")]
    if len(base_models) < 2:
        return pd.DataFrame()

    # 按 (signal_date, symbol, pool) 合并基模型分数
    out_frames: list[pd.DataFrame] = []
    for (sd, pool), group in all_scores.groupby(["signal_date", "candidate_pool_version"], sort=True):
        test_scores: dict[str, pd.DataFrame] = {}
        for mname in base_models:
            sub = group[group["model"] == mname]
            if not sub.empty:
                test_scores[mname] = sub
        stacking = _predict_stacking_ensemble(test_scores, meta_learner, oof_cols)
        if not stacking.empty:
            stacking["feature_spec"] = spec.name
            stacking["feature_families"] = ",".join(spec.families)
            out_frames.append(stacking)

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


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

    # P2-3: OOF 预测收集
    stacking_enabled = cfg.stacking_enabled or ("stacking_ensemble" in requested)
    oof_records: list[dict] = []  # 每项: {signal_date, symbol, pool, oof_model, oof_score}

    for pool, pool_df in valid.groupby("candidate_pool_version", sort=True):
        months = sorted(pd.to_datetime(pool_df["signal_date"]).dropna().unique())
        for test_month in months:
            # P1-2: 扩张窗口 vs 滚动窗口
            if cfg.window_type == "rolling":
                cutoff = test_month - pd.DateOffset(months=cfg.min_train_months)
                train = pool_df[(pool_df["signal_date"] < test_month) & (pool_df["signal_date"] >= cutoff)].copy()
            else:
                train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
            train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)

            # P1-2: 半衰期样本权重
            sample_weight = None
            if cfg.halflife_months > 0:
                age_days = (test_month - pd.to_datetime(train_fit["signal_date"])).dt.days
                age_months = age_days / 30.0
                sample_weight = np.exp(-np.log(2) * age_months / cfg.halflife_months)
                sample_weight = sample_weight.clip(lower=0.01)

            # P1-1: HPO
            hpo_params: dict | None = None
            if cfg.hpo_enabled and train_fit["signal_date"].nunique() >= 6:
                try:
                    from src.pipeline.hpo_utils import tune_xgboost_ranker
                    hpo_params = tune_xgboost_ranker(
                        train_fit, feature_cols,
                        n_trials=cfg.hpo_n_trials,
                        cv_folds=cfg.hpo_cv_folds,
                        random_seed=cfg.random_seed,
                        relevance_grades=cfg.relevance_grades,
                        model_n_jobs=cfg.model_n_jobs,
                    )
                except Exception:
                    hpo_params = None

            current_scores: list[pd.DataFrame] = []
            if "xgboost_rank_ndcg" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_ndcg", objective="rank:ndcg",
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs, hpo_params=hpo_params,
                    sample_weight=sample_weight,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                    # P2-3: 收集 OOF 预测
                    if stacking_enabled:
                        _append_oof(oof_records, scores, "M6_xgboost_rank_ndcg")
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            if "xgboost_rank_pairwise" in requested:
                scores, imp = _train_predict_xgboost_ranker(
                    model_name="M6_xgboost_rank_pairwise", objective="rank:pairwise",
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs, hpo_params=hpo_params,
                    sample_weight=sample_weight,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                    if stacking_enabled:
                        _append_oof(oof_records, scores, "M6_xgboost_rank_pairwise")
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            # H1: LightGBM Ranker（per docs/plan-05-04.md）
            if "lightgbm_rank_ndcg" in requested:
                scores, imp = _train_predict_lightgbm_ranker(
                    model_name="M6_lightgbm_rank_ndcg", objective="lambdarank",
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, relevance_grades=cfg.relevance_grades,
                    model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                    if stacking_enabled:
                        _append_oof(oof_records, scores, "M6_lightgbm_rank_ndcg")
                if not imp.empty:
                    importance_frames.append(_tag_importance(imp, spec, pool, test_month))
            # H1: XGBoost + LightGBM 简单平均集成
            if "ranker_ensemble" in requested:
                # 从 current_scores 中找出 xgboost_rank_ndcg 和 lightgbm_rank_ndcg
                ensemble = _build_ranker_ensemble(
                    current_scores,
                    model_name="M6_ranker_ensemble",
                )
                if not ensemble.empty:
                    current_scores.append(ensemble)
            if "top20_calibrated" in requested or "ranker_top20_ensemble" in requested or "stacking_ensemble" in requested:
                scores, imp = _train_predict_top20_calibrated(
                    train=train_fit, test=test, feature_cols=feature_cols,
                    random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
                )
                if scores is not None and not scores.empty:
                    current_scores.append(scores)
                    if stacking_enabled:
                        _append_oof(oof_records, scores, "M6_top20_calibrated")
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

    # P2-3: 训练 meta-learner 并生成 Stacking 集成分数
    if stacking_enabled and oof_records:
        oof_frame = _collect_stacking_oof_frame(oof_records, label_df=valid)
        oof_cols = [c for c in oof_frame.columns if c.startswith("oof_")]
        if len(oof_cols) >= 2:
            meta = _train_stacking_meta_learner(
                oof_frame,
                oof_cols=oof_cols,
                meta_learner=cfg.stacking_meta_learner,
                meta_C=cfg.stacking_meta_C,
                random_seed=cfg.random_seed,
            )
            if meta is not None:
                # 使用 meta-learner 为所有已有分数重新生成 stacking ensemble
                stacking_scores = _build_stacking_from_existing_scores(
                    score_frames, meta, oof_cols, spec, cfg
                )
                if not stacking_scores.empty:
                    score_frames.append(stacking_scores)

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
