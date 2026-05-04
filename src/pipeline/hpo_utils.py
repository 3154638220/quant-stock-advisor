"""P1-1: XGBoost 超参数时序感知调优（基于 Optuna）。

提供 XGBoost Ranker 和 Regressor 的时序交叉验证超参数搜索。
不同行情周期最优超参数差异显著（2020-2021 强趋势 vs. 2022-2023 震荡），
固定超参数无法适配市场状态变化。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optuna 搜索空间定义
XGBOOST_RANKER_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "max_depth": {"low": 3, "high": 6, "step": 1},
    "learning_rate": {"low": 0.01, "high": 0.10},
    "min_child_weight": {"low": 20, "high": 100, "step": 10},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.6, "high": 1.0},
    "n_estimators": {"low": 60, "high": 200, "step": 20},
    "reg_alpha": {"low": 0.0, "high": 10.0},
    "reg_lambda": {"low": 0.0, "high": 10.0},
}

XGBOOST_CLASSIFIER_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "max_depth": {"low": 3, "high": 6, "step": 1},
    "learning_rate": {"low": 0.01, "high": 0.10},
    "min_child_weight": {"low": 20, "high": 100, "step": 10},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.6, "high": 1.0},
    "n_estimators": {"low": 60, "high": 200, "step": 20},
    "reg_alpha": {"low": 0.0, "high": 10.0},
    "reg_lambda": {"low": 0.0, "high": 10.0},
}

XGBOOST_REGRESSOR_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "max_depth": {"low": 3, "high": 6, "step": 1},
    "learning_rate": {"low": 0.01, "high": 0.10},
    "min_child_weight": {"low": 20, "high": 100, "step": 10},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.6, "high": 1.0},
    "n_estimators": {"low": 60, "high": 200, "step": 20},
    "reg_alpha": {"low": 0.0, "high": 10.0},
    "reg_lambda": {"low": 0.0, "high": 10.0},
}


def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def _suggest_params(
    trial: Any,
    space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """将搜索空间转换为 optuna trial 建议参数。"""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        low = spec["low"]
        high = spec["high"]
        step = spec.get("step")
        if step is not None and isinstance(step, int) and float(step) == step:
            # 整数型参数
            params[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
        elif step is not None:
            params[name] = trial.suggest_float(name, float(low), float(high), step=float(step))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def _time_series_cv_folds(
    train: pd.DataFrame,
    n_folds: int = 3,
    date_col: str = "signal_date",
    gap: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """时序交叉验证：早期数据训练，后期数据验证，逐折滚动。

    使用 sklearn ``TimeSeriesSplit`` 对按月份聚合后的截面进行划分，
    确保验证集始终在训练集之后，避免未来数据泄露。

    Parameters
    ----------
    train : 训练数据，须含 ``date_col`` 列
    n_folds : 交叉验证折数
    date_col : 日期列名
    gap : 训练集与验证集之间的月份间隔数（默认 1，即留 1 个月 gap）
    """
    from sklearn.model_selection import TimeSeriesSplit

    months = sorted(pd.to_datetime(train[date_col]).dropna().unique())
    if len(months) < n_folds + gap + 1:
        # 数据不足，退化为单折（最后一个月验证）
        split_point = max(1, len(months) - 1)
        train_months = months[:split_point]
        val_months = months[split_point:]
        train_part = train[train[date_col].isin(train_months)]
        val_part = train[train[date_col].isin(val_months)]
        return [(train_part, val_part)]

    # 将 months 数组作为输入，由 TimeSeriesSplit 划分
    month_indices = np.arange(len(months))
    tscv = TimeSeriesSplit(n_splits=n_folds, max_train_size=None, gap=gap)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in tscv.split(month_indices):
        train_months = months[train_idx]
        val_months = months[val_idx]
        if len(val_months) == 0:
            continue
        train_part = train[train[date_col].isin(train_months)]
        val_part = train[train[date_col].isin(val_months)]
        folds.append((train_part, val_part))
    return folds


def tune_xgboost_ranker(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_seed: int = 42,
    relevance_grades: int = 5,
    model_n_jobs: int = 1,
) -> dict[str, Any]:
    """使用 Optuna 在训练集上做时序 CV 超参数搜索（XGBoost Ranker）。

    Parameters
    ----------
    train : 训练数据，须含 signal_date 列
    feature_cols : 特征列名
    n_trials : Optuna trial 数
    cv_folds : 时序 CV 折数
    random_seed : 随机种子

    Returns
    -------
    dict : 最优超参数。若 Optuna 不可用或搜索失败，返回默认超参数。
    """
    if not _optuna_available():
        logger.warning("Optuna 不可用，使用默认 XGBoost Ranker 超参数")
        return _default_ranker_params()

    try:
        import optuna
        from xgboost import XGBRanker
    except Exception:
        return _default_ranker_params()

    from src.pipeline.monthly_ltr import build_ltr_relevance

    folds = _time_series_cv_folds(train, n_folds=cv_folds)
    if len(folds) < 2:
        logger.warning("时序 CV 折数不足（%d），使用默认超参数", len(folds))
        return _default_ranker_params()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, XGBOOST_RANKER_SEARCH_SPACE)
        scores: list[float] = []

        for fold_train, fold_val in folds:
            if fold_train.empty or fold_val.empty:
                continue

            fold_train_sorted = fold_train.sort_values(
                ["signal_date", "symbol"], kind="mergesort"
            ).reset_index(drop=True)
            fold_val_sorted = fold_val.sort_values(
                ["signal_date", "symbol"], kind="mergesort"
            ).reset_index(drop=True)

            y = build_ltr_relevance(fold_train_sorted, grades=relevance_grades)
            if y.nunique(dropna=True) < 2:
                continue

            group_train = fold_train_sorted.groupby("signal_date", sort=False).size().to_numpy(dtype=np.uint32)

            model = XGBRanker(
                n_estimators=params.pop("n_estimators", 120),
                max_depth=params.pop("max_depth", 3),
                learning_rate=params.pop("learning_rate", 0.045),
                subsample=params.pop("subsample", 0.85),
                colsample_bytree=params.pop("colsample_bytree", 0.85),
                min_child_weight=params.pop("min_child_weight", 20),
                reg_alpha=params.pop("reg_alpha", 0.0),
                reg_lambda=params.pop("reg_lambda", 1.0),
                objective="rank:ndcg",
                eval_metric="ndcg@20",
                random_state=random_seed,
                n_jobs=1,  # Optuna 内串行
                tree_method="hist",
                verbosity=0,
            )

            X_train = fold_train_sorted[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_val = fold_val_sorted[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            try:
                model.fit(X_train, y, group=group_train, verbose=False)
                pred = model.predict(X_val)
                # 使用验证集上预测分数的 IC（Spearman rank corr）作为目标
                val_labels = pd.to_numeric(fold_val_sorted.get("label_forward_1m_excess_vs_market"), errors="coerce")
                valid_mask = val_labels.notna() & np.isfinite(pred)
                if valid_mask.sum() >= 5:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(pred[valid_mask], val_labels[valid_mask])
                    if np.isfinite(ic):
                        scores.append(float(ic))
            except Exception:
                continue

        if not scores:
            return -1.0
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    logger.info(
        "XGBoost Ranker HPO 完成: best_ic=%.4f, params=%s",
        study.best_value,
        best,
    )
    return best


def tune_xgboost_regressor(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_seed: int = 42,
    model_n_jobs: int = 1,
) -> dict[str, Any]:
    """使用 Optuna 在训练集上做时序 CV 超参数搜索（XGBoost Regressor）。"""
    if not _optuna_available():
        return _default_regressor_params()

    try:
        import optuna
        from xgboost import XGBRegressor
    except Exception:
        return _default_regressor_params()

    folds = _time_series_cv_folds(train, n_folds=cv_folds)
    if len(folds) < 2:
        return _default_regressor_params()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, XGBOOST_REGRESSOR_SEARCH_SPACE)
        scores: list[float] = []

        for fold_train, fold_val in folds:
            if fold_train.empty or fold_val.empty:
                continue

            y_train = pd.to_numeric(fold_train.get("label_forward_1m_excess_vs_market"), errors="coerce")
            y_val = pd.to_numeric(fold_val.get("label_forward_1m_excess_vs_market"), errors="coerce")
            mask = y_train.notna()
            if mask.sum() < 50:
                continue

            model = XGBRegressor(
                n_estimators=params.pop("n_estimators", 80),
                max_depth=params.pop("max_depth", 3),
                learning_rate=params.pop("learning_rate", 0.05),
                subsample=params.pop("subsample", 0.85),
                colsample_bytree=params.pop("colsample_bytree", 0.85),
                min_child_weight=params.pop("min_child_weight", 20),
                reg_alpha=params.pop("reg_alpha", 0.0),
                reg_lambda=params.pop("reg_lambda", 1.0),
                objective="reg:squarederror",
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            )

            X_train = fold_train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_val = fold_val[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            try:
                model.fit(X_train.loc[mask], y_train.loc[mask], verbose=False)
                pred = model.predict(X_val)
                valid_mask = y_val.notna() & np.isfinite(pred)
                if valid_mask.sum() >= 5:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(pred[valid_mask], y_val[valid_mask])
                    if np.isfinite(ic):
                        scores.append(float(ic))
            except Exception:
                continue

        if not scores:
            return -1.0
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    logger.info("XGBoost Regressor HPO 完成: best_ic=%.4f, params=%s", study.best_value, best)
    return best


def _default_ranker_params() -> dict[str, Any]:
    return {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.045,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }


def _default_regressor_params() -> dict[str, Any]:
    return {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
