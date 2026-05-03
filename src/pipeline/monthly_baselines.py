"""Shared monthly-selection baseline helpers.

This module holds the small set of training and scoring helpers that are used
by the M5/M6 pipeline modules.  Keeping them here avoids importing CLI scripts
from core pipeline code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.research.gates import EXCESS_COL, INDUSTRY_EXCESS_COL, LABEL_COL, MARKET_COL, TOP20_COL


def normalize_model_n_jobs(n_jobs: int) -> int:
    """Normalize CLI/config thread count for sklearn and XGBoost estimators."""
    n = int(n_jobs)
    return -1 if n <= 0 else n


def model_n_jobs_token(n_jobs: int) -> str:
    normalized = normalize_model_n_jobs(n_jobs)
    return "all" if normalized == -1 else str(normalized)


def valid_pool_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    m = dataset["candidate_pool_pass"].astype(bool) & dataset[LABEL_COL].notna()
    return dataset.loc[m].copy()


def _rank_pct_score(score: pd.Series) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.rank(method="average", pct=True, na_option="bottom")


def _score_base_columns(part: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    base = pd.DataFrame(index=part.index)
    for col in cols:
        base[col] = part[col] if col in part.columns else np.nan
    return base


def _make_score_frame(base: pd.DataFrame, model_name: str, model_type: str, score: pd.Series) -> pd.DataFrame:
    out = base.copy()
    out["model"] = model_name
    out["model_type"] = model_type
    out["score"] = pd.to_numeric(score, errors="coerce")
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False
    )
    return out


def _ensure_top20_target(train: pd.DataFrame) -> pd.Series:
    if TOP20_COL in train.columns and pd.to_numeric(train[TOP20_COL], errors="coerce").notna().any():
        y = pd.to_numeric(train[TOP20_COL], errors="coerce").fillna(0).astype(int)
        return y.clip(0, 1)
    pct = train.groupby("signal_date")[LABEL_COL].rank(method="first", pct=True, ascending=False)
    return (pct <= 0.20).astype(int)


def _train_predict_sklearn(
    *,
    model_name: str,
    model_type: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    model_n_jobs: int = 1,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = train[feature_cols]
    x_test = test[feature_cols]
    sw = None
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw = np.where(np.isfinite(sw), sw, 1.0)
    y_reg = pd.to_numeric(train[EXCESS_COL], errors="coerce")
    if y_reg.notna().sum() == 0:
        y_reg = pd.to_numeric(train[LABEL_COL], errors="coerce")

    importance = pd.DataFrame()
    try:
        if model_name == "M4_elasticnet_excess":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(alpha=0.002, l1_ratio=0.15, random_state=random_seed, max_iter=5000),
            )
            m = y_reg.notna()
            fit_kwargs = {}
            if sw is not None and len(sw) == len(train):
                fit_kwargs["elasticnet__sample_weight"] = sw[m.to_numpy()]
            model.fit(x_train.loc[m], y_reg.loc[m], **fit_kwargs)
            pred = model.predict(x_test)
            coefs = model.named_steps["elasticnet"].coef_
            importance = pd.DataFrame(
                {"model": model_name, "feature": feature_cols, "importance": np.abs(coefs), "signed_weight": coefs}
            )
        elif model_name == "M4_logistic_top20":
            y_cls = _ensure_top20_target(train)
            if y_cls.nunique(dropna=True) < 2:
                return None, importance
            model = make_pipeline(
                SimpleImputer(strategy="median"),
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
            fit_kwargs = {}
            if sw is not None and len(sw) == len(train):
                fit_kwargs["logisticregression__sample_weight"] = sw
            model.fit(x_train, y_cls, **fit_kwargs)
            pred = model.predict_proba(x_test)[:, 1]
            coefs = model.named_steps["logisticregression"].coef_[0]
            importance = pd.DataFrame(
                {"model": model_name, "feature": feature_cols, "importance": np.abs(coefs), "signed_weight": coefs}
            )
        elif model_name == "M4_extratrees_excess":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=50,
                    random_state=random_seed,
                    n_jobs=normalize_model_n_jobs(model_n_jobs),
                ),
            )
            m = y_reg.notna()
            fit_kwargs = {}
            if sw is not None and len(sw) == len(train):
                fit_kwargs["extratreesregressor__sample_weight"] = sw[m.to_numpy()]
            model.fit(x_train.loc[m], y_reg.loc[m], **fit_kwargs)
            pred = model.predict(x_test)
            imp = model.named_steps["extratreesregressor"].feature_importances_
            importance = pd.DataFrame(
                {"model": model_name, "feature": feature_cols, "importance": imp, "signed_weight": np.nan}
            )
        else:
            raise ValueError(f"unsupported sklearn model: {model_name}")
    except Exception as exc:  # pragma: no cover - defensive for optional model failures
        warnings.warn(f"{model_name} 训练失败: {exc}", RuntimeWarning)
        return None, importance

    base = _score_base_columns(test)
    out = _make_score_frame(base, model_name, model_type, _rank_pct_score(pd.Series(pred, index=test.index)))
    return out, importance


def _train_predict_xgboost(
    *,
    model_name: str,
    model_type: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int,
    model_n_jobs: int = 1,
    hpo_params: dict | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        warnings.warn(f"跳过 XGBoost baseline，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    params = hpo_params or {}
    sw = None
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw = np.where(np.isfinite(sw), sw, 1.0)
    importance = pd.DataFrame()
    try:
        if model_name == "M4_xgboost_excess":
            y = pd.to_numeric(train[EXCESS_COL], errors="coerce")
            if y.notna().sum() == 0:
                y = pd.to_numeric(train[LABEL_COL], errors="coerce")
            m = y.notna()
            model = XGBRegressor(
                n_estimators=params.get("n_estimators", 80),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.85),
                colsample_bytree=params.get("colsample_bytree", 0.85),
                min_child_weight=params.get("min_child_weight", 1),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                objective="reg:squarederror",
                random_state=random_seed,
                n_jobs=normalize_model_n_jobs(model_n_jobs),
                tree_method="hist",
            )
            fit_sw = sw[m.to_numpy()] if sw is not None and len(sw) == len(train) else None
            model.fit(x_train.loc[m], y.loc[m], verbose=False, sample_weight=fit_sw)
            pred = model.predict(x_test)
        elif model_name == "M4_xgboost_top20":
            y = _ensure_top20_target(train)
            if y.nunique(dropna=True) < 2:
                return None, importance
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            model = XGBClassifier(
                n_estimators=params.get("n_estimators", 80),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.85),
                colsample_bytree=params.get("colsample_bytree", 0.85),
                min_child_weight=params.get("min_child_weight", 1),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=max(neg / max(pos, 1.0), 1.0),
                random_state=random_seed,
                n_jobs=normalize_model_n_jobs(model_n_jobs),
                tree_method="hist",
            )
            fit_sw = sw if sw is not None and len(sw) == len(train) else None
            model.fit(x_train, y, verbose=False, sample_weight=fit_sw)
            pred = model.predict_proba(x_test)[:, 1]
        else:
            raise ValueError(f"unsupported xgboost model: {model_name}")
        importance = pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_cols,
                "importance": model.feature_importances_,
                "signed_weight": np.nan,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for optional model failures
        warnings.warn(f"{model_name} 训练失败: {exc}", RuntimeWarning)
        return None, importance

    base = _score_base_columns(test)
    out = _make_score_frame(base, model_name, model_type, _rank_pct_score(pd.Series(pred, index=test.index)))
    return out, importance
