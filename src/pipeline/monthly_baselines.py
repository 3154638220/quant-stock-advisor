"""Shared monthly-selection baseline helpers.

This module holds training, scoring, evaluation and leaderboard helpers used
by the M4/M5/M6 pipeline modules.  Keeping them here avoids importing CLI scripts
from core pipeline code.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.gates import EXCESS_COL, INDUSTRY_EXCESS_COL, LABEL_COL, MARKET_COL, TOP20_COL

# ═══════════════════════════════════════════════════════════════════════════
# Feature specs & blend definitions
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_SPECS: tuple[tuple[str, str, int], ...] = (
    ("B2_momentum_20d", "feature_ret_20d_z", 1),
    ("B2_momentum_60d", "feature_ret_60d_z", 1),
    ("B2_short_momentum_5d", "feature_ret_5d_z", 1),
    ("B3_low_vol_20d", "feature_realized_vol_20d_z", -1),
    ("B3_liquidity_amount_20d", "feature_amount_20d_log_z", 1),
    ("B4_turnover_20d", "feature_turnover_20d_z", 1),
    ("B4_price_position_250d", "feature_price_position_250d_z", 1),
    ("B3_low_limit_move_hits_20d", "feature_limit_move_hits_20d_z", -1),
)

ML_FEATURE_COLS: tuple[str, ...] = tuple(col for _, col, _ in FEATURE_SPECS)

BLEND_SPECS: dict[str, dict[str, float]] = {
    "B1_current_s2_proxy_vol_to_turnover": {
        "feature_realized_vol_20d_z": 1.0,
        "feature_turnover_20d_z": -1.0,
    },
    "B2_momentum_blend": {
        "feature_ret_20d_z": 0.45,
        "feature_ret_60d_z": 0.45,
        "feature_ret_5d_z": 0.10,
    },
    "B3_low_vol_quality_proxy": {
        "feature_realized_vol_20d_z": -0.55,
        "feature_limit_move_hits_20d_z": -0.25,
        "feature_amount_20d_log_z": 0.20,
    },
    "B4_price_volume_equal_blend": {
        "feature_ret_20d_z": 1.0,
        "feature_ret_60d_z": 1.0,
        "feature_realized_vol_20d_z": -1.0,
        "feature_amount_20d_log_z": 1.0,
        "feature_turnover_20d_z": 1.0,
        "feature_price_position_250d_z": 1.0,
        "feature_limit_move_hits_20d_z": -1.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BaselineRunConfig:
    top_ks: tuple[int, ...] = (20, 30, 50)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    cost_bps: float = 10.0
    random_seed: int = 42
    include_xgboost: bool = True
    model_n_jobs: int = 0


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


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation & leaderboard helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cross_section_fill(part: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """截面中位数填充缺失值。"""
    out = part[cols].apply(pd.to_numeric, errors="coerce")
    med = out.median(axis=0, skipna=True).fillna(0.0)
    return out.fillna(med).replace([np.inf, -np.inf], 0.0)


def _safe_qcut(values: pd.Series, bucket_count: int) -> pd.Series:
    """安全分桶，样本不足时返回 NA。"""
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < bucket_count or x.nunique(dropna=True) < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    try:
        b = pd.qcut(x, bucket_count, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    return (b + 1).astype("Int64")


def _weighted_turnover(prev: set[str] | None, cur: set[str], k: int) -> float:
    """半换手率（weighted turnover），范围 [0, 1]。"""
    if prev is None:
        return np.nan
    if k <= 0:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


def build_static_scores(dataset: pd.DataFrame) -> pd.DataFrame:
    """计算单因子和线性混合的截面打分。"""
    valid = valid_pool_frame(dataset)
    rows: list[pd.DataFrame] = []
    if valid.empty:
        return pd.DataFrame()

    for (pool, signal_date), part in valid.groupby(["candidate_pool_version", "signal_date"], sort=True):
        base_cols = _score_base_columns(part)
        for model_name, col, direction in FEATURE_SPECS:
            if col not in part.columns:
                continue
            score = pd.to_numeric(part[col], errors="coerce") * int(direction)
            rows.append(_make_score_frame(base_cols, model_name, "single_factor", _rank_pct_score(score)))
        for model_name, weights in BLEND_SPECS.items():
            usable = [c for c in weights if c in part.columns]
            if not usable:
                continue
            x = _cross_section_fill(part, usable)
            score = sum(x[c] * float(weights[c]) for c in usable)
            rows.append(_make_score_frame(base_cols, model_name, "linear_blend", _rank_pct_score(score)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_walk_forward_scores(dataset: pd.DataFrame, cfg: BaselineRunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward ML 训练与打分。"""
    valid = valid_pool_frame(dataset)
    feature_cols = [c for c in ML_FEATURE_COLS if c in valid.columns]
    if valid.empty or not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_specs = [
        ("M4_elasticnet_excess", "elasticnet"),
        ("M4_logistic_top20", "logistic_classifier"),
        ("M4_extratrees_excess", "tree_sanity"),
    ]
    if cfg.include_xgboost:
        model_specs.extend(
            [
                ("M4_xgboost_excess", "xgboost_regression"),
                ("M4_xgboost_top20", "xgboost_classifier"),
            ]
        )

    for pool, pool_df in valid.groupby("candidate_pool_version", sort=True):
        months = sorted(pd.to_datetime(pool_df["signal_date"]).dropna().unique())
        for test_month in months:
            train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
            for model_name, model_type in model_specs:
                if model_name.startswith("M4_xgboost"):
                    scores, imp = _train_predict_xgboost(
                        model_name=model_name,
                        model_type=model_type,
                        train=train,
                        test=test,
                        feature_cols=feature_cols,
                        random_seed=cfg.random_seed,
                        model_n_jobs=cfg.model_n_jobs,
                    )
                else:
                    scores, imp = _train_predict_sklearn(
                        model_name=model_name,
                        model_type=model_type,
                        train=train,
                        test=test,
                        feature_cols=feature_cols,
                        random_seed=cfg.random_seed,
                        model_n_jobs=cfg.model_n_jobs,
                    )
                if scores is not None and not scores.empty:
                    score_frames.append(scores)
                if not imp.empty:
                    imp = imp.copy()
                    imp["candidate_pool_version"] = pool
                    imp["test_signal_date"] = pd.Timestamp(test_month)
                    importance_frames.append(imp)
    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def build_rank_ic(scores: pd.DataFrame) -> pd.DataFrame:
    """按月计算 Rank IC (Spearman)。"""
    rows: list[dict[str, Any]] = []
    if scores.empty:
        return pd.DataFrame()
    for (pool, model, signal_date), part in scores.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        x = pd.to_numeric(part["score"], errors="coerce")
        y = pd.to_numeric(part[LABEL_COL], errors="coerce")
        ok = x.notna() & y.notna()
        rank_ic = float(x.loc[ok].corr(y.loc[ok], method="spearman")) if ok.sum() >= 3 else np.nan
        rows.append(
            {
                "signal_date": signal_date,
                "candidate_pool_version": pool,
                "model": model,
                "rank_ic": rank_ic,
                "n": int(ok.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_market_benchmark_monthly(scores: pd.DataFrame, *, top_ks: list[int], cost_bps: float) -> pd.DataFrame:
    """构建市场等权基准月度序列。"""
    base = scores.drop_duplicates(["signal_date", "candidate_pool_version"])[
        ["signal_date", "candidate_pool_version", MARKET_COL]
    ].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row_data in base.iterrows():
        market_ret = float(row_data[MARKET_COL]) if pd.notna(row_data[MARKET_COL]) else np.nan
        for k in top_ks:
            rows.append(
                {
                    "signal_date": row_data["signal_date"],
                    "candidate_pool_version": row_data["candidate_pool_version"],
                    "model": "B0_market_ew",
                    "model_type": "benchmark",
                    "top_k": int(k),
                    "candidate_pool_width": np.nan,
                    "selected_count": 0,
                    "topk_return": market_ret,
                    "market_ew_return": market_ret,
                    "topk_excess_vs_market": 0.0 if np.isfinite(market_ret) else np.nan,
                    "topk_industry_neutral_excess": np.nan,
                    "candidate_pool_mean_return": np.nan,
                    "topk_minus_pool_mean": np.nan,
                    "nextk_return": np.nan,
                    "topk_minus_nextk": np.nan,
                    "turnover_half_l1": np.nan,
                    "cost_bps": float(cost_bps),
                    "cost_drag": 0.0,
                    "topk_excess_after_cost": 0.0 if np.isfinite(market_ret) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_monthly_long(scores: pd.DataFrame, *, top_ks: list[int], cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从截面打分构建月度多头组合收益序列及持仓明细。"""
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    prev_by_key: dict[tuple[str, str, int], set[str]] = {}
    ordered = scores.sort_values(
        ["candidate_pool_version", "model", "signal_date", "score", "symbol"],
        ascending=[True, True, True, False, True],
    )
    for (pool, model, signal_date), part in ordered.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        part = part.sort_values(["score", "symbol"], ascending=[False, True]).copy()
        candidate_width = int(part["symbol"].nunique())
        market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
        pool_ret = float(pd.to_numeric(part[LABEL_COL], errors="coerce").mean())
        for k in top_ks:
            top = part.head(k).copy()
            nextk = part.iloc[k : 2 * k].copy()
            if top.empty:
                continue
            cur_symbols = set(top["symbol"].astype(str))
            turnover = _weighted_turnover(prev_by_key.get((pool, model, k)), cur_symbols, k)
            prev_by_key[(pool, model, k)] = cur_symbols
            top_ret = float(pd.to_numeric(top[LABEL_COL], errors="coerce").mean())
            top_ind_excess = (
                float(pd.to_numeric(top[INDUSTRY_EXCESS_COL], errors="coerce").mean())
                if INDUSTRY_EXCESS_COL in top.columns
                else np.nan
            )
            next_ret = float(pd.to_numeric(nextk[LABEL_COL], errors="coerce").mean()) if not nextk.empty else np.nan
            top["top_k"] = int(k)
            top["selected_rank"] = np.arange(1, len(top) + 1)
            holdings.append(
                top[
                    [
                        "signal_date",
                        "candidate_pool_version",
                        "model",
                        "model_type",
                        "top_k",
                        "selected_rank",
                        "symbol",
                        "score",
                        LABEL_COL,
                        "industry_level1",
                        "risk_flags",
                    ]
                ].copy()
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "model": model,
                    "model_type": str(top["model_type"].iloc[0]),
                    "top_k": int(k),
                    "candidate_pool_width": candidate_width,
                    "selected_count": int(len(top)),
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
    monthly = pd.DataFrame(rows)
    h = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    b0 = build_market_benchmark_monthly(scores, top_ks=top_ks, cost_bps=cost_bps)
    if not b0.empty:
        monthly = pd.concat([monthly, b0], ignore_index=True)
    return monthly, h


def build_quantile_spread(scores: pd.DataFrame, *, bucket_count: int) -> pd.DataFrame:
    """构建分桶收益 spread 分析。"""
    if scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (pool, model, signal_date), part in scores.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        tmp = part.copy()
        tmp["score_bucket"] = _safe_qcut(tmp["score"], bucket_count)
        tmp = tmp[tmp["score_bucket"].notna()].copy()
        if tmp.empty:
            continue
        bucket_ret = (
            tmp.groupby("score_bucket", sort=True)
            .agg(
                n=("symbol", "nunique"),
                mean_forward_return=(LABEL_COL, "mean"),
                mean_excess_vs_market=(EXCESS_COL, "mean"),
            )
            .reset_index()
        )
        top_val = bucket_ret.loc[
            bucket_ret["score_bucket"] == bucket_ret["score_bucket"].max(), "mean_forward_return"
        ].mean()
        bottom_val = bucket_ret.loc[
            bucket_ret["score_bucket"] == bucket_ret["score_bucket"].min(), "mean_forward_return"
        ].mean()
        for row_data in bucket_ret.itertuples(index=False):
            rows.append(
                {
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "model": model,
                    "bucket": int(row_data.score_bucket),
                    "bucket_count": int(bucket_count),
                    "n": int(row_data.n),
                    "mean_forward_return": float(row_data.mean_forward_return),
                    "mean_excess_vs_market": float(row_data.mean_excess_vs_market),
                    "top_minus_bottom_return": float(top_val - bottom_val)
                    if np.isfinite(top_val) and np.isfinite(bottom_val)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_leaderboard(
    monthly: pd.DataFrame,
    rank_ic: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    regime_slice: pd.DataFrame,
) -> pd.DataFrame:
    """聚合月度序列、Rank IC、分桶 spread 和状态切片为综合 leaderboard。"""
    if monthly.empty:
        return pd.DataFrame()
    agg = (
        monthly.groupby(["candidate_pool_version", "model", "model_type", "top_k"], sort=True)
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

    return agg.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Summary helpers
# ═══════════════════════════════════════════════════════════════════════════

def summarize_industry_exposure(holdings: pd.DataFrame) -> pd.DataFrame:
    """汇总行业暴露分布。"""
    if holdings.empty:
        return pd.DataFrame()
    h = holdings.copy()
    h["industry_level1"] = h["industry_level1"].fillna("_UNKNOWN_").astype(str)
    total = (
        h.groupby(["signal_date", "candidate_pool_version", "model", "top_k"], sort=True)["symbol"]
        .nunique()
        .rename("topk_count")
        .reset_index()
    )
    out = (
        h.groupby(["signal_date", "candidate_pool_version", "model", "top_k", "industry_level1"], sort=True)[
            "symbol"
        ]
        .nunique()
        .rename("industry_count")
        .reset_index()
        .merge(total, on=["signal_date", "candidate_pool_version", "model", "top_k"], how="left")
    )
    out["industry_share"] = out["industry_count"] / out["topk_count"].replace(0, np.nan)
    return out


def summarize_year_slice(monthly: pd.DataFrame) -> pd.DataFrame:
    """按年度汇总超额收益。"""
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.copy()
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    return (
        df.groupby(["candidate_pool_version", "model", "top_k", "year"], sort=True)
        .agg(
            months=("signal_date", "nunique"),
            mean_topk_return=("topk_return", "mean"),
            mean_topk_excess=("topk_excess_vs_market", "mean"),
            median_topk_excess=("topk_excess_vs_market", "median"),
            hit_rate=("topk_excess_vs_market", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_topk_minus_nextk=("topk_minus_nextk", "mean"),
        )
        .reset_index()
    )


def summarize_regime_slice(monthly: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """按市场状态汇总超额收益。"""
    if monthly.empty or states.empty:
        return pd.DataFrame()
    df = monthly.merge(states[["signal_date", "realized_market_state"]], on="signal_date", how="left")
    return (
        df.groupby(["candidate_pool_version", "model", "top_k", "realized_market_state"], dropna=False, sort=True)
        .agg(
            months=("signal_date", "nunique"),
            mean_topk_return=("topk_return", "mean"),
            mean_topk_excess=("topk_excess_vs_market", "mean"),
            median_topk_excess=("topk_excess_vs_market", "median"),
            hit_rate=("topk_excess_vs_market", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_topk_minus_nextk=("topk_minus_nextk", "mean"),
        )
        .reset_index()
    )


def summarize_candidate_pool_width(dataset: pd.DataFrame) -> pd.DataFrame:
    """汇总候选池宽度。"""
    if dataset.empty:
        return pd.DataFrame()
    out = (
        dataset.groupby(["signal_date", "candidate_pool_version"], dropna=False)
        .agg(
            raw_universe_width=("symbol", "nunique"),
            candidate_pool_width=("candidate_pool_pass", "sum"),
            label_valid_count=(LABEL_COL, lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
    )
    out["candidate_pool_pass_ratio"] = out["candidate_pool_width"] / out["raw_universe_width"].replace(0, np.nan)
    return out


def summarize_candidate_pool_reject_reason(dataset: pd.DataFrame) -> pd.DataFrame:
    """汇总候选池拒绝原因。"""
    if "candidate_pool_reject_reason" not in dataset.columns or dataset.empty:
        return pd.DataFrame()
    df = dataset.copy()
    df["candidate_pool_reject_reason"] = df["candidate_pool_reject_reason"].fillna("").astype(str)
    out = (
        df.groupby(["candidate_pool_version", "candidate_pool_reject_reason"], dropna=False, sort=True)
        .agg(rows=("symbol", "size"), symbols=("symbol", "nunique"))
        .reset_index()
    )
    return out.sort_values(["candidate_pool_version", "rows"], ascending=[True, False])


def summarize_feature_importance(static_scores: pd.DataFrame, ml_importance: pd.DataFrame) -> pd.DataFrame:
    """汇总特征重要性（静态权重 + ML 重要性）。"""
    rows: list[dict[str, Any]] = []
    for model, weights in BLEND_SPECS.items():
        for feature, weight in weights.items():
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "importance": abs(float(weight)),
                    "signed_weight": float(weight),
                    "importance_source": "configured_linear_weight",
                }
            )
    for model, col, direction in FEATURE_SPECS:
        rows.append(
            {
                "model": model,
                "feature": col,
                "importance": 1.0,
                "signed_weight": float(direction),
                "importance_source": "single_factor_direction",
            }
        )
    static_imp = pd.DataFrame(rows)
    if not ml_importance.empty:
        ml = (
            ml_importance.groupby(["candidate_pool_version", "model", "feature"], dropna=False, sort=True)
            .agg(
                importance=("importance", "mean"),
                signed_weight=("signed_weight", "mean"),
                observations=("test_signal_date", "nunique"),
            )
            .reset_index()
        )
        ml["importance_source"] = "walk_forward_model"
        static_imp["candidate_pool_version"] = "_ALL_"
        static_imp["observations"] = np.nan
        return pd.concat([static_imp, ml], ignore_index=True, sort=False)
    static_imp["candidate_pool_version"] = "_ALL_"
    static_imp["observations"] = np.nan
    return static_imp


def build_realized_market_states(dataset: pd.DataFrame) -> pd.DataFrame:
    """基于全市场等权收益的 20/80 分位数划分市场状态。"""
    base = dataset[dataset[LABEL_COL].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=["signal_date", "market_ew_return", "realized_market_state"])
    monthly = (
        base.groupby("signal_date", sort=True)[MARKET_COL]
        .first()
        .reset_index()
        .rename(columns={MARKET_COL: "market_ew_return"})
    )
    vals = pd.to_numeric(monthly["market_ew_return"], errors="coerce")
    lo = vals.quantile(0.20)
    hi = vals.quantile(0.80)
    monthly["realized_market_state"] = np.select(
        [vals <= lo, vals >= hi],
        ["strong_down", "strong_up"],
        default="neutral",
    )
    monthly["state_p20"] = float(lo) if np.isfinite(lo) else np.nan
    monthly["state_p80"] = float(hi) if np.isfinite(hi) else np.nan
    return monthly


def load_baseline_dataset(path: Path, *, candidate_pools: list[str] | None = None) -> pd.DataFrame:
    """加载 M2 canonical dataset 并做基础清洗。"""
    df = pd.read_parquet(path)
    required = {"signal_date", "symbol", "candidate_pool_version", "candidate_pool_pass", LABEL_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"M2 dataset 缺少列: {missing}")
    out = df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    for col in [LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL, *ML_FEATURE_COLS]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if candidate_pools:
        out = out[out["candidate_pool_version"].isin(candidate_pools)].copy()
    return out


def build_summary_payload(*, quality: dict[str, Any], leaderboard: pd.DataFrame) -> dict[str, Any]:
    """构建 JSON 摘要（quality + top models）。"""
    best = pd.DataFrame()
    if not leaderboard.empty:
        best = leaderboard.sort_values(
            ["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"],
            ascending=[True, False, False],
        ).groupby("top_k", as_index=False).head(5)
    return {
        "quality": quality,
        "top_models_by_topk": best.to_dict(orient="records"),
    }
