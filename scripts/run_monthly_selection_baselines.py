#!/usr/bin/env python3
"""M4: 月度选股 baseline ranker。

输入 M2 canonical dataset：
``data/cache/monthly_selection_features.parquet``。

输出 price-volume-only 的可解释 baseline、walk-forward ML baseline，以及
Rank IC、Top-K 超额、分桶 spread、年度/市场状态、行业暴露和换手诊断。
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
from src.settings import load_config, resolve_config_path


LABEL_COL = "label_forward_1m_o2o_return"
EXCESS_COL = "label_forward_1m_excess_vs_market"
INDUSTRY_EXCESS_COL = "label_forward_1m_industry_neutral_excess"
MARKET_COL = "label_market_ew_o2o_return"
TOP20_COL = "label_future_top_20pct"

POOL_RULES: dict[str, str] = {
    "U0_all_tradable": "valid current OHLCV + buyable at next trading day's open",
    "U1_liquid_tradable": "U0 + minimum history length + 20d average amount threshold",
    "U2_risk_sane": "U1 + exclude extreme limit-move path, extreme volatility/turnover, and absolute-high names",
}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M4 baseline ranker")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_baselines")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--skip-xgboost", action="store_true", help="跳过 XGBoost baseline，便于快速烟雾测试")
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals))


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_无记录_"
    view = df.head(max_rows).copy()
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    suffix = [f"\n_仅展示前 {max_rows} 行，共 {len(df)} 行。_"] if len(df) > max_rows else []
    return "\n".join([header, sep, *rows, *suffix])


def load_baseline_dataset(path: Path, *, candidate_pools: list[str] | None = None) -> pd.DataFrame:
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


def valid_pool_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    m = dataset["candidate_pool_pass"].astype(bool) & dataset[LABEL_COL].notna()
    return dataset.loc[m].copy()


def build_realized_market_states(dataset: pd.DataFrame) -> pd.DataFrame:
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


def _cross_section_fill(part: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = part[cols].apply(pd.to_numeric, errors="coerce")
    med = out.median(axis=0, skipna=True).fillna(0.0)
    return out.fillna(med).replace([np.inf, -np.inf], 0.0)


def _rank_pct_score(score: pd.Series) -> pd.Series:
    x = pd.to_numeric(score, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.rank(method="average", pct=True, na_option="bottom")


def build_static_scores(dataset: pd.DataFrame) -> pd.DataFrame:
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
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = train[feature_cols]
    x_test = test[feature_cols]
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
            model.fit(x_train.loc[m], y_reg.loc[m])
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
            model.fit(x_train, y_cls)
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
                    n_jobs=1,
                ),
            )
            m = y_reg.notna()
            model.fit(x_train.loc[m], y_reg.loc[m])
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
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        warnings.warn(f"跳过 XGBoost baseline，import 失败: {exc}", RuntimeWarning)
        return None, pd.DataFrame()

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    importance = pd.DataFrame()
    try:
        if model_name == "M4_xgboost_excess":
            y = pd.to_numeric(train[EXCESS_COL], errors="coerce")
            if y.notna().sum() == 0:
                y = pd.to_numeric(train[LABEL_COL], errors="coerce")
            m = y.notna()
            model = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
            )
            model.fit(x_train.loc[m], y.loc[m], verbose=False)
            pred = model.predict(x_test)
        elif model_name == "M4_xgboost_top20":
            y = _ensure_top20_target(train)
            if y.nunique(dropna=True) < 2:
                return None, importance
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=max(neg / max(pos, 1.0), 1.0),
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
            )
            model.fit(x_train, y, verbose=False)
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


def build_walk_forward_scores(dataset: pd.DataFrame, cfg: BaselineRunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    )
                else:
                    scores, imp = _train_predict_sklearn(
                        model_name=model_name,
                        model_type=model_type,
                        train=train,
                        test=test,
                        feature_cols=feature_cols,
                        random_seed=cfg.random_seed,
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


def _weighted_turnover(prev: set[str] | None, cur: set[str], k: int) -> float:
    if prev is None:
        return np.nan
    if k <= 0:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


def build_monthly_long(scores: pd.DataFrame, *, top_ks: list[int], cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    prev_by_key: dict[tuple[str, str, int], set[str]] = {}
    ordered = scores.sort_values(["candidate_pool_version", "model", "signal_date", "score", "symbol"], ascending=[True, True, True, False, True])
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


def build_market_benchmark_monthly(scores: pd.DataFrame, *, top_ks: list[int], cost_bps: float) -> pd.DataFrame:
    base = scores.drop_duplicates(["signal_date", "candidate_pool_version"])[
        ["signal_date", "candidate_pool_version", MARKET_COL]
    ].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        market_ret = float(row[MARKET_COL]) if pd.notna(row[MARKET_COL]) else np.nan
        for k in top_ks:
            rows.append(
                {
                    "signal_date": row["signal_date"],
                    "candidate_pool_version": row["candidate_pool_version"],
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


def _safe_qcut(values: pd.Series, bucket_count: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < bucket_count or x.nunique(dropna=True) < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    try:
        b = pd.qcut(x, bucket_count, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    return (b + 1).astype("Int64")


def build_quantile_spread(scores: pd.DataFrame, *, bucket_count: int) -> pd.DataFrame:
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
        for row in bucket_ret.itertuples(index=False):
            rows.append(
                {
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "model": model,
                    "bucket": int(row.score_bucket),
                    "bucket_count": int(bucket_count),
                    "n": int(row.n),
                    "mean_forward_return": float(row.mean_forward_return),
                    "mean_excess_vs_market": float(row.mean_excess_vs_market),
                    "top_minus_bottom_return": float(top_val - bottom_val)
                    if np.isfinite(top_val) and np.isfinite(bottom_val)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_industry_exposure(holdings: pd.DataFrame) -> pd.DataFrame:
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


def build_leaderboard(
    monthly: pd.DataFrame,
    rank_ic: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    regime_slice: pd.DataFrame,
) -> pd.DataFrame:
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


def build_summary_payload(*, quality: dict[str, Any], leaderboard: pd.DataFrame) -> dict[str, Any]:
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


def build_doc(
    *,
    quality: dict[str, Any],
    leaderboard: pd.DataFrame,
    year_slice: pd.DataFrame,
    regime_slice: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    leader_view = leaderboard.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, True, False, False],
    )
    year_view = year_slice.sort_values(["candidate_pool_version", "model", "top_k", "year"]).head(40)
    regime_view = regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(40)
    industry_view = pd.DataFrame()
    if not industry_exposure.empty:
        industry_view = (
            industry_exposure.groupby(["candidate_pool_version", "model", "top_k", "industry_level1"], sort=True)
            .agg(mean_share=("industry_share", "mean"), months=("signal_date", "nunique"))
            .reset_index()
            .sort_values(["top_k", "candidate_pool_version", "model", "mean_share"], ascending=[True, True, True, False])
            .head(40)
        )
    best_u1_top20 = pd.DataFrame()
    best_u2_top20 = pd.DataFrame()
    if not leaderboard.empty:
        best_u1_top20 = leaderboard[
            (leaderboard["candidate_pool_version"] == "U1_liquid_tradable") & (leaderboard["top_k"] == 20)
        ].head(3)
        best_u2_top20 = leaderboard[
            (leaderboard["candidate_pool_version"] == "U2_risk_sane") & (leaderboard["top_k"] == 20)
        ].head(3)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection Baselines

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_baselines`
- 研究主题：`{quality.get('research_topic', '')}`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 训练/评估：静态 baseline 全样本打分；ML baseline 使用 walk-forward，只用测试月之前的数据训练。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`

## Leaderboard

{_format_markdown_table(leader_view, max_rows=40)}

## Year Slice

{_format_markdown_table(year_view, max_rows=40)}

## Realized Market State Slice

{_format_markdown_table(regime_view, max_rows=40)}

## Industry Exposure

{_format_markdown_table(industry_view, max_rows=40)}

## 口径

- 输入固定为 `data/cache/monthly_selection_features.parquet` 兼容的 M2 canonical dataset。
- 主训练池/主报告池为 `U1_liquid_tradable` 与 `U2_risk_sane`。
- 第一轮只使用 price-volume-only 特征：收益动量、低波、流动性、换手、价格位置和涨跌停路径特征。
- Top-K 并行报告 `20 / 30 / 50`；`B0_market_ew` 作为市场等权基准，非真实持仓模型。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于归因，不作为可交易信号。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- baseline overlap 不作为 M4 gate；M4 的核心是 Rank IC、Top-K 超额、Top-K vs next-K、分桶 spread、年度/状态稳定性、行业暴露和换手。

## 本轮结论

- 本轮新增：M4 price-volume-only baseline ranker，覆盖单因子、线性 blend、ElasticNet、Logistic top-bucket classifier、ExtraTrees sanity check、XGBoost regression/classifier。
- 数据质量：沿用 M2 canonical dataset 的 PIT 与候选池口径；本脚本只消费已落地特征，不引入新数据家族。
- `U1_liquid_tradable` Top20 当前领先模型：

{_format_markdown_table(best_u1_top20, max_rows=3)}

- `U2_risk_sane` Top20 当前领先模型：

{_format_markdown_table(best_u2_top20, max_rows=3)}

- 静态单因子/线性 blend 多数无法稳定跑赢市场，应保留为低门槛对照，不作为推荐候选。
- walk-forward ML baseline 有弱正向起点，但 strong-up / strong-down 切片仍不稳；M4 不进入生产。
- 下一步进入 M5，逐个验证 industry breadth、fund flow、fundamental、shareholder 等增量是否能稳定改善 Rank IC、Top-K 超额、分桶 spread 和强市参与度。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    args = parse_args()
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    docs_dir = ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    run_cfg = BaselineRunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        include_xgboost=not bool(args.skip_xgboost),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_topic = "monthly_selection_baselines"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_buckets_{int(args.bucket_count)}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    valid = valid_pool_frame(dataset)
    static_scores = build_static_scores(dataset)
    wf_scores, ml_importance = build_walk_forward_scores(dataset, run_cfg)
    scores = pd.concat([x for x in [static_scores, wf_scores] if not x.empty], ignore_index=True)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=run_cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=run_cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_feature_importance(static_scores, ml_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)

    quality = {
        "result_type": "monthly_selection_baselines",
        "research_topic": research_topic,
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": pools,
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
        "top_ks": top_ks,
        "bucket_count": int(args.bucket_count),
        "cost_assumption": f"{float(args.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_spec": "price_volume_only_v1",
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
        "pit_policy": "features are consumed from M2 PIT-safe monthly signal rows; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month",
        "hyperparameter_policy": "fixed conservative defaults; no random CV",
        "random_seed": int(args.random_seed),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
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
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = build_summary_payload(quality=quality, leaderboard=leaderboard)
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
            year_slice=year_slice,
            regime_slice=regime_slice,
            industry_exposure=industry_exposure,
            artifacts=[*artifact_paths, str(paths_out["manifest"].relative_to(ROOT))],
        ),
        encoding="utf-8",
    )

    print(f"[monthly-baselines] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-baselines] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-baselines] doc={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
