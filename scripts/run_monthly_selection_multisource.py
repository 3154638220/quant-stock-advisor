#!/usr/bin/env python3
"""M5: 月度选股多源特征扩展。

本脚本消费 M2 canonical dataset，并逐个特征族做增量评估：
price-volume baseline -> industry breadth -> fund flow -> fundamental -> shareholder。
评估口径复用 M4 的 walk-forward、Rank IC、Top-K、分桶、年度/市场状态和行业暴露诊断。
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

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from scripts.run_monthly_selection_baselines import (
    EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    ML_FEATURE_COLS,
    POOL_RULES,
    TOP20_COL,
    _format_markdown_table,
    _json_sanitize,
    _train_predict_sklearn,
    _train_predict_xgboost,
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
from src.features.fundamental_factors import DEFAULT_FUNDAMENTAL_COLS, pit_safe_fundamental_rows
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
from src.settings import config_path_candidates, load_config, resolve_config_path

# ═══════════════════════════════════════════════════════════════════════════
# 以下核心管线函数已提取到 src.pipeline.monthly_multisource：
#   attach_industry_breadth_features, attach_fund_flow_features,
#   attach_fundamental_features, attach_shareholder_features,
#   build_feature_specs, build_all_m5_scores, summarize_feature_coverage_by_spec,
#   summarize_feature_importance, build_incremental_delta, 等。
# 本脚本保留本地副本仅为向后兼容；后续维护请直接修改 monthly_multisource 中的版本。
# ═══════════════════════════════════════════════════════════════════════════
from src.pipeline.monthly_multisource import (  # noqa: F401
    FeatureSpec,
    M5RunConfig,
    add_zscore_and_missing_flags,
    attach_enabled_families,
    attach_fund_flow_features,
    attach_fundamental_features,
    attach_industry_breadth_features,
    attach_shareholder_features,
    build_all_m5_scores,
    build_feature_specs,
    build_incremental_delta,
    summarize_feature_coverage_by_spec,
    summarize_feature_importance,
    _cap_fit_rows,
)

PRICE_VOLUME_FEATURES: tuple[str, ...] = ML_FEATURE_COLS

INDUSTRY_BREADTH_RAW_FEATURES: tuple[str, ...] = (
    "feature_industry_ret20_mean",
    "feature_industry_ret60_mean",
    "feature_industry_positive_ret20_ratio",
    "feature_industry_amount20_mean",
    "feature_industry_low_vol20_mean",
)

FUND_FLOW_RAW_FEATURES: tuple[str, ...] = (
    "feature_fund_flow_main_inflow_5d",
    "feature_fund_flow_main_inflow_10d",
    "feature_fund_flow_main_inflow_20d",
    "feature_fund_flow_super_inflow_10d",
    "feature_fund_flow_divergence_20d",
    "feature_fund_flow_main_inflow_streak",
)

FUNDAMENTAL_RAW_FEATURES: tuple[str, ...] = tuple(
    f"feature_fundamental_{c}"
    for c in DEFAULT_FUNDAMENTAL_COLS
    if c not in {"northbound_net_inflow", "margin_buy_ratio"}
)

SHAREHOLDER_RAW_FEATURES: tuple[str, ...] = (
    "feature_shareholder_holder_count_log",
    "feature_shareholder_holder_change_rate",
    "feature_shareholder_concentration_proxy",
)


@dataclass(frozen=True)
class M5RunConfig:
    top_ks: tuple[int, ...] = (20, 30, 50)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    include_xgboost: bool = True
    availability_lag_days: int = 45
    pit_fallback_lag_days: int = 45
    hpo_enabled: bool = False
    hpo_n_trials: int = 30
    hpo_cv_folds: int = 3
    window_type: str = "expanding"
    halflife_months: float = 36.0
    ml_models: tuple[str, ...] = ("elasticnet", "logistic", "extratrees", "xgboost_excess", "xgboost_top20")
    model_n_jobs: int = 0


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    families: tuple[str, ...]
    feature_cols: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M5 多源特征扩展")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_m5_multisource")
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
    p.add_argument(
        "--model-n-jobs",
        type=int,
        default=0,
        help="模型训练线程数；0 表示使用全部 CPU 核心，1 保持旧的单线程行为。",
    )
    p.add_argument("--availability-lag-days", type=int, default=30)
    p.add_argument("--skip-xgboost", action="store_true", help="跳过 XGBoost，便于快速烟雾测试")
    p.add_argument(
        "--ml-models",
        type=str,
        default="elasticnet,logistic,extratrees,xgboost_excess,xgboost_top20",
        help="M5 walk-forward 模型列表，可选 elasticnet,logistic,extratrees,xgboost_excess,xgboost_top20",
    )
    p.add_argument(
        "--families",
        type=str,
        default="industry_breadth,fund_flow,fundamental,shareholder",
        help="要启用的增量家族，逗号分隔。price_volume_only 始终作为 baseline。",
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


def _normalize_symbol_date(df: pd.DataFrame, *, date_col: str = "signal_date") -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize().astype("datetime64[ns]")
    return out


def _winsor_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 3:
        return pd.Series(0.0, index=series.index)
    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    clipped = x.clip(lo, hi)
    med = clipped.median()
    filled = clipped.fillna(med)
    std = filled.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((filled - filled.mean()) / std).clip(-5.0, 5.0)


def add_zscore_and_missing_flags(
    dataset: pd.DataFrame,
    raw_cols: tuple[str, ...],
    *,
    date_col: str = "signal_date",
) -> pd.DataFrame:
    out = dataset.copy()
    for col in raw_cols:
        if col not in out.columns:
            out[col] = np.nan
        vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"is_missing_{col}"] = vals.isna().astype(int)
        out[f"{col}_z"] = vals.groupby(out[date_col], sort=False).transform(_winsor_zscore)
    return out


def _unique_signal_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "symbol",
        "industry_level1",
        "feature_ret_20d",
        "feature_ret_60d",
        "feature_realized_vol_20d",
        "feature_amount_20d_log",
    ]
    keep = [c for c in cols if c in dataset.columns]
    base = (
        dataset.sort_values(["signal_date", "symbol", "candidate_pool_version"])
        .drop_duplicates(["signal_date", "symbol"], keep="first")[keep]
        .copy()
    )
    return _normalize_symbol_date(base)


def attach_industry_breadth_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """用信号日已知的行业截面强度构造 industry breadth 特征。"""
    base = _unique_signal_frame(dataset)
    if base.empty or "industry_level1" not in base.columns:
        return add_zscore_and_missing_flags(dataset, INDUSTRY_BREADTH_RAW_FEATURES)

    base["industry_level1"] = base["industry_level1"].fillna("_UNKNOWN_").astype(str)
    base["_ret20_positive"] = (pd.to_numeric(base.get("feature_ret_20d"), errors="coerce") > 0).astype(float)
    grouped = base.groupby(["signal_date", "industry_level1"], dropna=False, sort=False)
    ind = (
        grouped.agg(
            feature_industry_ret20_mean=("feature_ret_20d", "mean"),
            feature_industry_ret60_mean=("feature_ret_60d", "mean"),
            feature_industry_positive_ret20_ratio=("_ret20_positive", "mean"),
            feature_industry_amount20_mean=("feature_amount_20d_log", "mean"),
            feature_industry_low_vol20_mean=("feature_realized_vol_20d", lambda s: -pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )
    out = dataset.merge(ind, on=["signal_date", "industry_level1"], how="left")
    return add_zscore_and_missing_flags(out, INDUSTRY_BREADTH_RAW_FEATURES)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()
    return bool(row and int(row[0]) > 0)


def _read_table_if_exists(db_path: Path, table: str, cols: list[str]) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=cols)
    with duckdb.connect(str(db_path), read_only=True) as con:
        if not _table_exists(con, table):
            return pd.DataFrame(columns=cols)
        info = con.execute(f"PRAGMA table_info('{table}')").df()
        available = [c for c in cols if c in set(info["name"].astype(str))]
        if not available:
            return pd.DataFrame(columns=cols)
        return con.execute(f"SELECT {', '.join(available)} FROM {table}").df()


def _merge_asof_by_symbol(signal: pd.DataFrame, raw: pd.DataFrame, *, left_date: str, right_date: str) -> pd.DataFrame:
    if signal.empty:
        return signal.copy()
    if raw.empty:
        return signal.copy()
    left = _normalize_symbol_date(signal, date_col=left_date).sort_values([left_date, "symbol"], kind="mergesort")
    right = _normalize_symbol_date(raw, date_col=right_date).sort_values([right_date, "symbol"], kind="mergesort")
    right = right.dropna(subset=[right_date]).copy()
    if right.empty:
        return left
    return pd.merge_asof(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        left_on=left_date,
        right_on=right_date,
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )


def _filter_pit_safe_fundamental_rows(
    raw: pd.DataFrame,
    *,
    signal_date: pd.Timestamp | None = None,
    fallback_lag_days: int = 45,
    disclosure_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop statement rows whose announcement date is not after the report period.

    P0-1: 优先使用实际披露日历，缺失时回退为法定截止日保守估计，
    最后兜底为 fallback_lag_days 固定延迟。
    """
    if raw.empty:
        return raw
    out = raw.copy()
    return out[
        pit_safe_fundamental_rows(
            out,
            announcement_col="_fund_announcement_date",
            signal_date=signal_date,
            fallback_lag_days=fallback_lag_days,
            disclosure_calendar=disclosure_calendar,
        )
    ].copy()


def attach_fund_flow_features(dataset: pd.DataFrame, db_path: Path, *, table: str = "a_share_fund_flow") -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw = _read_table_if_exists(
        db_path,
        table,
        [
            "symbol",
            "trade_date",
            "main_net_inflow_pct",
            "super_large_net_inflow_pct",
            "small_net_inflow_pct",
        ],
    )
    if raw.empty:
        out = dataset.copy()
        for col in FUND_FLOW_RAW_FEATURES:
            out[col] = np.nan
        return add_zscore_and_missing_flags(out, FUND_FLOW_RAW_FEATURES)

    raw = raw.rename(columns={"trade_date": "_flow_trade_date"})
    raw = _normalize_symbol_date(raw, date_col="_flow_trade_date")
    for col in ["main_net_inflow_pct", "super_large_net_inflow_pct", "small_net_inflow_pct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.drop_duplicates(["symbol", "_flow_trade_date"], keep="last")
    raw = raw.sort_values(["symbol", "_flow_trade_date"], kind="mergesort").reset_index(drop=True)
    g = raw.groupby("symbol", sort=False)
    for w in (5, 10, 20):
        raw[f"feature_fund_flow_main_inflow_{w}d"] = g["main_net_inflow_pct"].transform(
            lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean()
        )
    raw["feature_fund_flow_super_inflow_10d"] = g["super_large_net_inflow_pct"].transform(
        lambda s: s.rolling(10, min_periods=5).mean()
    )
    small20 = g["small_net_inflow_pct"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    raw["feature_fund_flow_divergence_20d"] = raw["feature_fund_flow_main_inflow_20d"] - small20
    raw["feature_fund_flow_main_inflow_streak"] = _compute_signed_streak(raw, "main_net_inflow_pct")
    raw_keep = raw[["symbol", "_flow_trade_date", *FUND_FLOW_RAW_FEATURES]].copy()
    attached = _merge_asof_by_symbol(signal, raw_keep, left_date="signal_date", right_date="_flow_trade_date")
    attached = attached.drop(columns=["_flow_trade_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, FUND_FLOW_RAW_FEATURES)


def _compute_signed_streak(df: pd.DataFrame, col: str) -> pd.Series:
    streak = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby("symbol", sort=False):
        cur = 0
        sign = 0
        vals: list[float] = []
        for val in pd.to_numeric(grp[col], errors="coerce"):
            if pd.isna(val):
                cur = 0
                sign = 0
            elif val > 0:
                cur = cur + 1 if sign == 1 else 1
                sign = 1
            else:
                cur = cur - 1 if sign == -1 else -1
                sign = -1
            vals.append(float(cur))
        streak.loc[grp.index] = vals
    return streak


def attach_fundamental_features(
    dataset: pd.DataFrame,
    db_path: Path,
    *,
    table: str = "a_share_fundamental",
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw_cols = [
        "symbol",
        "report_period",
        "announcement_date",
        "source",
        *[c.replace("feature_fundamental_", "") for c in FUNDAMENTAL_RAW_FEATURES],
    ]
    raw = _read_table_if_exists(db_path, table, raw_cols)
    if raw.empty:
        out = dataset.copy()
        for col in FUNDAMENTAL_RAW_FEATURES:
            out[col] = np.nan
        return add_zscore_and_missing_flags(out, FUNDAMENTAL_RAW_FEATURES)

    raw = raw.rename(columns={"announcement_date": "_fund_announcement_date"})
    raw = _normalize_symbol_date(raw, date_col="_fund_announcement_date")
    raw["report_period"] = pd.to_datetime(raw.get("report_period"), errors="coerce").dt.normalize()
    raw = raw[raw["_fund_announcement_date"].notna()].copy()
    raw = _filter_pit_safe_fundamental_rows(raw)
    raw = raw.sort_values(["symbol", "_fund_announcement_date", "report_period"], kind="mergesort")
    raw = raw.drop_duplicates(["symbol", "_fund_announcement_date", "report_period"], keep="last")
    rename_map = {c.replace("feature_fundamental_", ""): c for c in FUNDAMENTAL_RAW_FEATURES}
    raw = raw.rename(columns=rename_map)
    for col in FUNDAMENTAL_RAW_FEATURES:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    attached = _merge_asof_by_symbol(
        signal,
        raw[["symbol", "_fund_announcement_date", *FUNDAMENTAL_RAW_FEATURES]],
        left_date="signal_date",
        right_date="_fund_announcement_date",
    )
    attached = attached.drop(columns=["_fund_announcement_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, FUNDAMENTAL_RAW_FEATURES)


def attach_shareholder_features(
    dataset: pd.DataFrame,
    db_path: Path,
    *,
    table: str = "a_share_shareholder",
    availability_lag_days: int = 30,
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw = _read_table_if_exists(
        db_path,
        table,
        ["symbol", "end_date", "notice_date", "holder_count", "holder_change"],
    )
    if raw.empty:
        out = dataset.copy()
        for col in SHAREHOLDER_RAW_FEATURES:
            out[col] = np.nan
        return add_zscore_and_missing_flags(out, SHAREHOLDER_RAW_FEATURES)

    raw["symbol"] = raw["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    raw["end_date"] = pd.to_datetime(raw.get("end_date"), errors="coerce").dt.normalize()
    raw["notice_date"] = pd.to_datetime(raw.get("notice_date"), errors="coerce").dt.normalize()
    raw["holder_count"] = pd.to_numeric(raw.get("holder_count"), errors="coerce")
    raw["holder_change"] = pd.to_numeric(raw.get("holder_change"), errors="coerce")
    raw["_holder_availability_date"] = raw["notice_date"]
    fallback = raw["_holder_availability_date"].isna() | (raw["_holder_availability_date"] < raw["end_date"])
    raw.loc[fallback, "_holder_availability_date"] = raw.loc[fallback, "end_date"] + pd.to_timedelta(
        int(availability_lag_days),
        unit="D",
    )
    raw = raw.dropna(subset=["_holder_availability_date"]).copy()
    raw = raw.sort_values(["symbol", "_holder_availability_date", "end_date"], kind="mergesort")
    raw = raw.drop_duplicates(["symbol", "_holder_availability_date", "end_date"], keep="last")
    holder_count = pd.to_numeric(raw["holder_count"], errors="coerce")
    raw["feature_shareholder_holder_count_log"] = np.log(holder_count.where(holder_count > 0))
    raw["feature_shareholder_holder_change_rate"] = pd.to_numeric(raw["holder_change"], errors="coerce") / holder_count.replace(0, np.nan)
    raw["feature_shareholder_concentration_proxy"] = -raw["feature_shareholder_holder_count_log"]
    raw = raw.rename(columns={"_holder_availability_date": "holder_availability_date"})
    raw = _normalize_symbol_date(raw, date_col="holder_availability_date")
    attached = _merge_asof_by_symbol(
        signal,
        raw[["symbol", "holder_availability_date", *SHAREHOLDER_RAW_FEATURES]],
        left_date="signal_date",
        right_date="holder_availability_date",
    )
    attached = attached.drop(columns=["holder_availability_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, SHAREHOLDER_RAW_FEATURES)


def build_feature_specs(enabled_families: list[str]) -> list[FeatureSpec]:
    enabled = [x for x in enabled_families if x and x != "price_volume_only"]
    specs = [
        FeatureSpec(
            name="price_volume_only",
            families=("price_volume",),
            feature_cols=PRICE_VOLUME_FEATURES,
        )
    ]
    cumulative: list[str] = list(PRICE_VOLUME_FEATURES)
    family_cols: dict[str, tuple[str, ...]] = {
        "industry_breadth": tuple(f"{c}_z" for c in INDUSTRY_BREADTH_RAW_FEATURES),
        "fund_flow": tuple(f"{c}_z" for c in FUND_FLOW_RAW_FEATURES),
        "fundamental": tuple(f"{c}_z" for c in FUNDAMENTAL_RAW_FEATURES),
        "shareholder": tuple(f"{c}_z" for c in SHAREHOLDER_RAW_FEATURES),
    }
    family_order = ["industry_breadth", "fund_flow", "fundamental", "shareholder"]
    active_families = ["price_volume"]
    for family in family_order:
        if family not in enabled:
            continue
        cumulative.extend(family_cols[family])
        active_families.append(family)
        specs.append(
            FeatureSpec(
                name="plus_" + "_plus_".join(active_families[1:]),
                families=tuple(active_families),
                feature_cols=tuple(dict.fromkeys(cumulative)),
            )
        )
    return specs


def attach_enabled_families(dataset: pd.DataFrame, db_path: Path, cfg: M5RunConfig, enabled_families: list[str]) -> pd.DataFrame:
    out = dataset.copy()
    if "industry_breadth" in enabled_families:
        out = attach_industry_breadth_features(out)
    if "fund_flow" in enabled_families:
        out = attach_fund_flow_features(out, db_path)
    if "fundamental" in enabled_families:
        out = attach_fundamental_features(out, db_path)
    if "shareholder" in enabled_families:
        out = attach_shareholder_features(out, db_path, availability_lag_days=cfg.availability_lag_days)
    return out


def summarize_feature_coverage_by_spec(dataset: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    base = dataset[dataset["candidate_pool_version"] == "U1_liquid_tradable"].copy()
    rows: list[dict[str, Any]] = []
    if base.empty:
        return pd.DataFrame()
    pool_pass = base["candidate_pool_pass"].astype(bool)
    for spec in specs:
        for col in spec.feature_cols:
            raw_col = col[:-2] if col.endswith("_z") else col
            vals = pd.to_numeric(base[raw_col], errors="coerce") if raw_col in base.columns else pd.Series(np.nan, index=base.index)
            rows.append(
                {
                    "feature_spec": spec.name,
                    "families": ",".join(spec.families),
                    "feature": col,
                    "raw_feature": raw_col,
                    "rows": int(len(base)),
                    "non_null": int(vals.notna().sum()),
                    "coverage_ratio": float(vals.notna().mean()) if len(base) else np.nan,
                    "candidate_pool_pass_coverage_ratio": float(vals.loc[pool_pass].notna().mean()) if pool_pass.any() else np.nan,
                    "first_signal_date": str(base.loc[vals.notna(), "signal_date"].min().date()) if vals.notna().any() else "",
                    "last_signal_date": str(base.loc[vals.notna(), "signal_date"].max().date()) if vals.notna().any() else "",
                }
            )
    return pd.DataFrame(rows)


def build_walk_forward_scores_for_spec(
    dataset: pd.DataFrame,
    spec: FeatureSpec,
    cfg: M5RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_pool_frame(dataset)
    feature_cols = [c for c in spec.feature_cols if c in valid.columns]
    if valid.empty or not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    requested = set(cfg.ml_models)
    model_specs: list[tuple[str, str]] = []
    if "elasticnet" in requested:
        model_specs.append(("M4_elasticnet_excess", "elasticnet"))
    if "logistic" in requested:
        model_specs.append(("M4_logistic_top20", "logistic_classifier"))
    if "extratrees" in requested:
        model_specs.append(("M4_extratrees_excess", "tree_sanity"))
    if cfg.include_xgboost:
        if "xgboost_excess" in requested:
            model_specs.append(("M4_xgboost_excess", "xgboost_regression"))
        if "xgboost_top20" in requested:
            model_specs.append(("M4_xgboost_top20", "xgboost_classifier"))
    if not model_specs:
        return pd.DataFrame(), pd.DataFrame()

    for pool, pool_df in valid.groupby("candidate_pool_version", sort=True):
        months = sorted(pd.to_datetime(pool_df["signal_date"]).dropna().unique())
        for test_month in months:
            if str(cfg.window_type).lower() == "rolling":
                prev_months = [m for m in months if m < test_month]
                keep_months = prev_months[-int(cfg.min_train_months):]
                train = pool_df[pool_df["signal_date"].isin(keep_months)].copy()
            else:
                train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
            print(
                "[monthly-m5] "
                f"spec={spec.name} pool={pool} test_month={pd.Timestamp(test_month).date()} "
                f"train_months={train['signal_date'].nunique()} train_rows={len(train)} test_rows={len(test)}",
                flush=True,
            )
            train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
            sample_weight = None
            if cfg.halflife_months > 0:
                age_months = (
                    (pd.Timestamp(test_month).normalize() - pd.to_datetime(train_fit["signal_date"]).dt.normalize())
                    .dt.days
                    / 30.0
                )
                sample_weight = np.exp(-np.log(2) * age_months / float(cfg.halflife_months))
                sample_weight = sample_weight.clip(lower=0.01)
            hpo_params: dict | None = None
            if cfg.hpo_enabled and train_fit["signal_date"].nunique() >= 4:
                try:
                    from src.pipeline.hpo_utils import tune_xgboost_regressor

                    hpo_params = tune_xgboost_regressor(
                        train_fit,
                        feature_cols,
                        n_trials=cfg.hpo_n_trials,
                        cv_folds=cfg.hpo_cv_folds,
                        random_seed=cfg.random_seed,
                        model_n_jobs=cfg.model_n_jobs,
                    )
                except Exception:
                    hpo_params = None
            for base_model_name, model_type in model_specs:
                if base_model_name.startswith("M4_xgboost"):
                    scores, imp = _train_predict_xgboost(
                        model_name=base_model_name,
                        model_type=model_type,
                        train=train_fit,
                        test=test,
                        feature_cols=feature_cols,
                        random_seed=cfg.random_seed,
                        model_n_jobs=cfg.model_n_jobs,
                        hpo_params=hpo_params,
                        sample_weight=sample_weight,
                    )
                else:
                    scores, imp = _train_predict_sklearn(
                        model_name=base_model_name,
                        model_type=model_type,
                        train=train_fit,
                        test=test,
                        feature_cols=feature_cols,
                        random_seed=cfg.random_seed,
                        model_n_jobs=cfg.model_n_jobs,
                        sample_weight=sample_weight,
                    )
                model_name = f"M5_{spec.name}_{base_model_name.replace('M4_', '')}"
                if scores is not None and not scores.empty:
                    scores = scores.copy()
                    scores["model"] = model_name
                    scores["feature_spec"] = spec.name
                    scores["feature_families"] = ",".join(spec.families)
                    scores["rank"] = scores.groupby(
                        ["signal_date", "candidate_pool_version", "model"], sort=False
                    )["score"].rank(method="first", ascending=False)
                    score_frames.append(scores)
                if not imp.empty:
                    imp = imp.copy()
                    imp["model"] = model_name
                    imp["feature_spec"] = spec.name
                    imp["feature_families"] = ",".join(spec.families)
                    imp["candidate_pool_version"] = pool
                    imp["test_signal_date"] = pd.Timestamp(test_month)
                    importance_frames.append(imp)
    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def _cap_fit_rows(train: pd.DataFrame, *, max_rows: int, random_seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(train) <= max_rows:
        return train
    months = sorted(pd.to_datetime(train["signal_date"]).dropna().unique())
    if not months:
        return train.sample(n=max_rows, random_state=random_seed).sort_values(["signal_date", "symbol"])
    per_month = max(int(np.ceil(max_rows / len(months))), 1)
    chunks: list[pd.DataFrame] = []
    for i, month in enumerate(months):
        part = train[train["signal_date"] == month]
        if len(part) <= per_month:
            chunks.append(part)
        else:
            chunks.append(part.sample(n=per_month, random_state=random_seed + i))
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_seed)
    return out.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def build_all_m5_scores(dataset: pd.DataFrame, specs: list[FeatureSpec], cfg: M5RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for spec in specs:
        scores, imp = build_walk_forward_scores_for_spec(dataset, spec, cfg)
        if not scores.empty:
            score_frames.append(scores)
        if not imp.empty:
            importance_frames.append(imp)
    scores_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    imp_out = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return scores_out, imp_out


def summarize_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
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
    return out.sort_values(["feature_spec", "candidate_pool_version", "model", "importance"], ascending=[True, True, True, False])


def _base_model_key(model: str) -> str:
    if "_price_volume_only_" in model:
        return model.split("_price_volume_only_", 1)[1]
    if "_plus_" in model:
        return model.split("_plus_", 1)[1].split("_", 1)[-1]
    return model


def build_incremental_delta(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return pd.DataFrame()
    df = leaderboard[leaderboard["model"].astype(str).str.startswith("M5_")].copy()
    if df.empty:
        return pd.DataFrame()
    df["feature_spec"] = df["model"].astype(str).str.replace(r"^M5_", "", regex=True).str.extract(
        r"(.+)_(elasticnet_excess|logistic_top20|extratrees_excess|xgboost_excess|xgboost_top20)$",
        expand=False,
    )[0]
    df["base_model"] = df["model"].astype(str).str.extract(
        r"(elasticnet_excess|logistic_top20|extratrees_excess|xgboost_excess|xgboost_top20)$",
        expand=False,
    )
    baseline = df[df["feature_spec"] == "price_volume_only"][
        [
            "candidate_pool_version",
            "top_k",
            "base_model",
            "topk_excess_mean",
            "topk_excess_after_cost_mean",
            "rank_ic_mean",
            "quantile_top_minus_bottom_mean",
        ]
    ].rename(
        columns={
            "topk_excess_mean": "baseline_topk_excess_mean",
            "topk_excess_after_cost_mean": "baseline_topk_excess_after_cost_mean",
            "rank_ic_mean": "baseline_rank_ic_mean",
            "quantile_top_minus_bottom_mean": "baseline_quantile_top_minus_bottom_mean",
        }
    )
    out = df.merge(baseline, on=["candidate_pool_version", "top_k", "base_model"], how="left")
    for col in [
        "topk_excess_mean",
        "topk_excess_after_cost_mean",
        "rank_ic_mean",
        "quantile_top_minus_bottom_mean",
    ]:
        out[f"delta_{col}"] = out[col] - out[f"baseline_{col}"]
    return out[out["feature_spec"] != "price_volume_only"].sort_values(
        ["top_k", "candidate_pool_version", "delta_topk_excess_after_cost_mean", "delta_rank_ic_mean"],
        ascending=[True, True, False, False],
    )


def build_quality_payload(
    *,
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    specs: list[FeatureSpec],
    cfg: M5RunConfig,
    dataset_path: Path,
    db_path: Path,
    output_stem: str,
    config_source: str,
    research_config_id: str,
) -> dict[str, Any]:
    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m5_multisource",
        "research_topic": "monthly_selection_m5_multisource",
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
        "feature_specs": [{"name": s.name, "families": list(s.families), "feature_count": len(s.feature_cols)} for s in specs],
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
        "pit_policy": "monthly rows use signal-date-or-earlier features; fundamental uses announcement_date <= signal_date; shareholder uses notice_date or conservative end_date lag; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month",
        "hyperparameter_policy": "fixed conservative defaults; no random CV",
        "ml_models": list(cfg.ml_models),
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
    incremental_delta: pd.DataFrame,
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
    delta_view = incremental_delta.head(40).copy()
    cov_view = feature_coverage.copy()
    if not cov_view.empty:
        cov_view = cov_view[
            ~cov_view["feature_spec"].eq("price_volume_only") | cov_view["feature"].isin(list(PRICE_VOLUME_FEATURES)[:3])
        ].head(80)
    year_view = year_slice.sort_values(["candidate_pool_version", "model", "top_k", "year"]).head(40)
    regime_view = regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(40)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M5 Multisource

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_m5_multisource`
- 研究主题：`{quality.get('research_topic', '')}`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 数据库：`{quality.get('duckdb_path', '')}`
- 训练/评估：每个 feature spec 独立 walk-forward，只用测试月之前的数据训练。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`
- 单窗训练行上限：`{quality.get('max_fit_rows', 0)}`（`0` 表示不抽样）

## Leaderboard

{_format_markdown_table(leader_view, max_rows=40)}

## Incremental Delta vs Price-Volume

{_format_markdown_table(delta_view, max_rows=40)}

## Feature Coverage

{_format_markdown_table(cov_view, max_rows=80)}

## Year Slice

{_format_markdown_table(year_view, max_rows=40)}

## Realized Market State Slice

{_format_markdown_table(regime_view, max_rows=40)}

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 `industry_breadth -> fund_flow -> fundamental -> shareholder` 累积加入。
- `industry_breadth` 只使用信号日已知的行业内价量截面统计。
- `fund_flow` 使用 `trade_date <= signal_date` 的资金流滚动聚合；覆盖不足时保留缺失标记并在 coverage 表中披露。
- `fundamental` 使用 `announcement_date <= signal_date` 的最新快照，禁止使用报告期直接穿越。
- `shareholder` 优先使用 `notice_date`，异常或缺失时使用 `end_date + fallback lag` 的保守可用日。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- 若设置 `max_fit_rows > 0`，训练窗内按月份做确定性均匀抽样；OOS 测试月和候选池不抽样。

## 本轮结论

- 本轮新增：M5 多源特征扩展 runner，逐族输出覆盖率、Rank IC、Top-K 超额、分桶 spread、特征重要性、年度和 regime slice。
- 当前数据覆盖呈近端特征特征：资金流、基本面、shareholder 的历史覆盖短于 price-volume，因此增量结果应与 coverage 表一起解释。
- M5 仍为研究诊断，不进入生产；若某个家族显示稳定增量，下一步进入 M6 learning-to-rank。

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
    cfg = M5RunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        include_xgboost=not bool(args.skip_xgboost),
        availability_lag_days=int(args.availability_lag_days),
        ml_models=tuple(_parse_str_list(args.ml_models)),
        model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_models_{'-'.join(slugify_token(x) for x in _parse_str_list(args.ml_models))}"
        f"_maxfit_{int(args.max_fit_rows)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m5_multisource",
        research_topic="monthly_selection_m5_multisource",
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    research_config_id = identity.research_config_id
    output_stem = identity.output_stem

    print(f"[monthly-m5] research_config_id={research_config_id}")
    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    dataset = attach_enabled_families(dataset, db_path, cfg, enabled_families)
    specs = build_feature_specs(enabled_families)
    feature_coverage = summarize_feature_coverage_by_spec(dataset, specs)
    scores, raw_importance = build_all_m5_scores(dataset, specs, cfg)
    if scores.empty:
        warnings.warn("M5 未生成任何 score；请检查训练窗、候选池或特征覆盖。", RuntimeWarning)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_feature_importance(raw_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)
    incremental_delta = build_incremental_delta(leaderboard)
    quality = build_quality_payload(
        dataset=dataset,
        scores=scores,
        specs=specs,
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
        "incremental_delta": results_dir / f"{output_stem}_incremental_delta.csv",
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
    incremental_delta.to_csv(paths_out["incremental_delta"], index=False)
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
        "best_incremental_by_topk": incremental_delta.groupby("top_k", as_index=False).head(5).to_dict(orient="records")
        if not incremental_delta.empty
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
            incremental_delta=incremental_delta,
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
    feature_columns = tuple(dict.fromkeys(col for spec in specs for col in spec.feature_cols))
    data_slice = DataSlice(
        dataset_name="monthly_selection_m5_multisource",
        source_tables=(_project_relative(dataset_path), _project_relative(db_path)),
        date_start=min_signal_date,
        date_end=max_signal_date,
        asof_trade_date=max_signal_date or None,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(pools),
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="m5_" + "_".join(["price_volume_only", *enabled_families]),
        feature_columns=feature_columns,
        label_columns=(LABEL_COL, EXCESS_COL, MARKET_COL, TOP20_COL),
        pit_policy=quality["pit_policy"],
        config_path=config_source,
        extra={
            "dataset_path": _project_relative(dataset_path),
            "duckdb_path": _project_relative(db_path),
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "enabled_families": enabled_families,
            "feature_specs": quality["feature_specs"],
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
            "availability_lag_days": int(args.availability_lag_days),
            "cv_policy": quality["cv_policy"],
        },
    )
    artifact_refs = (
        ArtifactRef("summary_json", _project_relative(paths_out["summary_json"]), "json"),
        ArtifactRef("leaderboard_csv", _project_relative(paths_out["leaderboard"]), "csv"),
        ArtifactRef("incremental_delta_csv", _project_relative(paths_out["incremental_delta"]), "csv"),
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
        "feature_spec_count": int(len(specs)),
        "feature_coverage_rows": int(len(feature_coverage)),
        "incremental_delta_rows": int(len(incremental_delta)),
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
            "passed": bool(not incremental_delta.empty),
            "incremental_delta_rows": int(len(incremental_delta)),
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
                "include_xgboost": cfg.include_xgboost,
                "availability_lag_days": cfg.availability_lag_days,
                "ml_models": list(cfg.ml_models),
                "model_n_jobs": normalize_model_n_jobs(cfg.model_n_jobs),
            },
            "overrides": {
                key: value
                for key, value in {
                    "dataset": args.dataset,
                    "duckdb_path": args.duckdb_path.strip(),
                    "results_dir": args.results_dir.strip(),
                    "top_k": args.top_k,
                    "candidate_pools": args.candidate_pools,
                    "families": args.families,
                    "ml_models": args.ml_models,
                    "skip_xgboost": args.skip_xgboost,
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
            "blocking_reasons": ["m5_multisource_research_only_not_promotion_candidate"],
        },
        notes="Monthly selection M5 multisource contract; model outputs are unchanged.",
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

    print(f"[monthly-m5] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-m5] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-m5] manifest={paths_out['manifest']}")
    print(f"[monthly-m5] research_index={experiments_dir / 'research_results.jsonl'}")
    print(f"[monthly-m5] doc={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
