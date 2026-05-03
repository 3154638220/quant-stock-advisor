"""月度选股多源特征扩展管线。

从 scripts/run_monthly_selection_multisource.py 提取核心算法逻辑：
- 特征家族 attachment（industry_breadth, fund_flow, fundamental, shareholder）
- Walk-forward 多模型打分
- 特征覆盖率 / 重要性 / 增量 delta 汇总
- FeatureSpec 与 M5RunConfig 数据类

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.pipeline.monthly_baselines import (
    _train_predict_sklearn,
    _train_predict_xgboost,
    normalize_model_n_jobs,
    valid_pool_frame,
)
from src.features.fundamental_factors import DEFAULT_FUNDAMENTAL_COLS, pit_safe_fundamental_rows

# ── 特征列常量 ───────────────────────────────────────────────────────────

PRICE_VOLUME_FEATURES: tuple[str, ...] = (
    "feature_ret_5d",
    "feature_ret_20d",
    "feature_ret_60d",
    "feature_realized_vol_20d",
    "feature_amount_20d_log",
    "feature_turnover_20d",
    "feature_price_position_250d",
    "feature_limit_move_hits_20d",
)

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


# ── 配置数据类 ───────────────────────────────────────────────────────────

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
    availability_lag_days: int = 30
    ml_models: tuple[str, ...] = ("elasticnet", "logistic", "extratrees", "xgboost_excess", "xgboost_top20")
    model_n_jobs: int = 0


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    families: tuple[str, ...]
    feature_cols: tuple[str, ...]


# ── 辅助函数 ─────────────────────────────────────────────────────────────

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
        "signal_date", "symbol", "industry_level1",
        "feature_ret_20d", "feature_ret_60d",
        "feature_realized_vol_20d", "feature_amount_20d_log",
    ]
    keep = [c for c in cols if c in dataset.columns]
    base = (
        dataset.sort_values(["signal_date", "symbol", "candidate_pool_version"])
        .drop_duplicates(["signal_date", "symbol"], keep="first")[keep]
        .copy()
    )
    return _normalize_symbol_date(base)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
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


def _merge_asof_by_symbol(
    signal: pd.DataFrame, raw: pd.DataFrame, *, left_date: str, right_date: str
) -> pd.DataFrame:
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
        left.reset_index(drop=True), right.reset_index(drop=True),
        left_on=left_date, right_on=right_date, by="symbol",
        direction="backward", allow_exact_matches=True,
    )


def _filter_pit_safe_fundamental_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    out = raw.copy()
    return out[
        pit_safe_fundamental_rows(out, announcement_col="_fund_announcement_date")
    ].copy()


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


# ── 特征家族 attachment ──────────────────────────────────────────────────

def attach_industry_breadth_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """用信号日已知的行业截面强度构造 industry breadth 特征。"""
    base = _unique_signal_frame(dataset)
    if base.empty or "industry_level1" not in base.columns:
        return add_zscore_and_missing_flags(dataset, INDUSTRY_BREADTH_RAW_FEATURES)

    base["industry_level1"] = base["industry_level1"].fillna("_UNKNOWN_").astype(str)
    base["_ret20_positive"] = (pd.to_numeric(base.get("feature_ret_20d"), errors="coerce") > 0).astype(float)
    grouped = base.groupby(["signal_date", "industry_level1"], dropna=False, sort=False)
    ind = grouped.agg(
        feature_industry_ret20_mean=("feature_ret_20d", "mean"),
        feature_industry_ret60_mean=("feature_ret_60d", "mean"),
        feature_industry_positive_ret20_ratio=("_ret20_positive", "mean"),
        feature_industry_amount20_mean=("feature_amount_20d_log", "mean"),
        feature_industry_low_vol20_mean=("feature_realized_vol_20d", lambda s: -pd.to_numeric(s, errors="coerce").mean()),
    ).reset_index()
    out = dataset.merge(ind, on=["signal_date", "industry_level1"], how="left")
    return add_zscore_and_missing_flags(out, INDUSTRY_BREADTH_RAW_FEATURES)


def attach_fund_flow_features(
    dataset: pd.DataFrame, db_path: Path, *, table: str = "a_share_fund_flow"
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw = _read_table_if_exists(
        db_path, table,
        ["symbol", "trade_date", "main_net_inflow_pct", "super_large_net_inflow_pct", "small_net_inflow_pct"],
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


def attach_fundamental_features(
    dataset: pd.DataFrame, db_path: Path, *, table: str = "a_share_fundamental",
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw_cols = [
        "symbol", "report_period", "announcement_date", "source",
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
        left_date="signal_date", right_date="_fund_announcement_date",
    )
    attached = attached.drop(columns=["_fund_announcement_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, FUNDAMENTAL_RAW_FEATURES)


def attach_shareholder_features(
    dataset: pd.DataFrame, db_path: Path, *, table: str = "a_share_shareholder", availability_lag_days: int = 30,
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw = _read_table_if_exists(
        db_path, table,
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
        int(availability_lag_days), unit="D",
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
        signal, raw[["symbol", "holder_availability_date", *SHAREHOLDER_RAW_FEATURES]],
        left_date="signal_date", right_date="holder_availability_date",
    )
    attached = attached.drop(columns=["holder_availability_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, SHAREHOLDER_RAW_FEATURES)


# ── FeatureSpec 构建 ─────────────────────────────────────────────────────

def build_feature_specs(enabled_families: list[str]) -> list[FeatureSpec]:
    enabled = [x for x in enabled_families if x and x != "price_volume_only"]
    specs = [
        FeatureSpec(name="price_volume_only", families=("price_volume",), feature_cols=PRICE_VOLUME_FEATURES)
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


def attach_enabled_families(
    dataset: pd.DataFrame, db_path: Path, cfg: M5RunConfig, enabled_families: list[str]
) -> pd.DataFrame:
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


# ── 覆盖率 & 重要性 ──────────────────────────────────────────────────────

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
            rows.append({
                "feature_spec": spec.name, "families": ",".join(spec.families),
                "feature": col, "raw_feature": raw_col,
                "rows": int(len(base)), "non_null": int(vals.notna().sum()),
                "coverage_ratio": float(vals.notna().mean()) if len(base) else np.nan,
                "candidate_pool_pass_coverage_ratio": float(vals.loc[pool_pass].notna().mean()) if pool_pass.any() else np.nan,
                "first_signal_date": str(base.loc[vals.notna(), "signal_date"].min().date()) if vals.notna().any() else "",
                "last_signal_date": str(base.loc[vals.notna(), "signal_date"].max().date()) if vals.notna().any() else "",
            })
    return pd.DataFrame(rows)


def summarize_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    out = importance.groupby(
        ["feature_spec", "feature_families", "candidate_pool_version", "model", "feature"], sort=True,
    ).agg(
        importance=("importance", "mean"),
        signed_weight=("signed_weight", "mean"),
        observations=("test_signal_date", "nunique"),
    ).reset_index()
    return out.sort_values(["feature_spec", "candidate_pool_version", "model", "importance"], ascending=[True, True, True, False])


# ── Walk-forward 打分 ────────────────────────────────────────────────────

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


def build_walk_forward_scores_for_spec(
    dataset: pd.DataFrame, spec: FeatureSpec, cfg: M5RunConfig,
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
            train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
            train_fit = _cap_fit_rows(train, max_rows=cfg.max_fit_rows, random_seed=cfg.random_seed)
            for base_model_name, model_type in model_specs:
                if base_model_name.startswith("M4_xgboost"):
                    scores, imp = _train_predict_xgboost(
                        model_name=base_model_name, model_type=model_type,
                        train=train_fit, test=test, feature_cols=feature_cols,
                        random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
                    )
                else:
                    scores, imp = _train_predict_sklearn(
                        model_name=base_model_name, model_type=model_type,
                        train=train_fit, test=test, feature_cols=feature_cols,
                        random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
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


def build_all_m5_scores(
    dataset: pd.DataFrame, specs: list[FeatureSpec], cfg: M5RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


# ── 增量 delta ───────────────────────────────────────────────────────────

def build_incremental_delta(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return pd.DataFrame()
    df = leaderboard[leaderboard["model"].astype(str).str.startswith("M5_")].copy()
    if df.empty:
        return pd.DataFrame()
    df["feature_spec"] = df["model"].astype(str).str.replace(r"^M5_", "", regex=True).str.extract(
        r"(.+)_(elasticnet_excess|logistic_top20|extratrees_excess|xgboost_excess|xgboost_top20)$", expand=False,
    )[0]
    df["base_model"] = df["model"].astype(str).str.extract(
        r"(elasticnet_excess|logistic_top20|extratrees_excess|xgboost_excess|xgboost_top20)$", expand=False,
    )
    baseline = df[df["feature_spec"] == "price_volume_only"][
        ["candidate_pool_version", "top_k", "base_model",
         "topk_excess_mean", "topk_excess_after_cost_mean", "rank_ic_mean", "quantile_top_minus_bottom_mean"]
    ].rename(columns={
        "topk_excess_mean": "baseline_topk_excess_mean",
        "topk_excess_after_cost_mean": "baseline_topk_excess_after_cost_mean",
        "rank_ic_mean": "baseline_rank_ic_mean",
        "quantile_top_minus_bottom_mean": "baseline_quantile_top_minus_bottom_mean",
    })
    out = df.merge(baseline, on=["candidate_pool_version", "top_k", "base_model"], how="left")
    for col in ["topk_excess_mean", "topk_excess_after_cost_mean", "rank_ic_mean", "quantile_top_minus_bottom_mean"]:
        out[f"delta_{col}"] = out[col] - out[f"baseline_{col}"]
    return out[out["feature_spec"] != "price_volume_only"].sort_values(
        ["top_k", "candidate_pool_version", "delta_topk_excess_after_cost_mean", "delta_rank_ic_mean"],
        ascending=[True, True, False, False],
    )
