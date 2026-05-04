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

from src.features.fundamental_factors import (
    LOW_COVERAGE_THRESHOLD,
    filter_low_coverage_cols,
    pit_safe_fundamental_rows,
)
from src.features.registry import (  # D1: 统一因子注册中心（单一权威来源）
    FUND_FLOW_FEATURES_REGISTRY,
    FUNDAMENTAL_FEATURES_REGISTRY,
    INDUSTRY_BREADTH_FEATURES_REGISTRY,
    PRICE_VOLUME_FEATURES_REGISTRY,
    SHAREHOLDER_FEATURES_REGISTRY,
)
from src.pipeline.monthly_baselines import (
    _train_predict_sklearn,
    _train_predict_xgboost,
    valid_pool_frame,
)

# ── 特征列常量（D1: 迁移至 FeatureRegistry 单一来源）─────────────────────
# 以下常量现在是 registry 生成的别名，保留旧名称以保持向后兼容。

PRICE_VOLUME_FEATURES: tuple[str, ...] = PRICE_VOLUME_FEATURES_REGISTRY
INDUSTRY_BREADTH_RAW_FEATURES: tuple[str, ...] = INDUSTRY_BREADTH_FEATURES_REGISTRY
FUND_FLOW_RAW_FEATURES: tuple[str, ...] = FUND_FLOW_FEATURES_REGISTRY
FUNDAMENTAL_RAW_FEATURES: tuple[str, ...] = FUNDAMENTAL_FEATURES_REGISTRY
SHAREHOLDER_RAW_FEATURES: tuple[str, ...] = SHAREHOLDER_FEATURES_REGISTRY


# ── P1-2: 特征家族数据起始日期（用于 walk-forward 分期训练）─────────────

FAMILY_DATA_START: dict[str, str] = {
    "price_volume": "2018-01-01",
    "industry_breadth": "2018-01-01",
    "fundamental": "2018-01-01",
    "shareholder": "2019-01-01",
    "fund_flow": "2023-10-01",   # 实际数据起始日
}

_FAMILY_FEATURE_PREFIX: dict[str, str] = {
    "price_volume": "feature_ret_|feature_realized_|feature_amount_|feature_turnover_|feature_price_position_|feature_limit_move_",
    "industry_breadth": "feature_industry_",
    "fund_flow": "feature_fund_flow_",
    "fundamental": "feature_fundamental_",
    "shareholder": "feature_shareholder_",
}


def _infer_feature_family(feature_name: str) -> str:
    """根据特征名前缀推断其所属家族。"""
    import re
    for family, pattern in _FAMILY_FEATURE_PREFIX.items():
        if re.search(pattern, feature_name):
            return family
    return "price_volume"


def get_active_features_for_fold(
    all_features: list[str],
    fold_train_end: pd.Timestamp,
    family_start: dict[str, str] | None = None,
) -> list[str]:
    """P1-2: 按 fold 训练末尾日期过滤不可用特征。

    fund_flow 数据始于 2023-10，当 fold train_end < 2023-10-01 时，
    排除所有 fund_flow z-score 列。is_missing 标志始终保留。
    """
    fm_start = family_start or FAMILY_DATA_START
    active: list[str] = []
    for feat in all_features:
        if feat.startswith("is_missing_"):
            active.append(feat)  # 缺失标志始终可用
            continue
        family = _infer_feature_family(feat)
        start_str = fm_start.get(family, "2000-01-01")
        start = pd.Timestamp(start_str)
        if fold_train_end >= start:
            active.append(feat)
    return active


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
    # P0-1: 已弃用，保留向后兼容。新逻辑优先使用实际披露日历，
    # 缺失时回退为 pit_fallback_lag_days（默认 45 天，比旧的 30 天更保守）。
    availability_lag_days: int = 45
    pit_fallback_lag_days: int = 45
    # P1-1: XGBoost 超参数时序感知调优
    hpo_enabled: bool = False
    hpo_n_trials: int = 30
    hpo_cv_folds: int = 3
    # P1-2: Walk-forward 窗口配置
    window_type: str = "expanding"  # "rolling" | "expanding"
    halflife_months: float = 36.0   # 扩张窗口样本半衰期（月），0 表示等权
    ml_models: tuple[str, ...] = (
        "elasticnet",
        "extratrees",
        "xgboost_excess",
        # "logistic",         # DEPRECATED P1-4: after-cost excess 长期为负，移入 ablation
        # "xgboost_top20",    # DEPRECATED P1-4: 同上
    )
    model_n_jobs: int = 0
    # P0-1: 对基本面因子使用行业内 z-score 中性化（默认 False，保持向后兼容）
    use_industry_neutral_zscore: bool = False
    # P2-1: z-score 后应用 rank transform（将截面秩映射到 [-1,1]，缓解上限堆积）
    use_rank_transform: bool = False
    # D2: IC 衰减因子自动排除（per docs/plan-05-04.md）
    ic_decay_enabled: bool = True
    ic_decay_window: int = 20
    ic_decay_threshold: float = 0.02


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
    use_rank_transform: bool = False,
) -> pd.DataFrame:
    """截面 z-score + 缺失标志，可选 rank transform（P2-1）。"""
    out = dataset.copy()
    for col in raw_cols:
        if col not in out.columns:
            out[col] = np.nan
        vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"is_missing_{col}"] = vals.isna().astype(int)
        z_col = f"{col}_z"
        out[z_col] = vals.groupby(out[date_col], sort=False).transform(_winsor_zscore)
        # P2-1: 可选秩变换，缓解 z-score 上限堆积
        if use_rank_transform:
            out[z_col] = out.groupby(date_col, sort=False)[z_col].transform(_rank_transform)
    return out


def industry_neutral_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    date_col: str = "signal_date",
    industry_col: str = "industry_level1",
) -> pd.DataFrame:
    """P0-1: 行业内 z-score 中性化。

    对每个 (signal_date, industry) 分组，将 feature_cols 做 z-score。
    消除行业间特征分布差异，防止高 asset_turnover 行业（如证券）系统性排名靠前。

    返回新增列 {col}_ind_z 的 DataFrame。
    """
    out = df.copy()
    if industry_col not in out.columns:
        return out
    for col in feature_cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_ind_z"] = (
            vals.groupby([out[date_col], out[industry_col]], sort=False, group_keys=False)
            .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
            .clip(-5, 5)
        )
    return out


def _rank_transform(series: pd.Series) -> pd.Series:
    """P2-1: 将 z-score 转换为截面秩分（均匀分布于 [-1, 1]）。

    对极值堆积的 z-score 分布更鲁棒。
    """
    n = series.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=series.index)
    ranked = series.rank(method="average", na_option="keep")
    return ((ranked - 1) / (n - 1) * 2 - 1).clip(-1, 1)


# ── D2: IC 衰减因子自动排除 ─────────────────────────────────────────────

# 特征家族原始列名集合（用于 IC 衰减检查时的因子名匹配）
_IC_DECAY_CHECK_FEATURES: tuple[str, ...] = (
    PRICE_VOLUME_FEATURES
    + INDUSTRY_BREADTH_RAW_FEATURES
    + FUND_FLOW_RAW_FEATURES
    + FUNDAMENTAL_RAW_FEATURES
    + SHAREHOLDER_RAW_FEATURES
)


def _get_ic_decayed_features(
    dataset: pd.DataFrame,
    db_path: Path,
    *,
    window: int = 20,
    threshold: float = 0.02,
) -> list[str]:
    """检查 IC Monitor 中的因子衰减状态，返回已衰减的特征列名列表。

    衰减因子的 ``_z`` 列将被置为 NaN，使其在后续模型训练中被自然排除。
    若 IC Monitor 数据不可用（无 DuckDB 或无 IC 记录），返回空列表。

    Parameters
    ----------
    dataset : DataFrame
        已附加全部特征的月度数据集。
    db_path : Path
        DuckDB 数据库路径（IC Monitor 存储于此）。
    window : int
        IC 滚动窗口大小（交易日数）。
    threshold : float
        |滚动 IC 均值| 低于此值时视为衰减。

    Returns
    -------
    list[str]
        已衰减的特征列名（原始列名，非 _z 后缀）。
    """
    if not db_path.exists():
        return []

    # 确定需要检查的特征列（仅检查数据集中实际存在的列）
    candidate_features = [c for c in _IC_DECAY_CHECK_FEATURES if c in dataset.columns]
    if not candidate_features:
        return []

    try:
        from src.features.ic_monitor import ICMonitor

        monitor = ICMonitor(db_path=db_path)
        decayed = monitor.get_decayed_factors(
            window=window, threshold=threshold, factors=candidate_features,
        )
        monitor.close()
    except Exception:
        # IC Monitor 数据不可用时静默跳过（如首次运行尚无 IC 记录）
        return []

    # 将衰减因子的 _z 列置为 NaN，使其在后续训练中被排除
    actual_decayed: list[str] = []
    for feat in decayed:
        z_col = f"{feat}_z"
        if z_col in dataset.columns:
            dataset[z_col] = np.nan
            actual_decayed.append(feat)
        # 同时处理行业内中性化列 _ind_z
        ind_z_col = f"{feat}_ind_z"
        if ind_z_col in dataset.columns:
            dataset[ind_z_col] = np.nan

    return actual_decayed


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


def _filter_pit_safe_fundamental_rows(
    raw: pd.DataFrame,
    *,
    signal_date: pd.Timestamp | None = None,
    fallback_lag_days: int = 45,
    disclosure_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    disclosure_calendar: pd.DataFrame | None = None,
    fallback_lag_days: int = 45,
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
    # P0-1: 按信号日逐批过滤，优先使用实际披露日历
    raw = _filter_pit_safe_fundamental_rows(
        raw, fallback_lag_days=fallback_lag_days, disclosure_calendar=disclosure_calendar,
    )
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

    # P0-2: 排除覆盖率过低的特征列（如 ev_ebitda 覆盖率=0），仅保留 is_missing 标志
    active_fundamental, dropped = filter_low_coverage_cols(
        out, list(FUNDAMENTAL_RAW_FEATURES),
        threshold=LOW_COVERAGE_THRESHOLD,
    )
    if dropped:
        import warnings
        warnings.warn(
            f"P0-2: 基本面特征覆盖率过低已排除: {dropped}，阈值={LOW_COVERAGE_THRESHOLD}",
            RuntimeWarning,
        )

    return add_zscore_and_missing_flags(out, tuple(active_fundamental))


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

def build_feature_specs(
    enabled_families: list[str],
    *,
    use_industry_neutral_zscore: bool = False,
) -> list[FeatureSpec]:
    """构建增量 FeatureSpec 列表。

    Parameters
    ----------
    use_industry_neutral_zscore : bool
        P0-1: 若为 True，基本面因子使用 _ind_z（行业内 z-score）替代 _z（全截面 z-score）。
    """
    enabled = [x for x in enabled_families if x and x != "price_volume_only"]
    specs = [
        FeatureSpec(name="price_volume_only", families=("price_volume",), feature_cols=PRICE_VOLUME_FEATURES)
    ]
    cumulative: list[str] = list(PRICE_VOLUME_FEATURES)
    family_cols: dict[str, tuple[str, ...]] = {
        "industry_breadth": tuple(f"{c}_z" for c in INDUSTRY_BREADTH_RAW_FEATURES),
        "fund_flow": tuple(f"{c}_z" for c in FUND_FLOW_RAW_FEATURES),
        "fundamental": tuple(
            f"{c}_ind_z" if use_industry_neutral_zscore else f"{c}_z"
            for c in FUNDAMENTAL_RAW_FEATURES
        ),
        "shareholder": tuple(f"{c}_z" for c in SHAREHOLDER_RAW_FEATURES),
    }
    family_order = ["industry_breadth", "fund_flow", "fundamental", "shareholder"]
    active_families = ["price_volume"]
    for family in family_order:
        if family not in enabled:
            continue
        cumulative.extend(family_cols[family])
        active_families.append(family)
        suffix = "_ind" if use_industry_neutral_zscore and family == "fundamental" else ""
        specs.append(
            FeatureSpec(
                name="plus_" + "_plus_".join(active_families[1:]) + suffix,
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
        out = attach_fundamental_features(
            out, db_path,
            fallback_lag_days=cfg.pit_fallback_lag_days if hasattr(cfg, 'pit_fallback_lag_days') else cfg.availability_lag_days,
        )
    if "shareholder" in enabled_families:
        out = attach_shareholder_features(out, db_path, availability_lag_days=cfg.availability_lag_days)

    # P0-1: 可选行业内 z-score 中性化（生成 _ind_z 列）
    if getattr(cfg, 'use_industry_neutral_zscore', False) and "fundamental" in enabled_families:
        fundamental_z_cols = [f"{c}_z" for c in FUNDAMENTAL_RAW_FEATURES if f"{c}_z" in out.columns]
        if fundamental_z_cols and "industry_level1" in out.columns:
            out = industry_neutral_zscore(out, fundamental_z_cols)

    # P2-1: 可选 rank transform（对已有 _z 列施加秩变换）
    if getattr(cfg, 'use_rank_transform', False):
        all_z_cols = [c for c in out.columns if c.endswith("_z")]
        if all_z_cols:
            for col in all_z_cols:
                out[col] = out.groupby("signal_date", sort=False)[col].transform(_rank_transform)

    # D2: IC 衰减因子自动排除（per docs/plan-05-04.md）
    # 检查 IC Monitor 中的因子衰减状态，自动将衰减因子的 _z 值置 NaN，
    # 使其在后续训练中被自然排除。
    ic_decay_window = getattr(cfg, 'ic_decay_window', 20)
    ic_decay_threshold = getattr(cfg, 'ic_decay_threshold', 0.02)
    if getattr(cfg, 'ic_decay_enabled', True):
        decayed = _get_ic_decayed_features(
            out, db_path, window=ic_decay_window, threshold=ic_decay_threshold,
        )
        if decayed:
            import warnings
            warnings.warn(
                f"D2: IC 衰减因子已自动排除 (window={ic_decay_window}, "
                f"threshold={ic_decay_threshold}): {sorted(decayed)}",
                RuntimeWarning,
            )

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
            if str(cfg.window_type).lower() == "rolling":
                prev_months = [m for m in months if m < test_month]
                keep_months = prev_months[-int(cfg.min_train_months):]
                train = pool_df[pool_df["signal_date"].isin(keep_months)].copy()
            else:
                train = pool_df[pool_df["signal_date"] < test_month].copy()
            test = pool_df[pool_df["signal_date"] == test_month].copy()
            if train["signal_date"].nunique() < cfg.min_train_months or len(train) < cfg.min_train_rows or test.empty:
                continue
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
                        model_name=base_model_name, model_type=model_type,
                        train=train_fit, test=test, feature_cols=feature_cols,
                        random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
                        hpo_params=hpo_params,
                        sample_weight=sample_weight,
                    )
                else:
                    scores, imp = _train_predict_sklearn(
                        model_name=base_model_name, model_type=model_type,
                        train=train_fit, test=test, feature_cols=feature_cols,
                        random_seed=cfg.random_seed, model_n_jobs=cfg.model_n_jobs,
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
