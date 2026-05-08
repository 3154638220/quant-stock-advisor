"""Shared PIT-safe feature family loaders for monthly pipelines.

The public entry point is :class:`DataLoader`.  Pipeline modules declare the
feature families they need, and the loader applies them in the canonical M5
order while preserving each family's PIT behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.features.fundamental_factors import (
    LOW_COVERAGE_THRESHOLD,
    filter_low_coverage_cols,
    pit_safe_fundamental_rows,
)
from src.features.registry import (
    CONCEPT_FEATURES_REGISTRY,
    FUND_FLOW_FEATURES_REGISTRY,
    FUNDAMENTAL_FEATURES_REGISTRY,
    INDUSTRY_BREADTH_FEATURES_REGISTRY,
    LHB_FEATURES_REGISTRY,
    MARGIN_TRADING_FEATURES_REGISTRY,
    NORTHBOUND_FEATURES_REGISTRY,
    NORTHBOUND_REGIME_FEATURES_REGISTRY,
    SHAREHOLDER_FEATURES_REGISTRY,
)
from src.features.standardize import winsor_zscore

FeatureAttachFunc = Callable[[pd.DataFrame], pd.DataFrame]

INDUSTRY_BREADTH_RAW_FEATURES: tuple[str, ...] = INDUSTRY_BREADTH_FEATURES_REGISTRY
FUND_FLOW_RAW_FEATURES: tuple[str, ...] = FUND_FLOW_FEATURES_REGISTRY
FUNDAMENTAL_RAW_FEATURES: tuple[str, ...] = FUNDAMENTAL_FEATURES_REGISTRY
SHAREHOLDER_RAW_FEATURES: tuple[str, ...] = SHAREHOLDER_FEATURES_REGISTRY
NORTHBOUND_RAW_FEATURES: tuple[str, ...] = NORTHBOUND_FEATURES_REGISTRY
NORTHBOUND_REGIME_RAW_FEATURES: tuple[str, ...] = NORTHBOUND_REGIME_FEATURES_REGISTRY
MARGIN_TRADING_RAW_FEATURES: tuple[str, ...] = MARGIN_TRADING_FEATURES_REGISTRY
CONCEPT_RAW_FEATURES: tuple[str, ...] = CONCEPT_FEATURES_REGISTRY
LHB_RAW_FEATURES: tuple[str, ...] = LHB_FEATURES_REGISTRY

CANONICAL_FAMILY_ORDER: tuple[str, ...] = (
    "industry_breadth",
    "fund_flow",
    "fundamental",
    "shareholder",
    "northbound",
    "northbound_regime",
    "margin_trading",
    "concept",
    "lhb",
)

FAMILY_RAW_FEATURES: dict[str, tuple[str, ...]] = {
    "industry_breadth": INDUSTRY_BREADTH_RAW_FEATURES,
    "fund_flow": FUND_FLOW_RAW_FEATURES,
    "fundamental": FUNDAMENTAL_RAW_FEATURES,
    "shareholder": SHAREHOLDER_RAW_FEATURES,
    "northbound": NORTHBOUND_RAW_FEATURES,
    "northbound_regime": NORTHBOUND_REGIME_RAW_FEATURES,
    "margin_trading": MARGIN_TRADING_RAW_FEATURES,
    "concept": CONCEPT_RAW_FEATURES,
    "lhb": LHB_RAW_FEATURES,
}


@dataclass(frozen=True)
class DataLoaderConfig:
    """Common settings used by PIT-safe feature attachment."""

    availability_lag_days: int = 30
    pit_fallback_lag_days: int = 45
    disclosure_calendar: pd.DataFrame | None = None
    strict_unknown_families: bool = True


class DataLoader:
    """Attach configured feature families to a monthly signal dataset."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        config: DataLoaderConfig | None = None,
        registry: Mapping[str, FeatureAttachFunc] | None = None,
        family_order: Sequence[str] = CANONICAL_FAMILY_ORDER,
    ) -> None:
        self.db_path = Path(db_path)
        self.config = config or DataLoaderConfig()
        self.family_order = tuple(family_order)
        self.registry: dict[str, FeatureAttachFunc] = self._default_registry()
        if registry:
            self.registry.update(dict(registry))

    def attach(self, dataset: pd.DataFrame, families: Sequence[str]) -> pd.DataFrame:
        requested = {family for family in families if family and family != "price_volume_only"}
        unknown = requested.difference(self.registry)
        if unknown and self.config.strict_unknown_families:
            raise ValueError(f"Unknown feature families: {sorted(unknown)}")

        out = dataset.copy()
        for family in self.family_order:
            if family not in requested:
                continue
            attach = self.registry.get(family)
            if attach is None:
                continue
            out = attach(out)
        return out

    def _default_registry(self) -> dict[str, FeatureAttachFunc]:
        return {
            "industry_breadth": self.attach_industry_breadth,
            "fund_flow": self.attach_fund_flow,
            "fundamental": self.attach_fundamental,
            "shareholder": self.attach_shareholder,
            "northbound": self.attach_northbound,
            "northbound_regime": self.attach_northbound_regime,
            "margin_trading": self.attach_margin_trading,
            "concept": self.attach_concept,
            "lhb": self.attach_lhb,
        }

    def attach_industry_breadth(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return attach_industry_breadth_features(dataset)

    def attach_fund_flow(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return attach_fund_flow_features(dataset, self.db_path)

    def attach_fundamental(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return attach_fundamental_features(
            dataset,
            self.db_path,
            disclosure_calendar=self.config.disclosure_calendar,
            fallback_lag_days=self.config.pit_fallback_lag_days,
        )

    def attach_shareholder(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return attach_shareholder_features(
            dataset,
            self.db_path,
            availability_lag_days=self.config.availability_lag_days,
        )

    def attach_northbound(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from src.features.northbound_factors import attach_northbound_features

        return attach_northbound_features(dataset, str(self.db_path))

    def attach_northbound_regime(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from src.features.northbound_regime_factors import attach_northbound_regime_features

        return attach_northbound_regime_features(dataset, str(self.db_path))

    def attach_margin_trading(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from src.features.margin_trading_factors import attach_margin_trading_features

        return attach_margin_trading_features(dataset, str(self.db_path))

    def attach_concept(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from src.features.concept_factors import attach_concept_features

        return attach_concept_features(dataset, str(self.db_path))

    def attach_lhb(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from src.features.lhb_factors import attach_lhb_features

        return attach_lhb_features(dataset, str(self.db_path))


def _normalize_symbol_date(df: pd.DataFrame, *, date_col: str = "signal_date") -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize().astype("datetime64[ns]")
    return out


def _rank_transform(series: pd.Series) -> pd.Series:
    n = series.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=series.index)
    ranked = series.rank(method="average", na_option="keep")
    return ((ranked - 1) / (n - 1) * 2 - 1).clip(-1, 1)


def add_zscore_and_missing_flags(
    dataset: pd.DataFrame,
    raw_cols: tuple[str, ...],
    *,
    date_col: str = "signal_date",
    use_rank_transform: bool = False,
) -> pd.DataFrame:
    """Attach cross-sectional z-score columns and missing flags."""

    out = dataset.copy()
    for col in raw_cols:
        if col not in out.columns:
            out[col] = np.nan
        vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"is_missing_{col}"] = vals.isna().astype(int)
        z_col = f"{col}_z"
        out[z_col] = vals.groupby(out[date_col], sort=False).transform(lambda s: winsor_zscore(s, clip_val=5.0))
        if use_rank_transform:
            out[z_col] = out.groupby(date_col, sort=False)[z_col].transform(_rank_transform)
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


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
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
    signal: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    left_date: str,
    right_date: str,
) -> pd.DataFrame:
    if signal.empty:
        return signal.copy()
    if raw.empty:
        return signal.copy()
    left = _normalize_symbol_date(signal, date_col=left_date).sort_values(
        [left_date, "symbol"], kind="mergesort"
    )
    right = _normalize_symbol_date(raw, date_col=right_date).sort_values(
        [right_date, "symbol"], kind="mergesort"
    )
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


def attach_industry_breadth_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Use same-signal-date industry cross-section strength as breadth features."""

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
        feature_industry_low_vol20_mean=(
            "feature_realized_vol_20d",
            lambda s: -pd.to_numeric(s, errors="coerce").mean(),
        ),
    ).reset_index()
    out = dataset.merge(ind, on=["signal_date", "industry_level1"], how="left")
    return add_zscore_and_missing_flags(out, INDUSTRY_BREADTH_RAW_FEATURES)


def attach_fund_flow_features(
    dataset: pd.DataFrame,
    db_path: Path,
    *,
    table: str = "a_share_fund_flow",
) -> pd.DataFrame:
    signal = dataset[["signal_date", "symbol"]].drop_duplicates(["signal_date", "symbol"]).copy()
    raw = _read_table_if_exists(
        db_path,
        table,
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
    dataset: pd.DataFrame,
    db_path: Path,
    *,
    table: str = "a_share_fundamental",
    disclosure_calendar: pd.DataFrame | None = None,
    fallback_lag_days: int = 45,
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
    raw = _filter_pit_safe_fundamental_rows(
        raw,
        fallback_lag_days=fallback_lag_days,
        disclosure_calendar=disclosure_calendar,
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
        left_date="signal_date",
        right_date="_fund_announcement_date",
    )
    attached = attached.drop(columns=["_fund_announcement_date"], errors="ignore")
    out = dataset.merge(attached, on=["signal_date", "symbol"], how="left")

    active_fundamental, dropped = filter_low_coverage_cols(
        out,
        list(FUNDAMENTAL_RAW_FEATURES),
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
    raw["feature_shareholder_holder_change_rate"] = pd.to_numeric(raw["holder_change"], errors="coerce") / (
        holder_count.replace(0, np.nan)
    )
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


def attach_feature_families(
    dataset: pd.DataFrame,
    db_path: str | Path,
    families: Sequence[str],
    *,
    config: DataLoaderConfig | None = None,
    registry: Mapping[str, FeatureAttachFunc] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for one-shot family attachment."""

    return DataLoader(db_path, config=config, registry=registry).attach(dataset, families)
