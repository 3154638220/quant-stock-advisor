"""基本面因子截面处理：缩尾、缺失填充与可选中性化。"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .neutralize import neutralize_size_industry_regression
from .standardize import fill_missing, winsorize_by_date

DEFAULT_FUNDAMENTAL_COLS: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "ev_ebitda",
    "roe_ttm",
    "net_profit_yoy",
    "gross_margin_change",
    "gross_margin_delta",
    "debt_to_assets_change",
    "ocf_to_net_profit",
    "ocf_to_asset",
    "asset_turnover",
    "net_margin_stability",
    "northbound_net_inflow",
    "margin_buy_ratio",
)

DAILY_VALUATION_SOURCES: tuple[str, ...] = ("stock_value_em",)


def pit_safe_fundamental_rows(
    df: pd.DataFrame,
    *,
    announcement_col: str = "announcement_date",
    report_period_col: str = "report_period",
    source_col: str = "source",
) -> pd.Series:
    """Return rows whose fundamental values were observable at announcement time.

    Statement-derived fundamentals need a real disclosure lag
    (announcement date after report period). Daily valuation snapshots are already
    dated by trading day and may use the same date for availability/report period.
    """
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    ann = pd.to_datetime(df.get(announcement_col), errors="coerce").dt.normalize()
    if report_period_col in df.columns:
        period = pd.to_datetime(df[report_period_col], errors="coerce").dt.normalize()
        has_unknown_period = period.isna()
        has_positive_notice_lag = ann > period
    else:
        has_unknown_period = pd.Series(True, index=df.index)
        has_positive_notice_lag = pd.Series(True, index=df.index)
    if source_col in df.columns:
        source = df[source_col].astype(object).where(df[source_col].notna(), "").astype(str)
    else:
        source = pd.Series("", index=df.index)
    is_daily_valuation = source.isin(DAILY_VALUATION_SOURCES)
    return ann.notna() & (is_daily_valuation | has_unknown_period | has_positive_notice_lag)


def preprocess_fundamental_cross_section(
    df: pd.DataFrame,
    *,
    factor_cols: Optional[Iterable[str]] = None,
    date_col: str = "trade_date",
    size_col: str = "log_market_cap",
    industry_col: Optional[str] = None,
    neutralize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    对基本面因子做最小预处理：
    1) 按日期分组缩尾
    2) 可选按市值/行业做截面回归中性化
    3) 缺失按日期中位数填充（兜底为 0）
    """
    out = df.copy()
    cols = [c for c in (factor_cols or DEFAULT_FUNDAMENTAL_COLS) if c in out.columns]
    if not cols:
        return out

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out = winsorize_by_date(
            out,
            c,
            date_col=date_col,
            lower_q=lower_q,
            upper_q=upper_q,
            out_col=c,
        )
        if neutralize and size_col in out.columns:
            ind_col = industry_col if (industry_col and industry_col in out.columns) else "__no_industry__"
            tmp = out.copy()
            if ind_col == "__no_industry__":
                tmp[ind_col] = "NA"
            tmp = neutralize_size_industry_regression(
                tmp,
                c,
                size_col=size_col,
                industry_col=ind_col,
                date_col=date_col,
                suffix="_fund_neut",
            )
            nc = f"{c}_fund_neut"
            if nc in tmp.columns:
                out[c] = tmp[nc]
        out[c] = fill_missing(
            out[c],
            method="cs_median",
            by_group=out[date_col],
        ).fillna(0.0)
    return out
