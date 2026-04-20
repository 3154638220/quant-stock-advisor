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
