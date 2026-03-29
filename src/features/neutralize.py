"""
截面中性化：全市场去均值、行业内去均值（行业中性化前后对比用）。

输入为长表 ``DataFrame``，须含 ``trade_date``（或自定义日期列）与因子列；
行业版另须 ``industry``（或自定义分组列），可为申万一级等类别编码。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def neutralize_cross_section(
    df: pd.DataFrame,
    factor_col: str,
    *,
    date_col: str = "trade_date",
    suffix: str = "_cs_neutral",
) -> pd.DataFrame:
    """
    每个交易日截面上对因子做去均值（不控制行业）。

    Returns
    -------
    DataFrame
        原表副本，并增加 ``{factor_col}{suffix}`` 列。
    """
    out = df.copy()
    name = f"{factor_col}{suffix}"
    g = out.groupby(date_col, sort=False)[factor_col]
    out[name] = g.transform(lambda s: s - s.mean())
    return out


def neutralize_industry(
    df: pd.DataFrame,
    factor_col: str,
    *,
    industry_col: str = "industry",
    date_col: str = "trade_date",
    suffix: str = "_ind_neutral",
) -> pd.DataFrame:
    """
    每个 ``(trade_date, industry)`` 组内对因子去均值，即常见「行业中性」近似。
    """
    out = df.copy()
    name = f"{factor_col}{suffix}"
    keys = [date_col, industry_col]
    if industry_col not in out.columns:
        raise ValueError(f"缺少行业列: {industry_col!r}")
    g = out.groupby(keys, sort=False)[factor_col]
    out[name] = g.transform(lambda s: s - s.mean())
    return out


def neutralize_size_industry_regression(
    df: pd.DataFrame,
    factor_col: str,
    *,
    size_col: str = "log_market_cap",
    industry_col: str = "industry",
    date_col: str = "trade_date",
    suffix: str = "_si_neutral",
) -> pd.DataFrame:
    """
    市值 + 行业双重中性化（横截面回归取残差）。

    对每个交易日截面做回归：
        ``Factor_i = β_1 * Industry_dummies + β_2 * log(Size_i) + ε_i``
    取残差 ε_i 作为中性化后的因子。

    这样，动量因子反映的是「在同等市值和同行业中表现较好的股票」，
    而不是「全市场里绝对盘子最小、炒作最猛的妖股」。
    """
    import numpy as np

    out = df.copy()
    target_col = f"{factor_col}{suffix}"

    has_size = size_col in out.columns
    has_industry = industry_col in out.columns

    if not has_size and not has_industry:
        out[target_col] = out.groupby(date_col, sort=False)[factor_col].transform(
            lambda s: s - s.mean()
        )
        return out

    residuals = pd.Series(index=out.index, dtype=float)

    for _, grp in out.groupby(date_col, sort=False):
        idx = grp.index
        y = pd.to_numeric(grp[factor_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(y)

        design_cols = []
        if has_size:
            sz = pd.to_numeric(grp[size_col], errors="coerce").to_numpy(dtype=np.float64)
            sz_valid = np.isfinite(sz)
            sz = np.where(sz_valid, sz, 0.0)
            valid &= sz_valid
            design_cols.append(sz)

        if has_industry:
            ind = grp[industry_col].astype(str).fillna("_NA_")
            dummies = pd.get_dummies(ind, drop_first=True, dtype=np.float64)
            for c in dummies.columns:
                design_cols.append(dummies[c].to_numpy(dtype=np.float64))

        if not design_cols or valid.sum() < 3:
            residuals.loc[idx] = y - np.nanmean(y)
            continue

        X = np.column_stack(design_cols)
        ones = np.ones((len(y), 1), dtype=np.float64)
        X = np.hstack([ones, X])

        m = valid
        if m.sum() < X.shape[1] + 1:
            residuals.loc[idx] = y - np.nanmean(y[m]) if m.any() else y
            continue

        try:
            beta, _, _, _ = np.linalg.lstsq(X[m], y[m], rcond=None)
            pred = X @ beta
            res = y - pred
        except np.linalg.LinAlgError:
            res = y - np.nanmean(y[m])

        residuals.loc[idx] = res

    out[target_col] = residuals
    return out


def attach_neutralized_pair(
    df: pd.DataFrame,
    factor_col: str,
    *,
    industry_col: Optional[str] = None,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    同时附加截面中性、可选行业中性列，便于对比「中性化前 / 行业中性后」。

    - 中性化前：使用原始 ``factor_col``。
    - 行业中性后：若提供 ``industry_col`` 则增加 ``*_ind_neutral``；始终增加 ``*_cs_neutral``。
    """
    out = neutralize_cross_section(df, factor_col, date_col=date_col)
    if industry_col is not None:
        out = neutralize_industry(
            out,
            factor_col,
            industry_col=industry_col,
            date_col=date_col,
        )
    return out
