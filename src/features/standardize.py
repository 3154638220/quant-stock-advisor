"""
因子标准化流程：截面 winsorize → 截面 z-score → 缺失填充。

面向长表（每日多标的一行），按 ``trade_date`` 分组做截面处理。
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

FillMethod = Literal["none", "zero", "cs_mean", "cs_median", "global_mean"]


def winsorize_cross_section(
    s: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """单截面向量：分位数缩尾。"""
    if lower_q < 0 or upper_q > 1 or lower_q >= upper_q:
        raise ValueError("须满足 0 <= lower_q < upper_q <= 1")
    valid = s.dropna()
    if valid.empty:
        return s
    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def winsorize_by_date(
    df: pd.DataFrame,
    col: str,
    *,
    date_col: str = "trade_date",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """按交易日分组对 ``col`` 做 winsorize，写入 ``out_col``（默认覆盖 ``col`` 或 ``{col}_wins``）。"""
    out = df.copy()
    target = out_col or f"{col}_wins"
    out[target] = out.groupby(date_col, sort=False)[col].transform(
        lambda x: winsorize_cross_section(x, lower_q=lower_q, upper_q=upper_q)
    )
    return out


def zscore_cross_section(
    s: pd.Series,
    *,
    eps: float = 1e-12,
) -> pd.Series:
    """单截面向量 z-score（``nan`` 保持 ``nan``，使用 np.nanmean/np.nanstd 统一口径）。"""
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(arr)
    if not valid.any():
        return pd.Series(np.nan, index=s.index, dtype=float)
    m = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=0))
    if not np.isfinite(sd) or sd < eps:
        out = pd.Series(0.0, index=s.index, dtype=float)
        out[~valid] = np.nan
        return out
    result = (arr - m) / sd
    result[~valid] = np.nan
    return pd.Series(result, index=s.index, dtype=float)


def zscore_by_date(
    df: pd.DataFrame,
    col: str,
    *,
    date_col: str = "trade_date",
    out_col: Optional[str] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """按交易日截面 z-score。"""
    out = df.copy()
    target = out_col or f"{col}_z"
    out[target] = out.groupby(date_col, sort=False)[col].transform(
        lambda x: zscore_cross_section(x, eps=eps)
    )
    return out


def fill_missing(
    s: pd.Series,
    method: FillMethod = "cs_median",
    *,
    global_fill: Optional[float] = None,
    by_group: Optional[pd.Series] = None,
) -> pd.Series:
    """
    缺失填充（用于因子列）。

    - ``none``：不填。
    - ``zero``：填 0。
    - ``cs_mean`` / ``cs_median``：须与 ``by_group`` 同索引的日期列对齐，按日期填截面均值/中位数。
    - ``global_mean``：全样本均值（仅当 ``global_fill`` 传入时使用，否则现算）。
    """
    if method == "none":
        return s
    if method == "zero":
        return s.fillna(0.0)
    if method == "global_mean":
        v = float(global_fill) if global_fill is not None else float(np.nanmean(s.to_numpy()))
        return s.fillna(v)
    if method in ("cs_mean", "cs_median"):
        if by_group is None:
            raise ValueError("cs_mean/cs_median 需要 by_group（与 s 同索引的日期序列）")
        df = pd.DataFrame({"v": s, "d": by_group})
        if method == "cs_mean":
            fill = df.groupby("d")["v"].transform("mean")
        else:
            fill = df.groupby("d")["v"].transform("median")
        return s.fillna(fill)
    raise ValueError(f"未知 method: {method!r}")


def factor_standardize_pipeline(
    df: pd.DataFrame,
    factor_col: str,
    *,
    date_col: str = "trade_date",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    fill: FillMethod = "cs_median",
    out_col: str = "factor_std",
) -> pd.DataFrame:
    """
    组合流程：winsorize（按日）→ z-score（按日）→ 缺失填充。

    填充在 z-score 之后执行，避免极值拉高截面矩；若希望先填再 winsorize，请分步调用本模块函数。
    """
    tmp = f"__w_{factor_col}"
    step = winsorize_by_date(
        df,
        factor_col,
        date_col=date_col,
        lower_q=lower_q,
        upper_q=upper_q,
        out_col=tmp,
    )
    step = zscore_by_date(step, tmp, date_col=date_col, out_col=out_col)
    step = step.drop(columns=[tmp], errors="ignore")
    step[out_col] = fill_missing(
        step[out_col],
        fill,
        by_group=step[date_col] if fill in ("cs_mean", "cs_median") else None,
    )
    return step
