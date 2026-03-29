"""
因子评估：IC、RankIC、分层（分位）收益、滚动窗口稳定性。
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

Method = Literal["pearson", "spearman"]


def _safe_corr(a: pd.Series, b: pd.Series, method: Method) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return float("nan")
    return float(a[m].corr(b[m], method=method))


def information_coefficient(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    *,
    date_col: str = "trade_date",
    method: Method = "pearson",
) -> pd.Series:
    """
    逐日截面 IC：因子与前瞻收益的相关系数（Pearson 即经典 IC；Spearman 常为 RankIC 的别名之一）。

    Returns
    -------
    Series
        索引为交易日，值为当日 IC。
    """

    idx: list = []
    vals: list[float] = []
    for d, sub in df.groupby(date_col, sort=True):
        idx.append(d)
        vals.append(_safe_corr(sub[factor_col], sub[forward_ret_col], method))
    return pd.Series(vals, index=pd.Index(idx, name=date_col))


def rank_ic(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    *,
    date_col: str = "trade_date",
) -> pd.Series:
    """RankIC：逐日 Spearman 秩相关，与 ``information_coefficient(..., method='spearman')`` 等价。"""
    return information_coefficient(
        df,
        factor_col,
        forward_ret_col,
        date_col=date_col,
        method="spearman",
    )


def ic_summary(ic: pd.Series) -> pd.Series:
    """IC 序列的均值、标准差、IR（信息比率）、正比例。"""
    x = ic.dropna()
    if x.empty:
        return pd.Series(
            {"mean": np.nan, "std": np.nan, "ir": np.nan, "hit_rate": np.nan, "n": 0}
        )
    m = float(x.mean())
    s = float(x.std(ddof=0))
    ir = m / s if s > 1e-15 else float("nan")
    hit = float((x > 0).mean())
    return pd.Series(
        {"mean": m, "std": s, "ir": ir, "hit_rate": hit, "n": int(len(x))}
    )


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    *,
    date_col: str = "trade_date",
    n_quantiles: int = 5,
    labels: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    分层收益：每个交易日按因子分位分组，计算各组前瞻收益均值，再对时间平均。

    Returns
    -------
    DataFrame
        行：分位组；列：``mean_ret``、``n_days`` 等。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 须 >= 2")
    parts: list[pd.Series] = []
    for d, sub in df.groupby(date_col, sort=True):
        s = sub[[factor_col, forward_ret_col]].dropna()
        if len(s) < n_quantiles * 2:
            continue
        try:
            s = s.copy()
            s["_q"] = pd.qcut(
                s[factor_col],
                n_quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        g = s.groupby("_q", observed=True)[forward_ret_col].mean()
        parts.append(g.rename(d))
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, axis=1)
    mean_layer = panel.mean(axis=1)
    out = pd.DataFrame(
        {
            "quantile": mean_layer.index.astype(int),
            "mean_fwd_ret": mean_layer.values,
            "n_dates": [panel.loc[q].notna().sum() for q in mean_layer.index],
        }
    )
    if labels:
        out["label"] = [labels[i] if i < len(labels) else str(i) for i in out["quantile"]]
    return out


def rolling_ic_stability(
    ic: pd.Series,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    滚动窗口 IC 稳定性：滚动均值、滚动标准差、滚动 IR（均值/标准差）。

    Parameters
    ----------
    ic
        日度 IC 序列（索引为日期）。
    window
        交易日窗口长度（观测数，非日历日）。
    """
    if window < 2:
        raise ValueError("window 须 >= 2")
    mp = min_periods or max(2, window // 2)
    s = ic.sort_index()
    roll_mean = s.rolling(window=window, min_periods=mp).mean()
    roll_std = s.rolling(window=window, min_periods=mp).std(ddof=0)
    roll_ir = roll_mean / roll_std.replace(0, np.nan)
    return pd.DataFrame(
        {
            "ic": s,
            "roll_mean_ic": roll_mean,
            "roll_std_ic": roll_std,
            "roll_ir": roll_ir,
        }
    )


def long_table_from_wide(
    wide: pd.DataFrame,
    factor_name: str,
    factor_values: np.ndarray,
) -> pd.DataFrame:
    """
    将「行=标的、列=日期」因子宽表转为长表，便于与收益评估拼接。

    ``factor_values`` 形状 ``(len(symbols), len(dates))``，与 ``wide`` 行列对齐。
    """
    w = pd.DataFrame(factor_values, index=wide.index, columns=wide.columns)
    ser = w.stack()
    ser.name = factor_name
    out = ser.reset_index()
    out.columns = ["symbol", "trade_date", factor_name]
    return out
