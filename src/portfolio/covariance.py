"""从日线宽表估计 Top-K 标的日收益协方差与期望收益（用于组合优化）。"""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
except ImportError:  # pragma: no cover
    LedoitWolf = None  # type: ignore


ShrinkageMethod = Literal["ledoit_wolf", "sample"]


def mean_cov_returns_from_wide(
    wide: pd.DataFrame,
    symbols: Sequence[str],
    *,
    lookback_days: int,
    ridge: float = 1e-6,
    shrinkage: ShrinkageMethod = "ledoit_wolf",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 ``wide`` 的最后若干个交易日上计算简单日收益的样本均值与协方差。

    Parameters
    ----------
    wide
        行索引为 ``symbol``，列为交易日，值为收盘价。
    symbols
        与组合行顺序一致的标的列表。
    lookback_days
        用于估计的**交易日**收益条数（使用最近 ``lookback_days + 1`` 根收盘计算
        ``lookback_days`` 条日收益）。

    Returns
    -------
    mu : ndarray, shape (n,)
        各标的日收益样本均值。
    cov : ndarray, shape (n, n)
        收缩协方差（默认 Ledoit–Wolf）或样本协方差 + ``ridge * I``（对称正定化）。
    shrinkage
        ``ledoit_wolf``：对样本协方差做 Ledoit–Wolf 收缩（样本外更稳健）；``sample``：仅用样本协方差。
    """
    syms = [str(s) for s in symbols]
    n = len(syms)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    cols = list(wide.columns)
    if len(cols) < 2:
        mu = np.zeros(n, dtype=np.float64)
        cov = np.eye(n, dtype=np.float64) * float(ridge)
        return mu, cov

    take = min(int(lookback_days) + 1, len(cols))
    widx = wide.reindex(syms)
    sl = widx.loc[:, cols[-take:]]

    ret_rows: list[np.ndarray] = []
    for i in range(n):
        row = sl.iloc[i].to_numpy(dtype=np.float64)
        prev = row[:-1]
        nxt = row[1:]
        r = np.where(
            np.isfinite(prev) & np.isfinite(nxt) & (np.abs(prev) > 1e-12),
            (nxt - prev) / prev,
            np.nan,
        )
        ret_rows.append(r)

    R = np.vstack(ret_rows)
    valid = np.all(np.isfinite(R), axis=0)
    R = R[:, valid]
    if R.shape[1] < 1:
        mu = np.zeros(n, dtype=np.float64)
        cov = np.eye(n, dtype=np.float64) * float(ridge)
        return mu, cov
    if R.shape[1] < 2:
        mu = np.mean(R, axis=1)
        vol = np.std(R, axis=1, ddof=1)
        vol = np.where(np.isfinite(vol) & (vol > 1e-12), vol, 1e-4)
        cov = np.diag(vol**2) + np.eye(n) * float(ridge)
        return mu.astype(np.float64), cov.astype(np.float64)

    mu = np.mean(R, axis=1)
    T = R.shape[1]
    if str(shrinkage).lower() == "ledoit_wolf" and LedoitWolf is not None and T >= 2 and n >= 2:
        # sklearn: 行样本为时间点，列为资产收益
        Xlw = R.T.astype(np.float64)
        lw = LedoitWolf().fit(Xlw)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
    else:
        cov = np.cov(R, rowvar=True, ddof=1)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(n) * float(ridge)
    return mu.astype(np.float64), cov.astype(np.float64)


def mean_cov_returns_from_daily_long(
    daily_df: pd.DataFrame,
    symbols: Sequence[str],
    *,
    asof: pd.Timestamp,
    lookback_days: int,
    ridge: float = 1e-6,
    shrinkage: ShrinkageMethod = "ledoit_wolf",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从长表日线估计 ``asof`` 当日之前窗口内的日收益均值与协方差（与 eval 一致）。
    """
    syms = [str(s).zfill(6) for s in symbols]
    n = len(syms)
    if n == 0 or daily_df.empty:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    sub = daily_df[daily_df["symbol"].astype(str).str.zfill(6).isin(syms)].copy()
    if sub.empty:
        mu = np.zeros(n, dtype=np.float64)
        return mu, np.eye(n, dtype=np.float64) * float(ridge)

    sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.normalize()
    end = pd.Timestamp(asof).normalize()
    sub = sub[sub["trade_date"] <= end]
    wide = sub.pivot_table(
        index="symbol",
        columns="trade_date",
        values="close",
        aggfunc="last",
    ).sort_index(axis=1)

    wide.index = wide.index.astype(str).str.zfill(6)
    return mean_cov_returns_from_wide(
        wide.reindex(syms),
        syms,
        lookback_days=lookback_days,
        ridge=ridge,
        shrinkage=shrinkage,
    )
