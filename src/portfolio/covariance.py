"""从日线宽表估计 Top-K 标的日收益协方差与期望收益（用于组合优化）。"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
except ImportError:  # pragma: no cover
    LedoitWolf = None  # type: ignore


CovarianceMethod = Literal["ledoit_wolf", "sample", "ewma", "industry_factor"]


def _returns_matrix_from_wide(
    wide: pd.DataFrame,
    symbols: Sequence[str],
    *,
    lookback_days: int,
) -> np.ndarray:
    """将宽表收盘价转为收益率矩阵 R（shape: n_assets x T）。"""
    syms = [str(s) for s in symbols]
    n = len(syms)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cols = list(wide.columns)
    if len(cols) < 2:
        return np.zeros((n, 0), dtype=np.float64)

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
    return R[:, valid]


def _ewma_covariance(R: np.ndarray, *, halflife: float, ridge: float) -> np.ndarray:
    """指数加权协方差（近期样本权重更高）。"""
    n, t = R.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if t <= 1:
        vol = np.std(R, axis=1, ddof=0)
        vol = np.where(np.isfinite(vol) & (vol > 1e-12), vol, 1e-4)
        return np.diag(vol**2) + np.eye(n, dtype=np.float64) * float(ridge)
    hl = float(max(halflife, 1.0))
    decay = float(np.exp(np.log(0.5) / hl))
    # 越近的样本权重越大
    raw = decay ** np.arange(t - 1, -1, -1, dtype=np.float64)
    raw /= raw.sum()
    mu_w = R @ raw
    xc = R - mu_w[:, None]
    cov = np.zeros((n, n), dtype=np.float64)
    for i in range(t):
        xi = xc[:, i : i + 1]
        cov += raw[i] * (xi @ xi.T)
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(n, dtype=np.float64) * float(ridge)
    return cov


def _industry_factor_covariance(
    R: np.ndarray,
    industry_labels: Sequence[str],
    *,
    ridge: float,
) -> np.ndarray:
    """
    Barra 风格的行业因子协方差近似：Σ = B F B' + D。
    其中 B 为行业虚拟变量暴露矩阵，F 为行业因子协方差，D 为特异风险对角阵。
    """
    n, t = R.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if t <= 1:
        vol = np.std(R, axis=1, ddof=0)
        vol = np.where(np.isfinite(vol) & (vol > 1e-12), vol, 1e-4)
        return np.diag(vol**2) + np.eye(n, dtype=np.float64) * float(ridge)

    labels = np.asarray([str(x) if x is not None else "_NA_" for x in industry_labels])
    if labels.size != n:
        raise ValueError("industry_labels 长度须与 symbols 一致")
    uniq = np.unique(labels)
    k = int(uniq.size)
    if k <= 1:
        # 单行业退化为样本协方差
        cov = np.cov(R, rowvar=True, ddof=1)
        cov = 0.5 * (cov + cov.T) + np.eye(n) * float(ridge)
        return cov.astype(np.float64)

    B = np.zeros((n, k), dtype=np.float64)
    for j, g in enumerate(uniq):
        B[:, j] = (labels == g).astype(np.float64)
    # 行标准化，避免空行或数值放大
    row_sum = B.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    B = B / row_sum

    BtB = B.T @ B
    BtB_reg = BtB + np.eye(k, dtype=np.float64) * 1e-8
    inv_btbr = np.linalg.inv(BtB_reg)
    P = inv_btbr @ B.T  # (k, n)

    F_ts = np.zeros((k, t), dtype=np.float64)
    E = np.zeros((n, t), dtype=np.float64)
    for i in range(t):
        rt = R[:, i]
        ft = P @ rt
        et = rt - B @ ft
        F_ts[:, i] = ft
        E[:, i] = et
    F_cov = np.cov(F_ts, rowvar=True, ddof=1)
    if np.ndim(F_cov) == 0:
        F_cov = np.array([[float(F_cov)]], dtype=np.float64)
    spec_var = np.var(E, axis=1, ddof=1)
    spec_var = np.where(np.isfinite(spec_var) & (spec_var > 1e-12), spec_var, 1e-8)
    D = np.diag(spec_var)

    cov = B @ F_cov @ B.T + D
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(n, dtype=np.float64) * float(ridge)
    return cov


def mean_cov_returns_from_wide(
    wide: pd.DataFrame,
    symbols: Sequence[str],
    *,
    lookback_days: int,
    ridge: float = 1e-6,
    shrinkage: CovarianceMethod = "ledoit_wolf",
    ewma_halflife: float = 20.0,
    industry_labels: Optional[Sequence[str]] = None,
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
        ``ledoit_wolf``：样本协方差做 Ledoit–Wolf 收缩；
        ``sample``：样本协方差；
        ``ewma``：指数加权协方差（近期权重更高）；
        ``industry_factor``：行业因子分解协方差（需 ``industry_labels``）。
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

    R = _returns_matrix_from_wide(wide, syms, lookback_days=lookback_days)
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
    method = str(shrinkage).lower()
    if method == "ledoit_wolf" and LedoitWolf is not None and T >= 2 and n >= 2:
        # sklearn: 行样本为时间点，列为资产收益
        Xlw = R.T.astype(np.float64)
        lw = LedoitWolf().fit(Xlw)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
    elif method == "ewma":
        cov = _ewma_covariance(R, halflife=ewma_halflife, ridge=ridge)
    elif method == "industry_factor":
        if industry_labels is None:
            raise ValueError("shrinkage=industry_factor 需要 industry_labels")
        cov = _industry_factor_covariance(R, industry_labels, ridge=ridge)
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
    shrinkage: CovarianceMethod = "ledoit_wolf",
    ewma_halflife: float = 20.0,
    industry_col: Optional[str] = None,
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
    industry_labels: Optional[list[str]] = None
    if str(shrinkage).lower() == "industry_factor":
        if industry_col is None or industry_col not in sub.columns:
            raise ValueError("shrinkage=industry_factor 需要 daily_df 提供 industry_col")
        latest_ind = (
            sub.sort_values(["symbol", "trade_date"])
            .groupby("symbol", as_index=False)[industry_col]
            .last()
        )
        ind_map = dict(
            zip(
                latest_ind["symbol"].astype(str).str.zfill(6),
                latest_ind[industry_col].astype(str).fillna("_NA_"),
            )
        )
        industry_labels = [str(ind_map.get(s, "_NA_")) for s in syms]

    return mean_cov_returns_from_wide(
        wide.reindex(syms),
        syms,
        lookback_days=lookback_days,
        ridge=ridge,
        shrinkage=shrinkage,
        ewma_halflife=ewma_halflife,
        industry_labels=industry_labels,
    )
