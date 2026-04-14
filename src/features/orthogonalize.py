"""
因子矩阵正交化（Orthogonalization）。

在将因子传入 XGBoost 或线性模型前，对因子矩阵进行正交化，
剥离高度相关因子之间的冗余，提升模型的区分度与稳定性。

支持两种方法：
1. 对称正交化（Symmetric / Löwdin Orthogonalization）
   - 最大化与原始因子的总体相似度（Frobenius 范数意义下最近）
   - 所有正交化后的因子地位等价，无先后顺序之分
   - 推荐在因子地位对称时使用（如传入 XGBoost 前）

2. Gram-Schmidt 正交化
   - 严格按列顺序逐步正交，先出现的因子保持原始信息更多
   - 推荐在因子有优先级时使用（如线性回归模型中控制顺序）

两种方法均支持：
- 逐截面（per cross-section）正交化：每个交易日截面分别正交化
- 全局（global panel）正交化：用全部样本估计正交化矩阵，再应用到每行

Usage
-----
>>> import numpy as np
>>> from src.features.orthogonalize import symmetric_orthogonalize, gram_schmidt_orthogonalize
>>> X = np.random.randn(500, 10)  # 500 only, 10 factors
>>> X_orth = symmetric_orthogonalize(X)
>>> assert X_orth.shape == (500, 10)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _center_and_scale(
    X: np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对因子矩阵按列做去均值 + 归一化（处理 nan，忽略缺失）。

    Returns
    -------
    X_scaled, col_mean, col_std
    """
    X = np.asarray(X, dtype=np.float64)
    col_mean = np.nanmean(X, axis=0) if center else np.zeros(X.shape[1])
    col_std = np.nanstd(X, axis=0, ddof=0) if scale else np.ones(X.shape[1])
    col_std = np.where(col_std < eps, 1.0, col_std)
    X_out = (X - col_mean) / col_std
    return X_out, col_mean, col_std


def symmetric_orthogonalize(
    X: np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
    fill_nan_with_zero: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    对称正交化（Löwdin / Symmetric Orthogonalization）。

    对因子矩阵 X (n_samples × n_factors) 进行对称正交化：
        X_orth = X @ V @ diag(1/sqrt(s)) @ V^T

    其中 V, s 来自因子相关矩阵的特征值分解：C = V diag(s) V^T。

    性质：
    - 正交化后的因子两两不相关
    - 在所有正交化方法中，与原始因子整体相似度（Frobenius 距离）最小
    - 因子之间地位对称，无先后顺序

    Parameters
    ----------
    X
        形状 ``(n_samples, n_factors)``，可含 ``nan``（缺失处先用 0 填充再处理）。
    center, scale
        是否先做截面去均值/标准化（推荐 True）。
    fill_nan_with_zero
        缺失值是否用 0 填充（标准化后 0 等价于截面均值）；False 则输出原样保留 nan。
    eps
        特征值下限，防止除零。

    Returns
    -------
    X_orth : ndarray，形状与 X 相同
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X 须为二维 (n_samples, n_factors)")
    n, p = X.shape
    if p < 2:
        return X.copy()

    nan_mask = ~np.isfinite(X)
    X_s, col_mean, col_std = _center_and_scale(X, center=center, scale=scale, eps=eps)

    if fill_nan_with_zero:
        X_filled = np.where(np.isfinite(X_s), X_s, 0.0)
    else:
        X_filled = X_s.copy()
        X_filled[~np.isfinite(X_filled)] = 0.0

    # 因子相关矩阵 C = X'X / n（用有限样本估计）
    # 使用 filled 矩阵计算
    C = X_filled.T @ X_filled
    # 归一化为相关矩阵（避免方差量纲影响）
    d = np.diag(C)
    d_safe = np.where(d > eps, d, 1.0)
    D_inv_half = np.diag(1.0 / np.sqrt(d_safe))
    R = D_inv_half @ C @ D_inv_half  # 相关矩阵
    R = 0.5 * (R + R.T)  # 保证对称

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, eps)
    S_inv_half = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ S_inv_half @ eigvecs.T  # 对称正交化矩阵

    X_orth = X_filled @ W

    if not fill_nan_with_zero:
        X_orth[nan_mask] = np.nan

    return X_orth


def gram_schmidt_orthogonalize(
    X: np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
    fill_nan_with_zero: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Gram-Schmidt 正交化（修正版，Modified Gram-Schmidt）。

    按列顺序逐步将第 j 列对前 j-1 列投影后取残差，
    先出现的因子保留更多原始信息，后出现的因子被剥离与前列的相关性。

    适用于因子有优先级的场景（如：先保留动量，再正交化其他因子）。

    Parameters
    ----------
    X
        形状 ``(n_samples, n_factors)``。
    center, scale
        是否先做截面去均值/标准化。

    Returns
    -------
    X_orth : ndarray，形状与 X 相同，各列已两两正交
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X 须为二维 (n_samples, n_factors)")
    n, p = X.shape
    if p < 2:
        return X.copy()

    nan_mask = ~np.isfinite(X)
    X_s, col_mean, col_std = _center_and_scale(X, center=center, scale=scale, eps=eps)
    X_filled = np.where(np.isfinite(X_s), X_s, 0.0)

    Q = X_filled.copy()
    for j in range(p):
        for i in range(j):
            norm_i = float(Q[:, i] @ Q[:, i])
            if norm_i < eps:
                continue
            proj = float(Q[:, i] @ Q[:, j]) / norm_i
            Q[:, j] -= proj * Q[:, i]
        # 归一化列长度到与原始因子方差一致（可选，保留量纲）
        norm_j = np.sqrt(float(Q[:, j] @ Q[:, j]))
        if norm_j > eps:
            orig_std = float(np.nanstd(X_filled[:, j])) or 1.0
            Q[:, j] *= orig_std / norm_j

    if not fill_nan_with_zero:
        Q[nan_mask] = np.nan

    return Q


def orthogonalize_panel_by_date(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    method: str = "symmetric",
    date_col: str = "trade_date",
    center: bool = True,
    scale: bool = True,
    suffix: str = "_orth",
    eps: float = 1e-10,
) -> pd.DataFrame:
    """
    按交易日截面对因子矩阵正交化，为每列追加 ``{col}{suffix}`` 列。

    每个截面（同一 ``trade_date``）独立正交化，保持截面内因子的横截面相关性被剥离，
    但不同截面之间相互独立（等价于"截面式"因子处理）。

    Parameters
    ----------
    df
        长表，须含 ``date_col`` 和 ``factor_cols``。
    factor_cols
        要正交化的因子列名列表。
    method
        ``symmetric``：对称正交化（推荐）；``gram_schmidt``：Gram-Schmidt 正交化。
    suffix
        正交化后列名后缀，默认 ``_orth``。

    Returns
    -------
    DataFrame，新增正交化因子列。
    """
    method = str(method).lower().strip()
    if method not in ("symmetric", "gram_schmidt"):
        raise ValueError("method 须为 symmetric 或 gram_schmidt")

    ortho_fn = symmetric_orthogonalize if method == "symmetric" else gram_schmidt_orthogonalize

    out = df.copy()
    orth_cols = [f"{c}{suffix}" for c in factor_cols]
    for c in orth_cols:
        out[c] = np.nan

    for date, grp in out.groupby(date_col, sort=False):
        idx = grp.index
        X = grp[factor_cols].to_numpy(dtype=np.float64)
        if len(X) < 3 or X.shape[1] < 2:
            for orig, new in zip(factor_cols, orth_cols):
                out.loc[idx, new] = grp[orig].to_numpy()
            continue
        X_orth = ortho_fn(X, center=center, scale=scale, eps=eps)
        for j, new_col in enumerate(orth_cols):
            out.loc[idx, new_col] = X_orth[:, j]

    return out


def orthogonalize_matrix(
    X: np.ndarray,
    *,
    method: str = "symmetric",
    center: bool = True,
    scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    单次矩阵正交化入口：按 ``method`` 调度到具体实现。

    Parameters
    ----------
    X
        形状 ``(n_samples, n_factors)``。
    method
        ``symmetric`` 或 ``gram_schmidt``。

    Returns
    -------
    X_orth : ndarray，形状与 X 相同
    """
    method = str(method).lower().strip()
    if method == "symmetric":
        return symmetric_orthogonalize(X, center=center, scale=scale, eps=eps)
    elif method == "gram_schmidt":
        return gram_schmidt_orthogonalize(X, center=center, scale=scale, eps=eps)
    else:
        raise ValueError(f"未知正交化方法: {method!r}（期望 symmetric | gram_schmidt）")
