"""风险平价与均值–方差（长仅、权重和为 1）。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


def _symmetrize(Sigma: np.ndarray) -> np.ndarray:
    s = np.asarray(Sigma, dtype=np.float64)
    return 0.5 * (s + s.T)


def _risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.zeros_like(w)
    w = w / s
    return w * (Sigma @ w)


def optimize_risk_parity(
    Sigma: np.ndarray,
    *,
    max_iter: int = 3000,
    ftol: float = 1e-11,
) -> np.ndarray:
    """
    等风险贡献（ERC）：最小化各资产风险贡献相对均值的平方偏差，约束 ``sum w = 1``、``w > 0``。
    """
    Sigma = _symmetrize(Sigma)
    n = Sigma.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        rc = _risk_contributions(w, Sigma)
        m = float(np.mean(rc))
        return float(np.sum((rc - m) ** 2))

    x0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    bounds = [(1e-8, 1.0)] * n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": int(max_iter), "ftol": float(ftol)},
    )
    w = np.maximum(res.x, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return np.ones(n) / n
    w = w / sw
    return w.astype(np.float64)


def optimize_min_variance(
    Sigma: np.ndarray,
    *,
    max_iter: int = 3000,
    ftol: float = 1e-11,
) -> np.ndarray:
    """长仅最小方差组合：``min w' Σ w``，``sum w = 1``。"""
    Sigma = _symmetrize(Sigma)
    n = Sigma.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            return 1e12
        w = w / s
        return float(w @ Sigma @ w)

    x0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": int(max_iter), "ftol": float(ftol)},
    )
    w = np.maximum(res.x, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return np.ones(n) / n
    return (w / sw).astype(np.float64)


def optimize_mean_variance(
    Sigma: np.ndarray,
    mu: np.ndarray,
    *,
    risk_aversion: float = 1.0,
    max_iter: int = 3000,
    ftol: float = 1e-11,
) -> np.ndarray:
    """
    长仅均值–方差：``max mu'w - (λ/2) w'Σw``，等价于最小化 ``(λ/2) w'Σw - mu'w``。
    """
    Sigma = _symmetrize(Sigma)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    n = Sigma.shape[0]
    if mu.size != n:
        raise ValueError("mu 与 Sigma 维度不一致")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    lam = float(max(risk_aversion, 1e-12))

    def objective(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            return 1e12
        w = w / s
        return float(0.5 * lam * (w @ Sigma @ w) - (mu @ w))

    x0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": int(max_iter), "ftol": float(ftol)},
    )
    w = np.maximum(res.x, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return np.ones(n) / n
    return (w / sw).astype(np.float64)


def weights_from_cov_method(
    method: str,
    Sigma: np.ndarray,
    *,
    mu: Optional[np.ndarray] = None,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    m = str(method).lower()
    if m == "risk_parity":
        return optimize_risk_parity(Sigma)
    if m == "min_variance":
        return optimize_min_variance(Sigma)
    if m == "mean_variance":
        if mu is None:
            raise ValueError("mean_variance 需要 expected_returns (mu)")
        return optimize_mean_variance(Sigma, mu, risk_aversion=risk_aversion)
    raise ValueError(f"未知协方差优化方法: {method!r}")
