"""风险平价与均值–方差（长仅、权重和为 1），并提供可观测性诊断摘要。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


def covariance_diagnostics(
    Sigma: np.ndarray,
    *,
    shrinkage_intensity: float = float("nan"),
) -> Dict[str, Any]:
    """输出协方差矩阵稳定性摘要，便于解释优化器是否有足够信号可用。

    P0-3: 支持外部传入 Ledoit-Wolf 收缩强度 α（0=样本协方差，1=收缩至均值相关）。
    """
    cov = _symmetrize(np.asarray(Sigma, dtype=np.float64))
    n = int(cov.shape[0]) if cov.ndim == 2 else 0
    if n == 0:
        return {
            "n_assets": 0,
            "diag_share": float("nan"),
            "mean_abs_offdiag": float("nan"),
            "mean_correlation": float("nan"),
            "condition_number": float("nan"),
            "min_eigenvalue": float("nan"),
            "max_eigenvalue": float("nan"),
            "shrinkage_intensity": float(shrinkage_intensity) if np.isfinite(shrinkage_intensity) else float("nan"),
        }
    diag = np.diag(cov).astype(np.float64)
    diag_sum = float(np.nansum(np.abs(diag)))
    total_sum = float(np.nansum(np.abs(cov)))
    offdiag_mask = ~np.eye(n, dtype=bool)
    offdiag = cov[offdiag_mask]
    vol = np.sqrt(np.maximum(diag, 0.0))
    denom = np.outer(vol, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 1e-18)
    corr_offdiag = corr[offdiag_mask]
    eigvals = np.linalg.eigvalsh(cov)
    abs_eig = np.abs(eigvals)
    max_abs_eig = float(np.max(abs_eig)) if abs_eig.size else float("nan")
    min_abs_eig = float(np.min(abs_eig)) if abs_eig.size else float("nan")
    cond = float(max_abs_eig / min_abs_eig) if min_abs_eig > 1e-18 else float("inf")
    return {
        "n_assets": n,
        "diag_share": float(diag_sum / total_sum) if total_sum > 0 else float("nan"),
        "mean_abs_offdiag": float(np.nanmean(np.abs(offdiag))) if offdiag.size else 0.0,
        "mean_correlation": float(np.nanmean(corr_offdiag)) if corr_offdiag.size else 0.0,
        "condition_number": cond,
        "min_eigenvalue": float(np.min(eigvals)) if eigvals.size else float("nan"),
        "max_eigenvalue": float(np.max(eigvals)) if eigvals.size else float("nan"),
        "shrinkage_intensity": float(shrinkage_intensity) if np.isfinite(shrinkage_intensity) else float("nan"),
    }


def weight_diagnostics(weights: np.ndarray, *, reference: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """输出权重离散度与相对参考组合的差异摘要。"""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return {
            "n_assets": 0,
            "effective_n": float("nan"),
            "weight_std": float("nan"),
            "max_weight": float("nan"),
            "min_weight": float("nan"),
            "nonzero_count": 0,
        }
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    s = float(w.sum())
    if s > 0:
        w = w / s
    nonzero = int(np.count_nonzero(w > 1e-10))
    sq = float(np.sum(w**2))
    out: Dict[str, Any] = {
        "n_assets": int(w.size),
        "effective_n": float(1.0 / sq) if sq > 1e-18 else float("nan"),
        "weight_std": float(np.std(w)),
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w)),
        "nonzero_count": nonzero,
    }
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64).ravel()
        if ref.size == w.size:
            ref = np.where(np.isfinite(ref), np.maximum(ref, 0.0), 0.0)
            rs = float(ref.sum())
            if rs > 0:
                ref = ref / rs
            diff = w - ref
            out.update(
                {
                    "l1_diff_vs_reference": float(np.sum(np.abs(diff))),
                    "max_abs_diff_vs_reference": float(np.max(np.abs(diff))),
                    "is_close_to_reference": bool(np.allclose(w, ref, rtol=0.0, atol=1e-8)),
                }
            )
    return out


def _finalize_solver_result(
    res: Any,
    n: int,
    *,
    fallback_reason: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """统一规范化 scipy 优化结果，并在异常时回退为等权。"""
    equal_w = np.ones(n, dtype=np.float64) / float(n)
    if res is None:
        return equal_w, {
            "solver_success": False,
            "solver_status": -1,
            "solver_message": fallback_reason or "missing_result",
            "solver_iterations": 0,
            "used_fallback": True,
            "fallback_reason": fallback_reason or "missing_result",
        }
    x = np.asarray(getattr(res, "x", np.array([], dtype=np.float64)), dtype=np.float64).ravel()
    if x.size != n or not np.all(np.isfinite(x)):
        return equal_w, {
            "solver_success": bool(getattr(res, "success", False)),
            "solver_status": int(getattr(res, "status", -1)),
            "solver_message": str(getattr(res, "message", fallback_reason or "invalid_solution")),
            "solver_iterations": int(getattr(res, "nit", 0) or 0),
            "used_fallback": True,
            "fallback_reason": fallback_reason or "invalid_solution",
        }
    w = np.maximum(x, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return equal_w, {
            "solver_success": bool(getattr(res, "success", False)),
            "solver_status": int(getattr(res, "status", -1)),
            "solver_message": str(getattr(res, "message", fallback_reason or "zero_sum_solution")),
            "solver_iterations": int(getattr(res, "nit", 0) or 0),
            "used_fallback": True,
            "fallback_reason": fallback_reason or "zero_sum_solution",
        }
    return (w / sw).astype(np.float64), {
        "solver_success": bool(getattr(res, "success", False)),
        "solver_status": int(getattr(res, "status", -1)),
        "solver_message": str(getattr(res, "message", "")),
        "solver_iterations": int(getattr(res, "nit", 0) or 0),
        "used_fallback": False,
        "fallback_reason": "",
        "objective_value": float(getattr(res, "fun", np.nan)),
    }


def optimize_risk_parity(
    Sigma: np.ndarray,
    *,
    max_iter: int = 3000,
    ftol: float = 1e-11,
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
) -> np.ndarray:
    """
    等风险贡献（ERC）：最小化各资产风险贡献相对均值的平方偏差，约束 ``sum w = 1``、``w >= 0``。

    P2-6: 可选换手约束 ``max_turnover``（0~1），限制权重与 ``prev_weights`` 的 L1 距离。
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
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    bounds = [(1e-8, 1.0)] * n

    # P2-6: 换手约束
    if prev_weights is not None and max_turnover < 1.0:
        prev = np.asarray(prev_weights, dtype=np.float64).ravel()
        if prev.size == n:
            # 使用线性化：sum(|w_i - prev_i|) <= max_turnover
            # 引入辅助变量（通过两次约束实现：w - prev <= slack, prev - w <= slack）
            # 简化处理：用 scipy 的 NonlinearConstraint
            cons.append({
                "type": "ineq",
                "fun": lambda w: float(max_turnover) - float(np.sum(np.abs(w - prev))),
            })

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
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
) -> np.ndarray:
    """长仅最小方差组合：``min w' Σ w``，``sum w = 1``。

    P2-6: 可选换手约束 ``max_turnover``。
    """
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
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # P2-6: 换手约束
    if prev_weights is not None and max_turnover < 1.0:
        prev = np.asarray(prev_weights, dtype=np.float64).ravel()
        if prev.size == n:
            cons.append({
                "type": "ineq",
                "fun": lambda w: float(max_turnover) - float(np.sum(np.abs(w - prev))),
            })

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
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
) -> np.ndarray:
    """
    长仅均值–方差：``max mu'w - (λ/2) w'Σw``，等价于最小化 ``(λ/2) w'Σw - mu'w``。

    P2-6: 可选换手约束 ``max_turnover``。
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
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # P2-6: 换手约束
    if prev_weights is not None and max_turnover < 1.0:
        prev = np.asarray(prev_weights, dtype=np.float64).ravel()
        if prev.size == n:
            cons.append({
                "type": "ineq",
                "fun": lambda w: float(max_turnover) - float(np.sum(np.abs(w - prev))),
            })

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
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
) -> np.ndarray:
    m = str(method).lower()
    if m == "risk_parity":
        return optimize_risk_parity(Sigma, prev_weights=prev_weights, max_turnover=max_turnover)
    if m == "min_variance":
        return optimize_min_variance(Sigma, prev_weights=prev_weights, max_turnover=max_turnover)
    if m == "mean_variance":
        if mu is None:
            raise ValueError("mean_variance 需要 expected_returns (mu)")
        return optimize_mean_variance(
            Sigma,
            mu,
            risk_aversion=risk_aversion,
            prev_weights=prev_weights,
            max_turnover=max_turnover,
        )
    raise ValueError(f"未知协方差优化方法: {method!r}")


def solve_weights_from_cov_method(
    method: str,
    Sigma: np.ndarray,
    *,
    mu: Optional[np.ndarray] = None,
    risk_aversion: float = 1.0,
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """求解协方差驱动的组合权重，并返回优化器与协方差诊断。

    P2-6: 支持换手约束 ``max_turnover`` 和 ``prev_weights``。
    """
    cov = _symmetrize(np.asarray(Sigma, dtype=np.float64))
    n = int(cov.shape[0]) if cov.ndim == 2 else 0
    if n == 0:
        return np.zeros(0, dtype=np.float64), {
            "method": str(method).lower(),
            "covariance": covariance_diagnostics(cov),
            "weights": weight_diagnostics(np.zeros(0, dtype=np.float64)),
        }
    if cov.shape != (n, n):
        raise ValueError(f"Sigma 形状非法: {cov.shape}")

    m = str(method).lower()
    equal_w = np.ones(n, dtype=np.float64) / float(n)
    diag: Dict[str, Any] = {
        "method": m,
        "covariance": covariance_diagnostics(cov),
    }

    # P2-6: 验证换手约束参数
    prev_vec: Optional[np.ndarray] = None
    if prev_weights is not None:
        prev_vec = np.asarray(prev_weights, dtype=np.float64).ravel()
        if prev_vec.size != n:
            prev_vec = None

    if n == 1:
        w = np.ones(1, dtype=np.float64)
        diag["weights"] = weight_diagnostics(w, reference=equal_w)
        diag.update(
            {
                "solver_success": True,
                "solver_status": 0,
                "solver_message": "single_asset",
                "solver_iterations": 0,
                "used_fallback": False,
                "fallback_reason": "",
            }
        )
        return w, diag

    # 构建约束列表（含换手约束）
    def _build_cons() -> list:
        cons_list = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
        if prev_vec is not None and max_turnover < 1.0:
            cons_list.append({
                "type": "ineq",
                "fun": lambda w: float(max_turnover) - float(np.sum(np.abs(w - prev_vec))),
            })
        return cons_list

    res = None
    if m == "risk_parity":
        def objective_rp(w: np.ndarray) -> float:
            w = np.maximum(w, 1e-12)
            w = w / w.sum()
            rc = _risk_contributions(w, cov)
            m_rc = float(np.mean(rc))
            return float(np.sum((rc - m_rc) ** 2))

        x0 = np.ones(n) / n
        bounds = [(1e-8, 1.0)] * n
        res = minimize(
            objective_rp,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=_build_cons(),
            options={"maxiter": 3000, "ftol": 1e-11},
        )
        w, solver_info = _finalize_solver_result(res, n)
        rc = _risk_contributions(w, cov)
        diag["risk_contribution_std"] = float(np.std(rc))
        diag["risk_contribution_mean"] = float(np.mean(rc))
    elif m == "min_variance":
        def objective_mv(w: np.ndarray) -> float:
            w = np.maximum(w, 0.0)
            s = float(w.sum())
            if s <= 0:
                return 1e12
            w = w / s
            return float(w @ cov @ w)

        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        res = minimize(
            objective_mv,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=_build_cons(),
            options={"maxiter": 3000, "ftol": 1e-11},
        )
        w, solver_info = _finalize_solver_result(res, n)
    elif m == "mean_variance":
        if mu is None:
            raise ValueError("mean_variance 需要 expected_returns (mu)")
        mu_vec = np.asarray(mu, dtype=np.float64).ravel()
        if mu_vec.size != n:
            raise ValueError("mu 与 Sigma 维度不一致")
        lam = float(max(risk_aversion, 1e-12))

        def objective_meanvar(w: np.ndarray) -> float:
            w = np.maximum(w, 0.0)
            s = float(w.sum())
            if s <= 0:
                return 1e12
            w = w / s
            return float(0.5 * lam * (w @ cov @ w) - (mu_vec @ w))

        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        res = minimize(
            objective_meanvar,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=_build_cons(),
            options={"maxiter": 3000, "ftol": 1e-11},
        )
        w, solver_info = _finalize_solver_result(res, n)
        diag["expected_return_mean"] = float(np.mean(mu_vec))
        diag["expected_return_std"] = float(np.std(mu_vec))
    else:
        raise ValueError(f"未知协方差优化方法: {method!r}")

    diag.update(solver_info)
    diag["weights"] = weight_diagnostics(w, reference=equal_w)
    if diag["weights"].get("is_close_to_reference", False) and not diag.get("fallback_reason"):
        diag["fallback_reason"] = "equal_like_solution"
    return w, diag
