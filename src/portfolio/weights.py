"""从排序/得分表构造可交易权重，并施加单票、行业、换手约束。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


def infer_score_column(df: pd.DataFrame) -> str:
    """按优先级选择用于加权的得分列。"""
    for c in (
        "deep_sequence_score",
        "tree_score",
        "composite_extended_score",
        "composite_score",
        "momentum",
        "forward_ret_1d",
    ):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    if "rank" in df.columns:
        return "rank"
    raise ValueError(
        "无法推断 score 列：需要 deep_sequence_score、tree_score、composite_extended_score、composite_score、momentum 或 rank"
    )


def _scores_from_column(df: pd.DataFrame, score_col: str) -> np.ndarray:
    if score_col == "rank":
        r = pd.to_numeric(df["rank"], errors="coerce").to_numpy(dtype=np.float64)
        inv = np.where(np.isfinite(r), 1.0 / np.maximum(r, 1.0), np.nan)
        return inv
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float64)
    return s


def _nonnegative_weights_from_scores(scores: np.ndarray, *, method: str) -> np.ndarray:
    method = str(method).lower()
    if method == "equal":
        m = np.isfinite(scores)
        w = np.zeros_like(scores, dtype=np.float64)
        k = int(m.sum())
        if k == 0:
            return w
        w[m] = 1.0 / k
        return w
    if method == "score":
        x = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s <= 0:
            return _nonnegative_weights_from_scores(scores, method="equal")
        return x / s
    raise ValueError(
        f"未知 weight_method: {method!r}（期望 equal | score | risk_parity | "
        "min_variance | mean_variance）"
    )


def redistribute_individual_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """
    将非负权重归一化后，将超过 ``cap`` 的部分迭代地按「未触顶」仓位比例再分配（水线法）。

    若 ``cap`` 过小导致 ``n * cap < 1``，无法同时满足和为 1 与单票上限；此时仍归一化，
    调用方应保证 ``cap >= 1/n`` 或接受约束松弛。
    """
    w = np.asarray(w, dtype=np.float64).ravel().copy()
    if w.size == 0:
        return w
    w = np.maximum(w, 0.0)
    sw = w.sum()
    if sw <= 0:
        return w
    w /= sw
    cap = float(cap)
    if cap >= 1.0 or cap <= 0:
        return w
    for _ in range(4096):
        over = w > cap + 1e-14
        if not np.any(over):
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = w < cap - 1e-14
        if not np.any(free):
            break
        free_sum = float(w[free].sum())
        if free_sum > 1e-14:
            w[free] += excess * (w[free] / free_sum)
        else:
            n_free = int(np.count_nonzero(free))
            if n_free > 0:
                w[free] += excess / n_free
    s = w.sum()
    if s > 0:
        w /= s
    return w


def _scale_groups_to_cap(
    w: np.ndarray,
    group_ids: np.ndarray,
    cap_group: float,
) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).ravel().copy()
    gid = np.asarray(group_ids)
    if w.size != gid.size:
        raise ValueError("weights 与 group_ids 长度须一致")
    cap_group = float(cap_group)
    if cap_group >= 1.0:
        return w
    for g in np.unique(gid):
        idx = gid == g
        sg = float(w[idx].sum())
        if sg > cap_group + 1e-12:
            w[idx] *= cap_group / sg
    s = float(w.sum())
    if s > 0:
        w /= s
    return w


def apply_turnover_constraint(
    w_new: np.ndarray,
    w_old_aligned: np.ndarray,
    max_turnover: float,
    turnover_cost_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    在总权重均为 1 的长仅向量上，将换手约束为
    ``0.5 * sum(|w - w_old|) <= max_turnover``（单边换手为 L1/2）。

    若传入 ``turnover_cost_coeffs``，则约束变为
    ``0.5 * sum(coeff_i * |w_i - w_old_i|) <= max_turnover``，用于近似交易冲击成本。
    若 ``max_turnover >= 1`` 视为不限制。
    """
    w = np.asarray(w_new, dtype=np.float64).ravel().copy()
    o = np.asarray(w_old_aligned, dtype=np.float64).ravel().copy()
    if w.size != o.size:
        raise ValueError("w_new 与 w_old_aligned 长度须一致")
    max_turnover = float(max(max_turnover, 0.0))
    if max_turnover >= 1.0 - 1e-12:
        return w
    if turnover_cost_coeffs is None:
        tv = 0.5 * float(np.sum(np.abs(w - o)))
    else:
        c = np.asarray(turnover_cost_coeffs, dtype=np.float64).ravel().copy()
        if c.size != w.size:
            raise ValueError("turnover_cost_coeffs 长度须与 w_new 一致")
        c = np.where(np.isfinite(c) & (c > 0), c, 1.0)
        tv = 0.5 * float(np.sum(c * np.abs(w - o)))
    if tv <= max_turnover + 1e-12 or tv <= 0:
        return w
    lam = max_turnover / tv
    return o + lam * (w - o)


def turnover_cost_coeffs_from_size(
    size_values: np.ndarray,
    *,
    small_cap_coeff: float = 1.6,
    mid_cap_coeff: float = 1.0,
    large_cap_coeff: float = 0.7,
    q_small: float = 0.33,
    q_large: float = 0.67,
) -> np.ndarray:
    """按市值分档构造换手成本系数（小市值成本更高）。"""
    x = np.asarray(size_values, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return x
    coeff = np.ones(n, dtype=np.float64) * float(mid_cap_coeff)
    fin = np.isfinite(x)
    if int(fin.sum()) < 2:
        return coeff
    v = x[fin]
    q1 = float(np.quantile(v, float(q_small)))
    q2 = float(np.quantile(v, float(q_large)))
    coeff[fin & (x <= q1)] = float(small_cap_coeff)
    coeff[fin & (x >= q2)] = float(large_cap_coeff)
    coeff = np.where(np.isfinite(coeff) & (coeff > 0), coeff, 1.0)
    return coeff


def build_portfolio_weights(
    df: pd.DataFrame,
    *,
    weight_method: str = "score",
    score_col: str = "auto",
    max_single_weight: float = 1.0,
    max_industry_weight: Optional[float] = None,
    industry_col: Optional[str] = None,
    prev_weights_aligned: Optional[np.ndarray] = None,
    max_turnover: float = 1.0,
    cov_matrix: Optional[np.ndarray] = None,
    expected_returns: Optional[np.ndarray] = None,
    risk_aversion: float = 1.0,
    turnover_cost_model: Optional[Mapping[str, Any]] = None,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    返回与 ``df`` 行对齐、和为 1 的长仅权重向量（若无可投则全 0）。

    ``weight_method`` 为 ``risk_parity`` / ``min_variance`` / ``mean_variance`` 时须提供
    与行数一致的 ``cov_matrix``；``mean_variance`` 另须 ``expected_returns``（与行顺序一致）。
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    method = str(weight_method).lower()
    cov_methods = ("risk_parity", "min_variance", "mean_variance")
    diagnostics: Dict[str, Any] = {
        "method": method,
        "constraints": {
            "max_single_weight": float(max_single_weight),
            "max_industry_weight": float(max_industry_weight) if max_industry_weight is not None else None,
            "max_turnover": float(max_turnover),
        },
    }
    if method in cov_methods:
        if cov_matrix is None:
            raise ValueError(
                f"weight_method={method!r} 需要 cov_matrix（Top-K 收益协方差）"
            )
        cov = np.asarray(cov_matrix, dtype=np.float64)
        if cov.shape != (n, n):
            raise ValueError(
                f"cov_matrix 形状 {cov.shape} 与推荐表行数 {n} 不一致"
            )
        from src.portfolio.optimizer import solve_weights_from_cov_method, weight_diagnostics

        mu = None
        if method == "mean_variance":
            if expected_returns is None:
                raise ValueError("weight_method=mean_variance 需要 expected_returns (mu)")
            mu = np.asarray(expected_returns, dtype=np.float64).ravel()
            if mu.size != n:
                raise ValueError("expected_returns 长度须与推荐表行数一致")
        w, opt_diag = solve_weights_from_cov_method(
            method,
            cov,
            mu=mu,
            risk_aversion=float(risk_aversion),
        )
        diagnostics["optimizer"] = opt_diag
    else:
        sc = infer_score_column(df) if score_col == "auto" else str(score_col)
        scores = _scores_from_column(df, sc)
        w = _nonnegative_weights_from_scores(scores, method=weight_method)
        from src.portfolio.optimizer import weight_diagnostics

        diagnostics["optimizer"] = {
            "method": method,
            "score_col": sc,
            "weights": weight_diagnostics(w),
        }
    w_before_constraints = np.asarray(w, dtype=np.float64).copy()
    ind_c = industry_col if industry_col and industry_col in df.columns else None
    g = None
    if ind_c is not None and max_industry_weight is not None:
        g = df[ind_c].astype(str).fillna("_NA_").to_numpy()
        cap_g = float(max_industry_weight)
        ms = float(max_single_weight)
        for _ in range(256):
            w0 = w.copy()
            w = redistribute_individual_cap(w, ms)
            for _ in range(32):
                w = _scale_groups_to_cap(w, g, cap_g)
                w = redistribute_individual_cap(w, ms)
                if bool(np.all(w <= ms + 1e-9)):
                    break
            s = float(w.sum())
            if s > 0:
                w /= s
            if np.allclose(w, w0, rtol=0, atol=1e-9):
                break
    else:
        w = redistribute_individual_cap(w, float(max_single_weight))

    if prev_weights_aligned is not None:
        coeffs = None
        m = dict(turnover_cost_model or {})
        if bool(m.get("enabled", False)):
            size_col = str(m.get("size_col", "log_market_cap"))
            if size_col in df.columns:
                size_vals = pd.to_numeric(df[size_col], errors="coerce").to_numpy(dtype=np.float64)
                coeffs = turnover_cost_coeffs_from_size(
                    size_vals,
                    small_cap_coeff=float(m.get("small_cap_coeff", 1.6)),
                    mid_cap_coeff=float(m.get("mid_cap_coeff", 1.0)),
                    large_cap_coeff=float(m.get("large_cap_coeff", 0.7)),
                    q_small=float(m.get("q_small", 0.33)),
                    q_large=float(m.get("q_large", 0.67)),
                )
        w = apply_turnover_constraint(
            w,
            prev_weights_aligned,
            max_turnover,
            turnover_cost_coeffs=coeffs,
        )
    s = w.sum()
    if s > 0:
        w /= s
    from src.portfolio.optimizer import weight_diagnostics

    equal_reference = np.ones_like(w, dtype=np.float64) / float(len(w)) if len(w) > 0 else None
    diagnostics["post_constraints"] = weight_diagnostics(w, reference=equal_reference)
    diagnostics["post_constraint_l1_shift"] = float(np.sum(np.abs(w - w_before_constraints)))
    if return_diagnostics:
        return w, diagnostics
    return w


def load_prev_weights_series(
    path: Union[str, Path],
    *,
    symbols: Tuple[str, ...],
) -> np.ndarray:
    """
    读取 ``symbol,weight`` CSV，与 ``symbols`` 顺序对齐；缺失视为 0。
    """
    p = Path(path)
    t = pd.read_csv(
        p,
        encoding="utf-8-sig",
        converters={
            "symbol": lambda v: str(v).strip(),
            "代码": lambda v: str(v).strip(),
        },
    )
    sym_c = "symbol" if "symbol" in t.columns else "代码"
    w_c = "weight" if "weight" in t.columns else "权重"
    if sym_c not in t.columns or w_c not in t.columns:
        raise ValueError(f"上一期权重文件需含 {sym_c} 与 {w_c} 列: {p}")
    t[sym_c] = t[sym_c].astype(str).str.zfill(6)
    m = dict(zip(t[sym_c], pd.to_numeric(t[w_c], errors="coerce")))
    out = np.array([float(m.get(s, 0.0) or 0.0) for s in symbols], dtype=np.float64)
    s = out.sum()
    if s > 0:
        out /= s
    return out


def portfolio_config_from_mapping(sig: Mapping[str, Any]) -> Dict[str, Any]:
    """从 ``config['portfolio']`` 解析组合参数（缺省安全）。"""
    p = dict(sig) if isinstance(sig, dict) else {}
    mi = p.get("max_industry_weight")
    ra = p.get("risk_aversion", 1.0)
    cr = p.get("cov_ridge", 1e-6)
    cs = p.get("cov_shrinkage", "ledoit_wolf")
    tcm = p.get("turnover_cost_model") or {}
    return {
        "weight_method": str(p.get("weight_method", "score")).lower(),
        "score_col": str(p.get("score_col", "auto")),
        "max_single_weight": float(p.get("max_single_weight", 0.1)),
        "max_industry_weight": float(mi) if mi is not None else None,
        "industry_col": p.get("industry_col"),
        "max_turnover": float(p.get("max_turnover", 1.0)),
        "cov_lookback_days": int(p.get("cov_lookback_days", 60)),
        "risk_aversion": float(ra) if ra is not None else 1.0,
        "cov_ridge": float(cr) if cr is not None else 1e-6,
        "cov_shrinkage": str(cs).lower() if cs is not None else "ledoit_wolf",
        "cov_ewma_halflife": float(p.get("cov_ewma_halflife", 20.0)),
        "turnover_cost_model": {
            "enabled": bool((tcm or {}).get("enabled", False)),
            "size_col": str((tcm or {}).get("size_col", "log_market_cap")),
            "small_cap_coeff": float((tcm or {}).get("small_cap_coeff", 1.6)),
            "mid_cap_coeff": float((tcm or {}).get("mid_cap_coeff", 1.0)),
            "large_cap_coeff": float((tcm or {}).get("large_cap_coeff", 0.7)),
            "q_small": float((tcm or {}).get("q_small", 0.33)),
            "q_large": float((tcm or {}).get("q_large", 0.67)),
        },
    }
