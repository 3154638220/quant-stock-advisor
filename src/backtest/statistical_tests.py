"""回测统计显著性检验：Newey-West t-统计量、bootstrap 置信区间、Information Ratio。

P2-10: M10 评估报告需要 NW-adjusted t-stat、bootstrap CI 和 IR 指标，
用于判断历史超额是否显著非零（考虑截面相关性和时序自相关）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _newey_west_se(
    series: np.ndarray,
    *,
    max_lag: int = 6,
) -> float:
    """计算 Newey-West 异方差自相关一致标准误（HAC SE）。

    使用 Bartlett 核（线性衰减权重 1 - j/(max_lag+1)）。
    """
    n = len(series)
    if n < 3:
        return float(np.std(series, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    mu = float(np.mean(series))
    residuals = series - mu
    # Lag-0 方差
    w0 = float(np.sum(residuals**2) / n)
    # 自协方差加权和（Bartlett kernel）
    w_lag = 0.0
    for j in range(1, min(max_lag + 1, n - 1)):
        ac = float(np.dot(residuals[j:], residuals[:-j]) / n)
        w_lag += (1.0 - j / (max_lag + 1.0)) * ac
    hac_var = w0 + 2.0 * w_lag
    hac_var = max(hac_var, 1e-15)
    return float(np.sqrt(hac_var / n))


def newey_west_t_statistic(
    returns: np.ndarray,
    *,
    max_lag: int = 6,
    null_hypothesis: float = 0.0,
) -> dict[str, float]:
    """对超额收益序列计算 Newey-West 调整 t-统计量。

    Parameters
    ----------
    returns: (n,) ndarray — 超额收益（或绝对收益），通常是月度序列。
    max_lag: 最大自相关滞后阶数（默认 6 对应月度数据的半年依赖）。
    null_hypothesis: 零假设下的均值（默认 0，即检验超额是否显著 > 0）。

    Returns
    -------
    dict with keys: mean, nw_se, nw_t, p_value_onesided, n_obs, max_lag
    """
    x = np.asarray(returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float(np.nanmean(x)), "nw_se": float("nan"), "nw_t": float("nan"),
                "p_value_onesided": float("nan"), "n_obs": n, "max_lag": int(max_lag)}

    mu = float(np.mean(x))
    se = _newey_west_se(x, max_lag=max_lag)
    t_stat = float((mu - null_hypothesis) / se) if se > 0 else float("nan")

    # 单侧 p 值（近似，使用标准正态；小样本下偏保守）
    from math import erfc, sqrt
    p_onesided = float(0.5 * erfc(t_stat / sqrt(2.0))) if np.isfinite(t_stat) else float("nan")

    return {
        "mean": mu,
        "nw_se": se,
        "nw_t": t_stat,
        "p_value_onesided": p_onesided,
        "n_obs": n,
        "max_lag": int(max_lag),
    }


def newey_west_ic_t_statistic(
    rank_ic: np.ndarray,
    *,
    max_lag: int = 6,
) -> dict[str, float]:
    """对 Rank IC 序列计算 NW-adjusted t-统计量与 IR。

    Parameters
    ----------
    rank_ic: (n_months,) ndarray — 月度 Rank IC 序列
    max_lag: 最大滞后

    Returns
    -------
    dict with: ic_mean, ic_std, ic_ir, nw_se, nw_t, p_value_onesided, n_obs
    """
    base = newey_west_t_statistic(rank_ic, max_lag=max_lag)
    ic_std = float(np.std(rank_ic, ddof=1))
    ic_ir = float(base["mean"] / ic_std) if ic_std > 0 else float("nan")
    return {
        "ic_mean": base["mean"],
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "nw_se": base["nw_se"],
        "nw_t": base["nw_t"],
        "p_value_onesided": base["p_value_onesided"],
        "n_obs": base["n_obs"],
        "max_lag": int(max_lag),
    }


def bootstrap_excess_ci(
    monthly_excess: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """对月度超额序列计算 bootstrap 95% 置信区间。

    使用块 bootstrap（保留时序顺序，块大小 = 3 个月）。
    """
    x = np.asarray(monthly_excess, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return {"mean_excess": float(np.nanmean(x)), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_months": n, "n_bootstrap": n_bootstrap, "alpha": alpha}

    rng = np.random.RandomState(seed)
    block_size = min(3, max(1, n // 4))
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        block_idx = rng.randint(0, n_blocks, size=n_blocks)
        sample = []
        for bi in block_idx:
            start = bi * block_size
            end = min(start + block_size, n)
            sample.append(x[start:end])
        boot_series = np.concatenate(sample)[:n]
        means[b] = float(np.mean(boot_series))

    ci_lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_excess": float(np.mean(x)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_months": n,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def information_ratio(
    monthly_excess: np.ndarray,
    *,
    return_monthly: bool = True,
) -> dict[str, float]:
    """计算 Information Ratio = mean(excess) / std(excess)。

    Parameters
    ----------
    monthly_excess: 月度超额序列
    return_monthly: 若 True，IR 基于月度序列（不年化）；否则年化

    Returns
    -------
    dict with: ir, mean_excess, std_excess, n_months, annualized
    """
    x = np.asarray(monthly_excess, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"ir": float("nan"), "mean_excess": float(np.nanmean(x)),
                "std_excess": float("nan"), "n_months": n, "annualized": False}

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    ir = float(mu / sd) if sd > 0 else float("nan")
    if not return_monthly:
        ir = float(ir * np.sqrt(12.0))

    return {"ir": ir, "mean_excess": mu, "std_excess": sd, "n_months": n,
            "annualized": not return_monthly}


def turnover_adjusted_ir(
    monthly_excess: np.ndarray,
    monthly_turnover: np.ndarray,
    *,
    cost_slope: float = 0.0004,
) -> dict[str, float]:
    """换手调整后 IR = mean(excess) / (std(excess) + λ * mean(turnover))

    Parameters
    ----------
    monthly_excess: 月度超额序列
    monthly_turnover: 月度换手率序列（half L1, 0-1 之间）
    cost_slope: λ — 成本斜率，默认 ~20bps 单边（0.0004 = 4bps 以小数表示）

    Returns
    -------
    dict with: ir_adj, ir_raw, mean_excess, std_excess, mean_turnover,
              turnover_penalty, lambda
    """
    x = np.asarray(monthly_excess, dtype=np.float64)
    t = np.asarray(monthly_turnover, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(t)
    x, t = x[valid], t[valid]
    n = len(x)
    if n < 2:
        return {"ir_adj": float("nan"), "ir_raw": float("nan"),
                "mean_excess": float(np.nanmean(x)) if n > 0 else float("nan"),
                "std_excess": float("nan"), "mean_turnover": float("nan"),
                "turnover_penalty": float("nan"), "lambda": cost_slope}

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    mean_turn = float(np.mean(t))
    ir_raw = float(mu / sd) if sd > 0 else float("nan")
    penalty = sd + cost_slope * mean_turn
    ir_adj = float(mu / penalty) if penalty > 0 else float("nan")

    return {
        "ir_adj": ir_adj,
        "ir_raw": ir_raw,
        "mean_excess": mu,
        "std_excess": sd,
        "mean_turnover": mean_turn,
        "turnover_penalty": penalty,
        "lambda": cost_slope,
    }


def turnover_excess_by_month(
    months: list[str],
    monthly_excess: np.ndarray,
    monthly_turnover: np.ndarray,
    *,
    regime_labels: Optional[list[str]] = None,
) -> list[dict[str, object]]:
    """构建换手-超额散点数据（按月），供 M10 报告使用。

    Returns
    -------
    list of dict: 每月一条记录 {month, excess, turnover, regime}
    """
    x = np.asarray(monthly_excess, dtype=np.float64)
    t = np.asarray(monthly_turnover, dtype=np.float64)
    n = min(len(months), len(x), len(t))
    records: list[dict[str, object]] = []
    for i in range(n):
        rec: dict[str, object] = {
            "month": str(months[i]),
            "excess": float(x[i]) if np.isfinite(x[i]) else None,
            "turnover": float(t[i]) if np.isfinite(t[i]) else None,
        }
        if regime_labels and i < len(regime_labels):
            rec["regime"] = regime_labels[i]
        records.append(rec)
    return records


def turnover_excess_correlation(
    monthly_excess: np.ndarray,
    monthly_turnover: np.ndarray,
) -> dict[str, float]:
    """计算换手与超额的 Pearson/Spearman 相关性，检验超额是否依赖高换手。"""
    x = np.asarray(monthly_excess, dtype=np.float64)
    t = np.asarray(monthly_turnover, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(t)
    x, t = x[valid], t[valid]
    n = len(x)
    if n < 3:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": n}

    # Pearson
    xm, tm = x - x.mean(), t - t.mean()
    pearson = float(np.dot(xm, tm) / (np.sqrt(np.dot(xm, xm) * np.dot(tm, tm)) + 1e-15))

    # Spearman (rank-based)
    from scipy.stats import spearmanr
    sr = spearmanr(x, t)
    spearman = float(sr.statistic) if hasattr(sr, 'statistic') else float(sr[0])

    return {"pearson_r": pearson, "spearman_r": spearman, "n": n}
