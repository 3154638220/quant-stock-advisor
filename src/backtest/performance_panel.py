"""统一绩效面板：年化、夏普、Calmar、最大回撤、胜率、换手率。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from src.backtest.risk_metrics import max_drawdown_from_returns


@dataclass(frozen=True)
class PerformancePanel:
    """回测/切片统一输出指标（与具体引擎解耦，便于 walk-forward 汇总）。

    P2-2: 新增 dsr / dsr_pvalue 字段，用于 Deflated Sharpe Ratio 多重比较检验。
    """

    annualized_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    turnover_mean: float
    n_periods: int
    total_return: float
    periods_per_year: float
    # P2-2: Deflated Sharpe Ratio 多重比较检验
    dsr: float = float("nan")
    dsr_pvalue: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64).ravel()
    return r[np.isfinite(r)]  # type: ignore[no-any-return]


def total_return_from_simple_returns(returns: np.ndarray) -> float:
    """简单收益序列的复利总收益：prod(1+r)-1。"""
    r = _finite_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)  # type: ignore[no-any-return]


def annualized_return_cagr(
    returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """
    由日（或 bar）简单收益序列推算年化复合收益：(1+R)^{PY/n}-1。
    """
    r = _finite_returns(returns)
    n = r.size
    if n == 0:
        return float("nan")
    cum = float(np.prod(1.0 + r))
    if cum <= 0:
        return float("nan")
    return float(cum ** (periods_per_year / n) - 1.0)


def sharpe_ratio(
    returns: np.ndarray,
    *,
    risk_free_daily: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """超额收益夏普（简单收益、样本标准差）。"""
    r = _finite_returns(returns)
    if r.size < 2:
        return float("nan")
    ex = r - float(risk_free_daily)
    mu = float(np.mean(ex))
    sd = float(np.std(ex, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(mu / sd * np.sqrt(periods_per_year))


def calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    """年化收益 / 最大回撤（回撤为正值）。"""
    if not np.isfinite(annualized_return) or not np.isfinite(max_drawdown):
        return float("nan")
    if max_drawdown <= 1e-15:
        return float("nan")
    return float(annualized_return / max_drawdown)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> tuple[float, float]:
    """P2-2: Deflated Sharpe Ratio (Bailey & López de Prado, 2014)。

    修正多重比较和非正态性对 Sharpe Ratio 的通货膨胀效应。
    同时测试多个策略配置时，随机抽到高 Sharpe 的概率不可忽视。

    Parameters
    ----------
    sharpe : 观测到的年化 Sharpe Ratio
    n_trials : 并行测试的策略/配置数量（多重比较次数）
    skewness : 收益分布的偏度（默认 0，正态）
    kurtosis : 收益分布的峰度（默认 3，正态）
    n_obs : 观测样本数（默认 252 个交易日）

    Returns
    -------
    (dsr, p_value) : DSR 值和对应的 p-value
        DSR < 0.05 表示统计显著；p_value > 0.05 表示不显著。
    """
    import numpy as np
    from scipy.stats import norm

    sr = float(sharpe)
    T = max(int(n_trials), 1)
    N = max(int(n_obs), 1)

    if not np.isfinite(sr) or sr <= 0:
        return 0.0, 1.0

    # E[max(Sharpe)] under null (zero mean), corrected for non-normality
    # Bailey & López de Prado (2014), Eq. (6)
    sk = float(skewness)
    ku = float(kurtosis)

    # Expected maximum Sharpe from T independent trials under null
    z_score = (1.0 - np.exp(-1.0)) * norm.ppf(1.0 - 1.0 / T)
    if not np.isfinite(z_score):
        z_score = norm.ppf(1.0 - 1.0 / T)

    # Non-normality correction: variance of Sharpe estimator
    var_sharpe = 1.0 + 0.5 * sr**2 - sk * sr + (ku - 3.0) / 4.0 * sr**2
    se_sharpe = np.sqrt(max(var_sharpe, 1e-8) / N)

    # DSR: P(max SR >= observed SR) under null
    # = 1 - ((P(SR < observed))^T)
    prob_single = norm.cdf(sr, loc=0.0, scale=se_sharpe)
    if prob_single >= 1.0:
        dsr = 0.0
    else:
        dsr = 1.0 - prob_single ** T
    dsr = float(np.clip(dsr, 0.0, 1.0))

    # p-value approximation
    p_value = 1.0 - float(dsr) if dsr < 1.0 else 0.0

    return dsr, p_value


def win_rate(returns: np.ndarray) -> float:
    """正收益 bar 占比。"""
    r = _finite_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0.0))


def compute_performance_panel(
    returns: np.ndarray,
    *,
    turnover: Optional[np.ndarray] = None,
    risk_free_daily: float = 0.0,
    periods_per_year: float = 252.0,
    # P2-2: 多重比较检验参数
    n_concurrent_strategies: int = 1,
) -> PerformancePanel:
    """
    由单条收益序列（通常为日收益）计算统一绩效面板。

    Parameters
    ----------
    returns
        简单收益序列（如日度）。
    turnover
        可选，与 ``returns`` 对齐或可聚合的换手序列；若提供则取 ``nanmean`` 为 turnover_mean，
        否则 turnover_mean 为 nan。
    n_concurrent_strategies
        P2-2: 并行测试的策略/配置数量（多重比较次数）。
        当 > 1 时，计算 Deflated Sharpe Ratio 并对 Sharpe 进行多重比较修正。
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r_fin = _finite_returns(r)
    n = int(r_fin.size)
    tot = total_return_from_simple_returns(r_fin)
    ann = annualized_return_cagr(r_fin, periods_per_year=periods_per_year)
    mdd = max_drawdown_from_returns(r_fin)
    sh = sharpe_ratio(r_fin, risk_free_daily=risk_free_daily, periods_per_year=periods_per_year)
    cal = calmar_ratio(ann, mdd)
    wr = win_rate(r_fin)

    if turnover is not None:
        t = np.asarray(turnover, dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
        t_mean = float(np.nanmean(t)) if t.size else float("nan")
    else:
        t_mean = float("nan")

    # P2-2: Deflated Sharpe Ratio 多重比较检验
    dsr = float("nan")
    dsr_pvalue = float("nan")
    n_trials = max(int(n_concurrent_strategies), 1)
    if n_trials > 1 and np.isfinite(sh) and n >= 2:
        # 计算收益分布的偏度和峰度
        r_valid = r_fin[np.isfinite(r_fin)]
        if r_valid.size >= 3:
            from scipy.stats import kurtosis as scipy_kurtosis
            from scipy.stats import skew
            try:
                sk = float(skew(r_valid))
                ku = float(scipy_kurtosis(r_valid, fisher=False))  # Pearson 峰度
            except Exception:
                sk, ku = 0.0, 3.0
        else:
            sk, ku = 0.0, 3.0
        dsr, dsr_pvalue = deflated_sharpe_ratio(
            sh,
            n_trials,
            skewness=sk,
            kurtosis=ku,
            n_obs=max(n, 1),
        )

    return PerformancePanel(
        annualized_return=ann,
        sharpe_ratio=sh,
        calmar_ratio=cal,
        max_drawdown=mdd,
        win_rate=wr,
        turnover_mean=t_mean,
        n_periods=n,
        total_return=tot,
        periods_per_year=float(periods_per_year),
        dsr=dsr,
        dsr_pvalue=dsr_pvalue,
    )


def aggregate_walk_forward_panels(
    panels: list[PerformancePanel],
    *,
    method: str = "mean",
) -> dict[str, Any]:
    """
    多折 walk-forward 面板的简单聚合（均值或中位数），便于总览稳定性。

    method: mean | median
    """
    if not panels:
        return {"n_folds": 0}
    keys = (
        "annualized_return",
        "sharpe_ratio",
        "calmar_ratio",
        "max_drawdown",
        "win_rate",
        "turnover_mean",
        "total_return",
    )
    method = str(method).lower()
    agg: dict[str, Any] = {"n_folds": len(panels), "method": method}
    for k in keys:
        vals = [getattr(p, k) for p in panels]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            agg[f"{k}_agg"] = float("nan")
        elif method == "median":
            agg[f"{k}_agg"] = float(np.median(arr))
        else:
            agg[f"{k}_agg"] = float(np.mean(arr))
    return agg


def panel_from_mapping(m: Mapping[str, Any]) -> PerformancePanel:
    """从字典构造 PerformancePanel（便于序列化往返）。"""
    return PerformancePanel(
        annualized_return=float(m["annualized_return"]),
        sharpe_ratio=float(m["sharpe_ratio"]),
        calmar_ratio=float(m["calmar_ratio"]),
        max_drawdown=float(m["max_drawdown"]),
        win_rate=float(m["win_rate"]),
        turnover_mean=float(m["turnover_mean"]),
        n_periods=int(m["n_periods"]),
        total_return=float(m["total_return"]),
        periods_per_year=float(m["periods_per_year"]),
    )
