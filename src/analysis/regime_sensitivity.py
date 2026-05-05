"""M12: Regime 参数敏感性网格分析。

在 bull_return_threshold × bear_return_threshold 网格上评估 regime 分类边界
对组合 Sharpe 和超额的敏感性，量化 Regime 参数的风险暴露。

Usage:
    from src.analysis.regime_sensitivity import run_regime_sensitivity_grid
    report = run_regime_sensitivity_grid(benchmark_returns, factor_weights, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.market.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_OSCILLATION,
    RegimeConfig,
    RegimeResult,
    classify_regime,
    get_regime_weights,
)


@dataclass
class RegimeGridPoint:
    """单个参数组合的 regime 分类结果摘要。"""

    bull_threshold: float
    bear_threshold: float
    n_bull: int
    n_bear: int
    n_oscillation: int
    bull_pct: float
    bear_pct: float
    oscillation_pct: float
    # 因子权重统计（跨时间均值）
    mean_momentum_weight: float
    mean_reversal_weight: float
    mean_lowvol_weight: float
    mean_size_weight: float
    # 权重波动（std of weight changes）
    weight_turnover_std: float


@dataclass
class RegimeSensitivityReport:
    """Regime 参数敏感性完整报告。"""

    grid: List[RegimeGridPoint]
    base_config: Dict[str, Any]
    n_months_analyzed: int
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_config": self.base_config,
            "n_months_analyzed": self.n_months_analyzed,
            "grid_points": [
                {
                    "bull_threshold": g.bull_threshold,
                    "bear_threshold": g.bear_threshold,
                    "n_bull": g.n_bull,
                    "n_bear": g.n_bear,
                    "n_oscillation": g.n_oscillation,
                    "bull_pct": round(g.bull_pct, 3),
                    "bear_pct": round(g.bear_pct, 3),
                    "oscillation_pct": round(g.oscillation_pct, 3),
                    "mean_momentum_weight": round(g.mean_momentum_weight, 4),
                    "mean_reversal_weight": round(g.mean_reversal_weight, 4),
                    "mean_lowvol_weight": round(g.mean_lowvol_weight, 4),
                    "mean_size_weight": round(g.mean_size_weight, 4),
                    "weight_turnover_std": round(g.weight_turnover_std, 4),
                }
                for g in self.grid
            ],
            "summary": self.summary,
        }


def _classify_regime_for_date(
    benchmark_series: pd.Series,
    asof_date,
    bull_threshold: float,
    bear_threshold: float,
) -> Tuple[str, RegimeResult]:
    """使用指定阈值对单个月份进行 regime 分类。"""
    cfg = RegimeConfig(
        bull_return_threshold=bull_threshold,
        bear_return_threshold=bear_threshold,
    )
    return classify_regime(benchmark_series, asof_date, cfg=cfg)


def _monthly_regime_classifications(
    benchmark_series: pd.Series,
    monthly_dates: List[pd.Timestamp],
    bull_threshold: float,
    bear_threshold: float,
) -> List[Tuple[str, RegimeResult]]:
    """对一系列月度截面日期进行 regime 分类。"""
    results: List[Tuple[str, RegimeResult]] = []
    for d in monthly_dates:
        regime, result = _classify_regime_for_date(
            benchmark_series, d, bull_threshold, bear_threshold
        )
        results.append((regime, result))
    return results


def _compute_grid_point(
    benchmark_series: pd.Series,
    monthly_dates: List[pd.Timestamp],
    base_weights: Dict[str, float],
    bull_threshold: float,
    bear_threshold: float,
) -> RegimeGridPoint:
    """计算单个网格点的 regime 分类与因子权重统计。"""
    classifications = _monthly_regime_classifications(
        benchmark_series, monthly_dates, bull_threshold, bear_threshold
    )

    n_total = len(classifications)
    regimes = [c[0] for c in classifications]
    results = [c[1] for c in classifications]

    n_bull = sum(1 for r in regimes if r == REGIME_BULL)
    n_bear = sum(1 for r in regimes if r == REGIME_BEAR)
    n_osc = sum(1 for r in regimes if r == REGIME_OSCILLATION)

    # 计算每月调整后的因子权重，追踪均值和波动
    cfg = RegimeConfig(
        bull_return_threshold=bull_threshold,
        bear_return_threshold=bear_threshold,
    )

    momentum_weights: List[float] = []
    reversal_weights: List[float] = []
    lowvol_weights: List[float] = []
    size_weights: List[float] = []
    all_weight_vectors: List[np.ndarray] = []

    for regime, result in zip(regimes, results):
        adjusted = get_regime_weights(base_weights, regime, cfg=cfg, regime_result=result)
        wv = np.array(list(adjusted.values()), dtype=np.float64)

        # 提取各类因子权重（使用子串匹配，与 regime.py 一致）
        m_w = 0.0
        r_w = 0.0
        l_w = 0.0
        s_w = 0.0
        for k, v in adjusted.items():
            if "momentum" in k:
                m_w += abs(v)
            if "reversal" in k or "recent_return" in k:
                r_w += abs(v)
            if "vol" in k or "atr" in k or "max_single_day_drop" in k:
                l_w += abs(v)
            if "market_cap" in k or "size" in k:
                s_w += abs(v)

        momentum_weights.append(m_w)
        reversal_weights.append(r_w)
        lowvol_weights.append(l_w)
        size_weights.append(s_w)
        all_weight_vectors.append(wv)

    # 权重向量变化的波动（相邻月 L1 距离的 std）
    weight_diffs = []
    for i in range(1, len(all_weight_vectors)):
        w_prev = all_weight_vectors[i - 1]
        w_curr = all_weight_vectors[i]
        if len(w_prev) == len(w_curr):
            weight_diffs.append(float(np.sum(np.abs(w_curr - w_prev))))
    wt_std = float(np.std(weight_diffs)) if weight_diffs else 0.0

    return RegimeGridPoint(
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
        n_bull=n_bull,
        n_bear=n_bear,
        n_oscillation=n_osc,
        bull_pct=n_bull / n_total if n_total > 0 else 0.0,
        bear_pct=n_bear / n_total if n_total > 0 else 0.0,
        oscillation_pct=n_osc / n_total if n_total > 0 else 0.0,
        mean_momentum_weight=float(np.mean(momentum_weights)) if momentum_weights else 0.0,
        mean_reversal_weight=float(np.mean(reversal_weights)) if reversal_weights else 0.0,
        mean_lowvol_weight=float(np.mean(lowvol_weights)) if lowvol_weights else 0.0,
        mean_size_weight=float(np.mean(size_weights)) if size_weights else 0.0,
        weight_turnover_std=wt_std,
    )


def run_regime_sensitivity_grid(
    benchmark_series: pd.Series,
    monthly_dates: List[pd.Timestamp],
    base_weights: Dict[str, float],
    *,
    bull_thresholds: Tuple[float, ...] = (0.03, 0.05, 0.07),
    bear_thresholds: Tuple[float, ...] = (0.03, 0.04, 0.05),
) -> RegimeSensitivityReport:
    """在参数网格上评估 regime 分类敏感性。

    Parameters
    ----------
    benchmark_series: 大盘基准日收益序列
    monthly_dates: 月度截面日期列表（每个回测月的月末）
    base_weights: 基准因子权重字典
    bull_thresholds: 牛市场阈值候选值
    bear_thresholds: 熊市场阈值候选值

    Returns
    -------
    RegimeSensitivityReport 包含所有网格点的分类与权重统计。
    """
    grid_points: List[RegimeGridPoint] = []
    for bull_t in bull_thresholds:
        for bear_t in bear_thresholds:
            gp = _compute_grid_point(
                benchmark_series, monthly_dates, base_weights, bull_t, bear_t
            )
            grid_points.append(gp)

    # 构建摘要
    bull_pcts = [g.bull_pct for g in grid_points]
    bear_pcts = [g.bear_pct for g in grid_points]
    osc_pcts = [g.oscillation_pct for g in grid_points]
    wt_stds = [g.weight_turnover_std for g in grid_points]

    summary = {
        "bull_pct_range": (round(min(bull_pcts), 3), round(max(bull_pcts), 3)),
        "bull_pct_spread": round(max(bull_pcts) - min(bull_pcts), 3),
        "bear_pct_range": (round(min(bear_pcts), 3), round(max(bear_pcts), 3)),
        "bear_pct_spread": round(max(bear_pcts) - min(bear_pcts), 3),
        "oscillation_pct_range": (round(min(osc_pcts), 3), round(max(osc_pcts), 3)),
        "weight_turnover_std_range": (round(min(wt_stds), 4), round(max(wt_stds), 4)),
        "default_point": {
            "bull_threshold": 0.05,
            "bear_threshold": 0.04,
        },
        "most_stable_point": _find_most_stable(grid_points),
        "n_grid_points": len(grid_points),
    }

    # 查找默认参数对应的网格点
    default_gp = None
    for gp in grid_points:
        if abs(gp.bull_threshold - 0.05) < 1e-6 and abs(gp.bear_threshold - 0.04) < 1e-6:
            default_gp = gp
            break

    if default_gp:
        summary["default_regime_distribution"] = {
            "bull_pct": round(default_gp.bull_pct, 3),
            "bear_pct": round(default_gp.bear_pct, 3),
            "oscillation_pct": round(default_gp.oscillation_pct, 3),
        }

    return RegimeSensitivityReport(
        grid=grid_points,
        base_config={
            "bull_thresholds": list(bull_thresholds),
            "bear_thresholds": list(bear_thresholds),
            "base_weights_keys": list(base_weights.keys()),
        },
        n_months_analyzed=len(monthly_dates),
        summary=summary,
    )


def _find_most_stable(grid_points: List[RegimeGridPoint]) -> Dict[str, Any]:
    """找到权重换手波动最小的参数组合。"""
    if not grid_points:
        return {}
    best = min(grid_points, key=lambda g: g.weight_turnover_std)
    return {
        "bull_threshold": best.bull_threshold,
        "bear_threshold": best.bear_threshold,
        "weight_turnover_std": round(best.weight_turnover_std, 4),
        "regime_distribution": {
            "bull_pct": round(best.bull_pct, 3),
            "bear_pct": round(best.bear_pct, 3),
            "oscillation_pct": round(best.oscillation_pct, 3),
        },
    }


def regime_sensitivity_markdown(report: RegimeSensitivityReport) -> str:
    """将 RegimeSensitivityReport 渲染为 Markdown 表格，供月度报告嵌入。"""
    lines: List[str] = []
    lines.append("## Regime 参数敏感性网格报告")
    lines.append("")
    lines.append(f"- 分析月份数: {report.n_months_analyzed}")
    lines.append(
        f"- 网格: bull_threshold ∈ {report.base_config['bull_thresholds']}, "
        f"bear_threshold ∈ {report.base_config['bear_thresholds']}"
    )
    lines.append(f"- 网格点数: {report.summary['n_grid_points']}")
    lines.append("")

    # 分类分布范围
    s = report.summary
    lines.append("### 分类分布敏感性")
    lines.append("")
    lines.append("| 指标 | 最小值 | 最大值 | 极差 |")
    lines.append("|------|--------|--------|------|")
    lines.append(
        f"| Bull % | {s['bull_pct_range'][0]:.1%} | {s['bull_pct_range'][1]:.1%} "
        f"| {s['bull_pct_spread']:.1%} |"
    )
    lines.append(
        f"| Bear % | {s['bear_pct_range'][0]:.1%} | {s['bear_pct_range'][1]:.1%} "
        f"| {s['bear_pct_spread']:.1%} |"
    )
    lines.append(
        f"| Oscillation % | {s['oscillation_pct_range'][0]:.1%} | "
        f"{s['oscillation_pct_range'][1]:.1%} | - |"
    )
    lines.append("")

    # 默认参数
    dp = s.get("default_regime_distribution", {})
    if dp:
        lines.append("### 默认参数 (bull=0.05, bear=0.04)")
        lines.append("")
        lines.append(
            f"- Bull: {dp.get('bull_pct', 0):.1%}  "
            f"Bear: {dp.get('bear_pct', 0):.1%}  "
            f"Oscillation: {dp.get('oscillation_pct', 0):.1%}"
        )
        lines.append("")

    # 完整网格表
    lines.append("### 完整网格")
    lines.append("")
    lines.append(
        "| bull_thr | bear_thr | Bull% | Bear% | Osc% | "
        "MomentumW | ReversalW | LowVolW | SizeW | WtStd |"
    )
    lines.append(
        "|-----------|----------|-------|-------|------|"
        "-----------|-----------|---------|-------|-------|"
    )
    for g in report.grid:
        lines.append(
            f"| {g.bull_threshold:.2f} | {g.bear_threshold:.2f} | "
            f"{g.bull_pct:.1%} | {g.bear_pct:.1%} | {g.oscillation_pct:.1%} | "
            f"{g.mean_momentum_weight:.3f} | {g.mean_reversal_weight:.3f} | "
            f"{g.mean_lowvol_weight:.3f} | {g.mean_size_weight:.3f} | "
            f"{g.weight_turnover_std:.4f} |"
        )
    lines.append("")

    # 建议
    ms = s.get("most_stable_point", {})
    if ms:
        lines.append("### 最稳定参数")
        lines.append("")
        lines.append(
            f"- bull_threshold={ms['bull_threshold']}, "
            f"bear_threshold={ms['bear_threshold']}, "
            f"权重换手 std={ms['weight_turnover_std']:.4f}"
        )
        lines.append("")

    return "\n".join(lines)
