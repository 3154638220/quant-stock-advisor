"""M12: 组合方法比较 —— 等权 vs 风险平价 vs 最小方差 side-by-side。

在相同信号/候选池上对比不同组合构建方法的 Sharpe、超额、回撤、换手，
输出供 M10 压力测试和 M12 Promotion Package 使用。

Usage:
    from src.analysis.portfolio_method_compare import compare_portfolio_methods
    report = compare_portfolio_methods(asset_returns, weights_signal, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.statistical_tests import (
    newey_west_t_statistic,
    bootstrap_excess_ci,
    information_ratio,
    turnover_adjusted_ir,
)


@dataclass
class PortfolioMethodRow:
    """单个组合方法的回测结果摘要。"""

    method: str
    mean_excess_bps: float
    net_excess_bps: float
    sharpe: float
    max_drawdown: float
    mean_turnover: float
    volatility_ann: float
    win_rate: float
    n_months: int
    positive_months: int
    ir: float
    ir_adj: float
    nw_t: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    # 优化器诊断（仅协方差方法有）
    solver_success_rate: Optional[float] = None
    mean_effective_n: Optional[float] = None
    mean_condition_number: Optional[float] = None


@dataclass
class PortfolioMethodReport:
    """组合方法对比完整报告。"""

    rows: List[PortfolioMethodRow]
    base_config: Dict[str, Any]
    n_months: int
    summary: Dict[str, Any] = field(default_factory=dict)

    def best_by_sharpe(self) -> PortfolioMethodRow:
        return max(self.rows, key=lambda r: r.sharpe)

    def best_by_ir(self) -> PortfolioMethodRow:
        return max(self.rows, key=lambda r: r.ir if np.isfinite(r.ir) else -999)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_config": self.base_config,
            "n_months": self.n_months,
            "methods": [
                {
                    "method": r.method,
                    "mean_excess_bps": round(r.mean_excess_bps, 1),
                    "net_excess_bps": round(r.net_excess_bps, 1),
                    "sharpe": round(r.sharpe, 3),
                    "max_drawdown": round(r.max_drawdown, 4),
                    "mean_turnover": round(r.mean_turnover, 3),
                    "volatility_ann": round(r.volatility_ann, 4),
                    "win_rate": round(r.win_rate, 3),
                    "ir": round(r.ir, 3),
                    "ir_adj": round(r.ir_adj, 3),
                    "nw_t": round(r.nw_t, 2),
                    "bootstrap_ci": (round(r.bootstrap_ci_lower, 1), round(r.bootstrap_ci_upper, 1)),
                    "solver_success_rate": (
                        round(r.solver_success_rate, 3) if r.solver_success_rate is not None else None
                    ),
                    "mean_effective_n": (
                        round(r.mean_effective_n, 1) if r.mean_effective_n is not None else None
                    ),
                }
                for r in self.rows
            ],
            "summary": self.summary,
        }


def _run_single_method_backtest(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    base_config: BacktestConfig,
) -> Dict[str, Any]:
    """运行单次回测并提取月度统计。"""
    bt = run_backtest(asset_returns, weights_signal, config=base_config)
    monthly = bt.panel.to_monthly_table() if hasattr(bt.panel, "to_monthly_table") else pd.DataFrame()

    if monthly.empty:
        return {
            "mean_excess": 0.0,
            "net_excess": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "mean_turnover": 0.0,
            "volatility_ann": 0.0,
            "win_rate": 0.0,
            "n_months": 0,
            "positive_months": 0,
            "monthly_excess": np.array([]),
            "monthly_turnover": np.array([]),
        }

    excess = monthly.get("excess_return", pd.Series([0.0]))
    turnover = monthly.get("turnover", pd.Series([0.0]))
    n_months = len(monthly)
    pos = int((excess > 0).sum())

    return {
        "mean_excess": float(bt.panel.mean_excess_return),
        "net_excess": float(bt.panel.net_excess_return),
        "sharpe": float(bt.panel.sharpe_ratio),
        "max_drawdown": float(bt.panel.max_drawdown),
        "mean_turnover": float(bt.panel.mean_turnover),
        "volatility_ann": float(bt.panel.volatility_ann)
        if hasattr(bt.panel, "volatility_ann")
        else float(np.std(excess) * np.sqrt(12)),
        "win_rate": pos / n_months if n_months > 0 else 0.0,
        "n_months": n_months,
        "positive_months": pos,
        "monthly_excess": excess.to_numpy(dtype=np.float64),
        "monthly_turnover": turnover.to_numpy(dtype=np.float64),
    }


def compare_portfolio_methods(
    asset_returns: pd.DataFrame,
    weights_equal: pd.DataFrame,
    weights_score: Optional[pd.DataFrame] = None,
    weights_risk_parity: Optional[pd.DataFrame] = None,
    weights_min_variance: Optional[pd.DataFrame] = None,
    weights_mean_variance: Optional[pd.DataFrame] = None,
    *,
    base_config: Optional[BacktestConfig] = None,
    solver_diag: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> PortfolioMethodReport:
    """对比多种组合构建方法的回测表现。

    Parameters
    ----------
    asset_returns: 资产日收益宽表
    weights_equal: 等权配置的权重宽表
    weights_score: score-weight 配置的权重宽表（可选）
    weights_risk_parity: 风险平价权重宽表（可选）
    weights_min_variance: 最小方差权重宽表（可选）
    weights_mean_variance: 均值-方差权重宽表（可选）
    base_config: 基础回测配置
    solver_diag: 每种协方差方法的优化器诊断（按月），key 为方法名

    Returns
    -------
    PortfolioMethodReport 包含所有方法的对比结果。
    """
    cfg = base_config or BacktestConfig()
    rows: List[PortfolioMethodRow] = []

    methods_to_test: List[Tuple[str, pd.DataFrame]] = [
        ("equal_weight", weights_equal),
    ]
    if weights_score is not None:
        methods_to_test.append(("score_weight", weights_score))
    if weights_risk_parity is not None:
        methods_to_test.append(("risk_parity", weights_risk_parity))
    if weights_min_variance is not None:
        methods_to_test.append(("min_variance", weights_min_variance))
    if weights_mean_variance is not None:
        methods_to_test.append(("mean_variance", weights_mean_variance))

    for method, wdf in methods_to_test:
        result = _run_single_method_backtest(asset_returns, wdf, cfg)

        monthly_excess = result["monthly_excess"]
        monthly_turnover = result["monthly_turnover"]

        # 统计检验
        nw = newey_west_t_statistic(monthly_excess, max_lag=6)
        boot = bootstrap_excess_ci(monthly_excess)
        ir = information_ratio(monthly_excess)
        ir_adj = turnover_adjusted_ir(monthly_excess, monthly_turnover)

        # 优化器诊断（仅协方差方法）
        solver_success_rate = None
        mean_eff_n = None
        mean_cond = None
        if solver_diag and method in solver_diag:
            diags = solver_diag[method]
            if diags:
                n_success = sum(1 for d in diags if d.get("solver_success", False))
                solver_success_rate = n_success / len(diags) if diags else None
                eff_ns = [
                    d.get("weights", {}).get("effective_n", float("nan"))
                    for d in diags
                ]
                mean_eff_n = float(np.nanmean(eff_ns)) if eff_ns else None
                conds = [
                    d.get("covariance", {}).get("condition_number", float("nan"))
                    for d in diags
                ]
                mean_cond = float(np.nanmean(conds)) if conds else None

        rows.append(PortfolioMethodRow(
            method=method,
            mean_excess_bps=float(result["mean_excess"]) * 1e4,
            net_excess_bps=float(result["net_excess"]) * 1e4,
            sharpe=float(result["sharpe"]),
            max_drawdown=float(result["max_drawdown"]),
            mean_turnover=float(result["mean_turnover"]),
            volatility_ann=float(result["volatility_ann"]),
            win_rate=float(result["win_rate"]),
            n_months=int(result["n_months"]),
            positive_months=int(result["positive_months"]),
            ir=float(ir.get("ir", float("nan"))),
            ir_adj=float(ir_adj.get("ir_adj", float("nan"))),
            nw_t=float(nw.get("nw_t", float("nan"))),
            bootstrap_ci_lower=float(boot.get("ci_lower", float("nan"))),
            bootstrap_ci_upper=float(boot.get("ci_upper", float("nan"))),
            solver_success_rate=solver_success_rate,
            mean_effective_n=mean_eff_n,
            mean_condition_number=mean_cond,
        ))

    # 构建摘要
    best_sharpe = max(rows, key=lambda r: r.sharpe)
    best_ir = max(rows, key=lambda r: r.ir if np.isfinite(r.ir) else -999)
    eq_row = next((r for r in rows if r.method == "equal_weight"), None)

    summary = {
        "best_by_sharpe": best_sharpe.method,
        "best_sharpe": round(best_sharpe.sharpe, 3),
        "best_by_ir": best_ir.method,
        "best_ir": round(best_ir.ir, 3),
        "equal_weight_sharpe": round(eq_row.sharpe, 3) if eq_row else None,
        "methods_tested": len(rows),
    }

    # 等权 vs 最优的差异
    if eq_row and best_sharpe.method != "equal_weight":
        summary["sharpe_improvement"] = round(best_sharpe.sharpe - eq_row.sharpe, 3)
        summary["sharpe_improvement_pct"] = round(
            (best_sharpe.sharpe - eq_row.sharpe) / max(abs(eq_row.sharpe), 0.01) * 100, 1
        )

    cost_bps = 10.0
    if cfg.cost_params is not None:
        cost_bps = round((cfg.cost_params.buy_fraction() + cfg.cost_params.sell_fraction()) * 1e4, 1)

    return PortfolioMethodReport(
        rows=rows,
        base_config={
            "execution_mode": cfg.execution_mode,
            "cost_total_bps": cost_bps,
        },
        n_months=rows[0].n_months if rows else 0,
        summary=summary,
    )


def portfolio_method_markdown(report: PortfolioMethodReport) -> str:
    """将 PortfolioMethodReport 渲染为 Markdown 表格。"""
    lines: List[str] = []
    lines.append("## 组合方法对比报告")
    lines.append("")
    lines.append(f"- 回测月份数: {report.n_months}")
    lines.append(f"- 执行模式: {report.base_config.get('execution_mode', 'N/A')}")
    lines.append(f"- 成本假设: {report.base_config.get('cost_total_bps', 'N/A')} bps 双边")
    lines.append("")

    lines.append("### 核心指标对比")
    lines.append("")
    lines.append(
        "| 方法 | 超额(bps) | 净超额(bps) | Sharpe | 最大回撤 | "
        "换手率 | 年化波动 | 胜率 | IR | IR_adj | NW t |"
    )
    lines.append(
        "|------|-----------|-------------|--------|----------|"
        "--------|----------|------|-----|--------|-------|"
    )
    for r in report.rows:
        lines.append(
            f"| {r.method} | {r.mean_excess_bps:.1f} | {r.net_excess_bps:.1f} | "
            f"{r.sharpe:.3f} | {r.max_drawdown:.4f} | {r.mean_turnover:.3f} | "
            f"{r.volatility_ann:.4f} | {r.win_rate:.1%} | {r.ir:.3f} | "
            f"{r.ir_adj:.3f} | {r.nw_t:.2f} |"
        )
    lines.append("")

    # Bootstrap CI
    lines.append("### Bootstrap 95% 置信区间 (超额 bps)")
    lines.append("")
    lines.append("| 方法 | 均值 | CI 下限 | CI 上限 |")
    lines.append("|------|------|---------|---------|")
    for r in report.rows:
        lines.append(
            f"| {r.method} | {r.mean_excess_bps:.1f} | "
            f"{r.bootstrap_ci_lower:.1f} | {r.bootstrap_ci_upper:.1f} |"
        )
    lines.append("")

    # 优化器诊断
    cov_methods = [r for r in report.rows if r.solver_success_rate is not None]
    if cov_methods:
        lines.append("### 优化器诊断")
        lines.append("")
        lines.append("| 方法 | 求解成功率 | 平均有效持仓数 | 平均条件数 |")
        lines.append("|------|------------|----------------|------------|")
        for r in cov_methods:
            lines.append(
                f"| {r.method} | {r.solver_success_rate:.1%} | "
                f"{r.mean_effective_n:.1f} | {r.mean_condition_number:.0f} |"
            )
        lines.append("")

    # 摘要
    s = report.summary
    lines.append("### 结论")
    lines.append("")
    lines.append(f"- Sharpe 最优: **{s['best_by_sharpe']}** (Sharpe={s['best_sharpe']:.3f})")
    lines.append(f"- IR 最优: **{s['best_by_ir']}** (IR={s['best_ir']:.3f})")
    if "sharpe_improvement" in s:
        lines.append(
            f"- 相对等权 Sharpe 提升: {s['sharpe_improvement']:.3f} "
            f"({s['sharpe_improvement_pct']:+.1f}%)"
        )
    lines.append("")

    return "\n".join(lines)
