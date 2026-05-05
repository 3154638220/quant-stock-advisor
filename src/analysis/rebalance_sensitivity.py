"""H2: 换仓频率 vs 净超额敏感性分析。

分析不同换仓频率（周/月/双月/季）下的净超额变化，
帮助选择最优换仓节奏。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.pipeline.monthly_dataset import select_month_end_signal_dates

SUPPORTED_RULES = ("W", "M", "BM", "Q")
RULE_LABELS = {"W": "周频", "M": "月频", "BM": "双月频", "Q": "季频"}


@dataclass
class RebalanceSensitivityResult:
    rule: str
    signal_count: int
    mean_excess_bps: float
    mean_turnover: float
    mean_cost_bps: float
    net_excess_bps: float
    sharpe: float
    max_drawdown: float


def analyze_rebalance_sensitivity(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    *,
    rules: tuple[str, ...] = SUPPORTED_RULES,
    base_config: BacktestConfig | None = None,
) -> list[RebalanceSensitivityResult]:
    """遍历换仓频率，比较各频率的净超额与换手成本。

    Parameters
    ----------
    asset_returns: 资产日收益宽表
    weights_signal: 信号权重（按调仓日索引）
    rules: 要测试的换仓频率
    base_config: 基础回测配置（成本、执行模式等共享）

    Returns
    -------
    各频率的结果列表，按 net_excess_bps 降序排列。
    """
    cfg = base_config or BacktestConfig()
    results: list[RebalanceSensitivityResult] = []

    for rule in rules:
        rule_cfg = BacktestConfig(
            cost_params=cfg.cost_params,
            risk_free_daily=cfg.risk_free_daily,
            periods_per_year=cfg.periods_per_year,
            max_gross_exposure=cfg.max_gross_exposure,
            execution_mode=cfg.execution_mode,
            execution_lag=cfg.execution_lag,
            limit_up_mode=cfg.limit_up_mode,
            limit_up_open_mask=cfg.limit_up_open_mask,
            rebalance_rule=rule,
            vwap_slippage_bps_per_side=cfg.vwap_slippage_bps_per_side,
            vwap_impact_bps=cfg.vwap_impact_bps,
            use_tiered_impact=cfg.use_tiered_impact,
            tiered_impact=cfg.tiered_impact,
        )
        bt = run_backtest(asset_returns, weights_signal, config=rule_cfg)
        signal_dates = select_month_end_signal_dates(
            asset_returns.index, rebalance_rule=rule,
        )

        results.append(RebalanceSensitivityResult(
            rule=rule,
            signal_count=len(signal_dates),
            mean_excess_bps=float(bt.panel.mean_excess_return) * 1e4,
            mean_turnover=float(bt.panel.mean_turnover),
            mean_cost_bps=float(bt.panel.mean_cost) * 1e4,
            net_excess_bps=float(bt.panel.net_excess_return) * 1e4,
            sharpe=float(bt.panel.sharpe_ratio),
            max_drawdown=float(bt.panel.max_drawdown),
        ))

    return sorted(results, key=lambda r: r.net_excess_bps, reverse=True)


def sensitivity_summary_df(results: list[RebalanceSensitivityResult]) -> pd.DataFrame:
    """将敏感性结果转为 DataFrame 摘要。"""
    rows = []
    for r in results:
        rows.append({
            "换仓频率": RULE_LABELS.get(r.rule, r.rule),
            "rule": r.rule,
            "信号日数": r.signal_count,
            "年化超额(bps)": round(r.mean_excess_bps, 1),
            "平均换手": round(r.mean_turnover, 3),
            "成本(bps)": round(r.mean_cost_bps, 1),
            "净超额(bps)": round(r.net_excess_bps, 1),
            "Sharpe": round(r.sharpe, 3),
            "最大回撤": f"{r.max_drawdown:.2%}",
        })
    return pd.DataFrame(rows)
