"""H2 换仓敏感性分析单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.rebalance_sensitivity import (
    RULE_LABELS,
    RebalanceSensitivityResult,
    sensitivity_summary_df,
)


def make_mock_returns(n_days=300, n_assets=10, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = rng.normal(0.0005, 0.02, (n_days, n_assets))
    symbols = [f"{600000 + i:06d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=symbols)


def make_mock_weights(returns_index, n_assets=10, top_k=5, seed=42):
    """生成等权 Top-K 权重（月度调仓）。"""
    rng = np.random.default_rng(seed)
    syms = [f"{600000 + i:06d}" for i in range(n_assets)]
    # 每月末一个信号日
    monthly_dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    data = np.zeros((len(monthly_dates), n_assets))
    for i, d in enumerate(monthly_dates):
        chosen = rng.choice(n_assets, top_k, replace=False)
        data[i, chosen] = 1.0 / top_k
    return pd.DataFrame(data, index=monthly_dates, columns=syms)


class TestRebalanceSensitivityResult:
    def test_dataclass_fields(self):
        r = RebalanceSensitivityResult(
            rule="M", signal_count=24,
            mean_excess_bps=15.0, mean_turnover=0.3,
            mean_cost_bps=5.0, net_excess_bps=10.0,
            sharpe=0.8, max_drawdown=-0.15,
        )
        assert r.rule == "M"
        assert r.net_excess_bps == 10.0


class TestSensitivitySummaryDf:
    def test_single_rule(self):
        results = [
            RebalanceSensitivityResult(
                rule="M", signal_count=24,
                mean_excess_bps=15.0, mean_turnover=0.3,
                mean_cost_bps=5.0, net_excess_bps=10.0,
                sharpe=0.8, max_drawdown=-0.15,
            ),
        ]
        df = sensitivity_summary_df(results)
        assert len(df) == 1
        assert df.iloc[0]["rule"] == "M"
        assert "月频" in str(df.iloc[0]["换仓频率"])

    def test_multiple_rules_sorted(self):
        results = [
            RebalanceSensitivityResult(
                rule=rule, signal_count=12 if rule == "Q" else 24,
                mean_excess_bps=20.0 if rule == "W" else 15.0,
                mean_turnover=0.5 if rule == "W" else 0.3,
                mean_cost_bps=10.0 if rule == "W" else 5.0,
                net_excess_bps=10.0 if rule == "W" else 12.0,
                sharpe=0.7, max_drawdown=-0.15,
            )
            for rule in ["W", "M", "Q"]
        ]
        df = sensitivity_summary_df(results)
        assert len(df) == 3
        # 不要求特定排序（summary 不做排序）


class TestRuleLabels:
    def test_all_supported_rules_have_labels(self):
        from src.analysis.rebalance_sensitivity import SUPPORTED_RULES
        for rule in SUPPORTED_RULES:
            assert rule in RULE_LABELS
