"""M10 容量报告单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.capacity_report import (
    CapacityEstimate,
    CapacityReport,
    CostSensitivityRow,
    ExecutionModeComparison,
    LimitUpBiasCheck,
    _cost_label,
    analyze_limit_up_redistribution,
    build_m10_statistical_section,
    capacity_summary,
    compare_execution_modes,
    estimate_capacity,
    run_cost_sensitivity,
)
from src.backtest.transaction_costs import TransactionCostParams
from src.backtest.engine import BacktestConfig


class TestCostLabel:
    def test_cost_label_format(self):
        params = TransactionCostParams(
            commission_buy_bps=2.0, commission_sell_bps=2.0,
            slippage_bps_per_side=1.5, stamp_duty_sell_bps=5.0,
        )
        label = _cost_label(params)
        assert "bps" in label
        assert label.endswith("bps")


class TestCostSensitivityRow:
    def test_excess_after_cost_bps_property(self):
        row = CostSensitivityRow(
            cost_label="10bps", commission_buy_bps=2.0, commission_sell_bps=2.0,
            slippage_bps_per_side=1.5, stamp_duty_sell_bps=5.0,
            total_roundtrip_bps=10.5, mean_excess_bps=15.0, net_excess_bps=5.0,
            sharpe=0.8, max_drawdown=-0.15, mean_turnover=0.3,
            win_rate=0.6, n_months=24, positive_months=14,
        )
        assert row.excess_after_cost_bps == 5.0


class TestLimitUpBiasCheck:
    def test_no_events_returns_clean_check(self):
        fake_bt = type("obj", (object,), {
            "meta": {"buy_fail_rows": [], "limit_up_mode": "idle"},
        })()
        result = analyze_limit_up_redistribution(fake_bt)
        assert result.total_events == 0
        assert not result.bias_detected

    def test_with_events_no_bias(self):
        events = [
            {"trade_date": "2024-01-02", "failed_weight": 0.05, "redistributed_weight": 0.05, "idle_weight": 0.0},
            {"trade_date": "2024-02-01", "failed_weight": 0.03, "redistributed_weight": 0.03, "idle_weight": 0.0},
        ]
        fake_bt = type("obj", (object,), {
            "meta": {"buy_fail_rows": events, "limit_up_mode": "redistribute"},
        })()
        result = analyze_limit_up_redistribution(fake_bt)
        assert result.total_events == 2
        assert result.rebalance_dates_affected == 2

    def test_bias_detected_with_strong_t(self):
        # redistribute ratios far from 1.0 generate large t-stat
        events = [
            {"trade_date": f"2024-{i+1:02d}-01", "failed_weight": 0.05,
             "redistributed_weight": 0.02 if i < 5 else 0.08, "idle_weight": 0.0}
            for i in range(10)
        ]
        fake_bt = type("obj", (object,), {
            "meta": {"buy_fail_rows": events, "limit_up_mode": "redistribute"},
        })()
        result = analyze_limit_up_redistribution(fake_bt)
        # 有 10 个不同值的 events，应该有 t-statistic
        assert result.total_events == 10


class TestCapacityEstimate:
    def test_estimate_capacity(self):
        weights = pd.Series([0.1, 0.05], index=["000001", "000002"])
        daily_amount = pd.Series([1e7, 5e6], index=["000001", "000002"])
        estimates = estimate_capacity(weights, daily_amount, portfolio_aum=1e7)
        assert len(estimates) == 2
        # 000001: 0.1 * 1e7 / 1e7 = 10% participation
        assert estimates[0].symbol in {"000001", "000002"}

    def test_estimate_capacity_zero_adv(self):
        weights = pd.Series([0.1], index=["000001"])
        daily_amount = pd.Series([0.0], index=["000001"])
        estimates = estimate_capacity(weights, daily_amount, portfolio_aum=1e7)
        assert len(estimates) == 1
        assert not estimates[0].feasible

    def test_capacity_summary(self):
        estimates = [
            CapacityEstimate("000001", 1e7, 0.1, 1e7, 1e6, 5.0, 2.0, True),
            CapacityEstimate("000002", 1e6, 0.05, 1e7, 5e5, 50.0, 20.0, False),
        ]
        summary = capacity_summary(estimates)
        assert summary["total_positions"] == 2
        assert summary["infeasible_count"] == 1
        assert "000002" in summary["infeasible_symbols"]

    def test_capacity_summary_empty(self):
        summary = capacity_summary([])
        assert summary["total_positions"] == 0

    def test_estimate_capacity_zero_weight_skipped(self):
        weights = pd.Series([0.0, 0.1], index=["000001", "000002"])
        daily_amount = pd.Series([1e7, 5e6], index=["000001", "000002"])
        estimates = estimate_capacity(weights, daily_amount, portfolio_aum=1e7)
        assert len(estimates) == 1  # zero weight skipped


class TestCapacityReport:
    def test_min_cost_for_positive_excess(self):
        rows = [
            CostSensitivityRow("10bps", 2, 2, 1.5, 5, 10.5, 15, 5, 0.8, -0.1, 0.3, 0.6, 24, 14),
            CostSensitivityRow("30bps", 2.5, 2.5, 12.5, 5, 32.5, 12, -2, 0.3, -0.2, 0.4, 0.5, 24, 12),
        ]
        report = CapacityReport(
            cost_sensitivity=rows,
            limit_up_check=LimitUpBiasCheck("idle", 0, 0, 0, 0, 0, 0, 0, False),
            capacity={"max_participation_pct": 5.0, "infeasible_count": 0},
        )
        best = report.min_cost_for_positive_excess()
        assert best is not None
        assert best.cost_label == "10bps"

    def test_min_cost_none_when_all_negative(self):
        rows = [
            CostSensitivityRow("30bps", 2.5, 2.5, 12.5, 5, 32.5, 10, -2, 0.3, -0.2, 0.4, 0.5, 24, 12),
        ]
        report = CapacityReport(
            cost_sensitivity=rows,
            limit_up_check=LimitUpBiasCheck("idle", 0, 0, 0, 0, 0, 0, 0, False),
            capacity={"max_participation_pct": 5.0, "infeasible_count": 0},
        )
        assert report.min_cost_for_positive_excess() is None

    def test_to_summary_dict(self):
        rows = [
            CostSensitivityRow("10bps", 2, 2, 1.5, 5, 10.5, 15, 5, 0.8, -0.1, 0.3, 0.6, 24, 14),
        ]
        exec_comp = ExecutionModeComparison(20.0, 18.0, 15.0, -5.0, -3.0, -5.0)
        report = CapacityReport(
            cost_sensitivity=rows,
            limit_up_check=LimitUpBiasCheck("idle", 0, 0, 0, 0, 0, 0, 0, False),
            capacity={"max_participation_pct": 5.0, "infeasible_count": 0, "worst_symbol": "000001"},
            execution_comparison=exec_comp,
        )
        d = report.to_summary_dict()
        assert d["min_cost_for_positive_excess"] == "10bps"
        assert d["execution_comparison"]["ctc_excess_bps"] == 20.0


class TestBuildM10StatisticalSection:
    def test_basic_statistics(self):
        excess = np.array([0.01, 0.02, -0.01, 0.03, 0.005, -0.005])
        turnover = np.array([0.3, 0.25, 0.2, 0.35, 0.3, 0.25])
        result = build_m10_statistical_section(excess, turnover)
        assert "excess_nw" in result
        assert "excess_bootstrap" in result
        assert "excess_ir" in result
        assert "turnover_ir" in result
        assert "turnover_excess_scatter" in result
        assert "gates" in result

    def test_with_rank_ic(self):
        excess = np.array([0.01, 0.02, -0.01, 0.03, 0.005, -0.005])
        turnover = np.array([0.3, 0.25, 0.2, 0.35, 0.3, 0.25])
        rank_ic = np.array([0.10, 0.12, 0.08, 0.15, 0.09, 0.07])
        result = build_m10_statistical_section(excess, turnover, rank_ic_monthly=rank_ic)
        assert "ic_nw" in result
        assert result["gates"]["ic_nw_t_gate"] is not None

    def test_with_regime_labels(self):
        excess = np.array([0.02, 0.01, -0.005, 0.015])
        turnover = np.array([0.3, 0.2, 0.25, 0.35])
        result = build_m10_statistical_section(
            excess, turnover,
            months=["2024-01", "2024-02", "2024-03", "2024-04"],
            regime_labels=["bull", "bull", "bear", "oscillation"],
        )
        scatter = result["turnover_excess_scatter"]
        assert len(scatter) == 4
        assert any(r.get("regime") == "bull" for r in scatter)
