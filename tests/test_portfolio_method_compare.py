"""Tests for src/analysis/portfolio_method_compare.py — M12 portfolio method comparison."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.portfolio_method_compare import (
    PortfolioMethodReport,
    PortfolioMethodRow,
    compare_portfolio_methods,
    portfolio_method_markdown,
)


class MockBacktestResult:
    """Simulate a BacktestResult with a panel that has the required attributes."""
    class Panel:
        mean_excess_return = 0.003
        net_excess_return = 0.002
        sharpe_ratio = 0.8
        max_drawdown = -0.15
        mean_turnover = 0.3
        volatility_ann = 0.18

        def to_monthly_table(self):
            n_months = 36
            rng = np.random.default_rng(42)
            return pd.DataFrame({
                "excess_return": rng.normal(0.003, 0.04, size=n_months),
                "turnover": np.full(n_months, 0.3),
            })

    def __init__(self):
        self.panel = self.Panel()
        self.meta = {}


def _make_mock_returns(n_assets: int = 10, n_days: int = 500) -> pd.DataFrame:
    """生成模拟资产日收益宽表（index=dates, columns=symbols）。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-02-01", periods=n_days, freq="B")
    symbols = [f"{i:06d}" for i in range(1, n_assets + 1)]
    data = rng.normal(0.0005, 0.02, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=symbols)


def _make_equal_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """从收益表生成每月等权配置（index=dates, columns=symbols）。"""
    s = pd.Series(0, index=returns.index)
    monthly = s.resample("ME").last().index
    symbols = returns.columns.tolist()
    n = len(symbols)
    records = [{"date": d, **{sym: 1.0 / n for sym in symbols}} for d in monthly]
    wdf = pd.DataFrame(records).set_index("date")
    wdf.index = pd.to_datetime(wdf.index)
    return wdf


def _make_score_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """生成模拟 score-weight（基于随机得分）。"""
    s = pd.Series(0, index=returns.index)
    monthly = s.resample("ME").last().index
    symbols = returns.columns.tolist()
    n = len(symbols)
    rng = np.random.default_rng(123)
    records = []
    for d in monthly:
        scores = np.maximum(rng.uniform(0.5, 1.5, size=n), 0)
        w = scores / scores.sum()
        records.append({"date": d, **{sym: w[i] for i, sym in enumerate(symbols)}})
    wdf = pd.DataFrame(records).set_index("date")
    wdf.index = pd.to_datetime(wdf.index)
    return wdf


class TestComparePortfolioMethods:
    """测试带 mock 的组合方法对比逻辑。"""

    def test_basic_comparison_equal_only(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq)
        assert len(report.rows) == 1
        assert report.rows[0].method == "equal_weight"
        assert "best_by_sharpe" in report.summary

    def test_comparison_with_score_weights(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        w_sc = _make_score_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq, weights_score=w_sc)
        assert len(report.rows) == 2
        methods = {r.method for r in report.rows}
        assert methods == {"equal_weight", "score_weight"}

    def test_comparison_with_cov_methods(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        w_rp = w_eq.copy()
        w_mv = w_eq.copy()
        solver_diags = {
            "risk_parity": [
                {"solver_success": True, "weights": {"effective_n": 8.5},
                 "covariance": {"condition_number": 120.0}}
            ] * 12,
            "min_variance": [
                {"solver_success": True, "weights": {"effective_n": 5.2},
                 "covariance": {"condition_number": 350.0}}
            ] * 12,
        }
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(
                rets, w_eq, weights_risk_parity=w_rp, weights_min_variance=w_mv,
                solver_diag=solver_diags,
            )
        assert len(report.rows) == 3
        rp_row = next(r for r in report.rows if r.method == "risk_parity")
        assert rp_row.solver_success_rate == 1.0
        assert rp_row.mean_effective_n == 8.5

    def test_statistical_fields_populated(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq)
        r = report.rows[0]
        assert np.isfinite(r.ir) or np.isnan(r.ir)
        assert np.isfinite(r.nw_t) or np.isnan(r.nw_t)
        assert r.bootstrap_ci_lower is not None

    def test_to_dict_serializable(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq)
        d = report.to_dict()
        assert "methods" in d
        assert len(d["methods"]) == 1
        assert "summary" in d

    def test_best_by_sharpe_and_ir(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        w_sc = _make_score_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq, weights_score=w_sc)
        best_s = report.best_by_sharpe()
        best_i = report.best_by_ir()
        assert best_s.method in ("equal_weight", "score_weight")
        assert best_i.method in ("equal_weight", "score_weight")


class TestPortfolioMethodMarkdown:
    def test_produces_valid_markdown(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(rets, w_eq)
        md = portfolio_method_markdown(report)
        assert "## 组合方法对比报告" in md
        assert "| 方法 | 超额(bps)" in md
        assert "### Bootstrap" in md
        assert "### 结论" in md

    def test_includes_cov_diagnostics_when_present(self):
        rets = _make_mock_returns(10, 500)
        w_eq = _make_equal_weights(rets)
        w_rp = w_eq.copy()
        solver_diags = {
            "risk_parity": [
                {"solver_success": True, "weights": {"effective_n": 8.0},
                 "covariance": {"condition_number": 100.0}}
            ] * 12,
        }
        with patch("src.analysis.portfolio_method_compare.run_backtest",
                   return_value=MockBacktestResult()):
            report = compare_portfolio_methods(
                rets, w_eq, weights_risk_parity=w_rp, solver_diag=solver_diags,
            )
        md = portfolio_method_markdown(report)
        assert "### 优化器诊断" in md
        assert "求解成功率" in md


class TestPortfolioMethodRow:
    def test_row_creation(self):
        row = PortfolioMethodRow(
            method="test",
            mean_excess_bps=50.0,
            net_excess_bps=30.0,
            sharpe=0.8,
            max_drawdown=-0.15,
            mean_turnover=0.3,
            volatility_ann=0.18,
            win_rate=0.65,
            n_months=48,
            positive_months=31,
            ir=0.55,
            ir_adj=0.42,
            nw_t=2.1,
            bootstrap_ci_lower=10.0,
            bootstrap_ci_upper=50.0,
        )
        assert row.method == "test"
        assert row.sharpe == 0.8
        assert row.solver_success_rate is None

    def test_row_with_solver_diag(self):
        row = PortfolioMethodRow(
            method="risk_parity",
            mean_excess_bps=40.0,
            net_excess_bps=25.0,
            sharpe=0.7,
            max_drawdown=-0.12,
            mean_turnover=0.25,
            volatility_ann=0.15,
            win_rate=0.60,
            n_months=36,
            positive_months=22,
            ir=0.50,
            ir_adj=0.40,
            nw_t=1.8,
            bootstrap_ci_lower=5.0,
            bootstrap_ci_upper=45.0,
            solver_success_rate=0.95,
            mean_effective_n=7.5,
            mean_condition_number=150.0,
        )
        assert row.solver_success_rate == 0.95
        assert row.mean_effective_n == 7.5


class TestPortfolioMethodReport:
    def test_best_by_sharpe_selects_highest(self):
        r1 = PortfolioMethodRow("a", 50, 40, 0.5, -0.1, 0.2, 0.15, 0.6, 12, 7,
                                0.4, 0.3, 1.5, 10, 40)
        r2 = PortfolioMethodRow("b", 60, 45, 0.9, -0.1, 0.2, 0.15, 0.6, 12, 7,
                                0.7, 0.5, 2.0, 20, 50)
        r3 = PortfolioMethodRow("c", 40, 30, 0.3, -0.1, 0.2, 0.15, 0.6, 12, 7,
                                0.3, 0.2, 1.0, 5, 30)
        report = PortfolioMethodReport(
            rows=[r1, r2, r3],
            base_config={},
            n_months=12,
            summary={"best_by_sharpe": "b", "best_sharpe": 0.9},
        )
        assert report.best_by_sharpe().method == "b"
        assert report.best_by_ir().method == "b"

    def test_empty_rows_best_by_sharpe_raises(self):
        report = PortfolioMethodReport(rows=[], base_config={}, n_months=0)
        with pytest.raises(ValueError):
            report.best_by_sharpe()
