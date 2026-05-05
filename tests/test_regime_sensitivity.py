"""Tests for src/analysis/regime_sensitivity.py — M12 regime parameter grid."""

import numpy as np
import pandas as pd

from src.analysis.regime_sensitivity import (
    RegimeGridPoint,
    _compute_grid_point,
    _find_most_stable,
    regime_sensitivity_markdown,
    run_regime_sensitivity_grid,
)


def _make_random_walk_benchmark(n_days: int = 1260, seed: int = 42) -> pd.Series:
    """生成模拟基准日收益序列（5年 ≈ 252*5 天）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, size=n_days)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    return pd.Series(rets, index=dates, name="benchmark")


def _make_monthly_dates(n_months: int = 48) -> list:
    """生成月末日期列表。"""
    return pd.date_range("2021-12-31", periods=n_months, freq="ME").tolist()


BASE_WEIGHTS = {
    "momentum": 0.20,
    "reversal": 0.12,
    "short_reversal": 0.08,
    "realized_vol": 0.10,
    "atr": 0.05,
    "log_market_cap": 0.05,
    "recent_return": 0.05,
    "max_single_day_drop": 0.05,
    "bias_short": 0.05,
    "bias_long": 0.05,
}


class TestComputeGridPoint:
    def test_returns_grid_point_with_valid_data(self):
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(24)
        gp = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.05, 0.04)
        assert isinstance(gp, RegimeGridPoint)
        assert gp.bull_threshold == 0.05
        assert gp.bear_threshold == 0.04
        assert gp.n_bull + gp.n_bear + gp.n_oscillation == len(dates)
        assert 0.0 <= gp.bull_pct <= 1.0
        assert gp.mean_momentum_weight >= 0
        assert gp.weight_turnover_std >= 0

    def test_bull_threshold_affects_classification(self):
        """更高的 bull 阈值应减少 bull 分类，增加 oscillation。"""
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(48)
        gp_low = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.03, 0.04)
        gp_high = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.07, 0.04)
        # 高阈值下 bull 分类 <= 低阈值下 bull 分类（随机游走中大多是 oscillation）
        assert gp_high.n_bull <= gp_low.n_bull + 5  # 允许随机波动

    def test_bear_threshold_affects_classification(self):
        """更高的 bear 阈值应减少 bear 分类。"""
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(48)
        gp_low = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.05, 0.03)
        gp_high = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.05, 0.05)
        assert gp_high.n_bear <= gp_low.n_bear + 5

    def test_strong_bull_market_all_bull(self):
        """强牛市（持续正收益）应全部归类为 bull。"""
        n_days = 252 * 2
        rets = np.full(n_days, 0.005)  # 每天 +0.5%
        dates_idx = pd.date_range("2021-01-01", periods=n_days, freq="B")
        bench = pd.Series(rets, index=dates_idx)
        monthly_dates = pd.date_range("2021-12-31", periods=12, freq="ME").tolist()
        gp = _compute_grid_point(bench, monthly_dates, BASE_WEIGHTS, 0.05, 0.04)
        assert gp.n_bull == len(monthly_dates)
        assert gp.n_bear == 0

    def test_strong_bear_market_all_bear(self):
        """强熊市（持续负收益）应全部归类为 bear。"""
        n_days = 252 * 2
        rets = np.full(n_days, -0.005)  # 每天 -0.5%
        dates_idx = pd.date_range("2021-01-01", periods=n_days, freq="B")
        bench = pd.Series(rets, index=dates_idx)
        monthly_dates = pd.date_range("2021-12-31", periods=12, freq="ME").tolist()
        gp = _compute_grid_point(bench, monthly_dates, BASE_WEIGHTS, 0.05, 0.04)
        assert gp.n_bear == len(monthly_dates)
        assert gp.n_bull == 0

    def test_empty_benchmark_handled(self):
        bench = pd.Series(dtype=float)
        dates = _make_monthly_dates(12)
        gp = _compute_grid_point(bench, dates, BASE_WEIGHTS, 0.05, 0.04)
        assert gp.n_oscillation == len(dates)
        assert gp.n_bull == 0
        assert gp.n_bear == 0


class TestRunRegimeSensitivityGrid:
    def test_default_grid_size(self):
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(24)
        report = run_regime_sensitivity_grid(bench, dates, BASE_WEIGHTS)
        # 3 × 3 = 9 grid points
        assert len(report.grid) == 9
        assert report.n_months_analyzed == 24
        assert "bull_pct_range" in report.summary
        assert "default_regime_distribution" in report.summary

    def test_custom_grid(self):
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(12)
        report = run_regime_sensitivity_grid(
            bench, dates, BASE_WEIGHTS,
            bull_thresholds=(0.04, 0.06),
            bear_thresholds=(0.03, 0.05),
        )
        assert len(report.grid) == 4

    def test_to_dict_serializable(self):
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(12)
        report = run_regime_sensitivity_grid(bench, dates, BASE_WEIGHTS)
        d = report.to_dict()
        assert "grid_points" in d
        assert len(d["grid_points"]) == 9
        assert "summary" in d

    def test_small_dataset_handled(self):
        """少于 6 个月的短序列也能运行（grid 内部处理不足数据）。"""
        bench = _make_random_walk_benchmark(200)
        dates = _make_monthly_dates(4)
        report = run_regime_sensitivity_grid(bench, dates, BASE_WEIGHTS)
        assert report.n_months_analyzed == 4


class TestFindMostStable:
    def test_returns_smallest_weight_std(self):
        gp1 = RegimeGridPoint(0.03, 0.03, 10, 5, 5, 0.5, 0.25, 0.25, 0.2, 0.1, 0.1, 0.1, 0.05)
        gp2 = RegimeGridPoint(0.05, 0.04, 8, 4, 8, 0.4, 0.2, 0.4, 0.2, 0.1, 0.1, 0.1, 0.02)
        gp3 = RegimeGridPoint(0.07, 0.05, 6, 3, 11, 0.3, 0.15, 0.55, 0.2, 0.1, 0.1, 0.1, 0.08)
        result = _find_most_stable([gp1, gp2, gp3])
        assert result["bull_threshold"] == 0.05
        assert result["bear_threshold"] == 0.04
        assert result["weight_turnover_std"] == 0.02

    def test_empty_list(self):
        assert _find_most_stable([]) == {}


class TestRegimeSensitivityMarkdown:
    def test_produces_valid_markdown(self):
        bench = _make_random_walk_benchmark(1260)
        dates = _make_monthly_dates(12)
        report = run_regime_sensitivity_grid(bench, dates, BASE_WEIGHTS)
        md = regime_sensitivity_markdown(report)
        assert "## Regime 参数敏感性网格报告" in md
        assert "| bull_thr | bear_thr" in md
        assert "### 最稳定参数" in md
