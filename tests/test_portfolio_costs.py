"""组合权重、交易成本与风险指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.risk_metrics import max_drawdown_from_returns, realized_volatility
from src.backtest.transaction_costs import (
    TransactionCostParams,
    net_simple_return_from_long_hold,
)
from src.market.regime import RegimeConfig, RegimeResult, get_regime_weights
from src.portfolio.covariance import mean_cov_returns_from_wide
from src.portfolio.optimizer import optimize_min_variance, optimize_risk_parity
from src.portfolio.weights import (
    apply_turnover_constraint,
    build_portfolio_weights,
    redistribute_individual_cap,
    turnover_cost_coeffs_from_size,
)


def test_redistribute_individual_cap():
    w = np.array([0.5, 0.3, 0.2])
    # 须 cap >= 1/n，否则无法同时满足和为 1 与单票上限
    out = redistribute_individual_cap(w, 0.34)
    assert out.sum() == pytest.approx(1.0)
    assert np.max(out) <= 0.34 + 1e-6


def test_build_portfolio_equal():
    df = pd.DataFrame(
        {
            "symbol": ["600519", "000001", "300750"],
            "momentum": [1.0, 2.0, 3.0],
            "rank": [1, 2, 3],
        }
    )
    w = build_portfolio_weights(
        df,
        weight_method="equal",
        score_col="auto",
        max_single_weight=0.5,
        max_industry_weight=None,
        industry_col=None,
        max_turnover=1.0,
    )
    assert len(w) == 3
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 1.0 / 3.0)


def test_build_portfolio_industry_cap():
    df = pd.DataFrame(
        {
            "symbol": ["a", "b", "c", "d"],
            "composite_score": [4.0, 3.0, 2.0, 1.0],
            "industry": ["X", "X", "Y", "Y"],
        }
    )
    w = build_portfolio_weights(
        df,
        weight_method="score",
        score_col="composite_score",
        max_single_weight=0.35,
        max_industry_weight=0.48,
        industry_col="industry",
        max_turnover=1.0,
    )
    assert w.sum() == pytest.approx(1.0)
    assert w[np.array([0, 1])].sum() <= 0.48 + 0.03
    assert np.max(w) <= 0.35 + 0.03


def test_net_simple_return_from_long_hold():
    c = TransactionCostParams(
        commission_buy_bps=0.0,
        commission_sell_bps=0.0,
        slippage_bps_per_side=0.0,
        stamp_duty_sell_bps=0.0,
    )
    g = 0.05
    assert net_simple_return_from_long_hold(g, c) == pytest.approx(0.05)
    c2 = TransactionCostParams(
        commission_buy_bps=250.0,
        commission_sell_bps=250.0,
        slippage_bps_per_side=0.0,
        stamp_duty_sell_bps=0.0,
    )
    assert net_simple_return_from_long_hold(0.0, c2) < 0


def test_covariance_and_risk_parity_small():
    # 3 标的 × 41 日收盘，40 条日收益
    rng = np.random.default_rng(0)
    days = 41
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    syms = ["600000", "600001", "600002"]
    prices = 10 + np.cumsum(rng.standard_normal((3, days)), axis=1) * 0.1
    wide = pd.DataFrame(prices, index=syms, columns=dates)
    mu, cov = mean_cov_returns_from_wide(wide, syms, lookback_days=40)
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)
    w_rp = optimize_risk_parity(cov)
    assert w_rp.sum() == pytest.approx(1.0)
    assert np.all(w_rp >= -1e-8)
    w_mv = optimize_min_variance(cov)
    assert w_mv.sum() == pytest.approx(1.0)


def test_build_portfolio_risk_parity():
    df = pd.DataFrame(
        {
            "symbol": ["600519", "000001", "300750"],
            "momentum": [1.0, 2.0, 3.0],
        }
    )
    cov = np.array(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.04, 0.01],
            [0.01, 0.01, 0.04],
        ],
        dtype=np.float64,
    )
    w = build_portfolio_weights(
        df,
        weight_method="risk_parity",
        score_col="auto",
        max_single_weight=1.0,
        max_industry_weight=None,
        industry_col=None,
        max_turnover=1.0,
        cov_matrix=cov,
    )
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)


def test_covariance_ewma_and_industry_factor():
    rng = np.random.default_rng(1)
    days = 81
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    syms = ["600000", "600001", "600002", "600003"]
    prices = 10 + np.cumsum(rng.standard_normal((4, days)), axis=1) * 0.08
    wide = pd.DataFrame(prices, index=syms, columns=dates)

    _, cov_ewma = mean_cov_returns_from_wide(
        wide,
        syms,
        lookback_days=60,
        shrinkage="ewma",
        ewma_halflife=15,
    )
    assert cov_ewma.shape == (4, 4)
    assert np.all(np.diag(cov_ewma) > 0)

    _, cov_ind = mean_cov_returns_from_wide(
        wide,
        syms,
        lookback_days=60,
        shrinkage="industry_factor",
        industry_labels=["bank", "bank", "new_energy", "new_energy"],
    )
    assert cov_ind.shape == (4, 4)
    assert np.all(np.diag(cov_ind) > 0)
    assert np.allclose(cov_ind, cov_ind.T, atol=1e-10)


def test_turnover_cost_weighted_constraint():
    w_old = np.array([0.34, 0.33, 0.33], dtype=np.float64)
    w_new = np.array([0.90, 0.05, 0.05], dtype=np.float64)
    coeffs = np.array([1.8, 0.9, 0.7], dtype=np.float64)
    out = apply_turnover_constraint(w_new, w_old, 0.20, turnover_cost_coeffs=coeffs)
    tv_cost = 0.5 * float(np.sum(coeffs * np.abs(out - w_old)))
    assert tv_cost <= 0.20 + 1e-8
    assert np.isclose(out.sum(), 1.0)

    size_vals = np.array([8.0, 9.0, 15.0], dtype=np.float64)
    tc = turnover_cost_coeffs_from_size(size_vals)
    assert tc[0] >= tc[1] >= tc[2]


def test_dynamic_regime_weights():
    base = {"momentum": 0.6, "short_reversal": -0.4}
    cfg = RegimeConfig(
        dynamic_weighting_enabled=True,
        dynamic_vol_target_ann=0.2,
        dynamic_vol_scale=0.05,
        dynamic_trend_scale=0.04,
        dynamic_strength=4.0,
    )
    res_risk_on = RegimeResult(
        regime="bull",
        short_return=0.08,
        long_return=0.12,
        realized_vol_ann=0.15,
        n_days_used=60,
    )
    w_on = get_regime_weights(base, "bull", cfg=cfg, regime_result=res_risk_on)

    res_risk_off = RegimeResult(
        regime="bear",
        short_return=-0.06,
        long_return=-0.09,
        realized_vol_ann=0.34,
        n_days_used=60,
    )
    w_off = get_regime_weights(base, "bear", cfg=cfg, regime_result=res_risk_off)
    assert abs(w_on["momentum"]) > abs(w_off["momentum"])
    assert abs(w_on["short_reversal"]) < abs(w_off["short_reversal"])


def test_max_drawdown_and_vol():
    r = np.array([0.1, -0.05, -0.02])
    assert max_drawdown_from_returns(r) >= 0
    assert np.isfinite(realized_volatility(r, periods_per_year=252.0))
