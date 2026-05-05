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
from src.portfolio.covariance import (
    _ewma_covariance,
    _factor_model_covariance,
    _industry_factor_covariance,
    _returns_matrix_from_wide,
    estimate_covariance,
    mean_cov_returns_from_daily_long,
    mean_cov_returns_from_wide,
)
from src.portfolio.optimizer import (
    _finalize_solver_result,
    _risk_contributions,
    covariance_diagnostics,
    optimize_mean_variance,
    optimize_min_variance,
    optimize_risk_parity,
    solve_weights_from_cov_method,
    weight_diagnostics,
    weights_from_cov_method,
)
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


def test_build_portfolio_tiered_equal_weight():
    df = pd.DataFrame(
        {
            "symbol": ["300750", "600519", "000001", "600000"],
            "momentum": [1.0, 4.0, 2.0, 3.0],
        }
    )
    w = build_portfolio_weights(
        df,
        weight_method="tiered_equal_weight",
        score_col="momentum",
        top_tier_count=2,
        top_tier_weight_share=0.6,
        max_single_weight=1.0,
        max_industry_weight=None,
        industry_col=None,
        max_turnover=1.0,
    )
    assert w.sum() == pytest.approx(1.0)
    expected = np.array([0.2, 0.3, 0.2, 0.3], dtype=np.float64)
    assert np.allclose(w, expected)


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


def test_solve_weights_from_cov_method_reports_equal_like_risk_parity():
    cov = np.array(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.04, 0.01],
            [0.01, 0.01, 0.04],
        ],
        dtype=np.float64,
    )
    w, diag = solve_weights_from_cov_method("risk_parity", cov)
    assert w.sum() == pytest.approx(1.0)
    assert diag["solver_success"] is True
    assert diag["weights"]["is_close_to_reference"] is True
    assert diag["fallback_reason"] == "equal_like_solution"
    assert diag["covariance"]["diag_share"] > 0


def test_build_portfolio_weights_return_diagnostics_for_min_variance():
    df = pd.DataFrame(
        {
            "symbol": ["600519", "000001", "300750"],
            "momentum": [1.0, 2.0, 3.0],
        }
    )
    cov = np.array(
        [
            [0.01, 0.002, 0.001],
            [0.002, 0.09, 0.003],
            [0.001, 0.003, 0.16],
        ],
        dtype=np.float64,
    )
    w, diag = build_portfolio_weights(
        df,
        weight_method="min_variance",
        score_col="auto",
        max_single_weight=1.0,
        max_industry_weight=None,
        industry_col=None,
        max_turnover=1.0,
        cov_matrix=cov,
        return_diagnostics=True,
    )
    assert w.sum() == pytest.approx(1.0)
    assert diag["optimizer"]["solver_success"] is True
    assert diag["post_constraints"]["l1_diff_vs_reference"] > 0.05
    assert diag["post_constraint_l1_shift"] == pytest.approx(0.0)


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


def test_covariance_factor_method_is_positive_definite():
    rng = np.random.default_rng(7)
    factors = rng.normal(0.0, 0.01, size=(80, 3))
    beta = rng.normal(0.5, 0.2, size=(3, 5))
    noise = rng.normal(0.0, 0.003, size=(80, 5))
    returns = (factors @ beta + noise).T

    cov = estimate_covariance(returns, method="factor", factor_returns=factors)

    eigvals = np.linalg.eigvalsh(cov)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-10)
    assert float(np.min(eigvals)) > 0.0


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


# ── optimizer edge cases ────────────────────────────────────────────────────

def test_risk_contributions_zero_sum():
    w = np.array([0.0, 0.0])
    Sigma = np.eye(2)
    rc = _risk_contributions(w, Sigma)
    assert np.all(rc == 0.0)


def test_covariance_diagnostics_n0():
    diag = covariance_diagnostics(np.zeros((0, 0)))
    assert diag["n_assets"] == 0
    assert np.isnan(diag["condition_number"])


def test_weight_diagnostics_n0():
    diag = weight_diagnostics(np.array([]))
    assert diag["n_assets"] == 0
    assert diag["nonzero_count"] == 0


def test_weight_diagnostics_with_reference():
    w = np.array([0.5, 0.3, 0.2])
    ref = np.array([0.4, 0.35, 0.25])
    diag = weight_diagnostics(w, reference=ref)
    assert "l1_diff_vs_reference" in diag
    assert diag["l1_diff_vs_reference"] > 0


def test_finalize_solver_result_missing():
    w, info = _finalize_solver_result(None, 3, fallback_reason="test_none")
    assert np.allclose(w, 1.0 / 3.0)
    assert info["used_fallback"] is True
    assert info["fallback_reason"] == "test_none"


def test_finalize_solver_result_invalid_solution():
    class BadResult:
        x = np.array([np.nan, np.nan, np.nan])
        success = False
        status = 1
        message = "bad"
        nit = 0
    w, info = _finalize_solver_result(BadResult(), 3)
    assert np.allclose(w, 1.0 / 3.0)
    assert info["used_fallback"] is True


def test_finalize_solver_result_zero_sum():
    class ZeroResult:
        x = np.array([0.0, 0.0, 0.0])
        success = False
        status = 2
        message = "zero"
        nit = 5
        fun = 0.5
    w, info = _finalize_solver_result(ZeroResult(), 3)
    assert np.allclose(w, 1.0 / 3.0)
    assert info["used_fallback"] is True


def test_optimize_mean_variance_basic():
    Sigma = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.16]])
    mu = np.array([0.01, 0.02, 0.03])
    w = optimize_mean_variance(Sigma, mu, risk_aversion=0.5)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= -1e-8)
    # highest return asset should have meaningful weight
    assert w[2] > 0.3


def test_optimize_mean_variance_n1():
    w = optimize_mean_variance(np.eye(1), np.array([0.05]))
    assert w[0] == pytest.approx(1.0)


def test_weights_from_cov_method_mean_variance():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    mu = np.array([0.02, 0.04])
    w = weights_from_cov_method("mean_variance", Sigma, mu=mu, risk_aversion=2.0)
    assert w.sum() == pytest.approx(1.0)


def test_weights_from_cov_method_invalid():
    with pytest.raises(ValueError):
        weights_from_cov_method("nonexistent", np.eye(2))


def test_solve_weights_from_cov_method_min_variance():
    cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.16]])
    w, diag = solve_weights_from_cov_method("min_variance", cov)
    assert w.sum() == pytest.approx(1.0)
    assert diag["solver_success"] is True
    assert diag["weights"]["nonzero_count"] >= 1


def test_solve_weights_from_cov_method_mean_variance():
    cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.16]])
    mu = np.array([0.02, 0.04, 0.06])
    w, diag = solve_weights_from_cov_method("mean_variance", cov, mu=mu, risk_aversion=1.5)
    assert w.sum() == pytest.approx(1.0)
    assert diag["expected_return_mean"] > 0


def test_solve_weights_from_cov_method_n0():
    w, diag = solve_weights_from_cov_method("risk_parity", np.zeros((0, 0)))
    assert len(w) == 0


def test_solve_weights_from_cov_method_n1():
    w, diag = solve_weights_from_cov_method("min_variance", np.eye(1))
    assert w[0] == pytest.approx(1.0)
    assert diag["solver_message"] == "single_asset"


def test_solve_weights_from_cov_method_invalid():
    with pytest.raises(ValueError):
        solve_weights_from_cov_method("bad_method", np.eye(3))


def test_solve_weights_from_cov_method_with_turnover():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    prev = np.array([0.7, 0.3])
    w, diag = solve_weights_from_cov_method(
        "risk_parity", cov, prev_weights=prev, max_turnover=0.3,
    )
    assert w.sum() == pytest.approx(1.0)
    # turnover should be constrained
    turnover = np.sum(np.abs(w - prev))
    assert turnover <= 0.3 + 1e-6 or diag.get("used_fallback", False)


# ── covariance edge cases ──────────────────────────────────────────────────

def test_estimate_covariance_n0():
    cov = estimate_covariance(np.zeros((0, 0)), method="auto")
    assert cov.shape == (0, 0)


def test_estimate_covariance_t1():
    R = np.array([[0.01], [0.02]])
    cov = estimate_covariance(R, method="auto")
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0)


def test_estimate_covariance_ledoit_wolf():
    rng = np.random.default_rng(9)
    R = rng.normal(0.0, 0.02, size=(5, 100))
    cov = estimate_covariance(R, method="ledoit_wolf")
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T)


def test_estimate_covariance_auto_high_condition():
    """n > t 时样本协方差奇异，auto 应触发 Ledoit-Wolf 或至少返回有效矩阵."""
    rng = np.random.default_rng(11)
    R = rng.normal(0.0, 0.02, size=(8, 5))  # n=8, t=5: singular
    cov, meta = estimate_covariance(R, method="auto", return_meta=True)
    assert cov.shape == (8, 8)
    assert np.all(np.diag(cov) > 0)


def test_factor_model_covariance_with_factors():
    rng = np.random.default_rng(13)
    factors = rng.normal(0.0, 0.01, size=(80, 2))
    beta = rng.normal(0.3, 0.15, size=(2, 6))
    noise = rng.normal(0.0, 0.002, size=(80, 6))
    returns = (factors @ beta + noise).T
    cov = _factor_model_covariance(returns, factor_returns=factors, ridge=1e-6)
    assert cov.shape == (6, 6)
    eigvals = np.linalg.eigvalsh(cov)
    assert float(np.min(eigvals)) > 0


def test_factor_model_covariance_n0():
    cov = _factor_model_covariance(np.zeros((0, 0)), factor_returns=None, ridge=1e-6)
    assert cov.shape == (0, 0)


def test_factor_model_covariance_t1():
    R = np.array([[0.01], [0.02], [0.03]])
    cov = _factor_model_covariance(R, factor_returns=None, ridge=1e-6)
    assert cov.shape == (3, 3)


def test_ewma_covariance_t1():
    R = np.array([[0.01], [0.02]])
    cov = _ewma_covariance(R, halflife=20, ridge=1e-6)
    assert cov.shape == (2, 2)


def test_industry_factor_covariance_single_industry():
    rng = np.random.default_rng(15)
    R = rng.normal(0.0, 0.01, size=(4, 60))
    cov = _industry_factor_covariance(R, ["A", "A", "A", "A"], ridge=1e-6)
    assert cov.shape == (4, 4)


def test_industry_factor_covariance_t1():
    R = np.array([[0.01], [0.02], [0.03]])
    cov = _industry_factor_covariance(R, ["A", "B", "C"], ridge=1e-6)
    assert cov.shape == (3, 3)


def test_industry_factor_covariance_n0():
    cov = _industry_factor_covariance(np.zeros((0, 0)), [], ridge=1e-6)
    assert cov.shape == (0, 0)


def test_returns_matrix_from_wide_basic():
    dates = pd.date_range("2024-01-01", periods=11, freq="B")
    prices = np.array([
        [10.0, 10.1, 10.2, 10.15, 10.3, 10.4, 10.35, 10.5, 10.6, 10.55, 10.7],
        [20.0, 20.2, 20.1, 20.3, 20.5, 20.4, 20.6, 20.8, 20.7, 20.9, 21.0],
    ])
    wide = pd.DataFrame(prices, index=["a", "b"], columns=dates)
    R = _returns_matrix_from_wide(wide, ["a", "b"], lookback_days=10)
    assert R.shape[0] == 2
    assert R.shape[1] >= 1


def test_returns_matrix_from_wide_empty():
    R = _returns_matrix_from_wide(pd.DataFrame(), [], lookback_days=10)
    assert R.shape == (0, 0)


def test_mean_cov_returns_from_daily_long_basic():
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    rows = []
    for dt in dates:
        for sym in ["000001", "000002", "000003"]:
            rows.append({"trade_date": dt, "symbol": sym, "close": 10.0 + np.random.randn() * 0.1})
    daily = pd.DataFrame(rows)
    mu, cov = mean_cov_returns_from_daily_long(
        daily, ["000001", "000002", "000003"],
        asof=pd.Timestamp("2025-03-15"), lookback_days=30,
    )
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)


def test_mean_cov_returns_from_daily_long_empty():
    mu, cov = mean_cov_returns_from_daily_long(
        pd.DataFrame(), [], asof=pd.Timestamp("2025-01-01"), lookback_days=30,
    )
    assert mu.shape == (0,)
    assert cov.shape == (0, 0)


def test_mean_cov_returns_from_daily_long_with_industry():
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    rows = []
    for dt in dates:
        for sym, ind in [("000001", "bank"), ("000002", "tech"), ("000003", "bank")]:
            rows.append({"trade_date": dt, "symbol": sym, "close": 10.0 + np.random.randn() * 0.1, "industry": ind})
    daily = pd.DataFrame(rows)
    mu, cov = mean_cov_returns_from_daily_long(
        daily, ["000001", "000002", "000003"],
        asof=pd.Timestamp("2025-03-01"), lookback_days=20,
        shrinkage="industry_factor", industry_col="industry",
    )
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)


def test_optimize_risk_parity_with_turnover():
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    prev = np.array([0.8, 0.2])
    w = optimize_risk_parity(cov, prev_weights=prev, max_turnover=0.3)
    assert w.sum() == pytest.approx(1.0)


def test_optimize_min_variance_with_turnover():
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    prev = np.array([0.8, 0.2])
    w = optimize_min_variance(cov, prev_weights=prev, max_turnover=0.3)
    assert w.sum() == pytest.approx(1.0)


def test_optimize_mean_variance_with_turnover():
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    mu = np.array([0.02, 0.05])
    prev = np.array([0.8, 0.2])
    w = optimize_mean_variance(cov, mu, prev_weights=prev, max_turnover=0.3)
    assert w.sum() == pytest.approx(1.0)
