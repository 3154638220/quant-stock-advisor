"""Tests for backtest engine: limit-up handling, tiered impact, gross exposure, helpers."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    _align_weights_columns,
    _amount_tier_label,
    _apply_gross_exposure,
    _apply_limit_up_buy_fail,
    _redistribute_limit_up_weights,
    BacktestConfig,
    TieredImpactConfig,
    build_daily_weights,
    run_backtest,
)


# ── _amount_tier_label ────────────────────────────────────────────────────────


def test_amount_tier_label_three_tiers():
    tc = TieredImpactConfig()
    assert _amount_tier_label(1_000_000_000, tc) == "large"
    assert _amount_tier_label(500_000_000, tc) == "large"
    assert _amount_tier_label(200_000_000, tc) == "mid"
    assert _amount_tier_label(100_000_000, tc) == "mid"
    assert _amount_tier_label(50_000_000, tc) == "small"
    assert _amount_tier_label(0, tc) == "small"
    assert _amount_tier_label(float("nan"), tc) == "small"


def test_amount_tier_label_custom_thresholds():
    tc = TieredImpactConfig(large_cap_threshold=1_000_000_000, mid_cap_threshold=200_000_000)
    assert _amount_tier_label(1_000_000_000, tc) == "large"
    assert _amount_tier_label(500_000_000, tc) == "mid"
    assert _amount_tier_label(100_000_000, tc) == "small"


def test_amount_tier_label_none_cfg():
    assert _amount_tier_label(100_000_000, None) == "mid"


# ── TieredImpactConfig.get_params ─────────────────────────────────────────────


def test_tiered_impact_get_params_defaults():
    tc = TieredImpactConfig()
    assert tc.get_params(1_000_000_000) == (1.5, 4.0)
    assert tc.get_params(200_000_000) == (3.0, 8.0)
    assert tc.get_params(50_000_000) == (5.0, 20.0)
    assert tc.get_params(0) == (5.0, 20.0)
    assert tc.get_params(float("nan")) == (5.0, 20.0)


# ── _align_weights_columns ────────────────────────────────────────────────────


def test_align_weights_columns():
    w = pd.DataFrame({"A": [0.5], "C": [0.5]})
    out = _align_weights_columns(w, ["A", "B", "C"])
    assert out.columns.tolist() == ["A", "B", "C"]
    assert out.iloc[0, 1] == 0.0


# ── _apply_gross_exposure ─────────────────────────────────────────────────────


def test_gross_exposure_no_cap():
    w = np.array([0.6, 0.5])
    out = _apply_gross_exposure(w, 0.0)
    np.testing.assert_array_equal(out, w)


def test_gross_exposure_scale_down():
    w = np.array([0.6, 0.5])
    out = _apply_gross_exposure(w, 0.5)
    assert out.sum() <= 0.5 + 1e-12


def test_gross_exposure_all_zero():
    out = _apply_gross_exposure(np.zeros(5), 1.0)
    np.testing.assert_array_equal(out, np.zeros(5))


# ── _redistribute_limit_up_weights ────────────────────────────────────────────


def test_redistribute_no_limit_up():
    w = np.array([0.3, 0.3, 0.4])
    out = _redistribute_limit_up_weights(w, np.array([False, False, False]))
    np.testing.assert_array_almost_equal(out, w)


def test_redistribute_single_limit_up():
    w = np.array([0.3, 0.3, 0.4])  # first stock limit-up
    mask = np.array([True, False, False])
    out = _redistribute_limit_up_weights(w, mask)
    assert out[0] == 0.0
    np.testing.assert_almost_equal(out.sum(), 1.0)  # weights re-normalized
    # Remaining stocks got proportional share of stranded 0.3


def test_redistribute_proportional():
    """Redistribution should preserve relative weights of non-limit-up stocks."""
    w = np.array([0.2, 0.3, 0.5])
    mask = np.array([True, False, False])  # 0.2 redistributed
    out = _redistribute_limit_up_weights(w, mask)
    assert out[0] == 0.0
    assert out.sum() > 0.99  # weights re-normalized


def test_redistribute_all_limit_up():
    w = np.array([0.3, 0.3, 0.4])
    out = _redistribute_limit_up_weights(w, np.array([True, True, True]))
    assert out.sum() == 0.0  # all funds idle


def test_redistribute_multi_limit_up():
    """Two stocks hit limit-up, remaining one gets all weight."""
    w = np.array([0.2, 0.2, 0.6])
    mask = np.array([True, True, False])
    out = _redistribute_limit_up_weights(w, mask)
    assert out[0] == 0.0
    assert out[1] == 0.0
    np.testing.assert_almost_equal(out[2], 1.0)


# ── _apply_limit_up_buy_fail ──────────────────────────────────────────────────


def test_apply_limit_up_idle_mode():
    target = np.array([0.3, 0.3, 0.4])
    prior = np.array([0.25, 0.35, 0.4])
    mask = np.array([True, False, False])
    effective, failed, ftotal, redist, idle = _apply_limit_up_buy_fail(target, prior, mask, mode="idle")
    # Only the delta (0.05) on first stock fails, not the prior 0.25
    assert failed[0] > 0
    assert effective[0] < target[0]
    assert redist == 0.0
    assert idle > 0


def test_apply_limit_up_redistribute_mode():
    target = np.array([0.3, 0.3, 0.4])
    prior = np.array([0.25, 0.35, 0.4])
    mask = np.array([True, False, False])
    effective, failed, ftotal, redist, idle = _apply_limit_up_buy_fail(target, prior, mask, mode="redistribute")
    assert ftotal > 0
    assert redist == ftotal  # all redistributed
    assert idle == 0.0


def test_apply_limit_up_no_new_buy():
    """If prior weight >= target weight (selling), no buy failure."""
    target = np.array([0.1, 0.4, 0.5])
    prior = np.array([0.3, 0.3, 0.4])  # first stock: target 0.1 < prior 0.3, no buy
    mask = np.array([True, False, False])
    _, failed, ftotal, _, _ = _apply_limit_up_buy_fail(target, prior, mask, mode="idle")
    assert failed[0] == 0.0  # no new buy to fail
    assert ftotal == 0.0


def test_apply_limit_up_prior_zero():
    """New stock entering portfolio (prior=0) hitting limit-up."""
    target = np.array([0.2, 0.4, 0.4])
    prior = np.array([0.0, 0.5, 0.5])
    mask = np.array([True, False, False])
    effective, failed, ftotal, _, _ = _apply_limit_up_buy_fail(target, prior, mask, mode="idle")
    assert failed[0] == 0.2  # entire target of new stock fails
    assert effective[0] == 0.0


# ── build_daily_weights ───────────────────────────────────────────────────────


def test_build_daily_weights_forward_fill():
    dates = pd.date_range("2021-01-04", periods=5, freq="B")
    weights = pd.DataFrame(
        {"A": [0.5, 0.6], "B": [0.5, 0.4]},
        index=pd.to_datetime(["2021-01-04", "2021-01-06"]),
    )
    dw = build_daily_weights(dates, weights, max_gross_exposure=1.0)
    assert dw.shape == (5, 2)
    np.testing.assert_array_almost_equal(dw.sum(axis=1), np.ones(5))


# ── run_backtest: VWAP + tiered impact ────────────────────────────────────────


def test_run_backtest_vwap_tiered_impact():
    """End-to-end: VWAP mode with tiered impact produces impact tier breakdown."""
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    syms = ["000001", "000002", "000003"]
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(60, 3) * 0.02, index=dates, columns=syms)
    signal = pd.DataFrame({"000001": [0.3], "000002": [0.3], "000003": [0.4]},
                          index=pd.to_datetime(["2021-01-04"]))
    # Provide daily_amount so tiered impact activates
    amount = pd.DataFrame(np.full((60, 3), 300_000_000.0), index=dates, columns=syms)

    tc = TieredImpactConfig(
        large_cap_threshold=1.0,
        mid_cap_threshold=0.5,
    )
    cfg = BacktestConfig(execution_mode="vwap", use_tiered_impact=True, tiered_impact=tc)
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="", daily_amount=amount)
    assert "impact_cost_by_tier" in result.meta
    assert result.meta["impact_tier_used"] is True
    assert result.meta["impact_cost_total"] >= 0


def test_run_backtest_vwap_flat_fallback():
    """Without daily_amount, tiered impact falls back to flat VWAP params."""
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    syms = ["000001", "000002"]
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(60, 2) * 0.02, index=dates, columns=syms)
    signal = pd.DataFrame({"000001": [0.5], "000002": [0.5]},
                          index=pd.to_datetime(["2021-01-04"]))
    cfg = BacktestConfig(execution_mode="vwap")
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="")
    assert result.meta["impact_tier_used"] is False


def test_run_backtest_vwap_with_daily_amount():
    """VWAP with actual daily_amount uses tiered params per stock."""
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    syms = ["000001", "000002"]
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(60, 2) * 0.02, index=dates, columns=syms)
    amount = pd.DataFrame(index=dates, columns=syms)
    amount["000001"] = 600_000_000.0
    amount["000002"] = 50_000_000.0
    signal = pd.DataFrame({"000001": [0.5], "000002": [0.5]},
                          index=pd.to_datetime(["2021-01-04"]))
    tc = TieredImpactConfig()
    cfg = BacktestConfig(execution_mode="vwap", use_tiered_impact=True, tiered_impact=tc)
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="", daily_amount=amount)
    assert result.meta["impact_tier_used"] is True
    assert result.meta["impact_cost_by_tier"]["small"] >= 0


# ── run_backtest: limit-up idle vs redistribute ───────────────────────────────


def _make_limit_up_backtest_data(limit_up_symbols=None):
    """Helper: create returns + signal + limit-up mask for 3-stock test.

    Uses a single rebalance on the first trading day, mask on t+1 entry day.
    Since prior weights are zero (no previous position), every stock has a buy delta,
    so limit-up stocks will immediately trigger buy failure.
    """
    if limit_up_symbols is None:
        limit_up_symbols = []
    dates = pd.date_range("2021-01-04", periods=20, freq="B")
    syms = ["000001", "000002", "000003"]
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(20, 3) * 0.01 + 0.001, index=dates, columns=syms)
    # Single rebalance on the first day
    signal = pd.DataFrame(
        {"000001": [0.3], "000002": [0.3], "000003": [0.4]},
        index=pd.to_datetime(["2021-01-04"]),
    )
    # Limit-up mask: True on the ENTRY day (t+1 after rebalance) for specified symbols.
    # The engine applies mask at index i (entry day with open-to-open), not jw (signal day).
    mask = pd.DataFrame(False, index=dates, columns=syms)
    for sym in limit_up_symbols:
        if sym in syms:
            mask.loc[pd.Timestamp("2021-01-05"), sym] = True  # t+1 after Jan 4 rebalance
    return returns, signal, mask


def test_run_backtest_limit_up_idle():
    """Limit-up idle mode: some weight frozen, buy_fail_diagnostic populated."""
    returns, signal, mask = _make_limit_up_backtest_data(["000001"])
    cfg = BacktestConfig(
        execution_mode="tplus1_open",
        limit_up_mode="idle",
        limit_up_open_mask=mask,
    )
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="")
    # Check diagnostic
    assert result.meta["limit_up_mode"] == "idle"
    assert result.meta["buy_fail_event_count"] > 0
    assert result.meta["buy_fail_idle_weight"] > 0
    assert result.meta["buy_fail_redistributed_weight"] == 0.0


def test_run_backtest_limit_up_redistribute():
    """Limit-up redistribute mode: failed weight redistributed."""
    returns, signal, mask = _make_limit_up_backtest_data(["000001"])
    cfg = BacktestConfig(
        execution_mode="tplus1_open",
        limit_up_mode="redistribute",
        limit_up_open_mask=mask,
    )
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="")
    assert result.meta["limit_up_mode"] == "redistribute"
    assert result.meta["buy_fail_redistributed_weight"] > 0


def test_run_backtest_limit_up_multi_stock():
    """Multiple stocks limit-up simultaneously."""
    returns, signal, mask = _make_limit_up_backtest_data(["000001", "000002"])
    cfg = BacktestConfig(
        execution_mode="tplus1_open",
        limit_up_mode="redistribute",
        limit_up_open_mask=mask,
    )
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="")
    events = result.meta["buy_fail_event_count"]
    assert events >= 2  # 2 stocks fail buy on entry day
    # With 2 of 3 stocks hitting limit-up, redistribution still works
    assert result.meta["buy_fail_redistributed_weight"] > 0


def test_run_backtest_no_limit_up():
    """No limit-up events -> no diagnostic rows."""
    returns, signal, mask = _make_limit_up_backtest_data([])
    cfg = BacktestConfig(
        execution_mode="tplus1_open",
        limit_up_mode="idle",
        limit_up_open_mask=mask,
    )
    result = run_backtest(returns, signal, config=cfg, rebalance_rule="")
    assert result.meta["buy_fail_event_count"] == 0


# ── run_backtest: execution modes ──────────────────────────────────────────────


def test_run_backtest_close_to_close():
    dates = pd.date_range("2021-01-04", periods=60, freq="B")
    syms = ["A", "B"]
    returns = pd.DataFrame(np.random.randn(60, 2) * 0.02, index=dates, columns=syms)
    signal = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=pd.to_datetime(["2021-01-04"]))
    result = run_backtest(returns, signal, config=BacktestConfig(execution_mode="close_to_close"), rebalance_rule="")
    assert len(result.daily_returns) == 60
    assert result.meta["execution_mode"] == "close_to_close"


def test_run_backtest_invalid_execution_mode():
    dates = pd.date_range("2021-01-04", periods=10, freq="B")
    returns = pd.DataFrame(np.random.randn(10, 1) * 0.02, index=dates, columns=["A"])
    signal = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2021-01-04"]))
    with pytest.raises(ValueError):
        run_backtest(returns, signal, config=BacktestConfig(execution_mode="invalid"))


# ── Edge cases ─────────────────────────────────────────────────────────────────


def test_empty_weights_signal():
    dates = pd.date_range("2021-01-04", periods=10, freq="B")
    returns = pd.DataFrame(np.random.randn(10, 1) * 0.02, index=dates, columns=["A"])
    empty_signal = pd.DataFrame(columns=["A"], dtype=float)
    with pytest.raises(ValueError):
        run_backtest(returns, empty_signal)


def test_mismatched_columns():
    dates = pd.date_range("2021-01-04", periods=10, freq="B")
    returns = pd.DataFrame(np.random.randn(10, 2) * 0.02, index=dates, columns=["A", "B"])
    signal = pd.DataFrame({"C": [0.5], "D": [0.5]}, index=pd.to_datetime(["2021-01-04"]))
    # Should not crash; missing columns become zeros
    result = run_backtest(returns, signal, config=BacktestConfig(), rebalance_rule="")
    assert result.meta["symbols"] == ["A", "B"]
