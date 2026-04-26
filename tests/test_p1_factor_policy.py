from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_backtest_eval import apply_p1_factor_policy, compute_factors


def test_compute_factors_includes_p1_columns():
    idx = pd.bdate_range("2023-01-02", periods=320)
    daily = pd.DataFrame(
        {
            "symbol": ["000001"] * len(idx),
            "trade_date": idx,
            "open": np.linspace(10.0, 16.0, len(idx)),
            "close": np.linspace(10.1, 16.2, len(idx)),
            "high": np.linspace(10.3, 16.4, len(idx)),
            "low": np.linspace(9.9, 16.0, len(idx)),
            "volume": np.linspace(1_000_000, 1_500_000, len(idx)),
            "amount": np.linspace(10_000_000, 15_000_000, len(idx)),
            "turnover": np.linspace(0.5, 2.0, len(idx)),
            "pct_chg": np.full(len(idx), 0.2),
        }
    )
    fac = compute_factors(daily, min_hist_days=260)
    assert "momentum_12_1" in fac.columns
    assert "turnover_rel_pct_252" in fac.columns
    assert "weekly_kdj_j" in fac.columns
    assert "weekly_kdj_oversold_depth" in fac.columns
    assert fac["momentum_12_1"].notna().sum() > 0
    assert fac["turnover_rel_pct_252"].notna().sum() > 0
    assert fac["weekly_kdj_j"].notna().sum() > 0


def test_apply_p1_factor_policy_remove_zero_flip():
    base = {"momentum": -0.2, "rsi": -0.1, "pb": -0.1, "momentum_12_1": 0.2}
    summary = pd.DataFrame(
        [
            {"factor": "momentum", "horizon_key": "tplus1_open_1d", "ic_mean": -0.01},
            {"factor": "momentum", "horizon_key": "close_21d", "ic_mean": 0.03},
            {"factor": "rsi", "horizon_key": "tplus1_open_1d", "ic_mean": 0.001},
            {"factor": "rsi", "horizon_key": "close_21d", "ic_mean": 0.01},
            {"factor": "pb", "horizon_key": "tplus1_open_1d", "ic_mean": -0.02},
            {"factor": "pb", "horizon_key": "close_21d", "ic_mean": -0.01},
            {"factor": "momentum_12_1", "horizon_key": "tplus1_open_1d", "ic_mean": 0.02},
            {"factor": "momentum_12_1", "horizon_key": "close_21d", "ic_mean": 0.04},
        ]
    )
    out, actions = apply_p1_factor_policy(
        base,
        summary,
        remove_if_t1_and_t21_negative=True,
        zero_if_abs_t1_below=0.005,
        flip_if_t1_negative_and_t21_above=0.02,
    )
    act = {r["factor"]: r["action"] for r in actions.to_dict(orient="records")}
    assert act["momentum"] == "flip"
    assert act["rsi"] == "zero"
    assert act["pb"] == "remove"
    assert act["momentum_12_1"] == "keep"
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9


def test_apply_p1_factor_policy_relaxed_weak_tplus1_zero():
    """P1v2：zero_if_abs_t1_below=0 时弱 |T+1| 因子保留；flip 门槛 0.005。"""
    base = {"momentum": -0.2, "rsi": -0.1, "pb": -0.1, "momentum_12_1": 0.2}
    summary = pd.DataFrame(
        [
            {"factor": "momentum", "horizon_key": "tplus1_open_1d", "ic_mean": -0.01},
            {"factor": "momentum", "horizon_key": "close_21d", "ic_mean": 0.03},
            {"factor": "rsi", "horizon_key": "tplus1_open_1d", "ic_mean": 0.001},
            {"factor": "rsi", "horizon_key": "close_21d", "ic_mean": 0.01},
            {"factor": "pb", "horizon_key": "tplus1_open_1d", "ic_mean": -0.02},
            {"factor": "pb", "horizon_key": "close_21d", "ic_mean": -0.01},
            {"factor": "momentum_12_1", "horizon_key": "tplus1_open_1d", "ic_mean": 0.02},
            {"factor": "momentum_12_1", "horizon_key": "close_21d", "ic_mean": 0.04},
        ]
    )
    out, actions = apply_p1_factor_policy(
        base,
        summary,
        remove_if_t1_and_t21_negative=True,
        zero_if_abs_t1_below=0.0,
        flip_if_t1_negative_and_t21_above=0.005,
    )
    act = {r["factor"]: r["action"] for r in actions.to_dict(orient="records")}
    assert act["momentum"] == "flip"
    assert act["rsi"] == "keep"
    assert act["pb"] == "remove"
    assert act["momentum_12_1"] == "keep"
    assert abs(sum(abs(v) for v in out.values()) - 1.0) < 1e-9


def test_apply_p1_factor_policy_removes_missing_ic_factors():
    base = {"ev_ebitda": -0.2, "ocf_to_net_profit": 0.1}
    summary = pd.DataFrame(
        [
            {"factor": "ocf_to_net_profit", "horizon_key": "tplus1_open_1d", "ic_mean": 0.01},
            {"factor": "ocf_to_net_profit", "horizon_key": "close_21d", "ic_mean": 0.02},
        ]
    )
    out, actions = apply_p1_factor_policy(base, summary)
    act = {r["factor"]: r["action"] for r in actions.to_dict(orient="records")}
    assert act["ev_ebitda"] == "remove"
    assert act["ocf_to_net_profit"] == "keep"
    assert set(out) == {"ocf_to_net_profit"}
    assert abs(out["ocf_to_net_profit"] - 1.0) < 1e-9
