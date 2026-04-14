"""标准回测接口、绩效面板与 walk-forward。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, build_open_to_open_returns, run_backtest
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.transaction_costs import TransactionCostParams
from src.backtest.walk_forward import (
    contiguous_time_splits,
    rolling_walk_forward_windows,
    walk_forward_backtest,
)


def _dummy_market(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=n)
    sym = ["AAA", "BBB", "CCC"]
    r = rng.normal(0.0005, 0.02, size=(n, len(sym)))
    ar = pd.DataFrame(r, index=idx, columns=sym)
    # 每 5 日调仓：等权略扰动
    rb = idx[::5]
    rows = []
    for t in rb:
        w = np.array([0.34, 0.33, 0.33], dtype=np.float64) + rng.normal(0, 0.02, size=3)
        w = np.maximum(w, 0)
        w /= w.sum()
        rows.append(w)
    ws = pd.DataFrame(rows, index=rb, columns=sym)
    return ar, ws


def test_compute_performance_panel_basic():
    r = np.array([0.01, -0.005, 0.02, 0.001])
    p = compute_performance_panel(r, periods_per_year=252.0)
    assert p.n_periods == 4
    assert np.isfinite(p.annualized_return)
    assert np.isfinite(p.sharpe_ratio)
    assert 0 <= p.max_drawdown <= 1
    assert 0 <= p.win_rate <= 1


def test_build_open_to_open_returns_smoke():
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "open": [10.0, 11.0, 20.0, 21.0],
            "close": [10.0, 11.0, 20.0, 21.0],
        }
    )
    wide = build_open_to_open_returns(df, zero_if_limit_up_open=False)
    assert "000001" in wide.columns
    d0 = wide.index[0]
    assert abs(wide.loc[d0, "000001"] - (11.0 / 10.0 - 1.0)) < 1e-9


def test_run_backtest_tplus1_open_execution_mode():
    ar, ws = _dummy_market(80)
    # 用同一矩阵充当「开盘到次日开盘」收益宽表（仅接口烟测）
    res = run_backtest(ar, ws, config=BacktestConfig(execution_mode="tplus1_open"))
    assert res.meta.get("execution_mode") == "tplus1_open"
    assert np.isfinite(res.panel.sharpe_ratio)


def test_run_backtest_vwap_execution_mode_penalizes_turnover():
    ar, ws = _dummy_market(80)
    base = run_backtest(ar, ws, config=BacktestConfig(execution_mode="close_to_close"))
    vwap = run_backtest(
        ar,
        ws,
        config=BacktestConfig(
            execution_mode="vwap",
            vwap_slippage_bps_per_side=5.0,
            vwap_impact_bps=20.0,
        ),
    )
    assert vwap.meta.get("execution_mode") == "vwap"
    assert vwap.panel.total_return <= base.panel.total_return + 1e-12


def test_run_backtest_equal_weight_no_cost():
    ar, ws = _dummy_market(80)
    res = run_backtest(ar, ws, config=BacktestConfig(cost_params=None))
    assert len(res.daily_returns) == len(ar)
    assert res.panel.n_periods == len(ar)
    assert np.isfinite(res.panel.sharpe_ratio)


def test_run_backtest_with_cost_reduces_return():
    ar, ws = _dummy_market(80)
    c = TransactionCostParams(
        commission_buy_bps=30.0,
        commission_sell_bps=30.0,
        slippage_bps_per_side=10.0,
        stamp_duty_sell_bps=10.0,
    )
    r0 = run_backtest(ar, ws, config=BacktestConfig(cost_params=None)).panel.total_return
    r1 = run_backtest(ar, ws, config=BacktestConfig(cost_params=c)).panel.total_return
    assert r1 < r0 or (not np.isfinite(r0) and not np.isfinite(r1))


def test_contiguous_time_splits_and_walk_forward():
    ar, ws = _dummy_market(120)
    idx = ar.index
    splits = contiguous_time_splits(idx, n_splits=4, min_train_days=15)
    assert len(splits) >= 1
    panels, detail, agg = walk_forward_backtest(ar, ws, splits, use_test_only=True)
    assert len(panels) == len(detail)
    assert agg.get("n_folds", 0) >= 1


def test_rolling_walk_forward():
    ar, ws = _dummy_market(200)
    wins = rolling_walk_forward_windows(ar.index, train_days=40, test_days=20, step_days=20)
    assert len(wins) >= 2
    panels, detail, agg = walk_forward_backtest(ar, ws, wins, use_test_only=True)
    assert len(panels) >= 1
