from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_p1_strong_up_attribution import classify_regimes, compute_r1_extra_features
from src.models.xtree.p1_workflow import build_market_ew_open_to_open_benchmark


def test_r1_limit_path_features_split_up_down_and_move_hits():
    dates = pd.bdate_range("2024-01-02", periods=20)
    pct = [10.0, -10.0] + [0.0] * 18
    daily = pd.DataFrame(
        {
            "symbol": ["000001"] * len(dates),
            "trade_date": dates,
            "open": [10.0] * len(dates),
            "close": [10.0] * len(dates),
            "amount": [100_000_000.0] * len(dates),
            "volume": [1_000_000.0] * len(dates),
            "turnover": [1.0] * len(dates),
            "pct_chg": pct,
        }
    )

    out = compute_r1_extra_features(daily)
    last = out.iloc[-1]

    assert last["limit_up_hits_20d"] == pytest.approx(1.0)
    assert last["limit_down_hits_20d"] == pytest.approx(1.0)
    assert last["limit_move_hits_20d"] == pytest.approx(2.0)


def test_open_to_open_market_benchmark_matches_strategy_execution_clock():
    daily = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "symbol": ["000001"] * 3 + ["000002"] * 3,
            "open": [10.0, 11.0, 12.1, 20.0, 20.0, 22.0],
            "close": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        }
    )

    bench = build_market_ew_open_to_open_benchmark(
        daily,
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-04"),
        min_days=1,
    )

    assert bench.loc[pd.Timestamp("2024-01-02")] == pytest.approx((0.10 + 0.0) / 2.0)
    assert bench.loc[pd.Timestamp("2024-01-03")] == pytest.approx((0.10 + 0.10) / 2.0)


def test_expanding_regime_thresholds_do_not_label_with_future_quantiles():
    months = pd.date_range("2023-01-31", periods=14, freq="ME")
    monthly = pd.DataFrame(
        {
            "month_end": months,
            "benchmark_return": [-0.01] * 12 + [0.20, -0.20],
            "strategy_return": [0.0] * 14,
            "excess_return": [0.0] * 14,
        }
    )
    breadth = pd.Series([0.5] * 14, index=months)

    out = classify_regimes(monthly, breadth, threshold_mode="expanding", min_periods=12)

    assert set(out.loc[:10, "regime"]) == {"neutral"}
    assert out.loc[12, "regime"] == "strong_up"
    assert out.loc[13, "regime"] == "strong_down"
    assert set(out["lookahead_check"]) == {"pass"}
    assert out.loc[12, "threshold_observations"] == 13
    assert pd.notna(out.loc[12, "regime_p20"])
