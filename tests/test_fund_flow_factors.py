from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fund_flow_factors import _compute_flow_factors


def test_compute_flow_factors_resets_streak_by_symbol_and_nan():
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001", "000002", "000002", "000002"],
            "trade_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            "main_net_inflow_pct": [1.0, 2.0, np.nan, -1.0, -2.0, 3.0],
            "super_large_net_inflow_pct": [0.5, 0.4, np.nan, -0.2, -0.3, 0.7],
            "small_net_inflow_pct": [-0.5, -0.6, np.nan, 0.2, 0.3, -0.4],
        }
    )

    out = _compute_flow_factors(df, windows=(3,))

    assert out["main_inflow_streak"].tolist() == [1.0, 2.0, 0.0, -1.0, -2.0, 1.0]


def test_compute_flow_factors_builds_cross_sectional_columns_and_drops_raw_inputs():
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001", "000002", "000002", "000002"],
            "trade_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            "main_net_inflow_pct": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            "super_large_net_inflow_pct": [0.2, 0.5, 0.8, 0.1, 0.3, 0.4],
            "small_net_inflow_pct": [-0.5, -1.0, -1.5, -0.2, -0.4, -0.6],
        }
    )

    out = _compute_flow_factors(df, windows=(3,))

    assert "main_inflow_z_3d" in out.columns
    assert "super_inflow_z_3d" in out.columns
    assert "flow_divergence_3d" in out.columns
    assert "main_net_inflow_pct" not in out.columns
    assert "super_large_net_inflow_pct" not in out.columns
    assert "small_net_inflow_pct" not in out.columns
    last_day = out.loc[out["trade_date"] == pd.Timestamp("2026-01-06")]
    assert np.isclose(float(last_day["main_inflow_z_3d"].mean()), 0.0)
