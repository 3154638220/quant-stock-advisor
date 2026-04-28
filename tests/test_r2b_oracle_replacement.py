import pandas as pd
import pytest

from scripts.run_r2b_oracle_replacement_attribution import (
    build_forward_returns_from_open,
    build_oracle_weights,
    select_oracle_replacements,
)


def test_build_forward_returns_from_open_uses_next_open_entry():
    daily = pd.DataFrame(
        {
            "symbol": ["000001"] * 4,
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 11.0, 12.0, 15.0],
        }
    )

    out = build_forward_returns_from_open(daily, horizon=1)

    jan1 = out[out["trade_date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert jan1["forward_ret_1d"] == pytest.approx(12.0 / 11.0 - 1.0)


def test_select_oracle_replacements_enforces_unique_old_and_new():
    pairs = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2024-01-31"), "old_symbol": "000001", "new_symbol": "000101", "edge": 0.05},
            {"trade_date": pd.Timestamp("2024-01-31"), "old_symbol": "000001", "new_symbol": "000102", "edge": 0.04},
            {"trade_date": pd.Timestamp("2024-01-31"), "old_symbol": "000002", "new_symbol": "000101", "edge": 0.03},
            {"trade_date": pd.Timestamp("2024-01-31"), "old_symbol": "000002", "new_symbol": "000102", "edge": 0.02},
            {"trade_date": pd.Timestamp("2024-01-31"), "old_symbol": "000003", "new_symbol": "000103", "edge": -0.01},
        ]
    )

    selected = select_oracle_replacements(pairs, edge_col="edge", max_replace=3)

    assert selected["old_symbol"].tolist() == ["000001", "000002"]
    assert selected["new_symbol"].tolist() == ["000101", "000102"]


def test_build_oracle_weights_replaces_old_slot_weight_with_new_name():
    defensive = pd.DataFrame(
        [[0.5, 0.5, 0.0]],
        index=[pd.Timestamp("2024-01-31")],
        columns=["000001", "000002", "000101"],
    )
    selected = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-01-31"),
                "old_symbol": "000001",
                "new_symbol": "000101",
            }
        ]
    )

    out = build_oracle_weights(defensive, selected)

    assert out.loc[pd.Timestamp("2024-01-31"), "000001"] == 0.0
    assert out.loc[pd.Timestamp("2024-01-31"), "000002"] == pytest.approx(0.5)
    assert out.loc[pd.Timestamp("2024-01-31"), "000101"] == pytest.approx(0.5)
    assert out.loc[pd.Timestamp("2024-01-31")].sum() == pytest.approx(1.0)
