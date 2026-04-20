from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_backtest_eval import _select_topk_with_holding_buffer, build_topk_weights


def test_select_topk_with_holding_buffer_keeps_previous_holdings_inside_buffer():
    day_df = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "000004", "000005"],
            "score": [0.80, 0.70, 0.95, 0.90, 0.10],
        }
    )

    out = _select_topk_with_holding_buffer(
        day_df,
        top_k=2,
        entry_top_k=2,
        hold_buffer_top_k=4,
        prev_holdings={"000001", "000002"},
        industry_map=None,
        industry_cap_count=None,
    )

    assert set(out["symbol"]) == {"000001", "000002"}


def test_build_topk_weights_supports_holding_buffer_config():
    score_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2026-01-31"] * 5 + ["2026-02-28"] * 5
            ),
            "symbol": [
                "000001",
                "000002",
                "000003",
                "000004",
                "000005",
                "000001",
                "000002",
                "000003",
                "000004",
                "000005",
            ],
            "score": [0.95, 0.90, 0.50, 0.40, 0.10, 0.80, 0.70, 0.99, 0.98, 0.05],
        }
    )
    factor_df = score_df[["trade_date", "symbol"]].copy()
    factor_df["turnover_roll_mean"] = 1.0
    factor_df["price_position"] = 0.5
    factor_df["limit_move_hits_5d"] = 0.0
    daily_df = pd.DataFrame()

    weights = build_topk_weights(
        score_df=score_df,
        factor_df=factor_df,
        daily_df=daily_df,
        top_k=2,
        rebalance_rule="M",
        prefilter_cfg={"enabled": False},
        max_turnover=1.0,
        entry_top_k=2,
        hold_buffer_top_k=4,
        portfolio_method="equal_weight",
    )

    second = weights.loc[pd.Timestamp("2026-02-28")]
    held = set(second[second > 0].index.tolist())
    assert held == {"000001", "000002"}


def test_build_topk_weights_supports_tiered_equal_weight_config():
    score_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-31"] * 4),
            "symbol": ["000001", "000002", "000003", "000004"],
            "score": [0.95, 0.90, 0.50, 0.40],
        }
    )
    factor_df = score_df[["trade_date", "symbol"]].copy()
    factor_df["turnover_roll_mean"] = 1.0
    factor_df["price_position"] = 0.5
    factor_df["limit_move_hits_5d"] = 0.0
    daily_df = pd.DataFrame()

    weights = build_topk_weights(
        score_df=score_df,
        factor_df=factor_df,
        daily_df=daily_df,
        top_k=4,
        rebalance_rule="M",
        prefilter_cfg={"enabled": False},
        max_turnover=1.0,
        top_tier_count=2,
        top_tier_weight_share=0.6,
        portfolio_method="tiered_equal_weight",
    )

    row = weights.loc[pd.Timestamp("2026-01-31")]
    assert row["000001"] == pytest.approx(0.3)
    assert row["000002"] == pytest.approx(0.3)
    assert row["000003"] == pytest.approx(0.2)
    assert row["000004"] == pytest.approx(0.2)
