from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_monthly_selection_oracle import (
    build_oracle_topk_tables,
    build_realized_market_states,
    summarize_baseline_overlap,
    summarize_feature_bucket_monotonicity,
    summarize_oracle_by_candidate_pool,
    summarize_regime_oracle_capacity,
)


def _oracle_sample() -> pd.DataFrame:
    rows = []
    for date, market_ret, shift in [
        (pd.Timestamp("2024-01-31"), 0.01, 0.00),
        (pd.Timestamp("2024-02-29"), -0.02, 0.01),
    ]:
        for i in range(6):
            ret = -0.03 + i * 0.02 + shift
            rows.append(
                {
                    "signal_date": date,
                    "symbol": f"00000{i + 1}",
                    "candidate_pool_version": "U1_liquid_tradable",
                    "candidate_pool_pass": True,
                    "label_forward_1m_o2o_return": ret,
                    "label_market_ew_o2o_return": market_ret,
                    "label_future_top_20pct": 1 if i >= 4 else 0,
                    "feature_ret_20d_z": float(i),
                    "feature_realized_vol_20d_z": float(5 - i),
                    "industry_level1": "银行" if i % 2 == 0 else "计算机",
                }
            )
    return pd.DataFrame(rows)


def test_build_oracle_topk_tables_selects_future_winners_by_pool_month():
    df = _oracle_sample()

    monthly, holdings = build_oracle_topk_tables(df, top_ks=[2])
    summary = summarize_oracle_by_candidate_pool(monthly)

    jan = monthly[monthly["signal_date"] == pd.Timestamp("2024-01-31")].iloc[0]
    assert jan["oracle_topk_count"] == 2
    assert jan["oracle_topk_mean_return"] == pytest.approx((0.05 + 0.07) / 2.0)
    assert jan["oracle_topk_excess_vs_market"] == pytest.approx(((0.05 + 0.07) / 2.0) - 0.01)
    assert set(holdings[holdings["signal_date"] == pd.Timestamp("2024-01-31")]["symbol"]) == {
        "000005",
        "000006",
    }
    assert summary.iloc[0]["months"] == 2


def test_feature_bucket_monotonicity_and_baseline_overlap_detect_obvious_signal():
    df = _oracle_sample()

    buckets = summarize_feature_bucket_monotonicity(df, bucket_count=3)
    momentum = buckets[
        (buckets["candidate_pool_version"] == "U1_liquid_tradable")
        & (buckets["feature"] == "momentum_20d")
    ]
    assert momentum["bucket_return_spearman"].dropna().iloc[0] > 0

    overlap = summarize_baseline_overlap(df, top_ks=[2])
    row = overlap[
        (overlap["candidate_pool_version"] == "U1_liquid_tradable")
        & (overlap["baseline"] == "momentum_20d")
        & (overlap["top_k"] == 2)
    ].iloc[0]
    assert row["mean_oracle_topk_overlap_share"] == pytest.approx(1.0)
    assert row["mean_oracle_top20_bucket_hit_share"] == pytest.approx(1.0)


def test_regime_oracle_capacity_uses_realized_market_state_slice():
    df = _oracle_sample()
    monthly, _ = build_oracle_topk_tables(df, top_ks=[2])
    states = build_realized_market_states(df)

    out = summarize_regime_oracle_capacity(monthly, states)

    assert set(out["realized_market_state"]) == {"strong_down", "strong_up"}
    assert out["months"].sum() == 2
