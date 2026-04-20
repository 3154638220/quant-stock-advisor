from __future__ import annotations

import pandas as pd

from scripts.run_benchmark_gap_diagnostics import build_rank_coverage_tables, summarize_market_capture, summarize_monthly_excess


def test_summarize_monthly_excess_and_capture_split_up_down_months():
    idx = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-02-03",
            "2026-02-20",
            "2026-03-03",
            "2026-03-20",
        ]
    )
    strat = pd.Series([0.02, 0.01, -0.01, 0.00, -0.03, 0.01], index=idx)
    bench = pd.Series([0.01, 0.01, 0.02, 0.00, -0.02, 0.00], index=idx)

    monthly_df, summary = summarize_monthly_excess(strat, bench)
    capture_df = summarize_market_capture(monthly_df)

    assert list(monthly_df["month"]) == [1, 2, 3]
    assert summary["months"] == 3
    assert summary["negative_excess_months"] >= 1
    assert set(capture_df["regime"]) == {"benchmark_up", "benchmark_down"}


def test_build_rank_coverage_tables_flags_buffer_candidates():
    score_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-01-31",
                    "2026-01-31",
                    "2026-01-31",
                    "2026-02-28",
                    "2026-02-28",
                    "2026-02-28",
                    "2026-02-28",
                ]
            ),
            "symbol": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "score": [4.0, 3.0, 2.0, 1.0, 1.5, 5.0, 4.0, 3.0],
        }
    )
    asset_returns = pd.DataFrame(
        {
            "A": [0.0, 0.06, 0.0],
            "B": [0.0, 0.01, 0.0],
            "C": [0.0, -0.02, 0.0],
            "D": [0.0, -0.03, 0.0],
        },
        index=pd.to_datetime(["2026-01-31", "2026-02-15", "2026-02-28"]),
    )

    _, bucket_summary, threshold_df, summary = build_rank_coverage_tables(
        score_df=score_df,
        asset_returns=asset_returns,
        rebalance_rule="M",
    )

    assert not bucket_summary.empty
    assert not threshold_df.empty
    assert summary["rebalances"] == 1
