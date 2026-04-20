from __future__ import annotations

import pandas as pd

from scripts.run_score_ablation import summarize_overlap, summarize_pairwise_score_correlations


def test_summarize_overlap_counts_and_ratios():
    reference = {
        pd.Timestamp("2026-01-31"): {"000001", "000002", "000003"},
        pd.Timestamp("2026-02-28"): {"000004", "000005"},
    }
    candidate = {
        pd.Timestamp("2026-01-31"): {"000002", "000003", "000006"},
        pd.Timestamp("2026-02-28"): {"000004", "000007"},
    }

    out = summarize_overlap(reference, candidate)

    assert list(out["overlap_count"]) == [2, 1]
    assert list(out["candidate_only_count"]) == [1, 1]
    assert list(out["reference_only_count"]) == [1, 1]
    assert out.loc[0, "overlap_ratio_vs_candidate"] == 2 / 3
    assert out.loc[1, "overlap_ratio_vs_reference"] == 1 / 2


def test_summarize_pairwise_score_correlations_uses_daily_median():
    s1 = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-31", "2026-01-31", "2026-02-28", "2026-02-28"]),
            "symbol": ["000001", "000002", "000001", "000002"],
            "score": [1.0, 2.0, 2.0, 1.0],
        }
    )
    s2 = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-31", "2026-01-31", "2026-02-28", "2026-02-28"]),
            "symbol": ["000001", "000002", "000001", "000002"],
            "score": [10.0, 20.0, 10.0, 20.0],
        }
    )

    out = summarize_pairwise_score_correlations({"left": s1, "right": s2})

    assert len(out) == 1
    assert out.loc[0, "common_rows"] == 4
    assert out.loc[0, "mean_daily_corr"] == 0.0
    assert out.loc[0, "median_daily_corr"] == 0.0
