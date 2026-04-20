from __future__ import annotations

import pandas as pd

from scripts.run_p1_residual_diagnostics import summarize_rebalance_churn, summarize_weight_profile


def test_summarize_weight_profile_reports_breadth_and_concentration():
    weights = pd.DataFrame(
        [
            {"000001": 0.5, "000002": 0.5, "000003": 0.0},
            {"000001": 0.7, "000002": 0.2, "000003": 0.1},
        ],
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
    )

    out = summarize_weight_profile(weights)

    assert out["mean_name_count"] == 2.5
    assert round(out["mean_effective_n"], 6) == round((2.0 + (1.0 / (0.49 + 0.04 + 0.01))) / 2.0, 6)
    assert out["mean_top1_weight"] == 0.6
    assert out["mean_top5_weight_sum"] == 1.0


def test_summarize_rebalance_churn_reports_overlap_and_turnover_sources():
    weights = pd.DataFrame(
        [
            {"000001": 0.5, "000002": 0.5, "000003": 0.0},
            {"000001": 0.4, "000002": 0.0, "000003": 0.6},
            {"000001": 0.0, "000002": 0.5, "000003": 0.5},
        ],
        index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]),
    )

    out = summarize_rebalance_churn(weights)

    assert out["mean_prev_overlap_count"] == 1.0
    assert out["mean_enter_count"] == 1.0
    assert out["mean_exit_count"] == 1.0
    assert out["mean_prev_overlap_ratio"] == 0.5
