from __future__ import annotations

import pandas as pd

from scripts.run_prefilter_universe_validation import summarize_subperiods, summarize_yearly


def test_summarize_yearly_reports_counts_and_extremes():
    idx = pd.to_datetime(["2025-01-02", "2025-12-31", "2026-01-02", "2026-12-31"])
    daily = pd.Series([0.10, -0.05, -0.10, 0.20], index=idx)
    bench = pd.Series([0.02, 0.02, 0.01, 0.01], index=idx)

    yearly_df, summary = summarize_yearly(daily, bench)

    assert list(yearly_df["year"]) == [2025, 2026]
    assert summary["positive_years"] == 2
    assert summary["negative_years"] == 0
    assert summary["best_year_return"] >= summary["worst_year_return"]


def test_summarize_subperiods_splits_fixed_buckets():
    idx = pd.to_datetime(["2021-01-04", "2022-12-30", "2023-01-03", "2024-12-31", "2025-01-02", "2026-04-18"])
    daily = pd.Series([0.01, -0.01, 0.02, -0.01, 0.03, -0.02], index=idx)

    subperiod_df, summary = summarize_subperiods(daily)

    assert list(subperiod_df["period"]) == ["2021-2022", "2023-2024", "2025-2026"]
    assert summary["subperiod_count"] == 3
    assert summary["max_subperiod_drawdown"] >= 0.0
