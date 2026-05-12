from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_trend_filter_audit import build_monthly_filter_audit, summarize_filter_audit
from src.research.gates import EXCESS_COL, LABEL_COL, TOP20_COL


def _dataset() -> pd.DataFrame:
    rows = []
    for month in ("2025-01-31", "2025-02-28"):
        for i in range(6):
            rows.append(
                {
                    "signal_date": pd.Timestamp(month),
                    "candidate_pool_version": "U1_liquid_tradable",
                    "candidate_pool_pass": True,
                    "symbol": f"{month[-2:]}{i:05d}",
                    LABEL_COL: float(i),
                    EXCESS_COL: float(i),
                    TOP20_COL: 1.0 if i >= 4 else 0.0,
                    "feature_trend_bull_state": 1.0 if i >= 3 else 0.0,
                    "feature_trend_flip_days_ago": float(i),
                }
            )
    return pd.DataFrame(rows)


def test_build_monthly_filter_audit_measures_selected_delta() -> None:
    monthly = build_monthly_filter_audit(_dataset())
    bull = monthly[
        monthly["filter_name"].eq("bull_state")
        & monthly["signal_date"].eq("2025-01-31")
    ].iloc[0]
    bear = monthly[
        monthly["filter_name"].eq("bear_state_reverse")
        & monthly["signal_date"].eq("2025-01-31")
    ].iloc[0]

    assert int(bull["after_count"]) == 3
    assert int(bear["after_count"]) == 3
    assert np.isclose(float(bull["selected_minus_all_excess"]), 1.5)
    assert np.isclose(float(bear["selected_minus_all_excess"]), -1.5)
    assert bool(bull["below_min_count"]) is True


def test_summarize_filter_audit_groups_by_pool_and_filter() -> None:
    summary = summarize_filter_audit(build_monthly_filter_audit(_dataset()))
    bull = summary[summary["filter_name"].eq("bull_state")].iloc[0]
    stable_q80 = summary[summary["filter_name"].eq("stable_q80")].iloc[0]

    assert int(bull["months"]) == 2
    assert np.isclose(float(bull["mean_after_count"]), 3.0)
    assert np.isclose(float(bull["selected_minus_all_excess_mean"]), 1.5)
    assert np.isclose(float(stable_q80["mean_after_count"]), 2.0)
