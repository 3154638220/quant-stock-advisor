from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_trend_reverse_ic_audit import build_monthly_reverse_ic, summarize_reverse_ic
from src.research.gates import EXCESS_COL, LABEL_COL


def _dataset() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "signal_date": pd.Timestamp("2025-01-31"),
                "candidate_pool_version": "U1_liquid_tradable",
                "candidate_pool_pass": True,
                "symbol": f"00000{i}",
                LABEL_COL: float(i),
                EXCESS_COL: float(i),
                "feature_trend_bull_state": float(5 - i),
                "feature_trend_streak_days": float(5 - i),
                "feature_trend_bull_ratio_20d": float(5 - i),
                "feature_trend_bull_ratio_60d": float(5 - i),
                "feature_trend_flip_days_ago": float(i),
                "feature_trend_ema_spread": float(5 - i),
            }
        )
    second_month_bearish_scores = [5.0, 4.0, 3.0, 3.0, 1.0, 0.0]
    for i in range(6):
        bearish_score = second_month_bearish_scores[i]
        rows.append(
            {
                "signal_date": pd.Timestamp("2025-02-28"),
                "candidate_pool_version": "U1_liquid_tradable",
                "candidate_pool_pass": True,
                "symbol": f"00001{i}",
                LABEL_COL: float(i),
                EXCESS_COL: float(i),
                "feature_trend_bull_state": bearish_score,
                "feature_trend_streak_days": bearish_score,
                "feature_trend_bull_ratio_20d": bearish_score,
                "feature_trend_bull_ratio_60d": bearish_score,
                "feature_trend_flip_days_ago": float(i),
                "feature_trend_ema_spread": bearish_score,
            }
        )
    return pd.DataFrame(rows)


def test_build_monthly_reverse_ic_flips_rank_ic_sign() -> None:
    monthly = build_monthly_reverse_ic(
        _dataset(),
        factor_cols=("feature_trend_bull_state",),
        min_samples=3,
        bucket_count=3,
    )

    original = monthly[monthly["direction_variant"].eq("original")].sort_values("signal_date")
    reversed_ = monthly[monthly["direction_variant"].eq("reversed")].sort_values("signal_date")

    assert len(original) == 2
    assert len(reversed_) == 2
    assert np.allclose(original["rank_ic"].to_numpy(), -reversed_["rank_ic"].to_numpy())
    assert (original["rank_ic"] < 0).all()
    assert (reversed_["rank_ic"] > 0).all()


def test_summarize_reverse_ic_marks_only_reversed_gate_pass() -> None:
    monthly = build_monthly_reverse_ic(
        _dataset(),
        factor_cols=("feature_trend_bull_state", "feature_trend_flip_days_ago"),
        min_samples=3,
        bucket_count=3,
    )
    summary = summarize_reverse_ic(monthly)

    bull_reversed = summary[
        summary["factor"].eq("feature_trend_bull_state")
        & summary["direction_variant"].eq("reversed")
    ].iloc[0]
    flip_original = summary[
        summary["factor"].eq("feature_trend_flip_days_ago")
        & summary["direction_variant"].eq("original")
    ].iloc[0]

    assert bool(bull_reversed["reverse_gate_pass"]) is True
    assert bool(flip_original["reverse_gate_pass"]) is False
