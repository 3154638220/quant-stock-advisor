"""W7 follow-up: trend-overheat reversal factors.

This family reuses the PIT-safe EMA12/EMA26 trend-state reconstruction from
``trend_persistence`` but exposes only the reverse-direction signals that passed
the offline reverse IC audit in both U1 and U2 pools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.trend_persistence_factors import (
    attach_trend_persistence_factors,
    compute_trend_persistence_factors,
)

TREND_OVERHEAT_REVERSAL_FACTOR_COLS: tuple[str, ...] = (
    "feature_trend_overheat_bear_state",
    "feature_trend_overheat_cooling_streak_days",
    "feature_trend_overheat_ema_spread_reversal",
)

TREND_OVERHEAT_REVERSAL_FACTOR_DIRECTION: dict[str, int] = {
    "feature_trend_overheat_bear_state": 1,
    "feature_trend_overheat_cooling_streak_days": 1,
    "feature_trend_overheat_ema_spread_reversal": 1,
}


def compute_trend_overheat_reversal_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    table_name: str = "a_share_daily",
    min_history_days: int = 90,
    min_coverage: float = 0.30,
    fast_span: int = 12,
    slow_span: int = 26,
) -> pd.DataFrame:
    """Compute the independent trend-overheat reversal factor family."""

    trend = compute_trend_persistence_factors(
        db_path,
        signal_date,
        table_name=table_name,
        min_history_days=min_history_days,
        min_coverage=min_coverage,
        fast_span=fast_span,
        slow_span=slow_span,
    )
    return _build_overheat_from_trend(trend, min_coverage=min_coverage, keep_input_columns=False)


def attach_trend_overheat_reversal_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_daily",
) -> pd.DataFrame:
    """Attach trend-overheat reversal factors to an existing monthly frame."""

    out = attach_trend_persistence_factors(factors, db_path, table_name=table_name)
    out = _build_overheat_from_trend(out, keep_input_columns=True)
    drop_cols = [
        "feature_trend_bull_state",
        "feature_trend_streak_days",
        "feature_trend_bull_ratio_20d",
        "feature_trend_bull_ratio_60d",
        "feature_trend_flip_days_ago",
        "feature_trend_ema_spread",
    ]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def _build_overheat_from_trend(
    trend: pd.DataFrame,
    *,
    min_coverage: float = 0.30,
    keep_input_columns: bool = True,
) -> pd.DataFrame:
    if trend.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *TREND_OVERHEAT_REVERSAL_FACTOR_COLS])

    out = trend.copy()
    bull_state = pd.to_numeric(out.get("feature_trend_bull_state"), errors="coerce")
    streak_days = pd.to_numeric(out.get("feature_trend_streak_days"), errors="coerce")
    ema_spread = pd.to_numeric(out.get("feature_trend_ema_spread"), errors="coerce")

    out["feature_trend_overheat_bear_state"] = 1.0 - bull_state
    out["feature_trend_overheat_cooling_streak_days"] = -streak_days
    out["feature_trend_overheat_ema_spread_reversal"] = -ema_spread

    for col in TREND_OVERHEAT_REVERSAL_FACTOR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cov = out[col].notna().mean() if len(out) else 0.0
        if cov < min_coverage:
            out[col] = np.nan

    if keep_input_columns:
        return out

    keep = [c for c in ["symbol", "trade_date", "signal_date"] if c in out.columns]
    return out[keep + list(TREND_OVERHEAT_REVERSAL_FACTOR_COLS)]
