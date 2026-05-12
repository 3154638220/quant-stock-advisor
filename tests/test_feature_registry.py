from __future__ import annotations

from src.features.registry import FACTOR_REGISTRY, get_factor_cols


def test_trend_persistence_family_is_registered() -> None:
    cols = get_factor_cols("trend_persistence", use_zscore=False)

    assert cols == [
        "feature_trend_bull_state",
        "feature_trend_streak_days",
        "feature_trend_bull_ratio_20d",
        "feature_trend_bull_ratio_60d",
        "feature_trend_flip_days_ago",
        "feature_trend_ema_spread",
    ]
    specs = [spec for spec in FACTOR_REGISTRY.values() if spec.family == "trend_persistence"]
    assert len(specs) == 6
    assert all(spec.direction == 1 for spec in specs)
    assert all(spec.min_coverage == 0.30 for spec in specs)


def test_trend_overheat_reversal_family_is_registered() -> None:
    cols = get_factor_cols("trend_overheat_reversal", use_zscore=False)

    assert cols == [
        "feature_trend_overheat_bear_state",
        "feature_trend_overheat_cooling_streak_days",
        "feature_trend_overheat_ema_spread_reversal",
    ]
    specs = [spec for spec in FACTOR_REGISTRY.values() if spec.family == "trend_overheat_reversal"]
    assert len(specs) == 3
    assert all(spec.direction == 1 for spec in specs)
    assert all(spec.min_coverage == 0.30 for spec in specs)
