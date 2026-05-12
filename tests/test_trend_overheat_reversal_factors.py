"""Tests for W7 trend-overheat reversal factors."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.features.trend_overheat_reversal_factors import (
    TREND_OVERHEAT_REVERSAL_FACTOR_COLS,
    attach_trend_overheat_reversal_factors,
    compute_trend_overheat_reversal_factors,
)
from src.features.trend_persistence_factors import compute_trend_persistence_factors
from src.pipeline.shared_loaders import attach_feature_families

_DAILY_DDL = """
    CREATE TABLE a_share_daily (
        symbol VARCHAR, trade_date DATE, close DOUBLE, amount DOUBLE,
        turnover DOUBLE, volume DOUBLE, high DOUBLE, low DOUBLE, open DOUBLE
    )
"""


def _build_trend_daily_db(db_path: Path) -> None:
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="B")
    rows = []
    for i, d in enumerate(dates):
        up_close = 10.0 + i * 0.08
        down_close = 30.0 - i * 0.08
        rows.append((
            "000001",
            d.strftime("%Y-%m-%d"),
            round(up_close, 4),
            1_000_000.0,
            2.0,
            100_000.0,
            round(up_close * 1.01, 4),
            round(up_close * 0.99, 4),
            round(up_close * 0.995, 4),
        ))
        rows.append((
            "000002",
            d.strftime("%Y-%m-%d"),
            round(down_close, 4),
            1_000_000.0,
            2.0,
            100_000.0,
            round(down_close * 1.01, 4),
            round(down_close * 0.99, 4),
            round(down_close * 1.005, 4),
        ))

    with duckdb.connect(str(db_path)) as con:
        con.execute(_DAILY_DDL)
        con.executemany("INSERT INTO a_share_daily VALUES (?,?,?,?,?,?,?,?,?)", rows)


def test_compute_trend_overheat_reversal_is_independent_reverse_family(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)

    trend = compute_trend_persistence_factors(str(db_path), "2025-06-30").sort_values("symbol")
    overheat = compute_trend_overheat_reversal_factors(str(db_path), "2025-06-30").sort_values("symbol")

    assert list(overheat["symbol"]) == ["000001", "000002"]
    for col in TREND_OVERHEAT_REVERSAL_FACTOR_COLS:
        assert col in overheat.columns
        assert overheat[col].notna().all()

    joined = trend.merge(overheat, on=["symbol", "trade_date"])
    assert (
        joined["feature_trend_overheat_bear_state"]
        == 1.0 - joined["feature_trend_bull_state"]
    ).all()
    assert (
        joined["feature_trend_overheat_cooling_streak_days"]
        == -joined["feature_trend_streak_days"]
    ).all()
    assert (
        joined["feature_trend_overheat_ema_spread_reversal"].round(12)
        == (-joined["feature_trend_ema_spread"]).round(12)
    ).all()


def test_attach_trend_overheat_reversal_keeps_only_new_family_columns(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)
    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002"],
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30", "2025-06-30"]),
            "value": [1.0, 2.0, 3.0],
        }
    )

    out = attach_trend_overheat_reversal_factors(factors, str(db_path))

    assert len(out) == len(factors)
    assert "value" in out.columns
    for col in TREND_OVERHEAT_REVERSAL_FACTOR_COLS:
        assert col in out.columns
    assert "feature_trend_ema_spread" not in out.columns


def test_shared_loader_attaches_trend_overheat_reversal_zscores(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)
    dataset = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30"]),
            "symbol": ["000001", "000002"],
            "candidate_pool_version": ["U1_liquid_tradable", "U1_liquid_tradable"],
        }
    )

    out = attach_feature_families(dataset, db_path, ["trend_overheat_reversal"])

    for col in TREND_OVERHEAT_REVERSAL_FACTOR_COLS:
        assert col in out.columns
        assert f"{col}_z" in out.columns
        assert f"is_missing_{col}" in out.columns
