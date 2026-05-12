"""Tests for W7 trend persistence factors."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.features.trend_persistence_factors import (
    TREND_PERSISTENCE_FACTOR_COLS,
    attach_trend_persistence_factors,
    compute_trend_persistence_factors,
)
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


def test_compute_trend_persistence_returns_empty_for_missing_table(tmp_path: Path):
    result = compute_trend_persistence_factors(str(tmp_path / "empty.duckdb"), "2025-06-30")

    assert result.empty


def test_compute_trend_persistence_factor_columns_and_direction(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)

    result = compute_trend_persistence_factors(str(db_path), "2025-06-30").sort_values("symbol")

    assert not result.empty
    for col in TREND_PERSISTENCE_FACTOR_COLS:
        assert col in result.columns
        assert result[col].notna().any(), f"All NaN in {col}"

    up = result[result["symbol"] == "000001"].iloc[0]
    down = result[result["symbol"] == "000002"].iloc[0]
    assert up["feature_trend_bull_state"] == 1.0
    assert down["feature_trend_bull_state"] == 0.0
    assert up["feature_trend_streak_days"] > 0
    assert down["feature_trend_streak_days"] < 0
    assert up["feature_trend_ema_spread"] > 0
    assert down["feature_trend_ema_spread"] < 0
    assert up["feature_trend_bull_ratio_60d"] > down["feature_trend_bull_ratio_60d"]
    assert np.isfinite(result["feature_trend_flip_days_ago"]).all()


def test_attach_trend_persistence_factors_keeps_duplicate_rows(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)
    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002"],
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30", "2025-06-30"]),
            "value": [1.0, 2.0, 3.0],
        }
    )

    out = attach_trend_persistence_factors(factors, str(db_path))

    assert len(out) == len(factors)
    for col in TREND_PERSISTENCE_FACTOR_COLS:
        assert col in out.columns


def test_shared_loader_attaches_trend_persistence_zscores(tmp_path: Path):
    db_path = tmp_path / "trend.duckdb"
    _build_trend_daily_db(db_path)
    dataset = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30"]),
            "symbol": ["000001", "000002"],
            "candidate_pool_version": ["U1_liquid_tradable", "U1_liquid_tradable"],
        }
    )

    out = attach_feature_families(dataset, db_path, ["trend_persistence"])

    for col in TREND_PERSISTENCE_FACTOR_COLS:
        assert col in out.columns
        assert f"{col}_z" in out.columns
        assert f"is_missing_{col}" in out.columns
