"""Tests for W1 Phase 2 reversal & volume anomaly factors."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.features.reversal_volume_factors import (
    REVERSAL_VOLUME_FACTOR_COLS,
    attach_reversal_volume_factors,
    compute_reversal_volume_factors,
)

_DAILY_DDL = """
    CREATE TABLE a_share_daily (
        symbol VARCHAR, trade_date DATE, close DOUBLE, amount DOUBLE,
        turnover DOUBLE, volume DOUBLE, high DOUBLE, low DOUBLE, open DOUBLE
    )
"""


def _build_daily_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_DAILY_DDL)
        symbols = ["000001", "000002"]
        dates = pd.date_range("2025-01-01", "2025-06-30", freq="B")
        rows = []
        np.random.seed(42)
        for sym in symbols:
            price = 10.0 if sym == "000001" else 20.0
            for d in dates:
                ret = np.random.normal(0.0005, 0.02)
                price = price * (1 + ret)
                amt = abs(price * np.random.uniform(5e6, 5e7))
                vol = amt / price
                rows.append((
                    sym, d.strftime("%Y-%m-%d"), round(price, 2),
                    round(amt, 0), round(np.random.uniform(1, 8), 2),
                    round(vol, 0), round(price * 1.01, 2),
                    round(price * 0.99, 2), round(price * (1 - ret * 0.5), 2),
                ))
        con.executemany("INSERT INTO a_share_daily VALUES (?,?,?,?,?,?,?,?,?)", rows)
    finally:
        con.close()


class TestComputeReversalVolumeFactors:
    def test_returns_empty_for_missing_table(self, tmp_path: Path):
        db_path = tmp_path / "empty.duckdb"
        # No table created — should return empty DataFrame
        result = compute_reversal_volume_factors(str(db_path), "2025-06-30")
        assert result.empty

    def test_computes_all_factor_cols(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_reversal_volume_factors(str(db_path), "2025-06-30")
        assert not result.empty
        for col in REVERSAL_VOLUME_FACTOR_COLS:
            assert col in result.columns, f"Missing column: {col}"
        assert "symbol" in result.columns
        assert "trade_date" in result.columns

    def test_non_empty_values(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_reversal_volume_factors(str(db_path), "2025-06-30")
        for col in REVERSAL_VOLUME_FACTOR_COLS:
            notna = result[col].notna().sum()
            assert notna > 0, f"All NaN in {col}"

    def test_volume_spike_positive(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_reversal_volume_factors(str(db_path), "2025-06-30")
        spike = result["feature_reversal_volume_spike"].dropna()
        assert (spike > 0).all(), "volume_spike should be positive (ratio)"

    def test_st_reversal_is_finite(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_reversal_volume_factors(str(db_path), "2025-06-30")
        rev = result["feature_reversal_st_reversal_1m"].dropna()
        assert len(rev) > 0
        assert np.isfinite(rev).all()


class TestAttachReversalVolumeFactors:
    def test_attaches_to_factor_frame(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        factors = pd.DataFrame({
            "symbol": ["000001", "000002"],
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30"]),
        })
        out = attach_reversal_volume_factors(factors, str(db_path))
        assert not out.empty
        for col in REVERSAL_VOLUME_FACTOR_COLS:
            assert col in out.columns, f"Missing column: {col}"

    def test_handles_duplicate_dates(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        factors = pd.DataFrame({
            "symbol": ["000001", "000001"],
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30"]),
            "value": [1.0, 2.0],
        })
        out = attach_reversal_volume_factors(factors, str(db_path))
        assert len(out) == 2

    def test_empty_db_returns_nan_columns(self, tmp_path: Path):
        db_path = tmp_path / "empty.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(_DAILY_DDL)
        con.close()
        factors = pd.DataFrame({
            "symbol": ["000001"],
            "signal_date": pd.to_datetime(["2025-06-30"]),
        })
        out = attach_reversal_volume_factors(factors, str(db_path))
        for col in REVERSAL_VOLUME_FACTOR_COLS:
            assert col in out.columns
            assert out[col].isna().all()
