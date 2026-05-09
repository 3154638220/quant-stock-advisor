"""Tests for W1 Phase 3 liquidity & price position extension factors."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.features.liquidity_position_factors import (
    LIQUIDITY_POSITION_FACTOR_COLS,
    attach_liquidity_position_factors,
    compute_liquidity_position_factors,
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
        dates = pd.date_range("2024-12-01", "2025-06-30", freq="B")
        rows = []
        np.random.seed(42)
        for sym in symbols:
            price = 10.0 if sym == "000001" else 20.0
            for d in dates:
                ret = np.random.normal(0.0005, 0.02)
                price = price * (1 + ret)
                amt = abs(price * np.random.uniform(5e6, 5e7))
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                rows.append((
                    sym, d.strftime("%Y-%m-%d"), round(price, 2),
                    round(amt, 0), round(np.random.uniform(1, 8), 2),
                    round(amt / price, 0), round(high, 2),
                    round(low, 2), round(price * 0.995, 2),
                ))
        con.executemany("INSERT INTO a_share_daily VALUES (?,?,?,?,?,?,?,?,?)", rows)
    finally:
        con.close()


class TestComputeLiquidityPositionFactors:
    def test_returns_empty_for_missing_table(self, tmp_path: Path):
        db_path = tmp_path / "empty.duckdb"
        result = compute_liquidity_position_factors(str(db_path), "2025-06-30")
        assert result.empty

    def test_computes_all_factor_cols(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_liquidity_position_factors(str(db_path), "2025-06-30")
        assert not result.empty
        for col in LIQUIDITY_POSITION_FACTOR_COLS:
            assert col in result.columns, f"Missing column: {col}"
        assert "symbol" in result.columns
        assert "trade_date" in result.columns

    def test_non_empty_values(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_liquidity_position_factors(str(db_path), "2025-06-30")
        for col in LIQUIDITY_POSITION_FACTOR_COLS:
            notna = result[col].notna().sum()
            assert notna > 0, f"All NaN in {col}"

    def test_amihud_positive(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_liquidity_position_factors(str(db_path), "2025-06-30")
        illiq = result["feature_liquidity_amihud"].dropna()
        assert (illiq >= 0).all(), "Amihud should be non-negative"

    def test_high52w_ratio_bounded(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        result = compute_liquidity_position_factors(str(db_path), "2025-06-30")
        ratio = result["feature_liquidity_high52w_ratio"].dropna()
        assert (ratio > 0).all(), "high52w_ratio should be positive"
        assert (ratio <= 1.01).all(), "high52w_ratio should be <= 1"


class TestAttachLiquidityPositionFactors:
    def test_attaches_to_factor_frame(self, tmp_path: Path):
        db_path = tmp_path / "test.duckdb"
        _build_daily_db(db_path)
        factors = pd.DataFrame({
            "symbol": ["000001", "000002"],
            "signal_date": pd.to_datetime(["2025-06-30", "2025-06-30"]),
        })
        out = attach_liquidity_position_factors(factors, str(db_path))
        assert not out.empty
        for col in LIQUIDITY_POSITION_FACTOR_COLS:
            assert col in out.columns, f"Missing column: {col}"

    def test_empty_db_returns_nan_columns(self, tmp_path: Path):
        db_path = tmp_path / "empty.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(_DAILY_DDL)
        con.close()
        factors = pd.DataFrame({
            "symbol": ["000001"],
            "signal_date": pd.to_datetime(["2025-06-30"]),
        })
        out = attach_liquidity_position_factors(factors, str(db_path))
        for col in LIQUIDITY_POSITION_FACTOR_COLS:
            assert col in out.columns
            assert out[col].isna().all()
