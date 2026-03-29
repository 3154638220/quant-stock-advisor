"""数据质量检查单元测试。"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.data_fetcher.data_quality import (
    QualityConfig,
    run_quality_checks,
    validate_daily_frame,
)


def test_validate_daily_frame_ohlc_ok() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.5],
            "high": [10.2, 11.0],
            "low": [9.9, 10.4],
            "close": [10.1, 10.8],
            "volume": [1e6, 1e6],
        }
    )
    r = validate_daily_frame(df)
    assert r.ok
    assert r.duplicate_pk_rows == 0
    assert r.ohlc_invalid_rows == 0


def test_validate_daily_frame_duplicate_pk() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "open": [10.0, 10.0],
            "high": [10.2, 10.2],
            "low": [9.9, 9.9],
            "close": [10.1, 10.1],
        }
    )
    r = validate_daily_frame(df)
    assert not r.ok
    assert r.duplicate_pk_rows == 1


def test_validate_daily_frame_ohlc_bad() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [9.5],
            "low": [9.9],
            "close": [10.1],
        }
    )
    r = validate_daily_frame(df)
    assert not r.ok
    assert r.ohlc_invalid_rows == 1


def test_validate_daily_frame_ohlc_bad_ignored_when_not_fail_on() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [9.5],
            "low": [9.9],
            "close": [10.1],
        }
    )
    r = validate_daily_frame(df, cfg=QualityConfig(fail_on_ohlc_invalid=False))
    assert r.ok
    assert r.ohlc_invalid_rows == 1


def test_run_quality_checks_sql_dup_and_gap() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE t (
          symbol VARCHAR,
          trade_date DATE,
          open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
          volume DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO t VALUES
        ('000001', '2024-01-02', 10, 10.5, 9.9, 10.1, 1e6),
        ('000001', '2024-01-03', 10.1, 10.8, 10.0, 10.5, 1e6),
        ('000001', '2024-02-15', 10.5, 11, 10.4, 10.9, 1e6)
        """
    )
    # 1-03 到 2-15 间隔 > 20 天 → large_gap_rows >= 1
    cfg = QualityConfig(max_calendar_gap_days=20, null_ratio_max=1.0)
    r = run_quality_checks(con, "t", cfg)
    assert r.large_gap_rows >= 1

    con.execute(
        """
        INSERT INTO t VALUES
        ('000002', '2024-01-02', 1, 1, 1, 1, 1),
        ('000002', '2024-01-02', 1, 1, 1, 1, 1)
        """
    )
    r2 = run_quality_checks(con, "t", cfg)
    assert r2.duplicate_pk_rows >= 1


def test_run_quality_checks_gaps_ok_when_not_fail_on() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE t2 (
          symbol VARCHAR,
          trade_date DATE,
          open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
          volume DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO t2 VALUES
        ('000001', '2024-01-02', 10, 10.5, 9.9, 10.1, 1e6),
        ('000001', '2024-02-15', 10.5, 11, 10.4, 10.9, 1e6)
        """
    )
    cfg = QualityConfig(
        max_calendar_gap_days=20,
        null_ratio_max=1.0,
        fail_on_large_gaps=False,
    )
    r = run_quality_checks(con, "t2", cfg)
    assert r.large_gap_rows >= 1
    assert r.ok
