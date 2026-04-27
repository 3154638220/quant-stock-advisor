"""数据质量检查单元测试。"""

from __future__ import annotations

import duckdb
import pandas as pd

from scripts.run_newdata_quality_checks import (
    build_newdata_quality_output_stem,
    build_newdata_quality_research_config_id,
)
from src.data_fetcher.data_quality import (
    QualityConfig,
    run_fund_flow_quality_checks,
    run_quality_checks,
    run_shareholder_quality_checks,
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


def test_run_fund_flow_quality_checks_detects_alignment_and_zero_rows() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE a_share_daily (
          symbol VARCHAR,
          trade_date DATE
        )
        """
    )
    con.execute(
        """
        INSERT INTO a_share_daily VALUES
        ('000001', '2024-01-02'),
        ('000001', '2024-01-03'),
        ('000002', '2024-01-02')
        """
    )
    con.execute(
        """
        CREATE TABLE a_share_fund_flow (
          symbol VARCHAR,
          trade_date DATE,
          main_net_inflow_pct DOUBLE,
          super_large_net_inflow_pct DOUBLE,
          small_net_inflow_pct DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO a_share_fund_flow VALUES
        ('000001', '2024-01-02', 1.0, 0.5, -0.2),
        ('000001', '2024-01-02', 1.0, 0.5, -0.2),
        ('000001', '2024-01-03', 0.0, 0.0, 0.0),
        ('000002', '2024-01-05', 2.0, NULL, -1.0)
        """
    )
    r = run_fund_flow_quality_checks(con)
    assert not r.ok
    assert r.table_exists
    assert r.duplicate_pk_rows == 1
    assert r.rows_without_daily_match == 1
    assert r.rows_after_daily_max_date == 1
    assert r.rows_without_daily_match_within_daily_span == 0
    assert r.rows_without_daily_match_absent_symbols == 0
    assert r.rows_without_daily_match_known_symbols == 0
    assert r.absent_symbol_count == 0
    assert r.daily_max_trade_date == "2024-01-03"
    assert r.all_zero_flow_rows == 1
    assert r.null_ratio_by_col["super_large_net_inflow_pct"] > 0.0
    assert r.coverage_ratio_vs_daily is not None


def test_run_fund_flow_quality_checks_separates_in_span_mismatch_from_stale_daily() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE a_share_daily (symbol VARCHAR, trade_date DATE)")
    con.execute(
        """
        INSERT INTO a_share_daily VALUES
        ('000001', '2024-01-02'),
        ('000001', '2024-01-04'),
        ('000002', '2024-01-02')
        """
    )
    con.execute(
        """
        CREATE TABLE a_share_fund_flow (
          symbol VARCHAR,
          trade_date DATE,
          main_net_inflow_pct DOUBLE,
          super_large_net_inflow_pct DOUBLE,
          small_net_inflow_pct DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO a_share_fund_flow VALUES
        ('000001', '2024-01-03', 1.0, 0.5, -0.2),
        ('000002', '2024-01-05', 2.0, 1.0, -1.0)
        """
    )

    r = run_fund_flow_quality_checks(con)

    assert r.rows_without_daily_match == 2
    assert r.rows_without_daily_match_within_daily_span == 1
    assert r.rows_after_daily_max_date == 1
    assert r.rows_without_daily_match_absent_symbols == 0
    assert r.rows_without_daily_match_known_symbols == 1
    assert any("日线已覆盖标的" in note for note in r.notes)


def test_run_fund_flow_quality_checks_treats_extra_market_symbols_as_nonblocking_note() -> None:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE a_share_daily (symbol VARCHAR, trade_date DATE)")
    con.execute("INSERT INTO a_share_daily VALUES ('000001', '2024-01-02')")
    con.execute(
        """
        CREATE TABLE a_share_fund_flow (
          symbol VARCHAR,
          trade_date DATE,
          main_net_inflow_pct DOUBLE,
          super_large_net_inflow_pct DOUBLE,
          small_net_inflow_pct DOUBLE
        )
        """
    )
    con.execute("INSERT INTO a_share_fund_flow VALUES ('920001', '2024-01-02', 1.0, 0.5, -0.2)")

    r = run_fund_flow_quality_checks(con)

    assert r.rows_without_daily_match_within_daily_span == 1
    assert r.rows_without_daily_match_absent_symbols == 1
    assert r.rows_without_daily_match_known_symbols == 0
    assert r.absent_symbol_count == 1
    assert any("完全不在日线表" in note for note in r.notes)
    assert r.ok


def test_run_shareholder_quality_checks_reports_notice_and_width() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE a_share_shareholder (
          symbol VARCHAR,
          end_date DATE,
          notice_date DATE,
          holder_count BIGINT,
          holder_change BIGINT
        )
        """
    )
    con.execute(
        """
        INSERT INTO a_share_shareholder VALUES
        ('000001', '2024-03-31', '2024-04-10', 1000, 50),
        ('000002', '2024-03-31', NULL, 1200, -20),
        ('000003', '2024-03-31', '2024-03-20', 900, 10),
        ('000001', '2024-03-31', '2024-04-10', 1000, 50)
        """
    )
    r = run_shareholder_quality_checks(con, min_effective_width=2)
    assert not r.ok
    assert r.table_exists
    assert r.duplicate_pk_rows == 1
    assert r.notice_date_coverage_ratio == 0.75
    assert r.fallback_lag_usage_ratio == 0.25
    assert r.negative_notice_lag_rows == 1
    assert r.median_symbols_per_end_date == 3.0
    assert r.effective_factor_dates_ge_min_width == 2


def test_run_shareholder_quality_checks_falls_back_for_notice_before_end_date() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE a_share_shareholder (
          symbol VARCHAR,
          end_date DATE,
          notice_date DATE,
          holder_count BIGINT,
          holder_change BIGINT
        )
        """
    )
    con.execute(
        """
        INSERT INTO a_share_shareholder VALUES
        ('000001', '2024-03-31', '2024-03-30', 1000, 50),
        ('000002', '2024-03-31', '2024-04-30', 1200, -20)
        """
    )

    r = run_shareholder_quality_checks(con, fallback_lag_days=30, min_effective_width=2)

    assert r.ok
    assert r.negative_notice_lag_rows == 1
    assert r.notice_date_coverage_ratio == 1.0
    assert r.fallback_lag_usage_ratio == 0.0
    assert any("保守处理" in note for note in r.notes)


def test_newdata_quality_research_identity_is_stable() -> None:
    config_id = build_newdata_quality_research_config_id(
        families=["fund_flow", "shareholder"],
        flow_table="a_share_fund_flow",
        shareholder_table="a_share_shareholder",
        daily_table="a_share_daily",
        shareholder_fallback_lag_days=30,
        min_effective_width=100,
    )
    assert (
        config_id
        == "families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100"
    )
    stem = build_newdata_quality_output_stem(
        output_prefix="newdata_quality",
        research_config_id=config_id,
    )
    assert (
        stem
        == "newdata_quality_families_fund_flow-shareholder_flow_a_share_fund_flow_holder_a_share_shareholder_daily_a_share_daily_lag_30_width_100"
    )
