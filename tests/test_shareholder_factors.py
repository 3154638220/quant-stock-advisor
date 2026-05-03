from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.features.shareholder_factors import attach_shareholder_factors


def test_attach_shareholder_factors_respects_availability_lag(tmp_path: Path):
    db_path = tmp_path / "shareholder.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_shareholder (
                symbol VARCHAR,
                end_date DATE,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_shareholder VALUES
            ('000001', DATE '2026-01-31', DATE '2026-02-20', 1000, 100, 'test', NOW()),
            ('000001', DATE '2026-02-28', DATE '2026-03-20', 900, -100, 'test', NOW()),
            ('000002', DATE '2026-01-31', DATE '2026-02-25', 2000, 0, 'test', NOW())
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000001"],
            "trade_date": pd.to_datetime(["2026-02-15", "2026-03-05", "2026-03-05", "2026-04-10"]),
            "vol_to_turnover": [1.0, 1.1, 0.9, 1.2],
        }
    )

    out = attach_shareholder_factors(factors, str(db_path), availability_lag_days=30)
    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    assert np.isnan(out.loc[0, "holder_count"])
    assert out.loc[1, "holder_count"] == 1000
    assert out.loc[2, "holder_count"] == 2000
    assert out.loc[3, "holder_count"] == 900


def test_attach_shareholder_factors_falls_back_for_notice_before_end_date(tmp_path: Path):
    db_path = tmp_path / "shareholder_negative_notice.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_shareholder (
                symbol VARCHAR,
                end_date DATE,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_shareholder VALUES
            ('000001', DATE '2024-03-31', DATE '2024-03-30', 1000, 50, 'test', NOW())
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "trade_date": pd.to_datetime(["2024-04-15", "2024-04-30"]),
        }
    )
    out = attach_shareholder_factors(factors, str(db_path), availability_lag_days=30).sort_values("trade_date")

    assert np.isnan(out.loc[out["trade_date"] == pd.Timestamp("2024-04-15"), "holder_count"]).all()
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-04-30"), "holder_count"].iloc[0] == 1000


def test_attach_shareholder_factors_builds_cross_sectional_outputs(tmp_path: Path):
    db_path = tmp_path / "shareholder_cs.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_shareholder (
                symbol VARCHAR,
                end_date DATE,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_shareholder VALUES
            ('000001', DATE '2026-01-31', DATE '2026-02-20', 1000, 100, 'test', NOW()),
            ('000002', DATE '2026-01-31', DATE '2026-02-20', 2000, -100, 'test', NOW())
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2026-03-05", "2026-03-05"]),
        }
    )

    out = attach_shareholder_factors(factors, str(db_path), availability_lag_days=30).sort_values("symbol")

    assert "holder_change_rate_z" in out.columns
    assert "holder_count_log_z" in out.columns
    assert "holder_concentration_proxy" in out.columns
    assert np.isclose(float(out["holder_change_rate_z"].mean()), 0.0)


# ── P2-4: 除权跳变处理测试 ──────────────────────────────────────────────

def _make_dividend_calendar_table(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    con.execute(
        """
        CREATE TABLE a_share_dividend_calendar (
            symbol VARCHAR,
            ex_dividend_date DATE,
            dividend_plan VARCHAR,
            source VARCHAR,
            fetched_at TIMESTAMP
        )
        """
    )
    for row in rows:
        con.execute(
            """
            INSERT INTO a_share_dividend_calendar VALUES (?, ?, ?, 'test', NOW())
            """,
            [row["symbol"], row["ex_dividend_date"], row.get("dividend_plan", "")],
        )


def test_exdiv_calendar_nulls_holder_change_rate_near_ex_date(tmp_path: Path):
    """P2-4: 除权日历中存在的股票，其除权日附近的 holder_change_rate 应被置为 NaN。"""
    db_path = tmp_path / "shareholder_exdiv.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_shareholder (
                symbol VARCHAR,
                end_date DATE,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        # 000001 在 2024-06-15 除权，其 2024-03-31 的报告期应在窗口内
        con.execute(
            """
            INSERT INTO a_share_shareholder VALUES
            ('000001', DATE '2024-03-31', DATE '2024-04-20', 1000, 500, 'test', NOW()),
            ('000001', DATE '2024-06-30', DATE '2024-07-20', 800, -200, 'test', NOW()),
            ('000002', DATE '2024-03-31', DATE '2024-04-20', 2000, 0, 'test', NOW())
            """
        )
        _make_dividend_calendar_table(
            con,
            [
                {"symbol": "000001", "ex_dividend_date": pd.Timestamp("2024-06-15")},
            ],
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002"],
            "trade_date": pd.to_datetime(["2024-05-15", "2024-08-15", "2024-05-15"]),
        }
    )

    out = attach_shareholder_factors(factors, str(db_path), availability_lag_days=30)
    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    # 000001 @ 2024-05-15 的 end_date=2024-03-31，距除权日 6/15 在 92 天内 → holder_change_rate 应为 NaN
    row_near = out[(out["symbol"] == "000001") & (out["trade_date"] == pd.Timestamp("2024-05-15"))]
    assert len(row_near) == 1
    assert np.isnan(row_near["holder_change_rate"].iloc[0])

    # 000001 @ 2024-08-15 的 end_date=2024-06-30，也在窗口内
    row_far = out[(out["symbol"] == "000001") & (out["trade_date"] == pd.Timestamp("2024-08-15"))]
    assert len(row_far) == 1
    assert np.isnan(row_far["holder_change_rate"].iloc[0])

    # 000002 没有除权，holder_change_rate 应为正常值
    row_normal = out[(out["symbol"] == "000002") & (out["trade_date"] == pd.Timestamp("2024-05-15"))]
    assert len(row_normal) == 1
    assert not np.isnan(row_normal["holder_change_rate"].iloc[0])


def test_exdiv_calendar_no_table_no_effect(tmp_path: Path):
    """P2-4: 无除权日历表时，holder_change_rate 不受影响。"""
    db_path = tmp_path / "shareholder_no_exdiv.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_shareholder (
                symbol VARCHAR,
                end_date DATE,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_shareholder VALUES
            ('000001', DATE '2024-03-31', DATE '2024-04-20', 1000, 500, 'test', NOW())
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-05-15"]),
        }
    )
    out = attach_shareholder_factors(factors, str(db_path), availability_lag_days=30)
    assert not np.isnan(out["holder_change_rate"].iloc[0])
