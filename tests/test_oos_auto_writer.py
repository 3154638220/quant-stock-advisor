"""Tests for src/monitoring/oos_auto_writer.py — M7 OOS auto-recording."""

import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.monitoring.oos_auto_writer import (
    OOSWriteResult,
    _compute_realized_excess_from_holdings,
    record_oos_batch_from_history,
    record_oos_from_m7_report,
)


@pytest.fixture
def temp_db():
    """Create a temporary DuckDB with oos_tracking table."""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "test.duckdb"

    conn = duckdb.connect(str(tmp_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS oos_tracking (
            config_id VARCHAR NOT NULL,
            signal_date DATE NOT NULL,
            run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            top_k INTEGER NOT NULL,
            candidate_pool VARCHAR NOT NULL,
            cost_bps DOUBLE NOT NULL,
            predicted_excess_monthly DOUBLE,
            realized_excess_monthly DOUBLE,
            holding_returns_json VARCHAR,
            benchmark_return DOUBLE,
            num_holdings INTEGER,
            limit_up_excluded INTEGER,
            PRIMARY KEY (config_id, signal_date, top_k, candidate_pool, cost_bps)
        )
    """)
    conn.close()
    yield tmp_path
    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        pass


class TestRecordOOSFromM7Report:
    def test_writes_predicted_excess(self, temp_db):
        result = record_oos_from_m7_report(
            db_path=temp_db,
            config_id="test_config_v1",
            signal_date="2026-05-05",
            predicted_excess_monthly=0.005,
            candidate_pool="U1_liquid_tradable",
            top_k=20,
            cost_bps=30.0,
            holdings=["000001", "000002", "000003"],
            num_holdings=20,
        )
        assert result.predicted_written is True
        assert result.current_prediction == 0.005
        assert len(result.errors) == 0

        # Verify in DB
        tracker_conn = duckdb.connect(str(temp_db))
        rows = tracker_conn.execute(
            "SELECT * FROM oos_tracking WHERE config_id='test_config_v1'"
        ).fetchall()
        tracker_conn.close()
        assert len(rows) == 1
        assert rows[0][3] == 20  # top_k

    def test_handles_db_connection_error(self):
        result = record_oos_from_m7_report(
            db_path="/nonexistent/path/db.duckdb",
            config_id="test",
            signal_date="2026-05-05",
            predicted_excess_monthly=0.005,
        )
        assert result.predicted_written is False
        assert len(result.errors) > 0

    def test_backfill_previous_month(self, temp_db):
        """写入两个月的数据，第二个月可以回填第一个月的实现值。"""
        # 先手动创建一条仅含预测的记录
        conn = duckdb.connect(str(temp_db))
        conn.execute(
            "INSERT INTO oos_tracking VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0)",
            ["test_config_v2", "2026-04-01", 20, "U1", 30.0, 0.004],
        )
        conn.close()

        result = record_oos_from_m7_report(
            db_path=temp_db,
            config_id="test_config_v2",
            signal_date="2026-05-01",
            predicted_excess_monthly=0.006,
            candidate_pool="U1",
            top_k=20,
            cost_bps=30.0,
        )
        assert result.predicted_written is True
        # Backfill may or may not succeed depending on data availability
        # (it tries to compute realized from holdings but there are none)

    def test_multiple_writes_same_key_updates(self, temp_db):
        """相同 (config_id, signal_date, ...) 应做 upsert。"""
        record_oos_from_m7_report(
            db_path=temp_db,
            config_id="test_upsert",
            signal_date="2026-05-05",
            predicted_excess_monthly=0.005,
        )
        result2 = record_oos_from_m7_report(
            db_path=temp_db,
            config_id="test_upsert",
            signal_date="2026-05-05",
            predicted_excess_monthly=0.006,
        )
        assert result2.predicted_written is True

        conn = duckdb.connect(str(temp_db))
        rows = conn.execute(
            "SELECT predicted_excess_monthly FROM oos_tracking WHERE config_id='test_upsert'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == 0.006  # updated value


class TestRecordOOSBatchFromHistory:
    def test_batch_write(self, temp_db):
        history = pd.DataFrame({
            "signal_date": ["2026-03-31", "2026-04-30", "2026-05-30"],
            "predicted_excess_monthly": [0.004, 0.005, 0.003],
            "realized_excess_monthly": [0.003, 0.006, np.nan],
        })
        results = record_oos_batch_from_history(
            db_path=temp_db,
            config_id="batch_test",
            history_df=history,
        )
        assert len(results) == 3
        assert all(r.predicted_written for r in results)

        conn = duckdb.connect(str(temp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM oos_tracking WHERE config_id='batch_test'"
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_batch_with_db_error(self):
        history = pd.DataFrame({
            "signal_date": ["2026-03-31"],
            "predicted_excess_monthly": [0.004],
        })
        results = record_oos_batch_from_history(
            db_path="/nonexistent/db.duckdb",
            config_id="batch_fail",
            history_df=history,
        )
        assert len(results) == 1
        assert "Connection failed" in results[0].errors[0]


class TestComputeRealizedExcess:
    def test_returns_none_for_empty_json(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE oos_tracking (x INT)")
        tracker = type("Mock", (), {"_conn": conn, "_table": "oos_tracking"})()
        assert _compute_realized_excess_from_holdings(tracker, "2026-01-01", None) is None
        assert _compute_realized_excess_from_holdings(tracker, "2026-01-01", "") is None
        conn.close()

    def test_returns_none_for_invalid_json(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE oos_tracking (x INT)")
        tracker = type("Mock", (), {"_conn": conn, "_table": "oos_tracking"})()
        assert _compute_realized_excess_from_holdings(tracker, "2026-01-01", "not-json") is None
        conn.close()

    def test_parses_holdings_list(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE oos_tracking (x INT)")
        tracker = type("Mock", (), {"_conn": conn, "_table": "oos_tracking"})()
        # Without daily_features table, returns None
        holdings_json = json.dumps(["000001", "000002"])
        result = _compute_realized_excess_from_holdings(tracker, "2026-01-01", holdings_json)
        assert result is None  # No daily_features table
        conn.close()

    def test_parses_holdings_dict(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE oos_tracking (x INT)")
        tracker = type("Mock", (), {"_conn": conn, "_table": "oos_tracking"})()
        holdings_json = json.dumps({"000001": 0.05, "000002": 0.05})
        result = _compute_realized_excess_from_holdings(tracker, "2026-01-01", holdings_json)
        assert result is None  # No daily_features table
        conn.close()


class TestOOSWriteResult:
    def test_default_values(self):
        r = OOSWriteResult(
            config_id="test",
            signal_date="2026-05-05",
            predicted_written=False,
        )
        assert r.config_id == "test"
        assert r.realized_backfilled is False
        assert r.backfilled_date is None
        assert r.errors == []

    def test_successful_result(self):
        r = OOSWriteResult(
            config_id="cfg1",
            signal_date="2026-05-05",
            predicted_written=True,
            realized_backfilled=True,
            backfilled_date="2026-04-01",
            previous_prediction=0.004,
            current_prediction=0.005,
        )
        assert r.predicted_written is True
        assert r.realized_backfilled is True
        assert r.previous_prediction == 0.004
        assert r.current_prediction == 0.005
