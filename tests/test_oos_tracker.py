"""OOS Tracker 单元测试。使用 in-memory DuckDB 不依赖真实文件。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.monitoring.oos_tracker import (
    OOSDegradationResult,
    OOSRecord,
    OOSTracker,
    record_monthly_oos,
)


@pytest.fixture
def tracker() -> OOSTracker:
    """创建基于临时文件的 tracker 实例。"""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test_oos.duckdb"
        t = OOSTracker(str(db))
        yield t
        t.close()


class TestOOSRecordInsert:
    def test_record_and_retrieve_basic(self, tracker):
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-01-15",
            top_k=20,
            candidate_pool="U1_liquid_tradable",
            cost_bps=10.0,
            predicted_excess_monthly=0.005,
            realized_excess_monthly=0.003,
            holdings=["000001", "000002", "000003"],
        )
        history = tracker.get_history("test_config")
        assert len(history) == 1
        row = history.iloc[0]
        assert row["predicted_excess_monthly"] == 0.005
        assert row["realized_excess_monthly"] == 0.003
        assert row["num_holdings"] == 3

    def test_record_without_realized(self, tracker):
        """先写预测值，relevance 暂缺。"""
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-02-15",
            predicted_excess_monthly=0.008,
        )
        # history 只返回有 realized 的
        history = tracker.get_history("test_config")
        assert len(history) == 0

    def test_update_realized(self, tracker):
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-03-15",
            predicted_excess_monthly=0.006,
        )
        ok = tracker.update_realized(
            "test_config",
            "2025-03-15",
            realized_excess_monthly=0.004,
            benchmark_return=0.001,
        )
        assert ok
        history = tracker.get_history("test_config")
        assert len(history) == 1
        assert history.iloc[0]["realized_excess_monthly"] == 0.004

    def test_get_latest_predicted_excess(self, tracker):
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-04-15",
            predicted_excess_monthly=0.007,
        )
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-05-15",
            predicted_excess_monthly=0.009,
        )
        latest = tracker.get_latest_predicted_excess("test_config")
        assert latest == 0.009

    def test_lookup_missing_config_returns_none(self, tracker):
        result = tracker.get_latest_predicted_excess("nonexistent")
        assert result is None

    def test_record_with_holding_returns(self, tracker):
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-06-15",
            predicted_excess_monthly=0.005,
            realized_excess_monthly=0.003,
            holding_returns={"000001": 0.02, "000002": -0.01},
            limit_up_excluded=2,
        )
        history = tracker.get_history("test_config")
        assert history.iloc[0]["limit_up_excluded"] == 2


class TestOOSDegradation:
    def test_no_degradation_when_above_threshold(self, tracker):
        for i in range(6):
            tracker.record_oos(
                config_id="test_config",
                signal_date=f"2025-{i+1:02d}-15",
                predicted_excess_monthly=0.01,
                realized_excess_monthly=0.008,
            )
        result = tracker.check_degradation(
            "test_config", min_months=3, threshold_ratio=0.5,
        )
        assert not result.degraded
        assert "OOS OK" in result.message

    def test_degradation_when_below_threshold(self, tracker):
        for i in range(6):
            tracker.record_oos(
                config_id="test_config",
                signal_date=f"2025-{i+1:02d}-15",
                predicted_excess_monthly=0.01,
                realized_excess_monthly=0.002,
            )
        result = tracker.check_degradation(
            "test_config", min_months=3, threshold_ratio=0.5,
        )
        assert result.degraded
        assert "OOS DEGRADED" in result.message

    def test_insufficient_data_no_degradation(self, tracker):
        tracker.record_oos(
            config_id="test_config",
            signal_date="2025-01-15",
            predicted_excess_monthly=0.01,
            realized_excess_monthly=0.002,
        )
        result = tracker.check_degradation(
            "test_config", min_months=3, threshold_ratio=0.5,
        )
        assert not result.degraded
        assert "数据不足" in result.message

    def test_consecutive_degraded_count(self, tracker):
        # 前3月 OK，后3月 degraded
        for i in range(3):
            tracker.record_oos(
                config_id="test_config",
                signal_date=f"2025-{i+1:02d}-15",
                predicted_excess_monthly=0.01,
                realized_excess_monthly=0.008,
            )
        for i in range(3, 6):
            tracker.record_oos(
                config_id="test_config",
                signal_date=f"2025-{i+1:02d}-15",
                predicted_excess_monthly=0.01,
                realized_excess_monthly=0.002,
            )
        result = tracker.check_degradation(
            "test_config", min_months=3, threshold_ratio=0.5,
        )
        assert result.degraded
        assert result.consecutive_degraded_months == 3


class TestOOSSummary:
    def test_summary_single_config(self, tracker):
        for i in range(3):
            tracker.record_oos(
                config_id="test_config",
                signal_date=f"2025-{i+1:02d}-15",
                predicted_excess_monthly=0.01,
                realized_excess_monthly=0.005 + i * 0.002,
            )
        df = tracker.summary("test_config")
        assert len(df) == 1
        assert df.iloc[0]["n_months"] == 3

    def test_summary_all_configs(self, tracker):
        tracker.record_oos(
            config_id="cfg_a", signal_date="2025-01-15",
            predicted_excess_monthly=0.01, realized_excess_monthly=0.005,
        )
        tracker.record_oos(
            config_id="cfg_b", signal_date="2025-01-15",
            predicted_excess_monthly=0.02, realized_excess_monthly=0.015,
        )
        df = tracker.summary()
        assert len(df) == 2


class TestRecordMonthlyOOS:
    def test_convenience_function(self, tmp_path):
        db = tmp_path / "test.duckdb"
        record_monthly_oos(
            db, "test_config", "2025-07-15",
            predicted_excess_monthly=0.005,
            realized_excess_monthly=0.003,
            holdings=["000001"],
        )
        # Verify by reading back
        t = OOSTracker(db)
        history = t.get_history("test_config")
        assert len(history) == 1
        t.close()


class TestOOSRecordDataclass:
    def test_oos_record_creation(self):
        rec = OOSRecord(
            config_id="test", signal_date="2025-01-15",
            top_k=20, candidate_pool="U1", cost_bps=10.0,
            predicted_excess_monthly=0.01,
        )
        assert rec.config_id == "test"
        assert rec.realized_excess_monthly is None

    def test_degradation_result_attributes(self):
        result = OOSDegradationResult(
            config_id="test", degraded=False,
            consecutive_degraded_months=0,
            recent_realized_mean=0.005,
            backtest_predicted_mean=0.01,
            ratio=0.5, threshold_ratio=0.5,
            message="OK",
        )
        assert not result.degraded
        assert result.ratio == 0.5
