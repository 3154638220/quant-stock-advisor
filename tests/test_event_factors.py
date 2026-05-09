"""Tests for W4 structured event factors."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.data_fetcher.migrations import apply_migrations
from src.features.event_factors import (
    EVENT_FACTOR_COLS,
    attach_event_factors,
    compute_event_factors,
)
from src.pipeline.monthly_multisource import build_feature_specs


def _seed_event_tables(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        apply_migrations(con)
        con.execute(
            """
            INSERT INTO a_share_event_earnings_guidance
            (symbol, announce_date, report_period, guidance_direction, guidance_change_ratio,
             expected_net_profit_min, expected_net_profit_max, prev_year_net_profit, source)
            VALUES
            ('000001', DATE '2025-12-20', DATE '2025-12-31', '预增', 0.05, 105, 110, 100, 'unit'),
            ('000001', DATE '2026-02-15', DATE '2026-03-31', '预增', 0.10, 110, 115, 100, 'unit'),
            ('000001', DATE '2026-03-10', DATE '2026-03-31', '上调', 0.20, 120, 130, 100, 'unit'),
            ('000001', DATE '2026-04-20', DATE '2026-06-30', '下调', -0.08, 88, 92, 100, 'unit'),
            ('000002', DATE '2026-04-15', DATE '2026-06-30', '预增', 0.12, 112, 116, 100, 'unit')
            """
        )
        con.execute(
            """
            INSERT INTO a_share_event_buyback
            (symbol, announce_date, buyback_amount, market_cap, progress_status, source)
            VALUES
            ('000001', DATE '2026-04-10', 10, 1000, 'plan', 'unit'),
            ('000001', DATE '2026-04-28', 5, 1000, 'done', 'unit'),
            ('000001', DATE '2026-03-20', 9, 1000, 'plan', 'unit')
            """
        )
        con.execute(
            """
            INSERT INTO a_share_event_reduction
            (symbol, announce_date, reduction_ratio, reduction_amount, holder_name, source)
            VALUES
            ('000001', DATE '2026-04-25', 0.01, 8, 'holderA', 'unit')
            """
        )
        con.execute(
            """
            INSERT INTO a_share_event_unlock
            (symbol, announce_date, unlock_date, unlock_market_value, market_cap, source)
            VALUES
            ('000001', DATE '2026-04-01', DATE '2026-05-20', 30, 1000, 'unit'),
            ('000001', DATE '2026-04-01', DATE '2026-06-30', 20, 1000, 'unit')
            """
        )
    finally:
        con.close()


def test_compute_event_factors_is_pit_safe_and_windowed(tmp_path: Path):
    db_path = tmp_path / "events.duckdb"
    _seed_event_tables(db_path)

    early = compute_event_factors(str(db_path), "2026-04-01")
    assert not early.empty
    e001 = early.loc[early["symbol"] == "000001"].iloc[0]
    # 2026-04-20 的下调公告尚不可见（PIT）
    assert e001["feature_event_earnings_guidance_direction"] > 0
    # buyback 30 日窗口：03-20 计入，04-10 不计入
    assert e001["feature_event_buyback_recent_30d"] == 1

    late = compute_event_factors(str(db_path), "2026-04-30")
    l001 = late.loc[late["symbol"] == "000001"].iloc[0]
    assert l001["feature_event_earnings_guidance_direction"] < 0
    assert l001["feature_event_buyback_amount_ratio"] > 0
    assert l001["feature_event_reduction_plan_flag"] == 1
    # 仅统计 future 30 天解禁：2026-05-20 计入，2026-06-30 不计入
    assert abs(float(l001["feature_event_unlock_ratio_30d"]) - 0.03) < 1e-9
    assert pd.notna(l001["feature_event_earnings_surprise_ttm"])


def test_attach_event_factors_adds_columns_and_preserves_rows(tmp_path: Path):
    db_path = tmp_path / "events.duckdb"
    _seed_event_tables(db_path)
    factors = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-03-31", "2026-04-30", "2026-04-30"]),
            "symbol": ["000001", "000001", "000002"],
            "x": [1, 2, 3],
        }
    )
    out = attach_event_factors(factors, str(db_path))
    assert len(out) == 3
    for col in EVENT_FACTOR_COLS:
        assert col in out.columns
    before = out[(out["symbol"] == "000001") & (out["signal_date"] == pd.Timestamp("2026-03-31"))].iloc[0]
    assert before["feature_event_buyback_recent_30d"] in (0, 1)


def test_event_feature_spec_includes_event_columns():
    specs = build_feature_specs(["event"])
    event_spec = specs[-1]
    assert "feature_event_earnings_guidance_direction_z" in event_spec.feature_cols
    assert "feature_event_buyback_amount_ratio_z" in event_spec.feature_cols
    assert "feature_event_unlock_ratio_30d_z" in event_spec.feature_cols
