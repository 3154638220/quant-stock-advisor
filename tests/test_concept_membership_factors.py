from __future__ import annotations

import duckdb
import pandas as pd

from src.data_fetcher.migrations import apply_migrations
from src.features.concept_factors import attach_concept_features
from src.pipeline.monthly_multisource import build_feature_specs


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-02-29", "2024-03-31", "2024-06-30", "2024-03-31"]),
            "symbol": ["000001", "000001", "000001", "000002"],
        }
    )


def test_concept_membership_features_are_pit_safe_and_snapshot_bounded(tmp_path):
    db_path = tmp_path / "concept.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        apply_migrations(con)
        con.execute(
            """
            INSERT INTO a_share_concept_daily
            (concept_code, trade_date, open, close, high, low, pct_chg, volume, amount)
            VALUES
            ('C1', DATE '2024-03-01', 1, 1, 1, 1, 1.0, 1, 1),
            ('C1', DATE '2024-03-15', 1, 1, 1, 1, 1.0, 1, 1),
            ('C1', DATE '2024-03-29', 1, 1, 1, 1, 1.0, 1, 1),
            ('C2', DATE '2024-03-01', 1, 1, 1, 1, -1.0, 1, 1),
            ('C2', DATE '2024-03-15', 1, 1, 1, 1, -1.0, 1, 1),
            ('C2', DATE '2024-03-29', 1, 1, 1, 1, -1.0, 1, 1),
            ('C3', DATE '2024-03-01', 1, 1, 1, 1, 0.5, 1, 1),
            ('C3', DATE '2024-03-15', 1, 1, 1, 1, 0.5, 1, 1),
            ('C3', DATE '2024-03-29', 1, 1, 1, 1, 0.5, 1, 1),
            ('C4', DATE '2024-03-01', 1, 1, 1, 1, 0.2, 1, 1),
            ('C4', DATE '2024-03-15', 1, 1, 1, 1, 0.2, 1, 1),
            ('C4', DATE '2024-03-29', 1, 1, 1, 1, 0.2, 1, 1),
            ('C5', DATE '2024-03-01', 1, 1, 1, 1, 0.1, 1, 1),
            ('C5', DATE '2024-03-15', 1, 1, 1, 1, 0.1, 1, 1),
            ('C5', DATE '2024-03-29', 1, 1, 1, 1, 0.1, 1, 1)
            """
        )
        con.execute(
            """
            INSERT INTO a_share_concept_membership
            (symbol, concept_code, snapshot_date, concept_name, entry_date, exit_date, source)
            VALUES
            ('000001', 'C1', DATE '2024-03-01', 'Hot', DATE '2024-03-01', NULL, 'unit'),
            ('000001', 'C2', DATE '2024-03-01', 'Cold', DATE '2024-03-01', NULL, 'unit'),
            ('000002', 'C2', DATE '2024-03-01', 'Cold', DATE '2024-03-01', NULL, 'unit')
            """
        )
        con.execute(
            """
            INSERT INTO a_share_fund_flow
            (symbol, trade_date, main_net_inflow_pct, super_large_net_inflow_pct, small_net_inflow_pct)
            VALUES
            ('000001', DATE '2024-03-20', 2.0, 0.0, 0.0),
            ('000002', DATE '2024-03-20', -1.0, 0.0, 0.0)
            """
        )
    finally:
        con.close()

    out = attach_concept_features(_dataset(), str(db_path)).sort_values(["signal_date", "symbol"]).reset_index(drop=True)

    before_snapshot = out[(out["signal_date"] == pd.Timestamp("2024-02-29")) & (out["symbol"] == "000001")].iloc[0]
    assert pd.isna(before_snapshot["feature_concept_member_count"])

    active = out[(out["signal_date"] == pd.Timestamp("2024-03-31")) & (out["symbol"] == "000001")].iloc[0]
    assert active["feature_concept_member_count"] == 2
    assert active["feature_hot_concept_membership"] == 1
    assert active["feature_concept_max_return_1m"] > active["feature_concept_ew_return_1m"]
    assert 0 <= active["feature_concept_inflow_breadth"] <= 1

    stale = out[(out["signal_date"] == pd.Timestamp("2024-06-30")) & (out["symbol"] == "000001")].iloc[0]
    assert pd.isna(stale["feature_concept_member_count"])


def test_concept_feature_spec_includes_m13b_membership_columns():
    specs = build_feature_specs(["concept"])
    concept_spec = specs[-1]
    assert "feature_concept_member_count_z" in concept_spec.feature_cols
    assert "feature_concept_max_return_1m_z" in concept_spec.feature_cols
    assert "feature_concept_inflow_breadth_z" in concept_spec.feature_cols
