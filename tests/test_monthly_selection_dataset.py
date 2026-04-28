from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_monthly_selection_dataset import (
    MonthlySelectionConfig,
    build_monthly_selection_dataset,
    build_research_config_id,
    select_month_end_signal_dates,
    summarize_candidate_width,
)


def _sample_daily() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2024-03-08")
    rows = []
    for sym, base, drift in [("000001", 10.0, 0.01), ("000002", 20.0, -0.005)]:
        for i, d in enumerate(dates):
            open_px = base * (1.0 + drift * i)
            close_px = open_px * 1.001
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": d,
                    "open": open_px,
                    "close": close_px,
                    "high": open_px * 1.01,
                    "low": open_px * 0.99,
                    "volume": 1_000_000.0,
                    "amount": 100_000_000.0,
                    "turnover": 1.0,
                    "pct_chg": drift * 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_select_month_end_signal_dates_uses_last_available_trade_date_per_month():
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-31",
            "2024-02-01",
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
        ]
    )

    out = select_month_end_signal_dates(dates, start="2024-01-01", end="2024-02-29")

    assert out == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_monthly_selection_dataset_aligns_tplus1_open_label_and_candidate_pools():
    daily = _sample_daily()
    industry = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "industry_level1": ["银行", "计算机"],
            "industry_level2": ["股份制银行", "软件开发"],
        }
    )
    cfg = MonthlySelectionConfig(min_history_days=5, min_amount_20d=1.0, limit_move_max=20)

    out = build_monthly_selection_dataset(
        daily,
        start_date="2024-01-01",
        end_date="2024-03-08",
        industry_map=industry,
        cfg=cfg,
    )

    assert set(out["candidate_pool_version"]) == {
        "U0_all_tradable",
        "U1_liquid_tradable",
        "U2_risk_sane",
    }
    jan_u1 = out[
        (out["signal_date"] == pd.Timestamp("2024-01-31"))
        & (out["symbol"] == "000001")
        & (out["candidate_pool_version"] == "U1_liquid_tradable")
    ].iloc[0]
    feb1_open = daily[(daily["symbol"] == "000001") & (daily["trade_date"] == pd.Timestamp("2024-02-01"))][
        "open"
    ].iloc[0]
    mar1_open = daily[(daily["symbol"] == "000001") & (daily["trade_date"] == pd.Timestamp("2024-03-01"))][
        "open"
    ].iloc[0]
    assert jan_u1["candidate_pool_pass"] is np.True_ or bool(jan_u1["candidate_pool_pass"])
    assert jan_u1["label_forward_1m_o2o_return"] == pytest.approx(mar1_open / feb1_open - 1.0)
    assert jan_u1["label_future_top_20pct"] == 1
    assert "feature_ret_20d_z" in out.columns
    assert "is_missing_feature_ret_20d" in out.columns

    width = summarize_candidate_width(out)
    width_jan = width[
        (width["signal_date"] == pd.Timestamp("2024-01-31"))
        & (width["candidate_pool_version"] == "U1_liquid_tradable")
    ].iloc[0]
    assert width_jan["raw_universe_width"] == 2
    assert width_jan["candidate_pool_width"] == 2


def test_build_research_config_id_records_core_dataset_contract():
    out = build_research_config_id(
        start_date="2021-01-01",
        end_date="2026-04-13",
        min_history_days=120,
        min_amount_20d=50_000_000.0,
        limit_move_max=3,
        daily_table="a_share_daily",
    )

    assert "rb_m_exec_tplus1_open_label_o2o" in out
    assert "hist_120" in out
    assert "amt20m_50" in out
