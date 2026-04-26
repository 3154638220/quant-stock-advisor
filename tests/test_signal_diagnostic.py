from __future__ import annotations

import sys

import pandas as pd

from scripts.light_strategy_proxy import summarize_signal_diagnostic
from scripts.run_signal_diagnostic import build_signal_diagnostic_research_identity, parse_args


def test_summarize_signal_diagnostic_computes_frequency_aware_metrics():
    period_df = pd.DataFrame(
        [
            {
                "period": "2026-01",
                "strategy_return": 0.10,
                "benchmark_return": 0.02,
                "excess_return": 0.08,
                "benchmark_up": True,
                "strategy_up": True,
                "beat_benchmark": True,
            },
            {
                "period": "2026-02",
                "strategy_return": -0.05,
                "benchmark_return": -0.01,
                "excess_return": -0.04,
                "benchmark_up": False,
                "strategy_up": False,
                "beat_benchmark": False,
            },
            {
                "period": "2026-03",
                "strategy_return": 0.03,
                "benchmark_return": 0.01,
                "excess_return": 0.02,
                "benchmark_up": True,
                "strategy_up": True,
                "beat_benchmark": True,
            },
        ]
    )

    out = summarize_signal_diagnostic(period_df, periods_per_year=12.0)

    assert out["n_periods"] == 3
    assert out["period_win_rate"] == 2 / 3
    assert out["period_beat_rate"] == 2 / 3
    assert out["benchmark_up_rate"] == 2 / 3
    assert out["strategy_annualized_return"] > out["benchmark_annualized_return"]
    assert out["annualized_excess_vs_market"] > 0.0
    assert out["strategy_annualized_vol"] > 0.0
    assert out["strategy_max_drawdown"] >= 0.0


def test_summarize_signal_diagnostic_empty_input_returns_nan_payload():
    out = summarize_signal_diagnostic(pd.DataFrame(), periods_per_year=52.0)

    assert out["n_periods"] == 0
    assert pd.isna(out["strategy_annualized_return"])
    assert pd.isna(out["period_beat_rate"])


def test_run_signal_diagnostic_parse_args_accepts_rebalance_and_prepared_cache(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_signal_diagnostic.py",
            "--rebalance-rule",
            "W",
            "--prepared-factors-cache",
            "data/cache/prepared_signal_diag.parquet",
            "--refresh-prepared-factors-cache",
        ],
    )

    args = parse_args()

    assert args.rebalance_rule == "W"
    assert args.prepared_factors_cache == "data/cache/prepared_signal_diag.parquet"
    assert args.refresh_prepared_factors_cache is True


def test_signal_diagnostic_summary_rows_can_carry_canonical_result_type():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "single_vol_to_turnover",
                "signal_name": "vol_to_turnover",
                "factor_family": "baseline",
                "is_baseline": True,
                "result_type": "signal_diagnostic",
                "research_topic": "signal_diagnostic",
                "research_config_id": "base_vol_to_turnover_rb_m_top20_yrs_2021-2026_pool_auto_count_12_include_none_exclude_none",
                "output_stem": "signal_diagnostic_base_vol_to_turnover_rb_m_top20_yrs_2021-2026_pool_auto_count_12_include_none_exclude_none",
                "rebalance_rule": "M",
                "periods_per_year": 12.0,
                "diagnostic_periods": 12,
                "strategy_annualized_return": 0.11,
                "benchmark_annualized_return": 0.05,
                "annualized_excess_vs_market": 0.06,
                "strategy_annualized_vol": 0.18,
                "strategy_sharpe_ratio": 0.61,
                "strategy_max_drawdown": 0.14,
                "period_beat_rate": 0.58,
            }
        ]
    )

    assert summary_df.loc[0, "result_type"] == "signal_diagnostic"
    assert summary_df.loc[0, "research_topic"] == "signal_diagnostic"
    assert str(summary_df.loc[0, "research_config_id"]).startswith("base_vol_to_turnover_rb_m_top20_yrs_2021-2026_")
    assert str(summary_df.loc[0, "output_stem"]).startswith("signal_diagnostic_base_vol_to_turnover_rb_m_top20_yrs_2021-2026_")
    assert float(summary_df.loc[0, "periods_per_year"]) == 12.0
    assert float(summary_df.loc[0, "annualized_excess_vs_market"]) == 0.06


def test_build_signal_diagnostic_research_identity_is_stable_and_readable():
    out = build_signal_diagnostic_research_identity(
        output_prefix="Signal Diagnostic",
        baseline_factor="vol_to_turnover",
        rebalance_rule="2W",
        top_k=15,
        start_date="2021-01-01",
        end_date="2023-12-31",
        include=["weekly_kdj_j", "main_inflow_z_5d"],
        exclude=["holder_count_change_pct"],
        candidate_count=7,
    )

    assert out["research_topic"] == "signal_diagnostic"
    assert out["research_config_id"].startswith("base_vol_to_turnover_rb_2w_top15_yrs_2021-2022-2023_")
    assert "pool_explicit" in out["research_config_id"]
    assert "count_7" in out["research_config_id"]
    assert out["output_stem"].startswith("signal_diagnostic_base_vol_to_turnover_rb_2w_top15_yrs_2021-2022-2023_")
