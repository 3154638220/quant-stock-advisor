from __future__ import annotations

import sys

import pandas as pd

from scripts.run_alpha_factor_scout import (
    classify_factor_family,
    find_missing_included_factors,
    infer_candidate_factors,
    parse_args,
)
from scripts.research_identity import (
    build_full_backtest_research_identity,
    build_light_research_identity,
    canonical_research_config,
)
from scripts.run_factor_admission_validation import build_admission_table


def test_infer_candidate_factors_excludes_identity_and_utility_columns():
    n = 30
    factor_df = pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(n)],
            "trade_date": pd.to_datetime(["2026-01-31"] * n),
            "announcement_date": pd.to_datetime(["2026-01-15"] * n),
            "turnover_roll_mean": range(n),
            "price_position": range(n),
            "limit_move_hits_5d": range(n),
            "log_market_cap": range(n),
            "vol_to_turnover": [float(i) for i in range(n)],
            "momentum": [float(i) * 0.1 for i in range(n)],
            "_universe_eligible": [True] * n,
            "note": ["x"] * n,
        }
    )

    out = infer_candidate_factors(factor_df, baseline_factor="vol_to_turnover")

    assert out == ["vol_to_turnover", "momentum"]


def test_infer_candidate_factors_respects_include_list_ordering_baseline_first():
    factor_df = pd.DataFrame(
        {
            "symbol": ["000001"] * 30,
            "trade_date": pd.to_datetime(["2026-01-31"] * 30),
            "vol_to_turnover": range(30),
            "momentum": range(30),
            "rsi": range(30),
        }
    )

    out = infer_candidate_factors(
        factor_df,
        baseline_factor="vol_to_turnover",
        include=["rsi", "momentum"],
    )

    assert out == ["vol_to_turnover", "momentum", "rsi"]


def test_infer_candidate_factors_keeps_new_data_source_columns():
    n = 30
    factor_df = pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(n)],
            "trade_date": pd.to_datetime(["2026-01-31"] * n),
            "vol_to_turnover": [float(i) for i in range(n)],
            "main_inflow_z_5d": [float(i) * 0.1 for i in range(n)],
            "holder_change_rate_z": [float(i) * -0.05 for i in range(n)],
            "holder_concentration_proxy": [float(i) * 0.03 for i in range(n)],
        }
    )

    out = infer_candidate_factors(factor_df, baseline_factor="vol_to_turnover")

    assert "main_inflow_z_5d" in out
    assert "holder_change_rate_z" in out
    assert "holder_concentration_proxy" in out


def test_classify_factor_family_marks_new_data_sources():
    assert classify_factor_family("vol_to_turnover", baseline_factor="vol_to_turnover") == "baseline"
    assert classify_factor_family("main_inflow_z_10d", baseline_factor="vol_to_turnover") == "fund_flow"
    assert classify_factor_family("proxy_main_inflow_pct_5d", baseline_factor="vol_to_turnover") == "fund_flow"
    assert classify_factor_family("holder_change_rate_z", baseline_factor="vol_to_turnover") == "shareholder"
    assert classify_factor_family("ocf_to_asset", baseline_factor="vol_to_turnover") == "fundamental"
    assert classify_factor_family("momentum", baseline_factor="vol_to_turnover") == "price_volume"


def test_find_missing_included_factors_reports_unavailable_requested_columns():
    factor_df = pd.DataFrame(
        {
            "symbol": ["000001"] * 30,
            "trade_date": pd.to_datetime(["2026-01-31"] * 30),
            "vol_to_turnover": range(30),
            "proxy_main_inflow_pct_5d": range(30),
        }
    )

    out = find_missing_included_factors(
        factor_df,
        baseline_factor="vol_to_turnover",
        include=["proxy_main_inflow_pct_5d", "holder_change_rate_z"],
    )

    assert out == ["holder_change_rate_z"]


def test_alpha_scout_table_reuses_benchmark_first_gate():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "single_vol_to_turnover",
                "candidate_factor": "vol_to_turnover",
                "family": "single_factor",
                "is_baseline": True,
                "annualized_return": 0.03,
                "sharpe_ratio": 0.30,
                "max_drawdown": 0.20,
                "turnover_mean": 0.18,
                "rolling_oos_median_ann_return": 0.02,
                "slice_oos_median_ann_return": 0.01,
                "annualized_excess_vs_market": -0.01,
                "yearly_excess_median_vs_market": -0.05,
                "key_year_excess_mean_vs_market": -0.20,
                "key_year_excess_worst_vs_market": -0.35,
                "rolling_oos_median_ann_excess_vs_market": -0.02,
                "slice_oos_median_ann_excess_vs_market": -0.01,
            },
            {
                "scenario": "single_momentum",
                "candidate_factor": "momentum",
                "family": "single_factor",
                "is_baseline": False,
                "annualized_return": 0.06,
                "sharpe_ratio": 0.45,
                "max_drawdown": 0.17,
                "turnover_mean": 0.20,
                "rolling_oos_median_ann_return": 0.05,
                "slice_oos_median_ann_return": 0.03,
                "annualized_excess_vs_market": 0.02,
                "yearly_excess_median_vs_market": -0.01,
                "key_year_excess_mean_vs_market": -0.10,
                "key_year_excess_worst_vs_market": -0.15,
                "rolling_oos_median_ann_excess_vs_market": 0.01,
                "slice_oos_median_ann_excess_vs_market": 0.02,
            },
        ]
    )
    gate_df = pd.DataFrame(
        [
            {"factor": "momentum", "horizon_key": "close_21d", "ic_mean": 0.02, "ic_t_value": 3.0},
            {"factor": "momentum", "horizon_key": "tplus1_open_1d", "ic_mean": 0.003, "ic_t_value": 1.0},
        ]
    )

    out = build_admission_table(
        summary_df,
        gate_df,
        baseline_label="single_vol_to_turnover",
        f1_min_ic=0.01,
        f1_min_t=2.0,
    ).rename(columns={"admission_status": "scout_status"})
    row = out.loc[out["scenario"] == "single_momentum"].iloc[0]

    assert bool(row["pass_f1_gate"]) is True
    assert bool(row["pass_combo_gate"]) is True
    assert bool(row["pass_benchmark_gate"]) is True
    assert row["scout_status"] == "pass"


def test_alpha_factor_scout_parse_args_accepts_prepared_cache(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_alpha_factor_scout.py",
            "--prepared-factors-cache",
            "data/cache/prepared_alpha_factors.parquet",
            "--refresh-prepared-factors-cache",
        ],
    )

    args = parse_args()

    assert args.prepared_factors_cache == "data/cache/prepared_alpha_factors.parquet"
    assert args.refresh_prepared_factors_cache is True


def test_alpha_factor_scout_summary_can_carry_light_proxy_fields():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "single_vol_to_turnover",
                "candidate_factor": "vol_to_turnover",
                "family": "single_factor",
                "factor_family": "baseline",
                "is_baseline": True,
                "result_type": "light_strategy_proxy",
                "research_topic": "alpha_factor_scout",
                "research_config_id": "base_vol_to_turnover_rb_m_top20_yrs_2021-2025-2026_pool_auto",
                "output_stem": "alpha_factor_scout_2026_04_20_base_vol_to_turnover_rb_m_top20_yrs_2021-2025-2026_pool_auto",
                "rebalance_rule": "M",
                "periods_per_year": 12.0,
                "proxy_periods": 12,
                "proxy_annualized_return": 0.08,
                "proxy_benchmark_annualized_return": 0.03,
                "annualized_return": 0.07,
                "annualized_excess_vs_market": 0.05,
                "full_backtest_annualized_excess_vs_market": 0.04,
                "sharpe_ratio": 0.30,
                "max_drawdown": 0.20,
                "turnover_mean": 0.18,
                "rolling_oos_median_ann_return": 0.02,
                "slice_oos_median_ann_return": 0.01,
                "yearly_excess_median_vs_market": -0.05,
                "key_year_excess_mean_vs_market": -0.20,
                "key_year_excess_worst_vs_market": -0.35,
                "rolling_oos_median_ann_excess_vs_market": -0.02,
                "slice_oos_median_ann_excess_vs_market": -0.01,
            }
        ]
    )

    assert summary_df.loc[0, "result_type"] == "light_strategy_proxy"
    assert summary_df.loc[0, "research_topic"] == "alpha_factor_scout"
    assert float(summary_df.loc[0, "periods_per_year"]) == 12.0
    assert float(summary_df.loc[0, "annualized_excess_vs_market"]) == 0.05


def test_build_light_research_identity_is_stable_for_scout_outputs():
    out = build_light_research_identity(
        topic="alpha_factor_scout",
        output_prefix="alpha_factor_scout_2026-04-20",
        baseline_factor="vol_to_turnover",
        rebalance_rule="M",
        top_k=20,
        benchmark_key_years=[2021, 2025, 2026],
        selector_parts={"pool": "auto", "count": 6, "factors": ["momentum", "bias_long"]},
    )
    assert out["research_topic"] == "alpha_factor_scout"
    assert out["research_config_id"].startswith("base_vol_to_turnover_rb_m_top20_yrs_2021-2025-2026_")
    assert out["output_stem"].startswith("alpha_factor_scout_2026_04_20_base_vol_to_turnover_rb_m_top20_yrs_2021-2025-2026_")


def test_full_backtest_research_identity_and_canonical_snapshot_are_stable():
    out = build_full_backtest_research_identity(
        sort_by="xgboost",
        rebalance_rule="M",
        top_k=20,
        max_turnover=0.3,
        portfolio_method="equal_weight",
        execution_mode="tplus1_open",
        prefilter_enabled=False,
        universe_filter_enabled=True,
        benchmark_symbol="market_ew_proxy",
        start_date="2021-01-01",
        end_date="2026-04-24",
        selector_parts={"tree_group": "G1"},
    )
    assert out["result_type"] == "full_backtest"
    assert out["research_topic"] == "full_backtest"
    assert "score_xgboost" in out["research_config_id"]
    assert "tree_group_g1" in out["research_config_id"]
    assert out["output_stem"].startswith("full_backtest_score_xgboost_")

    snap = canonical_research_config("p1_tree_full_backtest")
    assert snap["result_type"] == "full_backtest"
    assert snap["score"] == "xgboost"
    assert snap["max_turnover"] == 1.0
    snap["score"] = "changed"
    assert canonical_research_config("p1_tree_full_backtest")["score"] == "xgboost"
