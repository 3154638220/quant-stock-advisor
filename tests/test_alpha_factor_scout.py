from __future__ import annotations

import sys

import pandas as pd

from scripts.run_alpha_factor_scout import infer_candidate_factors, parse_args
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
