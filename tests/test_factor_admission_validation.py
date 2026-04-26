from __future__ import annotations

import pandas as pd

from scripts.run_factor_admission_validation import (
    build_admission_output_stem,
    build_admission_research_config_id,
    build_admission_table,
    select_scenarios,
)


def test_build_admission_table_requires_both_ic_and_combo_gates():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "A0_baseline_s2",
                "candidate_factor": "vol_to_turnover",
                "family": "baseline",
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
                "scenario": "A2_gross_blend_10",
                "candidate_factor": "gross_margin_delta",
                "family": "gross",
                "is_baseline": False,
                "annualized_return": 0.05,
                "sharpe_ratio": 0.40,
                "max_drawdown": 0.18,
                "turnover_mean": 0.17,
                "rolling_oos_median_ann_return": 0.03,
                "slice_oos_median_ann_return": 0.02,
                "annualized_excess_vs_market": 0.01,
                "yearly_excess_median_vs_market": -0.01,
                "key_year_excess_mean_vs_market": -0.05,
                "key_year_excess_worst_vs_market": -0.10,
                "rolling_oos_median_ann_excess_vs_market": 0.01,
                "slice_oos_median_ann_excess_vs_market": 0.02,
            },
            {
                "scenario": "A5_ocf_asset_blend_10",
                "candidate_factor": "ocf_to_asset",
                "family": "ocf_asset",
                "is_baseline": False,
                "annualized_return": 0.04,
                "sharpe_ratio": 0.35,
                "max_drawdown": 0.19,
                "turnover_mean": 0.17,
                "rolling_oos_median_ann_return": 0.01,
                "slice_oos_median_ann_return": 0.00,
                "annualized_excess_vs_market": -0.02,
                "yearly_excess_median_vs_market": -0.06,
                "key_year_excess_mean_vs_market": -0.24,
                "key_year_excess_worst_vs_market": -0.40,
                "rolling_oos_median_ann_excess_vs_market": -0.03,
                "slice_oos_median_ann_excess_vs_market": -0.01,
            },
        ]
    )
    gate_df = pd.DataFrame(
        [
            {"factor": "gross_margin_delta", "horizon_key": "close_21d", "ic_mean": 0.012, "ic_t_value": 2.5},
            {"factor": "gross_margin_delta", "horizon_key": "tplus1_open_1d", "ic_mean": 0.003, "ic_t_value": 1.0},
            {"factor": "ocf_to_asset", "horizon_key": "close_21d", "ic_mean": 0.009, "ic_t_value": 5.0},
            {"factor": "ocf_to_asset", "horizon_key": "tplus1_open_1d", "ic_mean": 0.002, "ic_t_value": 1.0},
        ]
    )

    out = build_admission_table(summary_df, gate_df, baseline_label="A0_baseline_s2", f1_min_ic=0.01, f1_min_t=2.0)

    gross = out.loc[out["scenario"] == "A2_gross_blend_10"].iloc[0]
    ocf = out.loc[out["scenario"] == "A5_ocf_asset_blend_10"].iloc[0]

    assert bool(gross["pass_f1_gate"]) is True
    assert bool(gross["pass_combo_gate"]) is True
    assert bool(gross["pass_benchmark_gate"]) is True
    assert gross["admission_status"] == "pass"
    assert bool(ocf["pass_f1_gate"]) is False
    assert ocf["admission_status"] == "fail"


def test_build_admission_table_blocks_combo_winner_if_benchmark_gate_fails():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "A0_baseline_s2",
                "candidate_factor": "vol_to_turnover",
                "family": "baseline",
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
                "scenario": "A8_net_margin_blend_10",
                "candidate_factor": "net_margin_stability",
                "family": "net_margin",
                "is_baseline": False,
                "annualized_return": 0.05,
                "sharpe_ratio": 0.35,
                "max_drawdown": 0.18,
                "turnover_mean": 0.16,
                "rolling_oos_median_ann_return": 0.03,
                "slice_oos_median_ann_return": 0.02,
                "annualized_excess_vs_market": -0.005,
                "yearly_excess_median_vs_market": -0.03,
                "key_year_excess_mean_vs_market": -0.18,
                "key_year_excess_worst_vs_market": -0.32,
                "rolling_oos_median_ann_excess_vs_market": -0.01,
                "slice_oos_median_ann_excess_vs_market": 0.01,
            },
        ]
    )
    gate_df = pd.DataFrame(
        [
            {"factor": "net_margin_stability", "horizon_key": "close_21d", "ic_mean": 0.013, "ic_t_value": 2.6},
            {"factor": "net_margin_stability", "horizon_key": "tplus1_open_1d", "ic_mean": 0.003, "ic_t_value": 0.9},
        ]
    )

    out = build_admission_table(summary_df, gate_df, baseline_label="A0_baseline_s2", f1_min_ic=0.01, f1_min_t=2.0)
    row = out.loc[out["scenario"] == "A8_net_margin_blend_10"].iloc[0]

    assert bool(row["pass_f1_gate"]) is True
    assert bool(row["pass_combo_gate"]) is True
    assert bool(row["pass_benchmark_gate"]) is False
    assert row["admission_status"] == "fail"


def test_select_scenarios_can_focus_on_next_candidate_families():
    scenarios = select_scenarios(["net_margin", "asset_turnover"])
    labels = [scenario.label for scenario in scenarios]
    candidate_factors = {scenario.candidate_factor for scenario in scenarios if not scenario.is_baseline}

    assert labels[0] == "A0_baseline_s2"
    assert set(labels[1:]) == {
        "A7_net_margin_single",
        "A8_net_margin_blend_10",
        "A9_net_margin_blend_20",
        "A10_asset_turnover_single",
        "A11_asset_turnover_blend_10",
        "A12_asset_turnover_blend_20",
    }
    assert candidate_factors == {"net_margin_stability", "asset_turnover"}


def test_build_admission_table_prefers_full_backtest_excess_when_present():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "A0_baseline_s2",
                "candidate_factor": "vol_to_turnover",
                "family": "baseline",
                "is_baseline": True,
                "annualized_return": 0.03,
                "sharpe_ratio": 0.30,
                "max_drawdown": 0.20,
                "turnover_mean": 0.18,
                "rolling_oos_median_ann_return": 0.02,
                "slice_oos_median_ann_return": 0.01,
                "annualized_excess_vs_market": 0.02,
                "full_backtest_annualized_excess_vs_market": -0.01,
                "yearly_excess_median_vs_market": -0.05,
                "key_year_excess_mean_vs_market": -0.20,
                "key_year_excess_worst_vs_market": -0.35,
                "rolling_oos_median_ann_excess_vs_market": -0.02,
                "slice_oos_median_ann_excess_vs_market": -0.01,
            },
            {
                "scenario": "A2_gross_blend_10",
                "candidate_factor": "gross_margin_delta",
                "family": "gross",
                "is_baseline": False,
                "annualized_return": 0.05,
                "sharpe_ratio": 0.40,
                "max_drawdown": 0.18,
                "turnover_mean": 0.17,
                "rolling_oos_median_ann_return": 0.03,
                "slice_oos_median_ann_return": 0.02,
                "annualized_excess_vs_market": 0.05,
                "full_backtest_annualized_excess_vs_market": 0.03,
                "yearly_excess_median_vs_market": -0.01,
                "key_year_excess_mean_vs_market": -0.05,
                "key_year_excess_worst_vs_market": -0.10,
                "rolling_oos_median_ann_excess_vs_market": 0.01,
                "slice_oos_median_ann_excess_vs_market": 0.02,
            },
        ]
    )
    gate_df = pd.DataFrame(
        [
            {"factor": "gross_margin_delta", "horizon_key": "close_21d", "ic_mean": 0.012, "ic_t_value": 2.5},
            {"factor": "gross_margin_delta", "horizon_key": "tplus1_open_1d", "ic_mean": 0.003, "ic_t_value": 1.0},
        ]
    )

    out = build_admission_table(summary_df, gate_df, baseline_label="A0_baseline_s2", f1_min_ic=0.01, f1_min_t=2.0)
    gross = out.loc[out["scenario"] == "A2_gross_blend_10"].iloc[0]

    assert gross["benchmark_excess_metric"] == "full_backtest_annualized_excess_vs_market"
    assert abs(float(gross["delta_ann_excess_vs_baseline"]) - 0.04) < 1e-12


def test_build_admission_table_sets_canonical_result_type():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "A0_baseline_s2",
                "candidate_factor": "vol_to_turnover",
                "family": "baseline",
                "is_baseline": True,
                "research_topic": "factor_admission_validation",
                "research_config_id": "families_all_yrs_2021-2025-2026_ic100_t200",
                "output_stem": "factor_admission_validation_2026_04_19_families_all_yrs_2021-2025-2026_ic100_t200",
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
            }
        ]
    )
    out = build_admission_table(
        summary_df,
        pd.DataFrame(columns=["factor", "horizon_key", "ic_mean", "ic_t_value"]),
        baseline_label="A0_baseline_s2",
        f1_min_ic=0.01,
        f1_min_t=2.0,
    )
    row = out.iloc[0]
    assert row["result_type"] == "factor_admission"
    assert row["research_topic"] == "factor_admission_validation"
    assert row["research_config_id"] == "families_all_yrs_2021-2025-2026_ic100_t200"


def test_build_admission_research_config_id_and_output_stem_are_stable():
    config_id = build_admission_research_config_id(
        selected_families=["net_margin", "asset_turnover"],
        benchmark_key_years=[2021, 2025, 2026],
        f1_min_ic=0.01,
        f1_min_t=2.0,
    )
    assert config_id == "families_net_margin-asset_turnover_yrs_2021-2025-2026_ic100_t200"
    stem = build_admission_output_stem(
        output_prefix="factor_admission_validation_2026-04-19",
        research_config_id=config_id,
    )
    assert (
        stem
        == "factor_admission_validation_2026_04_19_families_net_margin-asset_turnover_yrs_2021-2025-2026_ic100_t200"
    )
