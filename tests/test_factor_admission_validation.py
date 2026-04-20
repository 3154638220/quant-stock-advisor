from __future__ import annotations

import pandas as pd

from scripts.run_factor_admission_validation import build_admission_table, select_scenarios


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
    assert gross["admission_status"] == "pass"
    assert bool(ocf["pass_f1_gate"]) is False
    assert ocf["admission_status"] == "fail"


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
