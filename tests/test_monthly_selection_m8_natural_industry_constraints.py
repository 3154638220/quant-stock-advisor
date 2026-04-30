import numpy as np
import pandas as pd

from scripts.run_monthly_selection_m8_natural_industry_constraints import (
    build_gate_table,
    copy_source_metric_for_optimizer,
    select_soft_industry_risk,
)


def test_select_soft_industry_risk_diversifies_without_hard_cap():
    rows = [
        {"symbol": "a1", "industry_level1": "A", "score": 1.00},
        {"symbol": "a2", "industry_level1": "A", "score": 0.99},
        {"symbol": "a3", "industry_level1": "A", "score": 0.98},
        {"symbol": "b1", "industry_level1": "B", "score": 0.70},
        {"symbol": "c1", "industry_level1": "C", "score": 0.69},
        {"symbol": "d1", "industry_level1": "D", "score": 0.68},
    ]
    selected = select_soft_industry_risk(pd.DataFrame(rows), k=4, gamma=0.8)

    assert selected["symbol"].tolist() == ["a1", "a2", "b1", "c1"]
    assert selected["industry_level1"].value_counts().loc["A"] == 2


def test_copy_source_metric_for_optimizer_maps_source_model():
    metric = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-31"]),
            "candidate_pool_version": ["U1_liquid_tradable"],
            "model": ["source_model"],
            "rank_ic": [0.12],
        }
    )
    optimizer_monthly = pd.DataFrame(
        {
            "candidate_pool_version": ["U1_liquid_tradable"],
            "model": ["source_model__soft_risk_budget_gamma0_2"],
            "source_model": ["source_model"],
        }
    )

    copied = copy_source_metric_for_optimizer(metric, optimizer_monthly)

    assert copied["model"].tolist() == ["source_model__soft_risk_budget_gamma0_2"]
    assert copied["rank_ic"].tolist() == [0.12]


def test_build_gate_table_accepts_natural_candidate_near_hardcap():
    leaderboard = pd.DataFrame(
        {
            "candidate_pool_version": ["U1_liquid_tradable"],
            "model": ["m__soft_risk_budget_gamma0_2"],
            "model_type": ["soft_industry_risk_budget_optimizer"],
            "selection_policy": ["soft_industry_risk_budget"],
            "score_family": ["optimizer_compare"],
            "label_variant": ["market_excess"],
            "soft_gamma": [0.2],
            "top_k": [20],
            "topk_excess_after_cost_mean": [0.0184],
            "rank_ic_mean": [0.10],
            "topk_minus_nextk_mean": [0.01],
            "max_industry_share_mean": [0.10],
            "concentration_pass_rate": [1.0],
            "industry_count_mean": [16.0],
            "strong_down_median_excess": [-0.01],
            "strong_up_median_excess": [0.02],
        }
    )
    hardcap = pd.DataFrame(
        {
            "candidate_pool_version": ["U1_liquid_tradable"],
            "top_k": [20],
            "hardcap_after_cost_baseline": [0.0223],
            "hardcap_rank_ic_baseline": [0.09],
            "hardcap_source_model": ["hardcap"],
        }
    )

    gate = build_gate_table(leaderboard, hardcap, tolerance=0.005)

    assert bool(gate.loc[0, "m8_natural_gate_pass"])
    assert np.isclose(gate.loc[0, "hardcap_delta_after_cost"], -0.0039)
