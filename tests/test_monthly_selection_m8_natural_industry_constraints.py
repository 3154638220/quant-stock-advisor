from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import scripts.run_monthly_selection_m8_natural_industry_constraints as natural
from scripts.run_monthly_selection_m8_natural_industry_constraints import (
    build_gate_table,
    copy_source_metric_for_optimizer,
    select_soft_industry_risk,
)
from src.research.contracts import validate_manifest


def _natural_sample(months: int = 4, symbols: int = 10) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    for m, date in enumerate(dates):
        market = -0.015 + 0.008 * m
        for i in range(symbols):
            industry = "电子" if i < symbols // 2 else "计算机"
            signal = float(i) + (1.5 if industry == "计算机" else 0.0)
            forward = -0.035 + 0.010 * i + 0.003 * m
            rows.append(
                {
                    "signal_date": date,
                    "symbol": f"000{i + 1:03d}",
                    "candidate_pool_version": "U2_risk_sane",
                    "candidate_pool_pass": True,
                    "candidate_pool_reject_reason": "",
                    "label_forward_1m_o2o_return": forward,
                    "label_market_ew_o2o_return": market,
                    "label_forward_1m_excess_vs_market": forward - market,
                    "label_forward_1m_industry_neutral_excess": forward - (0.015 if industry == "计算机" else 0.0),
                    "label_future_top_20pct": 1 if i >= symbols - 2 else 0,
                    "feature_ret_20d": signal,
                    "feature_ret_60d": signal / 2.0,
                    "feature_realized_vol_20d": 1.0 / (1.0 + signal),
                    "feature_amount_20d_log": 10.0 + signal,
                    "feature_ret_20d_z": signal,
                    "feature_ret_60d_z": signal / 2.0,
                    "feature_ret_5d_z": signal / 3.0,
                    "feature_realized_vol_20d_z": -signal,
                    "feature_amount_20d_log_z": signal / 4.0,
                    "feature_turnover_20d_z": signal / 5.0,
                    "feature_price_position_250d_z": signal / 6.0,
                    "feature_limit_move_hits_20d_z": -signal,
                    "industry_level1": industry,
                    "industry_level2": "A",
                    "log_market_cap": 10.0 + i,
                    "risk_flags": "",
                    "is_buyable_tplus1_open": True,
                }
            )
    return pd.DataFrame(rows)


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


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _natural_sample(months=4, symbols=10).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(natural, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_selection_m8_natural_industry_constraints.py",
            "--dataset",
            str(dataset_path),
            "--results-dir",
            "results",
            "--output-prefix",
            "m8_natural_contract_test",
            "--as-of-date",
            "2026-05-02",
            "--top-k",
            "2",
            "--candidate-pools",
            "U2_risk_sane",
            "--min-train-months",
            "2",
            "--min-train-rows",
            "10",
            "--families",
            "industry_breadth",
            "--soft-gamma",
            "0.2",
        ],
    )

    assert natural.main() == 0

    manifests = sorted((tmp_path / "results").glob("m8_natural_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m8_natural_industry_constraints"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"leaderboard_csv", "gate_csv", "optimizer_compare_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
