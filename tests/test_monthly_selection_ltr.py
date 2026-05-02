from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import scripts.run_monthly_selection_ltr as ltr
from scripts.run_monthly_selection_ltr import (
    M6RunConfig,
    build_leaderboard,
    build_ltr_relevance,
    build_m6_feature_spec,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
    build_walk_forward_ltr_scores,
)
from scripts.run_monthly_selection_multisource import attach_industry_breadth_features
from scripts.validate_research_contracts import validate_manifest


def _m6_sample(months: int = 5, symbols: int = 10) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    for m, date in enumerate(dates):
        market_ret = -0.01 + 0.006 * m
        for i in range(symbols):
            industry = "电子" if i < symbols // 2 else "计算机"
            signal = float(i) + (2.0 if industry == "计算机" else 0.0)
            forward = -0.03 + 0.010 * i + 0.003 * m
            rows.append(
                {
                    "signal_date": date,
                    "symbol": f"000{i + 1:03d}",
                    "candidate_pool_version": "U1_liquid_tradable",
                    "candidate_pool_pass": True,
                    "candidate_pool_reject_reason": "",
                    "label_forward_1m_o2o_return": forward,
                    "label_market_ew_o2o_return": market_ret,
                    "label_forward_1m_excess_vs_market": forward - market_ret,
                    "label_forward_1m_industry_neutral_excess": forward - (0.02 if industry == "计算机" else 0.0),
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


def test_ltr_relevance_is_monthly_cross_sectional_and_directional():
    df = _m6_sample(months=2, symbols=10)

    rel = build_ltr_relevance(df, grades=5)

    jan = df["signal_date"] == pd.Timestamp("2024-01-31")
    best_idx = df.loc[jan, "label_forward_1m_excess_vs_market"].idxmax()
    worst_idx = df.loc[jan, "label_forward_1m_excess_vs_market"].idxmin()
    assert rel.loc[best_idx] > rel.loc[worst_idx]
    assert rel.min() == 0
    assert rel.max() == 4


def test_m6_walk_forward_ltr_outputs_ranker_calibrated_and_ensemble_scores():
    pytest.importorskip("xgboost")
    df = attach_industry_breadth_features(_m6_sample(months=5, symbols=10))
    spec = build_m6_feature_spec(["industry_breadth"])
    cfg = M6RunConfig(
        top_ks=(2,),
        candidate_pools=("U1_liquid_tradable",),
        min_train_months=2,
        min_train_rows=10,
        max_fit_rows=0,
        random_seed=3,
        relevance_grades=5,
        ltr_models=("xgboost_rank_ndcg", "top20_calibrated", "ranker_top20_ensemble"),
    )

    scores, importance = build_walk_forward_ltr_scores(df, spec, cfg)

    assert not scores.empty
    assert scores["signal_date"].min() == pd.Timestamp("2024-03-31")
    assert {
        "M6_xgboost_rank_ndcg",
        "M6_top20_calibrated",
        "M6_ranker_top20_ensemble",
    }.issubset(set(scores["model"]))
    assert scores["model_type"].str.contains("ranker|classifier|ensemble", regex=True).all()
    assert not importance.empty

    rank_ic = build_rank_ic(scores)
    monthly, _ = build_monthly_long(scores, top_ks=[2], cost_bps=10.0)
    quantile = build_quantile_spread(scores, bucket_count=4)
    leaderboard = build_leaderboard(monthly, rank_ic, quantile, pd.DataFrame())

    top = leaderboard[
        (leaderboard["model"] == "M6_ranker_top20_ensemble")
        & (leaderboard["candidate_pool_version"] == "U1_liquid_tradable")
        & (leaderboard["top_k"] == 2)
    ].iloc[0]
    assert top["rank_ic_mean"] > 0.95
    assert top["topk_excess_mean"] > 0


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _m6_sample(months=4, symbols=10).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(ltr, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_selection_ltr.py",
            "--dataset",
            str(dataset_path),
            "--results-dir",
            "results",
            "--output-prefix",
            "m6_contract_test",
            "--top-k",
            "2",
            "--candidate-pools",
            "U1_liquid_tradable",
            "--min-train-months",
            "2",
            "--min-train-rows",
            "10",
            "--families",
            "industry_breadth",
            "--ltr-models",
            "top20_calibrated",
        ],
    )

    assert ltr.main() == 0

    manifests = sorted((tmp_path / "results").glob("m6_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m6_ltr"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"leaderboard_csv", "feature_importance_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
