from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import scripts.run_monthly_selection_baselines as baselines
from src.pipeline.monthly_baselines import (
    BaselineRunConfig,
    build_leaderboard,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
    build_static_scores,
    build_walk_forward_scores,
    summarize_industry_exposure,
    summarize_regime_slice,
    valid_pool_frame,
)
from src.research.contracts import validate_manifest


def _baseline_sample(months: int = 5, symbols: int = 8) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    for m, date in enumerate(dates):
        market_ret = -0.01 + 0.01 * m
        for i in range(symbols):
            signal = float(i)
            forward = -0.04 + 0.015 * i + 0.002 * m
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
                    "label_forward_1m_industry_neutral_excess": forward - (-0.01 if i % 2 else 0.0),
                    "label_future_top_20pct": 1 if i >= symbols - 2 else 0,
                    "feature_ret_20d_z": signal,
                    "feature_ret_60d_z": signal,
                    "feature_ret_5d_z": signal / 2.0,
                    "feature_realized_vol_20d_z": -signal,
                    "feature_amount_20d_log_z": signal / 3.0,
                    "feature_turnover_20d_z": signal / 4.0,
                    "feature_price_position_250d_z": signal / 5.0,
                    "feature_limit_move_hits_20d_z": -signal,
                    "industry_level1": "银行" if i % 2 == 0 else "计算机",
                    "industry_level2": "A",
                    "log_market_cap": 10.0 + i,
                    "risk_flags": "",
                    "is_buyable_tplus1_open": True,
                }
            )
    return pd.DataFrame(rows)


def test_static_baseline_reports_rank_ic_topk_nextk_and_quantile_spread():
    df = _baseline_sample()

    scores = build_static_scores(df)
    rank_ic = build_rank_ic(scores)
    monthly, holdings = build_monthly_long(scores, top_ks=[2], cost_bps=10.0)
    quantile = build_quantile_spread(scores, bucket_count=4)
    industry = summarize_industry_exposure(holdings)
    regime = summarize_regime_slice(monthly, pd.DataFrame(
        {
            "signal_date": sorted(df["signal_date"].unique()),
            "realized_market_state": ["strong_down", "neutral", "neutral", "neutral", "strong_up"],
        }
    ))
    leaderboard = build_leaderboard(monthly, rank_ic, quantile, regime)

    mom = leaderboard[
        (leaderboard["model"] == "B2_momentum_20d")
        & (leaderboard["candidate_pool_version"] == "U1_liquid_tradable")
        & (leaderboard["top_k"] == 2)
    ].iloc[0]
    assert mom["rank_ic_mean"] == pytest.approx(1.0)
    assert mom["topk_excess_mean"] > 0
    assert mom["topk_minus_nextk_mean"] > 0
    assert mom["quantile_top_minus_bottom_mean"] > 0
    assert not industry.empty


def test_walk_forward_scores_only_start_after_required_train_window():
    df = _baseline_sample(months=4, symbols=10)
    cfg = BaselineRunConfig(
        top_ks=(2,),
        candidate_pools=("U1_liquid_tradable",),
        min_train_months=2,
        min_train_rows=10,
        include_xgboost=False,
        random_seed=7,
    )

    scores, importance = build_walk_forward_scores(df, cfg)

    assert not scores.empty
    assert scores["signal_date"].min() == pd.Timestamp("2024-03-31")
    assert {"M4_elasticnet_excess", "M4_logistic_top20"}.issubset(set(scores["model"]))
    assert not importance.empty
    assert len(valid_pool_frame(df)) == len(df)


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _baseline_sample(months=4, symbols=10).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(baselines, "ROOT", tmp_path)
    from src.pipeline import cli_helpers
    monkeypatch.setattr(cli_helpers, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_selection_baselines.py",
            "--dataset",
            str(dataset_path),
            "--results-dir",
            "results",
            "--output-prefix",
            "baseline_contract_test",
            "--top-k",
            "2",
            "--candidate-pools",
            "U1_liquid_tradable",
            "--min-train-months",
            "2",
            "--min-train-rows",
            "10",
            "--skip-xgboost",
        ],
    )

    assert baselines.main() == 0

    manifests = sorted((tmp_path / "results").glob("baseline_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_baselines"
    assert {x["name"] for x in payload["artifacts"]} >= {"leaderboard_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
