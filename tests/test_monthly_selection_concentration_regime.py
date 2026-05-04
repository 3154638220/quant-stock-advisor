from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from src.cli.monthly_concentration import run_monthly_concentration_regime
from src.pipeline.monthly_baselines import (
    build_quantile_spread,
    build_rank_ic,
    summarize_regime_slice,
)
from src.pipeline.monthly_concentration import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    M8_POLICY_MODEL,
    MARKET_COL,
    STABLE_M5_ELASTICNET,
    STABLE_M5_EXTRATREES,
    WATCHLIST_M6_RANK,
    WATCHLIST_M6_TOP20,
    build_constrained_leaderboard,
    build_constrained_monthly,
    build_default_cap_grid,
    build_lagged_state_frame,
    build_regime_policy_scores,
    default_cap_values_for_topk,
    parse_cap_grid,
    resolve_topk_and_cap_grid,
    select_with_industry_cap,
    serialize_cap_grid,
    summarize_industry_concentration,
)
from src.research.contracts import validate_manifest


def _score_sample(months: int = 4) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    models = [
        (STABLE_M5_ELASTICNET, "elasticnet", 0.00),
        (STABLE_M5_EXTRATREES, "tree_sanity", 0.01),
        (WATCHLIST_M6_RANK, "xgboost_ranker", 0.02),
        (WATCHLIST_M6_TOP20, "top_bucket_classifier", 0.03),
    ]
    for m, date in enumerate(dates):
        market = -0.02 + 0.02 * m
        for model, model_type, bump in models:
            for i in range(8):
                industry = "非银金融" if i < 5 else ("计算机" if i < 7 else "医药生物")
                forward = -0.03 + 0.012 * i + 0.002 * m
                score = 1.0 - i * 0.03 + bump
                rows.append(
                    {
                        "signal_date": date,
                        "candidate_pool_version": "U2_risk_sane",
                        "candidate_pool_pass": True,
                        "candidate_pool_reject_reason": "",
                        "symbol": f"000{i + 1:03d}",
                        "model": model,
                        "model_type": model_type,
                        "score": score,
                        "rank": i + 1,
                        LABEL_COL: forward,
                        MARKET_COL: market,
                        EXCESS_COL: forward - market,
                        INDUSTRY_EXCESS_COL: forward - (0.01 if industry == "非银金融" else 0.0),
                        "label_future_top_20pct": 1 if i >= 6 else 0,
                        "industry_level1": industry,
                        "industry_level2": "L2",
                        "risk_flags": "",
                        "is_buyable_tplus1_open": True,
                        "feature_ret_20d": 1.0 if i >= 4 else -1.0,
                        "feature_ret_60d": score,
                        "feature_realized_vol_20d": 1.0 / (1.0 + i),
                        "feature_amount_20d_log": 10.0 + i,
                        "feature_ret_20d_z": score,
                        "feature_ret_60d_z": score / 2.0,
                        "feature_ret_5d_z": score / 3.0,
                        "feature_realized_vol_20d_z": -score,
                        "feature_amount_20d_log_z": score / 4.0,
                        "feature_turnover_20d_z": score / 5.0,
                        "feature_price_position_250d_z": score / 6.0,
                        "feature_limit_move_hits_20d_z": -score,
                        "next_trade_date": date + pd.offsets.BDay(1),
                    }
                )
    return pd.DataFrame(rows)


def test_select_with_industry_cap_limits_each_industry():
    df = _score_sample(months=1)
    part = df[df["model"] == STABLE_M5_EXTRATREES]

    selected = select_with_industry_cap(part, k=5, max_industry_names=2)

    assert len(selected) == 5
    assert selected["industry_level1"].value_counts().max() == 2
    assert selected.iloc[0]["symbol"] == "000001"


def test_constrained_monthly_and_leaderboard_report_concentration_drop():
    scores = _score_sample(months=4)
    monthly, holdings = build_constrained_monthly(scores, top_ks=[5], cap_grid={5: (2,)}, cost_bps=10.0)
    concentration = summarize_industry_concentration(holdings)
    rank_ic = build_rank_ic(scores)
    quantile = build_quantile_spread(scores, bucket_count=4)
    regime = summarize_regime_slice(
        monthly,
        pd.DataFrame(
            {
                "signal_date": sorted(scores["signal_date"].unique()),
                "realized_market_state": ["strong_down", "neutral", "neutral", "strong_up"],
            }
        ),
    )
    leaderboard = build_constrained_leaderboard(monthly, rank_ic, quantile, regime, concentration)

    uncapped = leaderboard[
        (leaderboard["base_model"] == STABLE_M5_EXTRATREES)
        & (leaderboard["selection_policy"] == "uncapped")
    ].iloc[0]
    capped = leaderboard[
        (leaderboard["base_model"] == STABLE_M5_EXTRATREES)
        & (leaderboard["selection_policy"] == "industry_names_cap")
    ].iloc[0]
    assert uncapped["max_industry_share_mean"] > capped["max_industry_share_mean"]
    assert capped["max_industry_share_mean"] <= 0.4
    assert capped["topk_minus_nextk_mean"] > 0
    first_holding = holdings.sort_values(["signal_date", "selected_rank"]).iloc[0]
    assert first_holding["buy_trade_date"] == "2024-02-01"
    assert first_holding["sell_trade_date"] == "2024-02-29"


def test_lagged_state_uses_only_prior_history_and_policy_scores_rank():
    scores = _score_sample(months=6)
    dataset = scores[scores["model"] == STABLE_M5_EXTRATREES].copy()
    dataset["candidate_pool_pass"] = True
    states = build_lagged_state_frame(dataset, min_history_months=2)

    assert states.loc[0, "lagged_regime"] == "neutral"
    assert states.loc[1, "lagged_regime"] == "neutral"
    assert states.loc[2, "state_history_months"] == 2

    policy = build_regime_policy_scores(scores, states)

    assert not policy.empty
    assert set(policy["model"]) == {M8_POLICY_MODEL}
    assert policy.groupby(["signal_date", "candidate_pool_version"])["score"].max().eq(1.0).all()


def test_parse_cap_grid_keeps_requested_topk_without_caps():
    out = parse_cap_grid("20:3,4;30:5", [20, 30, 50])

    assert out[20] == (3, 4)
    assert out[30] == (5,)
    assert out[50] == tuple()


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [
        (5, (1, 2)),
        (10, (2, 3)),
        (20, (3, 4, 5)),
        (30, (4, 5, 6, 8)),
        (50, (6, 8, 10)),
    ],
)
def test_default_cap_values_follow_topk_width(top_k: int, expected: tuple[int, ...]):
    assert default_cap_values_for_topk(top_k) == expected


def test_resolve_topk_and_cap_grid_supports_narrow_preset():
    top_ks, cap_grid = resolve_topk_and_cap_grid(preset="narrow", top_k_raw="", cap_grid_raw="")

    assert top_ks == [5, 10, 20]
    assert cap_grid == build_default_cap_grid([5, 10, 20])
    assert serialize_cap_grid(cap_grid) == "5:1,2;10:2,3;20:3,4,5"


def test_resolve_topk_and_cap_grid_auto_fills_manual_top10_top5():
    top_ks, cap_grid = resolve_topk_and_cap_grid(preset="default", top_k_raw="10,5", cap_grid_raw="")

    assert top_ks == [5, 10]
    assert cap_grid == {5: (1, 2), 10: (2, 3)}


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _score_sample(months=4).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(sys, "argv", ["run_monthly_selection_concentration_regime.py"])
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "reports" / "2026-05").mkdir(parents=True, exist_ok=True)

    assert run_monthly_concentration_regime(
        config=None,
        dataset=str(dataset_path),
        duckdb_path="",
        output_prefix="m8_contract_test",
        as_of_date="2026-05-02",
        results_dir=str(tmp_path / "results"),
        topk_preset="default",
        top_k="2",
        cap_grid="",
        candidate_pools="U2_risk_sane",
        bucket_count=5,
        min_train_months=2,
        min_train_rows=10,
        max_fit_rows=0,
        cost_bps=10.0,
        random_seed=42,
        availability_lag_days=30,
        model_n_jobs=0,
        min_state_history_months=2,
        families="industry_breadth",
        skip_m6=True,
        root=tmp_path,
    ) == 0

    manifests = sorted((tmp_path / "results").glob("m8_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m8_concentration_regime"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"leaderboard_csv", "gate_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
