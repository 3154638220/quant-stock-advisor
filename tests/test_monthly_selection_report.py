from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import scripts.run_monthly_selection_report as m7_report
from scripts.run_monthly_selection_ltr import build_m6_feature_spec, summarize_ltr_feature_importance
from scripts.run_monthly_selection_multisource import attach_industry_breadth_features
from scripts.run_monthly_selection_report import (
    M7RunConfig,
    apply_m9_feature_coverage_policy,
    attach_stock_names,
    build_full_fit_report_scores,
    build_recommendation_table,
    select_report_signal_date,
    summarize_m9_integrity,
    summarize_report_feature_coverage,
)
try:
    from scripts.validate_research_contracts import validate_manifest
except ImportError:
    validate_manifest = None  # type: ignore[assignment]


def _m7_sample(months: int = 5, symbols: int = 10) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    for m, date in enumerate(dates):
        market_ret = -0.01 + 0.006 * m
        for i in range(symbols):
            industry = "电子" if i < symbols // 2 else "计算机"
            signal = float(i) + (2.0 if industry == "计算机" else 0.0)
            forward = -0.03 + 0.010 * i + 0.003 * m if m < months - 1 else None
            candidate_pass = not (m == months - 1 and i == 0)
            rows.append(
                {
                    "signal_date": date,
                    "symbol": f"000{i + 1:03d}",
                    "name": f"样本{i + 1}",
                    "candidate_pool_version": "U2_risk_sane",
                    "candidate_pool_rule": "U1 + risk sanity filters",
                    "candidate_pool_pass": candidate_pass,
                    "candidate_pool_reject_reason": "" if candidate_pass else "open_limit_up_unbuyable",
                    "label_forward_1m_o2o_return": forward,
                    "label_market_ew_o2o_return": market_ret if forward is not None else None,
                    "label_forward_1m_excess_vs_market": forward - market_ret if forward is not None else None,
                    "label_forward_1m_industry_neutral_excess": forward - (0.02 if industry == "计算机" else 0.0)
                    if forward is not None
                    else None,
                    "label_future_top_20pct": 1 if i >= symbols - 2 and forward is not None else 0,
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
                    "risk_flags": "extreme_turnover" if i == symbols - 1 else "",
                    "is_buyable_tplus1_open": candidate_pass,
                    "buyability_reject_reason": "" if candidate_pass else "open_limit_up_unbuyable",
                    "next_trade_date": date + pd.Timedelta(days=1),
                }
            )
    return pd.DataFrame(rows)


def test_select_report_signal_date_uses_latest_candidate_pass_month():
    df = _m7_sample(months=3, symbols=4)
    latest = df["signal_date"].max()
    df.loc[df["signal_date"] == latest, "candidate_pool_pass"] = False

    selected = select_report_signal_date(df, candidate_pools=("U2_risk_sane",))

    assert selected == pd.Timestamp("2024-02-29")


def test_select_report_signal_date_skips_month_without_next_trade_date():
    df = _m7_sample(months=3, symbols=4)
    latest = df["signal_date"].max()
    prev = pd.Timestamp("2024-02-29")
    df.loc[df["signal_date"] == latest, "next_trade_date"] = pd.NaT

    selected = select_report_signal_date(df, candidate_pools=("U2_risk_sane",))

    assert selected == prev


def test_requested_signal_date_requires_next_trade_date_for_passed_rows():
    df = _m7_sample(months=3, symbols=4)
    latest = df["signal_date"].max()
    df.loc[df["signal_date"] == latest, "next_trade_date"] = pd.NaT

    with pytest.raises(ValueError, match="缺少 next_trade_date"):
        select_report_signal_date(df, candidate_pools=("U2_risk_sane",), requested=latest)


def test_build_recommendation_table_includes_m7_required_fields_and_previous_rank():
    df = _m7_sample(months=3, symbols=5)
    signal_date = pd.Timestamp("2024-02-29")
    scores = df[(df["signal_date"] == signal_date) & df["candidate_pool_pass"]].copy()
    scores["model"] = "M6_xgboost_rank_ndcg"
    scores["model_type"] = "xgboost_ranker"
    scores["score"] = scores["feature_ret_20d_z"].rank(pct=True)
    scores["score_percentile"] = scores["score"]
    scores["feature_spec"] = "m6_core_price_volume"
    imp = pd.DataFrame(
        {
            "candidate_pool_version": ["U2_risk_sane", "U2_risk_sane"],
            "model": ["M6_xgboost_rank_ndcg", "M6_xgboost_rank_ndcg"],
            "feature": ["feature_ret_20d_z", "feature_realized_vol_20d_z"],
            "importance": [0.7, 0.3],
            "signed_weight": [pd.NA, pd.NA],
            "feature_spec": ["m6_core_price_volume", "m6_core_price_volume"],
            "feature_families": ["price_volume", "price_volume"],
            "observations": [1, 1],
        }
    )
    previous = pd.DataFrame(
        {
            "signal_date": ["2024-01-31"],
            "candidate_pool_version": ["U2_risk_sane"],
            "model": ["M6_xgboost_rank_ndcg"],
            "top_k": [2],
            "selected_rank": [1],
            "symbol": ["000005"],
        }
    )

    rec, contrib = build_recommendation_table(
        scores,
        df,
        imp,
        feature_cols=["feature_ret_20d_z", "feature_realized_vol_20d_z"],
        top_ks=[2],
        previous_holdings=previous,
    )

    required = {
        "rank",
        "symbol",
        "name",
        "score",
        "score_percentile",
        "industry",
        "feature_contrib",
        "risk_flags",
        "last_month_rank",
        "buyability",
        "buy_trade_date",
        "sell_trade_date",
    }
    assert required.issubset(rec.columns)
    assert rec.iloc[0]["symbol"] == "000005"
    assert rec.iloc[0]["last_month_rank"] == 1
    assert "feature_ret_20d_z" in rec.iloc[0]["feature_contrib"]
    assert rec.iloc[0]["risk_flags"] == "extreme_turnover"
    assert rec.iloc[0]["buyability"] == "buyable_tplus1_open"
    assert rec.iloc[0]["next_trade_date"] == "2024-03-01"
    assert rec.iloc[0]["buy_trade_date"] == "2024-03-01"
    assert rec.iloc[0]["sell_trade_date"] == "2024-03-31"
    assert not contrib.empty


def test_attach_stock_names_fills_report_names_from_cache():
    df = _m7_sample(months=1, symbols=2).drop(columns=["name"])
    names = pd.DataFrame({"代码": ["000001", "000002"], "名称": ["平安银行", "万科A"]})

    out = attach_stock_names(df, names)

    assert out.loc[out["symbol"] == "000001", "name"].iloc[0] == "平安银行"
    assert out.loc[out["symbol"] == "000002", "name"].iloc[0] == "万科A"


def test_m9_feature_policy_removes_zero_coverage_core_feature():
    df = _m7_sample(months=2, symbols=5)
    df["feature_fundamental_ev_ebitda"] = pd.NA
    df["feature_fundamental_ev_ebitda_z"] = 0.0
    df["is_missing_feature_fundamental_ev_ebitda"] = 1
    spec = type(
        "Spec",
        (),
        {
            "name": "spec",
            "families": ("price_volume", "fundamental"),
            "feature_cols": ("feature_ret_20d_z", "feature_fundamental_ev_ebitda_z"),
        },
    )()
    coverage = summarize_report_feature_coverage(df, spec, candidate_pools=("U2_risk_sane",))

    active, policy = apply_m9_feature_coverage_policy(
        df,
        spec,
        coverage,
        candidate_pools=("U2_risk_sane",),
        min_core_coverage=0.30,
    )

    assert "feature_ret_20d_z" in active
    assert "feature_fundamental_ev_ebitda_z" not in active
    assert "is_missing_feature_fundamental_ev_ebitda" in active
    ev = policy[policy["feature"] == "feature_fundamental_ev_ebitda_z"].iloc[0]
    assert ev["m9_feature_policy"] == "missing_flag_only_low_coverage"


def test_m9_integrity_passes_when_names_buyability_and_coverage_are_clean():
    df = _m7_sample(months=2, symbols=5)
    signal_date = pd.Timestamp("2024-02-29")
    rec = pd.DataFrame(
        {
            "name": ["样本1", "样本2"],
            "buyability": ["buyable_tplus1_open", "buyable_tplus1_open"],
        }
    )
    feature_policy = pd.DataFrame(
        {
            "feature": ["feature_ret_20d_z"],
            "candidate_pool_pass_coverage_ratio": [1.0],
            "m9_feature_policy": ["core_feature"],
        }
    )

    out = summarize_m9_integrity(
        dataset=df,
        recommendations=rec,
        feature_coverage=pd.DataFrame(),
        feature_policy=feature_policy,
        report_signal_date=signal_date,
        candidate_pools=("U2_risk_sane",),
    )

    assert out["pass"].all()


def test_m9_integrity_fails_st_name_recommendations():
    df = _m7_sample(months=2, symbols=5)
    rec = pd.DataFrame(
        {
            "name": ["*ST样本"],
            "buyability": ["buyable_tplus1_open"],
        }
    )

    out = summarize_m9_integrity(
        dataset=df,
        recommendations=rec,
        feature_coverage=pd.DataFrame(),
        feature_policy=pd.DataFrame(),
        report_signal_date=pd.Timestamp("2024-02-29"),
        candidate_pools=("U2_risk_sane",),
    )

    row = out[out["check"] == "recommendation_excludes_st_names"].iloc[0]
    assert row["value"] == 1
    assert not row["pass"]


def test_m7_full_fit_report_scores_rank_latest_unlabeled_month():
    pytest.importorskip("xgboost")
    df = attach_industry_breadth_features(_m7_sample(months=5, symbols=10))
    spec = build_m6_feature_spec(["industry_breadth"])
    cfg = M7RunConfig(
        top_ks=(2,),
        report_top_k=2,
        candidate_pools=("U2_risk_sane",),
        min_train_months=2,
        min_train_rows=10,
        random_seed=3,
        relevance_grades=5,
    )

    scores, raw_importance = build_full_fit_report_scores(
        df,
        spec,
        cfg,
        report_signal_date=pd.Timestamp("2024-05-31"),
    )
    feature_importance = summarize_ltr_feature_importance(raw_importance)
    rec, _ = build_recommendation_table(
        scores,
        df,
        feature_importance,
        feature_cols=[c for c in spec.feature_cols if c in df.columns],
        top_ks=[2],
    )

    assert not scores.empty
    assert scores["signal_date"].nunique() == 1
    assert scores["model"].unique().tolist() == ["M6_xgboost_rank_ndcg"]
    assert rec["top_k"].unique().tolist() == [2]
    assert rec["rank"].tolist() == [1, 2]


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _m7_sample(months=4, symbols=10).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(m7_report, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_selection_report.py",
            "--dataset",
            str(dataset_path),
            "--results-dir",
            "results",
            "--output-prefix",
            "m7_contract_test",
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
            "--stock-name-cache",
            str(tmp_path / "nonexistent_names.csv"),
        ],
    )

    assert m7_report.main() == 0

    manifests = sorted((tmp_path / "results").glob("m7_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    if validate_manifest is not None:
        assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m7_recommendation_report"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"recommendations_csv", "leaderboard_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
