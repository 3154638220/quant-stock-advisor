from __future__ import annotations

import json
import sys

import duckdb
import pandas as pd

import scripts.run_monthly_selection_multisource as multisource
from src.pipeline.monthly_baselines import (
    build_leaderboard,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
)
from src.pipeline.monthly_multisource import (
    M5RunConfig,
    attach_fundamental_features,
    attach_industry_breadth_features,
    build_feature_specs,
    build_incremental_delta,
    build_walk_forward_scores_for_spec,
)
from src.research.contracts import validate_manifest


def _m5_sample(months: int = 5, symbols: int = 10) -> pd.DataFrame:
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


def test_industry_breadth_features_are_signal_date_cross_sectional():
    df = _m5_sample(months=2, symbols=10)

    out = attach_industry_breadth_features(df)

    cols = {
        "feature_industry_ret20_mean",
        "feature_industry_positive_ret20_ratio",
        "feature_industry_low_vol20_mean_z",
    }
    assert cols.issubset(out.columns)
    jan = out[out["signal_date"] == pd.Timestamp("2024-01-31")]
    electronics = jan[jan["industry_level1"] == "电子"]["feature_industry_ret20_mean"].unique()
    computers = jan[jan["industry_level1"] == "计算机"]["feature_industry_ret20_mean"].unique()
    assert len(electronics) == 1
    assert len(computers) == 1
    assert computers[0] > electronics[0]


def test_feature_specs_are_cumulative_in_m5_order():
    specs = build_feature_specs(["industry_breadth", "fund_flow", "fundamental"])

    names = [s.name for s in specs]
    assert names == [
        "price_volume_only",
        "plus_industry_breadth",
        "plus_industry_breadth_plus_fund_flow",
        "plus_industry_breadth_plus_fund_flow_plus_fundamental",
    ]
    assert len(specs[-1].feature_cols) > len(specs[0].feature_cols)
    assert set(specs[1].feature_cols).issubset(set(specs[2].feature_cols))


def test_fundamental_features_drop_statement_rows_without_positive_notice_lag(tmp_path):
    db_path = tmp_path / "fundamental.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_fundamental (
                symbol VARCHAR,
                report_period DATE,
                announcement_date DATE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                roe_ttm DOUBLE,
                net_profit_yoy DOUBLE,
                gross_margin_change DOUBLE,
                debt_to_assets_change DOUBLE,
                ocf_to_net_profit DOUBLE,
                ocf_to_asset DOUBLE,
                gross_margin_delta DOUBLE,
                asset_turnover DOUBLE,
                net_margin_stability DOUBLE,
                source VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_fundamental VALUES
            ('000001', DATE '2025-09-30', DATE '2025-10-31', 10, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'stock_financial_analysis_indicator_em'),
            ('000001', DATE '2025-12-31', DATE '2025-12-31', 999, 1, 99, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'stock_financial_analysis_indicator'),
            ('000002', DATE '2025-12-31', DATE '2025-12-31', 30, 3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'stock_value_em')
            """
        )
    finally:
        con.close()

    dataset = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-02-28", "2026-02-28"]),
            "symbol": ["000001", "000002"],
        }
    )

    out = attach_fundamental_features(dataset, db_path).sort_values("symbol").reset_index(drop=True)

    assert out.loc[0, "feature_fundamental_pe_ttm"] == 10
    assert out.loc[0, "feature_fundamental_roe_ttm"] == 1
    assert out.loc[1, "feature_fundamental_pe_ttm"] == 30


def test_m5_walk_forward_scores_and_delta_compare_to_price_volume_baseline():
    df = attach_industry_breadth_features(_m5_sample(months=5, symbols=10))
    specs = build_feature_specs(["industry_breadth"])
    cfg = M5RunConfig(
        top_ks=(2,),
        candidate_pools=("U1_liquid_tradable",),
        min_train_months=2,
        min_train_rows=10,
        include_xgboost=False,
        random_seed=3,
    )

    score_frames = []
    for spec in specs:
        scores, importance = build_walk_forward_scores_for_spec(df, spec, cfg)
        assert not scores.empty
        assert not importance.empty
        assert scores["model"].str.startswith(f"M5_{spec.name}_").all()
        score_frames.append(scores)
    scores = pd.concat(score_frames, ignore_index=True)
    rank_ic = build_rank_ic(scores)
    monthly, _ = build_monthly_long(scores, top_ks=[2], cost_bps=10.0)
    quantile = build_quantile_spread(scores, bucket_count=4)
    leaderboard = build_leaderboard(monthly, rank_ic, quantile, pd.DataFrame())
    delta = build_incremental_delta(leaderboard)

    assert not delta.empty
    assert set(delta["feature_spec"]) == {"plus_industry_breadth"}
    assert "delta_rank_ic_mean" in delta.columns


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    _m5_sample(months=4, symbols=10).to_parquet(dataset_path, index=False)
    monkeypatch.setattr(multisource, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_selection_multisource.py",
            "--dataset",
            str(dataset_path),
            "--results-dir",
            "results",
            "--output-prefix",
            "m5_contract_test",
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
            "--ml-models",
            "elasticnet,logistic",
            "--skip-xgboost",
        ],
    )

    assert multisource.main() == 0

    manifests = sorted((tmp_path / "results").glob("m5_contract_test_*_manifest.json"))
    assert len(manifests) == 1
    assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m5_multisource"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"leaderboard_csv", "incremental_delta_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
