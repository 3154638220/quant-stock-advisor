from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.run_monthly_selection_ltr as ltr
from src.pipeline.monthly_ltr import (
    M6RunConfig,
    build_ltr_relevance,
    build_m6_feature_spec,
    build_walk_forward_ltr_scores,
)
from src.pipeline.monthly_baselines import (
    build_leaderboard,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
)
from src.pipeline.monthly_multisource import attach_industry_breadth_features
try:
    from scripts.validate_research_contracts import validate_manifest
except ImportError:
    validate_manifest = None  # type: ignore[assignment]


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
    if validate_manifest is not None:
        assert validate_manifest(manifests[0], root=tmp_path) == []
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["result_type"] == "monthly_selection_m6_ltr"
    artifact_names = {x["name"] for x in payload["artifacts"]}
    assert artifact_names >= {"leaderboard_csv", "feature_importance_csv", "manifest_json", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()


# ── P2-3: Stacking 集成 OOF meta-learner 测试 ────────────────────────────

def test_stacking_meta_learner_returns_none_on_insufficient_data():
    """P2-3: 数据不足时 meta-learner 返回 None。"""
    from src.pipeline.monthly_ltr import _train_stacking_meta_learner

    empty = pd.DataFrame()
    result = _train_stacking_meta_learner(empty, oof_cols=[], label_col="label_future_top_20pct")
    assert result is None


def test_stacking_meta_learner_trains_on_valid_data():
    """P2-3: 有效 OOF 数据上 meta-learner 成功训练。"""
    from src.pipeline.monthly_ltr import _train_stacking_meta_learner

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "oof_M6_xgboost_rank_ndcg": np.random.randn(n),
        "oof_M6_top20_calibrated": np.random.randn(n),
        "label_future_top_20pct": (np.random.rand(n) > 0.8).astype(int),
    })
    meta = _train_stacking_meta_learner(
        df,
        oof_cols=["oof_M6_xgboost_rank_ndcg", "oof_M6_top20_calibrated"],
        label_col="label_future_top_20pct",
    )
    assert meta is not None
    assert hasattr(meta, "predict_proba")


def test_collect_stacking_oof_frame_pivots_correctly():
    """P2-3: OOF 记录正确 pivot 为特征矩阵。"""
    from src.pipeline.monthly_ltr import _collect_stacking_oof_frame

    records = [
        {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "000001", "candidate_pool_version": "U1_liquid_tradable", "oof_model": "M6_xgboost_rank_ndcg", "oof_score": 0.8},
        {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "000001", "candidate_pool_version": "U1_liquid_tradable", "oof_model": "M6_top20_calibrated", "oof_score": 0.6},
        {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "000002", "candidate_pool_version": "U1_liquid_tradable", "oof_model": "M6_xgboost_rank_ndcg", "oof_score": 0.3},
        {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "000002", "candidate_pool_version": "U1_liquid_tradable", "oof_model": "M6_top20_calibrated", "oof_score": 0.5},
    ]
    oof = _collect_stacking_oof_frame(records)
    assert len(oof) == 2
    assert "oof_M6_xgboost_rank_ndcg" in oof.columns or any("M6_xgboost_rank_ndcg" in c for c in oof.columns)
    assert "oof_M6_top20_calibrated" in oof.columns or any("M6_top20_calibrated" in c for c in oof.columns)


def test_build_walk_forward_ltr_scores_with_stacking_does_not_crash():
    """P2-3: 启用 stacking_ensemble 后 walk-forward 不崩溃。"""
    cfg = M6RunConfig(
        ltr_models=("xgboost_rank_ndcg", "top20_calibrated", "stacking_ensemble"),
        stacking_enabled=True,
        min_train_months=3,
        min_train_rows=10,
        window_type="expanding",
        halflife_months=0,
    )
    dataset = _m6_sample(months=5, symbols=10)
    spec = build_m6_feature_spec(["industry_breadth"])
    scores, imp = build_walk_forward_ltr_scores(dataset, spec, cfg)
    assert not scores.empty
    models = set(scores["model"].unique())
    # 应至少包含基模型，stacking 可能在数据不足时不产出
    assert "M6_xgboost_rank_ndcg" in models or "M6_top20_calibrated" in models


# ── P0-1: 行业内特征中性化 ──────────────────────────────────────────────

def test_industry_neutral_zscore_reduces_sector_bias():
    """P0-1: 行业内 z-score 消除行业间特征分布差异。

    构造一个两行业样本：证券行业 asset_turnover 系统性偏高。
    验证中性化后同一行业内 z-score 均值为 0，跨行业可比。
    """
    from src.pipeline.monthly_multisource import industry_neutral_zscore, FUNDAMENTAL_RAW_FEATURES

    np.random.seed(42)
    n = 100
    dates = [pd.Timestamp("2024-01-31")] * n
    # 证券行业 turnover 偏高（均值 2.0），食品饮料偏低（均值 0.5）
    industry = ["证券"] * (n // 2) + ["食品饮料"] * (n // 2)
    asset_turnover_raw = np.concatenate([
        np.random.normal(2.0, 0.3, n // 2),   # 证券：高 turnover
        np.random.normal(0.5, 0.1, n // 2),   # 食品饮料：低 turnover
    ])
    df = pd.DataFrame({
        "signal_date": dates,
        "industry_level1": industry,
        "symbol": [f"{i:06d}" for i in range(n)],
        "feature_asset_turnover": asset_turnover_raw,
    })
    # 模拟已有全截面 _z 列
    df["feature_asset_turnover_z"] = (df["feature_asset_turnover"] - df["feature_asset_turnover"].mean()) / df["feature_asset_turnover"].std(ddof=0)

    result = industry_neutral_zscore(df, ["feature_asset_turnover_z"])

    assert "feature_asset_turnover_z_ind_z" in result.columns, "应生成 _ind_z 列"

    # 行业内均值应接近 0
    for ind in ["证券", "食品饮料"]:
        ind_mean = result.loc[result["industry_level1"] == ind, "feature_asset_turnover_z_ind_z"].mean()
        assert abs(ind_mean) < 1e-6, f"{ind} 行业内 _ind_z 均值应接近 0，实际 {ind_mean}"

    # 原 _z 列行业内均值不应为 0（证券偏高，食品饮料偏低）
    sec_z = result.loc[result["industry_level1"] == "证券", "feature_asset_turnover_z"].mean()
    food_z = result.loc[result["industry_level1"] == "食品饮料", "feature_asset_turnover_z"].mean()
    assert sec_z > food_z, f"原 _z 证券行业均值 ({sec_z}) 应高于食品饮料 ({food_z})"


def test_industry_neutral_zscore_with_m6_walk_forward():
    """P0-1: use_industry_neutral_zscore=True 时 M6 walk-forward 正常产出结果。"""
    pytest.importorskip("xgboost")
    df = _m6_sample(months=5, symbols=20)
    # 添加基本面 _z 列（模拟已有全截面 z-score）
    for feat in ["feature_asset_turnover_z", "feature_roe_ttm_z", "feature_gross_margin_z"]:
        df[feat] = np.random.randn(len(df))
    spec = build_m6_feature_spec(
        ["industry_breadth", "fund_flow", "fundamental"],
        use_industry_neutral_zscore=True,
    )
    cfg = M6RunConfig(
        top_ks=(2,),
        candidate_pools=("U1_liquid_tradable",),
        min_train_months=2,
        min_train_rows=10,
        max_fit_rows=0,
        random_seed=3,
        use_industry_neutral_zscore=True,
        ltr_models=("xgboost_rank_ndcg",),
    )
    scores, importance = build_walk_forward_ltr_scores(df, spec, cfg)
    assert not scores.empty, "use_industry_neutral_zscore=True 时应正常产出分数"
    assert "M6_xgboost_rank_ndcg" in set(scores["model"]), "应包含 xgboost ranker 输出"
