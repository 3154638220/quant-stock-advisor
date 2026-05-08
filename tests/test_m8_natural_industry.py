"""Tests for src/pipeline/m8_natural_industry.py — 行业约束与软优化边界。

覆盖重点（P3-1）：
- select_soft_industry_risk 的 k=0/空输入/单行业全集中/全 UNKNOWN 行业边界
- _weighted_turnover 的 None prev/空集/全替换场景
- build_monthly_from_scores 的空输入与非空路径
- build_soft_penalty_scores / build_score_decomposition_scores 空输入
- _target_for_variant 的纯 NaN 列 fallback
- build_gate_table / build_leaderboard 的空输入与缺失列
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.m8_natural_industry import (
    LABEL_VARIANTS,
    LabelVariant,
    M8NaturalRunConfig,
    _rank_within_group,
    _target_for_variant,
    _weighted_turnover,
    build_gate_table,
    build_leaderboard,
    build_monthly_from_scores,
    build_soft_penalty_scores,
    build_score_decomposition_scores,
    build_soft_optimizer_scores,
    copy_source_metric_for_optimizer,
    select_soft_industry_risk,
    load_hardcap_baseline,
)
from src.research.gates import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
)

# ── 常量 ───────────────────────────────────────────────────────────────────


def test_label_variants_defined():
    assert len(LABEL_VARIANTS) == 3
    names = {v.name for v in LABEL_VARIANTS}
    assert names == {"market_excess", "industry_neutral_excess", "blended_excess_50_50"}


def test_m8_config_defaults():
    cfg = M8NaturalRunConfig()
    assert cfg.top_ks == (20, 30)
    assert cfg.soft_gammas == (0.05, 0.10, 0.20, 0.35, 0.50, 0.80)
    assert cfg.hardcap_tolerance == 0.005


# ═══════════════════════════════════════════════════════════════════════════
# _target_for_variant
# ═══════════════════════════════════════════════════════════════════════════


def test_target_for_variant_market():
    ds = pd.DataFrame({
        EXCESS_COL: [0.01, 0.02, -0.01],
        INDUSTRY_EXCESS_COL: [0.005, 0.01, -0.005],
    })
    v = LabelVariant("market_excess", "", 1.0, 0.0)
    result = _target_for_variant(ds, v)
    np.testing.assert_array_almost_equal(result.values, [0.01, 0.02, -0.01])


def test_target_for_variant_industry_neutral():
    ds = pd.DataFrame({
        EXCESS_COL: [0.01, 0.02, -0.01],
        INDUSTRY_EXCESS_COL: [0.005, 0.01, -0.005],
    })
    v = LabelVariant("industry_neutral_excess", "", 0.0, 1.0)
    result = _target_for_variant(ds, v)
    np.testing.assert_array_almost_equal(result.values, [0.005, 0.01, -0.005])


def test_target_for_variant_blended():
    ds = pd.DataFrame({
        EXCESS_COL: [0.1, 0.2],
        INDUSTRY_EXCESS_COL: [0.05, 0.1],
    })
    v = LabelVariant("blended", "", 0.5, 0.5)
    result = _target_for_variant(ds, v)
    expected = 0.5 * ds[EXCESS_COL] + 0.5 * ds[INDUSTRY_EXCESS_COL]
    np.testing.assert_array_almost_equal(result.values, expected.values)


def test_target_for_variant_fallback_to_market_when_industry_all_nan():
    """当 industry_neutral_excess 全 NaN 时应 fallback 到 market。"""
    ds = pd.DataFrame({
        EXCESS_COL: [0.01, 0.02],
        INDUSTRY_EXCESS_COL: [np.nan, np.nan],
    })
    v = LabelVariant("industry_neutral_excess", "", 0.0, 1.0)
    result = _target_for_variant(ds, v)
    np.testing.assert_array_almost_equal(result.values, [0.01, 0.02])


def test_target_for_variant_missing_column():
    """缺少列时，pd.to_numeric(None) → nan → fallback 到另一列或返回 0。"""
    ds = pd.DataFrame({
        EXCESS_COL: [0.02, 0.03],
    })
    v = LabelVariant("market_excess", "", 1.0, 0.0)
    result = _target_for_variant(ds, v)
    assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# _rank_within_group
# ═══════════════════════════════════════════════════════════════════════════


def test_rank_within_group_single_value():
    s = pd.Series([5.0], index=[0])
    result = _rank_within_group(s)
    assert result.iloc[0] == 0.5


def test_rank_within_group_all_nan():
    s = pd.Series([np.nan, np.nan])
    result = _rank_within_group(s)
    assert (result == 0.5).all()


def test_rank_within_group_basic():
    s = pd.Series([1.0, 3.0, 2.0])
    result = _rank_within_group(s)
    assert result.iloc[0] < result.iloc[2] < result.iloc[1]
    assert 0 <= result.min() <= 1
    assert 0 <= result.max() <= 1


# ═══════════════════════════════════════════════════════════════════════════
# _weighted_turnover
# ═══════════════════════════════════════════════════════════════════════════


def test_weighted_turnover_none_prev():
    assert np.isnan(_weighted_turnover(None, {"A", "B"}))


def test_weighted_turnover_empty_prev():
    # 空 prev → 所有 cur 标的权重从 0 → 等权，L1 距离 = 1.0，half = 0.5
    assert _weighted_turnover(set(), {"A", "B"}) == 0.5


def test_weighted_turnover_full_overlap():
    s = {"A", "B", "C"}
    assert _weighted_turnover(s, s) == 0.0


def test_weighted_turnover_half_overlap():
    prev = {"A", "B"}
    cur = {"B", "C"}
    to = _weighted_turnover(prev, cur)
    assert to == pytest.approx(0.5)


def test_weighted_turnover_empty_both():
    # 两个空集 → 没有标的，L1 距离为 0
    assert _weighted_turnover(set(), set()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# select_soft_industry_risk
# ═══════════════════════════════════════════════════════════════════════════


def _make_industry_part(n_stocks: int = 30) -> pd.DataFrame:
    np.random.seed(42)
    industries = (["银行"] * 10) + (["科技"] * 10) + (["消费"] * 10)
    return pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, n_stocks + 1)],
        "score": np.linspace(0.1, 1.0, n_stocks),
        "industry_level1": industries[:n_stocks],
    })


def test_select_soft_industry_risk_basic():
    part = _make_industry_part(30)
    result = select_soft_industry_risk(part, k=10, gamma=0.1)
    assert len(result) == 10
    assert "adjusted_score" in result.columns


def test_select_soft_industry_risk_empty():
    result = select_soft_industry_risk(pd.DataFrame(), k=10, gamma=0.1)
    assert result.empty


def test_select_soft_industry_risk_k_zero():
    part = _make_industry_part(10)
    result = select_soft_industry_risk(part, k=0, gamma=0.1)
    assert result.empty


def test_select_soft_industry_risk_k_larger_than_candidates():
    part = _make_industry_part(5)
    result = select_soft_industry_risk(part, k=20, gamma=0.1)
    assert len(result) == 5


def test_select_soft_industry_risk_single_industry():
    part = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 11)],
        "score": np.linspace(0.1, 1.0, 10),
        "industry_level1": ["银行"] * 10,
    })
    result = select_soft_industry_risk(part, k=5, gamma=0.5)
    assert len(result) == 5


def test_select_soft_industry_risk_all_nan_industry():
    part = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 11)],
        "score": np.linspace(0.1, 1.0, 10),
        "industry_level1": [np.nan] * 10,
    })
    result = select_soft_industry_risk(part, k=5, gamma=0.5)
    assert len(result) == 5


def test_select_soft_industry_risk_gamma_zero():
    """gamma=0 退化为纯 score 排序。"""
    part = _make_industry_part(30)
    result = select_soft_industry_risk(part, k=10, gamma=0.0)
    assert len(result) == 10
    top_scores = set(part.nlargest(10, "score")["symbol"])
    assert set(result["symbol"]) == top_scores


def test_select_soft_industry_risk_diversifies():
    """高 gamma 不应减少行业分散度。"""
    part = _make_industry_part(30)
    result_g0 = select_soft_industry_risk(part, k=10, gamma=0.0)
    result_high = select_soft_industry_risk(part, k=10, gamma=2.0)
    g0_ind = result_g0["industry_level1"].nunique()
    high_ind = result_high["industry_level1"].nunique()
    assert high_ind >= g0_ind


# ═══════════════════════════════════════════════════════════════════════════
# build_monthly_from_scores
# ═══════════════════════════════════════════════════════════════════════════


def test_build_monthly_from_scores_empty():
    empty = pd.DataFrame()
    monthly, holdings = build_monthly_from_scores(
        empty, top_ks=[10], cost_bps=10.0, selection_policy="top_k",
    )
    assert monthly.empty
    assert holdings.empty


def test_build_monthly_from_scores_topk_policy():
    np.random.seed(42)
    syms = [f"{i:06d}" for i in range(1, 41)]
    scores = pd.DataFrame({
        "signal_date": pd.Timestamp("2023-06-30"),
        "candidate_pool_version": "U1",
        "model": "test_model",
        "model_type": "elasticnet",
        "symbol": syms,
        "score": np.linspace(0.1, 1.0, len(syms)),
        "industry_level1": (["银行"] * 10 + ["科技"] * 10 + ["消费"] * 10 + ["地产"] * 10),
        LABEL_COL: np.random.randn(len(syms)) * 0.02,
        MARKET_COL: [0.01] * len(syms),
        "risk_flags": [""] * len(syms),
        "source_model": ["test_model"] * len(syms),
    })
    monthly, holdings = build_monthly_from_scores(
        scores, top_ks=[10, 20], cost_bps=10.0, selection_policy="top_k",
    )
    assert not monthly.empty
    assert not holdings.empty
    assert set(monthly["top_k"].unique()) == {10, 20}
    assert "topk_excess_vs_market" in monthly.columns


def test_build_monthly_from_scores_soft_risk_budget():
    scores = pd.DataFrame({
        "signal_date": pd.Timestamp("2023-06-30"),
        "candidate_pool_version": "U1",
        "model": "test_model",
        "model_type": "elasticnet",
        "symbol": [f"{i:06d}" for i in range(1, 41)],
        "score": np.linspace(0.1, 1.0, 40),
        "industry_level1": (["银行"] * 15 + ["科技"] * 15 + ["消费"] * 10),
        LABEL_COL: np.random.randn(40) * 0.02,
        MARKET_COL: [0.01] * 40,
        "risk_flags": [""] * 40,
        "source_model": ["test_model"] * 40,
    })
    monthly, holdings = build_monthly_from_scores(
        scores, top_ks=[10], cost_bps=10.0,
        selection_policy="soft_industry_risk_budget", soft_gamma=0.2,
    )
    assert not monthly.empty
    assert (monthly["selection_policy"] == "soft_industry_risk_budget").all()


# ═══════════════════════════════════════════════════════════════════════════
# 空输入安全测试
# ═══════════════════════════════════════════════════════════════════════════


def test_build_soft_penalty_scores_empty():
    result = build_soft_penalty_scores(pd.DataFrame(), gammas=[0.1, 0.5])
    assert result.empty


def test_build_score_decomposition_scores_empty():
    scores, summary = build_score_decomposition_scores(pd.DataFrame())
    assert scores.empty
    assert summary.empty


def test_build_soft_optimizer_scores_empty():
    # 空 DataFrame 传入 → 函数应返回空结果（需要含 model 列）
    empty = pd.DataFrame(columns=["model", "score", "symbol", "industry_level1"])
    result = build_soft_optimizer_scores(empty, gammas=[0.1])
    assert len(result) == 1
    assert result[0][1].empty


def test_copy_source_metric_empty():
    result = copy_source_metric_for_optimizer(pd.DataFrame(), pd.DataFrame())
    assert result.empty


# ═══════════════════════════════════════════════════════════════════════════
# build_leaderboard
# ═══════════════════════════════════════════════════════════════════════════


def test_build_leaderboard_empty():
    result = build_leaderboard(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(),
    )
    assert result.empty


def test_build_leaderboard_basic():
    monthly = pd.DataFrame({
        "candidate_pool_version": ["U1", "U1"],
        "model": ["m1", "m1"],
        "model_type": ["elasticnet", "elasticnet"],
        "selection_policy": ["top_k", "top_k"],
        "score_family": ["", ""],
        "label_variant": ["", ""],
        "soft_gamma": [np.nan, np.nan],
        "top_k": [10, 10],
        "signal_date": [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-31")],
        "topk_return": [0.015, 0.012],
        "topk_excess_vs_market": [0.005, 0.003],
        "topk_minus_nextk": [0.01, 0.008],
        "topk_industry_neutral_excess": [0.003, 0.002],
        "turnover_half_l1": [0.5, 0.6],
        "cost_drag": [0.0005, 0.0006],
        "topk_excess_after_cost": [0.0045, 0.0024],
    })
    # 提供 rank_ic + industry_concentration 以通过 sort_values
    rank_ic = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "model": ["m1"],
        "rank_ic": [0.05],
    })
    ind_conc = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "model": ["m1"],
        "top_k": [10],
        "selection_policy": ["top_k"],
        "max_industry_share": [0.3],
        "industry_count": [5],
        "concentration_pass": [1.0],
    })
    result = build_leaderboard(
        monthly, rank_ic, pd.DataFrame(),
        pd.DataFrame(), ind_conc,
    )
    assert not result.empty
    assert "topk_excess_mean" in result.columns
    assert "topk_excess_annualized" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# load_hardcap_baseline
# ═══════════════════════════════════════════════════════════════════════════


def test_load_hardcap_baseline_none():
    result = load_hardcap_baseline(None)
    assert result.empty


def test_load_hardcap_baseline_nonexistent(tmp_path):
    result = load_hardcap_baseline(tmp_path / "no.csv")
    assert result.empty


def test_load_hardcap_baseline_no_hardcap_rows(tmp_path):
    p = tmp_path / "lb.csv"
    p.write_text(
        "candidate_pool_version,top_k,selection_policy,model,topk_excess_after_cost_mean,rank_ic_mean\n"
        "U1,10,top_k,m1,0.005,0.05\n"
    )
    result = load_hardcap_baseline(p)
    assert result.empty


def test_load_hardcap_baseline_with_hardcap(tmp_path):
    p = tmp_path / "lb.csv"
    p.write_text(
        "candidate_pool_version,top_k,selection_policy,model,topk_excess_after_cost_mean,rank_ic_mean\n"
        "U1,10,industry_names_cap,hard_model,0.010,0.08\n"
    )
    result = load_hardcap_baseline(p)
    assert not result.empty
    assert result.iloc[0]["hardcap_after_cost_baseline"] == 0.010


# ═══════════════════════════════════════════════════════════════════════════
# build_gate_table
# ═══════════════════════════════════════════════════════════════════════════


def test_build_gate_table_empty():
    result = build_gate_table(pd.DataFrame(), pd.DataFrame(), tolerance=0.005)
    assert result.empty


def test_build_gate_table_basic():
    lb = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "model": ["m1"],
        "model_type": ["elasticnet"],
        "selection_policy": ["soft_industry_risk_budget"],
        "score_family": [""],
        "label_variant": [""],
        "soft_gamma": [0.2],
        "top_k": [10],
        "topk_excess_after_cost_mean": [0.008],
        "rank_ic_mean": [0.06],
        "topk_minus_nextk_mean": [0.01],
        "concentration_pass_rate": [1.0],
        "strong_down_median_excess": [0.01],
        "strong_up_median_excess": [0.02],
    })
    result = build_gate_table(lb, pd.DataFrame(), tolerance=0.005)
    assert not result.empty
    assert "m8_natural_gate_pass" in result.columns
    assert result.iloc[0]["natural_policy_gate"] == True


def test_build_gate_table_with_hardcap_close():
    lb = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "model": ["soft"],
        "model_type": ["soft"],
        "selection_policy": ["soft_industry_risk_budget"],
        "score_family": [""], "label_variant": [""],
        "soft_gamma": [0.2], "top_k": [10],
        "topk_excess_after_cost_mean": [0.009],
        "rank_ic_mean": [0.06],
        "topk_minus_nextk_mean": [0.01],
        "concentration_pass_rate": [1.0],
        "strong_down_median_excess": [0.01],
        "strong_up_median_excess": [0.02],
    })
    hc = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "top_k": [10],
        "hardcap_after_cost_baseline": [0.010],
        "hardcap_rank_ic_baseline": [0.08],
        "hardcap_source_model": ["hard"],
    })
    result = build_gate_table(lb, hc, tolerance=0.005)
    assert bool(result.iloc[0]["hardcap_closeness_gate"]) is True


def test_build_gate_table_hardcap_delta_fails():
    lb = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "model": ["weak"],
        "model_type": ["soft"],
        "selection_policy": ["soft_industry_risk_budget"],
        "score_family": [""], "label_variant": [""],
        "soft_gamma": [0.2], "top_k": [10],
        "topk_excess_after_cost_mean": [0.002],
        "rank_ic_mean": [0.02],
        "topk_minus_nextk_mean": [0.005],
        "concentration_pass_rate": [0.9],
        "strong_down_median_excess": [-0.05],
        "strong_up_median_excess": [-0.02],
    })
    hc = pd.DataFrame({
        "candidate_pool_version": ["U1"],
        "top_k": [10],
        "hardcap_after_cost_baseline": [0.010],
        "hardcap_rank_ic_baseline": [0.08],
        "hardcap_source_model": ["hard"],
    })
    result = build_gate_table(lb, hc, tolerance=0.001)
    assert bool(result.iloc[0]["hardcap_closeness_gate"]) is False
    assert bool(result.iloc[0]["m8_natural_gate_pass"]) is False
