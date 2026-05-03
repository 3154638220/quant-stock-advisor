"""M8 natural gate 回归测试。

验证 M8 natural（soft industry risk-budget）gate pass rate 的正确性，
使用小型 mock 数据集（5 行业 × 10 只股票 × 3 个月）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_monthly_selection_m8_natural_industry_constraints import (
    M8NaturalRunConfig,
    build_gate_table,
    copy_source_metric_for_optimizer,
    select_soft_industry_risk,
)
from src.pipeline.monthly_concentration import summarize_industry_concentration


# ── mock 数据集：5 行业 × 10 只股票 × 3 个月 ──────────────────────────────

def _mock_scores(
    months: int = 3,
    industries: tuple[str, ...] = ("电子", "计算机", "医药生物", "食品饮料", "银行"),
    stocks_per_industry: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """生成 mock 打分数据。

    行业间分数差异小（0.03/行业），使 soft risk-budget penalty 能实际改变选择。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-31", periods=months, freq="ME")
    rows = []
    for m, date in enumerate(dates):
        market = -0.02 + 0.01 * m
        for i_idx, industry in enumerate(industries):
            for s in range(stocks_per_industry):
                symbol = f"{600000 + i_idx * 100 + s:06d}"
                # 行业间仅有轻微分数差异（0.03/行业），同行业内高分在前
                base_score = 0.50 + 0.03 * i_idx + 0.02 * (stocks_per_industry - s) + rng.normal(0, 0.02)
                forward = -0.03 + 0.008 * i_idx + 0.002 * s + 0.001 * m + rng.normal(0, 0.02)
                rows.append(
                    {
                        "signal_date": date,
                        "candidate_pool_version": "U2_risk_sane",
                        "candidate_pool_pass": True,
                        "candidate_pool_reject_reason": "",
                        "symbol": symbol,
                        "model": "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess",
                        "model_type": "elasticnet_excess",
                        "score": base_score,
                        "rank": i_idx * stocks_per_industry + s + 1,
                        "label_forward_1m_o2o_return": forward,
                        "label_market_ew_o2o_return": market,
                        "label_forward_1m_excess_vs_market": forward - market,
                        "label_forward_1m_industry_neutral_excess": forward - (0.005 * i_idx),
                        "label_future_top_20pct": 1 if s >= stocks_per_industry - 2 else 0,
                        "industry_level1": industry,
                        "industry_level2": "L2",
                        "log_market_cap": 10.0 + s,
                        "risk_flags": "",
                        "is_buyable_tplus1_open": True,
                        "feature_ret_20d": base_score,
                        "feature_ret_60d": base_score / 2.0,
                        "feature_realized_vol_20d": 1.0 / (1.0 + base_score),
                        "feature_amount_20d_log": 10.0 + base_score,
                        "feature_ret_20d_z": base_score,
                        "feature_ret_60d_z": base_score / 2.0,
                        "feature_ret_5d_z": base_score / 3.0,
                        "feature_realized_vol_20d_z": -base_score,
                        "feature_amount_20d_log_z": base_score / 4.0,
                        "feature_turnover_20d_z": base_score / 5.0,
                        "feature_price_position_250d_z": base_score / 6.0,
                        "feature_limit_move_hits_20d_z": -base_score,
                        "next_trade_date": date + pd.offsets.BDay(1),
                    }
                )
    return pd.DataFrame(rows)


# ── 测试：soft industry risk-budget 约束生效 ──────────────────────────────

def _uncapped_topk(part: pd.DataFrame, k: int) -> pd.DataFrame:
    """无约束 TopK 选择（按 score 降序）。"""
    return part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").head(k)


@pytest.mark.parametrize("gamma", [0.1, 0.3, 0.5, 0.8])
def test_soft_industry_risk_reduces_concentration_vs_uncapped(gamma: float):
    """soft risk-budget 约束应比无约束 TopK 降低单行业集中度。"""
    scores = _mock_scores(months=1)
    part = scores[scores["model"] == scores["model"].iloc[0]].copy()

    k = 20
    uncapped = _uncapped_topk(part, k=k)
    selected = select_soft_industry_risk(part, k=k, gamma=gamma)

    assert len(selected) == k, f"应选出 {k} 只，实际选出 {len(selected)} 只"

    uncapped_max_share = uncapped["industry_level1"].value_counts().max() / k
    selected_max_share = selected["industry_level1"].value_counts().max() / k
    uncapped_n_ind = uncapped["industry_level1"].nunique()
    selected_n_ind = selected["industry_level1"].nunique()

    # soft 约束下行业数应 ≥ 无约束
    assert selected_n_ind >= uncapped_n_ind, (
        f"gamma={gamma}: soft({selected_n_ind} industries) vs uncapped({uncapped_n_ind})"
    )
    # soft 约束下最大行业占比应 ≤ 无约束
    assert selected_max_share <= uncapped_max_share + 0.01, (
        f"gamma={gamma}: soft max_share={selected_max_share:.3f} vs uncapped={uncapped_max_share:.3f}"
    )


def test_higher_gamma_produces_more_diversification():
    """更高的 gamma 应产生更分散的选股。"""
    scores = _mock_scores(months=1)
    part = scores[scores["model"] == scores["model"].iloc[0]].copy()
    k = 20

    low = select_soft_industry_risk(part, k=k, gamma=0.1)
    high = select_soft_industry_risk(part, k=k, gamma=0.5)

    low_max_share = low["industry_level1"].value_counts().max() / k
    high_max_share = high["industry_level1"].value_counts().max() / k

    # 更高 gamma 不应增加集中度（可以相等，但不应更集中）
    assert high_max_share <= low_max_share + 0.01, (
        f"gamma=0.5 max_share={high_max_share:.3f} > gamma=0.1 max_share={low_max_share:.3f}"
    )


def test_soft_industry_risk_diversifies_across_all_industries():
    """soft risk-budget 应在多个行业间分散选股（覆盖 ≥ 3 行业）。"""
    scores = _mock_scores(months=1)
    part = scores[scores["model"] == scores["model"].iloc[0]].copy()

    k = 20
    gamma = 0.3
    selected = select_soft_industry_risk(part, k=k, gamma=gamma)

    # 至少覆盖 3 个行业
    n_industries = selected["industry_level1"].nunique()
    assert n_industries >= 3, f"应覆盖至少 3 个行业，实际 {n_industries} 个"


def test_soft_industry_risk_stable_across_months():
    """连续 3 个月截面，每月 soft constraint 均比 uncapped 更分散。"""
    scores = _mock_scores(months=3)
    model_name = scores["model"].iloc[0]
    gamma = 0.3
    k = 20

    for date in scores["signal_date"].unique():
        part = scores[
            (scores["model"] == model_name) & (scores["signal_date"] == date)
        ].copy()
        uncapped = _uncapped_topk(part, k=k)
        selected = select_soft_industry_risk(part, k=k, gamma=gamma)

        uncapped_n = uncapped["industry_level1"].nunique()
        selected_n = selected["industry_level1"].nunique()

        assert selected_n >= uncapped_n, (
            f"日期 {date.date()}：soft({selected_n} ind) < uncapped({uncapped_n} ind)"
        )
        assert selected["industry_level1"].nunique() >= 2, (
            f"日期 {date.date()}：行业数不足"
        )


def test_soft_industry_risk_handles_small_k():
    """小 TopK（如 k=5）时 soft constraint 仍有效（不崩溃、选出正确数量）。"""
    scores = _mock_scores(months=1)
    part = scores[scores["model"] == scores["model"].iloc[0]].copy()

    k = 5
    gamma = 0.5
    selected = select_soft_industry_risk(part, k=k, gamma=gamma)

    assert len(selected) == k
    # 至少覆盖 2 个行业
    assert selected["industry_level1"].nunique() >= 2


# ── 测试：gate pass rate 统计函数输出格式正确 ─────────────────────────────

def _build_mock_leaderboard() -> pd.DataFrame:
    """构建模拟 leaderboard，覆盖 2 个 gamma、2 个 TopK。"""
    rows = []
    for gamma_val in [0.2, 0.4]:
        for k in [20, 30]:
            rows.append(
                {
                    "candidate_pool_version": "U2_risk_sane",
                    "model": f"M5_elasticnet__soft_risk_budget_gamma{str(gamma_val).replace('.', '_')}",
                    "model_type": "soft_industry_risk_budget_optimizer",
                    "selection_policy": "soft_industry_risk_budget",
                    "score_family": "optimizer_compare",
                    "label_variant": "market_excess",
                    "soft_gamma": gamma_val,
                    "top_k": k,
                    "topk_excess_after_cost_mean": 0.01 + gamma_val * 0.02,
                    "rank_ic_mean": 0.08 + gamma_val * 0.02,
                    "topk_minus_nextk_mean": 0.008,
                    "max_industry_share_mean": gamma_val * 0.5,
                    "concentration_pass_rate": 1.0,
                    "industry_count_mean": 10.0,
                    "strong_down_median_excess": -0.01,
                    "strong_up_median_excess": 0.025,
                }
            )
    return pd.DataFrame(rows)


def _build_mock_hardcap_baseline() -> pd.DataFrame:
    """模拟 hardcap baseline。"""
    return pd.DataFrame(
        {
            "candidate_pool_version": ["U2_risk_sane", "U2_risk_sane"],
            "top_k": [20, 30],
            "hardcap_after_cost_baseline": [0.015, 0.018],
            "hardcap_rank_ic_baseline": [0.09, 0.10],
            "hardcap_source_model": ["hardcap_ref", "hardcap_ref"],
        }
    )


def test_build_gate_table_output_format():
    """gate table 输出格式正确：包含必需的 gate 列且 m8_natural_gate_pass 为布尔。"""
    leaderboard = _build_mock_leaderboard()
    hardcap = _build_mock_hardcap_baseline()
    tolerance = 0.005

    gate = build_gate_table(leaderboard, hardcap, tolerance=tolerance)

    required_cols = [
        "m8_natural_gate_pass",
        "natural_policy_gate",
        "hardcap_closeness_gate",
        "rank_gate",
        "spread_gate",
        "concentration_gate",
        "year_regime_gate",
        "cost_gate_10bps",
        "fixed_parameter_gate",
    ]
    for col in required_cols:
        assert col in gate.columns, f"缺少 gate 列: {col}"

    # m8_natural_gate_pass 应为布尔类型
    assert gate["m8_natural_gate_pass"].dtype == bool
    # 不应有空 gate pass 值
    assert gate["m8_natural_gate_pass"].notna().all()


def test_build_gate_table_all_pass_with_good_scores():
    """当所有指标良好时，gate 应全部通过。"""
    leaderboard = _build_mock_leaderboard()
    hardcap = _build_mock_hardcap_baseline()
    tolerance = 0.005

    gate = build_gate_table(leaderboard, hardcap, tolerance=tolerance)

    # 所有行应通过 gate
    assert gate["m8_natural_gate_pass"].all(), (
        f"预期全部通过，实际:\n{gate[['model', 'top_k', 'm8_natural_gate_pass']].to_string()}"
    )


def test_build_gate_table_fails_on_negative_rank_ic():
    """Rank IC 为负时 rank_gate 应为 False。"""
    leaderboard = _build_mock_leaderboard()
    # 将第一条的 rank IC 设为负
    leaderboard.loc[0, "rank_ic_mean"] = -0.02
    hardcap = _build_mock_hardcap_baseline()

    gate = build_gate_table(leaderboard, hardcap, tolerance=0.005)

    assert not gate.loc[0, "rank_gate"]
    assert not gate.loc[0, "m8_natural_gate_pass"]


def test_build_gate_table_fails_on_excess_below_zero():
    """after-cost excess ≤ 0 时 cost_gate_10bps 应为 False。"""
    leaderboard = _build_mock_leaderboard()
    leaderboard.loc[1, "topk_excess_after_cost_mean"] = -0.005
    hardcap = _build_mock_hardcap_baseline()

    gate = build_gate_table(leaderboard, hardcap, tolerance=0.005)

    assert not gate.loc[1, "cost_gate_10bps"]
    assert not gate.loc[1, "m8_natural_gate_pass"]


def test_build_gate_table_fails_on_hardcap_delta_below_tolerance():
    """hardcap delta < -tolerance 时 hardcap_closeness_gate 应为 False。"""
    leaderboard = _build_mock_leaderboard()
    # hardcap baseline=0.015，当前=0.014（delta=-0.001 > -0.005 仍 pass）
    # 调低当前值使 delta=-0.02 < -0.005
    leaderboard.loc[0, "topk_excess_after_cost_mean"] = -0.005
    hardcap = _build_mock_hardcap_baseline()
    tolerance = 0.005

    gate = build_gate_table(leaderboard, hardcap, tolerance=tolerance)

    assert not gate.loc[0, "hardcap_closeness_gate"]
    assert not gate.loc[0, "m8_natural_gate_pass"]


def test_build_gate_table_no_hardcap_fallback():
    """无 hardcap baseline 时，hardcap_closeness_gate 应为 True（fallback）。"""
    leaderboard = _build_mock_leaderboard()
    empty_hardcap = pd.DataFrame()
    tolerance = 0.005

    gate = build_gate_table(leaderboard, empty_hardcap, tolerance=tolerance)

    # 无 hardcap 对比时 hardcap_closeness_gate 应 fallback 为 True
    assert gate["hardcap_closeness_gate"].all()


def test_build_gate_table_handles_empty_leaderboard():
    """空 leaderboard 应返回空 DataFrame。"""
    gate = build_gate_table(pd.DataFrame(), pd.DataFrame(), tolerance=0.005)
    assert gate.empty


# ── 测试：copy_source_metric_for_optimizer ─────────────────────────────────

def test_copy_source_metric_remaps_model_names():
    """source model 的指标应正确映射到 optimizer model 名。"""
    metric = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "candidate_pool_version": ["U2_risk_sane", "U2_risk_sane"],
            "model": ["base_model", "base_model"],
            "rank_ic": [0.12, 0.11],
        }
    )
    optimizer_monthly = pd.DataFrame(
        {
            "candidate_pool_version": ["U2_risk_sane"],
            "model": ["base_model__soft_risk_budget_gamma0_2"],
            "source_model": ["base_model"],
        }
    )

    copied = copy_source_metric_for_optimizer(metric, optimizer_monthly)

    assert len(copied) == 2
    assert (copied["model"] == "base_model__soft_risk_budget_gamma0_2").all()
    assert copied["rank_ic"].tolist() == [0.12, 0.11]


# ── 测试：industry concentration 汇总输出格式 ──────────────────────────────

def test_summarize_industry_concentration_output_format():
    """industry concentration 汇总包含必需的统计列。"""
    scores = _mock_scores(months=3)
    model_name = scores["model"].iloc[0]
    gamma = 0.3
    k = 20

    # 模拟 holdings DataFrame
    holdings_rows = []
    for date in scores["signal_date"].unique():
        part = scores[
            (scores["model"] == model_name) & (scores["signal_date"] == date)
        ].copy()
        selected = select_soft_industry_risk(part, k=k, gamma=gamma)
        for rank_i, (_, row) in enumerate(selected.iterrows()):
            holdings_rows.append(
                {
                    "signal_date": date,
                    "candidate_pool_version": row["candidate_pool_version"],
                    "base_model": row["model"],
                    "model": f"{row['model']}__soft_risk_budget_gamma0_3",
                    "top_k": k,
                    "selection_policy": "soft_industry_risk_budget",
                    "max_industry_names": -1,
                    "industry_level1": row["industry_level1"],
                    "symbol": row["symbol"],
                    "selected_rank": rank_i + 1,
                }
            )

    holdings = pd.DataFrame(holdings_rows)
    concentration = summarize_industry_concentration(holdings)

    required_cols = [
        "signal_date",
        "candidate_pool_version",
        "base_model",
        "model",
        "top_k",
        "selection_policy",
        "max_industry_names",
        "selected_count",
        "industry_count",
        "max_industry_count",
        "max_industry_share",
        "concentration_threshold",
        "concentration_pass",
    ]
    for col in required_cols:
        assert col in concentration.columns, f"缺少列: {col}"

    # max_industry_share 应在 [0, 1]
    assert concentration["max_industry_share"].between(0, 1).all()
    # concentration_pass 应为布尔
    assert concentration["concentration_pass"].dtype == bool
    # 每个月都应有记录
    assert concentration["signal_date"].nunique() == 3


def test_summarize_industry_concentration_pass_when_under_threshold():
    """单行业占比 ≤ 阈值时 concentration_pass 应为 True。"""
    scores = _mock_scores(months=1)
    model_name = scores["model"].iloc[0]
    gamma = 0.6  # 较高 gamma 确保软约束足够分散
    k = 20

    part = scores[
        (scores["model"] == model_name) & (scores["signal_date"] == scores["signal_date"].iloc[0])
    ].copy()
    selected = select_soft_industry_risk(part, k=k, gamma=gamma)

    holdings_rows = []
    for rank_i, (_, row) in enumerate(selected.iterrows()):
        holdings_rows.append(
            {
                "signal_date": row["signal_date"],
                "candidate_pool_version": row["candidate_pool_version"],
                "base_model": row["model"],
                "model": f"{row['model']}__soft",
                "top_k": k,
                "selection_policy": "soft_industry_risk_budget",
                "max_industry_names": -1,
                "industry_level1": row["industry_level1"],
                "symbol": row["symbol"],
                "selected_rank": rank_i + 1,
            }
        )

    concentration = summarize_industry_concentration(pd.DataFrame(holdings_rows))

    assert len(concentration) == 1
    # 高 gamma 下应通过集中度检查；若未通过，需评审 gamma 或数据
    max_share = concentration.loc[0, "max_industry_share"]
    threshold = concentration.loc[0, "concentration_threshold"]
    assert concentration.loc[0, "concentration_pass"], (
        f"单行业占比 {max_share:.3f} 应 ≤ {threshold:.3f}（gamma={gamma}）"
    )
