from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.run_p1_tree_groups import _parse_group_list, resolve_label_horizons_and_weights
from src.models.xtree.p1_workflow import (
    DEFAULT_P1_BENCHMARK_KEY_YEARS,
    FUND_FLOW_TREE_FEATURES,
    SHAREHOLDER_TREE_FEATURES,
    WEEKLY_KDJ_INTERACTION_TREE_FEATURES,
    WEEKLY_KDJ_TREE_FEATURES,
    attach_weekly_kdj_interaction_features,
    baseline_tree_feature_names,
    build_daily_proxy_first_leaderboard,
    build_group_comparison_table,
    build_investable_period_return_panel,
    build_market_state_table_from_daily,
    build_p1_monthly_investable_label,
    build_p1_daily_proxy_first_report,
    build_p1_training_label,
    build_p1_tree_output_stem,
    build_p1_tree_research_config_id,
    build_tree_daily_backtest_like_proxy_detail,
    build_tree_direction_diagnostic_table,
    build_tree_light_proxy_detail,
    build_tree_score_weight_matrix,
    build_tree_topk_boundary_diagnostic,
    build_tree_turnover_aware_proxy_detail,
    classify_daily_proxy_first_decision,
    p1_tree_feature_groups,
    resolve_available_feature_names,
    select_rebalance_dates,
    summarize_p1_full_backtest_payload,
    summarize_p1_label_diagnostics,
    summarize_tree_daily_backtest_like_proxy,
    summarize_tree_daily_proxy_state_slices,
    summarize_tree_group_result,
    summarize_tree_score_direction,
)


def test_p1_tree_feature_groups_match_plan_shape():
    groups = p1_tree_feature_groups()
    assert list(groups) == ["G0", "G1", "G2", "G3", "G4"]
    assert tuple(groups["G0"]) == baseline_tree_feature_names()
    assert all(name in groups["G1"] for name in WEEKLY_KDJ_TREE_FEATURES)
    assert all(name in groups["G2"] for name in FUND_FLOW_TREE_FEATURES)
    assert all(name in groups["G3"] for name in SHAREHOLDER_TREE_FEATURES)
    assert all(name in groups["G4"] for name in WEEKLY_KDJ_TREE_FEATURES + FUND_FLOW_TREE_FEATURES)
    assert all(name not in groups["G0"] for name in WEEKLY_KDJ_TREE_FEATURES)


def test_p1_tree_feature_groups_can_add_weekly_kdj_interaction_branch():
    groups = p1_tree_feature_groups(include_interaction_groups=True)
    assert list(groups) == ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]
    assert all(name in groups["G5"] for name in WEEKLY_KDJ_TREE_FEATURES)
    assert all(name in groups["G5"] for name in WEEKLY_KDJ_INTERACTION_TREE_FEATURES)
    assert all(name in groups["G6"] for name in WEEKLY_KDJ_INTERACTION_TREE_FEATURES)
    assert all(name not in groups["G6"] for name in WEEKLY_KDJ_TREE_FEATURES)


def test_attach_weekly_kdj_interaction_features_builds_gates_by_date():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 4),
            "symbol": ["000001", "000002", "000003", "000004"],
            "weekly_kdj_j": [-20.0, -5.0, 10.0, 30.0],
            "weekly_kdj_oversold_depth": [15.0, 0.0, 0.0, 0.0],
            "vol_to_turnover": [1.0, 2.0, 3.0, 4.0],
            "turnover_roll_mean": [0.2, 0.4, 0.8, 1.0],
            "momentum": [-0.1, 0.0, 0.2, 0.4],
            "realized_vol": [0.1, 0.2, 0.8, 1.0],
        }
    )
    out = attach_weekly_kdj_interaction_features(
        panel,
        low_turnover_quantile=0.5,
        weak_momentum_quantile=0.5,
        low_vol_quantile=0.5,
    )
    assert all(name in out.columns for name in WEEKLY_KDJ_INTERACTION_TREE_FEATURES)
    assert out["wk_j_contrarian_low_turnover_gate"].tolist() == [20.0, 5.0, 0.0, 0.0]
    assert out["wk_j_contrarian_weak_momentum_gate"].tolist() == [20.0, 5.0, 0.0, 0.0]
    assert out["wk_oversold_depth_low_vol_gate"].tolist() == [15.0, 0.0, 0.0, 0.0]
    assert out["wk_j_contrarian_x_vol_to_turnover"].notna().sum() == 4


def test_resolve_available_feature_names_drops_missing_and_all_nan():
    panel = pd.DataFrame(
        {
            "momentum": [1.0, 2.0],
            "weekly_kdj_j": [float("nan"), float("nan")],
            "main_inflow_z_5d": [0.1, -0.2],
        }
    )
    available, missing = resolve_available_feature_names(
        panel,
        ["momentum", "weekly_kdj_j", "main_inflow_z_5d", "holder_change_rate_z"],
    )
    assert available == ["momentum", "main_inflow_z_5d"]
    assert missing == ["weekly_kdj_j", "holder_change_rate_z"]


def test_select_rebalance_dates_monthly_uses_last_trade_date():
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-31",
            "2024-02-05",
            "2024-02-29",
            "2024-03-08",
        ]
    )
    out = select_rebalance_dates(dates, rebalance_rule="M")
    assert out["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-08",
    ]
    assert out["period"].tolist() == ["2024-01", "2024-02", "2024-03"]


def test_build_tree_light_proxy_detail_selects_topk_by_period():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-02-28",
                    "2024-02-28",
                    "2024-02-29",
                    "2024-02-29",
                ]
            ),
            "symbol": ["000001", "000002"] * 4,
            "tree_score": [0.1, 0.2, 0.9, 0.3, 0.4, 0.2, 0.8, 0.1],
            "forward_ret_5d": [0.01, 0.02, 0.08, 0.01, 0.03, 0.00, 0.05, -0.02],
        }
    )
    out = build_tree_light_proxy_detail(
        df,
        score_col="tree_score",
        proxy_return_col="forward_ret_5d",
        rebalance_rule="M",
        top_k=1,
        scenario="G1",
    )
    assert out["period"].tolist() == ["2024-01", "2024-02"]
    assert out["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-02-29"]
    assert out["strategy_return"].tolist() == [0.08, 0.05]
    assert out["benchmark_return"].tolist() == pytest.approx([0.045, 0.015])
    assert out["beat_benchmark"].tolist() == [True, True]


def test_build_tree_turnover_aware_proxy_keeps_prev_holdings_under_cap():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-31"] * 4
                + ["2024-02-29"] * 4
            ),
            "symbol": ["000001", "000002", "000003", "000004"] * 2,
            "tree_score": [0.9, 0.8, 0.1, 0.0, 0.2, 0.1, 0.95, 0.94],
            "forward_ret_5d": [0.05, 0.04, 0.00, 0.00, 0.01, 0.00, -0.10, -0.10],
        }
    )

    out = build_tree_turnover_aware_proxy_detail(
        df,
        score_col="tree_score",
        proxy_return_col="forward_ret_5d",
        rebalance_rule="M",
        top_k=2,
        max_turnover=0.5,
        scenario="G1",
    )

    assert out["period"].tolist() == ["2024-01", "2024-02"]
    assert out.loc[1, "retained_from_prev_count"] == 1
    assert out.loc[1, "turnover_half_l1"] == pytest.approx(0.5)
    assert out.loc[1, "strategy_return"] == pytest.approx((-0.10 + 0.01) / 2.0)


def test_build_tree_score_weight_matrix_applies_turnover_cap():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 4 + ["2024-02-29"] * 4),
            "symbol": ["000001", "000002", "000003", "000004"] * 2,
            "tree_score": [0.9, 0.8, 0.1, 0.0, 0.2, 0.1, 0.95, 0.94],
        }
    )

    weights, diag = build_tree_score_weight_matrix(
        df,
        score_col="tree_score",
        rebalance_rule="M",
        top_k=2,
        max_turnover=0.5,
    )

    assert weights.index.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-02-29"]
    assert weights.loc[pd.Timestamp("2024-01-31"), ["000001", "000002"]].tolist() == [0.5, 0.5]
    assert weights.loc[pd.Timestamp("2024-02-29"), "000001"] == pytest.approx(0.5)
    assert weights.loc[pd.Timestamp("2024-02-29"), "000003"] == pytest.approx(0.5)
    assert diag.loc[1, "turnover_half_l1"] == pytest.approx(0.5)


def test_daily_backtest_like_proxy_uses_daily_open_returns_and_market_benchmark():
    scored = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 3),
            "symbol": ["000001", "000002", "000003"],
            "tree_score": [0.9, 0.2, 0.1],
        }
    )
    daily = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"] * 3),
            "symbol": ["000001"] * 4 + ["000002"] * 4 + ["000003"] * 4,
            "open": [10.0, 10.0, 10.5, 11.025, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0],
            "close": [10.0, 10.0, 10.5, 11.025, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0],
        }
    )

    detail, meta = build_tree_daily_backtest_like_proxy_detail(
        scored,
        daily,
        score_col="tree_score",
        rebalance_rule="M",
        top_k=1,
        max_turnover=1.0,
        scenario="G0",
        cost_params=None,
        market_ew_min_days=1,
    )
    summary = summarize_tree_daily_backtest_like_proxy(detail)

    row = detail.loc[detail["trade_date"] == pd.Timestamp("2024-02-01")].iloc[0]
    assert row["strategy_return"] == pytest.approx(0.05)
    assert row["benchmark_return"] == pytest.approx(0.05 / 3.0)
    assert row["excess_return"] == pytest.approx(0.05 - 0.05 / 3.0)
    assert meta["n_rebalances"] == 1
    assert summary["periods_per_year"] == 252.0
    assert summary["n_periods"] >= 1


def test_topk_boundary_diagnostic_compares_topk_and_next_bucket():
    dates = pd.bdate_range("2024-01-25", "2024-03-05")
    daily_rows = []
    score_rows = []
    for i, d in enumerate(dates):
        for sym, step, score in [
            ("000001", 0.01, 0.9),
            ("000002", 0.0, 0.5),
            ("000003", -0.005, 0.1),
        ]:
            px = 10.0 * ((1.0 + step) ** i)
            daily_rows.append({"trade_date": d, "symbol": sym, "open": px, "close": px})
            score_rows.append({"trade_date": d, "symbol": sym, "tree_score": score})

    detail, summary = build_tree_topk_boundary_diagnostic(
        pd.DataFrame(score_rows),
        pd.DataFrame(daily_rows),
        score_col="tree_score",
        rebalance_rule="M",
        top_k=1,
        scenario="G0",
    )

    assert not detail.empty
    assert detail.loc[0, "topk_mean_return"] > detail.loc[0, "next_bucket_mean_return"]
    assert summary["topk_boundary_topk_minus_next_mean_return"] > 0.0
    assert summary["topk_boundary_periods"] == len(detail)


def test_daily_proxy_state_slices_include_strong_up_and_high_vol_states():
    dates = pd.bdate_range("2024-01-02", "2024-04-30")
    daily_rows = []
    for i, d in enumerate(dates):
        for sym, mult in [("000001", 1.0), ("000002", 0.8), ("000003", 1.2)]:
            month_step = {1: 0.002, 2: -0.002, 3: 0.006, 4: 0.0}[d.month] * mult
            px = 10.0 * ((1.0 + month_step) ** i)
            daily_rows.append({"trade_date": d, "symbol": sym, "open": px, "close": px})
    detail_rows = []
    for d in dates:
        detail_rows.append(
            {
                "period": d.strftime("%Y-%m-%d"),
                "trade_date": d,
                "strategy_return": 0.002 if d.month in {1, 3} else -0.001,
                "benchmark_return": 0.001 if d.month in {1, 3} else -0.0015,
                "excess_return": 0.001,
                "group": "G0",
                "proxy_variant": "daily_backtest_like",
            }
        )
    monthly, summary = summarize_tree_daily_proxy_state_slices(
        pd.DataFrame(detail_rows),
        pd.DataFrame(daily_rows),
    )

    assert not monthly.empty
    assert {"return_state", "vol_state", "breadth_state"} <= set(monthly.columns)
    assert not summary.empty
    assert "strong_up" in set(summary["state"])
    assert "high_vol" in set(summary["state"])


def test_market_state_table_from_daily_labels_monthly_states():
    dates = pd.bdate_range("2024-01-02", "2024-04-30")
    daily_rows = []
    for i, d in enumerate(dates):
        step = {1: 0.001, 2: -0.002, 3: 0.004, 4: 0.0}[d.month]
        for sym in ["000001", "000002"]:
            px = 10.0 * ((1.0 + step) ** i)
            daily_rows.append({"trade_date": d, "symbol": sym, "open": px, "close": px})
    out = build_market_state_table_from_daily(pd.DataFrame(daily_rows))
    assert not out.empty
    assert "strong_up" in set(out["return_state"])
    assert "strong_down" in set(out["return_state"])


def test_summarize_tree_group_result_uses_frequency_aware_annualization():
    detail = pd.DataFrame(
        {
            "period": ["2024-01", "2024-02"],
            "strategy_return": [0.10, 0.00],
            "benchmark_return": [0.02, 0.01],
            "scenario": ["G0", "G0"],
            "excess_return": [0.08, -0.01],
            "benchmark_up": [True, True],
            "strategy_up": [True, False],
            "beat_benchmark": [True, False],
        }
    )
    summary = summarize_tree_group_result(detail, rebalance_rule="M")
    assert summary["periods_per_year"] == 12.0
    assert summary["n_periods"] == 2
    assert math.isfinite(summary["strategy_annualized_return"])
    assert math.isfinite(summary["annualized_excess_vs_market"])


def test_build_group_comparison_table_adds_deltas_vs_g0():
    out = build_group_comparison_table(
        [
            {"group": "G0", "val_rank_ic": 0.03, "annualized_excess_vs_market": 0.01},
            {"group": "G1", "val_rank_ic": 0.05, "annualized_excess_vs_market": 0.03},
        ]
    )
    assert out.loc[out["group"] == "G0", "delta_vs_baseline_val_rank_ic"].iloc[0] == 0.0
    assert out.loc[out["group"] == "G1", "delta_vs_baseline_val_rank_ic"].iloc[0] == pytest.approx(0.02)
    assert out.loc[out["group"] == "G1", "delta_vs_baseline_proxy_excess"].iloc[0] == pytest.approx(0.02)


def test_summarize_tree_score_direction_matches_inference_auto_flip():
    out = summarize_tree_score_direction({"val_rank_ic": -0.12, "train_rank_ic": 0.02})
    assert out["tree_score_auto_flipped"] is True
    assert out["effective_val_rank_ic"] == pytest.approx(0.12)


def test_summarize_p1_label_diagnostics_checks_target_proxy_direction():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 4 + ["2024-02-29"] * 4),
            "forward_ret_fused": [-0.5, -0.1, 0.2, 0.5, -0.4, -0.2, 0.1, 0.4],
            "forward_ret_5d": [-0.03, -0.01, 0.02, 0.04, -0.02, -0.01, 0.01, 0.03],
        }
    )
    out = summarize_p1_label_diagnostics(
        panel,
        target_column="forward_ret_fused",
        label_columns=["forward_ret_5d", "forward_ret_10d"],
        label_weights=[0.6, 0.4],
        proxy_return_col="forward_ret_5d",
    )
    assert out["label_diagnostic_status"] == "ok"
    assert out["target_higher_is_better"] is True
    assert out["target_proxy_rank_corr_mean"] == pytest.approx(1.0)
    assert out["target_proxy_rank_corr_negative_rate"] == pytest.approx(0.0)


def test_build_p1_training_label_supports_market_relative_mode():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
            "symbol": ["000001", "000002", "000003"] * 2,
            "forward_ret_5d": [0.01, 0.03, 0.05, -0.02, 0.01, 0.04],
            "forward_ret_10d": [0.02, 0.04, 0.06, -0.03, 0.02, 0.05],
        }
    )
    out, target, meta = build_p1_training_label(
        panel,
        label_columns=["forward_ret_5d", "forward_ret_10d"],
        label_weights=[0.5, 0.5],
        label_mode="market_relative",
    )
    assert target == "forward_ret_fused"
    assert meta["label_scope"] == "market_relative"
    assert meta["label_market_proxy"] == "same_date_cross_section_equal_weight"
    assert out.groupby("trade_date")["forward_ret_fused"].mean().abs().max() == pytest.approx(0.0)


def test_build_p1_training_label_supports_top_bucket_rank_fusion_mode():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 5),
            "symbol": ["000001", "000002", "000003", "000004", "000005"],
            "forward_ret_5d": [-0.04, -0.02, 0.0, 0.02, 0.04],
        }
    )

    out, target, meta = build_p1_training_label(
        panel,
        label_columns=["forward_ret_5d"],
        label_weights=[1.0],
        label_mode="top_bucket_rank_fusion",
    )

    assert target == "forward_ret_fused"
    assert meta["label_scope"] == "cross_section_top_bottom_bucket"
    assert meta["label_top_bucket_quantile"] == pytest.approx(0.2)
    assert out.sort_values("symbol")["forward_ret_fused"].tolist() == pytest.approx([-1.0, 0.0, 0.0, 0.0, 1.0])


def test_build_p1_training_label_supports_up_capture_market_relative_mode():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
            "symbol": ["000001", "000002", "000003"] * 2,
            "forward_ret_5d": [0.01, 0.03, 0.05, -0.05, -0.03, -0.01],
        }
    )

    out, target, meta = build_p1_training_label(
        panel,
        label_columns=["forward_ret_5d"],
        label_weights=[1.0],
        label_mode="up_capture_market_relative",
    )

    jan = out[out["trade_date"] == pd.Timestamp("2024-01-31")]
    feb = out[out["trade_date"] == pd.Timestamp("2024-02-29")]
    assert target == "forward_ret_fused"
    assert meta["label_scope"] == "up_capture_market_relative"
    assert meta["label_market_proxy"] == "same_date_cross_section_equal_weight"
    assert meta["label_up_capture_multiplier"] == pytest.approx(2.0)
    assert jan["forward_ret_fused"].tolist() == pytest.approx([-0.04, 0.0, 0.04])
    assert feb["forward_ret_fused"].tolist() == pytest.approx([-0.02, 0.0, 0.02])


def test_build_p1_training_label_rank_fusion_preserves_direction():
    panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31"] * 3),
            "forward_ret_5d": [-0.01, 0.02, 0.05],
        }
    )
    out, target, meta = build_p1_training_label(
        panel,
        label_columns=["forward_ret_5d"],
        label_weights=[1.0],
        label_mode="rank_fusion",
    )
    assert target == "forward_ret_fused"
    assert meta["label_scope"] == "cross_section_relative"
    assert out["forward_ret_fused"].tolist() == pytest.approx([-1.0 / 6.0, 1.0 / 6.0, 0.5])


def test_build_monthly_investable_label_uses_rebalance_period_open_returns():
    dates = pd.bdate_range("2024-01-25", "2024-03-05")
    rows = []
    panel_rows = []
    for i, d in enumerate(dates):
        for sym, base, daily_step in [("000001", 10.0, 0.01), ("000002", 20.0, 0.0)]:
            rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "open": base * ((1.0 + daily_step) ** i),
                    "close": base * ((1.0 + daily_step) ** i),
                }
            )
            panel_rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "forward_ret_5d": 0.0,
                }
            )
    daily = pd.DataFrame(rows)
    panel = pd.DataFrame(panel_rows)

    out, target, meta = build_p1_monthly_investable_label(
        panel,
        daily,
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_mode="monthly_investable",
    )

    assert target == "forward_ret_investable"
    assert meta["label_rebalance_rule"] == "M"
    assert out["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-01-31", "2024-02-29"]
    jan_a = out[(out["trade_date"] == pd.Timestamp("2024-01-31")) & (out["symbol"] == "000001")].iloc[0]
    n_open_intervals = int(((dates > pd.Timestamp("2024-01-31")) & (dates <= pd.Timestamp("2024-02-29"))).sum())
    assert jan_a["forward_ret_investable"] == pytest.approx((1.01 ** n_open_intervals) - 1.0)
    jan_b = out[(out["trade_date"] == pd.Timestamp("2024-01-31")) & (out["symbol"] == "000002")].iloc[0]
    assert jan_b["forward_ret_investable"] == pytest.approx(0.0)


def test_monthly_investable_market_relative_label_demeans_rebalance_cross_section():
    dates = pd.bdate_range("2024-01-25", "2024-03-05")
    daily_rows = []
    panel_rows = []
    for i, d in enumerate(dates):
        for sym, step in [("000001", 0.01), ("000002", 0.0)]:
            daily_rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "open": 10.0 * ((1.0 + step) ** i),
                    "close": 10.0 * ((1.0 + step) ** i),
                }
            )
            panel_rows.append({"trade_date": d, "symbol": sym})
    out, _, meta = build_p1_monthly_investable_label(
        pd.DataFrame(panel_rows),
        pd.DataFrame(daily_rows),
        rebalance_rule="M",
        label_mode="monthly_investable_market_relative",
    )
    assert meta["label_market_proxy"] == "same_rebalance_date_cross_section_equal_weight"
    assert out.groupby("trade_date")["forward_ret_investable"].mean().abs().max() == pytest.approx(0.0)


def test_monthly_investable_up_capture_market_relative_label_weights_up_periods():
    dates = pd.bdate_range("2024-01-25", "2024-03-29")
    daily_rows = []
    panel_rows = []
    # Jan signal to Feb signal has positive market return; Feb signal to Mar signal is negative.
    prices = {"000001": 10.0, "000002": 10.0}
    for d in dates:
        if pd.Timestamp("2024-01-31") < d <= pd.Timestamp("2024-02-29"):
            prices["000001"] *= 1.01
        elif pd.Timestamp("2024-02-29") < d <= pd.Timestamp("2024-03-29"):
            prices["000001"] *= 0.98
        for sym, price in prices.items():
            daily_rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "open": price,
                    "close": price,
                }
            )
            panel_rows.append({"trade_date": d, "symbol": sym})

    out, _, meta = build_p1_monthly_investable_label(
        pd.DataFrame(panel_rows),
        pd.DataFrame(daily_rows),
        rebalance_rule="M",
        label_mode="monthly_investable_up_capture_market_relative",
    )

    assert meta["label_scope"] == "monthly_investable_up_capture_market_relative"
    assert meta["label_market_proxy"] == "same_rebalance_date_cross_section_equal_weight"
    assert meta["label_up_capture_multiplier"] == pytest.approx(2.0)
    jan = out[out["trade_date"] == pd.Timestamp("2024-01-31")].sort_values("symbol")
    feb = out[out["trade_date"] == pd.Timestamp("2024-02-29")].sort_values("symbol")
    jan_spread = jan["forward_ret_investable"].iloc[0] - jan["forward_ret_investable"].iloc[1]
    feb_spread = feb["forward_ret_investable"].iloc[0] - feb["forward_ret_investable"].iloc[1]
    assert jan_spread > 0
    assert feb_spread < 0
    assert abs(jan["forward_ret_investable"].sum()) == pytest.approx(0.0)
    assert abs(feb["forward_ret_investable"].sum()) == pytest.approx(0.0)


def test_build_investable_period_return_panel_requires_next_rebalance():
    dates = pd.bdate_range("2024-01-25", "2024-02-05")
    daily = pd.DataFrame(
        {
            "trade_date": list(dates) * 2,
            "symbol": ["000001"] * len(dates) + ["000002"] * len(dates),
            "open": [10.0 + i for i in range(len(dates))] + [20.0] * len(dates),
            "close": [10.0 + i for i in range(len(dates))] + [20.0] * len(dates),
        }
    )
    panel = daily[["trade_date", "symbol"]].copy()
    out = build_investable_period_return_panel(panel, daily, rebalance_rule="M")
    assert out["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-01-31"]


def test_build_tree_direction_diagnostic_table_flags_raw_negative_ic():
    out = build_tree_direction_diagnostic_table(
        [
            {
                "group": "G0",
                "val_rank_ic": -0.03,
                "train_rank_ic": 0.01,
                "tree_score_auto_flipped": True,
                "effective_val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.1,
            },
            {
                "group": "G1",
                "val_rank_ic": 0.02,
                "train_rank_ic": 0.03,
                "tree_score_auto_flipped": False,
                "effective_val_rank_ic": 0.02,
                "annualized_excess_vs_market": 0.2,
            },
        ]
    )
    assert bool(out.loc[out["group"] == "G0", "needs_direction_diagnosis"].iloc[0]) is True
    assert bool(out.loc[out["group"] == "G1", "needs_direction_diagnosis"].iloc[0]) is False


def test_build_group_comparison_table_prefers_effective_rank_ic_when_present():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": -0.10,
                "effective_val_rank_ic": 0.10,
                "annualized_excess_vs_market": 0.01,
            },
            {
                "group": "G6",
                "val_rank_ic": -0.12,
                "effective_val_rank_ic": 0.12,
                "annualized_excess_vs_market": 0.02,
            },
        ]
    )
    assert out.loc[out["group"] == "G6", "delta_vs_baseline_val_rank_ic"].iloc[0] == pytest.approx(0.02)


def test_build_group_comparison_table_adds_full_backtest_delta_when_present():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.01,
                "full_backtest_annualized_excess_vs_market": -0.02,
                "rolling_oos_median_ann_return": -0.05,
                "slice_oos_median_ann_return": -0.04,
                "yearly_excess_median_vs_market": -0.03,
                "key_year_excess_mean_vs_market": -0.02,
            },
            {
                "group": "G4",
                "val_rank_ic": 0.04,
                "annualized_excess_vs_market": 0.015,
                "full_backtest_annualized_excess_vs_market": 0.01,
                "rolling_oos_median_ann_return": -0.01,
                "slice_oos_median_ann_return": -0.03,
                "yearly_excess_median_vs_market": 0.00,
                "key_year_excess_mean_vs_market": 0.01,
            },
        ]
    )
    assert out.loc[out["group"] == "G4", "delta_vs_baseline_full_backtest_excess"].iloc[0] == pytest.approx(0.03)
    assert out.loc[out["group"] == "G4", "delta_vs_baseline_rolling_oos_ann_return"].iloc[0] == pytest.approx(0.04)
    assert out.loc[out["group"] == "G4", "delta_vs_baseline_key_year_excess_mean"].iloc[0] == pytest.approx(0.03)
    assert bool(out.loc[out["group"] == "G4", "pass_p1_promotion_gate"].iloc[0]) is True


def test_build_group_comparison_table_adds_daily_proxy_admission_gate():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.10,
                "daily_bt_like_proxy_annualized_excess_vs_market": -0.01,
            },
            {
                "group": "G1",
                "val_rank_ic": 0.04,
                "annualized_excess_vs_market": 0.15,
                "daily_bt_like_proxy_annualized_excess_vs_market": 0.002,
            },
        ]
    )
    row = out.loc[out["group"] == "G1"].iloc[0]
    assert row["delta_vs_baseline_daily_bt_like_proxy_excess"] == pytest.approx(0.012)
    assert bool(row["pass_p1_daily_proxy_admission_gate"]) is True
    assert bool(out.loc[out["group"] == "G0", "pass_p1_daily_proxy_admission_gate"].iloc[0]) is False
    assert row["primary_result_type"] == "daily_bt_like_proxy"
    assert row["legacy_proxy_decision_role"] == "diagnostic_only"


def test_build_group_comparison_table_respects_daily_proxy_threshold_column():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.10,
                "daily_bt_like_proxy_annualized_excess_vs_market": -0.005,
                "daily_proxy_admission_threshold": -0.01,
            },
            {
                "group": "G1",
                "val_rank_ic": 0.04,
                "annualized_excess_vs_market": 0.15,
                "daily_bt_like_proxy_annualized_excess_vs_market": -0.02,
                "daily_proxy_admission_threshold": -0.01,
            },
        ]
    )

    assert bool(out.loc[out["group"] == "G0", "pass_p1_daily_proxy_admission_gate"].iloc[0]) is True
    assert bool(out.loc[out["group"] == "G1", "pass_p1_daily_proxy_admission_gate"].iloc[0]) is False


def test_classify_daily_proxy_first_decision_uses_three_stage_gate():
    reject = classify_daily_proxy_first_decision(-0.001, admission_threshold=0.0, full_backtest_threshold=0.03)
    assert reject["daily_proxy_first_status"] == "reject"
    assert reject["pass_p1_daily_proxy_admission_gate"] is False
    gray = classify_daily_proxy_first_decision(0.01, admission_threshold=0.0, full_backtest_threshold=0.03)
    assert gray["daily_proxy_first_status"] == "gray_zone"
    assert gray["pass_p1_daily_proxy_admission_gate"] is True
    assert gray["pass_p1_daily_proxy_full_backtest_gate"] is False
    candidate = classify_daily_proxy_first_decision(0.031, admission_threshold=0.0, full_backtest_threshold=0.03)
    assert candidate["daily_proxy_first_status"] == "full_backtest_candidate"
    assert candidate["pass_p1_daily_proxy_full_backtest_gate"] is True
    missing = classify_daily_proxy_first_decision(float("nan"))
    assert missing["daily_proxy_first_status"] == "no_daily_proxy"


def test_build_group_comparison_table_marks_daily_proxy_gray_zone():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.10,
                "daily_bt_like_proxy_annualized_excess_vs_market": 0.01,
                "daily_proxy_admission_threshold": 0.0,
                "daily_proxy_full_backtest_threshold": 0.03,
            }
        ]
    )
    row = out.iloc[0]
    assert row["daily_proxy_first_status"] == "gray_zone"
    assert bool(row["pass_p1_daily_proxy_admission_gate"]) is True
    assert bool(row["pass_p1_daily_proxy_full_backtest_gate"]) is False
    assert row["daily_proxy_safety_margin_to_full_backtest"] == pytest.approx(-0.02)


def test_build_daily_proxy_first_leaderboard_sorts_candidates_first():
    summary = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.03,
                "annualized_excess_vs_market": 0.10,
                "daily_bt_like_proxy_annualized_excess_vs_market": -0.01,
                "daily_proxy_admission_threshold": 0.0,
                "daily_proxy_full_backtest_threshold": 0.03,
            },
            {
                "group": "G1",
                "val_rank_ic": 0.04,
                "annualized_excess_vs_market": 0.15,
                "daily_bt_like_proxy_annualized_excess_vs_market": 0.04,
                "daily_proxy_admission_threshold": 0.0,
                "daily_proxy_full_backtest_threshold": 0.03,
            },
            {
                "group": "G2",
                "val_rank_ic": 0.02,
                "annualized_excess_vs_market": 0.20,
                "daily_bt_like_proxy_annualized_excess_vs_market": 0.01,
                "daily_proxy_admission_threshold": 0.0,
                "daily_proxy_full_backtest_threshold": 0.03,
            },
        ]
    )
    leaderboard = build_daily_proxy_first_leaderboard(summary)
    assert leaderboard["group"].tolist() == ["G1", "G2", "G0"]
    assert leaderboard.loc[0, "daily_proxy_first_status"] == "full_backtest_candidate"
    assert leaderboard.loc[1, "daily_proxy_first_status"] == "gray_zone"
    assert leaderboard.loc[2, "daily_proxy_first_status"] == "reject"


def test_build_p1_daily_proxy_first_report_includes_required_sections():
    summary = pd.DataFrame(
        [
            {
                "group": "G0",
                "daily_proxy_first_status": "reject",
                "daily_bt_like_proxy_annualized_excess_vs_market": -0.28,
                "daily_proxy_safety_margin_to_full_backtest": -0.31,
                "topk_boundary_topk_minus_next_mean_return": 0.002,
                "topk_boundary_switch_in_minus_out_mean_return": -0.009,
                "topk_boundary_avg_turnover_half_l1": 0.78,
            }
        ]
    )
    leaderboard = build_daily_proxy_first_leaderboard(summary)
    state = pd.DataFrame(
        [
            {
                "group": "G0",
                "state_axis": "return_state",
                "state": "strong_up",
                "n_months": 6,
                "median_excess_return": -0.024,
                "beat_rate": 0.17,
                "strategy_mean_return": 0.01,
                "benchmark_mean_return": 0.034,
            }
        ]
    )
    boundary = pd.DataFrame(
        [
            {
                "group": "G0",
                "period": "2026-01",
                "topk_mean_return": 0.01,
                "next_bucket_mean_return": 0.008,
                "topk_minus_next_bucket_return": 0.002,
                "switched_in_minus_out_return": -0.009,
                "topk_turnover_half_l1": 0.78,
            }
        ]
    )
    payload = {
        "generated_at_utc": "2026-04-27T00:00:00+00:00",
        "research_topic": "p1_tree_groups",
        "research_config_id": "rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_obj_regression",
        "output_stem": "p1_report_smoke",
        "summary_csv": "data/results/p1_report_smoke_summary.csv",
        "daily_proxy_leaderboard_csv": "data/results/p1_report_smoke_daily_proxy_leaderboard.csv",
        "daily_proxy_state_summary_csv": "data/results/p1_report_smoke_daily_proxy_state_summary.csv",
        "topk_boundary_csv": "data/results/p1_report_smoke_topk_boundary.csv",
        "bundle_manifest_csv": "data/results/p1_report_smoke_bundle_manifest.csv",
        "label_meta": {"label_mode": "rank_fusion", "label_scope": "cross_section_relative"},
        "xgboost_objective": "regression",
        "config": {
            "p1_experiment_mode": "daily_proxy_first",
            "legacy_proxy_decision_role": "diagnostic_only",
            "config_source": "config.yaml.backtest",
            "top_k": 20,
            "rebalance_rule": "M",
            "portfolio_method": "equal_weight",
            "execution_mode": "tplus1_open",
            "proxy_horizon": 5,
            "proxy_max_turnover": 1.0,
            "label_horizons": [5, 10, 20],
            "label_weights": [1 / 3, 1 / 3, 1 / 3],
            "label_transform": "calmar",
            "label_truncate_quantile": "",
            "label_spec": {
                "label_component_columns": "forward_ret_5d,forward_ret_10d,forward_ret_20d",
                "label_transform": "calmar",
                "target_column": "forward_ret_fused",
            },
            "backtest_config": "config.yaml.backtest",
            "backtest_start": "2021-01-01",
            "backtest_end": "",
            "backtest_top_k": 20,
            "backtest_max_turnover": 1.0,
            "backtest_portfolio_method": "equal_weight",
            "backtest_prepared_factors_cache": "",
            "transaction_costs": {
                "commission_buy_bps": 2.5,
                "commission_sell_bps": 2.5,
                "slippage_bps_per_side": 2.0,
                "stamp_duty_sell_bps": 5.0,
            },
            "daily_proxy_admission_threshold": 0.0,
            "daily_proxy_full_backtest_threshold": 0.03,
        },
    }

    report = build_p1_daily_proxy_first_report(
        summary_df=summary,
        daily_leaderboard_df=leaderboard,
        state_summary_df=state,
        boundary_df=boundary,
        payload=payload,
    )

    assert "# P1 Daily Proxy First Report" in report
    assert "config_source" in report
    assert "daily_proxy_first" in report
    assert "daily_proxy_leaderboard_csv" in report
    assert "strong_up_median_excess" in report
    assert "switch_in_minus_out" in report
    assert "Top-K 边界样本" in report
    assert "portfolio_method" in report
    assert "transaction_costs_bps" in report
    assert "label_component_columns" in report
    assert "label_transform" in report
    assert "backtest_prepared_factors_cache" in report


def test_build_group_comparison_table_flags_failed_promotion_gate():
    out = build_group_comparison_table(
        [
            {
                "group": "G0",
                "val_rank_ic": 0.05,
                "annualized_excess_vs_market": 0.02,
                "full_backtest_annualized_excess_vs_market": 0.01,
                "rolling_oos_median_ann_return": 0.03,
                "slice_oos_median_ann_return": 0.02,
                "yearly_excess_median_vs_market": 0.00,
                "key_year_excess_mean_vs_market": -0.01,
            },
            {
                "group": "G2",
                "val_rank_ic": 0.04,
                "annualized_excess_vs_market": 0.01,
                "full_backtest_annualized_excess_vs_market": 0.00,
                "rolling_oos_median_ann_return": 0.01,
                "slice_oos_median_ann_return": 0.03,
                "yearly_excess_median_vs_market": 0.01,
                "key_year_excess_mean_vs_market": -0.03,
            },
        ]
    )
    row = out.loc[out["group"] == "G2"].iloc[0]
    assert bool(row["pass_p1_val_rank_ic_gate"]) is False
    assert bool(row["pass_p1_rolling_oos_gate"]) is False
    assert bool(row["pass_p1_key_year_gate"]) is False
    assert bool(row["pass_p1_promotion_gate"]) is False


def test_summarize_p1_full_backtest_payload_extracts_gate_metrics():
    payload = {
        "full_sample": {
            "with_cost": {
                "annualized_return": -0.16,
                "sharpe_ratio": -0.62,
                "max_drawdown": 0.69,
            },
            "excess_vs_market": {
                "annualized_return": -0.27,
            },
        },
        "research_topic": "p1_tree_groups",
        "research_config_id": "rb_m_top20_lh_5_px_5_val20",
        "output_stem": "p1_tree_full_backtest_g1",
        "meta": {
            "tree_model": {
                "bundle_dir": "data/models/xgboost_panel_abc",
                "feature_group": "G1",
                "label_spec": {"horizons": [5], "target_column": "forward_ret_5d"},
                "tree_score_auto_flipped": False,
            },
            "prepared_factors_cache": {
                "path": "data/cache/prepared.parquet",
                "schema_version": 3,
            },
        },
        "walk_forward_rolling": {
            "agg": {
                "median_ann_return": -0.11,
                "median_ann_excess_vs_market": -0.06,
            }
        },
        "walk_forward_slices": {
            "agg": {
                "median_ann_return": -0.09,
                "median_ann_excess_vs_market": -0.04,
            }
        },
        "yearly": [
            {"year": 2021, "excess": -0.20},
            {"year": 2024, "excess": 0.05},
            {"year": 2025, "excess": -0.10},
            {"year": 2026, "excess": 0.02},
        ],
    }
    out = summarize_p1_full_backtest_payload(payload)
    assert out["full_backtest_annualized_excess_vs_market"] == pytest.approx(-0.27)
    assert out["rolling_oos_median_ann_return"] == pytest.approx(-0.11)
    assert out["slice_oos_median_ann_return"] == pytest.approx(-0.09)
    assert out["rolling_oos_median_ann_excess_vs_market"] == pytest.approx(-0.06)
    assert out["slice_oos_median_ann_excess_vs_market"] == pytest.approx(-0.04)
    assert out["yearly_excess_median_vs_market"] == pytest.approx((-0.10 + 0.02) / 2.0)
    assert out["key_year_excess_mean_vs_market"] == pytest.approx((-0.20 - 0.10 + 0.02) / 3.0)
    assert out["benchmark_key_years"] == ",".join(str(y) for y in DEFAULT_P1_BENCHMARK_KEY_YEARS)
    assert out["full_backtest_research_topic"] == "p1_tree_groups"
    assert out["full_backtest_tree_bundle_dir"] == "data/models/xgboost_panel_abc"
    assert out["full_backtest_tree_feature_group"] == "G1"
    assert out["full_backtest_tree_score_auto_flipped"] is False
    assert out["full_backtest_prepared_cache_schema_version"] == "3"


def test_parse_group_list_normalizes_case_and_filters_empty():
    assert _parse_group_list("g2, G4 ,,g1") == ["G2", "G4", "G1"]


def test_resolve_label_horizons_cli_horizons_override_config_weights():
    horizons, weights = resolve_label_horizons_and_weights(
        cli_label_horizons="5",
        cli_label_weights="",
        default_horizon=10,
        label_cfg={"horizons": [5, 10, 20], "weights": [0.5, 0.3, 0.2]},
    )
    assert horizons == [5]
    assert weights == [1.0]


def test_p1_tree_parse_args_accepts_history_and_sample_start(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--history-start",
            "2020-01-01",
            "--sample-start",
            "2021-01-01",
        ],
    )
    args = parse_args()
    assert args.history_start == "2020-01-01"
    assert args.sample_start == "2021-01-01"


def test_p1_tree_parse_args_accepts_daily_proxy_gate_controls(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--daily-proxy-admission-threshold",
            "-0.01",
            "--daily-proxy-full-backtest-threshold",
            "0.03",
            "--disable-daily-proxy-admission-gate",
        ],
    )
    args = parse_args()
    assert args.daily_proxy_admission_threshold == pytest.approx(-0.01)
    assert args.daily_proxy_full_backtest_threshold == pytest.approx(0.03)
    assert args.disable_daily_proxy_admission_gate is True


def test_p1_tree_parse_args_defaults_allow_full_monthly_turnover(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(sys, "argv", ["run_p1_tree_groups.py"])

    args = parse_args()

    assert args.proxy_max_turnover == pytest.approx(1.0)
    assert args.backtest_max_turnover == pytest.approx(1.0)


def test_p1_tree_parse_args_accepts_monthly_investable_label(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--label-mode",
            "monthly_investable",
        ],
    )
    args = parse_args()
    assert args.label_mode == "monthly_investable"


def test_p1_tree_parse_args_accepts_top_bucket_rank_fusion_label(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--label-mode",
            "top_bucket_rank_fusion",
        ],
    )
    args = parse_args()
    assert args.label_mode == "top_bucket_rank_fusion"


def test_p1_tree_parse_args_accepts_monthly_up_capture_label(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--label-mode",
            "monthly_investable_up_capture_market_relative",
        ],
    )
    args = parse_args()
    assert args.label_mode == "monthly_investable_up_capture_market_relative"


def test_p1_tree_parse_args_accepts_up_capture_market_relative_label(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--label-mode",
            "up_capture_market_relative",
        ],
    )
    args = parse_args()
    assert args.label_mode == "up_capture_market_relative"


def test_p1_tree_parse_args_accepts_path_quality_label_transform(monkeypatch):
    import sys

    from scripts.run_p1_tree_groups import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p1_tree_groups.py",
            "--label-transform",
            "calmar",
        ],
    )
    args = parse_args()
    assert args.label_transform == "calmar"
    assert args.label_truncate_quantile == pytest.approx(0.98)


def test_build_p1_tree_research_config_id_is_stable_and_readable():
    config_id = build_p1_tree_research_config_id(
        rebalance_rule="2W",
        top_k=20,
        label_horizons=[5, 10, 20],
        proxy_horizon=10,
        val_frac=0.2,
    )
    assert config_id == "rb_2w_top20_lh_5-10-20_px_10_val20"


def test_build_p1_tree_research_config_id_can_include_label_and_objective():
    config_id = build_p1_tree_research_config_id(
        rebalance_rule="M",
        top_k=20,
        label_horizons=[5, 10, 20],
        proxy_horizon=5,
        val_frac=0.2,
        label_mode="market_relative",
        xgboost_objective="regression",
    )
    assert config_id == "rb_m_top20_lh_5-10-20_px_5_val20_lbl_market_relative_obj_regression"


def test_build_p1_tree_research_config_id_includes_non_default_label_weights():
    config_id = build_p1_tree_research_config_id(
        rebalance_rule="M",
        top_k=20,
        label_horizons=[5, 10, 20],
        label_weights=[0.1, 0.2, 0.7],
        proxy_horizon=5,
        val_frac=0.2,
        label_mode="rank_fusion",
        xgboost_objective="regression",
    )
    assert config_id == "rb_m_top20_lh_5-10-20_px_5_val20_lw_10-20-70_lbl_rank_fusion_obj_regression"


def test_build_p1_tree_research_config_id_includes_non_raw_label_transform():
    config_id = build_p1_tree_research_config_id(
        rebalance_rule="M",
        top_k=20,
        label_horizons=[5, 10, 20],
        proxy_horizon=5,
        val_frac=0.2,
        label_mode="rank_fusion",
        label_transform="calmar",
        xgboost_objective="regression",
    )
    assert config_id == "rb_m_top20_lh_5-10-20_px_5_val20_lbl_rank_fusion_lt_calmar_obj_regression"


def test_build_p1_tree_output_stem_includes_tag_config_and_timestamp():
    stem = build_p1_tree_output_stem(
        out_tag="P1 Tree Groups",
        research_config_id="rb_m_top20_lh_5_px_5_val20",
        generated_at="2026-04-25T12:34:56Z",
    )
    assert stem == "p1_tree_groups_rb_m_top20_lh_5_px_5_val20_20260425_123456"
