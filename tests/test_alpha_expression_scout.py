from __future__ import annotations

import sys

import pandas as pd

from scripts.run_alpha_expression_scout import (
    build_expression_promotion_summary,
    build_conditioned_flip_factor,
    build_expression_gate_table,
    build_expression_scenarios,
    classify_expression_promotion,
    cross_sectional_residualize,
    parse_args,
    resolve_expression_candidates,
    select_expression_source_candidates,
)
from scripts.light_strategy_proxy import (
    build_light_proxy_period_detail,
    infer_periods_per_year,
    summarize_light_strategy_proxy,
)
from scripts.run_alpha_expression_diagnostics import build_overlay_diagnostics_summary, classify_defensive_overlay


def test_cross_sectional_residualize_removes_linear_dependency_by_date():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-31"] * 4 + ["2026-02-28"] * 4),
            "vol_to_turnover": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            "realized_vol": [3.0, 5.0, 7.0, 9.0, 5.0, 9.0, 13.0, 17.0],
        }
    )

    out = cross_sectional_residualize(
        df,
        target_col="realized_vol",
        control_cols=["vol_to_turnover"],
        min_obs=3,
    )

    resid_col = "realized_vol_resid"
    assert resid_col in out.columns
    for _, grp in out.groupby("trade_date"):
        resid = pd.to_numeric(grp[resid_col], errors="coerce")
        assert float(resid.abs().max()) < 1e-10


def test_build_conditioned_flip_factor_only_keeps_upper_tail():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-31"] * 5),
            "intraday_range": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out = build_conditioned_flip_factor(df, factor_col="intraday_range", z_threshold=0.5)

    assert out["intraday_range_cond_flip"].tolist() == [0.0, 0.0, 0.0, 4.0, 5.0]


def test_build_expression_gate_table_keeps_expression_level_gate_and_attaches_metadata():
    gate_df = pd.DataFrame(
        [
            {"factor": "vol_to_turnover", "horizon_key": "close_21d", "ic_mean": 0.01, "ic_t_value": 1.1},
            {
                "factor": "flip_resid_realized_vol",
                "horizon_key": "close_21d",
                "ic_mean": -0.02,
                "ic_t_value": -1.5,
            },
            {
                "factor": "flip_resid_realized_vol",
                "horizon_key": "tplus1_open_1d",
                "ic_mean": -0.01,
                "ic_t_value": -0.8,
            },
        ]
    )

    out = build_expression_gate_table(
        gate_df,
        baseline_factor="vol_to_turnover",
        expression_specs=[
            {
                "expression_factor": "flip_resid_realized_vol",
                "source_factor": "realized_vol",
                "family": "flip_residual",
            }
        ],
    )

    row = out.loc[out["factor"] == "flip_resid_realized_vol"].sort_values("horizon_key").iloc[0]
    assert row["source_factor"] == "realized_vol"
    assert row["expression_family"] == "flip_residual"
    assert float(row["ic_mean"]) == -0.02
    assert float(row["ic_t_value"]) == -1.5


def test_build_expression_scenarios_starts_with_baseline_and_builds_variants():
    scenarios, specs = build_expression_scenarios(
        baseline_factor="vol_to_turnover",
        candidates=["realized_vol"],
        blend_weights=[0.1],
        condition_z_thresholds=[0.5],
    )

    labels = [item["scenario"] for item in scenarios]
    assert labels[0] == "baseline_vol_to_turnover"
    assert "flip_resid_blend_10_realized_vol" in labels
    assert "flip_cond_z0p5_blend_10_realized_vol" in labels
    assert {item["family"] for item in specs} == {"flip_residual", "flip_conditioned"}


def test_alpha_expression_scout_parse_args_accepts_prepared_cache(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_alpha_expression_scout.py",
            "--prepared-factors-cache",
            "data/cache/prepared_expression_factors.parquet",
            "--refresh-prepared-factors-cache",
        ],
    )

    args = parse_args()

    assert args.prepared_factors_cache == "data/cache/prepared_expression_factors.parquet"
    assert args.refresh_prepared_factors_cache is True


def test_classify_expression_promotion_marks_combo_winner_as_overlay_only_when_benchmark_fails():
    scout_df = pd.DataFrame(
        [
            {
                "scenario": "baseline_vol_to_turnover",
                "family": "baseline",
                "is_baseline": True,
                "pass_f1_gate": False,
                "pass_combo_gate": False,
                "pass_benchmark_gate": False,
            },
            {
                "scenario": "flip_resid_blend_20_realized_vol",
                "family": "flip_residual",
                "is_baseline": False,
                "pass_f1_gate": False,
                "pass_combo_gate": True,
                "pass_benchmark_gate": False,
            },
        ]
    )

    out = classify_expression_promotion(scout_df)
    row = out.loc[out["scenario"] == "flip_resid_blend_20_realized_vol"].iloc[0]

    assert row["promotion_decision"] == "overlay_only"
    assert "overlay" in row["promotion_note"]


def test_classify_defensive_overlay_marks_downside_improver_as_overlay_candidate():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "baseline_vol_to_turnover",
                "family": "baseline",
                "up_month_beat_rate": 0.18,
                "down_month_beat_rate": 0.81,
                "down_month_median_excess": 0.027,
            },
            {
                "scenario": "flip_resid_blend_20_realized_vol",
                "family": "flip_residual",
                "up_month_beat_rate": 0.20,
                "down_month_beat_rate": 0.85,
                "down_month_median_excess": 0.030,
            },
        ]
    )

    out = classify_defensive_overlay(summary_df)
    row = out.loc[out["scenario"] == "flip_resid_blend_20_realized_vol"].iloc[0]

    assert row["overlay_role"] == "defensive_overlay_candidate"
    assert "overlay" in row["overlay_note"]


def test_build_expression_promotion_summary_mentions_overlay_only_and_no_mainline():
    scout_df = pd.DataFrame(
        [
            {"scenario": "baseline_vol_to_turnover", "promotion_decision": "baseline"},
            {"scenario": "flip_resid_blend_20_realized_vol", "promotion_decision": "overlay_only"},
            {"scenario": "flip_cond_z0p5_blend_15_realized_vol", "promotion_decision": "reject"},
        ]
    )

    out = build_expression_promotion_summary(scout_df)

    assert "baseline_vol_to_turnover" in out
    assert "本轮没有 expression 候选同时通过" in out
    assert "flip_resid_blend_20_realized_vol" in out


def test_build_overlay_diagnostics_summary_mentions_defensive_overlay():
    summary_df = pd.DataFrame(
        [
            {"scenario": "baseline_vol_to_turnover", "overlay_role": "baseline"},
            {"scenario": "flip_resid_blend_20_realized_vol", "overlay_role": "defensive_overlay_candidate"},
            {"scenario": "flip_cond_z0p5_blend_15_realized_vol", "overlay_role": "not_overlay"},
        ]
    )

    out = build_overlay_diagnostics_summary(summary_df)

    assert "baseline_vol_to_turnover" in out
    assert "flip_resid_blend_20_realized_vol" in out
    assert "防守型 overlay" in out


def test_infer_periods_per_year_respects_rebalance_frequency():
    assert infer_periods_per_year("M") == 12.0
    assert infer_periods_per_year("2M") == 6.0
    assert infer_periods_per_year("W-FRI") == 52.0
    assert infer_periods_per_year("2W") == 26.0
    assert infer_periods_per_year("D") == 252.0


def test_summarize_light_strategy_proxy_annualizes_by_proxy_frequency():
    monthly_df = pd.DataFrame(
        {
            "strategy_return": [0.01, 0.02, -0.01],
            "benchmark_return": [0.00, 0.01, 0.00],
            "excess_return": [0.01, 0.01, -0.01],
        }
    )

    out = summarize_light_strategy_proxy(monthly_df, periods_per_year=12.0)

    assert out["n_periods"] == 3
    assert abs(out["strategy_annualized_return"] - ((1.01 * 1.02 * 0.99) ** 4 - 1.0)) < 1e-12
    assert abs(out["benchmark_annualized_return"] - ((1.0 * 1.01 * 1.0) ** 4 - 1.0)) < 1e-12
    assert abs(out["annualized_excess_vs_market"] - ((1.01 * 1.01 * 0.99) ** 4 - 1.0)) < 1e-12


def test_build_light_proxy_period_detail_groups_by_rebalance_rule():
    idx = pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03", "2026-02-19"])
    strat = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    bench = pd.Series([0.00, 0.01, 0.00, 0.01], index=idx)

    out = build_light_proxy_period_detail(strat, bench, rebalance_rule="M", scenario="demo")

    assert out["period"].tolist() == ["2026-01", "2026-02"]
    assert out["scenario"].tolist() == ["demo", "demo"]
    assert abs(float(out.iloc[0]["strategy_return"]) - (1.01 * 1.02 - 1.0)) < 1e-12
    assert abs(float(out.iloc[1]["benchmark_return"]) - (1.0 * 1.01 - 1.0)) < 1e-12


def test_build_light_proxy_period_detail_supports_month_alias_used_by_expression_diagnostics():
    idx = pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03"])
    strat = pd.Series([0.01, 0.02, -0.01], index=idx)
    bench = pd.Series([0.00, 0.01, 0.00], index=idx)

    monthly = build_light_proxy_period_detail(strat, bench, rebalance_rule="M", scenario="expr").rename(
        columns={"period": "month"}
    )

    assert "month" in monthly.columns
    assert monthly["month"].tolist() == ["2026-01", "2026-02"]
    assert monthly["scenario"].tolist() == ["expr", "expr"]


def test_select_expression_source_candidates_skips_overlay_only_by_default():
    scout_df = pd.DataFrame(
        [
            {
                "base_factor": "realized_vol",
                "promotion_decision": "overlay_only",
            },
            {
                "base_factor": "intraday_range",
                "promotion_decision": "mainline_eligible",
            },
        ]
    )

    out = select_expression_source_candidates(
        scout_df,
        baseline_factor="vol_to_turnover",
        allow_overlay_candidates=False,
    )

    assert out == ["intraday_range"]


def test_resolve_expression_candidates_prefers_scout_source_and_can_include_overlay(tmp_path):
    scout_path = tmp_path / "expr_scout.csv"
    pd.DataFrame(
        [
            {"base_factor": "realized_vol", "promotion_decision": "overlay_only"},
            {"base_factor": "intraday_range", "promotion_decision": "mainline_eligible"},
        ]
    ).to_csv(scout_path, index=False, encoding="utf-8-sig")

    out_default, src_default = resolve_expression_candidates(
        baseline_factor="vol_to_turnover",
        candidates_arg=["bias_short"],
        candidate_source_scout=str(scout_path),
        allow_overlay_candidates=False,
    )
    out_overlay, src_overlay = resolve_expression_candidates(
        baseline_factor="vol_to_turnover",
        candidates_arg=["bias_short"],
        candidate_source_scout=str(scout_path),
        allow_overlay_candidates=True,
    )

    assert out_default == ["intraday_range"]
    assert out_overlay == ["realized_vol", "intraday_range"]
    assert str(scout_path) in src_default
    assert str(scout_path) in src_overlay


def test_expression_scout_summary_can_carry_light_proxy_fields():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "flip_resid_blend_20_realized_vol",
                "candidate_factor": "flip_resid_realized_vol",
                "base_factor": "realized_vol",
                "family": "flip_residual",
                "expression_type": "residual_blend",
                "factor_weight": -0.2,
                "is_baseline": False,
                "result_type": "light_strategy_proxy",
                "research_topic": "alpha_expression_scout",
                "research_config_id": "demo_cfg",
                "output_stem": "demo_stem",
                "rebalance_rule": "M",
                "periods_per_year": 12.0,
                "proxy_periods": 12,
                "proxy_annualized_return": 0.10,
                "proxy_benchmark_annualized_return": 0.04,
                "annualized_return": 0.09,
                "annualized_excess_vs_market": 0.06,
                "full_backtest_annualized_excess_vs_market": 0.03,
            }
        ]
    )

    assert summary_df.loc[0, "result_type"] == "light_strategy_proxy"
    assert summary_df.loc[0, "research_topic"] == "alpha_expression_scout"
    assert float(summary_df.loc[0, "periods_per_year"]) == 12.0
    assert float(summary_df.loc[0, "full_backtest_annualized_excess_vs_market"]) == 0.03


def test_expression_diagnostics_summary_can_carry_research_identity_fields():
    summary_df = pd.DataFrame(
        [
            {
                "scenario": "flip_resid_blend_20_realized_vol",
                "candidate_factor": "flip_resid_realized_vol",
                "family": "flip_residual",
                "expression_type": "residual_blend",
                "result_type": "light_strategy_proxy",
                "research_topic": "alpha_expression_diagnostics",
                "research_config_id": "demo_cfg",
                "output_stem": "demo_stem",
                "periods_per_year": 12.0,
                "annualized_excess_vs_market": 0.04,
            }
        ]
    )
    assert summary_df.loc[0, "research_topic"] == "alpha_expression_diagnostics"
    assert summary_df.loc[0, "result_type"] == "light_strategy_proxy"
