from __future__ import annotations

import sys

from scripts.run_backtest_eval import parse_args


def test_parse_args_accepts_b3_b4_portfolio_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest_eval.py",
            "--top-k",
            "20",
            "--max-turnover",
            "0.3",
            "--entry-top-k",
            "20",
            "--hold-buffer-top-k",
            "40",
            "--portfolio-method",
            "tiered_equal_weight",
            "--top-tier-count",
            "10",
            "--top-tier-weight-share",
            "0.6",
            "--prepared-factors-cache",
            "data/cache/prepared_factors_v3.parquet",
            "--prepare-factors-only",
        ],
    )

    args = parse_args()

    assert args.top_k == 20
    assert args.max_turnover == 0.3
    assert args.entry_top_k == 20
    assert args.hold_buffer_top_k == 40
    assert args.portfolio_method == "tiered_equal_weight"
    assert args.top_tier_count == 10
    assert args.top_tier_weight_share == 0.6
    assert args.prepared_factors_cache == "data/cache/prepared_factors_v3.parquet"
    assert args.prepare_factors_only is True


def test_parse_args_accepts_xgboost_backtest_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest_eval.py",
            "--sort-by",
            "xgboost",
            "--tree-bundle-dir",
            "data/models/xgboost_panel_latest",
            "--tree-features",
            "momentum,rsi,weekly_kdj_j",
            "--tree-rsi-mode",
            "level",
            "--tree-feature-group",
            "G1",
            "--research-topic",
            "p1_tree_groups",
            "--research-config-id",
            "rb_m_top20_lh_5_px_5_val20",
            "--output-stem",
            "p1_tree_full_backtest_g1",
            "--canonical-config",
            "p1_tree_full_backtest",
            "--json-report",
            "data/results/tree_backtest.json",
        ],
    )

    args = parse_args()

    assert args.sort_by == "xgboost"
    assert args.tree_bundle_dir == "data/models/xgboost_panel_latest"
    assert args.tree_features == "momentum,rsi,weekly_kdj_j"
    assert args.tree_rsi_mode == "level"
    assert args.tree_feature_group == "G1"
    assert args.research_topic == "p1_tree_groups"
    assert args.research_config_id == "rb_m_top20_lh_5_px_5_val20"
    assert args.output_stem == "p1_tree_full_backtest_g1"
    assert args.canonical_config == "p1_tree_full_backtest"
