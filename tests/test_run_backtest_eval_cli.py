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
