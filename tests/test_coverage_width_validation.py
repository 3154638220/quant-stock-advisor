from __future__ import annotations

import sys
from pathlib import Path

import yaml

from scripts.run_coverage_width_validation import (
    build_variant_scenarios,
    materialize_scenario_config,
    parse_args,
    select_scenarios,
)


def test_select_scenarios_keeps_baseline_when_filtering_b3_b4():
    scenarios = select_scenarios(["b3", "b4"])

    assert [scenario.key for scenario in scenarios] == ["v3", "b3", "b4"]


def test_build_variant_scenarios_generates_restrained_b3_b4_grid():
    scenarios = build_variant_scenarios(
        top_k=20,
        variant_mode="both",
        entry_top_k_values=[20],
        hold_buffer_top_k_values=[35, 40],
        top_tier_count_values=[8],
        top_tier_weight_share_values=[0.65],
    )

    keys = [scenario.key for scenario in scenarios]
    assert keys == [
        "vb3_20_35",
        "vb3_20_40",
        "vb4_20_35_t8_s65",
        "vb4_20_40_t8_s65",
    ]
    assert scenarios[0].portfolio_overrides["weight_method"] == "equal_weight"
    assert scenarios[-1].portfolio_overrides["weight_method"] == "tiered_equal_weight"


def test_select_scenarios_can_filter_dynamic_variants_with_baseline():
    dynamic = build_variant_scenarios(
        top_k=20,
        variant_mode="b4",
        entry_top_k_values=[20],
        hold_buffer_top_k_values=[35],
        top_tier_count_values=[8],
        top_tier_weight_share_values=[0.65],
    )

    scenarios = select_scenarios(["vb4_20_35_t8_s65"], dynamic)

    assert [scenario.key for scenario in scenarios] == ["vb4_20_35_t8_s65"]


def test_materialize_scenario_config_writes_snapshot_for_dynamic_variant(tmp_path: Path):
    scenario = build_variant_scenarios(
        top_k=20,
        variant_mode="b4",
        entry_top_k_values=[20],
        hold_buffer_top_k_values=[35],
        top_tier_count_values=[8],
        top_tier_weight_share_values=[0.65],
    )[0]
    cfg = {
        "signals": {"top_k": 20},
        "portfolio": {"weight_method": "equal_weight", "max_turnover": 0.3},
    }

    snapshot_name = materialize_scenario_config(scenario, cfg, root_dir=tmp_path)
    snapshot_path = tmp_path / snapshot_name

    assert snapshot_name == "config.yaml.backtest.vb4_20_35_t8_s65"
    assert snapshot_path.exists()
    loaded = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert loaded["signals"]["top_k"] == 20
    assert loaded["portfolio"]["weight_method"] == "tiered_equal_weight"
    assert loaded["portfolio"]["hold_buffer_top_k"] == 35


def test_parse_args_accepts_scenario_filter(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_coverage_width_validation.py",
            "--scenarios",
            "b3,b4",
        ],
    )

    args = parse_args()

    assert args.scenarios == "b3,b4"


def test_parse_args_accepts_variant_grid_inputs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_coverage_width_validation.py",
            "--variant-mode",
            "both",
            "--variant-hold-buffer-top-k-values",
            "35,40",
            "--variant-top-tier-count-values",
            "8,12",
            "--variant-top-tier-weight-share-values",
            "0.55,0.65",
        ],
    )

    args = parse_args()

    assert args.variant_mode == "both"
    assert args.variant_hold_buffer_top_k_values == "35,40"
    assert args.variant_top_tier_count_values == "8,12"
    assert args.variant_top_tier_weight_share_values == "0.55,0.65"
