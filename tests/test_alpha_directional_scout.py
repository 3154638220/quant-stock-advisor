from __future__ import annotations

import pandas as pd

from scripts.run_alpha_directional_scout import (
    build_flipped_gate_table,
    resolve_directional_candidate_pool,
    select_directional_source_candidates,
    select_flip_candidate_factors,
)


def test_select_flip_candidate_factors_uses_both_horizons_and_skips_baseline():
    gate_df = pd.DataFrame(
        [
            {"factor": "vol_to_turnover", "horizon_key": "close_21d", "ic_mean": -0.03},
            {"factor": "vol_to_turnover", "horizon_key": "tplus1_open_1d", "ic_mean": -0.03},
            {"factor": "momentum", "horizon_key": "close_21d", "ic_mean": -0.04},
            {"factor": "momentum", "horizon_key": "tplus1_open_1d", "ic_mean": -0.05},
            {"factor": "rsi", "horizon_key": "close_21d", "ic_mean": -0.06},
            {"factor": "rsi", "horizon_key": "tplus1_open_1d", "ic_mean": -0.01},
            {"factor": "bias_long", "horizon_key": "close_21d", "ic_mean": -0.08},
            {"factor": "bias_long", "horizon_key": "tplus1_open_1d", "ic_mean": -0.07},
        ]
    )

    out = select_flip_candidate_factors(
        gate_df,
        baseline_factor="vol_to_turnover",
        close21_max_ic=-0.02,
        open1_max_ic=-0.02,
        max_candidates=3,
    )

    assert out == ["bias_long", "momentum"]


def test_build_flipped_gate_table_inverts_ic_signs_and_t_values():
    gate_df = pd.DataFrame(
        [
            {"factor": "momentum", "horizon_key": "close_21d", "ic_mean": -0.04, "ic_t_value": -3.2},
            {"factor": "momentum", "horizon_key": "tplus1_open_1d", "ic_mean": -0.03, "ic_t_value": -2.4},
        ]
    )

    out = build_flipped_gate_table(gate_df, ["momentum"])

    assert out["factor"].tolist() == ["flip_momentum", "flip_momentum"]
    assert out["ic_mean"].tolist() == [0.04, 0.03]
    assert out["ic_t_value"].tolist() == [3.2, 2.4]


def test_select_directional_source_candidates_skips_passed_candidates_by_default():
    scout_df = pd.DataFrame(
        [
            {"candidate_factor": "vol_to_turnover", "scout_status": "baseline"},
            {"candidate_factor": "momentum", "scout_status": "fail"},
            {"candidate_factor": "bias_long", "scout_status": "pass"},
        ]
    )

    out = select_directional_source_candidates(
        scout_df,
        baseline_factor="vol_to_turnover",
        allow_passed_candidates=False,
    )

    assert out == ["momentum"]


def test_resolve_directional_candidate_pool_prefers_scout_source(tmp_path):
    scout_path = tmp_path / "factor_scout.csv"
    pd.DataFrame(
        [
            {"candidate_factor": "vol_to_turnover", "scout_status": "baseline"},
            {"candidate_factor": "momentum", "scout_status": "fail"},
            {"candidate_factor": "bias_long", "scout_status": "pass"},
        ]
    ).to_csv(scout_path, index=False, encoding="utf-8-sig")

    out_default, src_default = resolve_directional_candidate_pool(
        baseline_factor="vol_to_turnover",
        inferred_candidates=["vol_to_turnover", "rsi"],
        candidate_source_scout=str(scout_path),
        allow_passed_candidates=False,
    )
    out_all, src_all = resolve_directional_candidate_pool(
        baseline_factor="vol_to_turnover",
        inferred_candidates=["vol_to_turnover", "rsi"],
        candidate_source_scout=str(scout_path),
        allow_passed_candidates=True,
    )

    assert out_default == ["momentum"]
    assert out_all == ["momentum", "bias_long"]
    assert str(scout_path) in src_default
    assert str(scout_path) in src_all
