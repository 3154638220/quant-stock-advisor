import pandas as pd
import pytest

from scripts.run_r2b_edge_gated_replacement_v2 import (
    add_pair_edge_score,
    build_edge_gated_weights,
    select_edge_gated_replacements,
)


def test_add_pair_edge_score_prefers_stronger_margin_and_relief():
    pairs = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-01-31"),
                "score_col": "score__u3_a",
                "old_pool": "S2_bottom_3",
                "candidate_pool": "candidate_top_pct_95",
                "candidate_score_pct": 0.99,
                "score_margin": 0.30,
                "rel_strength_diff": 0.20,
                "amount_expansion_diff": 0.10,
                "turnover_expansion_diff": 0.05,
                "old_defensive_score": 0.10,
                "overheat_diff": -1.0,
            },
            {
                "trade_date": pd.Timestamp("2024-01-31"),
                "score_col": "score__u3_a",
                "old_pool": "S2_bottom_3",
                "candidate_pool": "candidate_top_pct_95",
                "candidate_score_pct": 0.95,
                "score_margin": 0.05,
                "rel_strength_diff": -0.10,
                "amount_expansion_diff": -0.05,
                "turnover_expansion_diff": -0.02,
                "old_defensive_score": 0.50,
                "overheat_diff": 1.0,
            },
        ]
    )

    out = add_pair_edge_score(pairs)

    assert out.loc[0, "pair_edge_score"] > out.loc[1, "pair_edge_score"]
    assert out["pair_edge_score"].between(0.0, 1.0).all()


def test_select_edge_gated_replacements_does_not_fill_slots_without_edge():
    rd = pd.Timestamp("2024-01-31")
    panel = pd.DataFrame(
        [
            {"trade_date": rd, "symbol": "000001", "industry": "A"},
            {"trade_date": rd, "symbol": "000002", "industry": "A"},
            {"trade_date": rd, "symbol": "000003", "industry": "B"},
            {"trade_date": rd, "symbol": "000101", "industry": "C"},
            {"trade_date": rd, "symbol": "000102", "industry": "C"},
            {"trade_date": rd, "symbol": "000103", "industry": "D"},
        ]
    )
    defensive = pd.DataFrame(
        [[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]],
        index=[rd],
        columns=["000001", "000002", "000003", "000101", "000102", "000103"],
    )
    pairs = pd.DataFrame(
        [
            {
                "trade_date": rd,
                "score_col": "score__u3_c",
                "old_pool": "S2_bottom_5",
                "candidate_pool": "candidate_buyable",
                "old_symbol": "000001",
                "new_symbol": "000101",
                "old_industry": "A",
                "new_industry": "C",
                "state_strong_up_or_wide": True,
                "pair_edge_score": 0.80,
                "score_margin": 0.12,
            },
            {
                "trade_date": rd,
                "score_col": "score__u3_c",
                "old_pool": "S2_bottom_5",
                "candidate_pool": "candidate_buyable",
                "old_symbol": "000002",
                "new_symbol": "000102",
                "old_industry": "A",
                "new_industry": "C",
                "state_strong_up_or_wide": True,
                "pair_edge_score": 0.60,
                "score_margin": 0.20,
            },
            {
                "trade_date": rd,
                "score_col": "score__u3_c",
                "old_pool": "S2_bottom_5",
                "candidate_pool": "candidate_buyable",
                "old_symbol": "000003",
                "new_symbol": "000103",
                "old_industry": "B",
                "new_industry": "D",
                "state_strong_up_or_wide": True,
                "pair_edge_score": 0.90,
                "score_margin": 0.01,
            },
        ]
    )
    rule = {
        "id": "U3_C_pairwise_residual_edge__EDGE_GATED",
        "input_id": "U3_C_pairwise_residual_edge",
        "score_col": "score__u3_c",
        "old_pool": "S2_bottom_5",
        "candidate_pool": "candidate_buyable",
        "state_gate": "state_strong_up_or_wide",
        "edge_threshold": 0.72,
        "min_score_margin": 0.06,
    }

    selected = select_edge_gated_replacements(
        pairs,
        defensive_weights=defensive,
        panel=panel,
        rule=rule,
        max_replace=3,
        max_industry_names=3,
    )
    weights, diag = build_edge_gated_weights(defensive_weights=defensive, selected_pairs=selected, rule=rule)

    assert len(selected) == 1
    assert diag.loc[0, "replacement_count"] == 1
    assert weights.loc[rd, "000001"] == 0.0
    assert weights.loc[rd, "000101"] == pytest.approx(1 / 3)
    assert weights.loc[rd].sum() == pytest.approx(1.0)


def test_select_edge_gated_replacements_respects_industry_capacity():
    rd = pd.Timestamp("2024-01-31")
    panel = pd.DataFrame(
        [
            {"trade_date": rd, "symbol": "000001", "industry": "A"},
            {"trade_date": rd, "symbol": "000002", "industry": "B"},
            {"trade_date": rd, "symbol": "000101", "industry": "C"},
        ]
    )
    defensive = pd.DataFrame(
        [[0.5, 0.5, 0.0]],
        index=[rd],
        columns=["000001", "000002", "000101"],
    )
    pairs = pd.DataFrame(
        [
            {
                "trade_date": rd,
                "score_col": "score__u3_a",
                "old_pool": "S2_bottom_3",
                "candidate_pool": "candidate_top_pct_95",
                "old_symbol": "000001",
                "new_symbol": "000101",
                "old_industry": "A",
                "new_industry": "C",
                "state_strong_up_and_wide": True,
                "pair_edge_score": 0.90,
                "score_margin": 0.20,
            }
        ]
    )
    rule = {
        "id": "U3_A_real_industry_leadership__EDGE_GATED",
        "input_id": "U3_A_real_industry_leadership",
        "score_col": "score__u3_a",
        "old_pool": "S2_bottom_3",
        "candidate_pool": "candidate_top_pct_95",
        "state_gate": "state_strong_up_and_wide",
        "edge_threshold": 0.68,
        "min_score_margin": 0.10,
    }

    selected = select_edge_gated_replacements(
        pairs,
        defensive_weights=defensive,
        panel=panel,
        rule=rule,
        max_replace=3,
        max_industry_names=0,
    )

    assert selected.empty
