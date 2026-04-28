from pathlib import Path

import pandas as pd

from scripts.run_r2b_tradable_upside_replacement_v1 import (
    PREFIX_FALLBACK_INDUSTRY_SOURCE,
    _industry_source_status,
    _load_or_build_industry_map,
    build_replacement_weights,
)


def _panel() -> pd.DataFrame:
    rd = pd.Timestamp("2024-01-31")
    rows = []
    for i, sym in enumerate(["000001", "000002", "000003", "000004", "000005"]):
        rows.append(
            {
                "symbol": sym,
                "trade_date": rd,
                "industry": "A" if i < 4 else "B",
                "score__defensive": 1.0 - i * 0.1,
                "score__u2_a": 0.10 + i * 0.2,
                "score__u2_a_pct": 0.20 + i * 0.2,
                "limit_up_hits_20d": 0.0,
                "limit_down_hits_20d": 0.0,
                "amount_expansion_5_60": 0.1,
                "turnover_expansion_5_60": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_build_replacement_weights_caps_replacements_and_requires_lagged_state():
    rd = pd.Timestamp("2024-01-31")
    entry = pd.Timestamp("2024-02-01")
    defensive = pd.DataFrame(
        [[0.25, 0.25, 0.25, 0.25, 0.0]],
        index=[rd],
        columns=["000001", "000002", "000003", "000004", "000005"],
    )
    state = pd.DataFrame(
        [
            {
                "rebalance_date": rd,
                "lagged_regime": "strong_up",
                "lagged_breadth": "mid",
            }
        ]
    )
    limit_mask = pd.DataFrame(False, index=[entry], columns=defensive.columns)
    weights, diag = build_replacement_weights(
        panel=_panel(),
        defensive_weights=defensive,
        state_by_rebalance=state,
        score_col="score__u2_a",
        rule={"id": "R2B_REPLACE_3", "max_replace": 1, "overlay_weight": None},
        trading_index=pd.DatetimeIndex([rd, entry]),
        limit_up_open_mask=limit_mask,
        upside_pct=0.90,
        score_margin=0.10,
        max_industry_names=4,
        max_limit_up_hits_20d=2.0,
        max_expansion=1.5,
    )

    assert bool(diag.loc[0, "replacement_allowed"]) is True
    assert diag.loc[0, "replacement_count"] == 1
    assert weights.loc[rd, "000005"] == 0.25
    assert (weights.loc[rd] > 0).sum() == 4


def test_build_replacement_weights_holds_core_when_state_not_allowed():
    rd = pd.Timestamp("2024-01-31")
    entry = pd.Timestamp("2024-02-01")
    defensive = pd.DataFrame(
        [[0.25, 0.25, 0.25, 0.25, 0.0]],
        index=[rd],
        columns=["000001", "000002", "000003", "000004", "000005"],
    )
    state = pd.DataFrame(
        [
            {
                "rebalance_date": rd,
                "lagged_regime": "neutral",
                "lagged_breadth": "mid",
            }
        ]
    )
    weights, diag = build_replacement_weights(
        panel=_panel(),
        defensive_weights=defensive,
        state_by_rebalance=state,
        score_col="score__u2_a",
        rule={"id": "R2B_REPLACE_3", "max_replace": 3, "overlay_weight": None},
        trading_index=pd.DatetimeIndex([rd, entry]),
        limit_up_open_mask=pd.DataFrame(False, index=[entry], columns=defensive.columns),
        upside_pct=0.90,
        score_margin=0.10,
        max_industry_names=4,
        max_limit_up_hits_20d=2.0,
        max_expansion=1.5,
    )

    assert bool(diag.loc[0, "replacement_allowed"]) is False
    assert diag.loc[0, "replacement_count"] == 0
    pd.testing.assert_series_equal(weights.loc[rd], defensive.loc[rd], check_names=False)


def test_load_or_build_industry_map_uses_real_map_when_present(tmp_path):
    p = tmp_path / "industry_map.csv"
    pd.DataFrame(
        [
            {
                "symbol": "1",
                "industry": "银行",
                "industry_level1": "银行",
                "industry_level2": "银行",
                "source": "akshare.stock_board_industry_cons_em",
                "asof_date": "2026-04-28",
            }
        ]
    ).to_csv(p, index=False)

    out, source = _load_or_build_industry_map(["000001"], p)

    assert out.loc[0, "symbol"] == "000001"
    assert out.loc[0, "industry"] == "银行"
    assert _industry_source_status(source) == "real_industry_map"


def test_load_or_build_industry_map_reports_prefix_fallback_when_missing():
    out, source = _load_or_build_industry_map(["600519"], Path("/tmp/not-present-industry-map.csv"))

    assert out.loc[0, "symbol"] == "600519"
    assert source == PREFIX_FALLBACK_INDUSTRY_SOURCE
    assert _industry_source_status(source) == "prefix_fallback_no_alpha_evidence"
