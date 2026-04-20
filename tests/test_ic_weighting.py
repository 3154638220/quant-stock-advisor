from __future__ import annotations

import math

import pandas as pd

from scripts.update_ic_weights import build_weights_by_date


def test_build_weights_by_date_emits_normalized_snapshots():
    ic_df = pd.DataFrame(
        [
            {"factor": "factor_a", "trade_date": "2024-01-02", "ic": 0.10},
            {"factor": "factor_a", "trade_date": "2024-01-03", "ic": 0.12},
            {"factor": "factor_a", "trade_date": "2024-01-04", "ic": 0.08},
            {"factor": "factor_b", "trade_date": "2024-01-02", "ic": -0.05},
            {"factor": "factor_b", "trade_date": "2024-01-03", "ic": -0.06},
            {"factor": "factor_b", "trade_date": "2024-01-04", "ic": -0.07},
        ]
    )

    out = build_weights_by_date(
        ic_df,
        window=3,
        min_obs=2,
        half_life=2.0,
        clip_abs_weight=0.25,
    )

    assert sorted(out) == ["2024-01-03", "2024-01-04"]
    for weights in out.values():
        assert math.isclose(sum(abs(v) for v in weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
        assert weights["factor_a"] > 0.0
        assert weights["factor_b"] < 0.0

