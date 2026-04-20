"""F1 验收：T+21 IC / t 值门槛解析。"""
from __future__ import annotations

import pandas as pd

from src.features.ic_f1_gate import build_f1_gate_rows


def test_f1_gate_pass_and_fail():
    summary = pd.DataFrame(
        [
            {
                "factor": "ocf_to_asset",
                "horizon_key": "close_21d",
                "settlement": "close_to_close",
                "ic_mean": 0.02,
                "ic_t_value": 3.0,
                "n_dates": 100,
            },
            {
                "factor": "weak",
                "horizon_key": "close_21d",
                "settlement": "close_to_close",
                "ic_mean": 0.005,
                "ic_t_value": 1.0,
                "n_dates": 100,
            },
        ]
    )
    rows = build_f1_gate_rows(summary, ["ocf_to_asset", "weak", "missing"], ic_min=0.01, t_min=2.0)
    by = {r["factor"]: r for r in rows}
    assert by["ocf_to_asset"]["pass_f1"] is True
    assert by["weak"]["pass_f1"] is False
    assert by["missing"]["pass_f1"] is False
    assert by["missing"]["reason"] == "missing_or_no_close_21d"
