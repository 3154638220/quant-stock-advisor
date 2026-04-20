"""F1盈利质量因子验收：T+21 close Rank IC 与 t 值门槛。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_f1_gate_rows(
    summary_df: pd.DataFrame,
    f1_factors: list[str],
    *,
    ic_min: float,
    t_min: float,
) -> list[dict[str, Any]]:
    sub = summary_df[
        (summary_df["horizon_key"] == "close_21d") & (summary_df["settlement"] == "close_to_close")
    ]
    rows: list[dict[str, Any]] = []
    for fac in f1_factors:
        r = sub[sub["factor"] == fac]
        if r.empty:
            rows.append(
                {
                    "factor": fac,
                    "pass_f1": False,
                    "reason": "missing_or_no_close_21d",
                    "ic_mean": float("nan"),
                    "ic_t_value": float("nan"),
                    "n_dates": 0,
                }
            )
            continue
        row = r.iloc[0]
        ic_m = float(row["ic_mean"])
        t_v = float(row["ic_t_value"])
        n_d = int(row["n_dates"])
        ok = (
            np.isfinite(ic_m)
            and np.isfinite(t_v)
            and ic_m >= float(ic_min)
            and t_v >= float(t_min)
            and n_d > 1
        )
        rows.append(
            {
                "factor": fac,
                "pass_f1": bool(ok),
                "reason": "" if ok else "ic_or_t_below_threshold_or_insufficient_n",
                "ic_mean": ic_m,
                "ic_t_value": t_v,
                "n_dates": n_d,
            }
        )
    return rows
