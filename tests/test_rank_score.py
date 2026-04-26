"""截面组合打分单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rank_score import (
    composite_linear_score,
    cross_section_zscore,
    sort_key_for_dataframe,
)


def test_cross_section_zscore_constant():
    x = np.array([1.0, 1.0, 1.0])
    z = cross_section_zscore(x)
    assert np.allclose(z, 0.0)


def test_cross_section_zscore_all_nan():
    x = np.array([np.nan, np.nan, np.nan])
    z = cross_section_zscore(x)
    assert z.shape == x.shape
    assert np.isnan(z).all()


def test_composite_linear_score_weights():
    mom = np.array([0.1, 0.2, 0.3])
    rsi = np.array([40.0, 50.0, 60.0])
    score, dbg = composite_linear_score(
        mom,
        rsi,
        w_momentum=1.0,
        w_rsi=0.0,
        rsi_mode="level",
    )
    assert score.shape == (3,)
    assert dbg.shape == (3, 3)


def test_sort_key_composite():
    df = pd.DataFrame(
        {
            "symbol": ["000001", "600519"],
            "momentum": [0.5, 0.1],
            "rsi": [55.0, 45.0],
        }
    )
    out = sort_key_for_dataframe(df, sort_by="composite")
    assert "composite_score" in out.columns
    assert "rank" in out.columns
    assert len(out) == 2


def test_sort_key_unknown():
    df = pd.DataFrame({"symbol": ["1"], "momentum": [1.0], "rsi": [50.0]})
    with pytest.raises(ValueError):
        sort_key_for_dataframe(df, sort_by="unknown")
