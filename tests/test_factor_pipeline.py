"""因子标准化、评估、基础张量因子冒烟测试。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from src.features.factor_eval import (
    ic_summary,
    information_coefficient,
    long_table_from_wide,
    quantile_returns,
    rank_ic,
    rolling_ic_stability,
)
from src.features.neutralize import attach_neutralized_pair, neutralize_size_industry_regression
from src.features.standardize import factor_standardize_pipeline
from src.features.tensor_base_factors import (
    atr_wilder,
    compute_base_factor_bundle,
    daily_returns_from_close,
    forward_returns_from_close,
    forward_returns_tplus1_open,
    true_range,
)
from src.models.rank_score import sort_key_for_dataframe
from src.models.recommend_explain import build_recommend_reason


def test_true_range_and_atr():
    close = torch.tensor([[100.0, 102.0, 101.0]], dtype=torch.float64)
    high = torch.tensor([[101.0, 103.0, 102.0]], dtype=torch.float64)
    low = torch.tensor([[99.0, 101.0, 100.0]], dtype=torch.float64)
    tr = true_range(high, low, close)
    assert torch.isnan(tr[0, 0])
    assert tr[0, 1].item() > 0
    atr = atr_wilder(high, low, close, period=2)
    assert torch.isfinite(atr[0, 2])


def test_composite_extended_sort():
    df = pd.DataFrame(
        {
            "momentum": [0.1, 0.2, 0.15],
            "rsi": [55.0, 60.0, 50.0],
            "atr": [1.0, 2.0, 1.5],
            "realized_vol": [0.2, 0.3, 0.25],
        }
    )
    out = sort_key_for_dataframe(
        df,
        sort_by="composite_extended",
        composite_extended_weights={
            "momentum": 0.4,
            "rsi": 0.35,
            "atr": 0.15,
            "realized_vol": 0.1,
        },
    )
    assert "composite_extended_score" in out.columns
    assert "rank" in out.columns


def test_momentum_sort_has_z_momentum():
    df = pd.DataFrame(
        {
            "momentum": [0.1, 0.3, 0.2],
            "rsi": [50.0, 60.0, 55.0],
        }
    )
    out = sort_key_for_dataframe(df, sort_by="momentum")
    assert "z_momentum" in out.columns
    assert out.iloc[0]["momentum"] == 0.3


def test_recommend_reason_momentum():
    row = pd.Series({"momentum": 0.5, "z_momentum": 1.2})
    s = build_recommend_reason(row, sort_by="momentum")
    assert "动量" in s and "z" in s


def test_daily_returns_and_forward():
    c = torch.tensor([[100.0, 110.0, 105.0]], dtype=torch.float64)
    r = daily_returns_from_close(c)
    assert np.isnan(r[0, 0].item())
    assert abs(r[0, 1].item() - 0.1) < 1e-9
    f = forward_returns_from_close(c, horizon=1)
    assert abs(f[0, 0].item() - (110 / 100 - 1)) < 1e-9


def test_forward_returns_tplus1_open():
    o = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0, 14.0]],
        dtype=torch.float64,
    )
    f = forward_returns_tplus1_open(o, horizon=1)
    # t=0: open[2]/open[1]-1 = 12/11-1
    assert abs(f[0, 0].item() - (12.0 / 11.0 - 1.0)) < 1e-9
    f2 = forward_returns_tplus1_open(o, horizon=2)
    # t=0: open[3]/open[1]-1
    assert abs(f2[0, 0].item() - (13.0 / 11.0 - 1.0)) < 1e-9


def test_base_factor_bundle():
    close = torch.tensor(
        [[100.0, 101.0, 102.0, 103.0, 104.0] * 1],
        dtype=torch.float32,
    )
    vol = torch.ones_like(close) * 1e6
    to = torch.ones_like(close) * 0.02
    hi = close * 1.001
    lo = close * 0.999
    b = compute_base_factor_bundle(
        close,
        volume=vol,
        turnover=to,
        high=hi,
        low=lo,
        vol_window=3,
        turnover_window=2,
        vp_corr_window=3,
        reversal_window=2,
        atr_period=3,
    )
    assert b["realized_vol"] is not None
    assert b["turnover_roll_mean"] is not None
    assert b["vol_ret_corr"] is not None
    assert b["atr"] is not None
    assert b["vol_to_turnover"] is not None
    assert b["volume_skew_log"] is not None


def test_ic_and_rank_ic():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rows = []
    for d in dates:
        for i in range(20):
            rows.append(
                {
                    "trade_date": d,
                    "symbol": f"{i:06d}",
                    "f": rng.normal(),
                    "r": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    # 人为让 f 与 r 正相关
    df["r"] = df["f"] * 0.5 + rng.normal(size=len(df)) * 0.1
    ic = information_coefficient(df, "f", "r")
    assert len(ic) > 0
    assert ic_summary(ic)["n"] > 0
    ric = rank_ic(df, "f", "r")
    assert len(ric) == len(ic)


def test_quantile_and_rolling():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    rows = []
    for d in dates:
        for i in range(50):
            rows.append(
                {
                    "trade_date": d,
                    "symbol": f"{i:06d}",
                    "f": rng.normal(),
                    "r": rng.normal() * 0.02,
                }
            )
    df = pd.DataFrame(rows)
    q = quantile_returns(df, "f", "r", n_quantiles=5)
    assert not q.empty
    ic = information_coefficient(df, "f", "r")
    stab = rolling_ic_stability(ic, window=10)
    assert "roll_mean_ic" in stab.columns


def test_neutralize_and_standardize():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
            "symbol": ["000001", "000002", "600000", "600519"] * 2,
            "industry": ["A", "A", "B", "B"] * 2,
            "raw_f": [1.0, 3.0, 2.0, 4.0, 0.0, 2.0, 1.0, 3.0],
        }
    )
    out = attach_neutralized_pair(df, "raw_f", industry_col="industry")
    assert "raw_f_cs_neutral" in out.columns
    assert "raw_f_ind_neutral" in out.columns
    # 同一行业组内和应为 0（浮点误差内）
    g = out.groupby(["trade_date", "industry"])["raw_f_ind_neutral"].sum()
    assert (g.abs() < 1e-9).all()

    std = factor_standardize_pipeline(out, "raw_f", fill="zero", out_col="z")
    assert "z" in std.columns


def test_neutralize_size_industry_regression_handles_all_nan_group_without_warning():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"] * 3),
            "symbol": ["000001", "000002", "000003"],
            "industry": ["A", "A", "B"],
            "log_market_cap": [10.0, 11.0, 12.0],
            "raw_f": [np.nan, np.nan, np.nan],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", category=RuntimeWarning)
        out = neutralize_size_industry_regression(df, "raw_f")

    assert f"raw_f_si_neutral" in out.columns
    assert len(caught) == 0


def test_long_table_from_wide():
    wide = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["000001", "000002"],
        columns=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    lt = long_table_from_wide(wide, "f", arr)
    assert len(lt) == 4
    assert set(lt.columns) == {"symbol", "trade_date", "f"}
