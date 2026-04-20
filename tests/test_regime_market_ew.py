"""P0-C：全市场等权 proxy 与日收益聚合。"""
from __future__ import annotations

import pandas as pd

from src.market.regime import MARKET_EW_PROXY, _market_ew_daily_returns_from_frame


def test_market_ew_proxy_constant():
    assert MARKET_EW_PROXY == "market_ew_proxy"


def test_market_ew_daily_returns_mean_cross_section():
    df = pd.DataFrame(
        {
            "symbol": ["1", "1", "2", "2"],
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03"] * 2),
            "close": [10.0, 11.0, 20.0, 19.0],
        }
    )
    # 各标的仅一条有效日收益；min_symbol_obs=1 保留截面
    s = _market_ew_daily_returns_from_frame(df, min_symbol_obs=1)
    assert len(s) == 1
    r1 = 11.0 / 10.0 - 1.0
    r2 = 19.0 / 20.0 - 1.0
    assert abs(float(s.iloc[0]) - (r1 + r2) / 2.0) < 1e-12
