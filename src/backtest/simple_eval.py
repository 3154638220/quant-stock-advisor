"""简单事后评估：从日线长表计算未来若干日收益（基线，非完整回测引擎）。

默认与 A 股 T+1 及「次日开盘买入」对齐：``open(T+1+H)/open(T+1)-1``。可选 ``close_to_close`` 保留旧口径。
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.market.tradability import (
    is_open_limit_up_unbuyable,
    is_row_suspended_like,
)


def _col_one(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"需要以下任一列: {candidates}，当前列: {list(df.columns)}")


def forward_close_return(
    df: pd.DataFrame,
    symbol: str,
    from_date: Union[str, pd.Timestamp],
    horizon_days: int = 5,
    *,
    price_col: str = "close",
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> Optional[float]:
    """
    单标的：从 ``from_date`` 的收盘价到其后第 ``horizon_days`` 根 **交易日** 收盘的累计收益。

    若数据不足则返回 ``None``。
    """
    if horizon_days < 1:
        raise ValueError("horizon_days 须 >= 1")
    sub = df[df[sym_col].astype(str) == str(symbol)].copy()
    if sub.empty:
        return None
    sub[date_col] = pd.to_datetime(sub[date_col]).dt.normalize()
    sub = sub.sort_values(date_col)
    t0 = pd.Timestamp(from_date).normalize()
    idx = sub[date_col].searchsorted(t0, side="left")
    if idx >= len(sub) or sub.iloc[idx][date_col] != t0:
        return None
    j = idx + horizon_days
    if j >= len(sub):
        return None
    p0 = float(sub.iloc[idx][price_col])
    p1 = float(sub.iloc[j][price_col])
    if not np.isfinite(p0) or p0 == 0.0:
        return None
    return p1 / p0 - 1.0


def forward_tplus1_open_return(
    df: pd.DataFrame,
    symbol: str,
    from_date: Union[str, pd.Timestamp],
    horizon_days: int = 5,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    exclude_unbuyable_on_entry: bool = True,
) -> Optional[float]:
    """
    单标的：信号日 ``from_date``（T 收盘）后，T+1 开盘买入，持有 ``horizon_days`` 个开盘到开盘区间，
    收益 ``open(T+1+h)/open(T+1)-1``。若次日停牌或次日开盘一字涨停（难以买入），返回 ``None``。
    """
    if horizon_days < 1:
        raise ValueError("horizon_days 须 >= 1")
    sub = df[df[sym_col].astype(str) == str(symbol)].copy()
    if sub.empty:
        return None
    sub[date_col] = pd.to_datetime(sub[date_col]).dt.normalize()
    sub = sub.sort_values(date_col)
    t0 = pd.Timestamp(from_date).normalize()
    idx = sub[date_col].searchsorted(t0, side="left")
    if idx >= len(sub) or sub.iloc[idx][date_col] != t0:
        return None
    j = idx + 1 + horizon_days
    if j >= len(sub):
        return None
    r0 = sub.iloc[idx]
    r_entry = sub.iloc[idx + 1]
    r_exit = sub.iloc[j]
    prev_close = float(r0.get("close", np.nan))
    o_entry = float(r_entry.get("open", np.nan))
    c_entry = float(r_entry.get("close", np.nan))
    v_entry = float(r_entry.get("volume", np.nan))
    o_exit = float(r_exit.get("open", np.nan))
    if exclude_unbuyable_on_entry:
        if is_row_suspended_like(v_entry, o_entry, c_entry):
            return None
        if is_open_limit_up_unbuyable(o_entry, prev_close, str(symbol)):
            return None
    if not np.isfinite(o_entry) or o_entry == 0.0 or not np.isfinite(o_exit):
        return None
    return o_exit / o_entry - 1.0


def attach_forward_returns(
    rec_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    horizon_days: int = 5,
    symbol_col: Optional[str] = None,
    asof_col: Optional[str] = None,
    price_col: str = "close",
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    settlement: Literal["tplus1_open", "close_to_close"] = "tplus1_open",
) -> pd.DataFrame:
    """
    在推荐表上追加 ``forward_ret_{horizon}`` 列。

    settlement
        ``tplus1_open``（默认）：``open(T+1+h)/open(T+1)-1``，次日一字涨停/停牌无法买入时为 ``NaN``。
        ``close_to_close``：``close(T+h)/close(T)-1``（旧口径，可能高估 T+0 日内可套利）。
    """
    if rec_df.empty:
        return rec_df.copy()
    out = rec_df.copy()
    sym_c = symbol_col or _col_one(out, ("symbol", "代码"))
    asof_c = asof_col or _col_one(out, ("asof_trade_date", "trade_date", "date"))
    col_name = f"forward_ret_{horizon_days}d"
    rets: list[Optional[float]] = []
    for _, row in out.iterrows():
        sym = str(row[sym_c])
        asof = row[asof_c]
        if settlement == "close_to_close":
            r = forward_close_return(
                daily_df,
                sym,
                asof,
                horizon_days=horizon_days,
                price_col=price_col,
                date_col=date_col,
                sym_col=sym_col,
            )
        else:
            r = forward_tplus1_open_return(
                daily_df,
                sym,
                asof,
                horizon_days=horizon_days,
                date_col=date_col,
                sym_col=sym_col,
            )
        rets.append(r)
    out[col_name] = rets
    return out


def summarize_forward_returns(rec_with_fwd: pd.DataFrame, *, forward_col: str) -> dict:
    """对含前向收益列的推荐表做简单汇总（均值、有效样本数）。"""
    s = pd.to_numeric(rec_with_fwd[forward_col], errors="coerce")
    valid = s.dropna()
    return {
        "n": int(len(rec_with_fwd)),
        "n_valid": int(valid.shape[0]),
        "mean": float(valid.mean()) if len(valid) else float("nan"),
        "median": float(valid.median()) if len(valid) else float("nan"),
    }
