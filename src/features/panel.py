"""长表日线 → 宽表收盘价矩阵，供张量因子使用。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def pivot_field_wide(
    df: pd.DataFrame,
    value_col: str,
    *,
    min_valid_days: int = 20,
) -> Tuple[pd.DataFrame, List[str], List[pd.Timestamp]]:
    """
    将 ``symbol, trade_date, <value_col>`` 长表透视为「行=标的、列=交易日」宽表。

    与 ``pivot_close_wide`` 等价于 ``value_col="close"`` 的情形。
    """
    if df.empty:
        raise ValueError("日线长表为空")
    need = {"symbol", "trade_date", value_col}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少列: {need - set(df.columns)}")

    sub = df[["symbol", "trade_date", value_col]].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.normalize()
    wide = sub.pivot_table(
        index="symbol",
        columns="trade_date",
        values=value_col,
        aggfunc="last",
    )
    wide = wide.sort_index(axis=1)
    valid = wide.notna().sum(axis=1) >= min_valid_days
    wide = wide.loc[valid]
    if wide.empty:
        raise ValueError("过滤 min_valid_days 后无可用标的")

    symbols = wide.index.astype(str).tolist()
    dates = list(wide.columns)
    return wide, symbols, dates


def pivot_close_wide(
    df: pd.DataFrame,
    *,
    min_valid_days: int = 20,
) -> Tuple[pd.DataFrame, List[str], List[pd.Timestamp]]:
    """
    将 ``symbol, trade_date, close`` 长表透视为「行=标的、列=交易日」的收盘价宽表。

    Parameters
    ----------
    df : DataFrame
        须含 ``symbol``, ``trade_date``, ``close``。
    min_valid_days : int
        至少有多少个有效收盘日才保留该标的（过滤新股/缺失过多）。

    Returns
    -------
    wide : DataFrame
        索引为 ``symbol``，列为有序 ``trade_date``。
    symbols : list of str
    dates : list
        与宽表列对齐的交易日序列。
    """
    return pivot_field_wide(df, "close", min_valid_days=min_valid_days)


def wide_close_to_numpy(
    wide: pd.DataFrame,
) -> np.ndarray:
    """宽表 → ``(num_symbols, num_days)`` float64，含 NaN。"""
    return wide.to_numpy(dtype=np.float64)


def pivot_field_aligned_to_close(
    df: pd.DataFrame,
    value_col: str,
    ref_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 ``symbol, trade_date, value_col`` 透视为宽表，并按 ``ref_wide`` 的索引与列对齐（缺失为 NaN）。

    用于在 ``pivot_close_wide`` 之后拉齐 OHLCV、换手率等，保证与收盘价矩阵同形。
    """
    if value_col not in df.columns:
        return pd.DataFrame(
            index=ref_wide.index,
            columns=ref_wide.columns,
            dtype=np.float64,
        )
    sub = df[["symbol", "trade_date", value_col]].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.normalize()
    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        return pd.DataFrame(
            index=ref_wide.index,
            columns=ref_wide.columns,
            dtype=np.float64,
        )
    w = sub.pivot_table(
        index="symbol",
        columns="trade_date",
        values=value_col,
        aggfunc="last",
    )
    w = w.sort_index(axis=1)
    return w.reindex(index=ref_wide.index, columns=ref_wide.columns)


def wide_close_to_numpy_filled(
    wide: pd.DataFrame,
    *,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 ``wide_close_to_numpy`` 相同形状；另返回 ``valid`` 掩码（收盘价非 NaN）。

    ``fill_value`` 仅用于占位；默认保持 NaN 以便因子函数自行处理。
    """
    arr = wide.to_numpy(dtype=np.float64)
    valid = np.isfinite(arr)
    return arr, valid
