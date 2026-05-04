"""资金流因子族：基于日频主力资金/超大单/大单/中单/小单净流入构造截面因子。"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

DEFAULT_FUND_FLOW_WINDOWS: tuple[int, ...] = (5, 10, 20)
_FLOW_RAW_COLS: tuple[str, ...] = (
    "main_net_inflow_pct",
    "super_large_net_inflow_pct",
    "small_net_inflow_pct",
)


def attach_fund_flow(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_fund_flow",
    windows: tuple[int, ...] = DEFAULT_FUND_FLOW_WINDOWS,
) -> pd.DataFrame:
    """
    从 DuckDB 读取资金流数据，按 (symbol, trade_date) 左连接到因子长表，
    并构造滚动窗口聚合因子。

    新增因子列：
    - ``main_inflow_z_{w}d``: 主力净流入占比的 w 日滚动均值，截面 z-score
    - ``super_inflow_z_{w}d``: 超大单净流入占比的 w 日滚动均值，截面 z-score
    - ``flow_divergence_{w}d``: (主力净流入占比 - 小单净流入占比) 的 w 日滚动均值
    - ``main_inflow_streak``: 主力连续净流入天数（正值为连续流入，负值为连续流出）
    """
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()

    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return _attach_empty_flow_columns(out, windows=windows)

        raw = con.execute(
            f"""
            SELECT symbol, trade_date,
                   main_net_inflow_pct,
                   super_large_net_inflow_pct,
                   small_net_inflow_pct
            FROM {table_name}
            """,
        ).df()
    finally:
        con.close()

    if raw.empty:
        return _attach_empty_flow_columns(out, windows=windows)

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()

    for c in _FLOW_RAW_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["trade_date"])
    raw = raw.drop_duplicates(["symbol", "trade_date"], keep="last")

    out = out.merge(raw, on=["symbol", "trade_date"], how="left")

    out = _compute_flow_factors(out, windows=windows)
    return out


def _compute_flow_factors(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """在已 merge 资金流原始数据的长表上计算滚动因子。"""
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    for w in windows:
        main_roll = (
            df.groupby("symbol", sort=False)["main_net_inflow_pct"]
            .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
        )
        super_roll = (
            df.groupby("symbol", sort=False)["super_large_net_inflow_pct"]
            .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
        )
        div_roll = main_roll - (
            df.groupby("symbol", sort=False)["small_net_inflow_pct"]
            .transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
        )

        df[f"main_inflow_z_{w}d"] = _cross_sectional_z(df, main_roll)
        df[f"super_inflow_z_{w}d"] = _cross_sectional_z(df, super_roll)
        df[f"flow_divergence_{w}d"] = _cross_sectional_z(df, div_roll)

    df["main_inflow_streak"] = _compute_streak(df)

    drop_cols = list(_FLOW_RAW_COLS)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def _attach_empty_flow_columns(df: pd.DataFrame, *, windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy(deep=False)
    for w in windows:
        out[f"main_inflow_z_{w}d"] = np.nan
        out[f"super_inflow_z_{w}d"] = np.nan
        out[f"flow_divergence_{w}d"] = np.nan
    out["main_inflow_streak"] = np.nan
    return out


def _cross_sectional_z(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    """按截面（trade_date）做 z-score 标准化。"""
    grouped = series.groupby(df["trade_date"], sort=False)
    mu = grouped.transform("mean")
    sd = grouped.transform("std", ddof=1)
    sd = sd.replace(0, np.nan)
    z = (series - mu) / sd
    return z.clip(-3.0, 3.0)


def _compute_streak(df: pd.DataFrame) -> pd.Series:
    """
    计算主力净流入连续天数。
    正数 = 连续流入天数，负数 = 连续流出天数。
    """
    streaks = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby("symbol", sort=False):
        values = pd.to_numeric(grp["main_net_inflow_pct"], errors="coerce")
        current_streak = 0
        current_sign = 0
        group_streaks: list[float] = []
        for raw_val in values:
            if pd.isna(raw_val):
                current_streak = 0
                current_sign = 0
                group_streaks.append(0.0)
                continue
            if raw_val > 0:
                if current_sign == 1:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_sign = 1
            else:
                if current_sign == -1:
                    current_streak -= 1
                else:
                    current_streak = -1
                    current_sign = -1
            group_streaks.append(float(current_streak))
        streaks.loc[grp.index] = group_streaks
    return streaks
