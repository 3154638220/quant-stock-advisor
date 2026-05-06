"""北向资金因子族：基于日频北向持股/净买入构造月度截面因子。

数据源: a_share_northbound 表 (DuckDB)，由 stock_hsgt_individual_em (AkShare) 填充。
历史起点: 2017-03-17。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

NORTHBOUND_RAW_FEATURES: tuple[str, ...] = (
    "feature_northbound_hold_ratio",
    "feature_northbound_net_buy_1m",
    "feature_northbound_hold_change_1m",
    "feature_northbound_inflow_stability_1m",
)

_NB_TABLE = "a_share_northbound"


def attach_northbound_features(
    dataset: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = _NB_TABLE,
) -> pd.DataFrame:
    """从 DuckDB 读取北向资金日频数据，按 (symbol, signal_date) 构建月度因子。

    新增因子列：
    - feature_northbound_hold_ratio: 最近一日北向持股占比
    - feature_northbound_net_buy_1m: 近 20 个交易日北向净买入合计（元）
    - feature_northbound_hold_change_1m: 近 20 个交易日持股占比变化
    - feature_northbound_inflow_stability_1m: 近 20 个交易日净买入为正的天数占比
    """
    from src.pipeline.monthly_multisource import add_zscore_and_missing_flags

    out = dataset.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)

    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return add_zscore_and_missing_flags(out, NORTHBOUND_RAW_FEATURES)

        raw = con.execute(
            f"""
            SELECT symbol, trade_date, hold_amount, hold_ratio, net_buy_amount, hold_shares
            FROM {table_name}
            ORDER BY symbol, trade_date
            """,
        ).df()
    finally:
        con.close()

    if raw.empty:
        return add_zscore_and_missing_flags(out, NORTHBOUND_RAW_FEATURES)

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    for c in ["hold_amount", "hold_ratio", "net_buy_amount", "hold_shares"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["trade_date"])
    raw = raw.sort_values(["symbol", "trade_date"])

    signal_dates = (
        out[["signal_date", "symbol"]]
        .drop_duplicates()
        .assign(signal_date=lambda x: pd.to_datetime(x["signal_date"], errors="coerce").dt.normalize())
    )

    rows: list[dict] = []
    for symbol, grp in raw.groupby("symbol"):
        if symbol not in signal_dates["symbol"].values:
            continue
        sym_signal_dates = signal_dates[signal_dates["symbol"] == symbol]["signal_date"]
        grp = grp.set_index("trade_date").sort_index()
        for sd in sym_signal_dates:
            if pd.isna(sd):
                continue
            window = grp[grp.index <= sd].tail(21)
            if len(window) < 5:
                continue
            latest = window.iloc[-1]
            rows.append({
                "signal_date": sd,
                "symbol": symbol,
                "feature_northbound_hold_ratio": latest["hold_ratio"],
                "feature_northbound_net_buy_1m": window["net_buy_amount"].tail(20).sum(),
                "feature_northbound_hold_change_1m": (
                    latest["hold_ratio"]
                    - window["hold_ratio"].iloc[0]
                    if len(window) > 1
                    else np.nan
                ),
                "feature_northbound_inflow_stability_1m": (
                    (window["net_buy_amount"].tail(20) > 0).mean()
                ),
            })

    if not rows:
        return add_zscore_and_missing_flags(out, NORTHBOUND_RAW_FEATURES)

    nb_df = pd.DataFrame(rows)
    nb_df["signal_date"] = pd.to_datetime(nb_df["signal_date"], errors="coerce").dt.normalize()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    out = out.merge(nb_df, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, NORTHBOUND_RAW_FEATURES)
