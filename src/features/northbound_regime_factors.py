"""北向资金 regime 因子族（M13-C）：市场级北向净流入特征。

数据源: a_share_northbound_aggregate 表 (DuckDB)，由 stock_hsgt_hist_em (AkShare) 填充。
历史起点: 2014-11-17（沪股通），数据无监管截断，持续到当日。

与 northbound_factors.py 的区别：
- northbound_factors: 个股级 Alpha 因子（hold_ratio 等），2024-08 后不可用
- 本模块: 市场级 regime 辅助因子（per-signal_date），无个股维度，全年段可用
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

NORTHBOUND_REGIME_RAW_FEATURES: tuple[str, ...] = (
    "feature_north_net_inflow_1m",
    "feature_north_net_inflow_3m",
    "feature_north_inflow_zscore_6m",
    "feature_north_consecutive_outflow_days",
)

_TRADING_DAYS_1M = 21
_TRADING_DAYS_3M = 63
_TRADING_DAYS_6M = 126


def _signal_date_features(
    aggregate: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """从汇总日线计算每个 signal_date 的市场级 regime 特征。

    对每个 signal_date，使用该日及之前可见的日线数据计算：
    - north_net_inflow_1m: 近 21 个交易日净买入合计（亿元）
    - north_net_inflow_3m: 近 63 个交易日净买入合计（亿元）
    - north_inflow_zscore_6m: 近 1 月净买入相对于近 6 月均值的 z-score
    - north_consecutive_outflow_days: 截至信号日的连续净流出天数
    """
    ts = aggregate.set_index("trade_date").sort_index()
    net = ts["net_buy_total"].fillna(0)

    rows: list[dict] = []
    for sd in signal_dates:
        if pd.isna(sd):
            continue
        before = net[net.index <= sd]
        if len(before) < _TRADING_DAYS_1M:
            continue

        inflow_1m = before.tail(_TRADING_DAYS_1M).sum()
        inflow_3m = before.tail(_TRADING_DAYS_3M).sum() if len(before) >= _TRADING_DAYS_3M else np.nan

        if len(before) >= _TRADING_DAYS_6M:
            hist_1m_series = before.tail(_TRADING_DAYS_6M).rolling(_TRADING_DAYS_1M).sum()
            mean_6m = hist_1m_series.mean()
            std_6m = hist_1m_series.std(ddof=0)
            if np.isfinite(std_6m) and std_6m > 1e-12:
                zscore = (inflow_1m - mean_6m) / std_6m
            else:
                zscore = 0.0
        else:
            zscore = np.nan

        consecutive = 0
        for v in reversed(before.values):
            if v < 0:
                consecutive += 1
            else:
                break

        rows.append({
            "signal_date": sd,
            "feature_north_net_inflow_1m": inflow_1m,
            "feature_north_net_inflow_3m": inflow_3m,
            "feature_north_inflow_zscore_6m": zscore,
            "feature_north_consecutive_outflow_days": float(consecutive),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def attach_northbound_regime_features(
    dataset: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """将北向 regime 特征附加到月度数据集。

    市场级 regime 特征仅按 signal_date 匹配，所有个股获得相同的 regime 值。
    """
    from src.pipeline.monthly_multisource import add_zscore_and_missing_flags

    out = dataset.copy(deep=False)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'a_share_northbound_aggregate'"
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return add_zscore_and_missing_flags(out, NORTHBOUND_REGIME_RAW_FEATURES)

        aggregate = con.execute(
            "SELECT trade_date, net_buy_sh, net_buy_sz, net_buy_total "
            "FROM a_share_northbound_aggregate ORDER BY trade_date"
        ).df()
    finally:
        con.close()

    if aggregate.empty:
        return add_zscore_and_missing_flags(out, NORTHBOUND_REGIME_RAW_FEATURES)

    aggregate["trade_date"] = pd.to_datetime(aggregate["trade_date"], errors="coerce").dt.normalize()
    for c in ["net_buy_sh", "net_buy_sz", "net_buy_total"]:
        aggregate[c] = pd.to_numeric(aggregate[c], errors="coerce")

    sig_dates = pd.to_datetime(out["signal_date"].dropna().unique())
    regime_features = _signal_date_features(aggregate, sig_dates)

    if regime_features.empty:
        return add_zscore_and_missing_flags(out, NORTHBOUND_REGIME_RAW_FEATURES)

    regime_features["signal_date"] = pd.to_datetime(
        regime_features["signal_date"], errors="coerce"
    ).dt.normalize()

    out = out.merge(regime_features, on="signal_date", how="left")
    return add_zscore_and_missing_flags(out, NORTHBOUND_REGIME_RAW_FEATURES)
