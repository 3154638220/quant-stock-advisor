"""概念/主题 breadth 因子族：基于概念板块日线构造月度截面因子。

数据源: a_share_concept_daily 表 (DuckDB)，同花顺概念板块指数。
当前阶段: 仅市场级特征（板块层面 breadth），不做个股-板块绑定。
PIT-safety: 板块日线 OHLCV 为 T 日收盘后可见 → PIT-safe。

Factor list (all market-level, same value for all stocks on a given signal_date):
    - feature_concept_breadth_1m      概念板块近月上涨占比
    - feature_concept_dispersion_1m   概念板块近月收益率截面标准差
    - feature_concept_momentum_1m     概念板块近月平均收益
    - feature_concept_breadth_rank    概念 breadth 的百分位排名（相对该信号日前的历史）
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

CONCEPT_RAW_FEATURES: tuple[str, ...] = (
    "feature_concept_breadth_1m",
    "feature_concept_dispersion_1m",
    "feature_concept_momentum_1m",
)


def _get_market_breadth_features(
    con: duckdb.DuckDBPyConnection,
    signal_dates: pd.Series,
) -> pd.DataFrame:
    """从 a_share_concept_daily 计算市场级概念 breadth 特征。

    对每个 signal_date，回溯约 21 个交易日，计算：
    - concept_breadth_1m: 近月上涨概念占比（累积收益 > 0 的概念比例）
    - concept_dispersion_1m: 概念近月收益率截面标准差
    - concept_momentum_1m: 概念近月收益率截面均值
    """
    all_dates = sorted(signal_dates.dropna().unique())
    if len(all_dates) == 0:
        return pd.DataFrame()

    # 一次性拉取所有概念日线数据（远比逐月查询高效）
    date_min = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=60)
    date_max = pd.Timestamp(all_dates[-1])

    daily = con.execute(
        """
        SELECT concept_code, trade_date, close, pct_chg
        FROM a_share_concept_daily
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY concept_code, trade_date
        """,
        [date_min, date_max],
    ).df()

    if daily.empty:
        return pd.DataFrame()

    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["pct_chg"] = pd.to_numeric(daily["pct_chg"], errors="coerce")
    daily = daily.dropna(subset=["trade_date"])

    rows: list[dict] = []
    for sd in all_dates:
        sd_ts = pd.Timestamp(sd)
        start_ts = sd_ts - pd.Timedelta(days=35)

        window = daily[
            (daily["trade_date"] > start_ts)
            & (daily["trade_date"] <= sd_ts)
        ]
        if window.empty:
            continue

        # 对每个概念，计算窗口内累积收益
        cum_ret = (
            window.sort_values("trade_date")
            .groupby("concept_code")
            .apply(
                lambda g: (g["pct_chg"].fillna(0) / 100 + 1).prod() - 1,
                include_groups=False,
            )
        )
        cum_ret = cum_ret.replace([np.inf, -np.inf], np.nan).dropna()

        if len(cum_ret) < 5:
            continue

        breadth = (cum_ret > 0).mean()
        dispersion = cum_ret.std()
        momentum = cum_ret.mean()

        rows.append({
            "signal_date": sd_ts,
            "feature_concept_breadth_1m": breadth,
            "feature_concept_dispersion_1m": dispersion,
            "feature_concept_momentum_1m": momentum,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def attach_concept_features(
    dataset: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """将概念板块市场级因子附加到月度数据集。

    新增因子列: feature_concept_breadth_1m, feature_concept_dispersion_1m,
    feature_concept_momentum_1m.
    """
    from src.pipeline.monthly_multisource import add_zscore_and_missing_flags

    out = dataset.copy(deep=False)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'a_share_concept_daily'"
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return add_zscore_and_missing_flags(out, CONCEPT_RAW_FEATURES)

        mkt_features = _get_market_breadth_features(con, out["signal_date"])
    finally:
        con.close()

    if mkt_features.empty:
        return add_zscore_and_missing_flags(out, CONCEPT_RAW_FEATURES)

    mkt_features["signal_date"] = pd.to_datetime(
        mkt_features["signal_date"], errors="coerce"
    ).dt.normalize()
    out = out.merge(mkt_features, on="signal_date", how="left")

    return add_zscore_and_missing_flags(out, CONCEPT_RAW_FEATURES)
