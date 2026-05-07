"""龙虎榜 (LHB) 因子族：基于新浪 LHB 日明细构造月度截面因子。

数据源: a_share_lhb_daily 表 (DuckDB)，新浪财经龙虎榜每日明细。
PIT-safety: LHB 明细为 T 日收盘后披露 → PIT-safe。

Factor list (per-stock, rolling window):
    - feature_lhb_appearance_count_1m    近月上榜次数
    - feature_lhb_appearance_count_3m    近季上榜次数
    - feature_lhb_recent_5d             近 5 日是否上榜 (0/1)
    - feature_lhb_avg_change_1m         上榜日平均涨跌幅
    - feature_lhb_avg_amount_1m         上榜日平均成交额(万元)
    - feature_lhb_is_bullish_1m         近月上榜原因含看涨信号 (0/1)
    - feature_lhb_is_bearish_1m         近月上榜原因含看跌信号 (0/1)
    - feature_lhb_is_high_turnover_1m   近月因高换手上榜 (0/1)
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

LHB_RAW_FEATURES: tuple[str, ...] = (
    "feature_lhb_appearance_count_1m",
    "feature_lhb_appearance_count_3m",
    "feature_lhb_recent_5d",
    "feature_lhb_avg_change_1m",
    "feature_lhb_avg_amount_1m",
    "feature_lhb_is_bullish_1m",
    "feature_lhb_is_bearish_1m",
    "feature_lhb_is_high_turnover_1m",
)

_BULLISH_KEYWORDS = [
    "涨幅偏离值达7%",
    "连续三个交易日内，涨幅偏离值累计达20%",
    "连续三个交易日内，涨幅偏离值累计达到12%的ST证券",
    "连续三个交易日内，涨幅偏离值累计达到15%的ST证券",
]

_BEARISH_KEYWORDS = [
    "跌幅偏离值达7%",
    "连续三个交易日内，跌幅偏离值累计达20%",
    "连续三个交易日内，跌幅偏离值累计达到12%的ST证券",
    "连续三个交易日内，跌幅偏离值累计达到15%的ST证券",
]

_HIGH_TURNOVER_KEYWORDS = [
    "换手率达20%",
    "连续三个交易日内，日均换手率",
]


def _get_lhb_features(
    con: duckdb.DuckDBPyConnection,
    signal_dates: pd.Series,
) -> pd.DataFrame:
    """从 a_share_lhb_daily 计算个股 LHB 特征。"""
    all_dates = sorted(signal_dates.dropna().unique())
    if len(all_dates) == 0:
        return pd.DataFrame()

    date_min = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=120)
    date_max = pd.Timestamp(all_dates[-1])

    daily = con.execute(
        """
        SELECT symbol, data_date, change_pct, amount_wan, lhb_reason
        FROM a_share_lhb_daily
        WHERE data_date >= ? AND data_date <= ?
        ORDER BY symbol, data_date
        """,
        [date_min, date_max],
    ).df()

    if daily.empty:
        return pd.DataFrame()

    daily["data_date"] = pd.to_datetime(daily["data_date"], errors="coerce")
    daily["change_pct"] = pd.to_numeric(daily["change_pct"], errors="coerce")
    daily["amount_wan"] = pd.to_numeric(daily["amount_wan"], errors="coerce")
    daily = daily.dropna(subset=["data_date"])

    rows: list[dict] = []
    for sd in all_dates:
        sd_ts = pd.Timestamp(sd)
        # 1-month window: ~21 trading days
        window_1m_start = sd_ts - pd.Timedelta(days=35)
        # 3-month window: ~63 trading days
        window_3m_start = sd_ts - pd.Timedelta(days=100)
        # 5-day window
        window_5d_start = sd_ts - pd.Timedelta(days=10)

        recent_1m = daily[
            (daily["data_date"] > window_1m_start) & (daily["data_date"] <= sd_ts)
        ]
        recent_3m = daily[
            (daily["data_date"] > window_3m_start) & (daily["data_date"] <= sd_ts)
        ]
        recent_5d = daily[
            (daily["data_date"] > window_5d_start) & (daily["data_date"] <= sd_ts)
        ]

        # Per-symbol aggregations
        agg_1m = (
            recent_1m.groupby("symbol")
            .agg(
                appearance_count_1m=("data_date", "count"),
                avg_change_1m=("change_pct", "mean"),
                avg_amount_1m=("amount_wan", "mean"),
            )
            .reset_index()
        )

        agg_3m = (
            recent_3m.groupby("symbol")
            .agg(appearance_count_3m=("data_date", "count"))
            .reset_index()
        )

        agg_5d = recent_5d[["symbol"]].drop_duplicates().copy()
        agg_5d["recent_5d"] = 1

        # Reason-based flags
        def _any_match(grp, keywords):
            return int(any(
                any(kw in str(r) for kw in keywords)
                for r in grp["lhb_reason"].dropna()
            ))

        if len(recent_1m) > 0:
            reason_flags = (
                recent_1m.groupby("symbol")
                .apply(
                    lambda g: pd.Series({
                        "is_bullish_1m": _any_match(g, _BULLISH_KEYWORDS),
                        "is_bearish_1m": _any_match(g, _BEARISH_KEYWORDS),
                        "is_high_turnover_1m": _any_match(g, _HIGH_TURNOVER_KEYWORDS),
                    }, dtype=int),
                    include_groups=False,
                )
                .reset_index()
            )
        else:
            reason_flags = pd.DataFrame(columns=["symbol"])

        # Merge all
        combined = agg_1m.merge(agg_3m, on="symbol", how="outer")
        combined = combined.merge(agg_5d, on="symbol", how="left")
        combined["recent_5d"] = combined["recent_5d"].fillna(0).astype(int)
        if not reason_flags.empty:
            combined = combined.merge(reason_flags, on="symbol", how="left")

        for col in ["is_bullish_1m", "is_bearish_1m", "is_high_turnover_1m"]:
            if col not in combined.columns:
                combined[col] = 0
            combined[col] = combined[col].fillna(0).astype(int)

        combined["signal_date"] = sd_ts
        rows.append(combined)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={
        "appearance_count_1m": "feature_lhb_appearance_count_1m",
        "appearance_count_3m": "feature_lhb_appearance_count_3m",
        "recent_5d": "feature_lhb_recent_5d",
        "avg_change_1m": "feature_lhb_avg_change_1m",
        "avg_amount_1m": "feature_lhb_avg_amount_1m",
        "is_bullish_1m": "feature_lhb_is_bullish_1m",
        "is_bearish_1m": "feature_lhb_is_bearish_1m",
        "is_high_turnover_1m": "feature_lhb_is_high_turnover_1m",
    })
    for col in LHB_RAW_FEATURES:
        if col not in result.columns:
            result[col] = 0

    return result


def attach_lhb_features(
    dataset: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """将 LHB 因子附加到月度数据集。"""
    from src.pipeline.monthly_multisource import add_zscore_and_missing_flags

    out = dataset.copy(deep=False)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    con = duckdb.connect(db_path, read_only=True)
    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'a_share_lhb_daily'"
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return add_zscore_and_missing_flags(out, LHB_RAW_FEATURES)

        features = _get_lhb_features(con, out["signal_date"])
    finally:
        con.close()

    if features.empty:
        return add_zscore_and_missing_flags(out, LHB_RAW_FEATURES)

    features["signal_date"] = pd.to_datetime(
        features["signal_date"], errors="coerce"
    ).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    features["symbol"] = features["symbol"].astype(str).str.zfill(6)
    out = out.merge(features, on=["symbol", "signal_date"], how="left")

    return add_zscore_and_missing_flags(out, LHB_RAW_FEATURES)
