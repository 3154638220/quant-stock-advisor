"""概念/主题因子族：基于概念板块日线及成分股快照构造月度截面因子。

数据源:
    - a_share_concept_daily: 同花顺概念板块指数
    - a_share_concept_membership: 当前成分股快照（按 snapshot_date 使用）
PIT-safety:
    - 板块日线 OHLCV 为 T 日收盘后可见
    - 成分股只在 snapshot_date <= signal_date 且 snapshot_age <= 60 天时使用

Factor list:
    - feature_concept_breadth_1m      概念板块近月上涨占比
    - feature_concept_dispersion_1m   概念板块近月收益率截面标准差
    - feature_concept_momentum_1m     概念板块近月平均收益
    - feature_concept_member_count    个股所属概念数量
    - feature_hot_concept_membership   是否属于近月热门概念 Top-10
    - feature_concept_ew_return_1m     所属概念近月收益均值
    - feature_concept_max_return_1m    所属概念近月最高收益
    - feature_concept_inflow_breadth   所属概念中资金净流入成员占比均值
    - feature_concept_return_dispersion 所属概念近月收益离散度
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

CONCEPT_RAW_FEATURES: tuple[str, ...] = (
    "feature_concept_breadth_1m",
    "feature_concept_dispersion_1m",
    "feature_concept_momentum_1m",
    "feature_concept_member_count",
    "feature_hot_concept_membership",
    "feature_concept_ew_return_1m",
    "feature_concept_max_return_1m",
    "feature_concept_inflow_breadth",
    "feature_concept_return_dispersion",
)

MEMBERSHIP_MAX_AGE_DAYS = 60


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


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(row and int(row[0]) > 0)


def _available_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    if not _table_exists(con, table):
        return set()
    info = con.execute(f"PRAGMA table_info('{table}')").df()
    return set(info["name"].astype(str))


def _read_membership(
    con: duckdb.DuckDBPyConnection,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> pd.DataFrame:
    cols = _available_columns(con, "a_share_concept_membership")
    if not cols:
        return pd.DataFrame()

    concept_name_expr = "concept_name" if "concept_name" in cols else "NULL::VARCHAR AS concept_name"
    entry_expr = "entry_date" if "entry_date" in cols else "snapshot_date AS entry_date"
    exit_expr = "exit_date" if "exit_date" in cols else "NULL::DATE AS exit_date"
    membership = con.execute(
        f"""
        SELECT symbol, concept_code, snapshot_date, {concept_name_expr}, {entry_expr}, {exit_expr}
        FROM a_share_concept_membership
        WHERE snapshot_date >= ? AND snapshot_date <= ?
        """,
        [date_min - pd.Timedelta(days=MEMBERSHIP_MAX_AGE_DAYS), date_max],
    ).df()
    if membership.empty:
        return membership
    membership["symbol"] = (
        membership["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    )
    for c in ["snapshot_date", "entry_date", "exit_date"]:
        membership[c] = pd.to_datetime(membership[c], errors="coerce").dt.normalize()
    return membership.dropna(subset=["snapshot_date"])


def _concept_window_returns(daily: pd.DataFrame, sd_ts: pd.Timestamp) -> pd.Series:
    window = daily[
        (daily["trade_date"] > sd_ts - pd.Timedelta(days=35))
        & (daily["trade_date"] <= sd_ts)
    ]
    if window.empty:
        return pd.Series(dtype=float)
    cum_ret = (
        window.sort_values("trade_date")
        .groupby("concept_code")
        .apply(
            lambda g: (pd.to_numeric(g["pct_chg"], errors="coerce").fillna(0) / 100 + 1).prod() - 1,
            include_groups=False,
        )
    )
    return cum_ret.replace([np.inf, -np.inf], np.nan).dropna()


def _read_fund_flow(
    con: duckdb.DuckDBPyConnection,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> pd.DataFrame:
    if not _table_exists(con, "a_share_fund_flow"):
        return pd.DataFrame()
    cols = _available_columns(con, "a_share_fund_flow")
    if "main_net_inflow_pct" not in cols:
        return pd.DataFrame()
    flow = con.execute(
        """
        SELECT symbol, trade_date, main_net_inflow_pct
        FROM a_share_fund_flow
        WHERE trade_date >= ? AND trade_date <= ?
        """,
        [date_min - pd.Timedelta(days=35), date_max],
    ).df()
    if flow.empty:
        return flow
    flow["symbol"] = flow["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    flow["trade_date"] = pd.to_datetime(flow["trade_date"], errors="coerce").dt.normalize()
    flow["main_net_inflow_pct"] = pd.to_numeric(flow["main_net_inflow_pct"], errors="coerce")
    return flow.dropna(subset=["trade_date"])


def _concept_inflow_breadth(
    active_membership: pd.DataFrame,
    fund_flow: pd.DataFrame,
    sd_ts: pd.Timestamp,
) -> pd.DataFrame:
    if active_membership.empty or fund_flow.empty:
        return pd.DataFrame(columns=["concept_code", "concept_inflow_breadth"])
    recent = fund_flow[
        (fund_flow["trade_date"] > sd_ts - pd.Timedelta(days=35))
        & (fund_flow["trade_date"] <= sd_ts)
    ]
    if recent.empty:
        return pd.DataFrame(columns=["concept_code", "concept_inflow_breadth"])
    symbol_flow = (
        recent.groupby("symbol", sort=False)["main_net_inflow_pct"]
        .mean()
        .reset_index()
    )
    symbol_flow["_positive_inflow"] = (symbol_flow["main_net_inflow_pct"] > 0).astype(float)
    joined = active_membership[["symbol", "concept_code"]].merge(symbol_flow, on="symbol", how="left")
    return (
        joined.groupby("concept_code", sort=False)["_positive_inflow"]
        .mean()
        .rename("concept_inflow_breadth")
        .reset_index()
    )


def _get_membership_features(
    con: duckdb.DuckDBPyConnection,
    signal_dates: pd.Series,
) -> pd.DataFrame:
    """从 a_share_concept_membership 计算 M13-B 个股绑定概念特征。"""
    all_dates = sorted(signal_dates.dropna().unique())
    if len(all_dates) == 0:
        return pd.DataFrame()

    date_min = pd.Timestamp(all_dates[0]).normalize()
    date_max = pd.Timestamp(all_dates[-1]).normalize()
    membership = _read_membership(con, date_min, date_max)
    if membership.empty:
        return pd.DataFrame()

    daily = con.execute(
        """
        SELECT concept_code, trade_date, pct_chg
        FROM a_share_concept_daily
        WHERE trade_date >= ? AND trade_date <= ?
        """,
        [date_min - pd.Timedelta(days=35), date_max],
    ).df()
    if daily.empty:
        return pd.DataFrame()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    daily["pct_chg"] = pd.to_numeric(daily["pct_chg"], errors="coerce")
    daily = daily.dropna(subset=["trade_date"])
    fund_flow = _read_fund_flow(con, date_min, date_max)

    rows: list[pd.DataFrame] = []
    for sd in all_dates:
        sd_ts = pd.Timestamp(sd).normalize()
        active = membership[
            (membership["snapshot_date"] <= sd_ts)
            & (membership["snapshot_date"] > sd_ts - pd.Timedelta(days=MEMBERSHIP_MAX_AGE_DAYS))
            & (membership["entry_date"].fillna(membership["snapshot_date"]) <= sd_ts)
            & (membership["exit_date"].isna() | (membership["exit_date"] > sd_ts))
        ].copy()
        if active.empty:
            continue
        active = (
            active.sort_values(["symbol", "concept_code", "snapshot_date"], kind="mergesort")
            .drop_duplicates(["symbol", "concept_code"], keep="last")
        )
        concept_ret = _concept_window_returns(daily, sd_ts)
        if concept_ret.empty:
            continue
        concept_ret_df = concept_ret.rename("concept_return_1m").reset_index()
        hot_concepts = set(concept_ret.sort_values(ascending=False).head(10).index.astype(str))
        active = active.merge(concept_ret_df, on="concept_code", how="left")
        active["hot_concept"] = active["concept_code"].astype(str).isin(hot_concepts).astype(float)

        inflow = _concept_inflow_breadth(active, fund_flow, sd_ts)
        if not inflow.empty:
            active = active.merge(inflow, on="concept_code", how="left")
        else:
            active["concept_inflow_breadth"] = np.nan

        agg = (
            active.groupby("symbol", sort=False)
            .agg(
                feature_concept_member_count=("concept_code", "nunique"),
                feature_hot_concept_membership=("hot_concept", "max"),
                feature_concept_ew_return_1m=("concept_return_1m", "mean"),
                feature_concept_max_return_1m=("concept_return_1m", "max"),
                feature_concept_inflow_breadth=("concept_inflow_breadth", "mean"),
                feature_concept_return_dispersion=("concept_return_1m", "std"),
            )
            .reset_index()
        )
        agg["feature_concept_return_dispersion"] = agg["feature_concept_return_dispersion"].fillna(0.0)
        agg["signal_date"] = sd_ts
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def attach_concept_features(
    dataset: pd.DataFrame,
    db_path: str,
) -> pd.DataFrame:
    """将概念板块市场级与个股绑定因子附加到月度数据集。

    个股绑定因子遵守 rolling 60 天快照窗口：没有历史成分快照时，历史
    信号日保持缺失，不使用当前成分回填。
    """
    from src.pipeline.monthly_multisource import add_zscore_and_missing_flags

    out = dataset.copy(deep=False)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    con = duckdb.connect(db_path, read_only=True)
    try:
        if not _table_exists(con, "a_share_concept_daily"):
            return add_zscore_and_missing_flags(out, CONCEPT_RAW_FEATURES)

        mkt_features = _get_market_breadth_features(con, out["signal_date"])
        membership_features = _get_membership_features(con, out["signal_date"])
    finally:
        con.close()

    if not mkt_features.empty:
        mkt_features["signal_date"] = pd.to_datetime(
            mkt_features["signal_date"], errors="coerce"
        ).dt.normalize()
        out = out.merge(mkt_features, on="signal_date", how="left")

    if not membership_features.empty:
        membership_features["signal_date"] = pd.to_datetime(
            membership_features["signal_date"], errors="coerce"
        ).dt.normalize()
        out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
        membership_features["symbol"] = (
            membership_features["symbol"]
            .astype(str)
            .str.extract(r"(\d{1,6})", expand=False)
            .fillna("")
            .str.zfill(6)
        )
        out = out.merge(membership_features, on=["symbol", "signal_date"], how="left")

    return add_zscore_and_missing_flags(out, CONCEPT_RAW_FEATURES)
