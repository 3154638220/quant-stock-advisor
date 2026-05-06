"""融资融券因子族：基于日频融资融券数据构造月度截面因子。

数据源: a_share_margin_trading 表 (DuckDB)，由 stock_margin_detail_sse/szse (AkShare) 填充。
历史起点: 约 2012 年。

候选因子：
- feature_margin_fin_balance_ratio: 融资余额/总市值 (近似市值 = fin_balance / total_balance)
- feature_margin_net_fin_buy_1m: 近 20 个交易日融资买入额合计 - 偿还额合计 (仅 SSE)
- feature_margin_short_pressure_1m: 近 20 个交易日融券余量变化率
- feature_margin_fin_balance_momentum_1m: 近 20 个交易日融资余额增长率
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

MARGIN_TRADING_RAW_FEATURES: tuple[str, ...] = (
    "feature_margin_fin_balance_ratio",
    "feature_margin_net_fin_buy_1m",
    "feature_margin_short_pressure_1m",
    "feature_margin_fin_balance_momentum_1m",
)

_MT_TABLE = "a_share_margin_trading"


def attach_margin_trading_features(
    dataset: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = _MT_TABLE,
) -> pd.DataFrame:
    """从 DuckDB 读取融资融券日频数据，按 (symbol, signal_date) 构建月度因子。

    新增因子列：
    - feature_margin_fin_balance_ratio: 融资余额 / 融资融券余额（近似杠杆占比）
    - feature_margin_net_fin_buy_1m: 近 20 个交易日融资买入净额
    - feature_margin_short_pressure_1m: 近 20 个交易日融券余量变化率
    - feature_margin_fin_balance_momentum_1m: 近 20 个交易日融资余额增长率
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
            return add_zscore_and_missing_flags(out, MARGIN_TRADING_RAW_FEATURES)

        raw = con.execute(
            f"""
            SELECT symbol, trade_date, fin_balance, fin_buy_amount,
                   short_volume, short_sell_volume, fin_repay_amount
            FROM {table_name}
            ORDER BY symbol, trade_date
            """,
        ).df()
    finally:
        con.close()

    if raw.empty:
        return add_zscore_and_missing_flags(out, MARGIN_TRADING_RAW_FEATURES)

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    for c in ["fin_balance", "fin_buy_amount", "short_volume",
              "short_sell_volume", "fin_repay_amount"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["trade_date"])
    raw = raw.sort_values(["symbol", "trade_date"])

    signal_dates = (
        out[["signal_date", "symbol"]]
        .drop_duplicates()
        .assign(signal_date=lambda x: pd.to_datetime(x["signal_date"], errors="coerce").dt.normalize())
    )

    has_repay = "fin_repay_amount" in raw.columns

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
            fin_bal = latest.get("fin_balance", np.nan)
            short_vol = latest.get("short_volume", np.nan)

            # fin_balance_ratio: 融资余额/融资买入额 (近似杠杆倾向)
            fin_buy_latest = latest.get("fin_buy_amount", np.nan)
            fin_balance_ratio = fin_bal / fin_buy_latest if fin_buy_latest and fin_buy_latest > 0 else np.nan

            # net_fin_buy_1m: SSE-only (has fin_repay_amount)
            net_buy_1m = np.nan
            if has_repay:
                buy_sum = window["fin_buy_amount"].tail(20).sum()
                repay_sum = window["fin_repay_amount"].tail(20).sum()
                if pd.notna(buy_sum) and pd.notna(repay_sum):
                    net_buy_1m = buy_sum - repay_sum

            # short_pressure_1m: 融券余量变化率
            short_first = window["short_volume"].iloc[0] if len(window) > 1 else np.nan
            short_pressure_1m = (
                (short_vol - short_first) / abs(short_first)
                if short_first and pd.notna(short_first) and short_first != 0
                else np.nan
            )

            # fin_balance_momentum_1m: 融资余额增长率
            fin_bal_first = window["fin_balance"].iloc[0] if len(window) > 1 else np.nan
            fin_bal_mom_1m = (
                (fin_bal - fin_bal_first) / abs(fin_bal_first)
                if fin_bal_first and pd.notna(fin_bal_first) and fin_bal_first != 0
                else np.nan
            )

            rows.append({
                "signal_date": sd,
                "symbol": symbol,
                "feature_margin_fin_balance_ratio": fin_balance_ratio,
                "feature_margin_net_fin_buy_1m": net_buy_1m,
                "feature_margin_short_pressure_1m": short_pressure_1m,
                "feature_margin_fin_balance_momentum_1m": fin_bal_mom_1m,
            })

    if not rows:
        return add_zscore_and_missing_flags(out, MARGIN_TRADING_RAW_FEATURES)

    mt_df = pd.DataFrame(rows)
    mt_df["signal_date"] = pd.to_datetime(mt_df["signal_date"], errors="coerce").dt.normalize()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()

    out = out.merge(mt_df, on=["signal_date", "symbol"], how="left")
    return add_zscore_and_missing_flags(out, MARGIN_TRADING_RAW_FEATURES)
