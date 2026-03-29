"""
A 股可交易性：涨跌停比例、次日开盘一字涨停（难以买入）、停牌近似。

荐股列表仅做「可买性」过滤，不实现 T+1 仓位状态机；回测与标签对齐见 T+1 开盘价序列。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def limit_up_ratio(symbol: str) -> float:
    """
    普通股涨跌幅限制比例（不含 ST）。
    创业板/科创板 20%，北交所 30%，其余主板 10%。
    """
    s = str(symbol).zfill(6)
    if s.startswith(("300", "688")):
        return 0.20
    if s.startswith(("8", "4")):
        return 0.30
    return 0.10


def limit_up_px(prev_close: float, symbol: str) -> float:
    """涨停价（简化：未做 tick 舍入，与行情比较时用相对容差）。"""
    pc = float(prev_close)
    r = limit_up_ratio(symbol)
    return pc * (1.0 + r)


def is_open_limit_up_unbuyable(
    open_px: float,
    prev_close: float,
    symbol: str,
    *,
    rel_tol: float = 1e-4,
) -> bool:
    """一字涨停开盘：开盘价触及涨停价（约等于），散户无法按开盘价成交。"""
    if not np.isfinite(open_px) or not np.isfinite(prev_close) or prev_close <= 0:
        return True
    lim = limit_up_px(prev_close, symbol)
    return open_px >= lim * (1.0 - rel_tol)


def is_row_suspended_like(
    volume: float,
    open_px: float,
    close_px: float,
) -> bool:
    """停牌近似：无成交量或 OHLC 无效。"""
    if not np.isfinite(open_px) or not np.isfinite(close_px):
        return True
    if not np.isfinite(volume) or volume <= 0:
        return True
    return False


def prefilter_stock_pool(
    df: pd.DataFrame,
    asof_date,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    limit_move_lookback: int = 5,
    limit_move_max: int = 2,
    turnover_low_pct: float = 0.10,
    turnover_high_pct: float = 0.98,
    price_position_high_pct: float = 0.90,
    price_position_lookback: int = 250,
    log=None,
) -> Tuple[list, dict]:
    """
    硬规则预过滤：在进入因子计算或打分前剔除高风险标的。

    规则：
    1. 剔除 ST / \\*ST（股票代码名称含 ST，需 name 列；无 name 列则跳过）。
    2. 剔除过去 ``limit_move_lookback`` 天内发生 >= ``limit_move_max`` 次
       涨停或跌停（|日收益| >= 对应板块限幅）的股票。
    3. 剔除换手率长期处于市场后 ``turnover_low_pct`` 或前 ``1 - turnover_high_pct``
       的标的。
    4. 剔除当前价格处于过去 ``price_position_lookback`` 天区间
       ``price_position_high_pct`` 分位以上的绝对高位股。

    Returns
    -------
    kept_symbols : list
        通过过滤的标的代码列表。
    stats : dict
        各规则剔除数量的统计。
    """
    if df.empty:
        return [], {"total": 0}

    asof_ts = pd.Timestamp(asof_date).normalize()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[symbol_col] = df[symbol_col].astype(str).str.zfill(6)

    all_syms = df[symbol_col].unique().tolist()
    removed_st: set = set()
    removed_limit: set = set()
    removed_turnover: set = set()
    removed_high: set = set()

    if "name" in df.columns:
        latest_names = df.sort_values(date_col).groupby(symbol_col)["name"].last()
        for sym, name in latest_names.items():
            if isinstance(name, str) and "ST" in name.upper():
                removed_st.add(sym)

    lookback_start = asof_ts - pd.offsets.BDay(limit_move_lookback + 5)
    recent = df[(df[date_col] >= lookback_start) & (df[date_col] <= asof_ts)]

    for sym, grp in recent.groupby(symbol_col, sort=False):
        if sym in removed_st:
            continue
        grp = grp.sort_values(date_col)
        if len(grp) < 2:
            continue
        closes = grp["close"].to_numpy(dtype=np.float64)
        rets = closes[1:] / closes[:-1] - 1.0
        tail = rets[-limit_move_lookback:] if len(rets) >= limit_move_lookback else rets
        lim = limit_up_ratio(str(sym))
        threshold = lim - 0.005
        n_hits = int(np.sum(np.abs(tail) >= threshold))
        if n_hits >= limit_move_max:
            removed_limit.add(sym)

    if "turnover" in recent.columns:
        last_day = recent[recent[date_col] == asof_ts]
        if not last_day.empty:
            to_vals = pd.to_numeric(last_day["turnover"], errors="coerce")
            lo = to_vals.quantile(turnover_low_pct)
            hi = to_vals.quantile(turnover_high_pct)
            for _, row in last_day.iterrows():
                sym = row[symbol_col]
                if sym in removed_st or sym in removed_limit:
                    continue
                tv = float(row.get("turnover", np.nan))
                if np.isfinite(tv) and (tv < lo or tv > hi):
                    removed_turnover.add(sym)

    pp_start = asof_ts - pd.offsets.BDay(price_position_lookback + 20)
    hist = df[(df[date_col] >= pp_start) & (df[date_col] <= asof_ts)]
    for sym, grp in hist.groupby(symbol_col, sort=False):
        if sym in removed_st or sym in removed_limit or sym in removed_turnover:
            continue
        closes = pd.to_numeric(grp["close"], errors="coerce").dropna()
        if len(closes) < 20:
            continue
        current = closes.iloc[-1]
        pct = float((closes < current).sum()) / len(closes)
        if pct >= price_position_high_pct:
            removed_high.add(sym)

    all_removed = removed_st | removed_limit | removed_turnover | removed_high
    kept = [s for s in all_syms if s not in all_removed]

    stats = {
        "total_before": len(all_syms),
        "removed_st": len(removed_st),
        "removed_limit_move": len(removed_limit),
        "removed_turnover_extreme": len(removed_turnover),
        "removed_absolute_high": len(removed_high),
        "total_removed": len(all_removed),
        "total_after": len(kept),
    }
    if log:
        log.info(
            "预过滤: ST=%d, 涨跌停=%d, 换手率=%d, 绝对高位=%d → 剩余 %d/%d",
            stats["removed_st"],
            stats["removed_limit_move"],
            stats["removed_turnover_extreme"],
            stats["removed_absolute_high"],
            stats["total_after"],
            stats["total_before"],
        )
    return kept, stats


def _sorted_dates(daily_df: pd.DataFrame, date_col: str) -> np.ndarray:
    d = pd.to_datetime(daily_df[date_col]).dt.normalize().unique()
    return np.sort(d)


def _next_trading_date(
    dates_sorted: np.ndarray,
    asof: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    asof_n = pd.Timestamp(asof).normalize()
    for i, d in enumerate(dates_sorted):
        if pd.Timestamp(d).normalize() != asof_n:
            continue
        if i + 1 < len(dates_sorted):
            return pd.Timestamp(dates_sorted[i + 1]).normalize()
        return None
    return None


def filter_recommend_tradable_next_day(
    rec_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    asof_date,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    log=None,
) -> Tuple[pd.DataFrame, int]:
    """
    剔除次日停牌、次日开盘一字涨停（难以买入）的标的。

    若次日尚未产生行情（例如尚未产生 T+1），则**不剔除**该标的，并打日志（可选）。

    Returns
    -------
    filtered_df, n_dropped
    """
    if rec_df.empty or daily_df.empty:
        return rec_df.copy(), 0

    dates_sorted = _sorted_dates(daily_df, date_col)
    asof_ts = pd.Timestamp(asof_date).normalize()
    next_day = _next_trading_date(dates_sorted, asof_ts)
    if next_day is None:
        if log:
            log.warning(
                "asof=%s 后无下一交易日行情，跳过「次日可买性」过滤。",
                asof_ts.date(),
            )
        return rec_df.copy(), 0

    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    sym_col_db = "symbol" if "symbol" in df.columns else "代码"

    # 预取 (symbol, date) -> row
    keep_mask: list[bool] = []
    dropped = 0
    for _, row in rec_df.iterrows():
        sym = str(row[symbol_col]).zfill(6)
        sub = df[(df[sym_col_db].astype(str).str.zfill(6) == sym)]
        if sub.empty:
            keep_mask.append(True)
            continue
        r0 = sub[sub[date_col] == asof_ts]
        r1 = sub[sub[date_col] == next_day]
        if r0.empty or r1.empty:
            keep_mask.append(True)
            continue
        r0 = r0.iloc[0]
        r1 = r1.iloc[0]
        prev_close = float(r0.get("close", np.nan))
        o1 = float(r1.get("open", np.nan))
        c1 = float(r1.get("close", np.nan))
        v1 = float(r1.get("volume", np.nan))

        if is_row_suspended_like(v1, o1, c1):
            keep_mask.append(False)
            dropped += 1
            continue
        if is_open_limit_up_unbuyable(o1, prev_close, sym):
            keep_mask.append(False)
            dropped += 1
            continue
        keep_mask.append(True)

    out = rec_df.loc[keep_mask].reset_index(drop=True)
    return out, dropped
