"""基本面因子截面处理：缩尾、缺失填充与可选中性化。"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .neutralize import neutralize_size_industry_regression
from .standardize import fill_missing, winsorize_by_date

DEFAULT_FUNDAMENTAL_COLS: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "ev_ebitda",
    "roe_ttm",
    "net_profit_yoy",
    "gross_margin_change",
    "gross_margin_delta",
    "debt_to_assets_change",
    "ocf_to_net_profit",
    "ocf_to_asset",
    "asset_turnover",
    "net_margin_stability",
    "northbound_net_inflow",
    "margin_buy_ratio",
)

DAILY_VALUATION_SOURCES: tuple[str, ...] = ("stock_value_em",)

# P0-1: A 股财报实际披露截止日
# 年报披露截止 4/30、半年报 8/31、三季报 10/31
# 在这些日期之前使用 availability_lag_days 固定延迟会系统性地引入未来数据
_REPORT_PERIOD_DEADLINES: dict[int, tuple[int, int]] = {
    # month: (day, month_day_label)
    3: (31, 3),   # Q1 报告：3 月 31 日（部分提前披露，保守取季度末）
    6: (31, 8),   # 半年报：8 月 31 日截止
    9: (31, 10),  # 三季报：10 月 31 日截止
    12: (30, 4),  # 年报：次年 4 月 30 日截止
}


def _report_period_statutory_deadline(report_period: pd.Timestamp) -> pd.Timestamp:
    """返回财报报告期的法定披露截止日（保守估计）。"""
    m = report_period.month
    if m in _REPORT_PERIOD_DEADLINES:
        day, deadline_month = _REPORT_PERIOD_DEADLINES[m]
        year = report_period.year if deadline_month >= m else report_period.year + 1
        return pd.Timestamp(year=year, month=deadline_month, day=day)
    # fallback: report_period + 120 天
    return report_period + pd.Timedelta(days=120)


def pit_safe_fundamental_rows(
    df: pd.DataFrame,
    *,
    announcement_col: str = "announcement_date",
    report_period_col: str = "report_period",
    source_col: str = "source",
    disclosure_calendar: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
    fallback_lag_days: int = 45,
) -> pd.Series:
    """Return rows whose fundamental values were observable at announcement time.

    P0-1: 优先使用实际披露日历（disclosure_calendar），缺失时回退为
    法定截止日保守估计，最后兜底为 fallback_lag_days 固定延迟。

    Statement-derived fundamentals need a real disclosure lag
    (announcement date after report period). Daily valuation snapshots are already
    dated by trading day and may use the same date for availability/report period.

    Parameters
    ----------
    df : DataFrame
    disclosure_calendar : DataFrame or None
        若提供，须包含 symbol, report_period, disclosure_dt 三列。
        优先使用实际公告日期判定可得性。
    signal_date : Timestamp or None
        信号日期；用于判定披露日是否不晚于信号日。
    fallback_lag_days : int
        披露日历缺失时的兜底延迟天数（默认 45 天，比旧的 30 天更保守）。
    """
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    ann = pd.to_datetime(df.get(announcement_col), errors="coerce").dt.normalize()
    symbol_col = "symbol"

    # ── 1. 优先使用实际披露日历 ──
    if disclosure_calendar is not None and not disclosure_calendar.empty and symbol_col in df.columns:
        cal = disclosure_calendar.copy()
        cal["symbol"] = cal["symbol"].astype(str).str.zfill(6)
        cal["report_period"] = pd.to_datetime(cal["report_period"], errors="coerce").dt.normalize()
        cal["disclosure_dt"] = pd.to_datetime(cal["disclosure_dt"], errors="coerce").dt.normalize()
        df_sym = df[symbol_col].astype(str).str.zfill(6)
        if report_period_col in df.columns:
            df_rp = pd.to_datetime(df[report_period_col], errors="coerce").dt.normalize()
            merged = pd.merge(
                pd.DataFrame({"symbol": df_sym, "report_period": df_rp, "_idx": df.index}),
                cal[["symbol", "report_period", "disclosure_dt"]],
                on=["symbol", "report_period"],
                how="left",
            )
            known_disclosure = merged["disclosure_dt"].notna()
            if signal_date is not None:
                sd = pd.Timestamp(signal_date).normalize()
                # 实际披露日不晚于信号日 → 可得
                calendar_safe = known_disclosure & (merged["disclosure_dt"] <= sd)
            else:
                # 无信号日时，只要实际公告日不早于报告期即可
                calendar_safe = known_disclosure & (merged["disclosure_dt"] >= merged["report_period"])
            # 对于有日历覆盖的行，直接返回结果
            result = pd.Series(False, index=df.index, dtype=bool)
            result[merged.loc[calendar_safe, "_idx"]] = True
            # 对于无日历覆盖的行，继续走后续逻辑
            no_cal_mask = pd.Series(True, index=df.index, dtype=bool)
            no_cal_mask[merged.loc[known_disclosure, "_idx"]] = False
        else:
            no_cal_mask = pd.Series(True, index=df.index, dtype=bool)
            result = pd.Series(False, index=df.index, dtype=bool)
    else:
        no_cal_mask = pd.Series(True, index=df.index, dtype=bool)
        result = pd.Series(False, index=df.index, dtype=bool)

    # ── 2. 无日历覆盖的行：使用法定截止日保守估计 ──
    if no_cal_mask.any():
        sub_df = df.loc[no_cal_mask]
        sub_ann = ann.loc[no_cal_mask]

        if report_period_col in df.columns:
            sub_period = pd.to_datetime(sub_df[report_period_col], errors="coerce").dt.normalize()
            # 对每条记录计算法定截止日
            statutory_deadlines = sub_period.apply(
                lambda rp: _report_period_statutory_deadline(rp) if pd.notna(rp) else pd.NaT
            )
            if signal_date is not None:
                sd = pd.Timestamp(signal_date).normalize()
                statutory_safe = statutory_deadlines.notna() & (statutory_deadlines <= sd)
            else:
                # Without an as-of signal date, the statutory deadline alone does
                # not prove a statement row was observable. Preserve the older
                # PIT guard: statement-derived rows need a real positive notice
                # lag, while daily valuation sources are handled below.
                statutory_safe = pd.Series(False, index=sub_df.index)
            has_positive_notice_lag = sub_ann.notna() & (sub_ann > sub_period)
        else:
            statutory_safe = pd.Series(False, index=sub_df.index)
            has_positive_notice_lag = pd.Series(True, index=sub_df.index)

        if source_col in df.columns:
            sub_source = sub_df[source_col].astype(object).where(sub_df[source_col].notna(), "").astype(str)
        else:
            sub_source = pd.Series("", index=sub_df.index)
        is_daily_valuation = sub_source.isin(DAILY_VALUATION_SOURCES)

        # ── 3. 兜底：fallback_lag_days 固定延迟 ──
        if signal_date is not None:
            sd = pd.Timestamp(signal_date).normalize()
            cutoff = sd - pd.Timedelta(days=int(fallback_lag_days))
            if report_period_col in df.columns:
                fallback_safe = sub_period.notna() & (sub_period <= cutoff)
            else:
                fallback_safe = pd.Series(True, index=sub_df.index)
        else:
            fallback_safe = pd.Series(True, index=sub_df.index)

        # 综合判定：每日估值源 | 法定截止日 OK | 公告日>报告期（兜底）
        sub_result = sub_ann.notna() & (
            is_daily_valuation | statutory_safe | (has_positive_notice_lag & fallback_safe)
        )
        result.loc[no_cal_mask] = sub_result.values

    return result


def preprocess_fundamental_cross_section(
    df: pd.DataFrame,
    *,
    factor_cols: Optional[Iterable[str]] = None,
    date_col: str = "trade_date",
    size_col: str = "log_market_cap",
    industry_col: Optional[str] = None,
    neutralize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    对基本面因子做最小预处理：
    1) 按日期分组缩尾
    2) 可选按市值/行业做截面回归中性化
    3) 缺失按日期中位数填充（兜底为 0）
    """
    out = df.copy()
    cols = [c for c in (factor_cols or DEFAULT_FUNDAMENTAL_COLS) if c in out.columns]
    if not cols:
        return out

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out = winsorize_by_date(
            out,
            c,
            date_col=date_col,
            lower_q=lower_q,
            upper_q=upper_q,
            out_col=c,
        )
        if neutralize and size_col in out.columns:
            ind_col = industry_col if (industry_col and industry_col in out.columns) else "__no_industry__"
            tmp = out.copy()
            if ind_col == "__no_industry__":
                tmp[ind_col] = "NA"
            tmp = neutralize_size_industry_regression(
                tmp,
                c,
                size_col=size_col,
                industry_col=ind_col,
                date_col=date_col,
                suffix="_fund_neut",
            )
            nc = f"{c}_fund_neut"
            if nc in tmp.columns:
                out[c] = tmp[nc]
        out[c] = fill_missing(
            out[c],
            method="cs_median",
            by_group=out[date_col],
        ).fillna(0.0)
    return out
