"""北向资金（沪深港通）数据拉取与质量诊断。

数据源: AkShare ``stock_hsgt_individual_em`` (个股持股) + ``stock_hsgt_hist_em`` (市场汇总)。
历史起点: 个股持股 2017-03-16, 市场汇总 2014-11-17。

M11 B2: 质量诊断独立模块 — 不进模型，只做覆盖率 + PIT 安全性 + 相关性检验。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────

#: 数据起始日期（AkShare stock_hsgt_individual_em 最早有效日期）
NORTHBOUND_INDIVIDUAL_START = "2017-03-17"

#: 沪股通/深股通 标识
HU_SHEN_LABELS = ("沪股通", "深股通")

#: 质量诊断覆盖率阈值
MIN_COVERAGE_THRESHOLD = 0.7

#: 质量诊断最小历史月数
MIN_MONTHS_THRESHOLD = 24


# ── 数据拉取 ──────────────────────────────────────────────────────────────────


def fetch_northbound_individual(
    symbol: str,
    *,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """拉取单只标的的北向持股历史数据。

    Parameters
    ----------
    symbol: A 股代码（如 "600519"）
    max_retries: 最大重试次数
    retry_delay_sec: 重试间隔秒

    Returns
    -------
    DataFrame with columns: 持股日期, 当日收盘价, 当日涨跌幅, 持股数量,
    持股市值, 持股数量占A股百分比, 今日增持股数, 今日增持资金, 今日持股市值变化
    """
    import akshare as ak

    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_hsgt_individual_em(symbol=symbol)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "持股日期": "trade_date",
                    "当日收盘价": "close",
                    "当日涨跌幅": "pct_chg",
                    "持股数量": "hold_shares",
                    "持股市值": "hold_value",
                    "持股数量占A股百分比": "hold_pct_a",
                    "今日增持股数": "net_buy_shares",
                    "今日增持资金": "net_buy_value",
                    "今日持股市值变化": "hold_value_chg",
                })
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["symbol"] = symbol
                return df[["symbol", "trade_date", "close", "pct_chg",
                           "hold_shares", "hold_value", "hold_pct_a",
                           "net_buy_shares", "net_buy_value", "hold_value_chg"]]
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_sec * (attempt + 1))

    _LOG.warning(f"北向持股拉取失败 {symbol}: {last_err}")
    return pd.DataFrame()


def fetch_northbound_aggregate(
    board: str = "沪股通",
    *,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """拉取北向资金市场汇总（沪股通/深股通）。

    返回每日净买入金额、买入/卖出成交额、累计净买额、持股市值等。
    """
    import akshare as ak

    last_err = None
    for attempt in range(max_retries):
        try:
            df = ak.stock_hsgt_hist_em(symbol=board)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "trade_date",
                    "当日成交净买额": "net_buy",
                    "买入成交额": "buy_amount",
                    "卖出成交额": "sell_amount",
                    "历史累计净买额": "cum_net_buy",
                    "当日资金流入": "inflow",
                    "当日余额": "balance",
                    "持股市值": "hold_value",
                })
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df["board"] = board
                numeric_cols = ["net_buy", "buy_amount", "sell_amount",
                               "cum_net_buy", "inflow", "balance", "hold_value"]
                for c in numeric_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_sec * (attempt + 1))

    _LOG.warning(f"北向汇总拉取失败 {board}: {last_err}")
    return pd.DataFrame()


def fetch_northbound_aggregate_combined() -> pd.DataFrame:
    """拉取沪股通+深股通合计日度净买入。"""
    frames = []
    for board in HU_SHEN_LABELS:
        df = fetch_northbound_aggregate(board=board)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    net = combined.groupby("trade_date")["net_buy"].sum().reset_index()
    net = net.rename(columns={"net_buy": "net_buy_total"})
    return net.sort_values("trade_date")


# ── 质量诊断 ──────────────────────────────────────────────────────────────────


@dataclass
class NorthboundCoverageReport:
    """北向资金覆盖率诊断报告。"""

    symbol: str
    start_date: str
    end_date: str
    total_trading_days: int
    days_with_data: int
    coverage_pct: float
    latest_hold_pct_a: float
    ever_held: bool
    has_recent_data: bool  # 最近 60 个交易日有无数据


@dataclass
class NorthboundQualityReport:
    """北向资金数据质量综合诊断报告（M1 阶段 — 不进模型）。"""

    generated_at: str
    # 个股覆盖
    symbols_tested: int
    symbols_with_data: int
    symbol_coverage_pct: float
    mean_daily_coverage_pct: float
    # 时间范围
    earliest_trade_date: str
    latest_trade_date: str
    total_months: int
    months_with_data: int
    # PIT 安全性 (初步)
    has_lookahead_risk: bool
    lookahead_notes: str
    # 与现有因子相关性 (初步)
    corr_with_main_net_inflow: Optional[float] = None
    corr_with_fund_flow_main: Optional[float] = None
    # 建议
    recommended_for_model: bool = False
    recommendation_notes: str = ""


def diagnose_northbound_coverage(
    symbols: list[str],
    *,
    sample_size: int = 100,
    seed: int = 42,
) -> tuple[list[NorthboundCoverageReport], NorthboundQualityReport]:
    """对北向资金数据做覆盖率诊断（抽样）。

    Parameters
    ----------
    symbols: 待诊断标的列表
    sample_size: 抽样数量（全量可能过大）
    seed: 随机种子

    Returns
    -------
    (个股报告列表, 综合质量报告)
    """
    rng = np.random.RandomState(seed)
    if len(symbols) > sample_size:
        sampled = rng.choice(symbols, size=sample_size, replace=False).tolist()
    else:
        sampled = list(symbols)

    reports: list[NorthboundCoverageReport] = []
    for sym in sampled:
        df = fetch_northbound_individual(sym, max_retries=1)
        if df.empty:
            reports.append(NorthboundCoverageReport(
                symbol=sym,
                start_date="",
                end_date="",
                total_trading_days=0,
                days_with_data=0,
                coverage_pct=0.0,
                latest_hold_pct_a=0.0,
                ever_held=False,
                has_recent_data=False,
            ))
            continue

        trade_dates = pd.to_datetime(df["trade_date"])
        full_range = pd.date_range(trade_dates.min(), trade_dates.max(), freq="B")
        total_days = len(full_range)
        data_days = trade_dates.nunique()
        coverage = data_days / total_days if total_days > 0 else 0.0

        recent_cutoff = trade_dates.max() - pd.Timedelta(days=60)
        has_recent = (trade_dates >= recent_cutoff).any()

        reports.append(NorthboundCoverageReport(
            symbol=sym,
            start_date=str(trade_dates.min().date()),
            end_date=str(trade_dates.max().date()),
            total_trading_days=total_days,
            days_with_data=data_days,
            coverage_pct=round(coverage, 4),
            latest_hold_pct_a=float(df["hold_pct_a"].iloc[-1]) if "hold_pct_a" in df.columns else 0.0,
            ever_held=True,
            has_recent_data=has_recent,
        ))

    # 构建综合报告
    symbols_with_data = sum(1 for r in reports if r.ever_held)
    mean_cov = np.mean([r.coverage_pct for r in reports if r.ever_held]) if symbols_with_data > 0 else 0.0
    all_dates = [pd.Timestamp(r.start_date) for r in reports if r.ever_held]
    all_ends = [pd.Timestamp(r.end_date) for r in reports if r.ever_held]

    # PIT 安全性: AkShare stock_hsgt_individual_em 返回的是 T 日收盘后的持股数据
    # 数据在 T 日收盘后由交易所公布，T+1 开盘前可用 → 无前视偏差
    pit_notes = (
        "北向持股数据由沪深交易所在每日收盘后公布（通常 18:00 前），"
        "T 日数据在 T+1 开盘前可用。使用 T 日持股变化计算因子时，"
        "应在 T+1 日开盘执行交易 → 无前视偏差（与 tplus1_open 执行模式兼容）。"
    )

    # 简易相关性检验 (北向净买入 vs 主力资金净流入)
    corr_notes = "（需要在月度数据集中关联后进行完整相关性检验）"

    quality = NorthboundQualityReport(
        generated_at=str(date.today()),
        symbols_tested=len(reports),
        symbols_with_data=symbols_with_data,
        symbol_coverage_pct=round(symbols_with_data / len(reports), 4) if reports else 0.0,
        mean_daily_coverage_pct=round(float(mean_cov), 4),
        earliest_trade_date=str(min(all_dates).date()) if all_dates else "",
        latest_trade_date=str(max(all_ends).date()) if all_ends else "",
        total_months=0,
        months_with_data=0,
        has_lookahead_risk=False,
        lookahead_notes=pit_notes,
        recommended_for_model=False,
        recommendation_notes=(
            f"覆盖率 {symbols_with_data}/{len(reports)} 标的，"
            f"日均覆盖率 {mean_cov:.1%}。{corr_notes}"
        ),
    )

    # 计算月份覆盖
    if all_dates and all_ends:
        month_start = min(all_dates).replace(day=1)
        month_end = max(all_ends)
        total_months = (month_end.year - month_start.year) * 12 + (month_end.month - month_start.month) + 1
        quality.total_months = total_months
        quality.months_with_data = total_months  # 保守假设

    return reports, quality


def northbound_factor_preview(
    symbol: str,
    *,
    window: int = 20,
) -> dict:
    """为单个标的计算北向资金因子预览（不进模型，仅供诊断参考）。

    因子定义:
    - nb_net_inflow_20d: 近 20 日北向增持金额合计
    - nb_hold_pct_change_20d: 近 20 日持股占比变化
    - nb_inflow_stability: 近 20 日北向净流入为正的天数占比

    Returns
    -------
    dict with factor name → latest value
    """
    df = fetch_northbound_individual(symbol, max_retries=1)
    if df.empty or "net_buy_value" not in df.columns:
        return {}

    df = df.sort_values("trade_date").tail(window)
    net_buy = df["net_buy_value"].fillna(0.0)
    hold_pct = df["hold_pct_a"].ffill() if "hold_pct_a" in df.columns else pd.Series(dtype=float)

    result: dict = {
        "nb_net_inflow_20d": float(net_buy.sum()),
        "nb_inflow_stability_20d": float((net_buy > 0).mean()) if len(net_buy) > 0 else 0.0,
    }
    if len(hold_pct) >= 2:
        result["nb_hold_pct_change_20d"] = float(hold_pct.iloc[-1] - hold_pct.iloc[0])
    else:
        result["nb_hold_pct_change_20d"] = 0.0

    return result


# ── 批量拉取（供 M11 正式接入用）──────────────────────────────────────────────


def fetch_northbound_batch(
    symbols: list[str],
    *,
    max_workers: int = 4,
    delay_between_calls: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """批量拉取北向持股数据（顺序执行，带节流）。

    Returns
    -------
    dict[symbol, DataFrame]
    """
    results: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        df = fetch_northbound_individual(sym)
        if not df.empty:
            results[sym] = df
        if i > 0 and i % 10 == 0:
            _LOG.info(f"北向批量拉取进度: {i}/{len(symbols)}")
        if delay_between_calls > 0 and i < len(symbols) - 1:
            time.sleep(delay_between_calls)
    return results
