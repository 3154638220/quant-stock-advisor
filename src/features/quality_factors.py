"""W1 Phase 1: 质量因子族 — ROE 稳定性、Accruals、资产增长率、盈利惊喜。

基于 a_share_fundamental 表的历史数据构造截面质量因子。
所有因子均需通过 PIT 验证后方可接入 M5 gate。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

# ── 因子列表常量 ─────────────────────────────────────────────────────────

QUALITY_FACTOR_COLS: tuple[str, ...] = (
    "feature_quality_roe_stability",
    "feature_quality_accruals_ratio",
    "feature_quality_asset_growth_rate",
    "feature_quality_earnings_surprise",
)

QUALITY_FACTOR_DIRECTION: dict[str, int] = {
    "feature_quality_roe_stability": 1,       # ROE 稳定 = 高质量，正向
    "feature_quality_accruals_ratio": -1,      # 高应计 = 盈利质量差，负向
    "feature_quality_asset_growth_rate": -1,   # 高资产增长 = 过度扩张，负向
    "feature_quality_earnings_surprise": 1,    # 盈利超预期 = 正向
}


def compute_quality_factors(
    db_path: str,
    signal_date: str | pd.Timestamp,
    *,
    table_name: str = "a_share_fundamental",
    lookback_quarters: int = 8,
    min_coverage: float = 0.30,
) -> pd.DataFrame:
    """从 fundamental 历史数据计算质量因子截面。

    Parameters
    ----------
    db_path : str
        DuckDB 数据库路径。
    signal_date : str or Timestamp
        信号日（月末），只使用 announcement_date <= signal_date 的数据。
    table_name : str
        基本面表名。
    lookback_quarters : int
        ROE 稳定性计算所需的回溯季度数。
    min_coverage : float
        最低截面覆盖率阈值。

    Returns
    -------
    pd.DataFrame
        包含 symbol, trade_date (=signal_date) 及质量因子列。
    """
    sd = pd.Timestamp(signal_date).normalize()
    con = duckdb.connect(db_path, read_only=True)

    try:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            return pd.DataFrame()

        # PIT-safe: 只使用 announcement_date <= signal_date 的行
        raw = con.execute(
            f"""
            SELECT symbol, report_period, announcement_date,
                   roe_ttm, net_profit_yoy, ocf_to_net_profit, ocf_to_asset,
                   debt_to_assets, gross_margin, gross_margin_change
            FROM {table_name}
            WHERE announcement_date IS NOT NULL
              AND CAST(announcement_date AS DATE) <= CAST(? AS DATE)
            ORDER BY symbol, report_period
            """,
            [sd.strftime("%Y-%m-%d")],
        ).df()
    finally:
        con.close()

    if raw.empty:
        return pd.DataFrame()

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["report_period"] = pd.to_datetime(raw["report_period"], errors="coerce").dt.normalize()
    raw["announcement_date"] = pd.to_datetime(raw["announcement_date"], errors="coerce").dt.normalize()

    for c in ["roe_ttm", "net_profit_yoy", "ocf_to_net_profit", "ocf_to_asset",
              "debt_to_assets", "gross_margin", "gross_margin_change"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # 按 symbol 排序，取每个 report_period 最新 announcement
    raw = raw.sort_values(["symbol", "report_period", "announcement_date"])
    raw = raw.drop_duplicates(["symbol", "report_period"], keep="last")

    # ── 1. roe_stability: ROE 过去 N 季度标准差（取负 = 越稳定越好）──
    roe_stability = _compute_roe_stability(raw, lookback_quarters=lookback_quarters)

    # ── 2. accruals_ratio: (1 - ocf_to_net_profit) 的符号版 ──
    # 高应计 = 净利润远大于经营现金流 = 盈利质量差
    accruals = _compute_accruals(raw)

    # ── 3. asset_growth_rate: 通过 ocf_to_asset 和 debt_to_assets 反推 ──
    # asset_growth ≈ -Δ(debt_to_assets) / debt_to_assets  (粗略近似)
    asset_growth = _compute_asset_growth_proxy(raw)

    # ── 4. earnings_surprise: net_profit_yoy 相对自身历史均值的偏离 ──
    surprise = _compute_earnings_surprise(raw)

    # ── 合并为截面 ──
    latest = raw.sort_values(["symbol", "report_period"]).groupby("symbol").tail(1)[
        ["symbol", "report_period"]
    ].copy()

    result = latest.merge(roe_stability, on="symbol", how="left")
    result = result.merge(accruals, on="symbol", how="left")
    result = result.merge(asset_growth, on="symbol", how="left")
    result = result.merge(surprise, on="symbol", how="left")
    # Use same date column name as the input expects (trade_date for standalone, signal_date for pipeline)
    result["trade_date"] = sd

    # 覆盖率过滤
    for col in QUALITY_FACTOR_COLS:
        if col in result.columns:
            cov = result[col].notna().mean()
            if cov < min_coverage:
                result[col] = np.nan

    return result


def _compute_roe_stability(
    raw: pd.DataFrame, *, lookback_quarters: int = 8
) -> pd.DataFrame:
    """ROE 过去 N 季度标准差（取负值，正值 = 高稳定性）。

    对每个 symbol，取最近 lookback_quarters 个 report_period 的 roe_ttm，
    计算标准差，取负。缺失超过一半的 symbol 返回 NaN。
    """
    df = raw[["symbol", "report_period", "roe_ttm"]].dropna(subset=["roe_ttm"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_quality_roe_stability"])

    df = df.sort_values(["symbol", "report_period"])
    result = (
        df.groupby("symbol")
        .apply(
            lambda g: g.tail(lookback_quarters).pipe(
                lambda x: np.nan
                if len(x) < max(3, lookback_quarters // 2)
                else float(x["roe_ttm"].std())
            ),
            include_groups=False,
        )
        .reset_index(name="roe_std")
    )
    # ROE 稳定性 = -std，越大越稳定
    result["feature_quality_roe_stability"] = -result["roe_std"].astype(float)
    return result[["symbol", "feature_quality_roe_stability"]]


def _compute_accruals(raw: pd.DataFrame) -> pd.DataFrame:
    """应计项目比率代理变量。

    当 ocf_to_net_profit 有效时：
      accruals ≈ (1 - ocf_to_net_profit)  当 ocf_to_net_profit > 0
      accruals ≈ 1 - 1/ocf_to_net_profit 当 ocf_to_net_profit > 1（利润<现金流）
    取最新报告期的值。
    """
    df = raw[["symbol", "report_period", "ocf_to_net_profit"]].dropna(
        subset=["ocf_to_net_profit"]
    ).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_quality_accruals_ratio"])

    df = df.sort_values(["symbol", "report_period"])
    latest = df.groupby("symbol").tail(1).copy()

    ocf_ratio = latest["ocf_to_net_profit"].astype(float)
    # 应计比率: 正值 = 利润 > 现金流 = 盈利质量差
    accruals = np.where(
        ocf_ratio > 0,
        np.where(ocf_ratio <= 1, 1.0 - ocf_ratio, 0.0),
        np.nan,
    )
    latest["feature_quality_accruals_ratio"] = accruals
    return latest[["symbol", "feature_quality_accruals_ratio"]]


def _compute_asset_growth_proxy(raw: pd.DataFrame) -> pd.DataFrame:
    """资产增长率代理：使用 debt_to_assets 的时序变化。

    debt_to_assets = total_debt / total_assets
    当 debt 变化慢于 assets 时，debt_to_assets ↓ 意味着 assets ↑
    近似: asset_growth ≈ (debt_to_assets_{t-4Q} - debt_to_assets_t) / debt_to_assets_t
    """
    df = raw[["symbol", "report_period", "debt_to_assets"]].dropna(
        subset=["debt_to_assets"]
    ).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_quality_asset_growth_rate"])

    df = df.sort_values(["symbol", "report_period"])

    def _growth(g: pd.DataFrame) -> float | None:
        if len(g) < 5:
            return None
        g = g.tail(5)  # 最近 5 个季度
        current = g["debt_to_assets"].iloc[-1]
        # 找大约 4 个季度前的值
        past = g["debt_to_assets"].iloc[0]
        if pd.isna(current) or pd.isna(past) or past == 0:
            return None
        return float((past - current) / past)  # debt_ratio 下降 = 资产增长

    result = df.groupby("symbol").apply(_growth, include_groups=False).reset_index(name="feature_quality_asset_growth_rate")
    result["feature_quality_asset_growth_rate"] = result["feature_quality_asset_growth_rate"].astype(float)
    return result


def _compute_earnings_surprise(raw: pd.DataFrame) -> pd.DataFrame:
    """盈利惊喜：最新 net_profit_yoy 相对自身历史均值的标准化偏离。

    surprise = (latest_np_yoy - mean_np_yoy) / std_np_yoy
    """
    df = raw[["symbol", "report_period", "net_profit_yoy"]].dropna(
        subset=["net_profit_yoy"]
    ).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_quality_earnings_surprise"])

    df = df.sort_values(["symbol", "report_period"])

    def _surprise(g: pd.DataFrame) -> float | None:
        if len(g) < 4:
            return None
        hist = g["net_profit_yoy"].iloc[:-1]
        latest = g["net_profit_yoy"].iloc[-1]
        mu = float(hist.mean())
        sigma = float(hist.std())
        if pd.isna(sigma) or sigma == 0:
            return 0.0 if pd.notna(latest) and pd.notna(mu) else None
        return float((latest - mu) / sigma)

    result = df.groupby("symbol").apply(_surprise, include_groups=False).reset_index(name="feature_quality_earnings_surprise")
    result["feature_quality_earnings_surprise"] = result["feature_quality_earnings_surprise"].astype(float)
    return result


def attach_quality_factors(
    factors: pd.DataFrame,
    db_path: str,
    *,
    table_name: str = "a_share_fundamental",
) -> pd.DataFrame:
    """将质量因子附加到现有因子长表。

    对每个 signal_date，逐月计算质量因子并合并。
    注意：此操作较慢（每日期需单独查询），适合月度批量场景。
    """
    out = factors.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    # Support both trade_date and signal_date as date column name
    date_col = "trade_date" if "trade_date" in out.columns else "signal_date"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    unique_dates = sorted(out[date_col].dropna().unique())
    all_quality: list[pd.DataFrame] = []

    for dt in unique_dates:
        q = compute_quality_factors(db_path, signal_date=dt, table_name=table_name)
        if q.empty:
            continue
        q[date_col] = pd.Timestamp(dt)
        all_quality.append(q)

    if not all_quality:
        for col in QUALITY_FACTOR_COLS:
            out[col] = np.nan
        return out

    quality_all = pd.concat(all_quality, ignore_index=True)
    quality_all[date_col] = pd.to_datetime(quality_all[date_col]).dt.normalize()

    out = out.merge(
        quality_all[["symbol", date_col] + list(QUALITY_FACTOR_COLS)],
        on=["symbol", date_col],
        how="left",
    )
    return out
