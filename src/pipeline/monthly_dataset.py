"""月度选股 canonical dataset 构建管线。

从 scripts/run_monthly_selection_dataset.py 提取核心逻辑，
只放算法与数据处理，不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.backtest.engine import build_open_to_open_returns
from src.market.tradability import is_open_limit_up_unbuyable, is_row_suspended_like, limit_up_ratio
from src.research.gates import POOL_RULES

# ── 特征 & 标签列定义 ────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "feature_ret_5d",
    "feature_ret_20d",
    "feature_ret_60d",
    "feature_realized_vol_20d",
    "feature_amount_20d_log",
    "feature_turnover_20d",
    "feature_price_position_250d",
    "feature_limit_move_hits_20d",
]

LABEL_COLS: list[str] = [
    "label_forward_1m_o2o_return",
    "label_forward_1m_excess_vs_market",
    "label_forward_1m_industry_neutral_excess",
    "label_future_return_percentile",
    "label_future_return_quantile",
    "label_future_top_20pct",
    "label_future_top_10pct",
    "label_future_bottom_20pct",
]


# ── 配置数据类 ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonthlySelectionConfig:
    min_history_days: int = 120
    min_amount_20d: float = 50_000_000.0
    limit_move_lookback: int = 20
    limit_move_max: int = 3
    volatility_high_pct: float = 0.98
    turnover_high_pct: float = 0.98
    price_position_high_pct: float = 0.95
    price_position_lookback: int = 250


# ── 研究配置 ID ──────────────────────────────────────────────────────────

def build_research_config_id(
    *,
    start_date: str,
    end_date: str,
    min_history_days: int,
    min_amount_20d: float,
    limit_move_max: int,
    daily_table: str,
) -> str:
    import re

    def _slug(s: str) -> str:
        t = str(s).strip().lower()
        t = re.sub(r"[^a-z0-9]+", "_", t)
        return re.sub(r"_+", "_", t).strip("_") or "na"

    amount_m = float(min_amount_20d) / 1_000_000.0
    return (
        f"rb_m_exec_tplus1_open_sell_mend_open_label_o2o"
        f"_start_{_slug(start_date)}"
        f"_end_{_slug(end_date or 'latest')}"
        f"_hist_{int(min_history_days)}"
        f"_amt20m_{amount_m:.0f}"
        f"_lmmax_{int(limit_move_max)}"
        f"_daily_{_slug(daily_table)}"
    )


# ── DuckDB 数据读取 ──────────────────────────────────────────────────────

def read_daily_from_duckdb(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    start: str,
    end: str | None,
    min_history_days: int,
    price_position_lookback: int,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start).normalize()
    read_start = start_ts - pd.offsets.BDay(max(min_history_days, price_position_lookback) + 40)
    params: list[Any] = [read_start.date()]
    cond = "trade_date >= ?"
    if end:
        cond += " AND trade_date <= ?"
        params.append(pd.Timestamp(end).date())
    q = f"""
        SELECT symbol, trade_date, open, close, high, low, volume, amount, turnover, pct_chg
        FROM {table}
        WHERE {cond}
        ORDER BY symbol, trade_date
    """
    df = con.execute(q, params).df()
    return normalize_daily_frame(df)


def normalize_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "trade_date", "open", "close", "high", "low", "volume", "amount"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily 缺少列: {missing}")
    df = daily.copy()
    df["symbol"] = (
        df["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    for col in ["open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["symbol"].str.len() == 6) & df["trade_date"].notna()].copy()
    return df.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")


# ── 行业映射 ─────────────────────────────────────────────────────────────

def load_industry_map(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "industry_level1", "industry_level2"]), "missing"
    df = pd.read_csv(path, converters={"symbol": lambda v: str(v).strip().zfill(6)})
    if "industry_level1" not in df.columns and "industry" in df.columns:
        df["industry_level1"] = df["industry"]
    if "industry_level2" not in df.columns:
        df["industry_level2"] = ""
    keep = ["symbol", "industry_level1", "industry_level2"]
    return df[keep].drop_duplicates("symbol", keep="last"), "real_industry_map"


# ── 信号日选择 ───────────────────────────────────────────────────────────

def select_month_end_signal_dates(
    dates: pd.Series | list[Any] | np.ndarray,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[pd.Timestamp]:
    d = pd.Series(pd.to_datetime(pd.Series(dates), errors="coerce")).dropna().dt.normalize()
    if start:
        d = d[d >= pd.Timestamp(start).normalize()]
    if end:
        d = d[d <= pd.Timestamp(end).normalize()]
    if d.empty:
        return []
    unique = pd.DataFrame({"trade_date": sorted(d.unique())})
    unique["period"] = unique["trade_date"].dt.to_period("M")
    out = unique.groupby("period", sort=True)["trade_date"].max().tolist()
    return [pd.Timestamp(x).normalize() for x in out]


# ── 特征工程 ─────────────────────────────────────────────────────────────

def attach_signal_features(daily: pd.DataFrame, cfg: MonthlySelectionConfig) -> pd.DataFrame:
    df = normalize_daily_frame(daily)
    g = df.groupby("symbol", sort=False)
    df["history_days"] = g.cumcount() + 1
    df["_ret_1d"] = g["close"].pct_change()
    for n in (5, 20, 60):
        df[f"feature_ret_{n}d"] = g["close"].pct_change(n)
    df["amount_20d"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["feature_amount_20d_log"] = np.log1p(df["amount_20d"].where(df["amount_20d"] > 0))
    df["feature_realized_vol_20d"] = g["_ret_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    df["feature_turnover_20d"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["market_cap"] = np.where(
        (df["turnover"] > 0) & np.isfinite(df["amount"]),
        df["amount"] / (df["turnover"] / 100.0),
        np.nan,
    )
    df["log_market_cap"] = np.log1p(
        pd.to_numeric(df["market_cap"], errors="coerce").where(df["market_cap"] > 0)
    )
    roll_min = g["close"].transform(
        lambda s: s.rolling(cfg.price_position_lookback, min_periods=60).min()
    )
    roll_max = g["close"].transform(
        lambda s: s.rolling(cfg.price_position_lookback, min_periods=60).max()
    )
    denom = (roll_max - roll_min).replace(0.0, np.nan)
    df["feature_price_position_250d"] = ((df["close"] - roll_min) / denom).clip(lower=0.0, upper=1.0)
    threshold = df["symbol"].map(lambda s: limit_up_ratio(str(s)) - 0.005).astype(float)
    ret_abs = df["_ret_1d"].abs()
    df["_limit_move_hit"] = (ret_abs >= threshold).astype(float)
    df["feature_limit_move_hits_20d"] = g["_limit_move_hit"].transform(
        lambda s: s.rolling(cfg.limit_move_lookback, min_periods=1).sum()
    )
    # P0-2: 对特征列做截面 Winsorization，消除退市/复牌极端值污染
    df = winsorize_features(df, FEATURE_COLS, date_col="trade_date")
    return df


def winsorize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    date_col: str = "signal_date",
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.DataFrame:
    """对每个截面日期，将 feature_cols 中的值截尾至 [q_low, q_high] 分位数。

    P0-2: 消除退市整理期、复牌后极端收益（±100%）对 XGBoost 分裂节点
    和 OLS 回归系数的污染。
    """
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(date_col, sort=False)[col].transform(
            lambda s: s.clip(
                lower=s.quantile(q_low),
                upper=s.quantile(q_high),
            )
        )
    return out


# ── T+1 可买性 ───────────────────────────────────────────────────────────

def attach_buyability(signal: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if signal.empty:
        return signal.copy()
    df = normalize_daily_frame(daily)
    dates = sorted(pd.to_datetime(df["trade_date"]).dropna().dt.normalize().unique())
    next_by_date = {
        pd.Timestamp(dates[i]).normalize(): pd.Timestamp(dates[i + 1]).normalize()
        for i in range(len(dates) - 1)
    }
    lookup = df.set_index(["symbol", "trade_date"], drop=False)
    out = signal.copy()
    out["next_trade_date"] = out["signal_date"].map(next_by_date)
    flags: list[bool] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        sym = str(getattr(row, "symbol")).zfill(6)
        signal_date = pd.Timestamp(getattr(row, "signal_date")).normalize()
        next_date = getattr(row, "next_trade_date")
        if pd.isna(next_date):
            flags.append(False)
            reasons.append("no_next_trade_date")
            continue
        try:
            r0 = lookup.loc[(sym, signal_date)]
            r1 = lookup.loc[(sym, pd.Timestamp(next_date).normalize())]
        except KeyError:
            flags.append(False)
            reasons.append("missing_next_day_bar")
            continue
        prev_close = float(r0.get("close", np.nan))
        open_px = float(r1.get("open", np.nan))
        close_px = float(r1.get("close", np.nan))
        volume = float(r1.get("volume", np.nan))
        if is_row_suspended_like(volume, open_px, close_px):
            flags.append(False)
            reasons.append("suspended_like_next_open")
        elif is_open_limit_up_unbuyable(open_px, prev_close, sym):
            flags.append(False)
            reasons.append("open_limit_up_unbuyable")
        else:
            flags.append(True)
            reasons.append("")
    out["is_buyable_tplus1_open"] = flags
    out["buyability_reject_reason"] = reasons
    return out


# ── 月度标签构建 ─────────────────────────────────────────────────────────

def build_monthly_labels(
    daily: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    *,
    industry_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if len(signal_dates) < 2:
        return pd.DataFrame(columns=["signal_date", "symbol", *LABEL_COLS])
    daily_norm = normalize_daily_frame(daily)
    returns = build_open_to_open_returns(daily_norm, zero_if_limit_up_open=False).sort_index()
    rows: list[pd.DataFrame] = []
    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        if next_signal_date not in returns.index:
            continue
        window = returns[(returns.index > signal_date) & (returns.index < next_signal_date)]
        if window.empty:
            continue
        window = window.replace([np.inf, -np.inf], np.nan)
        period_ret = (1.0 + window.fillna(0.0)).prod(axis=0) - 1.0
        part = (
            period_ret.rename("label_forward_1m_o2o_return")
            .reset_index()
            .rename(columns={"index": "symbol"})
        )
        part["signal_date"] = signal_date
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["signal_date", "symbol", *LABEL_COLS])
    labels = pd.concat(rows, ignore_index=True)
    labels["symbol"] = labels["symbol"].astype(str).str.zfill(6)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    labels = labels[
        np.isfinite(pd.to_numeric(labels["label_forward_1m_o2o_return"], errors="coerce"))
    ].copy()
    labels["label_market_ew_o2o_return"] = labels.groupby("signal_date", sort=False)[
        "label_forward_1m_o2o_return"
    ].transform("mean")
    labels["label_forward_1m_excess_vs_market"] = (
        labels["label_forward_1m_o2o_return"] - labels["label_market_ew_o2o_return"]
    )
    if industry_map is not None and not industry_map.empty:
        labels = labels.merge(industry_map[["symbol", "industry_level1"]], on="symbol", how="left")
    else:
        labels["industry_level1"] = "_UNKNOWN_"
    labels["industry_level1"] = labels["industry_level1"].fillna("_UNKNOWN_").astype(str)
    labels["_industry_mean"] = labels.groupby(["signal_date", "industry_level1"], sort=False)[
        "label_forward_1m_o2o_return"
    ].transform("mean")
    labels["label_forward_1m_industry_neutral_excess"] = (
        labels["label_forward_1m_o2o_return"] - labels["_industry_mean"]
    )
    pct = labels.groupby("signal_date", sort=False)["label_forward_1m_o2o_return"].rank(
        pct=True, method="average"
    )
    labels["label_future_return_percentile"] = pct
    labels["label_future_return_quantile"] = np.ceil(
        (pct * 10.0).clip(lower=0.0, upper=10.0)
    ).astype("Int64")
    labels["label_future_top_20pct"] = (pct >= 0.80).astype(int)
    labels["label_future_top_10pct"] = (pct >= 0.90).astype(int)
    labels["label_future_bottom_20pct"] = (pct <= 0.20).astype(int)
    return labels.drop(columns=["_industry_mean"], errors="ignore")


# ── 特征截面变换 ─────────────────────────────────────────────────────────

def _winsor_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 3:
        return pd.Series(np.nan, index=series.index)
    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    clipped = x.clip(lo, hi)
    med = clipped.median()
    filled = clipped.fillna(med)
    std = filled.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (filled - filled.mean()) / std


def attach_feature_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in FEATURE_COLS:
        out[f"is_missing_{col}"] = pd.to_numeric(out[col], errors="coerce").isna().astype(int)
        out[f"{col}_z"] = out.groupby("signal_date", sort=False)[col].transform(_winsor_zscore)
    return out


# ── 候选池构建 ───────────────────────────────────────────────────────────

def _join_reasons(items: list[str]) -> str:
    uniq = [x for x in dict.fromkeys(items) if x]
    return ";".join(uniq)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_candidate_pool_panel(
    base_panel: pd.DataFrame,
    cfg: MonthlySelectionConfig,
) -> pd.DataFrame:
    base = base_panel.copy()
    valid_current = (
        np.isfinite(pd.to_numeric(base["open"], errors="coerce"))
        & np.isfinite(pd.to_numeric(base["close"], errors="coerce"))
        & np.isfinite(pd.to_numeric(base["volume"], errors="coerce"))
        & np.isfinite(pd.to_numeric(base["amount"], errors="coerce"))
        & (pd.to_numeric(base["volume"], errors="coerce") > 0)
        & (pd.to_numeric(base["amount"], errors="coerce") > 0)
    )
    base["_u0_pass"] = valid_current & base["is_buyable_tplus1_open"].astype(bool)
    base["_u1_pass"] = (
        base["_u0_pass"]
        & (pd.to_numeric(base["history_days"], errors="coerce") >= int(cfg.min_history_days))
        & (pd.to_numeric(base["amount_20d"], errors="coerce") >= float(cfg.min_amount_20d))
    )
    vol_cut = base.groupby("signal_date", sort=False)["feature_realized_vol_20d"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").quantile(cfg.volatility_high_pct)
    )
    turnover_cut = base.groupby("signal_date", sort=False)["feature_turnover_20d"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").quantile(cfg.turnover_high_pct)
    )
    extreme_risk = (
        (pd.to_numeric(base["feature_limit_move_hits_20d"], errors="coerce") > int(cfg.limit_move_max))
        | (pd.to_numeric(base["feature_realized_vol_20d"], errors="coerce") > vol_cut)
        | (pd.to_numeric(base["feature_turnover_20d"], errors="coerce") > turnover_cut)
        | (
            pd.to_numeric(base["feature_price_position_250d"], errors="coerce")
            >= float(cfg.price_position_high_pct)
        )
    )
    base["_u2_pass"] = base["_u1_pass"] & ~extreme_risk.fillna(False)

    flag_rows: list[list[str]] = []
    for i, (_, row) in enumerate(base.iterrows()):
        flags: list[str] = []
        if not bool(row.get("_u0_pass", False)):
            if not bool(row.get("is_buyable_tplus1_open", False)):
                flags.append(str(row.get("buyability_reject_reason") or "not_buyable_tplus1_open"))
            else:
                flags.append("invalid_current_ohlcv")
        if _as_float(row.get("history_days")) < int(cfg.min_history_days):
            flags.append("insufficient_history")
        amount_20d = _as_float(row.get("amount_20d"))
        if not np.isfinite(amount_20d) or amount_20d < float(cfg.min_amount_20d):
            flags.append("low_liquidity")
        if _as_float(row.get("feature_limit_move_hits_20d")) > int(cfg.limit_move_max):
            flags.append("limit_move_path")
        vol = _as_float(row.get("feature_realized_vol_20d"))
        if np.isfinite(vol):
            row_vol_cut = vol_cut.iloc[i]
            if np.isfinite(row_vol_cut) and vol > row_vol_cut:
                flags.append("extreme_volatility")
        turn = _as_float(row.get("feature_turnover_20d"))
        if np.isfinite(turn):
            row_turn_cut = turnover_cut.iloc[i]
            if np.isfinite(row_turn_cut) and turn > row_turn_cut:
                flags.append("extreme_turnover")
        pp = _as_float(row.get("feature_price_position_250d"))
        if np.isfinite(pp) and pp >= float(cfg.price_position_high_pct):
            flags.append("absolute_high")
        flag_rows.append(flags)
    base["risk_flags"] = [_join_reasons(x) for x in flag_rows]

    frames: list[pd.DataFrame] = []
    for pool, pass_col in [
        ("U0_all_tradable", "_u0_pass"),
        ("U1_liquid_tradable", "_u1_pass"),
        ("U2_risk_sane", "_u2_pass"),
    ]:
        part = base.copy()
        part["candidate_pool_version"] = pool
        part["candidate_pool_rule"] = POOL_RULES[pool]
        part["candidate_pool_pass"] = part[pass_col].astype(bool)
        if pool == "U0_all_tradable":
            part["candidate_pool_reject_reason"] = np.where(
                part["candidate_pool_pass"],
                "",
                np.where(
                    part["is_buyable_tplus1_open"].astype(bool),
                    "invalid_current_ohlcv",
                    part["buyability_reject_reason"],
                ),
            )
        elif pool == "U1_liquid_tradable":
            part["candidate_pool_reject_reason"] = np.where(
                part["candidate_pool_pass"],
                "",
                part["risk_flags"].apply(
                    lambda x: _join_reasons(
                        [
                            r
                            for r in str(x).split(";")
                            if r
                            in {
                                "invalid_current_ohlcv",
                                "missing_next_day_bar",
                                "no_next_trade_date",
                                "suspended_like_next_open",
                                "open_limit_up_unbuyable",
                                "not_buyable_tplus1_open",
                                "insufficient_history",
                                "low_liquidity",
                            }
                        ]
                    )
                ),
            )
        else:
            part["candidate_pool_reject_reason"] = np.where(
                part["candidate_pool_pass"], "", part["risk_flags"]
            )
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    drop_cols = [c for c in out.columns if c.startswith("_u") or c in {"_ret_1d", "_limit_move_hit"}]
    return out.drop(columns=drop_cols, errors="ignore")


# ── 主数据集构建 ─────────────────────────────────────────────────────────

def build_monthly_selection_dataset(
    daily: pd.DataFrame,
    *,
    start_date: str,
    end_date: str | None = None,
    industry_map: pd.DataFrame | None = None,
    cfg: MonthlySelectionConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or MonthlySelectionConfig()
    daily_features = attach_signal_features(daily, cfg)
    signal_dates = select_month_end_signal_dates(
        daily_features["trade_date"], start=start_date, end=end_date
    )
    if not signal_dates:
        return pd.DataFrame()
    base = daily_features[daily_features["trade_date"].isin(signal_dates)].copy()
    base = base.rename(columns={"trade_date": "signal_date"})
    base["signal_date"] = pd.to_datetime(base["signal_date"]).dt.normalize()
    if industry_map is not None and not industry_map.empty:
        base = base.merge(industry_map, on="symbol", how="left")
    else:
        base["industry_level1"] = ""
        base["industry_level2"] = ""
    base["industry_level1"] = base["industry_level1"].fillna("_UNKNOWN_").astype(str)
    base["industry_level2"] = base["industry_level2"].fillna("").astype(str)
    base = attach_buyability(base, daily_features.rename(columns={"signal_date": "trade_date"}))
    labels = build_monthly_labels(
        daily_features.rename(columns={"signal_date": "trade_date"}),
        signal_dates,
        industry_map=industry_map,
    )
    if not labels.empty:
        labels = labels.drop(columns=["industry_level1"], errors="ignore")
        base = base.merge(labels, on=["signal_date", "symbol"], how="left")
    else:
        for col in LABEL_COLS:
            base[col] = np.nan
    base = attach_feature_transforms(base)
    out = build_candidate_pool_panel(base, cfg)
    out["dataset_version"] = "monthly_selection_features_v1"
    out["rebalance_rule"] = "M"
    out["execution_mode"] = "tplus1_open"
    out["label_return_mode"] = "open_to_open"
    out["sell_timing"] = "holding_month_last_trading_day_open"
    return out.sort_values(["signal_date", "candidate_pool_version", "symbol"]).reset_index(drop=True)


# ── 质量汇总 ─────────────────────────────────────────────────────────────

def summarize_candidate_width(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    out = (
        dataset.groupby(["signal_date", "candidate_pool_version"], dropna=False)
        .agg(
            raw_universe_width=("symbol", "nunique"),
            candidate_pool_width=("candidate_pool_pass", "sum"),
            label_valid_count=(
                "label_forward_1m_o2o_return",
                lambda s: pd.to_numeric(s, errors="coerce").notna().sum(),
            ),
        )
        .reset_index()
    )
    out["candidate_pool_pass_ratio"] = out["candidate_pool_width"] / out["raw_universe_width"].replace(0, np.nan)
    return out


def summarize_reject_reasons(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    failed = dataset[~dataset["candidate_pool_pass"].astype(bool)].copy()
    for (signal_date, pool), part in failed.groupby(
        ["signal_date", "candidate_pool_version"], sort=True
    ):
        counts: dict[str, int] = {}
        for raw in part["candidate_pool_reject_reason"].fillna("").astype(str):
            for reason in [x for x in raw.split(";") if x]:
                counts[reason] = counts.get(reason, 0) + 1
        for reason, count in sorted(counts.items()):
            rows.append(
                {
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "reject_reason": reason,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def summarize_feature_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    base = dataset[dataset["candidate_pool_version"] == "U1_liquid_tradable"].copy()
    rows: list[dict[str, Any]] = []
    for col in FEATURE_COLS:
        vals = pd.to_numeric(base[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "rows": int(len(base)),
                "non_null": int(vals.notna().sum()),
                "coverage_ratio": float(vals.notna().mean()) if len(base) else np.nan,
                "candidate_pool_pass_coverage_ratio": (
                    float(vals[base["candidate_pool_pass"].astype(bool)].notna().mean())
                    if base["candidate_pool_pass"].any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_label_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    base = dataset[dataset["candidate_pool_version"] == "U1_liquid_tradable"].copy()
    rows: list[dict[str, Any]] = []
    for signal_date, part in base.groupby("signal_date", sort=True):
        vals = pd.to_numeric(
            part.loc[
                part["candidate_pool_pass"].astype(bool), "label_forward_1m_o2o_return"
            ],
            errors="coerce",
        )
        rows.append(
            {
                "signal_date": signal_date,
                "candidate_pool_version": "U1_liquid_tradable",
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                "median": float(vals.median()) if vals.notna().any() else np.nan,
                "p10": float(vals.quantile(0.10)) if vals.notna().any() else np.nan,
                "p90": float(vals.quantile(0.90)) if vals.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_quality_summary(
    dataset: pd.DataFrame,
    *,
    research_topic: str,
    research_config_id: str,
    output_stem: str,
    config_source: str,
    industry_map_source_status: str,
) -> pd.DataFrame:
    if dataset.empty:
        n_rows = n_symbols = n_signal_dates = 0
        min_signal = max_signal = ""
        label_valid_rows = 0
    else:
        n_rows = len(dataset)
        n_symbols = dataset["symbol"].nunique()
        n_signal_dates = dataset["signal_date"].nunique()
        min_signal = str(pd.to_datetime(dataset["signal_date"]).min().date())
        max_signal = str(pd.to_datetime(dataset["signal_date"]).max().date())
        label_base = dataset[dataset["candidate_pool_version"] == "U1_liquid_tradable"].copy()
        label_valid_rows = int(
            pd.to_numeric(
                label_base["label_forward_1m_o2o_return"], errors="coerce"
            ).notna().sum()
        )
    return pd.DataFrame(
        [
            {
                "result_type": "monthly_selection_dataset_quality",
                "research_topic": research_topic,
                "research_config_id": research_config_id,
                "output_stem": output_stem,
                "config_source": config_source,
                "dataset_version": "monthly_selection_features_v1",
                "rebalance_rule": "M",
                "execution_mode": "tplus1_open",
                "benchmark_return_mode": "market_ew_open_to_open",
                "sell_timing": "holding_month_last_trading_day_open",
                "candidate_pool_versions": ",".join(POOL_RULES),
                "industry_map_source_status": industry_map_source_status,
                "rows": n_rows,
                "symbols": n_symbols,
                "signal_months": n_signal_dates,
                "min_signal_date": min_signal,
                "max_signal_date": max_signal,
                "label_valid_rows": label_valid_rows,
            }
        ]
    )
