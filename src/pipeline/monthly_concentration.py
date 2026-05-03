"""月度选股行业集中度约束核心算法。

从 scripts/run_monthly_selection_concentration_regime.py 提取：
- 行业 cap 贪婪选择算法
- 受约束月度选股构建
- TopK 与 cap grid 解析
- 交易日期附加与换手计算
- M8 分析辅助：行业集中度汇总、Leaderboard、Gate、Regime 策略、Lagged 状态

不放 CLI 参数解析与文件 I/O 编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.monthly_baselines import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    TOP20_COL,
)

TOPK_PRESET_DEFAULT = "default"
TOPK_PRESET_NARROW = "narrow"
TOPK_PRESETS: dict[str, str] = {
    TOPK_PRESET_DEFAULT: "20,30,50",
    TOPK_PRESET_NARROW: "5,10,20",
}


@dataclass(frozen=True)
class M8RunConfig:
    top_ks: tuple[int, ...] = (20, 30, 50)
    candidate_pools: tuple[str, ...] = ("U1_liquid_tradable", "U2_risk_sane")
    bucket_count: int = 5
    min_train_months: int = 24
    min_train_rows: int = 500
    max_fit_rows: int = 0
    cost_bps: float = 10.0
    random_seed: int = 42
    availability_lag_days: int = 30
    min_state_history_months: int = 24
    cap_grid: dict[int, tuple[int, ...]] | None = None
    model_n_jobs: int = 0


# ── 行业 cap 选择 ─────────────────────────────────────────────────────────

def select_with_industry_cap(
    part: pd.DataFrame, *, k: int, max_industry_names: int,
) -> pd.DataFrame:
    """Greedy score selection with an optional max names per industry cap.

    按 score 降序贪婪选取，每个行业最多选取 max_industry_names 只股票。
    若 max_industry_names <= 0 则不做行业限制，直接取 TopK。
    """
    if part.empty or int(k) <= 0:
        return part.iloc[0:0].copy()
    ordered = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
    if int(max_industry_names) <= 0:
        return ordered.head(int(k)).copy()
    counts: dict[str, int] = {}
    selected_idx: list[Any] = []
    for idx, row in ordered.iterrows():
        industry = str(row.get("industry_level1") if pd.notna(row.get("industry_level1")) else "_UNKNOWN_")
        if counts.get(industry, 0) >= int(max_industry_names):
            continue
        selected_idx.append(idx)
        counts[industry] = counts.get(industry, 0) + 1
        if len(selected_idx) >= int(k):
            break
    return ordered.loc[selected_idx].copy()


def _industry_threshold(top_k: int) -> float:
    if int(top_k) <= 20:
        return 0.30
    if int(top_k) <= 30:
        return 0.35
    return 0.35


# ── 换手计算 ──────────────────────────────────────────────────────────────

def _weighted_turnover(prev: set[str] | None, cur: set[str]) -> float:
    """计算两只组合之间的加权换手率（等权 L1 距离）。"""
    if prev is None:
        return np.nan
    all_symbols = prev | cur
    prev_w = {s: 1.0 / max(len(prev), 1) for s in prev}
    cur_w = {s: 1.0 / max(len(cur), 1) for s in cur}
    return float(0.5 * sum(abs(cur_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_symbols))


# ── 日期工具 ──────────────────────────────────────────────────────────────

def _date_iso(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def _first_date_iso(values: Any) -> str:
    if values is None:
        return ""
    s = pd.Series(values).dropna()
    if s.empty:
        return ""
    return _date_iso(s.iloc[0])


def _next_signal_date_map(
    signal_calendar: list[Any] | tuple[Any, ...] | pd.Series | np.ndarray | None,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """构建信号日到下个信号日的映射。"""
    if signal_calendar is None:
        return {}
    dates = pd.Series(pd.to_datetime(pd.Series(signal_calendar), errors="coerce")).dropna().dt.normalize()
    dates = sorted(pd.Timestamp(x).normalize() for x in dates.unique())
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


# ── TopK 与 cap grid ─────────────────────────────────────────────────────

def default_cap_values_for_topk(top_k: int) -> tuple[int, ...]:
    k = int(top_k)
    if k <= 5:
        return (1, 2)
    if k <= 10:
        return (2, 3)
    if k <= 20:
        return (3, 4, 5)
    if k <= 30:
        return (4, 5, 6, 8)
    return (6, 8, 10)


def build_default_cap_grid(top_ks: list[int]) -> dict[int, tuple[int, ...]]:
    return {int(k): default_cap_values_for_topk(int(k)) for k in top_ks}


def parse_cap_grid(raw: str, top_ks: list[int]) -> dict[int, tuple[int, ...]]:
    """解析 cap-grid 字符串，格式如 "20:3,4,5;30:4,5,6,8"。"""
    out: dict[int, tuple[int, ...]] = {}
    for chunk in str(raw).split(";"):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            raise ValueError(f"cap-grid 片段缺少 ':': {chunk}")
        k_raw, vals_raw = chunk.split(":", 1)
        k = int(k_raw.strip())
        vals = tuple(sorted({int(x.strip()) for x in vals_raw.split(",") if x.strip()}))
        if vals:
            out[k] = vals
    for k in top_ks:
        out.setdefault(k, tuple())
    return out


def serialize_cap_grid(cap_grid: dict[int, tuple[int, ...]]) -> str:
    parts: list[str] = []
    for k in sorted(cap_grid):
        vals = ",".join(str(v) for v in cap_grid[k])
        parts.append(f"{int(k)}:{vals}")
    return ";".join(parts)


def resolve_topk_and_cap_grid(
    *, preset: str, top_k_raw: str, cap_grid_raw: str,
) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    preset_topk = TOPK_PRESETS.get(str(preset), TOPK_PRESETS[TOPK_PRESET_DEFAULT])
    top_ks = sorted({int(x.strip()) for x in str(top_k_raw).strip().split(",") if x.strip()}) if str(top_k_raw).strip() else [
        int(x.strip()) for x in preset_topk.split(",")
    ]
    if str(cap_grid_raw).strip():
        cap_grid = parse_cap_grid(cap_grid_raw, top_ks)
    else:
        cap_grid = build_default_cap_grid(top_ks)
    return top_ks, cap_grid


# ── 交易日期附加 ──────────────────────────────────────────────────────────

def attach_trade_dates_to_scores(
    scores: pd.DataFrame, dataset: pd.DataFrame,
) -> pd.DataFrame:
    """将 dataset 中的 next_trade_date 附加到 scores。"""
    if scores.empty or dataset.empty or "next_trade_date" not in dataset.columns:
        return scores.copy()
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    if not set(keys).issubset(scores.columns) or not set(keys).issubset(dataset.columns):
        return scores.copy()
    dates = dataset[keys + ["next_trade_date"]].copy()
    dates["signal_date"] = pd.to_datetime(dates["signal_date"], errors="coerce").dt.normalize()
    dates["symbol"] = dates["symbol"].astype(str).str.zfill(6)
    dates = dates.drop_duplicates(keys, keep="last")
    out = scores.drop(columns=["next_trade_date"], errors="ignore").copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    return out.merge(dates, on=keys, how="left")


# ── 受约束月度选股构建 ────────────────────────────────────────────────────

def build_constrained_monthly(
    scores: pd.DataFrame,
    *,
    top_ks: list[int],
    cap_grid: dict[int, tuple[int, ...]],
    cost_bps: float,
    signal_calendar: list[Any] | tuple[Any, ...] | pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每个 (pool, base_model, signal_date) 构建行业 cap 约束下的选股组合。

    Returns:
        (summary_df, holdings_df): summary 每行一个变动，holdings 每行一只选股。
    """
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    holdings: list[pd.DataFrame] = []
    prev_by_key: dict[tuple[str, str, int, int], set[str]] = {}
    next_signal_by_date = _next_signal_date_map(
        signal_calendar if signal_calendar is not None else scores["signal_date"]
    )
    ordered = scores.sort_values(
        ["candidate_pool_version", "model", "signal_date", "score", "symbol"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    for (pool, base_model, signal_date), part in ordered.groupby(
        ["candidate_pool_version", "model", "signal_date"], sort=True
    ):
        part = part.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").copy()
        candidate_width = int(part["symbol"].nunique())
        market_ret = float(part[MARKET_COL].dropna().iloc[0]) if part[MARKET_COL].notna().any() else np.nan
        pool_ret = float(pd.to_numeric(part[LABEL_COL], errors="coerce").mean())
        model_type = str(part["model_type"].dropna().iloc[0]) if part["model_type"].notna().any() else ""
        for k in top_ks:
            caps = [0, *[int(x) for x in cap_grid.get(int(k), tuple())]]
            for cap in caps:
                selected = select_with_industry_cap(part, k=int(k), max_industry_names=int(cap))
                if selected.empty:
                    continue
                remaining = part.drop(index=selected.index, errors="ignore")
                nextk = select_with_industry_cap(remaining, k=int(k), max_industry_names=int(cap))
                variant_model = f"{base_model}__{'uncapped' if int(cap) <= 0 else f'indcap{int(cap)}'}"
                cur_symbols = set(selected["symbol"].astype(str))
                turnover = _weighted_turnover(
                    prev_by_key.get((pool, variant_model, int(k), int(cap))), cur_symbols,
                )
                prev_by_key[(pool, variant_model, int(k), int(cap))] = cur_symbols
                top_ret = float(pd.to_numeric(selected[LABEL_COL], errors="coerce").mean())
                next_ret = float(pd.to_numeric(nextk[LABEL_COL], errors="coerce").mean()) if not nextk.empty else np.nan
                top_ind_excess = (
                    float(pd.to_numeric(selected[INDUSTRY_EXCESS_COL], errors="coerce").mean())
                    if INDUSTRY_EXCESS_COL in selected.columns
                    else np.nan
                )
                buy_trade_date = (
                    _first_date_iso(selected["next_trade_date"]) if "next_trade_date" in selected.columns else ""
                )
                sell_trade_date = _date_iso(next_signal_by_date.get(pd.Timestamp(signal_date).normalize()))
                h = selected.copy()
                h["base_model"] = base_model
                h["model"] = variant_model
                h["base_model_type"] = model_type
                h["model_type"] = "unconstrained_selection" if int(cap) <= 0 else "industry_cap_selection"
                h["top_k"] = int(k)
                h["selected_rank"] = np.arange(1, len(h) + 1)
                h["selection_policy"] = "uncapped" if int(cap) <= 0 else "industry_names_cap"
                h["max_industry_names"] = int(cap)
                h["buy_trade_date"] = buy_trade_date
                h["sell_trade_date"] = sell_trade_date
                keep_cols = [
                    "signal_date", "candidate_pool_version", "base_model", "model",
                    "base_model_type", "model_type", "selection_policy",
                    "max_industry_names", "top_k", "selected_rank", "symbol", "score",
                    LABEL_COL, "industry_level1", "risk_flags",
                    "next_trade_date", "buy_trade_date", "sell_trade_date",
                ]
                holdings.append(h[[c for c in keep_cols if c in h.columns]].copy())
                rows.append({
                    "signal_date": signal_date,
                    "candidate_pool_version": pool,
                    "base_model": base_model,
                    "model": variant_model,
                    "base_model_type": model_type,
                    "model_type": "unconstrained_selection" if int(cap) <= 0 else "industry_cap_selection",
                    "selection_policy": "uncapped" if int(cap) <= 0 else "industry_names_cap",
                    "max_industry_names": int(cap),
                    "top_k": int(k),
                    "buy_trade_date": buy_trade_date,
                    "sell_trade_date": sell_trade_date,
                    "candidate_pool_width": candidate_width,
                    "selected_count": int(len(selected)),
                    "topk_return": top_ret,
                    "market_ew_return": market_ret,
                    "topk_excess_vs_market": float(top_ret - market_ret) if np.isfinite(market_ret) else np.nan,
                    "topk_industry_neutral_excess": top_ind_excess,
                    "candidate_pool_mean_return": pool_ret,
                    "topk_minus_pool_mean": float(top_ret - pool_ret),
                    "nextk_return": next_ret,
                    "topk_minus_nextk": float(top_ret - next_ret) if np.isfinite(next_ret) else np.nan,
                    "turnover_half_l1": turnover,
                    "cost_bps": float(cost_bps),
                    "cost_drag": float(turnover * cost_bps / 10000.0) if np.isfinite(turnover) else np.nan,
                    "topk_excess_after_cost": float(top_ret - market_ret - turnover * cost_bps / 10000.0)
                    if np.isfinite(market_ret) and np.isfinite(turnover)
                    else np.nan,
                })
    return pd.DataFrame(rows), pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()


# ── M8 模型名称常量 ──────────────────────────────────────────────────────

STABLE_M5_ELASTICNET = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_elasticnet_excess"
STABLE_M5_EXTRATREES = "M5_plus_industry_breadth_plus_fund_flow_plus_fundamental_extratrees_excess"
WATCHLIST_M6_RANK = "M6_xgboost_rank_ndcg"
WATCHLIST_M6_TOP20 = "M6_top20_calibrated"
M8_POLICY_MODEL = "M8_regime_aware_fixed_policy"


# ── 行业集中度汇总 ───────────────────────────────────────────────────────

def summarize_industry_concentration(holdings: pd.DataFrame) -> pd.DataFrame:
    """P3-2: 从 holdings 计算行业集中度统计。"""
    if holdings.empty:
        return pd.DataFrame()
    h = holdings.copy()
    h["industry_level1"] = h["industry_level1"].fillna("_UNKNOWN_").astype(str)
    counts = (
        h.groupby(
            [
                "signal_date", "candidate_pool_version", "base_model", "model",
                "top_k", "selection_policy", "max_industry_names", "industry_level1",
            ],
            sort=True,
        )["symbol"]
        .nunique()
        .rename("industry_symbol_count")
        .reset_index()
    )
    total = (
        counts.groupby(
            [
                "signal_date", "candidate_pool_version", "base_model", "model",
                "top_k", "selection_policy", "max_industry_names",
            ],
            sort=True,
        )
        .agg(
            selected_count=("industry_symbol_count", "sum"),
            industry_count=("industry_level1", "nunique"),
            max_industry_count=("industry_symbol_count", "max"),
        )
        .reset_index()
    )
    total["max_industry_share"] = total["max_industry_count"] / total["selected_count"].replace(0, np.nan)
    total["concentration_threshold"] = total["top_k"].map(_industry_threshold)
    total["concentration_pass"] = total["max_industry_share"] <= total["concentration_threshold"]
    return total


def _summarize_concentration_for_leaderboard(industry_concentration: pd.DataFrame) -> pd.DataFrame:
    if industry_concentration.empty:
        return pd.DataFrame()
    return (
        industry_concentration.groupby(
            ["candidate_pool_version", "model", "top_k", "selection_policy", "max_industry_names"], sort=True
        )
        .agg(
            max_industry_share_mean=("max_industry_share", "mean"),
            max_industry_share_max=("max_industry_share", "max"),
            industry_count_mean=("industry_count", "mean"),
            concentration_pass_rate=("concentration_pass", "mean"),
        )
        .reset_index()
    )


# ── Leaderboard ──────────────────────────────────────────────────────────

def build_constrained_leaderboard(
    monthly: pd.DataFrame,
    rank_ic: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    regime_slice: pd.DataFrame,
    industry_concentration: pd.DataFrame,
) -> pd.DataFrame:
    """P3-2: 构建 M8 约束选股 leaderboard。"""
    if monthly.empty:
        return pd.DataFrame()
    agg = (
        monthly.groupby(
            [
                "candidate_pool_version", "base_model", "model",
                "base_model_type", "model_type", "selection_policy",
                "max_industry_names", "top_k",
            ],
            sort=True,
        )
        .agg(
            months=("signal_date", "nunique"),
            mean_topk_return=("topk_return", "mean"),
            median_topk_return=("topk_return", "median"),
            topk_excess_mean=("topk_excess_vs_market", "mean"),
            topk_excess_median=("topk_excess_vs_market", "median"),
            topk_hit_rate=("topk_excess_vs_market", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            topk_minus_nextk_mean=("topk_minus_nextk", "mean"),
            industry_neutral_topk_excess_mean=("topk_industry_neutral_excess", "mean"),
            turnover_mean=("turnover_half_l1", "mean"),
            cost_drag_mean=("cost_drag", "mean"),
            topk_excess_after_cost_mean=("topk_excess_after_cost", "mean"),
        )
        .reset_index()
    )
    agg["topk_excess_annualized"] = (1.0 + agg["topk_excess_mean"]).pow(12) - 1.0

    if not rank_ic.empty:
        ic = (
            rank_ic.groupby(["candidate_pool_version", "model"], sort=True)
            .agg(rank_ic_mean=("rank_ic", "mean"), rank_ic_std=("rank_ic", "std"), rank_ic_months=("rank_ic", "count"))
            .reset_index()
            .rename(columns={"model": "base_model"})
        )
        ic["rank_ic_ir"] = ic["rank_ic_mean"] / ic["rank_ic_std"].replace(0.0, np.nan)
        agg = agg.merge(ic, on=["candidate_pool_version", "base_model"], how="left")

    if not quantile_spread.empty:
        qs = (
            quantile_spread.groupby(["candidate_pool_version", "model"], sort=True)
            .agg(quantile_top_minus_bottom_mean=("top_minus_bottom_return", "mean"))
            .reset_index()
            .rename(columns={"model": "base_model"})
        )
        agg = agg.merge(qs, on=["candidate_pool_version", "base_model"], how="left")

    if not regime_slice.empty:
        rs = regime_slice.pivot_table(
            index=["candidate_pool_version", "model", "top_k"],
            columns="realized_market_state",
            values="median_topk_excess",
            aggfunc="first",
        ).reset_index()
        rs.columns = [
            f"{c}_median_excess" if c in {"strong_up", "strong_down", "neutral"} else c for c in rs.columns
        ]
        agg = agg.merge(rs, on=["candidate_pool_version", "model", "top_k"], how="left")

    conc = _summarize_concentration_for_leaderboard(industry_concentration)
    if not conc.empty:
        agg = agg.merge(
            conc,
            on=["candidate_pool_version", "model", "top_k", "selection_policy", "max_industry_names"],
            how="left",
        )

    return agg.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "max_industry_share_mean", "rank_ic_mean"],
        ascending=[True, True, False, True, False],
    )


# ── Gate 表 ──────────────────────────────────────────────────────────────

def _concentration_gate_threshold(top_k: int) -> float:
    return 0.30 if int(top_k) <= 20 else 0.35


def build_gate_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """P3-2: 从 leaderboard 构建 gate 通行表。"""
    if leaderboard.empty:
        return pd.DataFrame()

    def _numeric_col(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(default, index=frame.index, dtype=float)
        return pd.to_numeric(frame[col], errors="coerce")

    baseline = leaderboard[
        (leaderboard["selection_policy"] == "uncapped")
        & (leaderboard["base_model"].isin([STABLE_M5_ELASTICNET, STABLE_M5_EXTRATREES]))
    ].copy()
    baseline = (
        baseline.groupby(["candidate_pool_version", "top_k"], sort=True)
        .agg(
            m5_stable_after_cost_baseline=("topk_excess_after_cost_mean", "max"),
            m5_stable_rank_ic_baseline=("rank_ic_mean", "max"),
        )
        .reset_index()
    )
    out = leaderboard.merge(baseline, on=["candidate_pool_version", "top_k"], how="left")
    out["baseline_delta_after_cost"] = out["topk_excess_after_cost_mean"] - out["m5_stable_after_cost_baseline"]
    out["baseline_gate"] = out["baseline_delta_after_cost"] >= -0.003
    out["rank_gate"] = _numeric_col(out, "rank_ic_mean") > 0
    out["spread_gate"] = _numeric_col(out, "topk_minus_nextk_mean") > 0
    out["year_regime_gate"] = (
        _numeric_col(out, "strong_down_median_excess").fillna(0.0) > -0.03
    ) & (_numeric_col(out, "strong_up_median_excess").fillna(0.0) > -0.03)
    out["concentration_gate"] = _numeric_col(out, "concentration_pass_rate").fillna(0.0) >= 0.999
    out["cost_gate_10bps"] = _numeric_col(out, "topk_excess_after_cost_mean") > 0
    gate_cols = [
        "candidate_pool_version", "base_model", "model", "selection_policy",
        "max_industry_names", "top_k", "topk_excess_after_cost_mean",
        "m5_stable_after_cost_baseline", "baseline_delta_after_cost",
        "rank_ic_mean", "topk_minus_nextk_mean", "max_industry_share_mean",
        "concentration_pass_rate", "industry_count_mean",
        "baseline_gate", "rank_gate", "spread_gate", "year_regime_gate",
        "concentration_gate", "cost_gate_10bps",
    ]
    out["m8_gate_pass"] = out[
        ["baseline_gate", "rank_gate", "spread_gate", "year_regime_gate", "concentration_gate", "cost_gate_10bps"]
    ].all(axis=1)
    return out[[c for c in gate_cols if c in out.columns] + ["m8_gate_pass"]].sort_values(
        ["top_k", "candidate_pool_version", "m8_gate_pass", "topk_excess_after_cost_mean"],
        ascending=[True, True, False, False],
    )


# ── Lagged 状态与 Regime 策略 ─────────────────────────────────────────────

def build_lagged_state_frame(dataset: pd.DataFrame, *, min_history_months: int = 24) -> pd.DataFrame:
    """P3-2: 构建滞后市场状态帧。"""
    base = dataset[dataset[LABEL_COL].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=["signal_date", "lagged_regime", "breadth_state"])
    monthly = (
        base.groupby("signal_date", sort=True)
        .agg(
            market_ew_return=(MARKET_COL, "first"),
            breadth_positive_ret20=("feature_ret_20d", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    monthly["signal_date"] = pd.to_datetime(monthly["signal_date"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    for i, row in monthly.iterrows():
        hist = monthly.iloc[:i].copy()
        lagged_regime = "neutral"
        breadth_state = "normal"
        if len(hist) >= int(min_history_months):
            prev = hist.iloc[-1]
            mvals = pd.to_numeric(hist["market_ew_return"], errors="coerce")
            lo = float(mvals.quantile(0.20))
            hi = float(mvals.quantile(0.80))
            prev_ret = float(prev["market_ew_return"])
            if np.isfinite(prev_ret) and np.isfinite(lo) and prev_ret <= lo:
                lagged_regime = "strong_down"
            elif np.isfinite(prev_ret) and np.isfinite(hi) and prev_ret >= hi:
                lagged_regime = "strong_up"
            bvals = pd.to_numeric(hist["breadth_positive_ret20"], errors="coerce")
            blo = float(bvals.quantile(0.20))
            bhi = float(bvals.quantile(0.80))
            cur_breadth = float(row["breadth_positive_ret20"])
            if np.isfinite(cur_breadth) and np.isfinite(blo) and cur_breadth <= blo:
                breadth_state = "narrow"
            elif np.isfinite(cur_breadth) and np.isfinite(bhi) and cur_breadth >= bhi:
                breadth_state = "wide"
        rows.append({
            "signal_date": row["signal_date"],
            "market_ew_return": row["market_ew_return"],
            "breadth_positive_ret20": row["breadth_positive_ret20"],
            "lagged_regime": lagged_regime,
            "breadth_state": breadth_state,
            "state_history_months": int(len(hist)),
        })
    return pd.DataFrame(rows)


def build_regime_policy_scores(scores: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """P3-2: 基于滞后市场状态的固定策略 blend 得分。

    使用 _rank_pct_score 做截面排名归一化（需要从调用方注入）。
    """
    from src.pipeline.monthly_baselines import _rank_pct_score as _rps

    if scores.empty:
        return pd.DataFrame()
    wanted = {
        STABLE_M5_ELASTICNET: "_elasticnet_score",
        STABLE_M5_EXTRATREES: "_extratrees_score",
        WATCHLIST_M6_RANK: "_rank_score",
        WATCHLIST_M6_TOP20: "_top20_score",
    }
    base_cols = [
        "signal_date", "candidate_pool_version", "symbol",
        LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL,
        "industry_level1", "industry_level2", "log_market_cap",
        "risk_flags", "is_buyable_tplus1_open", "next_trade_date",
    ]
    frames: list[pd.DataFrame] = []
    for model, score_col in wanted.items():
        part = scores[scores["model"] == model].copy()
        if part.empty:
            continue
        cols = [c for c in base_cols if c in part.columns]
        part = part[cols + ["score"]].rename(columns={"score": score_col})
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    keys = ["signal_date", "candidate_pool_version", "symbol"]
    for frame in frames[1:]:
        score_cols = [c for c in frame.columns if c.startswith("_") and c.endswith("_score")]
        merged = merged.merge(frame[keys + score_cols], on=keys, how="outer")
    merged = merged.merge(states[["signal_date", "lagged_regime", "breadth_state"]], on="signal_date", how="left")
    merged["lagged_regime"] = merged["lagged_regime"].fillna("neutral")
    merged["breadth_state"] = merged["breadth_state"].fillna("normal")
    for col in wanted.values():
        if col not in merged.columns:
            merged[col] = np.nan
    stable_mean = merged[["_elasticnet_score", "_extratrees_score"]].mean(axis=1)
    for col in wanted.values():
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(stable_mean).fillna(0.5)

    score = pd.Series(0.5 * merged["_elasticnet_score"] + 0.5 * merged["_extratrees_score"], index=merged.index)
    down = merged["lagged_regime"].eq("strong_down")
    up_or_wide = merged["lagged_regime"].eq("strong_up") | merged["breadth_state"].eq("wide")
    narrow = merged["breadth_state"].eq("narrow") & ~up_or_wide
    score.loc[down] = 0.80 * merged.loc[down, "_elasticnet_score"] + 0.20 * merged.loc[down, "_extratrees_score"]
    score.loc[up_or_wide] = (
        0.60 * merged.loc[up_or_wide, "_extratrees_score"]
        + 0.25 * merged.loc[up_or_wide, "_rank_score"]
        + 0.15 * merged.loc[up_or_wide, "_top20_score"]
    )
    score.loc[narrow] = 0.55 * merged.loc[narrow, "_elasticnet_score"] + 0.45 * merged.loc[narrow, "_extratrees_score"]

    out = merged[[c for c in base_cols if c in merged.columns] + ["lagged_regime", "breadth_state"]].copy()
    out["model"] = M8_POLICY_MODEL
    out["model_type"] = "lagged_regime_fixed_score_blend"
    out["score"] = score.groupby([out["signal_date"], out["candidate_pool_version"]], sort=False).transform(_rps)
    out["rank"] = out.groupby(["signal_date", "candidate_pool_version", "model"], sort=False)["score"].rank(
        method="first", ascending=False,
    )
    return out
