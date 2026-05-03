"""研究门控与过滤器：共享的候选池规则、预过滤器、因子过滤策略。

从 scripts/run_backtest_eval.py 和 scripts/run_monthly_selection_*.py 中提取，
确保各模块对过滤逻辑的定义唯一、可复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 月度选股共享的候选池规则
# ---------------------------------------------------------------------------

POOL_RULES: dict[str, str] = {
    "U0_all_tradable": "valid current OHLCV + buyable at next trading day's open",
    "U1_liquid_tradable": "U0 + minimum history length + 20d average amount threshold",
    "U2_risk_sane": (
        "U1 + exclude extreme limit-move path, extreme volatility/turnover, "
        "and absolute-high names"
    ),
}

# ---------------------------------------------------------------------------
# 共享的标签列名
# ---------------------------------------------------------------------------

LABEL_COL = "label_forward_1m_o2o_return"
EXCESS_COL = "label_forward_1m_excess_vs_market"
INDUSTRY_EXCESS_COL = "label_forward_1m_industry_neutral_excess"
MARKET_COL = "label_market_ew_o2o_return"
TOP20_COL = "label_future_top_20pct"


# ---------------------------------------------------------------------------
# 预过滤器（回测用）
# ---------------------------------------------------------------------------

def apply_prefilter(
    day_s: pd.DataFrame,
    top_k: int,
    *,
    enabled: bool = True,
    limit_move_max: int = 2,
    turnover_low_pct: float = 0.10,
    turnover_high_pct: float = 0.98,
    price_position_high_pct: float = 0.90,
    limit_move_col: str = "limit_move_hits_5d",
    turnover_col: str = "turnover_roll_mean",
    price_position_col: str = "price_position",
) -> pd.DataFrame:
    """对单日截面执行预过滤：涨跌停次数、极端换手、绝对高位。"""
    if not enabled or len(day_s) < top_k:
        return day_s

    filtered = day_s.copy()

    lm = pd.to_numeric(filtered.get(limit_move_col, np.nan), errors="coerce").fillna(0.0)
    filtered = filtered[lm <= float(limit_move_max)]

    to = pd.to_numeric(filtered.get(turnover_col, np.nan), errors="coerce")
    if to.notna().sum() >= max(50, top_k):
        lo = to.quantile(turnover_low_pct)
        hi = to.quantile(turnover_high_pct)
        filtered = filtered[(to >= lo) & (to <= hi) | to.isna()]

    pp = pd.to_numeric(filtered.get(price_position_col, np.nan), errors="coerce")
    filtered = filtered[(pp <= price_position_high_pct) | pp.isna()]

    if len(filtered) < top_k:
        return day_s
    return filtered


# ---------------------------------------------------------------------------
# Universe 过滤器（M2.4：流动性 + 盈利）
# ---------------------------------------------------------------------------

def attach_universe_filter(
    factors: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    enabled: bool,
    min_amount_20d: float,
    require_roe_ttm_positive: bool,
) -> pd.DataFrame:
    """M2.4：流动性（20 日日均成交额）+ 盈利（ROE_TTM>0）universe。"""
    out = factors.copy()
    if not enabled:
        out["_universe_eligible"] = True
        return out
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    amt = pd.to_numeric(d["amount"], errors="coerce")
    d["_amt20"] = amt.groupby(d["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    liq = d[["symbol", "trade_date", "_amt20"]].drop_duplicates(
        ["symbol", "trade_date"], keep="last"
    )
    out = out.merge(liq, on=["symbol", "trade_date"], how="left")
    roe = (
        pd.to_numeric(out["roe_ttm"], errors="coerce")
        if "roe_ttm" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    ok = pd.Series(True, index=out.index)
    if float(min_amount_20d) > 0.0:
        ok &= out["_amt20"].fillna(0.0) >= float(min_amount_20d)
    if bool(require_roe_ttm_positive):
        ok &= roe.fillna(-np.inf) > 0.0
    out["_universe_eligible"] = ok.to_numpy(dtype=bool)
    return out.drop(columns=["_amt20"])


# ---------------------------------------------------------------------------
# P1 因子过滤策略
# ---------------------------------------------------------------------------

def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化因子权重，使绝对值之和为 1。"""
    total = sum(abs(w) for w in weights.values())
    if total <= 1e-12:
        raise ValueError("所有权重绝对值之和为 0，无法归一化")
    return {k: v / total for k, v in weights.items()}


def load_factor_ic_summary(ic_report_path: str) -> pd.DataFrame:
    """加载因子 IC 汇总报告。"""
    from pathlib import Path

    p = Path(ic_report_path)
    if not p.is_absolute():
        # 相对于项目根
        from pathlib import Path as _Path
        p = _Path(__file__).resolve().parents[2] / p
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "factor" not in df.columns:
        return pd.DataFrame()
    return df


def apply_p1_factor_policy(
    base_weights: dict[str, float],
    ic_summary: pd.DataFrame,
    *,
    remove_if_t1_and_t21_negative: bool = True,
    zero_if_abs_t1_below: float = 0.0,
    flip_if_t1_negative_and_t21_above: float = 0.005,
) -> tuple[dict[str, float], pd.DataFrame]:
    """根据 IC 汇总对因子权重执行 P1 策略：剔除/归零/反转。"""
    updated = dict(base_weights)
    rows: list[dict[str, Any]] = []
    for fac, cur_w in base_weights.items():
        match = ic_summary[ic_summary["factor"].astype(str).str.strip() == str(fac).strip()]
        if match.empty:
            rows.append({"factor": fac, "action": "keep", "ic_mean_t1": np.nan, "ic_mean_t21": np.nan})
            continue
        t1 = float(match["ic_mean_t1"].iloc[0]) if "ic_mean_t1" in match.columns else np.nan
        t21 = float(match["ic_mean_t21"].iloc[0]) if "ic_mean_t21" in match.columns else np.nan
        action = "keep"
        if remove_if_t1_and_t21_negative and np.isfinite(t1) and np.isfinite(t21) and t1 < 0.0 and t21 < 0.0:
            updated.pop(fac, None)
            action = "remove"
        elif zero_if_abs_t1_below > 0 and np.isfinite(t1) and abs(t1) < float(zero_if_abs_t1_below):
            updated[fac] = 0.0
            action = "zero"
        elif (
            np.isfinite(t1)
            and np.isfinite(t21)
            and t1 < 0.0
            and t21 > float(flip_if_t1_negative_and_t21_above)
            and abs(cur_w) > 1e-12
        ):
            updated[fac] = -cur_w
            action = "flip"
        rows.append({"factor": fac, "action": action, "ic_mean_t1": t1, "ic_mean_t21": t21})
    try:
        normalized = normalize_weights(updated)
    except ValueError:
        normalized = dict(base_weights)
    return normalized, pd.DataFrame(rows).sort_values(["action", "factor"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 候选池汇总辅助
# ---------------------------------------------------------------------------

def summarize_candidate_pool_width(dataset: pd.DataFrame) -> pd.DataFrame:
    """汇总候选池宽度：各月各池的原始/通过/有效标签数。"""
    if dataset.empty:
        return pd.DataFrame()
    label_col = LABEL_COL if LABEL_COL in dataset.columns else "label_forward_1m_o2o_return"
    out = (
        dataset.groupby(["signal_date", "candidate_pool_version"], dropna=False)
        .agg(
            raw_universe_width=("symbol", "nunique"),
            candidate_pool_width=("candidate_pool_pass", "sum"),
            label_valid_count=(label_col, lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
    )
    out["candidate_pool_pass_ratio"] = out["candidate_pool_width"] / out["raw_universe_width"].replace(0, np.nan)
    return out


def summarize_candidate_pool_reject_reason(dataset: pd.DataFrame) -> pd.DataFrame:
    """汇总候选池拒绝原因分布。"""
    if "candidate_pool_reject_reason" not in dataset.columns or dataset.empty:
        return pd.DataFrame()
    df = dataset.copy()
    df["candidate_pool_reject_reason"] = df["candidate_pool_reject_reason"].fillna("").astype(str)
    out = (
        df.groupby(["candidate_pool_version", "candidate_pool_reject_reason"], dropna=False, sort=True)
        .agg(rows=("symbol", "size"), symbols=("symbol", "nunique"))
        .reset_index()
    )
    return out.sort_values(["candidate_pool_version", "rows"], ascending=[True, False])
