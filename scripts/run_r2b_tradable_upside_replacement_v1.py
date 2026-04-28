"""R2B tradable upside replacement v1.

This is the first replacement-style R2B experiment after the R0 contract fix
and R1F sanity rerun. It deliberately does not build a standalone upside sleeve
for promotion. The production-shaped candidates keep the S2 defensive Top-20
core, then allow only capped replacements when the lagged market state is
strong-up or breadth is wide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (  # noqa: E402
    attach_universe_filter,
    build_limit_up_open_mask,
    build_market_ew_open_to_open_benchmark,
    build_open_to_open_returns,
    load_config,
    load_daily_from_duckdb,
    transaction_cost_params_from_mapping,
)
from scripts.run_p1_strong_up_attribution import (  # noqa: E402
    REGIME_ORDER,
    _json_sanitize,
    build_groups_per_rebalance,
    build_switch_quality,
    compute_breadth,
    compute_r1_extra_features,
    summarize_breadth_capture,
    summarize_regime_capture,
    summarize_switch_by_regime,
)
from scripts.run_p2_regime_aware_dual_sleeve_v1 import (  # noqa: E402
    _attach_pit_roe_ttm,
    _compute_minimal_defensive_factors,
    _detail_from_weights,
    _filter_universe,
    _lagged_state_by_rebalance,
    _monthly_benchmark_frame,
    _monthly_with_regime,
)
from scripts.run_p2_upside_sleeve_v1 import (  # noqa: E402
    GATE_FULL_BACKTEST,
    GATE_REJECT,
    _add_cs_rank,
    _gate_decision,
)
from src.models.xtree.p1_workflow import (  # noqa: E402
    build_tree_score_weight_matrix,
    summarize_tree_daily_backtest_like_proxy,
)

EVAL_CONTRACT_VERSION = "r0_eval_execution_contract_2026-04-28"
EXECUTION_CONTRACT_VERSION = "tplus1_open_buy_delta_limit_mask_2026-04-28"

BASELINE_ID = "BASELINE_S2_FIXED"

UPSIDE_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "U2_A_industry_breadth_strength",
        "label": "industry breadth + intra-industry strength + amount expansion",
        "score_col": "score__u2_a",
        "standalone_score_col": "score__u2_a",
    },
    {
        "id": "U2_B_tradable_breakout_expansion",
        "label": "tradable breakout + amount/turnover expansion + overheat penalty",
        "score_col": "score__u2_b",
        "standalone_score_col": "score__u2_b",
    },
    {
        "id": "U2_C_s2_residual_elasticity",
        "label": "S2 residual elasticity after size/vol/S2 controls",
        "score_col": "score__u2_c",
        "standalone_score_col": "score__u2_c",
    },
]

REPLACEMENT_RULES: list[dict[str, Any]] = [
    {"id": "R2B_REPLACE_3", "max_replace": 3, "overlay_weight": None},
    {"id": "R2B_REPLACE_5", "max_replace": 5, "overlay_weight": None},
    {"id": "R2B_OVERLAY_10", "max_replace": 5, "overlay_weight": 0.10},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2B tradable upside replacement v1")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default="r2b_tradable_upside_replacement_v1_2026-04-28")
    p.add_argument("--industry-map", default="data/cache/industry_map.csv")
    p.add_argument("--upside-pct", type=float, default=0.90)
    p.add_argument("--score-margin", type=float, default=0.10)
    p.add_argument("--max-industry-names", type=int, default=5)
    p.add_argument("--max-limit-up-hits-20d", type=float, default=2.0)
    p.add_argument("--max-expansion", type=float, default=1.50)
    return p.parse_args()


def _long_with_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "candidate_id", cid)
    return out


PREFIX_FALLBACK_INDUSTRY_SOURCE = "symbol_prefix_proxy_missing_data_cache_industry_map"
DIAGNOSTIC_FALLBACK_SOURCE = "fallback_only_for_diagnostic"


def _industry_source_status(source: str) -> str:
    s = str(source)
    if s == PREFIX_FALLBACK_INDUSTRY_SOURCE:
        return "prefix_fallback_no_alpha_evidence"
    parts = {p.strip() for p in s.split(",") if p.strip()}
    if parts and parts <= {DIAGNOSTIC_FALLBACK_SOURCE}:
        return "fallback_only_for_diagnostic_no_alpha_evidence"
    return "real_industry_map"


def _load_or_build_industry_map(symbols: list[str], path: Path) -> tuple[pd.DataFrame, str]:
    """Load a real industry map if present; otherwise use a stable board proxy."""
    if path.exists():
        raw = pd.read_csv(path, encoding="utf-8-sig")
        if {"symbol", "industry"}.issubset(raw.columns):
            keep_cols = ["symbol", "industry"]
            if "source" in raw.columns:
                keep_cols.append("source")
            if "asof_date" in raw.columns:
                keep_cols.append("asof_date")
            out = raw[keep_cols].copy()
            out["symbol"] = out["symbol"].astype(str).str.zfill(6)
            out["industry"] = out["industry"].astype(str).replace({"": "unknown"}).fillna("unknown")
            if "source" in out.columns:
                source_series = out["source"]
            else:
                source_series = pd.Series(["unknown"] * len(out), index=out.index)
            sources = sorted(set(source_series.astype(str).str.strip()))
            source_label = ",".join(s for s in sources if s) or str(path)
            return out[["symbol", "industry"]].drop_duplicates("symbol", keep="last"), source_label

    rows = []
    for sym in sorted({str(s).zfill(6) for s in symbols}):
        if sym.startswith("688"):
            industry = "proxy_star_688"
        elif sym.startswith("300"):
            industry = "proxy_chinext_300"
        elif sym.startswith(("8", "4")):
            industry = "proxy_bse"
        elif sym.startswith("60"):
            industry = "proxy_sh_main_60"
        elif sym.startswith("00"):
            industry = "proxy_sz_main_00"
        elif sym.startswith("002"):
            industry = "proxy_sz_sme_002"
        else:
            industry = f"proxy_prefix_{sym[:2]}"
        rows.append({"symbol": sym, "industry": industry})
    return pd.DataFrame(rows), PREFIX_FALLBACK_INDUSTRY_SOURCE


def _add_basic_r2b_features(panel: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    d["symbol"] = d["symbol"].astype(str).str.zfill(6)
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d["turnover"] = pd.to_numeric(d["turnover"], errors="coerce")
    d["pct_chg_frac"] = pd.to_numeric(d.get("pct_chg"), errors="coerce") / 100.0
    g = d.groupby("symbol", sort=False)
    d["realized_vol_20d"] = g["pct_chg_frac"].transform(lambda s: s.rolling(20, min_periods=10).std())
    d["rolling_high_60d"] = g["close"].transform(lambda s: s.rolling(60, min_periods=30).max())
    d["breakout_60d"] = d["close"] / d["rolling_high_60d"].replace(0, np.nan) - 1.0
    d["log_market_cap_proxy"] = np.log1p((d["close"] * d["volume"]) / d["turnover"].replace(0, np.nan).abs())
    keep = [
        "symbol",
        "trade_date",
        "realized_vol_20d",
        "breakout_60d",
        "log_market_cap_proxy",
    ]
    return panel.merge(d[keep], on=["symbol", "trade_date"], how="left")


def _residualize_by_date(df: pd.DataFrame, y_col: str, x_cols: list[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    for _, idx in out.groupby("trade_date", sort=False).groups.items():
        sub = out.loc[idx, [y_col, *x_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < len(x_cols) + 20:
            continue
        y = sub[y_col].to_numpy(dtype=float)
        x = sub[x_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - x @ beta
        out.loc[sub.index, out_col] = resid
    return out


def build_r2b_scores(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["industry_positive_20d"] = (pd.to_numeric(df["rel_strength_20d"], errors="coerce") > 0).astype(float)
    df["industry_breadth_20d"] = df.groupby(["trade_date", "industry"])["industry_positive_20d"].transform("mean")
    df["intra_industry_strength"] = df.groupby(["trade_date", "industry"])["rel_strength_20d"].rank(
        pct=True,
        method="average",
    )
    df["overheat_penalty"] = (
        pd.to_numeric(df["limit_up_hits_20d"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["limit_down_hits_20d"], errors="coerce").fillna(0.0)
        + np.maximum(pd.to_numeric(df["turnover_expansion_5_60"], errors="coerce").fillna(0.0) - 1.0, 0.0)
    )
    df = _residualize_by_date(
        df,
        y_col="rel_strength_20d",
        x_cols=["log_market_cap_proxy", "realized_vol_20d", "vol_to_turnover"],
        out_col="s2_residual_elasticity",
    )

    rank_cols = [
        "vol_to_turnover",
        "industry_breadth_20d",
        "intra_industry_strength",
        "amount_expansion_5_60",
        "turnover_expansion_5_60",
        "breakout_60d",
        "tail_strength_20d",
        "overheat_penalty",
        "s2_residual_elasticity",
        "realized_vol_20d",
        "log_market_cap_proxy",
    ]
    for col in rank_cols:
        df = _add_cs_rank(df, col, f"_rk_{col}")

    df["score__defensive"] = df["_rk_vol_to_turnover"]
    df["score__u2_a"] = (
        0.40 * df["_rk_industry_breadth_20d"]
        + 0.35 * df["_rk_intra_industry_strength"]
        + 0.25 * df["_rk_amount_expansion_5_60"]
        - 0.25 * df["_rk_overheat_penalty"]
    )
    df["score__u2_b"] = (
        0.40 * df["_rk_breakout_60d"]
        + 0.25 * df["_rk_turnover_expansion_5_60"]
        + 0.20 * df["_rk_amount_expansion_5_60"]
        + 0.15 * df["_rk_tail_strength_20d"]
        - 0.35 * df["_rk_overheat_penalty"]
    )
    df["score__u2_c"] = (
        0.55 * df["_rk_s2_residual_elasticity"]
        + 0.20 * df["_rk_amount_expansion_5_60"]
        + 0.15 * df["_rk_breakout_60d"]
        - 0.10 * df["_rk_realized_vol_20d"]
        - 0.10 * df["_rk_log_market_cap_proxy"]
        - 0.20 * df["_rk_overheat_penalty"]
    )
    for col in ("score__u2_a", "score__u2_b", "score__u2_c"):
        df = _add_cs_rank(df, col, f"{col}_pct")
    return df


def _next_trading_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(pd.Timestamp(date).normalize(), side="right")
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos]).normalize()


def _state_allows_replacement(state_row: pd.Series | None) -> bool:
    if state_row is None:
        return False
    return str(state_row.get("lagged_regime", "")) == "strong_up" or str(state_row.get("lagged_breadth", "")) == "wide"


def build_replacement_weights(
    *,
    panel: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    state_by_rebalance: pd.DataFrame,
    score_col: str,
    rule: dict[str, Any],
    trading_index: pd.DatetimeIndex,
    limit_up_open_mask: pd.DataFrame,
    upside_pct: float,
    score_margin: float,
    max_industry_names: int,
    max_limit_up_hits_20d: float,
    max_expansion: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_map = state_by_rebalance.set_index("rebalance_date") if not state_by_rebalance.empty else pd.DataFrame()
    rows: list[pd.Series] = []
    diag_rows: list[dict[str, Any]] = []
    panel_by_date = {
        pd.Timestamp(k).normalize(): v.copy()
        for k, v in panel.groupby("trade_date", sort=False)
    }
    for rd, base_w in defensive_weights.sort_index().iterrows():
        rd = pd.Timestamp(rd).normalize()
        state_row = state_map.loc[rd] if rd in state_map.index else None
        base_holdings = [str(s).zfill(6) for s, w in base_w.items() if float(w) > 0.0]
        out_w = base_w.copy()
        replacements: list[tuple[str, str, float]] = []
        blocked_reason = ""
        if _state_allows_replacement(state_row) and rd in panel_by_date:
            day = panel_by_date[rd].copy()
            day["symbol"] = day["symbol"].astype(str).str.zfill(6)
            day = day.dropna(subset=[score_col, "score__defensive"])
            entry_date = _next_trading_date(trading_index, rd)
            if entry_date is None:
                blocked_reason = "no_next_entry_date"
            else:
                day["_is_base"] = day["symbol"].isin(base_holdings)
                pct_col = f"{score_col}_pct"
                if pct_col not in day.columns:
                    day = _add_cs_rank(day, score_col, pct_col)
                candidates = day[~day["_is_base"]].copy()
                if not candidates.empty:
                    lim_row = (
                        limit_up_open_mask.reindex(index=[entry_date], columns=candidates["symbol"], fill_value=False)
                        .fillna(False)
                        .iloc[0]
                    )
                    candidates["_buyable"] = ~candidates["symbol"].map(lim_row.to_dict()).fillna(False).astype(bool)
                    candidates["_passes_risk"] = (
                        candidates["_buyable"]
                        & (pd.to_numeric(candidates[pct_col], errors="coerce") >= float(upside_pct))
                        & (pd.to_numeric(candidates["limit_up_hits_20d"], errors="coerce").fillna(0.0) <= max_limit_up_hits_20d)
                        & (pd.to_numeric(candidates["limit_down_hits_20d"], errors="coerce").fillna(0.0) <= 0.0)
                        & (pd.to_numeric(candidates["amount_expansion_5_60"], errors="coerce").fillna(0.0) <= max_expansion)
                        & (pd.to_numeric(candidates["turnover_expansion_5_60"], errors="coerce").fillna(0.0) <= max_expansion)
                    )
                    candidates = candidates[candidates["_passes_risk"]].sort_values(score_col, ascending=False)
                base_day = day[day["_is_base"]].sort_values("score__defensive", ascending=True)
                industry_counts = day[day["_is_base"]].groupby("industry")["symbol"].count().to_dict()
                slots = int(rule["max_replace"])
                for old in base_day.itertuples(index=False):
                    if len(replacements) >= slots or candidates.empty:
                        break
                    old_sym = str(old.symbol).zfill(6)
                    old_score = float(getattr(old, score_col))
                    viable = candidates[pd.to_numeric(candidates[score_col], errors="coerce") >= old_score + score_margin]
                    chosen = None
                    for cand in viable.itertuples(index=False):
                        industry = str(cand.industry)
                        if int(industry_counts.get(industry, 0)) + 1 <= int(max_industry_names):
                            chosen = cand
                            break
                    if chosen is None:
                        continue
                    new_sym = str(chosen.symbol).zfill(6)
                    new_score = float(getattr(chosen, score_col))
                    replacements.append((old_sym, new_sym, new_score - old_score))
                    industry_counts[str(old.industry)] = max(0, int(industry_counts.get(str(old.industry), 0)) - 1)
                    industry_counts[str(chosen.industry)] = int(industry_counts.get(str(chosen.industry), 0)) + 1
                    candidates = candidates[candidates["symbol"] != new_sym]

                overlay_weight = rule.get("overlay_weight")
                if replacements and overlay_weight is not None:
                    per_new = float(overlay_weight) / float(len(replacements))
                    per_old = float(overlay_weight) / float(len(replacements))
                    for old_sym, new_sym, _ in replacements:
                        out_w.loc[old_sym] = max(float(out_w.get(old_sym, 0.0)) - per_old, 0.0)
                        if new_sym not in out_w.index:
                            out_w.loc[new_sym] = 0.0
                        out_w.loc[new_sym] = float(out_w.get(new_sym, 0.0)) + per_new
                elif replacements:
                    for old_sym, new_sym, _ in replacements:
                        weight = float(out_w.get(old_sym, 0.0))
                        out_w.loc[old_sym] = 0.0
                        if new_sym not in out_w.index:
                            out_w.loc[new_sym] = 0.0
                        out_w.loc[new_sym] = weight
                elif not blocked_reason:
                    blocked_reason = "no_candidate_passed_gate"
        elif state_row is None:
            blocked_reason = "missing_lagged_state"
        else:
            blocked_reason = "state_not_strong_up_or_wide"

        total = float(out_w.sum())
        if total > 0:
            out_w = out_w / total
        out_w.name = rd
        rows.append(out_w)
        diag_rows.append(
            {
                "trade_date": rd,
                "rule_id": rule["id"],
                "score_col": score_col,
                "lagged_regime": "" if state_row is None else str(state_row.get("lagged_regime", "")),
                "lagged_breadth": "" if state_row is None else str(state_row.get("lagged_breadth", "")),
                "replacement_allowed": bool(_state_allows_replacement(state_row)),
                "replacement_count": int(len(replacements)),
                "old_symbols": ",".join(r[0] for r in replacements),
                "new_symbols": ",".join(r[1] for r in replacements),
                "avg_score_margin": float(np.mean([r[2] for r in replacements])) if replacements else np.nan,
                "blocked_reason": blocked_reason,
                "n_names": int((out_w > 0).sum()),
                "max_industry_names": int(max_industry_names),
            }
        )
    weights = pd.DataFrame(rows).fillna(0.0)
    weights.index = pd.to_datetime(weights.index).normalize()
    return weights.sort_index(), pd.DataFrame(diag_rows)


def _score_for_groups(panel: pd.DataFrame, weights: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Use held names first, then upside score, so switch diagnostics track replacement boundary."""
    held_long = (
        weights.stack()
        .rename("weight")
        .reset_index()
        .rename(columns={"level_0": "trade_date", "level_1": "symbol"})
    )
    held_long["trade_date"] = pd.to_datetime(held_long["trade_date"]).dt.normalize()
    held_long["symbol"] = held_long["symbol"].astype(str).str.zfill(6)
    held_long = held_long[held_long["weight"] > 0.0]
    out = panel[["symbol", "trade_date", score_col]].copy().rename(columns={score_col: "score"})
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out = out.merge(held_long[["symbol", "trade_date", "weight"]], on=["symbol", "trade_date"], how="left")
    out["score"] = out["score"] + out["weight"].fillna(0.0) * 10.0
    return out[["symbol", "trade_date", "score"]].dropna(subset=["score"])


def _industry_exposure(weights: pd.DataFrame, industry_map: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    imap = industry_map.set_index("symbol")["industry"].astype(str).to_dict()
    rows = []
    for rd, row in weights.iterrows():
        active = row[row > 0.0]
        for sym, w in active.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "trade_date": pd.Timestamp(rd).normalize(),
                    "industry": imap.get(str(sym).zfill(6), "unknown"),
                    "weight": float(w),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return (
        out.groupby(["candidate_id", "trade_date", "industry"], as_index=False)["weight"]
        .sum()
        .sort_values(["candidate_id", "trade_date", "weight"], ascending=[True, True, False])
    )


def _overlap_vs_baseline(candidate_weights: pd.DataFrame, baseline_weights: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    rows = []
    for rd in candidate_weights.index.intersection(baseline_weights.index):
        cand = set(candidate_weights.loc[rd][candidate_weights.loc[rd] > 0.0].index.astype(str))
        base = set(baseline_weights.loc[rd][baseline_weights.loc[rd] > 0.0].index.astype(str))
        rows.append(
            {
                "candidate_id": candidate_id,
                "trade_date": pd.Timestamp(rd).normalize(),
                "baseline_names": int(len(base)),
                "candidate_names": int(len(cand)),
                "overlap_names": int(len(cand & base)),
                "overlap_share_of_baseline": float(len(cand & base) / len(base)) if base else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run_from_weights(
    *,
    candidate: dict[str, Any],
    score_df: pd.DataFrame,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    breadth_series: pd.Series,
    cost_params: Any,
    limit_up_open_mask: pd.DataFrame,
) -> dict[str, Any]:
    detail, meta = _detail_from_weights(
        weights=weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        cost_params=cost_params,
        scenario=str(candidate["id"]),
        limit_up_open_mask=limit_up_open_mask,
    )
    meta.update(
        {
            "primary_benchmark_return_mode": "open_to_open",
            "comparison_benchmark_return_mode": "close_to_close",
            "eval_contract_version": EVAL_CONTRACT_VERSION,
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        }
    )
    summary = summarize_tree_daily_backtest_like_proxy(detail)
    monthly = _monthly_with_regime(detail, breadth_series)
    regime_capture = summarize_regime_capture(monthly)
    breadth_capture = summarize_breadth_capture(monthly)
    rebalance_actual = sorted(pd.to_datetime(weights.index).normalize().unique().tolist())
    groups = build_groups_per_rebalance(score_df.rename(columns={"candidate_score": "score"}), rebalance_actual)
    switch_df = build_switch_quality(groups, monthly, asset_returns)
    switch_by_regime = summarize_switch_by_regime(switch_df)
    year_rows: list[dict[str, Any]] = []
    for year in (2021, 2025, 2026):
        for regime in REGIME_ORDER:
            part = monthly[(monthly["month_end"].dt.year == year) & (monthly["regime"] == regime)]
            if part.empty:
                continue
            year_rows.append(
                {
                    "year": year,
                    "regime": regime,
                    "months": int(len(part)),
                    "median_excess_return": float(part["excess_return"].median()),
                    "positive_excess_share": float((part["excess_return"] > 0).mean()),
                }
            )
    return {
        "candidate": candidate,
        "summary": summary,
        "meta": meta,
        "monthly": monthly,
        "regime_capture": regime_capture,
        "breadth_capture": breadth_capture,
        "year_capture": pd.DataFrame(year_rows),
        "switch_by_regime": switch_by_regime,
        "switch_detail": switch_df,
    }


def _build_leaderboard(results: list[dict[str, Any]], baseline_id: str) -> pd.DataFrame:
    base = next(r for r in results if r["candidate"]["id"] == baseline_id)

    def _regime_row(rc: pd.DataFrame, regime: str) -> dict[str, float]:
        sub = rc[rc["regime"] == regime] if not rc.empty else pd.DataFrame()
        if sub.empty:
            return {"median_excess": np.nan, "positive_share": np.nan, "capture": np.nan}
        r = sub.iloc[0]
        return {
            "median_excess": float(r["median_excess_return"]),
            "positive_share": float(r["positive_excess_share"]),
            "capture": float(r.get("capture_ratio", np.nan)),
        }

    def _breadth_row(bc: pd.DataFrame, breadth: str) -> dict[str, float]:
        sub = bc[bc["breadth"] == breadth] if not bc.empty else pd.DataFrame()
        if sub.empty:
            return {"median_excess": np.nan, "positive_share": np.nan}
        r = sub.iloc[0]
        return {
            "median_excess": float(r["median_excess_return"]),
            "positive_share": float(r["positive_excess_share"]),
        }

    def _switch_strong_up(sw: pd.DataFrame) -> dict[str, float]:
        sub = sw[sw["regime"] == "strong_up"] if not sw.empty else pd.DataFrame()
        if sub.empty:
            return {"strong_up_switch_in_minus_out": np.nan, "strong_up_topk_minus_next": np.nan}
        r = sub.iloc[0]
        return {
            "strong_up_switch_in_minus_out": float(r["mean_switch_in_minus_out"]),
            "strong_up_topk_minus_next": float(r["mean_topk_minus_next"]),
        }

    base_su = _regime_row(base["regime_capture"], "strong_up")
    base_sd = _regime_row(base["regime_capture"], "strong_down")
    base_wide = _breadth_row(base["breadth_capture"], "wide")
    base_proxy = float(base["summary"].get("annualized_excess_vs_market", np.nan))
    base_turnover = float(base["meta"].get("avg_turnover_half_l1", np.nan))
    rows: list[dict[str, Any]] = []
    for r in results:
        proxy = float(r["summary"].get("annualized_excess_vs_market", np.nan))
        su = _regime_row(r["regime_capture"], "strong_up")
        sd = _regime_row(r["regime_capture"], "strong_down")
        wide = _breadth_row(r["breadth_capture"], "wide")
        sw = _switch_strong_up(r["switch_by_regime"])
        avg_turnover = float(r["meta"].get("avg_turnover_half_l1", np.nan))
        rows.append(
            {
                "candidate_id": r["candidate"]["id"],
                "input_id": r["candidate"].get("input_id", ""),
                "rule_id": r["candidate"].get("rule_id", ""),
                "label": r["candidate"].get("label", ""),
                "daily_proxy_annualized_excess_vs_market": proxy,
                "delta_vs_baseline_proxy": proxy - base_proxy,
                "gate_decision": _gate_decision(proxy),
                "strong_up_median_excess": su["median_excess"],
                "delta_vs_baseline_strong_up_median_excess": su["median_excess"] - base_su["median_excess"],
                "strong_up_positive_share": su["positive_share"],
                "delta_vs_baseline_strong_up_positive_share": su["positive_share"] - base_su["positive_share"],
                "strong_down_median_excess": sd["median_excess"],
                "delta_vs_baseline_strong_down_median_excess": sd["median_excess"] - base_sd["median_excess"],
                "wide_breadth_median_excess": wide["median_excess"],
                "delta_vs_baseline_wide_breadth_median_excess": wide["median_excess"] - base_wide["median_excess"],
                "strong_up_switch_in_minus_out": sw["strong_up_switch_in_minus_out"],
                "strong_up_topk_minus_next": sw["strong_up_topk_minus_next"],
                "avg_turnover_half_l1": avg_turnover,
                "delta_vs_baseline_turnover": avg_turnover - base_turnover,
                "buy_fail_total_weight": float(r["meta"].get("buy_fail_total_weight", 0.0)),
                "n_periods": int(r["summary"].get("n_periods", 0)),
                "n_rebalances": int(r["meta"].get("n_rebalances", 0)),
                "primary_result_type": "daily_bt_like_proxy",
                "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
                "p1_experiment_mode": "daily_proxy_first",
                "legacy_proxy_decision_role": "diagnostic_only",
            }
        )
    out = pd.DataFrame(rows)
    out["_sort"] = out["candidate_id"].apply(lambda x: 0 if x == baseline_id else 1)
    out["_proxy_sort"] = -pd.to_numeric(out["daily_proxy_annualized_excess_vs_market"], errors="coerce")
    return out.sort_values(["_sort", "_proxy_sort"]).drop(columns=["_sort", "_proxy_sort"]).reset_index(drop=True)


def _accept_summary(row: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    if row["candidate_id"] == baseline["candidate_id"]:
        return {"status": "baseline", "checks": {}}
    checks: dict[str, str] = {}
    proxy = float(row["daily_proxy_annualized_excess_vs_market"])
    checks["daily_proxy_not_below_baseline"] = (
        "pass" if np.isfinite(proxy) and proxy >= float(baseline["daily_proxy_annualized_excess_vs_market"]) else "fail"
    )
    checks["daily_proxy_>=_0"] = "pass" if np.isfinite(proxy) and proxy >= 0.0 else "fail"
    su_med_delta = float(row["delta_vs_baseline_strong_up_median_excess"])
    su_pos_delta = float(row["delta_vs_baseline_strong_up_positive_share"])
    sd_med_delta = float(row["delta_vs_baseline_strong_down_median_excess"])
    turn_delta = float(row["delta_vs_baseline_turnover"])
    sw = float(row["strong_up_switch_in_minus_out"])
    checks["strong_up_median_+2pct_or_positive_share_+10pct"] = (
        "pass"
        if (np.isfinite(su_med_delta) and su_med_delta >= 0.02)
        or (np.isfinite(su_pos_delta) and su_pos_delta >= 0.10)
        else "fail"
    )
    checks["strong_down_not_worse_than_-2pct"] = "pass" if np.isfinite(sd_med_delta) and sd_med_delta >= -0.02 else "fail"
    checks["turnover_delta_<=_+0.10"] = "pass" if np.isfinite(turn_delta) and turn_delta <= 0.10 else "fail"
    checks["strong_up_switch_not_negative"] = "pass" if np.isfinite(sw) and sw >= 0.0 else "fail"
    status = "pass" if all(v == "pass" for v in checks.values()) else "fail"
    if status == "fail" and checks["daily_proxy_not_below_baseline"] == "pass" and (
        checks["strong_up_median_+2pct_or_positive_share_+10pct"] == "pass"
        or checks["strong_up_switch_not_negative"] == "pass"
    ):
        status = "gray_zone"
    return {"status": status, "checks": checks}


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    leaderboard: pd.DataFrame,
    accept_map: dict[str, dict[str, Any]],
    replacement_diag: pd.DataFrame,
    regime_long: pd.DataFrame,
    breadth_long: pd.DataFrame,
    switch_long: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2B Tradable Upside Replacement v1\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append(f"- `eval_contract_version`: `{EVAL_CONTRACT_VERSION}`")
    lines.append(f"- `execution_contract_version`: `{EXECUTION_CONTRACT_VERSION}`")
    lines.append(f"- `industry_map_source`: `{params.get('industry_map_source', '')}`")
    lines.append(f"- `industry_map_source_status`: `{params.get('industry_map_source_status', '')}`")
    lines.append("- 组合表达：`S2 defensive Top-20 + capped replacement slots`，不再使用完整 upside Top-20 sleeve。")
    lines.append("- 状态输入：上一已完成月份 `regime/breadth`；仅 `strong_up` 或 `wide` 允许 replacement gate。")
    lines.append("- primary benchmark：`open_to_open`；promotion metric：`daily_bt_like_proxy_annualized_excess_vs_market`。")
    lines.append("")
    lines.append("## 1. Leaderboard\n")
    cols = [
        "candidate_id",
        "daily_proxy_annualized_excess_vs_market",
        "delta_vs_baseline_proxy",
        "gate_decision",
        "strong_up_median_excess",
        "delta_vs_baseline_strong_up_median_excess",
        "strong_up_positive_share",
        "delta_vs_baseline_strong_up_positive_share",
        "strong_down_median_excess",
        "delta_vs_baseline_strong_down_median_excess",
        "strong_up_switch_in_minus_out",
        "avg_turnover_half_l1",
        "delta_vs_baseline_turnover",
        "industry_map_source_status",
        "industry_alpha_evidence_allowed",
        "n_rebalances",
    ]
    lines.append(leaderboard[[c for c in cols if c in leaderboard.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 2. R2B 验收\n")
    accept_rows = []
    for cid, info in accept_map.items():
        row = {"candidate_id": cid, "status": info["status"]}
        row.update(info.get("checks", {}))
        accept_rows.append(row)
    lines.append(pd.DataFrame(accept_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Replacement 触发统计\n")
    if replacement_diag.empty:
        lines.append("_无 replacement 诊断_")
    else:
        trigger = (
            replacement_diag.groupby(["candidate_id", "replacement_allowed"], dropna=False)
            .agg(
                rebalances=("trade_date", "count"),
                avg_replacement_count=("replacement_count", "mean"),
                max_replacement_count=("replacement_count", "max"),
                active_rebalance_share=("replacement_count", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        lines.append(trigger.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 4. Regime / Breadth 切片\n")
    lines.append(regime_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(breadth_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 5. Switch quality\n")
    lines.append("_无 switch 样本_" if switch_long.empty else switch_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 6. 行业暴露\n")
    if params.get("industry_map_source_status") != "real_industry_map":
        lines.append(
            f"> 行业来源为 `{params.get('industry_map_source_status')}`；"
            "industry breadth / 行业暴露仅可作诊断，不可作为 alpha 证据。"
        )
        lines.append("")
    if industry_exposure.empty:
        lines.append("_无行业暴露_")
    else:
        latest = industry_exposure.sort_values("trade_date").groupby("candidate_id").tail(5)
        lines.append(latest.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 7. 结论\n")
    non_base = leaderboard[leaderboard["candidate_id"] != BASELINE_ID]
    full_bt = non_base[non_base["daily_proxy_annualized_excess_vs_market"] >= GATE_FULL_BACKTEST]["candidate_id"].tolist()
    gray = [cid for cid, info in accept_map.items() if info["status"] == "gray_zone"]
    passed = [cid for cid, info in accept_map.items() if info["status"] == "pass"]
    if passed:
        lines.append(f"- R2B 通过候选：{', '.join(passed)}。")
    elif gray:
        lines.append(f"- R2B 进入 gray zone 候选：{', '.join(gray)}；可作为 R3 规则版 boundary gate 的输入。")
    else:
        lines.append("- R2B 第一轮无候选通过，也无 gray zone。规则版 R3 不应启动。")
    if full_bt:
        lines.append(f"- daily proxy `>= +3%` 候选：{', '.join(full_bt)}，允许补正式 full backtest。")
    else:
        lines.append("- 无候选达到 daily proxy `>= +3%`，不补正式 full backtest。")
    lines.append("")
    lines.append("## 8. 产出文件\n")
    for suf in [
        "leaderboard.csv",
        "replacement_diag_long.csv",
        "standalone_leaderboard.csv",
        "overlap_long.csv",
        "industry_exposure_long.csv",
        "regime_long.csv",
        "breadth_long.csv",
        "year_long.csv",
        "switch_long.csv",
        "monthly_long.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suf}`")
    lines.append("")
    lines.append("## 9. 参数\n")
    for k, v in params.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg, config_source = load_config(args.config)
    paths_cfg = cfg.get("paths", {}) or {}
    db_path_raw = paths_cfg.get("duckdb_path") or paths_cfg.get("database_path") or "data/market.duckdb"
    db_path = str(db_path_raw if Path(db_path_raw).is_absolute() else PROJECT_ROOT / db_path_raw)
    end_date = args.end or str(paths_cfg.get("asof_trade_date") or pd.Timestamp.today().strftime("%Y-%m-%d"))
    backtest_cfg = cfg.get("backtest", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    signals_cfg = cfg.get("signals", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    uf_cfg = cfg.get("universe_filter", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    top_k = int(signals_cfg.get("top_k", 20))
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    max_turnover = float(portfolio_cfg.get("max_turnover", 1.0))

    print(f"[1/8] load daily {args.start}->{end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)

    print("[2/8] compute S2 + R2B features + universe filter", flush=True)
    factors = _compute_minimal_defensive_factors(daily_df, min_hist_days=args.min_hist_days)
    factors = _attach_pit_roe_ttm(factors, db_path)
    factors = attach_universe_filter(
        factors,
        daily_df,
        enabled=bool(uf_cfg.get("enabled", False)),
        min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
        require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
    )
    extras = compute_r1_extra_features(daily_df)
    panel = factors.merge(extras, on=["symbol", "trade_date"], how="left")
    panel = _add_basic_r2b_features(panel, daily_df)
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start)].copy()
    panel = _filter_universe(panel)
    industry_map, industry_source = _load_or_build_industry_map(
        sorted(panel["symbol"].astype(str).str.zfill(6).unique().tolist()),
        PROJECT_ROOT / args.industry_map,
    )
    industry_status = _industry_source_status(industry_source)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["industry"] = panel["industry"].fillna("unknown").astype(str)
    panel = build_r2b_scores(panel)
    print(
        f"  panel(after filter)={panel.shape}; industry_source={industry_source}; industry_status={industry_status}",
        flush=True,
    )

    print("[3/8] precompute returns + benchmark + lagged state", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    limit_up_open_mask = build_limit_up_open_mask(daily_df).sort_index()
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_state = _monthly_benchmark_frame(bench_daily, breadth_series)

    print("[4/8] build baseline and standalone weights", flush=True)
    score_base = panel[["symbol", "trade_date", "score__defensive"]].dropna().copy()
    defensive_weights, defensive_diag = build_tree_score_weight_matrix(
        score_base,
        score_col="score__defensive",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    state_by_rebalance = _lagged_state_by_rebalance(defensive_weights.index, monthly_state)
    standalone_weights: dict[str, pd.DataFrame] = {}
    standalone_diag_rows = []
    for cand in UPSIDE_CANDIDATES:
        w, diag = build_tree_score_weight_matrix(
            panel[["symbol", "trade_date", cand["standalone_score_col"]]].dropna(),
            score_col=cand["standalone_score_col"],
            rebalance_rule=rebalance_rule,
            top_k=top_k,
            max_turnover=max_turnover,
        )
        standalone_weights[cand["id"]] = w
        diag["input_id"] = cand["id"]
        standalone_diag_rows.append(diag)

    print("[5/8] build replacement weights", flush=True)
    replacement_weights: dict[str, pd.DataFrame] = {}
    replacement_diag_frames = []
    for cand in UPSIDE_CANDIDATES:
        for rule in REPLACEMENT_RULES:
            cid = f"{cand['id']}__{rule['id']}"
            w, diag = build_replacement_weights(
                panel=panel,
                defensive_weights=defensive_weights,
                state_by_rebalance=state_by_rebalance,
                score_col=cand["score_col"],
                rule=rule,
                trading_index=open_returns.index,
                limit_up_open_mask=limit_up_open_mask,
                upside_pct=args.upside_pct,
                score_margin=args.score_margin,
                max_industry_names=args.max_industry_names,
                max_limit_up_hits_20d=args.max_limit_up_hits_20d,
                max_expansion=args.max_expansion,
            )
            replacement_weights[cid] = w
            diag.insert(0, "candidate_id", cid)
            diag.insert(1, "input_id", cand["id"])
            replacement_diag_frames.append(diag)

    print("[6/8] run daily proxy", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    limit_mask = limit_up_open_mask.reindex(columns=sym_universe, fill_value=False)

    results = [
        _run_from_weights(
            candidate={"id": BASELINE_ID, "label": "fixed S2 defensive Top-20 baseline"},
            score_df=score_base.rename(columns={"score__defensive": "candidate_score"}),
            weights=defensive_weights,
            asset_returns=asset_returns,
            bench_daily=bench_daily,
            breadth_series=breadth_series,
            cost_params=cost_params,
            limit_up_open_mask=limit_mask,
        )
    ]
    standalone_results = []
    for cand in UPSIDE_CANDIDATES:
        print(f"  standalone -> {cand['id']}", flush=True)
        standalone_results.append(
            _run_from_weights(
                candidate={"id": f"{cand['id']}__STANDALONE_TOP20", "label": cand["label"], "input_id": cand["id"]},
                score_df=panel[["symbol", "trade_date", cand["score_col"]]].rename(
                    columns={cand["score_col"]: "candidate_score"}
                ),
                weights=standalone_weights[cand["id"]],
                asset_returns=asset_returns,
                bench_daily=bench_daily,
                breadth_series=breadth_series,
                cost_params=cost_params,
                limit_up_open_mask=limit_mask,
            )
        )
    for cand in UPSIDE_CANDIDATES:
        for rule in REPLACEMENT_RULES:
            cid = f"{cand['id']}__{rule['id']}"
            print(f"  replacement -> {cid}", flush=True)
            results.append(
                _run_from_weights(
                    candidate={
                        "id": cid,
                        "label": f"{cand['label']} / {rule['id']}",
                        "input_id": cand["id"],
                        "rule_id": rule["id"],
                    },
                    score_df=_score_for_groups(panel, replacement_weights[cid], cand["score_col"]).rename(
                        columns={"score": "candidate_score"}
                    ),
                    weights=replacement_weights[cid],
                    asset_returns=asset_returns,
                    bench_daily=bench_daily,
                    breadth_series=breadth_series,
                    cost_params=cost_params,
                    limit_up_open_mask=limit_mask,
                )
            )

    print("[7/8] aggregate diagnostics", flush=True)
    leaderboard = _build_leaderboard(results, baseline_id=BASELINE_ID)
    standalone_leaderboard = _build_leaderboard(
        [
            results[0],
            *standalone_results,
        ],
        baseline_id=BASELINE_ID,
    )
    for table in (leaderboard, standalone_leaderboard):
        table["industry_map_source"] = industry_source
        table["industry_map_source_status"] = industry_status
        table["industry_alpha_evidence_allowed"] = industry_status == "real_industry_map"
    base_row = leaderboard[leaderboard["candidate_id"] == BASELINE_ID].iloc[0]
    accept_map = {row["candidate_id"]: _accept_summary(row, base_row) for _, row in leaderboard.iterrows()}
    regime_long = pd.concat([_long_with_id(r["regime_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    breadth_long = pd.concat([_long_with_id(r["breadth_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    year_long = pd.concat([_long_with_id(r["year_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    switch_long = pd.concat([_long_with_id(r["switch_by_regime"], r["candidate"]["id"]) for r in results], ignore_index=True)
    monthly_long = pd.concat([_long_with_id(r["monthly"], r["candidate"]["id"]) for r in results], ignore_index=True)
    replacement_diag = pd.concat(replacement_diag_frames, ignore_index=True) if replacement_diag_frames else pd.DataFrame()
    standalone_diag = pd.concat(standalone_diag_rows, ignore_index=True) if standalone_diag_rows else pd.DataFrame()
    overlap_long = pd.concat(
        [_overlap_vs_baseline(w, defensive_weights, cid) for cid, w in replacement_weights.items()],
        ignore_index=True,
    )
    industry_exposure = pd.concat(
        [_industry_exposure(defensive_weights, industry_map, BASELINE_ID)]
        + [_industry_exposure(w, industry_map, cid) for cid, w in replacement_weights.items()],
        ignore_index=True,
    )

    print("[8/8] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    leaderboard.to_csv(results_dir / f"{prefix}_leaderboard.csv", index=False, encoding="utf-8-sig")
    standalone_leaderboard.to_csv(results_dir / f"{prefix}_standalone_leaderboard.csv", index=False, encoding="utf-8-sig")
    replacement_diag.to_csv(results_dir / f"{prefix}_replacement_diag_long.csv", index=False, encoding="utf-8-sig")
    standalone_diag.to_csv(results_dir / f"{prefix}_standalone_diag_long.csv", index=False, encoding="utf-8-sig")
    overlap_long.to_csv(results_dir / f"{prefix}_overlap_long.csv", index=False, encoding="utf-8-sig")
    industry_exposure.to_csv(results_dir / f"{prefix}_industry_exposure_long.csv", index=False, encoding="utf-8-sig")
    regime_long.to_csv(results_dir / f"{prefix}_regime_long.csv", index=False, encoding="utf-8-sig")
    breadth_long.to_csv(results_dir / f"{prefix}_breadth_long.csv", index=False, encoding="utf-8-sig")
    year_long.to_csv(results_dir / f"{prefix}_year_long.csv", index=False, encoding="utf-8-sig")
    switch_long.to_csv(results_dir / f"{prefix}_switch_long.csv", index=False, encoding="utf-8-sig")
    monthly_long.to_csv(results_dir / f"{prefix}_monthly_long.csv", index=False, encoding="utf-8-sig")
    state_by_rebalance.to_csv(results_dir / f"{prefix}_lagged_state_by_rebalance.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "defensive_core_capped_replacement",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "state_lag": "previous_completed_month",
        "state_threshold_mode": "expanding",
        "replacement_state_gate": "lagged_regime==strong_up or lagged_breadth==wide",
        "upside_pct": args.upside_pct,
        "score_margin": args.score_margin,
        "max_industry_names": args.max_industry_names,
        "max_limit_up_hits_20d": args.max_limit_up_hits_20d,
        "max_expansion": args.max_expansion,
        "industry_map_source": industry_source,
        "industry_map_source_status": industry_status,
        "industry_alpha_evidence_allowed": industry_status == "real_industry_map",
        "prefilter": prefilter_cfg,
        "universe_filter": uf_cfg,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
        "benchmark_min_history_days": bench_min,
        "primary_benchmark_return_mode": "open_to_open",
        "comparison_benchmark_return_mode": "close_to_close",
        "config_source": config_source,
        "eval_contract_version": EVAL_CONTRACT_VERSION,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "p1_experiment_mode": "daily_proxy_first",
        "legacy_proxy_decision_role": "diagnostic_only",
        "primary_decision_metric": "daily_bt_like_proxy_annualized_excess_vs_market",
        "gate_thresholds": {"reject": GATE_REJECT, "full_backtest": GATE_FULL_BACKTEST},
        "defensive_weight_diag_rows": int(len(defensive_diag)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "upside_candidates": UPSIDE_CANDIDATES,
        "replacement_rules": REPLACEMENT_RULES,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "standalone_leaderboard": standalone_leaderboard.to_dict(orient="records"),
        "accept": accept_map,
        "monthly_state_thresholds": {
            "regime": monthly_state.attrs.get("regime_thresholds"),
            "breadth": monthly_state.attrs.get("breadth_thresholds"),
            "trace": monthly_state.attrs.get("threshold_trace"),
        },
        "candidates": [
            {
                "id": r["candidate"]["id"],
                "label": r["candidate"].get("label", ""),
                "summary": r["summary"],
                "meta": r["meta"],
            }
            for r in results
        ],
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        config_source=config_source,
        params=params,
        leaderboard=leaderboard,
        accept_map=accept_map,
        replacement_diag=replacement_diag,
        regime_long=regime_long,
        breadth_long=breadth_long,
        switch_long=switch_long,
        industry_exposure=industry_exposure,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc_text, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)


if __name__ == "__main__":
    main()
