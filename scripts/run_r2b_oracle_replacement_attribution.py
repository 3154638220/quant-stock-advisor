"""R2B oracle replacement attribution.

This script asks a deliberately narrow question before any new boundary model is
trained: if future open-to-open returns were known, is there enough replacement
edge around the S2 Top-20 boundary to justify continuing R2B?
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
    _json_sanitize,
    compute_breadth,
    compute_r1_extra_features,
)
from scripts.run_p2_regime_aware_dual_sleeve_v1 import (  # noqa: E402
    _attach_pit_roe_ttm,
    _compute_minimal_defensive_factors,
    _detail_from_weights,
    _filter_universe,
    _lagged_state_by_rebalance,
    _monthly_benchmark_frame,
)
from scripts.run_r2b_tradable_upside_replacement_v1 import (  # noqa: E402
    BASELINE_ID,
    EVAL_CONTRACT_VERSION,
    EXECUTION_CONTRACT_VERSION,
    UPSIDE_CANDIDATES,
    _industry_source_status,
    _load_or_build_industry_map,
    _add_basic_r2b_features,
    build_r2b_scores,
)
from src.models.xtree.p1_workflow import (  # noqa: E402
    build_tree_score_weight_matrix,
    summarize_tree_daily_backtest_like_proxy,
)
from src.models.experiment import append_experiment_result  # noqa: E402
from src.models.research_contract import (  # noqa: E402
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from scripts.research_identity import make_research_identity, slugify_token  # noqa: E402


DEFAULT_OUTPUT_PREFIX = "r2b_oracle_replacement_attribution_2026-04-28"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2B oracle replacement attribution")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--industry-map", default="data/cache/industry_map.csv")
    p.add_argument("--horizons", default="5,10,20")
    p.add_argument("--cost-buffers", default="0,0.0015,0.003")
    p.add_argument("--default-horizon", type=int, default=20)
    p.add_argument("--default-cost-buffer", type=float, default=0.0015)
    p.add_argument("--candidate-pcts", default="0.90,0.95")
    p.add_argument("--old-pool-sizes", default="3,5")
    p.add_argument("--max-oracle-replace", type=int, default=3)
    p.add_argument("--max-limit-up-hits-20d", type=float, default=2.0)
    p.add_argument("--max-expansion", type=float, default=1.50)
    p.add_argument(
        "--max-candidates-per-pool",
        type=int,
        default=0,
        help="Optional cap after sorting by upside score; 0 keeps the full pool.",
    )
    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("整数列表不能为空")
    return out


def _parse_float_list(s: str) -> list[float]:
    out = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("浮点列表不能为空")
    return out


def _stack_wide(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    try:
        s = wide.stack(future_stack=True)
    except TypeError:
        s = wide.stack()
    out = s.rename(value_name).reset_index()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out[np.isfinite(out[value_name])].copy()


def build_forward_returns_from_open(daily: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Return open(t+1+h) / open(t+1) - 1 for each signal date t."""
    if horizon < 1:
        raise ValueError("horizon 须 >= 1")
    df = daily[["symbol", "trade_date", "open"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["trade_date", "open"]).sort_values(["trade_date", "symbol"])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", f"forward_ret_{horizon}d"])
    wide = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    fwd = wide.shift(-(1 + int(horizon))) / wide.shift(-1) - 1.0
    return _stack_wide(fwd, f"forward_ret_{horizon}d")


def _next_trading_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(pd.Timestamp(date).normalize(), side="right")
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos]).normalize()


def _state_flags(state_row: pd.Series | None) -> dict[str, Any]:
    regime = "" if state_row is None else str(state_row.get("lagged_regime", ""))
    breadth = "" if state_row is None else str(state_row.get("lagged_breadth", ""))
    return {
        "lagged_regime": regime,
        "lagged_breadth": breadth,
        "state_strong_up_or_wide": bool(regime == "strong_up" or breadth == "wide"),
        "state_strong_up_and_wide": bool(regime == "strong_up" and breadth == "wide"),
        "state_up_or_wide_not_strong_down": bool((regime == "strong_up" or breadth == "wide") and regime != "strong_down"),
    }


def _candidate_pool_id(pct: float) -> str:
    return f"candidate_top_pct_{int(round(float(pct) * 100))}"


def build_oracle_pair_base(
    *,
    panel: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    state_by_rebalance: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    limit_up_open_mask: pd.DataFrame,
    forward_by_horizon: dict[int, pd.DataFrame],
    score_cols: list[str],
    old_pool_sizes: list[int],
    candidate_pcts: list[float],
    max_limit_up_hits_20d: float,
    max_expansion: float,
    max_candidates_per_pool: int = 0,
) -> pd.DataFrame:
    """Build pair candidates with forward returns, before horizon/cost labels."""
    if defensive_weights.empty:
        return pd.DataFrame()
    state_map = state_by_rebalance.set_index("rebalance_date") if not state_by_rebalance.empty else pd.DataFrame()
    fwd = None
    for horizon, df in forward_by_horizon.items():
        col = f"forward_ret_{int(horizon)}d"
        keep = df[["symbol", "trade_date", col]].copy()
        fwd = keep if fwd is None else fwd.merge(keep, on=["symbol", "trade_date"], how="outer")
    if fwd is None:
        return pd.DataFrame()

    feature_cols = [
        "symbol",
        "trade_date",
        "industry",
        "score__defensive",
        "rel_strength_20d",
        "amount_expansion_5_60",
        "turnover_expansion_5_60",
        "limit_up_hits_20d",
        "limit_down_hits_20d",
        *score_cols,
        *[f"{c}_pct" for c in score_cols],
    ]
    feature_cols = [c for c in feature_cols if c in panel.columns]
    px = panel[feature_cols].merge(fwd, on=["symbol", "trade_date"], how="left")
    px["symbol"] = px["symbol"].astype(str).str.zfill(6)
    px["trade_date"] = pd.to_datetime(px["trade_date"], errors="coerce").dt.normalize()
    panel_by_date = {pd.Timestamp(k).normalize(): v.copy() for k, v in px.groupby("trade_date", sort=False)}

    rows: list[pd.DataFrame] = []
    for rd, base_w in defensive_weights.sort_index().iterrows():
        rd = pd.Timestamp(rd).normalize()
        if rd not in panel_by_date:
            continue
        entry_date = _next_trading_date(trading_index, rd)
        if entry_date is None:
            continue
        day = panel_by_date[rd].copy()
        base_holdings = [str(s).zfill(6) for s, w in base_w.items() if float(w) > 0.0]
        if not base_holdings:
            continue
        day["_is_base"] = day["symbol"].isin(base_holdings)
        base_day = day[day["_is_base"]].dropna(subset=["score__defensive"]).sort_values("score__defensive")
        if base_day.empty:
            continue
        state_row = state_map.loc[rd] if rd in state_map.index else None
        state = _state_flags(state_row)
        non_base = day[~day["_is_base"]].copy()
        if non_base.empty:
            continue
        lim_row = (
            limit_up_open_mask.reindex(index=[entry_date], columns=non_base["symbol"], fill_value=False)
            .fillna(False)
            .iloc[0]
        )
        non_base["_buyable"] = ~non_base["symbol"].map(lim_row.to_dict()).fillna(False).astype(bool)
        non_base["_passes_risk"] = (
            non_base["_buyable"]
            & (pd.to_numeric(non_base["limit_up_hits_20d"], errors="coerce").fillna(0.0) <= max_limit_up_hits_20d)
            & (pd.to_numeric(non_base["limit_down_hits_20d"], errors="coerce").fillna(0.0) <= 0.0)
            & (pd.to_numeric(non_base["amount_expansion_5_60"], errors="coerce").fillna(0.0) <= max_expansion)
            & (pd.to_numeric(non_base["turnover_expansion_5_60"], errors="coerce").fillna(0.0) <= max_expansion)
        )
        non_base = non_base[non_base["_passes_risk"]].copy()
        if non_base.empty:
            continue

        old_pools = {
            f"S2_bottom_{int(n)}": base_day.head(int(n)).copy()
            for n in old_pool_sizes
            if int(n) > 0
        }
        for score_col in score_cols:
            pct_col = f"{score_col}_pct"
            if score_col not in non_base.columns:
                continue
            score_candidates = non_base.dropna(subset=[score_col]).copy()
            if score_candidates.empty:
                continue
            candidate_pools: dict[str, pd.DataFrame] = {"candidate_buyable": score_candidates.copy()}
            for pct in candidate_pcts:
                if pct_col in score_candidates.columns:
                    candidate_pools[_candidate_pool_id(pct)] = score_candidates[
                        pd.to_numeric(score_candidates[pct_col], errors="coerce") >= float(pct)
                    ].copy()
            for pool_id, cand in candidate_pools.items():
                cand = cand.sort_values(score_col, ascending=False)
                if max_candidates_per_pool > 0:
                    cand = cand.head(int(max_candidates_per_pool)).copy()
                if cand.empty:
                    continue
                for old_pool_id, old in old_pools.items():
                    old = old.dropna(subset=[score_col, "score__defensive"]).copy()
                    if old.empty:
                        continue
                    merged = old.merge(cand, how="cross", suffixes=("_old", "_new"))
                    out = pd.DataFrame(
                        {
                            "trade_date": rd,
                            "entry_date": entry_date,
                            "score_col": score_col,
                            "old_pool": old_pool_id,
                            "candidate_pool": pool_id,
                            "old_symbol": merged["symbol_old"].astype(str).str.zfill(6),
                            "new_symbol": merged["symbol_new"].astype(str).str.zfill(6),
                            "old_industry": merged.get("industry_old", "unknown"),
                            "new_industry": merged.get("industry_new", "unknown"),
                            "old_weight": merged["symbol_old"].map(base_w.to_dict()).astype(float),
                            "old_defensive_score": pd.to_numeric(merged["score__defensive_old"], errors="coerce"),
                            "old_score": pd.to_numeric(merged[f"{score_col}_old"], errors="coerce"),
                            "candidate_score": pd.to_numeric(merged[f"{score_col}_new"], errors="coerce"),
                            "candidate_score_pct": pd.to_numeric(merged.get(f"{pct_col}_new"), errors="coerce"),
                            "score_margin": pd.to_numeric(merged[f"{score_col}_new"], errors="coerce")
                            - pd.to_numeric(merged[f"{score_col}_old"], errors="coerce"),
                            "rel_strength_diff": pd.to_numeric(merged.get("rel_strength_20d_new"), errors="coerce")
                            - pd.to_numeric(merged.get("rel_strength_20d_old"), errors="coerce"),
                            "amount_expansion_diff": pd.to_numeric(merged.get("amount_expansion_5_60_new"), errors="coerce")
                            - pd.to_numeric(merged.get("amount_expansion_5_60_old"), errors="coerce"),
                            "turnover_expansion_diff": pd.to_numeric(merged.get("turnover_expansion_5_60_new"), errors="coerce")
                            - pd.to_numeric(merged.get("turnover_expansion_5_60_old"), errors="coerce"),
                            "overheat_diff": (
                                pd.to_numeric(merged.get("limit_up_hits_20d_new"), errors="coerce").fillna(0.0)
                                + pd.to_numeric(merged.get("limit_down_hits_20d_new"), errors="coerce").fillna(0.0)
                            )
                            - (
                                pd.to_numeric(merged.get("limit_up_hits_20d_old"), errors="coerce").fillna(0.0)
                                + pd.to_numeric(merged.get("limit_down_hits_20d_old"), errors="coerce").fillna(0.0)
                            ),
                            **state,
                        }
                    )
                    for horizon in forward_by_horizon:
                        col = f"forward_ret_{int(horizon)}d"
                        out[f"old_forward_{int(horizon)}d"] = pd.to_numeric(merged.get(f"{col}_old"), errors="coerce")
                        out[f"new_forward_{int(horizon)}d"] = pd.to_numeric(merged.get(f"{col}_new"), errors="coerce")
                        out[f"raw_pair_edge_{int(horizon)}d"] = (
                            out[f"new_forward_{int(horizon)}d"] - out[f"old_forward_{int(horizon)}d"]
                        )
                    rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _edge_col(horizon: int, cost_buffer: float) -> str:
    bp = int(round(float(cost_buffer) * 10000))
    return f"pair_edge_{int(horizon)}d_cost_{bp}bp"


def _with_edge(base: pd.DataFrame, *, horizon: int, cost_buffer: float) -> pd.DataFrame:
    out = base.copy()
    col = _edge_col(horizon, cost_buffer)
    out[col] = pd.to_numeric(out[f"raw_pair_edge_{int(horizon)}d"], errors="coerce") - float(cost_buffer)
    out["replace_win"] = out[col] > 0.0
    out["pair_edge"] = out[col]
    out["horizon"] = int(horizon)
    out["cost_buffer"] = float(cost_buffer)
    return out


def select_oracle_replacements(
    pairs: pd.DataFrame,
    *,
    edge_col: str,
    max_replace: int,
    state_gate: str | None = None,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    df = pairs.dropna(subset=[edge_col]).copy()
    df = df[df[edge_col] > 0.0].copy()
    if state_gate:
        df = df[df[state_gate].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for rd, sub in df.sort_values(edge_col, ascending=False).groupby("trade_date", sort=False):
        used_old: set[str] = set()
        used_new: set[str] = set()
        selected = []
        for row in sub.itertuples(index=False):
            old_sym = str(row.old_symbol).zfill(6)
            new_sym = str(row.new_symbol).zfill(6)
            if old_sym in used_old or new_sym in used_new:
                continue
            selected.append(row._asdict())
            used_old.add(old_sym)
            used_new.add(new_sym)
            if len(selected) >= int(max_replace):
                break
        if selected:
            rows.extend(selected)
    return pd.DataFrame(rows)


def build_oracle_weights(
    defensive_weights: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> pd.DataFrame:
    out = defensive_weights.copy()
    if selected_pairs.empty:
        return out
    for rd, sub in selected_pairs.groupby("trade_date", sort=False):
        rd = pd.Timestamp(rd).normalize()
        if rd not in out.index:
            continue
        for row in sub.itertuples(index=False):
            old_sym = str(row.old_symbol).zfill(6)
            new_sym = str(row.new_symbol).zfill(6)
            weight = float(out.loc[rd, old_sym]) if old_sym in out.columns else 0.0
            if weight <= 0:
                continue
            out.loc[rd, old_sym] = 0.0
            if new_sym not in out.columns:
                out[new_sym] = 0.0
            out.loc[rd, new_sym] = float(out.loc[rd, new_sym]) + weight
    out = out.fillna(0.0)
    totals = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(totals, axis=0).fillna(0.0).sort_index()


def _summarize_hit_rate(base_pairs: pd.DataFrame, horizons: list[int], cost_buffers: list[float]) -> pd.DataFrame:
    dims = [
        "score_col",
        "old_pool",
        "candidate_pool",
        "horizon",
        "cost_buffer",
        "lagged_regime",
        "lagged_breadth",
        "state_strong_up_or_wide",
        "state_strong_up_and_wide",
        "state_up_or_wide_not_strong_down",
    ]
    rows = []
    for horizon in horizons:
        for cost in cost_buffers:
            df = _with_edge(base_pairs, horizon=horizon, cost_buffer=cost).dropna(subset=["pair_edge"])
            if df.empty:
                continue
            agg = (
                df.groupby(dims, dropna=False)
                .agg(
                    pair_count=("pair_edge", "size"),
                    oracle_hit_rate=("replace_win", "mean"),
                    mean_pair_edge=("pair_edge", "mean"),
                    median_pair_edge=("pair_edge", "median"),
                    p75_pair_edge=("pair_edge", lambda s: float(np.nanquantile(s, 0.75))),
                )
                .reset_index()
            )
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=dims)


def _summarize_capacity(
    base_pairs: pd.DataFrame,
    *,
    horizons: list[int],
    cost_buffers: list[float],
    max_replace: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_rows = []
    summary_rows = []
    dims = ["score_col", "old_pool", "candidate_pool"]
    gates = {
        "all_states": None,
        "strong_up_or_wide": "state_strong_up_or_wide",
        "strong_up_and_wide": "state_strong_up_and_wide",
    }
    for horizon in horizons:
        for cost in cost_buffers:
            edge_col = _edge_col(horizon, cost)
            df = _with_edge(base_pairs, horizon=horizon, cost_buffer=cost).dropna(subset=[edge_col])
            if df.empty:
                continue
            state_cols = ["lagged_regime", "lagged_breadth", *[c for c in gates.values() if c]]
            month_state = df[[*dims, "trade_date", *state_cols]].drop_duplicates([*dims, "trade_date"])

            positive = df[df[edge_col] > 0.0][
                [*dims, "trade_date", "old_symbol", "new_symbol", edge_col, *[c for c in gates.values() if c]]
            ].copy()

            for gate_name, gate_col in gates.items():
                gate_positive = positive if gate_col is None else positive[positive[gate_col].astype(bool)]
                selected_rows = []
                if not gate_positive.empty:
                    for key, sub in gate_positive.groupby([*dims, "trade_date"], sort=False, dropna=False):
                        used_old: set[str] = set()
                        used_new: set[str] = set()
                        chosen_edges = []
                        for row in sub.sort_values(edge_col, ascending=False).itertuples(index=False):
                            old_sym = str(row.old_symbol).zfill(6)
                            new_sym = str(row.new_symbol).zfill(6)
                            if old_sym in used_old or new_sym in used_new:
                                continue
                            chosen_edges.append(float(getattr(row, edge_col)))
                            used_old.add(old_sym)
                            used_new.add(new_sym)
                            if len(chosen_edges) >= int(max_replace):
                                break
                        if chosen_edges:
                            key_dict = dict(zip([*dims, "trade_date"], key, strict=True))
                            selected_rows.append(
                                {
                                    **key_dict,
                                    "oracle_positive_slots": int(len(chosen_edges)),
                                    "oracle_mean_selected_edge": float(np.mean(chosen_edges)),
                                    "oracle_sum_selected_edge": float(np.sum(chosen_edges)),
                                }
                            )
                selected_month = pd.DataFrame(selected_rows)
                if selected_month.empty:
                    selected_month = pd.DataFrame(
                        columns=[
                            *dims,
                            "trade_date",
                            "oracle_positive_slots",
                            "oracle_mean_selected_edge",
                            "oracle_sum_selected_edge",
                        ]
                    )
                gate_month = month_state.merge(selected_month, on=[*dims, "trade_date"], how="left")
                if gate_col:
                    gate_month.loc[~gate_month[gate_col].astype(bool), "oracle_positive_slots"] = 0
                    gate_month.loc[~gate_month[gate_col].astype(bool), "oracle_mean_selected_edge"] = np.nan
                    gate_month.loc[~gate_month[gate_col].astype(bool), "oracle_sum_selected_edge"] = 0.0
                gate_month["oracle_positive_slots"] = gate_month["oracle_positive_slots"].fillna(0).astype(int)
                gate_month["oracle_sum_selected_edge"] = gate_month["oracle_sum_selected_edge"].fillna(0.0)
                gate_month["horizon"] = int(horizon)
                gate_month["cost_buffer"] = float(cost)
                gate_month["state_gate"] = gate_name
                month_rows.append(
                    gate_month[
                        [
                            *dims,
                            "horizon",
                            "cost_buffer",
                            "state_gate",
                            "trade_date",
                            "lagged_regime",
                            "lagged_breadth",
                            "oracle_positive_slots",
                            "oracle_mean_selected_edge",
                            "oracle_sum_selected_edge",
                        ]
                    ]
                )

                agg = (
                    gate_month.groupby(dims, dropna=False)
                    .agg(
                        months=("trade_date", "count"),
                        avg_oracle_positive_slots=("oracle_positive_slots", "mean"),
                        active_month_share=("oracle_positive_slots", lambda s: float((s > 0).mean())),
                        full_3_slot_month_share=("oracle_positive_slots", lambda s: float((s >= max_replace).mean())),
                        avg_selected_edge_when_active=(
                            "oracle_mean_selected_edge",
                            lambda s: float(s.dropna().mean()) if not s.dropna().empty else np.nan,
                        ),
                        avg_sum_edge_per_month=("oracle_sum_selected_edge", "mean"),
                    )
                    .reset_index()
                )
                agg["horizon"] = int(horizon)
                agg["cost_buffer"] = float(cost)
                agg["state_gate"] = gate_name
                summary_rows.append(agg)
    month_out = pd.concat(month_rows, ignore_index=True) if month_rows else pd.DataFrame()
    summary_out = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    if not summary_out.empty:
        summary_out = summary_out[
            [
                *dims,
                "horizon",
                "cost_buffer",
                "state_gate",
                "months",
                "avg_oracle_positive_slots",
                "active_month_share",
                "full_3_slot_month_share",
                "avg_selected_edge_when_active",
                "avg_sum_edge_per_month",
            ]
        ]
    return month_out, summary_out


def _summarize_state_gate_precision(
    base_pairs: pd.DataFrame,
    *,
    horizon: int,
    cost_buffer: float,
) -> pd.DataFrame:
    df = _with_edge(base_pairs, horizon=horizon, cost_buffer=cost_buffer).dropna(subset=["pair_edge"])
    rows = []
    gates = ["state_strong_up_or_wide", "state_strong_up_and_wide", "state_up_or_wide_not_strong_down"]
    dims = ["score_col", "old_pool", "candidate_pool"]
    for key, sub in df.groupby(dims, dropna=False):
        key_dict = dict(zip(dims, key, strict=True))
        base_hit = float(sub["replace_win"].mean()) if len(sub) else np.nan
        for gate in gates:
            for val, part in sub.groupby(gate, dropna=False):
                rows.append(
                    {
                        **key_dict,
                        "horizon": int(horizon),
                        "cost_buffer": float(cost_buffer),
                        "gate": gate,
                        "gate_value": bool(val),
                        "pair_count": int(len(part)),
                        "hit_rate": float(part["replace_win"].mean()) if len(part) else np.nan,
                        "lift_vs_all": float(part["replace_win"].mean() - base_hit) if len(part) and np.isfinite(base_hit) else np.nan,
                        "mean_pair_edge": float(part["pair_edge"].mean()) if len(part) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _summarize_feature_buckets(
    base_pairs: pd.DataFrame,
    *,
    horizon: int,
    cost_buffer: float,
) -> pd.DataFrame:
    df = _with_edge(base_pairs, horizon=horizon, cost_buffer=cost_buffer).dropna(subset=["pair_edge"]).copy()
    features = [
        "score_margin",
        "candidate_score",
        "candidate_score_pct",
        "old_defensive_score",
        "rel_strength_diff",
        "amount_expansion_diff",
        "turnover_expansion_diff",
        "overheat_diff",
    ]
    rows = []
    dims = ["score_col", "old_pool", "candidate_pool"]
    for key, sub in df.groupby(dims, dropna=False):
        key_dict = dict(zip(dims, key, strict=True))
        for feature in features:
            vals = pd.to_numeric(sub[feature], errors="coerce")
            valid = sub[np.isfinite(vals)].copy()
            if len(valid) < 50 or valid[feature].nunique(dropna=True) < 5:
                continue
            try:
                valid["_bucket"] = pd.qcut(pd.to_numeric(valid[feature], errors="coerce"), 5, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            for bucket, part in valid.groupby("_bucket", dropna=False):
                rows.append(
                    {
                        **key_dict,
                        "horizon": int(horizon),
                        "cost_buffer": float(cost_buffer),
                        "feature": feature,
                        "bucket": int(bucket),
                        "pair_count": int(len(part)),
                        "feature_min": float(pd.to_numeric(part[feature], errors="coerce").min()),
                        "feature_max": float(pd.to_numeric(part[feature], errors="coerce").max()),
                        "hit_rate": float(part["replace_win"].mean()),
                        "mean_pair_edge": float(part["pair_edge"].mean()),
                        "median_pair_edge": float(part["pair_edge"].median()),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mono_rows = []
    for key, sub in out.groupby([*dims, "feature"], dropna=False):
        ordered = sub.sort_values("bucket")
        corr = ordered["bucket"].corr(ordered["mean_pair_edge"], method="spearman") if len(ordered) >= 3 else np.nan
        mono_rows.append({**dict(zip([*dims, "feature"], key, strict=True)), "bucket_edge_spearman": corr})
    return out.merge(pd.DataFrame(mono_rows), on=[*dims, "feature"], how="left")


def _summarize_cost_sensitivity(capacity_summary: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    if capacity_summary.empty:
        return capacity_summary
    return capacity_summary[capacity_summary["horizon"] == int(horizon)].copy()


def _run_oracle_backtests(
    *,
    base_pairs: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    cost_params: Any,
    limit_up_open_mask: pd.DataFrame,
    horizon: int,
    cost_buffer: float,
    max_replace: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_col = _edge_col(horizon, cost_buffer)
    pairs_with_edge = _with_edge(base_pairs, horizon=horizon, cost_buffer=cost_buffer)
    base_detail, base_meta = _detail_from_weights(
        weights=defensive_weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        cost_params=cost_params,
        scenario=BASELINE_ID,
        limit_up_open_mask=limit_up_open_mask,
    )
    base_summary = summarize_tree_daily_backtest_like_proxy(base_detail)
    base_proxy = float(base_summary.get("annualized_excess_vs_market", np.nan))
    rows = [
        {
            "candidate_id": BASELINE_ID,
            "score_col": "score__defensive",
            "old_pool": "",
            "candidate_pool": "",
            "state_gate": "",
            "horizon": int(horizon),
            "cost_buffer": float(cost_buffer),
            "daily_proxy_annualized_excess_vs_market": base_proxy,
            "best_possible_replace_3_excess": 0.0,
            "avg_turnover_half_l1": float(base_meta.get("avg_turnover_half_l1", np.nan)),
            "oracle_selected_pairs": 0,
            "active_rebalance_share": 0.0,
        }
    ]
    selected_frames = []
    gates = {"all_states": None, "strong_up_or_wide": "state_strong_up_or_wide"}
    dims = ["score_col", "old_pool", "candidate_pool"]
    for key, sub in pairs_with_edge.groupby(dims, dropna=False):
        key_dict = dict(zip(dims, key, strict=True))
        for gate_name, gate_col in gates.items():
            selected = select_oracle_replacements(sub, edge_col=edge_col, max_replace=max_replace, state_gate=gate_col)
            if not selected.empty:
                selected = selected.copy()
                selected["state_gate"] = gate_name
                selected_frames.append(selected)
            weights = build_oracle_weights(defensive_weights, selected)
            detail, meta = _detail_from_weights(
                weights=weights,
                asset_returns=asset_returns,
                bench_daily=bench_daily,
                cost_params=cost_params,
                scenario=f"ORACLE_{key_dict['score_col']}__{key_dict['old_pool']}__{key_dict['candidate_pool']}__{gate_name}",
                limit_up_open_mask=limit_up_open_mask,
            )
            summary = summarize_tree_daily_backtest_like_proxy(detail)
            proxy = float(summary.get("annualized_excess_vs_market", np.nan))
            active_share = (
                float(selected.groupby("trade_date").size().reindex(defensive_weights.index, fill_value=0).gt(0).mean())
                if not selected.empty
                else 0.0
            )
            rows.append(
                {
                    "candidate_id": f"ORACLE_{key_dict['score_col']}__{key_dict['old_pool']}__{key_dict['candidate_pool']}__{gate_name}",
                    **key_dict,
                    "state_gate": gate_name,
                    "horizon": int(horizon),
                    "cost_buffer": float(cost_buffer),
                    "daily_proxy_annualized_excess_vs_market": proxy,
                    "best_possible_replace_3_excess": proxy - base_proxy,
                    "avg_turnover_half_l1": float(meta.get("avg_turnover_half_l1", np.nan)),
                    "oracle_selected_pairs": int(len(selected)),
                    "active_rebalance_share": active_share,
                }
            )
    selected_long = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return pd.DataFrame(rows).sort_values("daily_proxy_annualized_excess_vs_market", ascending=False), selected_long


def _long_with_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "candidate_id", cid)
    return out


def _build_doc(
    *,
    params: dict[str, Any],
    oracle_bt: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    state_gate_precision: pd.DataFrame,
    feature_buckets: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2B Oracle Replacement Attribution\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- `eval_contract_version`: `{EVAL_CONTRACT_VERSION}`")
    lines.append(f"- `execution_contract_version`: `{EXECUTION_CONTRACT_VERSION}`")
    lines.append(f"- `industry_map_source`: `{params.get('industry_map_source', '')}`")
    lines.append(f"- `industry_map_source_status`: `{params.get('industry_map_source_status', '')}`")
    lines.append("- 目标：先判断边界 replacement 是否存在事后可学习上限，再决定是否进入 R2B v2。")
    lines.append("")

    lines.append("## 1. Oracle Replace-3 理论上限\n")
    view_cols = [
        "candidate_id",
        "daily_proxy_annualized_excess_vs_market",
        "best_possible_replace_3_excess",
        "avg_turnover_half_l1",
        "oracle_selected_pairs",
        "active_rebalance_share",
    ]
    lines.append(oracle_bt[[c for c in view_cols if c in oracle_bt.columns]].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 2. Oracle Capacity\n")
    cap_view = capacity_summary[
        (capacity_summary["horizon"] == int(params["default_horizon"]))
        & np.isclose(capacity_summary["cost_buffer"], float(params["default_cost_buffer"]))
        & (capacity_summary["state_gate"].isin(["all_states", "strong_up_or_wide"]))
    ].copy()
    cap_cols = [
        "score_col",
        "old_pool",
        "candidate_pool",
        "state_gate",
        "avg_oracle_positive_slots",
        "active_month_share",
        "full_3_slot_month_share",
        "avg_selected_edge_when_active",
        "avg_sum_edge_per_month",
    ]
    lines.append(cap_view[cap_cols].sort_values("avg_sum_edge_per_month", ascending=False).head(20).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 3. State Gate Precision\n")
    sg_cols = ["score_col", "old_pool", "candidate_pool", "gate", "gate_value", "pair_count", "hit_rate", "lift_vs_all", "mean_pair_edge"]
    lines.append(state_gate_precision[sg_cols].sort_values("lift_vs_all", ascending=False).head(24).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 4. Feature Bucket Monotonicity\n")
    fb_cols = [
        "score_col",
        "old_pool",
        "candidate_pool",
        "feature",
        "bucket",
        "pair_count",
        "hit_rate",
        "mean_pair_edge",
        "bucket_edge_spearman",
    ]
    lines.append(feature_buckets[fb_cols].sort_values(["bucket_edge_spearman", "feature", "bucket"], ascending=[False, True, True]).head(30).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 5. 判定\n")
    non_base = oracle_bt[oracle_bt["candidate_id"] != BASELINE_ID].copy()
    gated = non_base[non_base.get("state_gate", pd.Series(dtype=str)).astype(str) == "strong_up_or_wide"].copy()
    best = gated.iloc[0] if not gated.empty else (non_base.iloc[0] if not non_base.empty else None)
    decision_scope = "strong_up_or_wide state-gated" if not gated.empty else "all_states diagnostic"
    if best is None or not np.isfinite(float(best["best_possible_replace_3_excess"])):
        lines.append("- 无有效 oracle replacement 结果；R2B 不应进入 v2。")
    elif float(best["best_possible_replace_3_excess"]) < 0.03:
        lines.append(
            f"- `{decision_scope}` 最优 oracle replace-3 的 daily proxy 增量为 `{float(best['best_possible_replace_3_excess']):.2%}`，"
            "低于 `+3%` 理论上限参考线；R2B v2 只能作为规则/数据诊断，不应启动复杂 R3。"
        )
    else:
        lines.append(
            f"- `{decision_scope}` 最优 oracle replace-3 的 daily proxy 增量为 `{float(best['best_possible_replace_3_excess']):.2%}`，"
            "说明候选池存在可学习上限；允许推进 R2B v2 edge-gated replacement。"
        )
    lines.append("- 若 feature bucket 单调性弱而 oracle 上限存在，下一步优先做 pairwise rule，不训练复杂模型。")
    lines.append("")

    lines.append("## 6. 产出文件\n")
    for suffix in [
        "oracle_hit_rate_by_state.csv",
        "oracle_capacity_by_month.csv",
        "oracle_capacity_summary.csv",
        "best_possible_replace_3_excess.csv",
        "feature_bucket_monotonicity.csv",
        "state_gate_precision.csv",
        "cost_sensitivity.csv",
        "oracle_selected_pairs.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suffix}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    horizons = _parse_int_list(args.horizons)
    cost_buffers = _parse_float_list(args.cost_buffers)
    candidate_pcts = _parse_float_list(args.candidate_pcts)
    old_pool_sizes = _parse_int_list(args.old_pool_sizes)
    if args.default_horizon not in horizons:
        horizons.append(args.default_horizon)
    if not any(np.isclose(c, args.default_cost_buffer) for c in cost_buffers):
        cost_buffers.append(float(args.default_cost_buffer))
    horizons = sorted(set(int(h) for h in horizons))
    cost_buffers = sorted(set(float(c) for c in cost_buffers))

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

    print("[2/8] compute S2 + R2B features + real industry map", flush=True)
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
    print(f"  panel={panel.shape}; industry_status={industry_status}", flush=True)

    print("[3/8] returns + benchmark + lagged state", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    limit_up_open_mask = build_limit_up_open_mask(daily_df).sort_index()
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_state = _monthly_benchmark_frame(bench_daily, breadth_series)

    print("[4/8] baseline S2 defensive weights", flush=True)
    score_base = panel[["symbol", "trade_date", "score__defensive"]].dropna().copy()
    defensive_weights, defensive_diag = build_tree_score_weight_matrix(
        score_base,
        score_col="score__defensive",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    state_by_rebalance = _lagged_state_by_rebalance(defensive_weights.index, monthly_state)

    print("[5/8] forward returns + oracle pair base", flush=True)
    forward_by_horizon = {h: build_forward_returns_from_open(daily_df, horizon=h) for h in horizons}
    score_cols = [str(c["score_col"]) for c in UPSIDE_CANDIDATES]
    base_pairs = build_oracle_pair_base(
        panel=panel,
        defensive_weights=defensive_weights,
        state_by_rebalance=state_by_rebalance,
        trading_index=open_returns.index,
        limit_up_open_mask=limit_up_open_mask,
        forward_by_horizon=forward_by_horizon,
        score_cols=score_cols,
        old_pool_sizes=old_pool_sizes,
        candidate_pcts=candidate_pcts,
        max_limit_up_hits_20d=args.max_limit_up_hits_20d,
        max_expansion=args.max_expansion,
        max_candidates_per_pool=args.max_candidates_per_pool,
    )
    print(f"  pair_base_rows={len(base_pairs):,}", flush=True)

    print("[6/8] attribution summaries", flush=True)
    hit_rate = _summarize_hit_rate(base_pairs, horizons, cost_buffers)
    capacity_month, capacity_summary = _summarize_capacity(
        base_pairs,
        horizons=horizons,
        cost_buffers=cost_buffers,
        max_replace=args.max_oracle_replace,
    )
    state_gate_precision = _summarize_state_gate_precision(
        base_pairs,
        horizon=args.default_horizon,
        cost_buffer=args.default_cost_buffer,
    )
    feature_buckets = _summarize_feature_buckets(
        base_pairs,
        horizon=args.default_horizon,
        cost_buffer=args.default_cost_buffer,
    )
    cost_sensitivity = _summarize_cost_sensitivity(capacity_summary, horizon=args.default_horizon)

    print("[7/8] oracle replace-3 daily proxy upper bound", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()) | set(defensive_weights.columns.astype(str)))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    limit_mask = limit_up_open_mask.reindex(columns=sym_universe, fill_value=False)
    defensive_weights = defensive_weights.reindex(columns=sym_universe, fill_value=0.0)
    oracle_bt, selected_pairs = _run_oracle_backtests(
        base_pairs=base_pairs,
        defensive_weights=defensive_weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        cost_params=cost_params,
        limit_up_open_mask=limit_mask,
        horizon=args.default_horizon,
        cost_buffer=args.default_cost_buffer,
        max_replace=args.max_oracle_replace,
    )

    print("[8/8] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    hit_rate.to_csv(results_dir / f"{prefix}_oracle_hit_rate_by_state.csv", index=False, encoding="utf-8-sig")
    capacity_month.to_csv(results_dir / f"{prefix}_oracle_capacity_by_month.csv", index=False, encoding="utf-8-sig")
    capacity_summary.to_csv(results_dir / f"{prefix}_oracle_capacity_summary.csv", index=False, encoding="utf-8-sig")
    oracle_bt.to_csv(results_dir / f"{prefix}_best_possible_replace_3_excess.csv", index=False, encoding="utf-8-sig")
    feature_buckets.to_csv(results_dir / f"{prefix}_feature_bucket_monotonicity.csv", index=False, encoding="utf-8-sig")
    state_gate_precision.to_csv(results_dir / f"{prefix}_state_gate_precision.csv", index=False, encoding="utf-8-sig")
    cost_sensitivity.to_csv(results_dir / f"{prefix}_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    selected_pairs.to_csv(results_dir / f"{prefix}_oracle_selected_pairs.csv", index=False, encoding="utf-8-sig")
    state_by_rebalance.to_csv(results_dir / f"{prefix}_lagged_state_by_rebalance.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "oracle_defensive_core_replace_3",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "state_lag": "previous_completed_month",
        "state_threshold_mode": "expanding",
        "horizons": horizons,
        "cost_buffers": cost_buffers,
        "default_horizon": int(args.default_horizon),
        "default_cost_buffer": float(args.default_cost_buffer),
        "candidate_pcts": candidate_pcts,
        "old_pool_sizes": old_pool_sizes,
        "max_oracle_replace": int(args.max_oracle_replace),
        "max_limit_up_hits_20d": float(args.max_limit_up_hits_20d),
        "max_expansion": float(args.max_expansion),
        "max_candidates_per_pool": int(args.max_candidates_per_pool),
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
        "defensive_weight_diag_rows": int(len(defensive_diag)),
        "pair_base_rows": int(len(base_pairs)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "upside_candidates": UPSIDE_CANDIDATES,
        "oracle_backtest": oracle_bt.to_dict(orient="records"),
        "best_oracle": oracle_bt[oracle_bt["candidate_id"] != BASELINE_ID].head(1).to_dict(orient="records"),
        "capacity_summary_default": capacity_summary[
            (capacity_summary["horizon"] == int(args.default_horizon))
            & np.isclose(capacity_summary["cost_buffer"], float(args.default_cost_buffer))
        ].to_dict(orient="records"),
        "monthly_state_thresholds": {
            "regime": monthly_state.attrs.get("regime_thresholds"),
            "breadth": monthly_state.attrs.get("breadth_thresholds"),
            "trace": monthly_state.attrs.get("threshold_trace"),
        },
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc_text = _build_doc(
        params=params,
        oracle_bt=oracle_bt,
        capacity_summary=capacity_summary,
        state_gate_precision=state_gate_precision,
        feature_buckets=feature_buckets,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc_text, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)

    # --- standard research contract ---
    def _project_relative(path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(p)

    manifest_path = results_dir / f"{prefix}_manifest.json"
    identity = make_research_identity(
        result_type="r2b_oracle_replacement_attribution",
        research_topic="r2b_oracle_replacement_attribution",
        research_config_id=f"r2b_oracle_{slugify_token(prefix)}",
        output_stem=prefix,
    )
    data_slice = DataSlice(
        dataset_name="r2b_oracle_attribution_backtest",
        source_tables=("a_share_daily",),
        date_start=args.start,
        date_end=end_date,
        asof_trade_date=end_date,
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="U1_liquid_tradable",
        rebalance_rule=rebalance_rule,
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="r2b_oracle_factors",
        feature_columns=(),
        label_columns=(),
        pit_policy="oracle_future_visible_for_attribution_only",
        config_path=config_source,
        extra={"top_k": int(top_k), "max_turnover": float(max_turnover), "default_horizon": int(args.default_horizon)},
    )
    artifact_refs = (
        ArtifactRef("oracle_hit_rate_by_state_csv", _project_relative(results_dir / f"{prefix}_oracle_hit_rate_by_state.csv"), "csv", False, "命中率按状态"),
        ArtifactRef("oracle_capacity_by_month_csv", _project_relative(results_dir / f"{prefix}_oracle_capacity_by_month.csv"), "csv", False, "容量按月"),
        ArtifactRef("oracle_capacity_summary_csv", _project_relative(results_dir / f"{prefix}_oracle_capacity_summary.csv"), "csv", False, "容量汇总"),
        ArtifactRef("best_possible_replace_3_excess_csv", _project_relative(results_dir / f"{prefix}_best_possible_replace_3_excess.csv"), "csv", False, "最优替换"),
        ArtifactRef("feature_bucket_monotonicity_csv", _project_relative(results_dir / f"{prefix}_feature_bucket_monotonicity.csv"), "csv", False, "特征单调性"),
        ArtifactRef("state_gate_precision_csv", _project_relative(results_dir / f"{prefix}_state_gate_precision.csv"), "csv", False, "状态 gate 精度"),
        ArtifactRef("cost_sensitivity_csv", _project_relative(results_dir / f"{prefix}_cost_sensitivity.csv"), "csv", False, "成本敏感性"),
        ArtifactRef("oracle_selected_pairs_csv", _project_relative(results_dir / f"{prefix}_oracle_selected_pairs.csv"), "csv", False, "Oracle 选中pair"),
        ArtifactRef("summary_json", _project_relative(results_dir / f"{prefix}_summary.json"), "json", False, "汇总"),
        ArtifactRef("report_md", _project_relative(docs_dir / f"{prefix}.md"), "md", False, "报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    )
    metrics = {
        "oracle_bt_rows": int(len(oracle_bt)),
        "capacity_summary_rows": int(len(capacity_summary)),
        "hit_rate_rows": int(len(hit_rate)),
    }
    gates = {
        "data_gate": {"passed": bool(len(oracle_bt) > 0)},
        "execution_gate": {"passed": True},
        "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=" ".join(sys.argv),
        created_at=utc_now_iso(),
        duration_sec=None,
        seed=None,
        data_slices=(data_slice,),
        config=config_snapshot(config_path=config_source),
        params={"cli": {k: str(v) for k, v in vars(args).items()}},
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={"production_eligible": False, "registry_status": "not_registered", "blocking_reasons": ["r2b_oracle_is_attribution_only"]},
        notes="R2B oracle replacement attribution; uses future data for analysis only, not a promotion candidate.",
    )
    write_research_manifest(manifest_path, result)
    append_experiment_result(PROJECT_ROOT / "data" / "experiments", result)
    # --- end standard research contract ---


if __name__ == "__main__":
    main()
