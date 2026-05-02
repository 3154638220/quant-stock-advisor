"""R2B v2 edge-gated replacement.

This runner keeps the S2 defensive Top-20 core and only replaces boundary names
when a rule-based pair edge score clears state, score, cost, buyability, risk,
industry, and capacity gates. It is intentionally still a daily-proxy-first
research artifact, not a production promotion path.
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
    _filter_universe,
    _lagged_state_by_rebalance,
    _monthly_benchmark_frame,
)
from scripts.run_p2_upside_sleeve_v1 import GATE_FULL_BACKTEST, _add_cs_rank  # noqa: E402
from scripts.run_r2b_oracle_replacement_attribution import (  # noqa: E402
    _edge_col,
    _with_edge,
    build_forward_returns_from_open,
    build_oracle_pair_base,
)
from scripts.run_r2b_tradable_upside_replacement_v1 import (  # noqa: E402
    BASELINE_ID,
    EVAL_CONTRACT_VERSION,
    EXECUTION_CONTRACT_VERSION,
    _accept_summary,
    _build_leaderboard,
    _industry_exposure,
    _industry_source_status,
    _load_or_build_industry_map,
    _long_with_id,
    _overlap_vs_baseline,
    _run_from_weights,
    _score_for_groups,
    _add_basic_r2b_features,
    build_r2b_scores,
)
from src.models.xtree.p1_workflow import build_tree_score_weight_matrix  # noqa: E402
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


DEFAULT_OUTPUT_PREFIX = "r2b_edge_gated_replacement_v2_2026-04-28"

U3_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "U3_A_real_industry_leadership__EDGE_GATED",
        "input_id": "U3_A_real_industry_leadership",
        "label": "real industry breadth + intra-industry leadership persistence",
        "score_col": "score__u3_a",
        "old_pool": "S2_bottom_3",
        "candidate_pool": "candidate_top_pct_95",
        "state_gate": "state_strong_up_and_wide",
        "edge_threshold": 0.68,
        "min_score_margin": 0.10,
    },
    {
        "id": "U3_B_buyable_leadership_persistence__EDGE_GATED",
        "input_id": "U3_B_buyable_leadership_persistence",
        "label": "buyable leadership persistence with breakout and overheat controls",
        "score_col": "score__u3_b",
        "old_pool": "S2_bottom_3",
        "candidate_pool": "candidate_top_pct_90",
        "state_gate": "state_up_or_wide_not_strong_down",
        "edge_threshold": 0.66,
        "min_score_margin": 0.08,
    },
    {
        "id": "U3_C_pairwise_residual_edge__EDGE_GATED",
        "input_id": "U3_C_pairwise_residual_edge",
        "label": "pairwise residual edge from score, strength, liquidity and overheat deltas",
        "score_col": "score__u3_c",
        "old_pool": "S2_bottom_5",
        "candidate_pool": "candidate_buyable",
        "state_gate": "state_strong_up_or_wide",
        "edge_threshold": 0.72,
        "min_score_margin": 0.06,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2B v2 edge-gated replacement")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--industry-map", default="data/cache/industry_map.csv")
    p.add_argument("--default-horizon", type=int, default=20)
    p.add_argument("--default-cost-buffer", type=float, default=0.0015)
    p.add_argument("--max-replace", type=int, default=3)
    p.add_argument("--max-industry-names", type=int, default=5)
    p.add_argument("--max-limit-up-hits-20d", type=float, default=2.0)
    p.add_argument("--max-expansion", type=float, default=1.50)
    p.add_argument(
        "--max-candidates-per-pool",
        type=int,
        default=0,
        help="Optional cap after sorting by candidate score; 0 keeps the full pool.",
    )
    return p.parse_args()


def add_u3_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the three U3 rule scores used by R2B v2."""
    df = panel.copy()
    df["industry_positive_60d"] = (pd.to_numeric(df["rel_strength_60d"], errors="coerce") > 0).astype(float)
    df["industry_breadth_60d"] = df.groupby(["trade_date", "industry"])["industry_positive_60d"].transform("mean")
    df["industry_breadth_persistence"] = (
        0.55 * pd.to_numeric(df["industry_breadth_20d"], errors="coerce")
        + 0.45 * pd.to_numeric(df["industry_breadth_60d"], errors="coerce")
    )
    df["industry_amount_expansion"] = df.groupby(["trade_date", "industry"])["amount_expansion_5_60"].transform("mean")
    df["trend_consistency"] = np.minimum(
        pd.to_numeric(df["rel_strength_20d"], errors="coerce"),
        pd.to_numeric(df["rel_strength_60d"], errors="coerce"),
    )
    df["leadership_overheat_penalty"] = (
        pd.to_numeric(df["limit_up_hits_20d"], errors="coerce").fillna(0.0)
        + 0.50 * pd.to_numeric(df["limit_down_hits_20d"], errors="coerce").fillna(0.0)
        + np.maximum(pd.to_numeric(df["overnight_gap_pos_20d"], errors="coerce").fillna(0.0) - 0.35, 0.0)
        + np.maximum(pd.to_numeric(df["amount_expansion_5_60"], errors="coerce").fillna(0.0) - 1.0, 0.0)
        + np.maximum(pd.to_numeric(df["turnover_expansion_5_60"], errors="coerce").fillna(0.0) - 1.0, 0.0)
    )

    rank_cols = [
        "industry_breadth_60d",
        "industry_breadth_persistence",
        "industry_amount_expansion",
        "rel_strength_60d",
        "trend_consistency",
        "overnight_gap_pos_20d",
        "leadership_overheat_penalty",
        "s2_residual_elasticity",
    ]
    for col in rank_cols:
        df = _add_cs_rank(df, col, f"_rk_{col}")

    df["score__u3_a"] = (
        0.30 * df["_rk_industry_breadth_persistence"]
        + 0.25 * df["_rk_intra_industry_strength"]
        + 0.20 * df["_rk_industry_amount_expansion"]
        + 0.15 * df["_rk_trend_consistency"]
        + 0.10 * df["_rk_amount_expansion_5_60"]
        - 0.25 * df["_rk_leadership_overheat_penalty"]
    )
    df["score__u3_b"] = (
        0.30 * df["_rk_trend_consistency"]
        + 0.25 * df["_rk_breakout_60d"]
        + 0.20 * df["_rk_tail_strength_20d"]
        + 0.15 * df["_rk_turnover_expansion_5_60"]
        + 0.10 * df["_rk_overnight_gap_pos_20d"]
        - 0.30 * df["_rk_leadership_overheat_penalty"]
    )
    df["score__u3_c"] = (
        0.35 * df["_rk_s2_residual_elasticity"]
        + 0.25 * df["_rk_trend_consistency"]
        + 0.15 * df["_rk_amount_expansion_5_60"]
        + 0.10 * df["_rk_turnover_expansion_5_60"]
        + 0.10 * df["_rk_intra_industry_strength"]
        - 0.25 * df["_rk_leadership_overheat_penalty"]
        - 0.10 * df["_rk_realized_vol_20d"]
    )
    for col in ("score__u3_a", "score__u3_b", "score__u3_c"):
        df = _add_cs_rank(df, col, f"{col}_pct")
    return df


def _rank_feature(
    df: pd.DataFrame,
    col: str,
    *,
    ascending: bool = True,
    group_cols: list[str] | None = None,
) -> pd.Series:
    group_cols = group_cols or ["trade_date", "score_col", "old_pool", "candidate_pool"]
    vals = pd.to_numeric(df[col], errors="coerce")
    ranked = vals.groupby([df[c] for c in group_cols], dropna=False).rank(pct=True, method="average", ascending=ascending)
    return ranked.fillna(0.5).clip(0.0, 1.0)


def add_pair_edge_score(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build a no-lookahead edge score for old/new replacement pairs."""
    if pairs.empty:
        return pairs.copy()
    out = pairs.copy()
    out["score_margin_rank"] = _rank_feature(out, "score_margin")
    out["rel_strength_diff_rank"] = _rank_feature(out, "rel_strength_diff")
    out["amount_expansion_diff_rank"] = _rank_feature(out, "amount_expansion_diff")
    out["turnover_expansion_diff_rank"] = _rank_feature(out, "turnover_expansion_diff")
    out["old_weakness_rank"] = _rank_feature(out, "old_defensive_score", ascending=False)
    out["overheat_relief_rank"] = _rank_feature(out, "overheat_diff", ascending=False)
    candidate_pct = pd.to_numeric(out["candidate_score_pct"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["pair_edge_score"] = (
        0.25 * candidate_pct
        + 0.20 * out["score_margin_rank"]
        + 0.20 * out["rel_strength_diff_rank"]
        + 0.10 * out["amount_expansion_diff_rank"]
        + 0.05 * out["turnover_expansion_diff_rank"]
        + 0.10 * out["old_weakness_rank"]
        + 0.10 * out["overheat_relief_rank"]
    )
    out["pair_edge_score"] = out["pair_edge_score"].clip(0.0, 1.0)
    return out


def _filter_rule_pairs(pairs: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    df = pairs[
        (pairs["score_col"].astype(str) == str(rule["score_col"]))
        & (pairs["old_pool"].astype(str) == str(rule["old_pool"]))
        & (pairs["candidate_pool"].astype(str) == str(rule["candidate_pool"]))
    ].copy()
    if df.empty:
        return df
    gate = str(rule["state_gate"])
    if gate in df.columns:
        df = df[df[gate].astype(bool)].copy()
    df["expected_edge_after_cost"] = pd.to_numeric(df["pair_edge_score"], errors="coerce") - float(rule["edge_threshold"])
    df = df[
        (pd.to_numeric(df["pair_edge_score"], errors="coerce") >= float(rule["edge_threshold"]))
        & (pd.to_numeric(df["score_margin"], errors="coerce") >= float(rule["min_score_margin"]))
        & (pd.to_numeric(df["expected_edge_after_cost"], errors="coerce") > 0.0)
    ].copy()
    return df


def select_edge_gated_replacements(
    pairs: pd.DataFrame,
    *,
    defensive_weights: pd.DataFrame,
    panel: pd.DataFrame,
    rule: dict[str, Any],
    max_replace: int,
    max_industry_names: int,
) -> pd.DataFrame:
    """Select 0..max_replace pairs per rebalance without using realized returns."""
    eligible = _filter_rule_pairs(pairs, rule)
    if eligible.empty:
        return pd.DataFrame()
    px = panel[["symbol", "trade_date", "industry"]].copy()
    px["symbol"] = px["symbol"].astype(str).str.zfill(6)
    px["trade_date"] = pd.to_datetime(px["trade_date"], errors="coerce").dt.normalize()
    panel_by_date = {pd.Timestamp(k).normalize(): v for k, v in px.groupby("trade_date", sort=False)}
    rows: list[dict[str, Any]] = []
    for rd, sub in eligible.sort_values(["pair_edge_score", "score_margin"], ascending=False).groupby("trade_date", sort=False):
        rd = pd.Timestamp(rd).normalize()
        if rd not in defensive_weights.index or rd not in panel_by_date:
            continue
        base_w = defensive_weights.loc[rd]
        base_symbols = [str(s).zfill(6) for s, w in base_w.items() if float(w) > 0.0]
        day_industry = panel_by_date[rd].set_index("symbol")["industry"].astype(str).to_dict()
        industry_counts: dict[str, int] = {}
        for sym in base_symbols:
            ind = day_industry.get(sym, "unknown")
            industry_counts[ind] = int(industry_counts.get(ind, 0)) + 1
        used_old: set[str] = set()
        used_new: set[str] = set()
        selected = 0
        for row in sub.itertuples(index=False):
            old_sym = str(row.old_symbol).zfill(6)
            new_sym = str(row.new_symbol).zfill(6)
            if old_sym in used_old or new_sym in used_new:
                continue
            old_ind = str(getattr(row, "old_industry", day_industry.get(old_sym, "unknown")))
            new_ind = str(getattr(row, "new_industry", day_industry.get(new_sym, "unknown")))
            projected = dict(industry_counts)
            projected[old_ind] = max(0, int(projected.get(old_ind, 0)) - 1)
            projected[new_ind] = int(projected.get(new_ind, 0)) + 1
            if projected[new_ind] > int(max_industry_names):
                continue
            rec = row._asdict()
            rec["candidate_id"] = str(rule["id"])
            rec["input_id"] = str(rule["input_id"])
            rec["state_gate"] = str(rule["state_gate"])
            rec["edge_threshold"] = float(rule["edge_threshold"])
            rec["min_score_margin"] = float(rule["min_score_margin"])
            rows.append(rec)
            industry_counts = projected
            used_old.add(old_sym)
            used_new.add(new_sym)
            selected += 1
            if selected >= int(max_replace):
                break
    return pd.DataFrame(rows)


def build_edge_gated_weights(
    *,
    defensive_weights: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = defensive_weights.copy()
    selected_by_date = (
        {pd.Timestamp(k).normalize(): v.copy() for k, v in selected_pairs.groupby("trade_date", sort=False)}
        if not selected_pairs.empty
        else {}
    )
    diag_rows: list[dict[str, Any]] = []
    for rd, base_w in defensive_weights.sort_index().iterrows():
        rd = pd.Timestamp(rd).normalize()
        sub = selected_by_date.get(rd, pd.DataFrame())
        if not sub.empty:
            for row in sub.itertuples(index=False):
                old_sym = str(row.old_symbol).zfill(6)
                new_sym = str(row.new_symbol).zfill(6)
                weight = float(out.loc[rd, old_sym]) if old_sym in out.columns else 0.0
                if weight <= 0.0:
                    continue
                out.loc[rd, old_sym] = 0.0
                if new_sym not in out.columns:
                    out[new_sym] = 0.0
                out.loc[rd, new_sym] = float(out.loc[rd, new_sym]) + weight
        if not sub.empty:
            blocked_reason = "selected"
        else:
            blocked_reason = "no_pair_passed_edge_gate"
        diag_rows.append(
            {
                "candidate_id": str(rule["id"]),
                "input_id": str(rule["input_id"]),
                "trade_date": rd,
                "score_col": str(rule["score_col"]),
                "old_pool": str(rule["old_pool"]),
                "candidate_pool": str(rule["candidate_pool"]),
                "state_gate": str(rule["state_gate"]),
                "edge_threshold": float(rule["edge_threshold"]),
                "min_score_margin": float(rule["min_score_margin"]),
                "replacement_count": int(len(sub)),
                "old_symbols": ",".join(sub["old_symbol"].astype(str).str.zfill(6).tolist()) if not sub.empty else "",
                "new_symbols": ",".join(sub["new_symbol"].astype(str).str.zfill(6).tolist()) if not sub.empty else "",
                "avg_pair_edge_score": float(pd.to_numeric(sub["pair_edge_score"], errors="coerce").mean()) if not sub.empty else np.nan,
                "avg_expected_edge_after_cost": (
                    float(pd.to_numeric(sub["expected_edge_after_cost"], errors="coerce").mean())
                    if "expected_edge_after_cost" in sub.columns and not sub.empty
                    else np.nan
                ),
                "avg_realized_pair_edge": (
                    float(pd.to_numeric(sub["pair_edge"], errors="coerce").mean())
                    if "pair_edge" in sub.columns and not sub.empty
                    else np.nan
                ),
                "blocked_reason": blocked_reason,
            }
        )
    out = out.fillna(0.0)
    totals = out.sum(axis=1).replace(0.0, np.nan)
    out = out.div(totals, axis=0).fillna(0.0).sort_index()
    return out, pd.DataFrame(diag_rows)


def _replacement_count_distribution(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    return (
        diag.groupby(["candidate_id", "replacement_count"], dropna=False)
        .agg(months=("trade_date", "count"))
        .reset_index()
        .assign(month_share=lambda d: d["months"] / d.groupby("candidate_id")["months"].transform("sum"))
        .sort_values(["candidate_id", "replacement_count"])
    )


def _year_improvement_table(year_long: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    if year_long.empty:
        return pd.DataFrame()
    base = year_long[(year_long["candidate_id"] == BASELINE_ID) & (year_long["regime"] == "strong_up")].copy()
    if base.empty:
        return pd.DataFrame()
    base = base[["year", "median_excess_return", "positive_excess_share"]].rename(
        columns={
            "median_excess_return": "baseline_strong_up_median_excess",
            "positive_excess_share": "baseline_strong_up_positive_share",
        }
    )
    cand = year_long[(year_long["candidate_id"] != BASELINE_ID) & (year_long["regime"] == "strong_up")].copy()
    out = cand.merge(base, on="year", how="left")
    out["delta_strong_up_median_excess"] = out["median_excess_return"] - out["baseline_strong_up_median_excess"]
    out["delta_strong_up_positive_share"] = out["positive_excess_share"] - out["baseline_strong_up_positive_share"]
    keep_ids = set(leaderboard["candidate_id"].astype(str))
    return out[out["candidate_id"].astype(str).isin(keep_ids)].copy()


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    leaderboard: pd.DataFrame,
    accept_map: dict[str, dict[str, Any]],
    replacement_diag: pd.DataFrame,
    replacement_count_dist: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    regime_long: pd.DataFrame,
    breadth_long: pd.DataFrame,
    switch_long: pd.DataFrame,
    year_improvement: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2B v2 Edge-Gated Replacement\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append(f"- `eval_contract_version`: `{EVAL_CONTRACT_VERSION}`")
    lines.append(f"- `execution_contract_version`: `{EXECUTION_CONTRACT_VERSION}`")
    lines.append(f"- `industry_map_source`: `{params.get('industry_map_source', '')}`")
    lines.append(f"- `industry_map_source_status`: `{params.get('industry_map_source_status', '')}`")
    lines.append("- 组合表达：`S2 defensive core + edge-gated replacement`；每月替换 `0/1/2/3`，不默认填满 slot。")
    lines.append("- pair gate 只使用当期可观测 feature；`realized_pair_edge` 只用于事后诊断。")
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
        "strong_up_topk_minus_next",
        "avg_turnover_half_l1",
        "delta_vs_baseline_turnover",
        "industry_map_source_status",
        "n_rebalances",
    ]
    lines.append(leaderboard[[c for c in cols if c in leaderboard.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 2. R2B v2 验收\n")
    accept_rows = []
    for cid, info in accept_map.items():
        row = {"candidate_id": cid, "status": info["status"]}
        row.update(info.get("checks", {}))
        accept_rows.append(row)
    lines.append(pd.DataFrame(accept_rows).to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Replacement Count Distribution\n")
    lines.append("_无 replacement 诊断_" if replacement_count_dist.empty else replacement_count_dist.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 4. Selected Pair Diagnostics\n")
    if selected_pairs.empty:
        lines.append("_无 selected pairs_")
    else:
        pair_cols = [
            "candidate_id",
            "state_gate",
            "old_pool",
            "candidate_pool",
            "selected_pairs",
            "avg_pair_edge_score",
            "avg_expected_edge_after_cost",
            "avg_realized_pair_edge",
            "realized_win_rate",
        ]
        pair_summary = (
            selected_pairs.groupby(["candidate_id", "state_gate", "old_pool", "candidate_pool"], dropna=False)
            .agg(
                selected_pairs=("trade_date", "count"),
                avg_pair_edge_score=("pair_edge_score", "mean"),
                avg_expected_edge_after_cost=("expected_edge_after_cost", "mean"),
                avg_realized_pair_edge=("pair_edge", "mean"),
                realized_win_rate=("replace_win", "mean"),
            )
            .reset_index()
        )
        lines.append(pair_summary[[c for c in pair_cols if c in pair_summary.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 5. Regime / Breadth\n")
    lines.append(regime_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(breadth_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 6. 2021 / 2025 / 2026 Strong-Up\n")
    if year_improvement.empty:
        lines.append("_无 strong-up 年度样本_")
    else:
        year_cols = [
            "candidate_id",
            "year",
            "months",
            "median_excess_return",
            "delta_strong_up_median_excess",
            "positive_excess_share",
            "delta_strong_up_positive_share",
        ]
        lines.append(year_improvement[[c for c in year_cols if c in year_improvement.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 7. Switch Quality\n")
    lines.append("_无 switch 样本_" if switch_long.empty else switch_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 8. 行业暴露\n")
    if params.get("industry_map_source_status") != "real_industry_map":
        lines.append(
            f"> 行业来源为 `{params.get('industry_map_source_status')}`；industry breadth 仅可诊断，不可作为 alpha 证据。"
        )
        lines.append("")
    if industry_exposure.empty:
        lines.append("_无行业暴露_")
    else:
        latest = industry_exposure.sort_values("trade_date").groupby("candidate_id").tail(5)
        lines.append(latest.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 9. 结论\n")
    non_base = leaderboard[leaderboard["candidate_id"] != BASELINE_ID].copy()
    full_bt = non_base[non_base["daily_proxy_annualized_excess_vs_market"] >= GATE_FULL_BACKTEST]["candidate_id"].tolist()
    gray = [cid for cid, info in accept_map.items() if info["status"] == "gray_zone"]
    passed = [cid for cid, info in accept_map.items() if info["status"] == "pass"]
    if passed:
        lines.append(f"- R2B v2 通过候选：{', '.join(passed)}。")
    elif gray:
        lines.append(f"- R2B v2 进入 gray zone 候选：{', '.join(gray)}；可补更细 slice，但仍非 production candidate。")
    else:
        lines.append("- R2B v2 无候选通过，也无 gray zone；不启动 R3 boundary model。")
    if full_bt:
        lines.append(f"- daily proxy `>= +3%` 候选：{', '.join(full_bt)}，允许补正式 full backtest。")
    else:
        lines.append("- 无候选达到 daily proxy `>= +3%`，不补正式 full backtest。")
    lines.append("")

    lines.append("## 10. 产出文件\n")
    for suffix in [
        "leaderboard.csv",
        "replacement_diag_long.csv",
        "replacement_count_distribution.csv",
        "selected_pairs.csv",
        "overlap_long.csv",
        "industry_exposure_long.csv",
        "regime_long.csv",
        "breadth_long.csv",
        "year_long.csv",
        "switch_long.csv",
        "monthly_long.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suffix}`")
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

    print("[2/8] compute S2 + U3 features + real industry map", flush=True)
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
    panel = add_u3_scores(build_r2b_scores(panel))
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

    print("[5/8] pair base + no-lookahead edge scores", flush=True)
    forward_by_horizon = {int(args.default_horizon): build_forward_returns_from_open(daily_df, horizon=int(args.default_horizon))}
    score_cols = [str(c["score_col"]) for c in U3_CANDIDATES]
    old_pool_sizes = sorted({int(str(c["old_pool"]).split("_")[-1]) for c in U3_CANDIDATES})
    candidate_pcts = sorted(
        {
            int(str(c["candidate_pool"]).split("_")[-1]) / 100.0
            for c in U3_CANDIDATES
            if str(c["candidate_pool"]).startswith("candidate_top_pct_")
        }
    )
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
        max_limit_up_hits_20d=float(args.max_limit_up_hits_20d),
        max_expansion=float(args.max_expansion),
        max_candidates_per_pool=int(args.max_candidates_per_pool),
    )
    base_pairs = _with_edge(base_pairs, horizon=int(args.default_horizon), cost_buffer=float(args.default_cost_buffer))
    base_pairs = add_pair_edge_score(base_pairs)
    print(f"  pair_base_rows={len(base_pairs):,}", flush=True)

    print("[6/8] select edge-gated replacements", flush=True)
    selected_frames = []
    replacement_weights: dict[str, pd.DataFrame] = {}
    replacement_diag_frames = []
    for rule in U3_CANDIDATES:
        selected = select_edge_gated_replacements(
            base_pairs,
            defensive_weights=defensive_weights,
            panel=panel,
            rule=rule,
            max_replace=int(args.max_replace),
            max_industry_names=int(args.max_industry_names),
        )
        if not selected.empty:
            selected_frames.append(selected)
        w, diag = build_edge_gated_weights(defensive_weights=defensive_weights, selected_pairs=selected, rule=rule)
        replacement_weights[str(rule["id"])] = w
        replacement_diag_frames.append(diag)
        print(f"  {rule['id']}: selected_pairs={len(selected)}", flush=True)

    selected_pairs = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    replacement_diag = pd.concat(replacement_diag_frames, ignore_index=True) if replacement_diag_frames else pd.DataFrame()
    replacement_count_dist = _replacement_count_distribution(replacement_diag)

    print("[7/8] run daily proxy + diagnostics", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(
        set(panel["symbol"].astype(str).str.zfill(6).unique())
        | set(defensive_weights.columns.astype(str))
        | set().union(*(set(w.columns.astype(str)) for w in replacement_weights.values()))
    )
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    limit_mask = limit_up_open_mask.reindex(columns=sym_universe, fill_value=False)
    defensive_weights = defensive_weights.reindex(columns=sym_universe, fill_value=0.0)
    replacement_weights = {cid: w.reindex(columns=sym_universe, fill_value=0.0) for cid, w in replacement_weights.items()}

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
    for rule in U3_CANDIDATES:
        cid = str(rule["id"])
        print(f"  edge-gated -> {cid}", flush=True)
        results.append(
            _run_from_weights(
                candidate={
                    "id": cid,
                    "label": str(rule["label"]),
                    "input_id": str(rule["input_id"]),
                    "rule_id": "EDGE_GATED_REPLACE_0_TO_3",
                },
                score_df=_score_for_groups(panel, replacement_weights[cid], str(rule["score_col"])).rename(
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

    leaderboard = _build_leaderboard(results, baseline_id=BASELINE_ID)
    for table in (leaderboard,):
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
    overlap_long = pd.concat(
        [_overlap_vs_baseline(w, defensive_weights, cid) for cid, w in replacement_weights.items()],
        ignore_index=True,
    )
    industry_exposure = pd.concat(
        [_industry_exposure(defensive_weights, industry_map, BASELINE_ID)]
        + [_industry_exposure(w, industry_map, cid) for cid, w in replacement_weights.items()],
        ignore_index=True,
    )
    year_improvement = _year_improvement_table(year_long, leaderboard)

    print("[8/8] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    leaderboard.to_csv(results_dir / f"{prefix}_leaderboard.csv", index=False, encoding="utf-8-sig")
    replacement_diag.to_csv(results_dir / f"{prefix}_replacement_diag_long.csv", index=False, encoding="utf-8-sig")
    replacement_count_dist.to_csv(results_dir / f"{prefix}_replacement_count_distribution.csv", index=False, encoding="utf-8-sig")
    selected_pairs.to_csv(results_dir / f"{prefix}_selected_pairs.csv", index=False, encoding="utf-8-sig")
    overlap_long.to_csv(results_dir / f"{prefix}_overlap_long.csv", index=False, encoding="utf-8-sig")
    industry_exposure.to_csv(results_dir / f"{prefix}_industry_exposure_long.csv", index=False, encoding="utf-8-sig")
    regime_long.to_csv(results_dir / f"{prefix}_regime_long.csv", index=False, encoding="utf-8-sig")
    breadth_long.to_csv(results_dir / f"{prefix}_breadth_long.csv", index=False, encoding="utf-8-sig")
    year_long.to_csv(results_dir / f"{prefix}_year_long.csv", index=False, encoding="utf-8-sig")
    year_improvement.to_csv(results_dir / f"{prefix}_year_strong_up_improvement.csv", index=False, encoding="utf-8-sig")
    switch_long.to_csv(results_dir / f"{prefix}_switch_long.csv", index=False, encoding="utf-8-sig")
    monthly_long.to_csv(results_dir / f"{prefix}_monthly_long.csv", index=False, encoding="utf-8-sig")
    state_by_rebalance.to_csv(results_dir / f"{prefix}_lagged_state_by_rebalance.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "defensive_core_edge_gated_replacement",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "state_lag": "previous_completed_month",
        "state_threshold_mode": "expanding",
        "default_horizon": int(args.default_horizon),
        "default_cost_buffer": float(args.default_cost_buffer),
        "max_replace": int(args.max_replace),
        "max_industry_names": int(args.max_industry_names),
        "max_limit_up_hits_20d": float(args.max_limit_up_hits_20d),
        "max_expansion": float(args.max_expansion),
        "max_candidates_per_pool": int(args.max_candidates_per_pool),
        "candidate_pool_pairs": [
            {"old_pool": c["old_pool"], "candidate_pool": c["candidate_pool"], "score_col": c["score_col"]}
            for c in U3_CANDIDATES
        ],
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
        "defensive_weight_diag_rows": int(len(defensive_diag)),
        "pair_base_rows": int(len(base_pairs)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "u3_candidates": U3_CANDIDATES,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "accept": accept_map,
        "replacement_count_distribution": replacement_count_dist.to_dict(orient="records"),
        "selected_pair_count": int(len(selected_pairs)),
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
        replacement_count_dist=replacement_count_dist,
        selected_pairs=selected_pairs,
        regime_long=regime_long,
        breadth_long=breadth_long,
        switch_long=switch_long,
        year_improvement=year_improvement,
        industry_exposure=industry_exposure,
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
        result_type="r2b_edge_gated_replacement",
        research_topic="r2b_edge_gated_replacement",
        research_config_id=f"r2b_edge_gated_{slugify_token(prefix)}",
        output_stem=prefix,
    )
    data_slice = DataSlice(
        dataset_name="r2b_edge_gated_backtest",
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
        feature_set_id="r2b_edge_gated_factors",
        feature_columns=(),
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=config_source,
        extra={"top_k": int(top_k), "max_turnover": float(max_turnover), "max_replace": int(args.max_replace)},
    )
    artifact_refs = (
        ArtifactRef("leaderboard_csv", _project_relative(results_dir / f"{prefix}_leaderboard.csv"), "csv", False, "leaderboard"),
        ArtifactRef("replacement_diag_long_csv", _project_relative(results_dir / f"{prefix}_replacement_diag_long.csv"), "csv", False, "替换诊断"),
        ArtifactRef("replacement_count_distribution_csv", _project_relative(results_dir / f"{prefix}_replacement_count_distribution.csv"), "csv", False, "替换分布"),
        ArtifactRef("selected_pairs_csv", _project_relative(results_dir / f"{prefix}_selected_pairs.csv"), "csv", False, "选中pair"),
        ArtifactRef("overlap_long_csv", _project_relative(results_dir / f"{prefix}_overlap_long.csv"), "csv", False, "重合度"),
        ArtifactRef("industry_exposure_long_csv", _project_relative(results_dir / f"{prefix}_industry_exposure_long.csv"), "csv", False, "行业暴露"),
        ArtifactRef("regime_long_csv", _project_relative(results_dir / f"{prefix}_regime_long.csv"), "csv", False, "状态长期"),
        ArtifactRef("breadth_long_csv", _project_relative(results_dir / f"{prefix}_breadth_long.csv"), "csv", False, "广度长期"),
        ArtifactRef("year_long_csv", _project_relative(results_dir / f"{prefix}_year_long.csv"), "csv", False, "年度长期"),
        ArtifactRef("switch_long_csv", _project_relative(results_dir / f"{prefix}_switch_long.csv"), "csv", False, "切换长期"),
        ArtifactRef("monthly_long_csv", _project_relative(results_dir / f"{prefix}_monthly_long.csv"), "csv", False, "月度长期"),
        ArtifactRef("summary_json", _project_relative(results_dir / f"{prefix}_summary.json"), "json", False, "汇总"),
        ArtifactRef("report_md", _project_relative(docs_dir / f"{prefix}.md"), "md", False, "报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    )
    metrics = {
        "candidate_count": int(len(leaderboard)),
        "selected_pair_count": int(len(selected_pairs)),
    }
    gates = {
        "data_gate": {"passed": bool(len(leaderboard) > 0)},
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
        promotion={"production_eligible": False, "registry_status": "not_registered", "blocking_reasons": ["r2b_is_research_only"]},
        notes="R2B edge-gated replacement experiment; not a promotion candidate.",
    )
    write_research_manifest(manifest_path, result)
    append_experiment_result(PROJECT_ROOT / "data" / "experiments", result)
    # --- end standard research contract ---


if __name__ == "__main__":
    main()
