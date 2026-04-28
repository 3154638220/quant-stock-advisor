"""R2B v2 gray-zone weight audit.

This script does not create a new production candidate. It audits whether the
R2B v2 U3_A gray-zone result is supported by stable pair-level evidence, simple
feature directionality, and robust narrow-rule sensitivity.
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
from scripts.run_p1_strong_up_attribution import _json_sanitize, compute_breadth, compute_r1_extra_features  # noqa: E402
from scripts.run_p2_regime_aware_dual_sleeve_v1 import (  # noqa: E402
    _attach_pit_roe_ttm,
    _compute_minimal_defensive_factors,
    _filter_universe,
    _lagged_state_by_rebalance,
    _monthly_benchmark_frame,
)
from scripts.run_r2b_edge_gated_replacement_v2 import (  # noqa: E402
    U3_CANDIDATES,
    add_pair_edge_score,
    add_u3_scores,
    build_edge_gated_weights,
    select_edge_gated_replacements,
)
from scripts.run_r2b_oracle_replacement_attribution import (  # noqa: E402
    _with_edge,
    build_forward_returns_from_open,
    build_oracle_pair_base,
)
from scripts.run_r2b_tradable_upside_replacement_v1 import (  # noqa: E402
    BASELINE_ID,
    EVAL_CONTRACT_VERSION,
    EXECUTION_CONTRACT_VERSION,
    _industry_source_status,
    _load_or_build_industry_map,
    _run_from_weights,
    _score_for_groups,
    _add_basic_r2b_features,
    build_r2b_scores,
)
from src.models.xtree.p1_workflow import build_tree_score_weight_matrix  # noqa: E402


DEFAULT_OUTPUT_PREFIX = "r2b_v2_weight_audit_2026-04-28"
TARGET_ID = "U3_A_real_industry_leadership__EDGE_GATED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R2B v2 gray-zone weight audit")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--industry-map", default="data/cache/industry_map.csv")
    p.add_argument("--default-horizon", type=int, default=20)
    p.add_argument("--default-cost-buffer", type=float, default=0.0015)
    p.add_argument("--max-industry-names", type=int, default=5)
    p.add_argument("--max-limit-up-hits-20d", type=float, default=2.0)
    p.add_argument("--max-expansion", type=float, default=1.50)
    p.add_argument("--max-candidates-per-pool", type=int, default=0)
    return p.parse_args()


def _target_rule() -> dict[str, Any]:
    return next(c for c in U3_CANDIDATES if c["id"] == TARGET_ID)


def _filter_rule_base(pairs: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    df = pairs[
        (pairs["score_col"].astype(str) == str(rule["score_col"]))
        & (pairs["old_pool"].astype(str) == str(rule["old_pool"]))
        & (pairs["candidate_pool"].astype(str) == str(rule["candidate_pool"]))
    ].copy()
    gate = str(rule["state_gate"])
    if gate in df.columns:
        df = df[df[gate].astype(bool)].copy()
    return df


def _feature_bucket_table(pairs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        vals = pd.to_numeric(pairs[feature], errors="coerce")
        valid = pairs[np.isfinite(vals)].copy()
        if len(valid) < 50 or valid[feature].nunique(dropna=True) < 5:
            continue
        try:
            valid["bucket"] = pd.qcut(pd.to_numeric(valid[feature], errors="coerce"), 5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for bucket, part in valid.groupby("bucket", dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "bucket": int(bucket),
                    "pair_count": int(len(part)),
                    "feature_min": float(pd.to_numeric(part[feature], errors="coerce").min()),
                    "feature_max": float(pd.to_numeric(part[feature], errors="coerce").max()),
                    "replace_win_rate": float(pd.to_numeric(part["replace_win"], errors="coerce").mean()),
                    "mean_pair_edge": float(pd.to_numeric(part["pair_edge"], errors="coerce").mean()),
                    "median_pair_edge": float(pd.to_numeric(part["pair_edge"], errors="coerce").median()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mono = []
    for feature, sub in out.groupby("feature", dropna=False):
        ordered = sub.sort_values("bucket")
        mono.append(
            {
                "feature": feature,
                "bucket_edge_spearman": float(ordered["bucket"].corr(ordered["mean_pair_edge"], method="spearman"))
                if len(ordered) >= 3
                else np.nan,
                "bucket_win_spearman": float(ordered["bucket"].corr(ordered["replace_win_rate"], method="spearman"))
                if len(ordered) >= 3
                else np.nan,
            }
        )
    return out.merge(pd.DataFrame(mono), on="feature", how="left")


def _add_selected_order(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    out = selected.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["replacement_order"] = out.groupby("trade_date", sort=False).cumcount() + 1
    return out


def _selected_pair_slices(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    df = _add_selected_order(selected)
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    by_year = (
        df.groupby("year", dropna=False)
        .agg(
            selected_pairs=("trade_date", "count"),
            realized_win_rate=("replace_win", "mean"),
            avg_realized_pair_edge=("pair_edge", "mean"),
            median_realized_pair_edge=("pair_edge", "median"),
            avg_pair_edge_score=("pair_edge_score", "mean"),
        )
        .reset_index()
    )
    by_slot = (
        df.groupby("replacement_order", dropna=False)
        .agg(
            selected_pairs=("trade_date", "count"),
            realized_win_rate=("replace_win", "mean"),
            avg_realized_pair_edge=("pair_edge", "mean"),
            median_realized_pair_edge=("pair_edge", "median"),
            avg_pair_edge_score=("pair_edge_score", "mean"),
        )
        .reset_index()
    )
    by_industry = (
        df.groupby("new_industry", dropna=False)
        .agg(
            selected_pairs=("trade_date", "count"),
            realized_win_rate=("replace_win", "mean"),
            avg_realized_pair_edge=("pair_edge", "mean"),
            median_realized_pair_edge=("pair_edge", "median"),
            avg_pair_edge_score=("pair_edge_score", "mean"),
        )
        .reset_index()
        .sort_values(["selected_pairs", "avg_realized_pair_edge"], ascending=[False, False])
    )
    return by_year, by_slot, by_industry


def _monthly_attribution(monthly_long: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monthly_long.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = monthly_long.copy()
    m["month_end"] = pd.to_datetime(m["month_end"], errors="coerce").dt.normalize()
    pivot = m.pivot_table(index="month_end", columns="candidate_id", values="excess_return", aggfunc="first")
    if BASELINE_ID not in pivot.columns or TARGET_ID not in pivot.columns:
        return pd.DataFrame(), pd.DataFrame()
    out = pivot[[BASELINE_ID, TARGET_ID]].rename(columns={BASELINE_ID: "baseline_excess", TARGET_ID: "target_excess"}).reset_index()
    out["monthly_delta_vs_baseline"] = out["target_excess"] - out["baseline_excess"]
    meta_cols = ["month_end", "regime", "breadth", "benchmark_return"]
    meta = m[m["candidate_id"] == TARGET_ID][[c for c in meta_cols if c in m.columns]].drop_duplicates("month_end")
    out = out.merge(meta, on="month_end", how="left")
    if selected.empty:
        out["active_replacement_month"] = False
        out["selected_pairs"] = 0
    else:
        s = selected.copy()
        month_ends = sorted(out["month_end"].dropna().unique().tolist())
        def _map_month_end(x: Any) -> pd.Timestamp | pd.NaT:
            ts = pd.Timestamp(x).normalize()
            pos = pd.Index(month_ends).searchsorted(ts, side="left")
            if pos >= len(month_ends):
                return pd.NaT
            return pd.Timestamp(month_ends[pos]).normalize()

        s["month_end"] = pd.to_datetime(s["trade_date"], errors="coerce").map(_map_month_end)
        counts = s.dropna(subset=["month_end"]).groupby("month_end").size().rename("selected_pairs").reset_index()
        out = out.merge(counts, on="month_end", how="left")
        out["selected_pairs"] = out["selected_pairs"].fillna(0).astype(int)
        out["active_replacement_month"] = out["selected_pairs"] > 0
    summary = (
        out.groupby(["active_replacement_month", "regime", "breadth"], dropna=False)
        .agg(
            months=("month_end", "count"),
            avg_selected_pairs=("selected_pairs", "mean"),
            mean_monthly_delta=("monthly_delta_vs_baseline", "mean"),
            median_monthly_delta=("monthly_delta_vs_baseline", "median"),
            positive_delta_share=("monthly_delta_vs_baseline", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["active_replacement_month", "regime", "breadth"])
    )
    return out.sort_values("monthly_delta_vs_baseline", ascending=False), summary


def _run_strategy_variant(
    *,
    rule: dict[str, Any],
    pairs: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    panel: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    breadth_series: pd.Series,
    cost_params: Any,
    limit_mask: pd.DataFrame,
    max_replace: int,
    max_industry_names: int,
    candidate_id: str,
) -> dict[str, Any]:
    selected = select_edge_gated_replacements(
        pairs,
        defensive_weights=defensive_weights,
        panel=panel,
        rule=rule,
        max_replace=max_replace,
        max_industry_names=max_industry_names,
    )
    weights, diag = build_edge_gated_weights(defensive_weights=defensive_weights, selected_pairs=selected, rule={**rule, "id": candidate_id})
    weights = weights.reindex(columns=asset_returns.columns, fill_value=0.0)
    result = _run_from_weights(
        candidate={"id": candidate_id, "label": str(rule.get("label", "")), "input_id": str(rule.get("input_id", ""))},
        score_df=_score_for_groups(panel, weights, str(rule["score_col"])).rename(columns={"score": "candidate_score"}),
        weights=weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        breadth_series=breadth_series,
        cost_params=cost_params,
        limit_up_open_mask=limit_mask,
    )
    return {"result": result, "selected": _add_selected_order(selected), "diag": diag}


def _regime_metric(result: dict[str, Any], regime: str) -> dict[str, float]:
    rc = result["regime_capture"]
    sub = rc[rc["regime"] == regime] if not rc.empty else pd.DataFrame()
    if sub.empty:
        return {"median_excess": np.nan, "positive_share": np.nan}
    row = sub.iloc[0]
    return {
        "median_excess": float(row["median_excess_return"]),
        "positive_share": float(row["positive_excess_share"]),
    }


def _variant_table(variant_runs: list[dict[str, Any]], base_result: dict[str, Any]) -> pd.DataFrame:
    base_proxy = float(base_result["summary"].get("annualized_excess_vs_market", np.nan))
    base_su = _regime_metric(base_result, "strong_up")
    rows: list[dict[str, Any]] = []
    for item in variant_runs:
        result = item["result"]
        selected = item["selected"]
        su = _regime_metric(result, "strong_up")
        proxy = float(result["summary"].get("annualized_excess_vs_market", np.nan))
        rows.append(
            {
                "candidate_id": result["candidate"]["id"],
                "daily_proxy_annualized_excess_vs_market": proxy,
                "delta_vs_baseline_proxy": proxy - base_proxy,
                "strong_up_median_excess": su["median_excess"],
                "delta_vs_baseline_strong_up_median_excess": su["median_excess"] - base_su["median_excess"],
                "strong_up_positive_share": su["positive_share"],
                "delta_vs_baseline_strong_up_positive_share": su["positive_share"] - base_su["positive_share"],
                "selected_pairs": int(len(selected)),
                "active_rebalance_share": float(selected["trade_date"].nunique() / max(int(result["meta"].get("n_rebalances", 0)), 1))
                if not selected.empty
                else 0.0,
                "avg_realized_pair_edge": float(pd.to_numeric(selected.get("pair_edge"), errors="coerce").mean())
                if not selected.empty
                else np.nan,
                "realized_win_rate": float(pd.to_numeric(selected.get("replace_win"), errors="coerce").mean())
                if not selected.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("daily_proxy_annualized_excess_vs_market", ascending=False)


def _simple_score_pairs(base: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = base.copy()
    vals = pd.to_numeric(out[score_col], errors="coerce")
    out["pair_edge_score"] = vals.groupby(out["trade_date"], dropna=False).rank(pct=True, method="average")
    out["pair_edge_score"] = out["pair_edge_score"].fillna(0.0).clip(0.0, 1.0)
    return out


def _build_doc(
    *,
    params: dict[str, Any],
    monthly_attr: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    feature_buckets: pd.DataFrame,
    selected_by_year: pd.DataFrame,
    selected_by_slot: pd.DataFrame,
    selected_by_industry: pd.DataFrame,
    sensitivity: pd.DataFrame,
    simple_baselines: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R2B v2 Weight Audit\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- `eval_contract_version`: `{EVAL_CONTRACT_VERSION}`")
    lines.append(f"- `execution_contract_version`: `{EXECUTION_CONTRACT_VERSION}`")
    lines.append(f"- `industry_map_source_status`: `{params.get('industry_map_source_status')}`")
    lines.append(f"- 目标：审计 `U3_A` gray zone 是否有稳定证据，不产生 production candidate。")
    lines.append("")

    lines.append("## 1. U3_A 月度归因\n")
    view = monthly_attr.head(12).copy()
    cols = ["month_end", "regime", "breadth", "baseline_excess", "target_excess", "monthly_delta_vs_baseline", "selected_pairs"]
    lines.append(view[[c for c in cols if c in view.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(monthly_summary.to_markdown(index=False, floatfmt=".4f") if not monthly_summary.empty else "_无月度归因_")
    lines.append("")

    lines.append("## 2. Selected Pair Slices\n")
    lines.append("### By Year\n")
    lines.append(selected_by_year.to_markdown(index=False, floatfmt=".4f") if not selected_by_year.empty else "_无 selected pairs_")
    lines.append("")
    lines.append("### By Replacement Slot\n")
    lines.append(selected_by_slot.to_markdown(index=False, floatfmt=".4f") if not selected_by_slot.empty else "_无 selected pairs_")
    lines.append("")
    lines.append("### By New Industry\n")
    lines.append(selected_by_industry.head(15).to_markdown(index=False, floatfmt=".4f") if not selected_by_industry.empty else "_无 selected pairs_")
    lines.append("")

    lines.append("## 3. Feature Bucket Monotonicity\n")
    fb_cols = [
        "feature",
        "bucket",
        "pair_count",
        "replace_win_rate",
        "mean_pair_edge",
        "bucket_edge_spearman",
        "bucket_win_spearman",
    ]
    lines.append(feature_buckets[[c for c in fb_cols if c in feature_buckets.columns]].to_markdown(index=False, floatfmt=".4f") if not feature_buckets.empty else "_无 feature bucket_")
    lines.append("")

    lines.append("## 4. Threshold / Capacity Sensitivity\n")
    lines.append(sensitivity.to_markdown(index=False, floatfmt=".4f") if not sensitivity.empty else "_无 sensitivity_")
    lines.append("")

    lines.append("## 5. Simple Baseline Comparison\n")
    lines.append(simple_baselines.to_markdown(index=False, floatfmt=".4f") if not simple_baselines.empty else "_无 simple baseline_")
    lines.append("")

    lines.append("## 6. 判定\n")
    if sensitivity.empty:
        lines.append("- 无法完成 sensitivity，维持 R2B v2 gray-zone 待复核。")
    else:
        best = sensitivity.iloc[0]
        base_case = sensitivity[sensitivity["candidate_id"].astype(str).str.contains("thr0.68_replace3", regex=False)]
        base_delta = float(base_case["delta_vs_baseline_proxy"].iloc[0]) if not base_case.empty else np.nan
        best_delta = float(best["delta_vs_baseline_proxy"])
        slot1 = sensitivity[sensitivity["candidate_id"].astype(str).str.contains("replace1", regex=False)]
        slot1_best = float(slot1["delta_vs_baseline_proxy"].max()) if not slot1.empty else np.nan
        if best_delta < 0.0:
            lines.append("- 所有 U3_A 窄规则 sensitivity 都低于 baseline，建议停止 R2B/R3 replacement 主线。")
        elif np.isfinite(slot1_best) and slot1_best >= max(base_delta, 0.0):
            lines.append("- `replace-1` 不弱于原 `replace-3`，说明 edge 主要集中在最高置信度 slot；若继续，应先做 R2B v2.1 replace-1，而不是复杂 R3。")
        else:
            lines.append("- U3_A gray-zone 仍弱，当前不足以启动复杂 R3；最多继续做窄规则复核。")
    lines.append("- 本轮 audit 仍不产生 production candidate，不补正式 full backtest，不写默认配置。")
    lines.append("")

    lines.append("## 7. 产出文件\n")
    for suffix in [
        "monthly_attribution.csv",
        "monthly_attribution_summary.csv",
        "feature_bucket_monotonicity.csv",
        "selected_pairs_by_year.csv",
        "selected_pairs_by_slot.csv",
        "selected_pairs_by_industry.csv",
        "threshold_capacity_sensitivity.csv",
        "simple_baseline_comparison.csv",
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
    rule = _target_rule()

    print(f"[1/8] load daily {args.start}->{end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)

    print("[2/8] compute U3 panel", flush=True)
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

    print("[3/8] returns + state", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    limit_up_open_mask = build_limit_up_open_mask(daily_df).sort_index()
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_state = _monthly_benchmark_frame(bench_daily, breadth_series)

    print("[4/8] baseline weights + pair base", flush=True)
    score_base = panel[["symbol", "trade_date", "score__defensive"]].dropna().copy()
    defensive_weights, defensive_diag = build_tree_score_weight_matrix(
        score_base,
        score_col="score__defensive",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    state_by_rebalance = _lagged_state_by_rebalance(defensive_weights.index, monthly_state)
    forward_by_horizon = {int(args.default_horizon): build_forward_returns_from_open(daily_df, horizon=int(args.default_horizon))}
    base_pairs = build_oracle_pair_base(
        panel=panel,
        defensive_weights=defensive_weights,
        state_by_rebalance=state_by_rebalance,
        trading_index=open_returns.index,
        limit_up_open_mask=limit_up_open_mask,
        forward_by_horizon=forward_by_horizon,
        score_cols=[str(rule["score_col"])],
        old_pool_sizes=[int(str(rule["old_pool"]).split("_")[-1])],
        candidate_pcts=[int(str(rule["candidate_pool"]).split("_")[-1]) / 100.0],
        max_limit_up_hits_20d=float(args.max_limit_up_hits_20d),
        max_expansion=float(args.max_expansion),
        max_candidates_per_pool=int(args.max_candidates_per_pool),
    )
    base_pairs = add_pair_edge_score(_with_edge(base_pairs, horizon=int(args.default_horizon), cost_buffer=float(args.default_cost_buffer)))
    rule_pairs = _filter_rule_base(base_pairs, rule)
    print(f"  pair_base_rows={len(base_pairs):,}; target_rule_pairs={len(rule_pairs):,}", flush=True)

    print("[5/8] selected pair and feature audit", flush=True)
    selected = select_edge_gated_replacements(
        base_pairs,
        defensive_weights=defensive_weights,
        panel=panel,
        rule=rule,
        max_replace=3,
        max_industry_names=int(args.max_industry_names),
    )
    selected = _add_selected_order(selected)
    by_year, by_slot, by_industry = _selected_pair_slices(selected)
    features = [
        "candidate_score_pct",
        "score_margin",
        "pair_edge_score",
        "rel_strength_diff",
        "amount_expansion_diff",
        "turnover_expansion_diff",
        "overheat_diff",
        "old_defensive_score",
    ]
    feature_buckets = _feature_bucket_table(rule_pairs, features)

    print("[6/8] strategy sensitivity", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()) | set(defensive_weights.columns.astype(str)))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    limit_mask = limit_up_open_mask.reindex(columns=sym_universe, fill_value=False)
    defensive_weights = defensive_weights.reindex(columns=sym_universe, fill_value=0.0)

    base_result = _run_from_weights(
        candidate={"id": BASELINE_ID, "label": "fixed S2 defensive Top-20 baseline"},
        score_df=score_base.rename(columns={"score__defensive": "candidate_score"}),
        weights=defensive_weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        breadth_series=breadth_series,
        cost_params=cost_params,
        limit_up_open_mask=limit_mask,
    )
    variant_runs: list[dict[str, Any]] = []
    for threshold in (0.68, 0.75, 0.80, 0.85):
        for max_replace in (1, 2, 3):
            vrule = {**rule, "edge_threshold": threshold}
            cid = f"U3_A_audit_thr{threshold:.2f}_replace{max_replace}"
            variant_runs.append(
                _run_strategy_variant(
                    rule=vrule,
                    pairs=base_pairs,
                    defensive_weights=defensive_weights,
                    panel=panel,
                    asset_returns=asset_returns,
                    bench_daily=bench_daily,
                    breadth_series=breadth_series,
                    cost_params=cost_params,
                    limit_mask=limit_mask,
                    max_replace=max_replace,
                    max_industry_names=int(args.max_industry_names),
                    candidate_id=cid,
                )
            )
    sensitivity = _variant_table(variant_runs, base_result)

    print("[7/8] simple baselines", flush=True)
    simple_runs = []
    simple_specs = [
        ("candidate_score_pct_only", "candidate_score_pct", 0.95),
        ("score_margin_only", "score_margin_rank", 0.80),
        ("rel_strength_diff_only", "rel_strength_diff_rank", 0.80),
        ("amount_expansion_diff_only", "amount_expansion_diff_rank", 0.80),
        ("overheat_relief_only", "overheat_relief_rank", 0.80),
    ]
    for name, col, threshold in simple_specs:
        simple_pairs = _simple_score_pairs(base_pairs, col)
        srule = {**rule, "id": f"U3_A_simple_{name}", "edge_threshold": threshold, "min_score_margin": 0.0}
        simple_runs.append(
            _run_strategy_variant(
                rule=srule,
                pairs=simple_pairs,
                defensive_weights=defensive_weights,
                panel=panel,
                asset_returns=asset_returns,
                bench_daily=bench_daily,
                breadth_series=breadth_series,
                cost_params=cost_params,
                limit_mask=limit_mask,
                max_replace=3,
                max_industry_names=int(args.max_industry_names),
                candidate_id=f"U3_A_simple_{name}",
            )
        )
    simple_baselines = _variant_table(simple_runs, base_result)

    print("[8/8] write outputs", flush=True)
    monthly_long_path = PROJECT_ROOT / "data" / "results" / "r2b_edge_gated_replacement_v2_2026-04-28_monthly_long.csv"
    monthly_long = pd.read_csv(monthly_long_path) if monthly_long_path.exists() else pd.DataFrame()
    monthly_attr, monthly_summary = _monthly_attribution(monthly_long, selected)

    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    monthly_attr.to_csv(results_dir / f"{prefix}_monthly_attribution.csv", index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(results_dir / f"{prefix}_monthly_attribution_summary.csv", index=False, encoding="utf-8-sig")
    feature_buckets.to_csv(results_dir / f"{prefix}_feature_bucket_monotonicity.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(results_dir / f"{prefix}_selected_pairs_by_year.csv", index=False, encoding="utf-8-sig")
    by_slot.to_csv(results_dir / f"{prefix}_selected_pairs_by_slot.csv", index=False, encoding="utf-8-sig")
    by_industry.to_csv(results_dir / f"{prefix}_selected_pairs_by_industry.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(results_dir / f"{prefix}_threshold_capacity_sensitivity.csv", index=False, encoding="utf-8-sig")
    simple_baselines.to_csv(results_dir / f"{prefix}_simple_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "target_candidate": TARGET_ID,
        "config_source": config_source,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "max_turnover": max_turnover,
        "default_horizon": int(args.default_horizon),
        "default_cost_buffer": float(args.default_cost_buffer),
        "max_industry_names": int(args.max_industry_names),
        "industry_map_source": industry_source,
        "industry_map_source_status": industry_status,
        "industry_alpha_evidence_allowed": industry_status == "real_industry_map",
        "prefilter": prefilter_cfg,
        "universe_filter": uf_cfg,
        "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
        "benchmark_min_history_days": bench_min,
        "primary_benchmark_return_mode": "open_to_open",
        "comparison_benchmark_return_mode": "close_to_close",
        "eval_contract_version": EVAL_CONTRACT_VERSION,
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "pair_base_rows": int(len(base_pairs)),
        "target_rule_pair_rows": int(len(rule_pairs)),
        "selected_pairs": int(len(selected)),
        "defensive_weight_diag_rows": int(len(defensive_diag)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "parameters": params,
        "selected_pairs_by_year": by_year.to_dict(orient="records"),
        "selected_pairs_by_slot": by_slot.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
        "simple_baselines": simple_baselines.to_dict(orient="records"),
        "feature_bucket_monotonicity": feature_buckets.to_dict(orient="records"),
    }
    with open(results_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    doc = _build_doc(
        params=params,
        monthly_attr=monthly_attr,
        monthly_summary=monthly_summary,
        feature_buckets=feature_buckets,
        selected_by_year=by_year,
        selected_by_slot=by_slot,
        selected_by_industry=by_industry,
        sensitivity=sensitivity,
        simple_baselines=simple_baselines,
        output_prefix=prefix,
    )
    (docs_dir / f"{prefix}.md").write_text(doc, encoding="utf-8")
    print(f"  doc -> {docs_dir / f'{prefix}.md'}", flush=True)


if __name__ == "__main__":
    main()
