"""R1F Fixed Baseline minimal rerun.

Runs only the three R1F sanity-check candidates after the R0 execution/eval
contract fixes:

    BASELINE_S2_FIXED
    UPSIDE_C_FIXED
    DUAL_V1_FIXED

The goal is not to extend the search grid. It is a narrow before/after check
against the archived pre-fix R2 outputs.
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
    _blend_weights,
    _build_leaderboard,
    _compute_minimal_defensive_factors,
    _detail_from_weights,
    _filter_universe,
    _lagged_state_by_rebalance,
    _monthly_benchmark_frame,
    _monthly_with_regime,
    _score_for_switch_groups,
    _attach_pit_roe_ttm,
    build_sleeve_scores,
)
from scripts.run_p2_upside_sleeve_v1 import (  # noqa: E402
    GATE_FULL_BACKTEST,
    GATE_REJECT,
    _accept_summary,
)
from src.models.xtree.p1_workflow import (  # noqa: E402
    build_tree_score_weight_matrix,
    summarize_tree_daily_backtest_like_proxy,
)


EVAL_CONTRACT_VERSION = "r0_eval_execution_contract_2026-04-28"
EXECUTION_CONTRACT_VERSION = "tplus1_open_buy_delta_limit_mask_2026-04-28"

BASELINE = {
    "id": "BASELINE_S2_FIXED",
    "label": "fixed S2 vol_to_turnover defensive baseline",
}
UPSIDE_C = {
    "id": "UPSIDE_C_FIXED",
    "label": "fixed UPSIDE_C = pure limit_up_hits_20d + tail_strength_20d",
}
DUAL_V1 = {
    "id": "DUAL_V1_FIXED",
    "label": "fixed lagged strong_up/wide: defensive 80% + upside 20%; otherwise defensive 100%",
    "strong_or_wide_upside": 0.20,
    "mild_or_mid_upside": 0.0,
    "neutral_upside": 0.0,
}

COMPARISON_MAP = {
    "BASELINE_S2_FIXED": {
        "old_file": "data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_leaderboard.csv",
        "old_candidate_id": "BASELINE_S2",
        "old_label": "pre_fix_dual_BASELINE_S2",
    },
    "UPSIDE_C_FIXED": {
        "old_file": "data/results/p2_upside_sleeve_v1_2026-04-27_leaderboard.csv",
        "old_candidate_id": "UPSIDE_C_limitup_tail",
        "old_label": "pre_fix_UPSIDE_C_limitup_tail",
    },
    "DUAL_V1_FIXED": {
        "old_file": "data/results/p2_regime_aware_dual_sleeve_v1_2026-04-28_leaderboard.csv",
        "old_candidate_id": "DUAL_V1_80_20_TRIGGER_ONLY",
        "old_label": "pre_fix_DUAL_V1_80_20_TRIGGER_ONLY",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1F fixed baseline minimal rerun")
    p.add_argument("--config", default="config.yaml.backtest")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--lookback-days", type=int, default=320)
    p.add_argument("--min-hist-days", type=int, default=130)
    p.add_argument("--output-prefix", default="r1f_fixed_baseline_rerun_2026-04-28")
    return p.parse_args()


def _long_with_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "candidate_id", cid)
    return out


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
        "state_diag": pd.DataFrame(),
    }


def _run_dual_v1(
    *,
    panel: pd.DataFrame,
    defensive_weights: pd.DataFrame,
    upside_weights: pd.DataFrame,
    state_by_rebalance: pd.DataFrame,
    asset_returns: pd.DataFrame,
    bench_daily: pd.Series,
    breadth_series: pd.Series,
    cost_params: Any,
    limit_up_open_mask: pd.DataFrame,
) -> dict[str, Any]:
    weights, state_diag = _blend_weights(defensive_weights, upside_weights, state_by_rebalance, DUAL_V1)
    score_df = _score_for_switch_groups(panel, state_by_rebalance, DUAL_V1).rename(
        columns={"score": "candidate_score"}
    )
    out = _run_from_weights(
        candidate=DUAL_V1,
        score_df=score_df,
        weights=weights,
        asset_returns=asset_returns,
        bench_daily=bench_daily,
        breadth_series=breadth_series,
        cost_params=cost_params,
        limit_up_open_mask=limit_up_open_mask,
    )
    out["state_diag"] = state_diag
    return out


def _load_old_row(path: Path, candidate_id: str) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "candidate_id" not in df.columns:
        return None
    sub = df[df["candidate_id"].astype(str) == str(candidate_id)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _build_comparison(leaderboard: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    meta_by_id = {str(r["candidate"]["id"]): r["meta"] for r in results}
    metrics = [
        "daily_proxy_annualized_excess_vs_market",
        "strong_up_median_excess",
        "strong_down_median_excess",
        "strong_up_switch_in_minus_out",
        "avg_turnover_half_l1",
    ]
    rows: list[dict[str, Any]] = []
    for cid, info in COMPARISON_MAP.items():
        fixed = leaderboard[leaderboard["candidate_id"].astype(str) == cid]
        if fixed.empty:
            continue
        fixed_row = fixed.iloc[0]
        old_row = _load_old_row(PROJECT_ROOT / str(info["old_file"]), str(info["old_candidate_id"]))
        row: dict[str, Any] = {
            "candidate_id": cid,
            "old_label": info["old_label"],
            "old_candidate_id": info["old_candidate_id"],
            "old_source_file": info["old_file"],
        }
        for metric in metrics:
            old_val = float(old_row[metric]) if old_row is not None and metric in old_row else np.nan
            fixed_val = float(fixed_row[metric]) if metric in fixed_row else np.nan
            row[f"old_{metric}"] = old_val
            row[f"fixed_{metric}"] = fixed_val
            row[f"delta_{metric}"] = fixed_val - old_val if np.isfinite(old_val) and np.isfinite(fixed_val) else np.nan
        meta = meta_by_id.get(cid, {})
        row["fixed_gate_decision"] = str(fixed_row.get("gate_decision", ""))
        row["fixed_buy_fail_event_count"] = int(meta.get("buy_fail_event_count", 0))
        row["fixed_buy_fail_total_weight"] = float(meta.get("buy_fail_total_weight", 0.0))
        row["fixed_buy_fail_redistributed_weight"] = float(meta.get("buy_fail_redistributed_weight", 0.0))
        row["fixed_buy_fail_idle_weight"] = float(meta.get("buy_fail_idle_weight", 0.0))
        row["fixed_limit_up_detection"] = str(meta.get("limit_up_detection", ""))
        row["fixed_primary_benchmark_return_mode"] = str(meta.get("primary_benchmark_return_mode", ""))
        row["fixed_comparison_benchmark_return_mode"] = str(meta.get("comparison_benchmark_return_mode", ""))
        row["old_buy_fail_fields_available"] = False
        row["old_benchmark_mode_fields_available"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _build_doc(
    *,
    config_source: str,
    params: dict[str, Any],
    leaderboard: pd.DataFrame,
    comparison: pd.DataFrame,
    accept_map: dict[str, dict[str, Any]],
    regime_long: pd.DataFrame,
    switch_long: pd.DataFrame,
    output_prefix: str,
) -> str:
    lines: list[str] = []
    lines.append("# R1F Fixed Baseline 最小复跑\n")
    lines.append(f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`")
    lines.append(f"- 配置快照：`{config_source}`")
    lines.append(f"- `eval_contract_version`: `{EVAL_CONTRACT_VERSION}`")
    lines.append(f"- `execution_contract_version`: `{EXECUTION_CONTRACT_VERSION}`")
    lines.append("- 固定候选：`BASELINE_S2_FIXED` / `UPSIDE_C_FIXED` / `DUAL_V1_FIXED`")
    lines.append("- 固定口径：`top_k=20` / `M` / `equal_weight` / `max_turnover=1.0` / `tplus1_open`")
    lines.append("- primary benchmark：`open_to_open`；comparison benchmark：`close_to_close`")
    lines.append("")

    lines.append("## 1. Fixed Leaderboard\n")
    cols_show = [
        "candidate_id",
        "daily_proxy_annualized_excess_vs_market",
        "delta_vs_baseline_proxy",
        "gate_decision",
        "strong_up_median_excess",
        "delta_vs_baseline_strong_up_median_excess",
        "strong_up_positive_share",
        "strong_down_median_excess",
        "delta_vs_baseline_strong_down_median_excess",
        "strong_up_switch_in_minus_out",
        "avg_turnover_half_l1",
        "n_rebalances",
    ]
    lines.append(leaderboard[[c for c in cols_show if c in leaderboard.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 2. 修复前后对照\n")
    compare_cols = [
        "candidate_id",
        "old_daily_proxy_annualized_excess_vs_market",
        "fixed_daily_proxy_annualized_excess_vs_market",
        "delta_daily_proxy_annualized_excess_vs_market",
        "old_strong_up_median_excess",
        "fixed_strong_up_median_excess",
        "delta_strong_up_median_excess",
        "old_strong_down_median_excess",
        "fixed_strong_down_median_excess",
        "delta_strong_down_median_excess",
        "old_strong_up_switch_in_minus_out",
        "fixed_strong_up_switch_in_minus_out",
        "delta_strong_up_switch_in_minus_out",
        "fixed_buy_fail_total_weight",
        "fixed_primary_benchmark_return_mode",
    ]
    lines.append(comparison[[c for c in compare_cols if c in comparison.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 3. R1F 判定\n")
    accept_rows = []
    for cid, info in accept_map.items():
        row = {"candidate_id": cid, "status": info["status"]}
        row.update(info.get("checks", {}))
        accept_rows.append(row)
    lines.append(pd.DataFrame(accept_rows).to_markdown(index=False))
    lines.append("")

    non_baseline = leaderboard[leaderboard["candidate_id"] != "BASELINE_S2_FIXED"].copy()
    non_reject = non_baseline[non_baseline["gate_decision"].astype(str) != "reject"]["candidate_id"].tolist()
    if non_reject:
        lines.append(f"- 非 reject 候选：{', '.join(non_reject)}。")
    else:
        lines.append("- `UPSIDE_C_FIXED` 与 `DUAL_V1_FIXED` 仍为 reject，旧 R2 失败结论成立。")
    full_bt = non_baseline[non_baseline["gate_decision"].astype(str) == "full_backtest_candidate"][
        "candidate_id"
    ].tolist()
    if full_bt:
        lines.append(f"- daily proxy `>= +3%` 候选：{', '.join(full_bt)}，可补正式 full backtest。")
    else:
        lines.append("- 无候选达到 `>= +3%`，不补正式 full backtest。")
    lines.append("- 后续应进入 R2B：重做更可交易的 upside 输入，而不是扩大 dual sleeve 权重网格。")
    lines.append("")

    lines.append("## 4. Regime 切片\n")
    lines.append(regime_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 5. Switch quality\n")
    lines.append("_无 switch 样本_" if switch_long.empty else switch_long.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## 6. 产出文件\n")
    for suf in [
        "leaderboard.csv",
        "comparison.csv",
        "regime_long.csv",
        "breadth_long.csv",
        "year_long.csv",
        "switch_long.csv",
        "monthly_long.csv",
        "state_diag_long.csv",
        "lagged_state_by_rebalance.csv",
        "summary.json",
    ]:
        lines.append(f"- `data/results/{output_prefix}_{suf}`")
    lines.append("")

    lines.append("## 7. 配置参数\n")
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

    print(f"[1/7] load daily {args.start}->{end_date}", flush=True)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)

    print("[2/7] compute minimal S2 + R1 extras + universe filter", flush=True)
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
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start)].copy()
    panel = _filter_universe(panel)
    panel = build_sleeve_scores(panel)
    print(f"  panel(after filter)={panel.shape}", flush=True)

    print("[3/7] precompute open-to-open returns + benchmark + state", flush=True)
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    n_trade_days = int(open_returns.index.nunique())
    bench_min = max(60, int(0.35 * max(n_trade_days, 1)))
    bench_daily = build_market_ew_open_to_open_benchmark(daily_df, args.start, end_date, min_days=bench_min)
    limit_up_open_mask = build_limit_up_open_mask(daily_df).sort_index()
    sym_counts = daily_df.groupby("symbol")["trade_date"].count()
    benchmark_symbols = set(sym_counts[sym_counts >= bench_min].index.astype(str))
    breadth_series = compute_breadth(daily_df, benchmark_symbols)
    monthly_state = _monthly_benchmark_frame(bench_daily, breadth_series)

    print("[4/7] build defensive/upside weights", flush=True)
    score_base = panel[["symbol", "trade_date", "score__defensive"]].dropna().copy()
    score_upside = panel[["symbol", "trade_date", "score__upside_c"]].dropna().copy()
    defensive_weights, defensive_diag = build_tree_score_weight_matrix(
        score_base,
        score_col="score__defensive",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    upside_weights, upside_diag = build_tree_score_weight_matrix(
        score_upside,
        score_col="score__upside_c",
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        max_turnover=max_turnover,
    )
    state_by_rebalance = _lagged_state_by_rebalance(defensive_weights.index, monthly_state)

    print("[5/7] run fixed candidates", flush=True)
    cost_params = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    sym_universe = sorted(set(panel["symbol"].astype(str).str.zfill(6).unique()))
    asset_returns = open_returns.reindex(columns=sym_universe).fillna(0.0)
    limit_mask = limit_up_open_mask.reindex(columns=sym_universe, fill_value=False)
    base_score_df = score_base.rename(columns={"score__defensive": "candidate_score"})
    upside_score_df = score_upside.rename(columns={"score__upside_c": "candidate_score"})
    results = [
        _run_from_weights(
            candidate=BASELINE,
            score_df=base_score_df,
            weights=defensive_weights,
            asset_returns=asset_returns,
            bench_daily=bench_daily,
            breadth_series=breadth_series,
            cost_params=cost_params,
            limit_up_open_mask=limit_mask,
        ),
        _run_from_weights(
            candidate=UPSIDE_C,
            score_df=upside_score_df,
            weights=upside_weights,
            asset_returns=asset_returns,
            bench_daily=bench_daily,
            breadth_series=breadth_series,
            cost_params=cost_params,
            limit_up_open_mask=limit_mask,
        ),
        _run_dual_v1(
            panel=panel,
            defensive_weights=defensive_weights,
            upside_weights=upside_weights,
            state_by_rebalance=state_by_rebalance,
            asset_returns=asset_returns,
            bench_daily=bench_daily,
            breadth_series=breadth_series,
            cost_params=cost_params,
            limit_up_open_mask=limit_mask,
        ),
    ]

    print("[6/7] aggregate diagnostics", flush=True)
    leaderboard = _build_leaderboard(results, baseline_id="BASELINE_S2_FIXED")
    base_row = leaderboard[leaderboard["candidate_id"] == "BASELINE_S2_FIXED"].iloc[0]
    accept_map = {row["candidate_id"]: _accept_summary(row, base_row) for _, row in leaderboard.iterrows()}
    comparison = _build_comparison(leaderboard, results)
    regime_long = pd.concat([_long_with_id(r["regime_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    breadth_long = pd.concat([_long_with_id(r["breadth_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    year_long = pd.concat([_long_with_id(r["year_capture"], r["candidate"]["id"]) for r in results], ignore_index=True)
    switch_long = pd.concat([_long_with_id(r["switch_by_regime"], r["candidate"]["id"]) for r in results], ignore_index=True)
    monthly_long = pd.concat([_long_with_id(r["monthly"], r["candidate"]["id"]) for r in results], ignore_index=True)
    state_diag_long = pd.concat(
        [df for df in (r["state_diag"] for r in results) if df is not None and not df.empty],
        ignore_index=True,
    )

    print("[7/7] write outputs", flush=True)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    leaderboard.to_csv(results_dir / f"{prefix}_leaderboard.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(results_dir / f"{prefix}_comparison.csv", index=False, encoding="utf-8-sig")
    regime_long.to_csv(results_dir / f"{prefix}_regime_long.csv", index=False, encoding="utf-8-sig")
    breadth_long.to_csv(results_dir / f"{prefix}_breadth_long.csv", index=False, encoding="utf-8-sig")
    year_long.to_csv(results_dir / f"{prefix}_year_long.csv", index=False, encoding="utf-8-sig")
    switch_long.to_csv(results_dir / f"{prefix}_switch_long.csv", index=False, encoding="utf-8-sig")
    monthly_long.to_csv(results_dir / f"{prefix}_monthly_long.csv", index=False, encoding="utf-8-sig")
    state_diag_long.to_csv(results_dir / f"{prefix}_state_diag_long.csv", index=False, encoding="utf-8-sig")
    state_by_rebalance.to_csv(results_dir / f"{prefix}_lagged_state_by_rebalance.csv", index=False, encoding="utf-8-sig")

    params = {
        "start": args.start,
        "end": end_date,
        "top_k": top_k,
        "rebalance_rule": rebalance_rule,
        "portfolio_method": "fixed_baseline_minimal_rerun",
        "max_turnover": max_turnover,
        "execution_mode": "tplus1_open",
        "state_lag": "previous_completed_month",
        "state_threshold_mode": "expanding",
        "defensive_sleeve": "vol_to_turnover",
        "upside_sleeve": "limit_up_hits_20d + tail_strength_20d",
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
        "upside_weight_diag_rows": int(len(upside_diag)),
    }
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": params,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "accept": accept_map,
        "monthly_state_thresholds": {
            "regime": monthly_state.attrs.get("regime_thresholds"),
            "breadth": monthly_state.attrs.get("breadth_thresholds"),
            "trace": monthly_state.attrs.get("threshold_trace"),
        },
        "candidates": [
            {
                "id": r["candidate"]["id"],
                "label": r["candidate"]["label"],
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
        comparison=comparison,
        accept_map=accept_map,
        regime_long=regime_long,
        switch_long=switch_long,
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
        result_type="r1f_fixed_baseline_rerun",
        research_topic="r1f_fixed_baseline_rerun",
        research_config_id=f"r1f_fixed_{slugify_token(prefix)}",
        output_stem=prefix,
    )
    data_slice = DataSlice(
        dataset_name="r1f_fixed_baseline_backtest",
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
        feature_set_id="r1f_fixed_baseline_factors",
        feature_columns=(),
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=config_source,
        extra={"top_k": int(top_k), "max_turnover": float(max_turnover), "candidates": R1F_CANDIDATES},
    )
    artifact_refs = (
        ArtifactRef("leaderboard_csv", _project_relative(results_dir / f"{prefix}_leaderboard.csv"), "csv", False, "leaderboard"),
        ArtifactRef("comparison_csv", _project_relative(results_dir / f"{prefix}_comparison.csv"), "csv", False, "新旧对比"),
        ArtifactRef("regime_long_csv", _project_relative(results_dir / f"{prefix}_regime_long.csv"), "csv", False, "状态长期"),
        ArtifactRef("breadth_long_csv", _project_relative(results_dir / f"{prefix}_breadth_long.csv"), "csv", False, "广度长期"),
        ArtifactRef("year_long_csv", _project_relative(results_dir / f"{prefix}_year_long.csv"), "csv", False, "年度长期"),
        ArtifactRef("switch_long_csv", _project_relative(results_dir / f"{prefix}_switch_long.csv"), "csv", False, "切换长期"),
        ArtifactRef("monthly_long_csv", _project_relative(results_dir / f"{prefix}_monthly_long.csv"), "csv", False, "月度长期"),
        ArtifactRef("state_diag_long_csv", _project_relative(results_dir / f"{prefix}_state_diag_long.csv"), "csv", False, "状态诊断"),
        ArtifactRef("lagged_state_by_rebalance_csv", _project_relative(results_dir / f"{prefix}_lagged_state_by_rebalance.csv"), "csv", False, "滞后状态"),
        ArtifactRef("summary_json", _project_relative(results_dir / f"{prefix}_summary.json"), "json", False, "汇总"),
        ArtifactRef("report_md", _project_relative(docs_dir / f"{prefix}.md"), "md", False, "报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    )
    metrics = {
        "candidate_count": int(len(leaderboard)),
        "comparison_rows": int(len(comparison)),
        "accept_count": int(sum(1 for v in accept_map.values() if v == "accept")),
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
        promotion={"production_eligible": False, "registry_status": "not_registered", "blocking_reasons": ["r1f_is_sanity_rerun_only"]},
        notes="R1F fixed baseline sanity rerun; not a promotion candidate.",
    )
    write_research_manifest(manifest_path, result)
    append_experiment_result(PROJECT_ROOT / "data" / "experiments", result)
    # --- end standard research contract ---


if __name__ == "__main__":
    main()
