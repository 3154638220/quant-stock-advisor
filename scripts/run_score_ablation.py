"""按 docs/plan.md 的 S1~S4 口径执行 score 消融，并输出汇总与诊断产物。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest_eval import (
    BacktestConfig,
    _attach_pit_fundamentals,
    _rebalance_dates,
    apply_p1_factor_policy,
    attach_universe_filter,
    build_asset_returns,
    build_ic_weights_from_monitor,
    build_open_to_open_returns,
    build_regime_weight_overrides,
    build_score,
    build_topk_weights,
    compare_full_vs_slices,
    compute_factors,
    load_config,
    load_daily_from_duckdb,
    load_factor_ic_summary,
    load_ic_weights_by_date,
    normalize_weights,
    resolve_industry_cap_and_map,
    rolling_walk_forward_windows,
    run_backtest,
    transaction_cost_params_from_mapping,
    walk_forward_backtest,
    contiguous_time_splits,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    config_path: str
    is_reference: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("s1", "S1_ocf_single", "config.yaml.backtest.s1_ocf"),
    Scenario("s2", "S2_vol_single", "config.yaml.backtest.s2_vol"),
    Scenario("s3", "S3_dual_7030", "config.yaml.backtest.s3_dual_7030"),
    Scenario("s4", "S4_legacy3", "config.yaml.backtest.s4_legacy3"),
    Scenario("s5", "S5_p1_static", "config.yaml.backtest.s5_p1_static", is_reference=True),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 score 消融实验并导出统一诊断产物")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="score_ablation_2026-04-19",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--wf-train-window", type=int, default=252, help="滚动 WF 训练窗")
    p.add_argument("--wf-test-window", type=int, default=63, help="滚动 WF 测试窗")
    p.add_argument("--wf-step-window", type=int, default=63, help="滚动 WF 步长")
    p.add_argument("--wf-slice-splits", type=int, default=5, help="时间切片折数")
    p.add_argument("--wf-slice-min-train-days", type=int, default=252, help="时间切片最少训练窗")
    p.add_argument("--wf-slice-fixed-window", action="store_true", help="时间切片使用固定训练窗")
    p.add_argument(
        "--scenarios",
        default="",
        help="仅运行指定场景，逗号分隔，如 s1,s3,s5；为空则跑全部",
    )
    return p.parse_args()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return _json_sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _weights_to_membership(weights: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    membership: dict[pd.Timestamp, set[str]] = {}
    for dt, row in weights.sort_index().iterrows():
        names = [str(sym).zfill(6) for sym, val in row.items() if float(val) > 0.0]
        membership[pd.Timestamp(dt)] = set(names)
    return membership


def summarize_overlap(
    reference: dict[pd.Timestamp, set[str]],
    candidate: dict[pd.Timestamp, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt in sorted(set(reference.keys()) & set(candidate.keys())):
        ref_set = reference.get(dt, set())
        cand_set = candidate.get(dt, set())
        if not ref_set and not cand_set:
            continue
        overlap = ref_set & cand_set
        rows.append(
            {
                "trade_date": pd.Timestamp(dt),
                "reference_count": len(ref_set),
                "candidate_count": len(cand_set),
                "overlap_count": len(overlap),
                "candidate_only_count": len(cand_set - ref_set),
                "reference_only_count": len(ref_set - cand_set),
                "overlap_ratio_vs_candidate": len(overlap) / len(cand_set) if cand_set else np.nan,
                "overlap_ratio_vs_reference": len(overlap) / len(ref_set) if ref_set else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise_score_correlations(score_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = list(score_frames.keys())
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(keys):
        lhs = score_frames[left].rename(columns={"score": "score_left"})
        for right in keys[i + 1 :]:
            rhs = score_frames[right].rename(columns={"score": "score_right"})
            merged = lhs.merge(rhs, on=["trade_date", "symbol"], how="inner")
            if merged.empty:
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "common_rows": 0,
                        "median_daily_corr": np.nan,
                        "mean_daily_corr": np.nan,
                    }
                )
                continue
            corr_by_date = (
                merged.groupby("trade_date", sort=True)
                .apply(lambda g: g["score_left"].corr(g["score_right"]), include_groups=False)
                .dropna()
            )
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "common_rows": int(len(merged)),
                    "median_daily_corr": float(corr_by_date.median()) if not corr_by_date.empty else np.nan,
                    "mean_daily_corr": float(corr_by_date.mean()) if not corr_by_date.empty else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["left", "right", "common_rows", "median_daily_corr", "mean_daily_corr"])


def _extract_report_summary(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    full_sample = payload["full_sample"]["with_cost"]
    wf_roll = payload["walk_forward_rolling"]["agg"]
    wf_slice = payload["walk_forward_slices"]["agg"]
    return {
        "scenario": label,
        "annualized_return": full_sample["annualized_return"],
        "sharpe_ratio": full_sample["sharpe_ratio"],
        "max_drawdown": full_sample["max_drawdown"],
        "turnover_mean": full_sample["turnover_mean"],
        "rolling_oos_median_ann_return": wf_roll.get("median_ann_return"),
        "rolling_oos_median_sharpe": wf_roll.get("median_sharpe_ratio"),
        "slice_oos_median_ann_return": wf_slice.get("median_ann_return"),
        "slice_oos_median_sharpe": wf_slice.get("median_sharpe_ratio"),
    }


def _coverage_rows(
    scenario: str,
    factors: pd.DataFrame,
    weights: dict[str, float],
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    fac_cols = [c for c in weights.keys() if c in factors.columns]
    rows: list[dict[str, Any]] = []
    scored_count = score_df.groupby("trade_date")["symbol"].nunique().rename("scored_symbols")
    for dt, g in factors.groupby("trade_date", sort=True):
        eligible = g
        if "_universe_eligible" in g.columns:
            eligible = g.loc[g["_universe_eligible"].to_numpy(dtype=bool)]
        universe_count = int(len(eligible))
        active_factor_count = 0
        for fac in fac_cols:
            col = pd.to_numeric(eligible[fac], errors="coerce")
            valid_count = int((col.notna() & np.isfinite(col)).sum())
            if valid_count >= 5 and abs(float(weights.get(fac, 0.0))) > 1e-15:
                active_factor_count += 1
            rows.append(
                {
                    "scenario": scenario,
                    "trade_date": pd.Timestamp(dt),
                    "factor": fac,
                    "eligible_symbols": universe_count,
                    "valid_count": valid_count,
                    "missing_rate": 1.0 - (valid_count / universe_count) if universe_count > 0 else np.nan,
                    "active_factor_count": np.nan,
                    "scored_symbols": int(scored_count.get(pd.Timestamp(dt), 0)),
                }
            )
        for idx in range(len(rows) - len(fac_cols), len(rows)):
            rows[idx]["active_factor_count"] = active_factor_count
    return pd.DataFrame(rows)


def _build_report_payload(
    *,
    config_source: str,
    parameters: dict[str, Any],
    res_nc: Any,
    res_wc: Any,
    benchmark_market: dict[str, Any],
    wf_detail: pd.DataFrame,
    wf_agg: dict[str, Any],
    sp_detail: pd.DataFrame,
    sp_agg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": config_source,
        "parameters": parameters,
        "full_sample": {
            "no_cost": _json_sanitize(res_nc.panel.to_dict()),
            "with_cost": _json_sanitize(res_wc.panel.to_dict()),
            "market_ew": _json_sanitize(benchmark_market),
        },
        "walk_forward_rolling": {
            "detail": _json_sanitize(wf_detail.to_dict(orient="records")),
            "agg": _json_sanitize(wf_agg),
        },
        "walk_forward_slices": {
            "detail": _json_sanitize(sp_detail.to_dict(orient="records")),
            "agg": _json_sanitize(sp_agg),
            "full_vs_slices": _json_sanitize(compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}),
        },
    }


def main() -> None:
    args = parse_args()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    scenario_filter = {x.strip().lower() for x in str(args.scenarios).split(",") if x.strip()}
    scenarios = tuple(s for s in SCENARIOS if not scenario_filter or s.key in scenario_filter or s.label.lower() in scenario_filter)
    if not scenarios:
        raise SystemExit("未匹配到任何场景，请检查 --scenarios")

    print(f"[1/4] load base data: start={args.start} end={end_date} scenarios={[s.key for s in scenarios]}", flush=True)
    base_cfg, _ = load_config("config.yaml.backtest.s3_dual_7030")
    db_path = str(PROJECT_ROOT / str(base_cfg["paths"]["duckdb_path"]))
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  factors={factors.shape}", flush=True)
    needed_base_cols = [
        "symbol",
        "trade_date",
        "vol_to_turnover",
        "lower_shadow_ratio",
        "turnover_roll_mean",
        "price_position",
        "limit_move_hits_5d",
        "log_market_cap",
    ]
    factors = factors[[c for c in needed_base_cols if c in factors.columns]].copy()
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    print(f"  factors@rebalance={factors.shape}", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    print(f"  factors+pit={factors.shape}", flush=True)

    scenario_payloads: dict[str, dict[str, Any]] = {}
    score_frames: dict[str, pd.DataFrame] = {}
    holdings_map: dict[str, dict[pd.Timestamp, set[str]]] = {}
    summary_rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"[2/4] scenario {idx}/{len(scenarios)}: {scenario.key} ({scenario.label})", flush=True)
        cfg, config_source = load_config(scenario.config_path)
        signals = cfg.get("signals", {})
        portfolio_cfg = cfg.get("portfolio", {})
        backtest_cfg = cfg.get("backtest", {})
        risk_cfg = cfg.get("risk", {}) or {}
        prefilter_cfg = cfg.get("prefilter", {}) or {}
        regime_cfg = cfg.get("regime", {}) or {}
        uf_cfg = cfg.get("universe_filter", {}) or {}
        p1_cfg = signals.get("p1_factor_filter", {}) or {}
        ic_cfg = signals.get("ic_weighting", {}) or {}

        ce_weights = normalize_weights(signals.get("composite_extended", {}))
        p1_filter_enabled = bool(p1_cfg.get("enabled", False))
        p1_ic_report_path = str(p1_cfg.get("ic_report_path", "")).strip()
        if p1_filter_enabled and p1_ic_report_path:
            ic_summary = load_factor_ic_summary(p1_ic_report_path)
            if not ic_summary.empty:
                ce_weights, _ = apply_p1_factor_policy(
                    ce_weights,
                    ic_summary,
                    remove_if_t1_and_t21_negative=bool(p1_cfg.get("remove_if_t1_and_t21_negative", True)),
                    zero_if_abs_t1_below=float(p1_cfg.get("zero_if_abs_t1_below", 0.0)),
                    flip_if_t1_negative_and_t21_above=float(p1_cfg.get("flip_if_t1_negative_and_t21_above", 0.005)),
                )

        scenario_factors = attach_universe_filter(
            factors,
            daily_df,
            enabled=bool(uf_cfg.get("enabled", False)),
            min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
            require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
        )

        regime_enabled = bool(regime_cfg.get("enabled", True))
        regime_overrides: dict[pd.Timestamp, dict[str, float]] = {}
        if regime_enabled:
            regime_overrides, _ = build_regime_weight_overrides(
                scenario_factors,
                daily_df,
                ce_weights,
                benchmark_symbol=str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
                regime_cfg_raw=regime_cfg,
                market_ew_min_days=439,
            )

        ic_weighting_enabled = bool(ic_cfg.get("enabled", False))
        ic_overrides: dict[pd.Timestamp, dict[str, float]] = {}
        if ic_weighting_enabled:
            weights_path = str(ic_cfg.get("weights_path", "")).strip()
            monitor_path = str(ic_cfg.get("monitor_path", "")).strip()
            if weights_path:
                ic_overrides = load_ic_weights_by_date(weights_path)
            elif monitor_path:
                ic_overrides = build_ic_weights_from_monitor(
                    monitor_path,
                    window=int(ic_cfg.get("window", 60)),
                    min_obs=int(ic_cfg.get("min_obs", 20)),
                    half_life=float(ic_cfg.get("half_life", 20.0)),
                    clip_abs_weight=float(ic_cfg.get("clip_abs_weight", 0.25)),
                )

        merged_overrides: dict[pd.Timestamp, dict[str, float]] = {}
        for dt in sorted(set(regime_overrides.keys()) | set(ic_overrides.keys())):
            merged = dict(ce_weights)
            if dt in regime_overrides:
                merged.update(regime_overrides[dt])
            if dt in ic_overrides:
                merged.update(ic_overrides[dt])
            merged_overrides[pd.Timestamp(dt)] = merged

        score_df = build_score(
            scenario_factors,
            ce_weights,
            weights_by_date=merged_overrides if merged_overrides else (regime_overrides if regime_overrides else None),
        )
        print(f"  score_df={score_df.shape}", flush=True)
        score_frames[scenario.label] = score_df.copy()

        industry_cap_count, industry_map, _ = resolve_industry_cap_and_map(
            int(portfolio_cfg.get("industry_cap_count", 5)),
            "data/cache/industry_map.csv",
        )
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(signals.get("top_k", 20)),
            rebalance_rule=str(backtest_cfg.get("eval_rebalance_rule", "M")),
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 1.0)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
            cov_lookback_days=int(portfolio_cfg.get("cov_lookback_days", 252)),
            cov_ridge=float(portfolio_cfg.get("cov_ridge", 1e-6)),
            cov_shrinkage=str(portfolio_cfg.get("cov_shrinkage", "ledoit_wolf")).lower(),
            cov_ewma_halflife=float(portfolio_cfg.get("cov_ewma_halflife", 20.0)),
            risk_aversion=float(portfolio_cfg.get("risk_aversion", 1.0)),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]
        print(f"  weights={weights.shape}", flush=True)
        holdings_map[scenario.label] = _weights_to_membership(weights)

        target_cols = sorted({str(col).zfill(6) for col in weights.columns})
        execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
        if execution_mode == "tplus1_open":
            open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
            open_returns = open_returns.sort_index()
            open_returns.index = pd.to_datetime(open_returns.index)
            asset_returns = open_returns[
                (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
            ]
        else:
            asset_returns = build_asset_returns(daily_df, target_cols, args.start, end_date)
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        weights = weights.reindex(columns=target_cols, fill_value=0.0)

        first_reb = weights.index.min()
        if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        asset_returns = asset_returns[asset_returns.index >= first_reb]

        costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
        bt_no_cost = BacktestConfig(cost_params=None, execution_mode=execution_mode, execution_lag=1)
        bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
        res_nc = run_backtest(asset_returns, weights, config=bt_no_cost)
        res_wc = run_backtest(asset_returns, weights, config=bt_cost)
        print(
            "  full_sample="
            f"{res_wc.panel.annualized_return:.4f}/{res_wc.panel.sharpe_ratio:.4f}/{res_wc.panel.max_drawdown:.4f}",
            flush=True,
        )

        benchmark_market = {
            "n_periods": int(len(res_wc.daily_returns)),
            "note": "score ablation script omits extra benchmark recomputation; use run_backtest_eval for full benchmark panel",
        }
        rolling = rolling_walk_forward_windows(
            asset_returns.index,
            train_days=max(20, int(args.wf_train_window)),
            test_days=max(5, int(args.wf_test_window)),
            step_days=max(5, int(args.wf_step_window)),
        )
        _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cost, use_test_only=True)
        slices = contiguous_time_splits(
            asset_returns.index,
            n_splits=max(2, int(args.wf_slice_splits)),
            min_train_days=max(20, int(args.wf_slice_min_train_days)),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )
        _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cost, use_test_only=True)

        payload = _build_report_payload(
            config_source=config_source,
            parameters={
                "start": args.start,
                "end": end_date,
                "top_k": int(signals.get("top_k", 20)),
                "rebalance_rule": str(backtest_cfg.get("eval_rebalance_rule", "M")),
                "max_turnover": float(portfolio_cfg.get("max_turnover", 1.0)),
                "portfolio_method": str(portfolio_cfg.get("weight_method", "equal_weight")),
                "execution_mode": execution_mode,
                "regime_enabled": regime_enabled,
                "composite_extended_weights": ce_weights,
                "prefilter": prefilter_cfg,
                "universe_filter": uf_cfg,
                "p1_factor_filter_enabled": p1_filter_enabled,
            },
            res_nc=res_nc,
            res_wc=res_wc,
            benchmark_market=benchmark_market,
            wf_detail=wf_detail,
            wf_agg=wf_agg,
            sp_detail=sp_detail,
            sp_agg=sp_agg,
        )
        scenario_payloads[scenario.label] = payload
        summary_rows.append(_extract_report_summary(scenario.label, payload))
        coverage_frames.append(_coverage_rows(scenario.label, scenario_factors, ce_weights, score_df))

        report_path = results_dir / f"{output_prefix}_{scenario.key}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {scenario.label}: {report_path.relative_to(PROJECT_ROOT)}")

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    coverage_df = pd.concat(coverage_frames, ignore_index=True)
    coverage_summary_df = (
        coverage_df.groupby(["scenario", "factor"], dropna=False)
        .agg(
            mean_missing_rate=("missing_rate", "mean"),
            median_missing_rate=("missing_rate", "median"),
            mean_valid_count=("valid_count", "mean"),
            mean_eligible_symbols=("eligible_symbols", "mean"),
            mean_active_factor_count=("active_factor_count", "mean"),
            mean_scored_symbols=("scored_symbols", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "factor"])
        .reset_index(drop=True)
    )
    pairwise_corr_df = summarize_pairwise_score_correlations(score_frames)
    if not pairwise_corr_df.empty:
        pairwise_corr_df = pairwise_corr_df.sort_values(["left", "right"]).reset_index(drop=True)

    ref_label = next(s.label for s in scenarios if s.is_reference) if any(s.is_reference for s in scenarios) else ""
    overlap_detail_frames: list[pd.DataFrame] = []
    overlap_summary_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not ref_label or scenario.label == ref_label:
            continue
        detail = summarize_overlap(holdings_map[ref_label], holdings_map[scenario.label])
        if detail.empty:
            continue
        detail.insert(0, "scenario", scenario.label)
        detail.insert(1, "reference", ref_label)
        overlap_detail_frames.append(detail)
        overlap_summary_rows.append(
            {
                "scenario": scenario.label,
                "reference": ref_label,
                "mean_overlap_count": float(detail["overlap_count"].mean()),
                "median_overlap_count": float(detail["overlap_count"].median()),
                "mean_overlap_ratio_vs_candidate": float(detail["overlap_ratio_vs_candidate"].mean()),
                "mean_overlap_ratio_vs_reference": float(detail["overlap_ratio_vs_reference"].mean()),
                "mean_candidate_only_count": float(detail["candidate_only_count"].mean()),
                "mean_reference_only_count": float(detail["reference_only_count"].mean()),
            }
        )

    overlap_detail_df = pd.concat(overlap_detail_frames, ignore_index=True) if overlap_detail_frames else pd.DataFrame()
    overlap_summary_df = pd.DataFrame(overlap_summary_rows).sort_values("scenario").reset_index(drop=True)

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    coverage_path = results_dir / f"{output_prefix}_coverage_daily.csv"
    coverage_summary_path = results_dir / f"{output_prefix}_coverage_summary.csv"
    score_corr_path = results_dir / f"{output_prefix}_score_corr.csv"
    overlap_detail_path = results_dir / f"{output_prefix}_topk_overlap_detail.csv"
    overlap_summary_path = results_dir / f"{output_prefix}_topk_overlap_summary.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    coverage_summary_df.to_csv(coverage_summary_path, index=False, encoding="utf-8-sig")
    pairwise_corr_df.to_csv(score_corr_path, index=False, encoding="utf-8-sig")
    overlap_detail_df.to_csv(overlap_detail_path, index=False, encoding="utf-8-sig")
    overlap_summary_df.to_csv(overlap_summary_path, index=False, encoding="utf-8-sig")

    doc_lines = [
        "# Score 消融实验",
        "",
        f"- 生成时间：`{pd.Timestamp.utcnow().isoformat()}`",
        f"- 区间：`{args.start}` 到 `{end_date}`",
        f"- 固定口径：`tplus1_open` / `M` / `top_k=20` / `max_turnover=0.3` / `universe_filter=false`（S5 为外部对照，沿用 P1 静态原参数）",
        "",
        "## 全样本与 OOS 汇总",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## 与 S5 的 Top-K 重合度",
        "",
        overlap_summary_df.to_markdown(index=False) if not overlap_summary_df.empty else "_无公共调仓日_",
        "",
        "## 分数相关性",
        "",
        pairwise_corr_df.to_markdown(index=False) if not pairwise_corr_df.empty else "_单场景运行，无 pairwise 相关性_",
        "",
        "## 产物",
        "",
        f"- `{summary_path.relative_to(PROJECT_ROOT)}`",
        f"- `{coverage_summary_path.relative_to(PROJECT_ROOT)}`",
        f"- `{score_corr_path.relative_to(PROJECT_ROOT)}`",
        f"- `{overlap_summary_path.relative_to(PROJECT_ROOT)}`",
    ]
    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text("\n".join(doc_lines), encoding="utf-8")
    print(f"[ok] summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[ok] doc: {doc_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
