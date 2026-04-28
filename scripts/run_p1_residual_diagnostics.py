"""解释 P1 静态为何仍优于 S2 基线的残余差异诊断。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
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
    compute_factors,
    contiguous_time_splits,
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
)
from scripts.run_score_ablation import (
    _build_report_payload,
    _extract_report_summary,
    _json_sanitize,
    _weights_to_membership,
    summarize_overlap,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    config_path: str
    is_reference: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("r1", "R1_s2_baseline", "config.yaml.backtest.r1_s2_baseline"),
    Scenario("r2", "R2_s2_top50", "config.yaml.backtest.r2_s2_top50"),
    Scenario("r3", "R3_s2_turnover05", "config.yaml.backtest.r3_s2_turnover05"),
    Scenario("r4", "R4_s2_top50_turnover05", "config.yaml.backtest.r4_s2_top50_turnover05"),
    Scenario("r5", "R5_s2_prefilter_off", "config.yaml.backtest.r5_s2_prefilter_off"),
    Scenario("r6", "R6_s2_universe_on", "config.yaml.backtest.r6_s2_universe_on"),
    Scenario("r7", "R7_s2_prefilter_off_universe_on", "config.yaml.backtest.r7_s2_prefilter_off_universe_on"),
    Scenario("r8", "R8_s2_no_regime", "config.yaml.backtest.r8_s2_no_regime"),
    Scenario("r9", "R9_p1_static_ref", "config.yaml.backtest.r9_p1_static_ref", is_reference=True),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 P1 残余差异诊断实验")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="p1_residual_diag_2026-04-19",
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
        help="仅运行指定场景，逗号分隔，如 r1,r4,r9；为空则跑全部",
    )
    return p.parse_args()


def summarize_weight_profile(weights: pd.DataFrame) -> dict[str, Any]:
    if weights.empty:
        return {
            "mean_name_count": np.nan,
            "median_name_count": np.nan,
            "mean_effective_n": np.nan,
            "median_effective_n": np.nan,
            "mean_top1_weight": np.nan,
            "mean_top5_weight_sum": np.nan,
        }
    rows: list[dict[str, float]] = []
    for _, row in weights.sort_index().iterrows():
        vals = pd.to_numeric(row, errors="coerce").fillna(0.0)
        vals = vals[vals > 0.0].sort_values(ascending=False)
        if vals.empty:
            continue
        vv = vals.to_numpy(dtype=np.float64)
        rows.append(
            {
                "name_count": float(len(vv)),
                "effective_n": float(1.0 / np.sum(np.square(vv))),
                "top1_weight": float(vv[0]),
                "top5_weight_sum": float(vv[: min(5, len(vv))].sum()),
            }
        )
    if not rows:
        return {
            "mean_name_count": np.nan,
            "median_name_count": np.nan,
            "mean_effective_n": np.nan,
            "median_effective_n": np.nan,
            "mean_top1_weight": np.nan,
            "mean_top5_weight_sum": np.nan,
        }
    tab = pd.DataFrame(rows)
    return {
        "mean_name_count": float(tab["name_count"].mean()),
        "median_name_count": float(tab["name_count"].median()),
        "mean_effective_n": float(tab["effective_n"].mean()),
        "median_effective_n": float(tab["effective_n"].median()),
        "mean_top1_weight": float(tab["top1_weight"].mean()),
        "mean_top5_weight_sum": float(tab["top5_weight_sum"].mean()),
    }


def summarize_rebalance_churn(weights: pd.DataFrame) -> dict[str, Any]:
    membership = _weights_to_membership(weights)
    dates = sorted(membership.keys())
    if len(dates) < 2:
        return {
            "mean_prev_overlap_count": np.nan,
            "mean_enter_count": np.nan,
            "mean_exit_count": np.nan,
            "mean_prev_overlap_ratio": np.nan,
        }
    rows: list[dict[str, float]] = []
    for prev_dt, cur_dt in zip(dates[:-1], dates[1:]):
        prev_set = membership.get(prev_dt, set())
        cur_set = membership.get(cur_dt, set())
        overlap = prev_set & cur_set
        rows.append(
            {
                "prev_overlap_count": float(len(overlap)),
                "enter_count": float(len(cur_set - prev_set)),
                "exit_count": float(len(prev_set - cur_set)),
                "prev_overlap_ratio": float(len(overlap) / len(cur_set)) if cur_set else np.nan,
            }
        )
    tab = pd.DataFrame(rows)
    return {
        "mean_prev_overlap_count": float(tab["prev_overlap_count"].mean()),
        "mean_enter_count": float(tab["enter_count"].mean()),
        "mean_exit_count": float(tab["exit_count"].mean()),
        "mean_prev_overlap_ratio": float(tab["prev_overlap_ratio"].mean()),
    }


def build_experiment_summary(label: str, payload: dict[str, Any], weights: pd.DataFrame) -> dict[str, Any]:
    out = _extract_report_summary(label, payload)
    out.update(summarize_weight_profile(weights))
    out.update(summarize_rebalance_churn(weights))
    return out


def build_doc(summary_df: pd.DataFrame, overlap_summary_df: pd.DataFrame, output_prefix: str) -> str:
    summary_md = summary_df.to_markdown(index=False)
    overlap_md = overlap_summary_df.to_markdown(index=False) if not overlap_summary_df.empty else "_无可用重合度数据_"
    generated_at = pd.Timestamp.utcnow().isoformat()
    return f"""# P1 残余优势诊断

- 生成时间：`{generated_at}`
- 固定主线：`S2 = vol_to_turnover` 单因子 / `tplus1_open`
- 诊断目的：拆分 `P1 静态` 相对 `S2` 的剩余优势，优先检查 `top_k`、`max_turnover`、`prefilter`、`universe_filter`、`regime`

## 全样本与 OOS 汇总

{summary_md}

## 与 P1 参考组合的持仓重合度

{overlap_md}

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_topk_overlap_summary.csv`
- `data/results/{output_prefix}_r*.json`
"""


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
    base_cfg, _ = load_config("config.yaml.backtest.r1_s2_baseline")
    db_path = str(PROJECT_ROOT / str(base_cfg["paths"]["duckdb_path"]))
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  factors={factors.shape}", flush=True)
    needed_base_cols = [
        "symbol",
        "trade_date",
        "vol_to_turnover",
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
    holdings_map: dict[str, dict[pd.Timestamp, set[str]]] = {}
    weights_map: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

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
        weights_map[scenario.label] = weights.copy()
        holdings_map[scenario.label] = _weights_to_membership(weights)
        print(f"  weights={weights.shape}", flush=True)

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
            "note": "p1 residual diagnostics omits extra benchmark recomputation; use run_backtest_eval for full benchmark panel",
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
        summary_rows.append(build_experiment_summary(scenario.label, payload, weights_map[scenario.label]))

        report_path = results_dir / f"{output_prefix}_{scenario.key}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {scenario.label}: {report_path.relative_to(PROJECT_ROOT)}", flush=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    summary_path = results_dir / f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    reference = next((s for s in scenarios if s.is_reference), None)
    overlap_rows: list[dict[str, Any]] = []
    if reference is not None:
        ref_membership = holdings_map.get(reference.label, {})
        for scenario in scenarios:
            if scenario.label == reference.label:
                continue
            overlap = summarize_overlap(ref_membership, holdings_map.get(scenario.label, {}))
            if overlap.empty:
                continue
            overlap_rows.append(
                {
                    "scenario": scenario.label,
                    "reference": reference.label,
                    "mean_overlap_count": float(overlap["overlap_count"].mean()),
                    "median_overlap_count": float(overlap["overlap_count"].median()),
                    "mean_overlap_ratio_vs_candidate": float(overlap["overlap_ratio_vs_candidate"].mean()),
                    "mean_overlap_ratio_vs_reference": float(overlap["overlap_ratio_vs_reference"].mean()),
                    "mean_candidate_only_count": float(overlap["candidate_only_count"].mean()),
                    "mean_reference_only_count": float(overlap["reference_only_count"].mean()),
                }
            )
    overlap_summary_df = pd.DataFrame(overlap_rows).sort_values("scenario").reset_index(drop=True)
    overlap_path = results_dir / f"{output_prefix}_topk_overlap_summary.csv"
    overlap_summary_df.to_csv(overlap_path, index=False, encoding="utf-8-sig")

    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text(build_doc(summary_df, overlap_summary_df, output_prefix), encoding="utf-8")
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] overlap: {overlap_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "scenarios": [s.__dict__ for s in scenarios],
        "artifacts": [
            str(summary_path.relative_to(PROJECT_ROOT)),
            str(overlap_path.relative_to(PROJECT_ROOT)),
            str(doc_path.relative_to(PROJECT_ROOT)),
        ]
        + [str((results_dir / f"{output_prefix}_{s.key}.json").relative_to(PROJECT_ROOT)) for s in scenarios],
    }
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
