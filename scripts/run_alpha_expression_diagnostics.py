"""诊断 expression 候选为何能改善组合却未通过正式 IC gate。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_alpha_expression_scout import (
    DEFAULT_CANDIDATES,
    build_conditioned_flip_factor,
    build_expression_scenarios,
    compute_expression_gate_table,
    cross_sectional_residualize,
)
from scripts.run_alpha_factor_scout import DEFAULT_BASELINE_FACTOR
from scripts.run_backtest_eval import (
    BacktestConfig,
    _prepared_factors_cache_expected_meta,
    _rebalance_dates,
    _resolve_optional_path,
    build_market_ew_benchmark,
    build_open_to_open_returns,
    build_score,
    build_topk_weights,
    load_config,
    load_daily_from_duckdb,
    prepare_factors_for_backtest,
    resolve_industry_cap_and_map,
    transaction_cost_params_from_mapping,
    run_backtest,
)
from scripts.run_factor_admission_validation import _json_sanitize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 alpha expression diagnostics")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="alpha_expression_diagnostics_2026-04-20",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--config", default="config.yaml.backtest.r7_s2_prefilter_off_universe_on", help="研究基线配置")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--top-k", type=int, default=20, help="统一使用的 Top-K")
    p.add_argument(
        "--prepared-factors-cache",
        default="",
        help="prepared factors parquet 缓存路径；命中后跳过 compute_factors/PIT/universe 预处理",
    )
    p.add_argument(
        "--refresh-prepared-factors-cache",
        action="store_true",
        help="忽略已有 prepared factors 缓存并强制重建",
    )
    return p.parse_args()


def _annualize_period_return(ret: float, n_periods: int, periods_per_year: float) -> float:
    if not np.isfinite(ret) or n_periods <= 0 or ret <= -1.0:
        return float("nan")
    return float((1.0 + ret) ** (periods_per_year / float(n_periods)) - 1.0)


def _build_monthly_detail(strategy_daily: pd.Series, benchmark_daily: pd.Series, scenario: str) -> pd.DataFrame:
    common = pd.DatetimeIndex(strategy_daily.index).intersection(pd.DatetimeIndex(benchmark_daily.index)).sort_values()
    strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0)
    bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0)
    month_key = common.to_period("M")

    monthly = pd.DataFrame(
        {
            "strategy_return": strat.groupby(month_key).apply(lambda s: float((1.0 + s).prod() - 1.0)),
            "benchmark_return": bench.groupby(month_key).apply(lambda s: float((1.0 + s).prod() - 1.0)),
        }
    ).reset_index(names="month")
    monthly["scenario"] = scenario
    monthly["excess_return"] = monthly["strategy_return"] - monthly["benchmark_return"]
    monthly["benchmark_up"] = monthly["benchmark_return"] > 0.0
    monthly["strategy_up"] = monthly["strategy_return"] > 0.0
    monthly["beat_benchmark"] = monthly["excess_return"] > 0.0
    return monthly


def _summarize_month_capture(monthly_df: pd.DataFrame) -> dict[str, Any]:
    if monthly_df.empty:
        return {}
    up = monthly_df[monthly_df["benchmark_up"]].copy()
    down = monthly_df[~monthly_df["benchmark_up"]].copy()
    return {
        "months_total": int(len(monthly_df)),
        "months_up": int(len(up)),
        "months_down": int(len(down)),
        "up_month_beat_rate": float(up["beat_benchmark"].mean()) if not up.empty else np.nan,
        "up_month_median_excess": float(pd.to_numeric(up["excess_return"], errors="coerce").median()) if not up.empty else np.nan,
        "down_month_beat_rate": float(down["beat_benchmark"].mean()) if not down.empty else np.nan,
        "down_month_median_excess": float(pd.to_numeric(down["excess_return"], errors="coerce").median()) if not down.empty else np.nan,
    }


def classify_defensive_overlay(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    if out.empty:
        out["overlay_role"] = pd.Series(dtype=object)
        out["overlay_note"] = pd.Series(dtype=object)
        return out

    out["overlay_role"] = "not_overlay"
    out["overlay_note"] = "没有显示出明确的防守型 overlay 特征。"
    baseline_mask = out["family"].astype(str).eq("baseline")
    out.loc[baseline_mask, "overlay_role"] = "baseline"
    out.loc[baseline_mask, "overlay_note"] = "当前默认研究基线。"

    baseline_row = out.loc[baseline_mask].head(1)
    if baseline_row.empty:
        return out

    baseline_up = float(pd.to_numeric(baseline_row["up_month_beat_rate"], errors="coerce").iloc[0])
    baseline_down = float(pd.to_numeric(baseline_row["down_month_beat_rate"], errors="coerce").iloc[0])
    baseline_down_excess = float(pd.to_numeric(baseline_row["down_month_median_excess"], errors="coerce").iloc[0])

    defensive_mask = (
        (~baseline_mask)
        & pd.to_numeric(out["down_month_beat_rate"], errors="coerce").gt(baseline_down)
        & pd.to_numeric(out["down_month_median_excess"], errors="coerce").gt(baseline_down_excess)
        & pd.to_numeric(out["up_month_beat_rate"], errors="coerce").le(max(baseline_up + 0.05, 0.25))
    )
    out.loc[defensive_mask, "overlay_role"] = "defensive_overlay_candidate"
    out.loc[defensive_mask, "overlay_note"] = "更像增强下跌月份防守的 overlay，不应作为新主排序信号升级。"
    return out


def build_overlay_diagnostics_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "- 本轮没有可总结的 diagnostics 结果。"

    lines: list[str] = []
    baseline_rows = summary_df.loc[summary_df["overlay_role"].astype(str) == "baseline", "scenario"].astype(str).tolist()
    if baseline_rows:
        lines.append(f"- 基线：`{baseline_rows[0]}` 作为上涨/下跌月份对照参考。")

    defensive_rows = summary_df.loc[
        summary_df["overlay_role"].astype(str) == "defensive_overlay_candidate",
        "scenario",
    ].astype(str).tolist()
    if defensive_rows:
        lines.append(
            "- 防守型 overlay："
            + "、".join(f"`{name}`" for name in defensive_rows)
            + " 更像增强下跌月份防守的组合层表达，后续只应作为 overlay 对照保留。"
        )
    else:
        lines.append("- 防守型 overlay：本轮没有出现明确优于基线下跌月份防守、且上涨参与仍不足的表达。")

    non_overlay_rows = summary_df.loc[
        summary_df["overlay_role"].astype(str) == "not_overlay",
        "scenario",
    ].astype(str).tolist()
    if non_overlay_rows:
        preview = "、".join(f"`{name}`" for name in non_overlay_rows[:5])
        suffix = " 等候选" if len(non_overlay_rows) > 5 else ""
        lines.append(f"- 非 overlay：{preview}{suffix} 当前没有显示出明确的防守增强器特征。")
    return "\n".join(lines)


def _build_doc(
    summary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    output_prefix: str,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    scenario_names = summary_df["scenario"].astype(str).tolist()
    overlay_summary = build_overlay_diagnostics_summary(summary_df)
    monthly_pivot = (
        monthly_df.pivot_table(index="month", columns="scenario", values="excess_return", aggfunc="first")
        .reset_index()
        .sort_values("month")
    )
    return f"""# Alpha Expression Diagnostics

- 生成时间：`{generated_at}`
- 目标：解释为何 expression-level IC 为负，但部分 expression 方案仍能改善组合结果
- 诊断场景：`{", ".join(scenario_names)}`

## 场景汇总

{summary_df.to_markdown(index=False)}

## Expression Gate

{gate_df.to_markdown(index=False)}

## 月度超额

{monthly_pivot.to_markdown(index=False)}

## 决策摘要

{overlay_summary}

## 结论提示

- 若某方案 `IC gate` 为负，但 `up_month_median_excess / beat_rate` 改善，说明它更像是在组合层抑制某类坏暴露，而不是提供稳定正向排序信号。
- 若某方案只改善全样本和关键年份，却无法改善 `up_month_beat_rate` 或 `slice OOS` 超额，它更接近可保留 overlay，而不是可升级主因子。
- `overlay_role=defensive_overlay_candidate` 表示该方案更像防守增强器，后续只应作为 overlay 对照保留。

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_factor_gate.csv`
- `data/results/{output_prefix}_monthly.csv`
- `docs/{output_prefix}.md`
"""


def main() -> None:
    args = parse_args()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()

    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load base data: start={args.start} end={end_date}", flush=True)
    cfg, _ = load_config(args.config)
    db_path = str(PROJECT_ROOT / str(cfg["paths"]["duckdb_path"]))
    prepared_factors_cache = _resolve_optional_path(args.prepared_factors_cache)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)
    results_cfg_dir = PROJECT_ROOT / str(cfg.get("paths", {}).get("results_dir", "data/results"))
    uf = cfg.get("universe_filter", {}) or {}
    prepared_factors_cache_meta = _prepared_factors_cache_expected_meta(
        start_date=args.start,
        end_date=end_date,
        lookback_days=int(args.lookback_days),
        min_hist_days=int(args.min_hist_days),
        db_path=db_path,
        results_dir=str(results_cfg_dir),
        universe_filter_cfg=uf,
    )
    factors, factors_cache_hit = prepare_factors_for_backtest(
        daily_df,
        min_hist_days=int(args.min_hist_days),
        db_path=db_path,
        results_dir=results_cfg_dir,
        universe_filter_cfg=uf,
        cache_path=prepared_factors_cache,
        refresh_cache=bool(args.refresh_prepared_factors_cache),
        cache_meta=prepared_factors_cache_meta,
    )
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    if prepared_factors_cache is not None:
        cache_state = "hit" if factors_cache_hit else "rebuilt"
        print(f"  prepared_factors_cache: {cache_state} -> {prepared_factors_cache}", flush=True)
    scenario_factors = factors

    baseline_factor = DEFAULT_BASELINE_FACTOR
    factor_name = "realized_vol"
    scenario_factors = cross_sectional_residualize(
        scenario_factors,
        target_col=factor_name,
        control_cols=[baseline_factor],
        suffix="_resid_against_v3",
    )
    scenario_factors["flip_resid_realized_vol"] = scenario_factors[f"{factor_name}_resid_against_v3"]
    scenario_factors = build_conditioned_flip_factor(
        scenario_factors,
        factor_col=factor_name,
        z_threshold=0.5,
        suffix="_cond_flip_z0p5",
    )
    scenario_factors["flip_cond_z0p5_realized_vol"] = scenario_factors[f"{factor_name}_cond_flip_z0p5"]

    all_scenarios, expression_specs = build_expression_scenarios(
        baseline_factor=baseline_factor,
        candidates=[factor_name],
        blend_weights=[0.15, 0.20],
        condition_z_thresholds=[0.5],
    )
    keep = {
        "baseline_vol_to_turnover",
        "flip_resid_blend_20_realized_vol",
        "flip_cond_z0p5_blend_15_realized_vol",
        "flip_cond_z0p5_blend_20_realized_vol",
    }
    scenario_defs = [item for item in all_scenarios if item["scenario"] in keep]
    expression_specs = [item for item in expression_specs if item["expression_factor"] in {"flip_resid_realized_vol", "flip_cond_z0p5_realized_vol"}]

    portfolio_cfg = cfg.get("portfolio", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
    bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
    industry_cap_count, industry_map, _ = resolve_industry_cap_and_map(
        int(portfolio_cfg.get("industry_cap_count", 5)),
        "data/cache/industry_map.csv",
    )

    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    open_returns.index = pd.to_datetime(open_returns.index)
    asset_returns = open_returns[
        (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
    ]
    benchmark_min_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    market_benchmark = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_days).sort_index()

    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []

    print(f"[2/4] run scenarios: {len(scenario_defs)}", flush=True)
    for scenario in scenario_defs:
        print(f"  scenario={scenario['scenario']}", flush=True)
        score_df = build_score(scenario_factors, scenario["score_weights"])
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(args.top_k),
            rebalance_rule=str(backtest_cfg.get("eval_rebalance_rule", "M")),
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 0.3)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]
        target_cols = sorted({str(col).zfill(6) for col in weights.columns})
        scenario_asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        weights = weights.reindex(columns=target_cols, fill_value=0.0)
        first_reb = weights.index.min()
        if not scenario_asset_returns.empty and first_reb > scenario_asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([scenario_asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        scenario_asset_returns = scenario_asset_returns[scenario_asset_returns.index >= first_reb]
        res = run_backtest(scenario_asset_returns, weights, config=bt_cost)

        monthly = _build_monthly_detail(res.daily_returns, market_benchmark, scenario["scenario"])
        monthly_rows.append(monthly)
        capture = _summarize_month_capture(monthly)

        key_years = pd.DataFrame({"year": monthly["month"].astype(str).str[:4].astype(int), "excess": monthly["excess_return"]})
        key_years = key_years[key_years["year"].isin([2021, 2025, 2026])]

        summary_rows.append(
            {
                "scenario": scenario["scenario"],
                "candidate_factor": scenario["candidate_factor"],
                "family": scenario["family"],
                "expression_type": scenario["expression_type"],
                "annualized_return": float(res.panel.annualized_return),
                "sharpe_ratio": float(res.panel.sharpe_ratio),
                "max_drawdown": float(res.panel.max_drawdown),
                "turnover_mean": float(res.panel.turnover_mean),
                "annualized_excess_vs_market": _annualize_period_return(float((1.0 + (res.daily_returns - market_benchmark.reindex(res.daily_returns.index).fillna(0.0))).prod() - 1.0), len(res.daily_returns), 252.0),
                "up_month_beat_rate": capture.get("up_month_beat_rate"),
                "up_month_median_excess": capture.get("up_month_median_excess"),
                "down_month_beat_rate": capture.get("down_month_beat_rate"),
                "down_month_median_excess": capture.get("down_month_median_excess"),
                "key_year_monthly_excess_mean": float(pd.to_numeric(key_years["excess"], errors="coerce").mean()) if not key_years.empty else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["annualized_excess_vs_market", "up_month_median_excess", "annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df = classify_defensive_overlay(summary_df)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    gate_df = compute_expression_gate_table(
        scenario_factors,
        daily_df,
        baseline_factor=baseline_factor,
        expression_specs=expression_specs,
    )

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    gate_path = results_dir / f"{output_prefix}_factor_gate.csv"
    monthly_path = results_dir / f"{output_prefix}_monthly.csv"
    doc_path = docs_dir / f"{output_prefix}.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    doc_path.write_text(_build_doc(summary_df, gate_df, monthly_df, output_prefix), encoding="utf-8")

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": [
            str(summary_path.relative_to(PROJECT_ROOT)),
            str(gate_path.relative_to(PROJECT_ROOT)),
            str(monthly_path.relative_to(PROJECT_ROOT)),
            str(doc_path.relative_to(PROJECT_ROOT)),
        ],
    }
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] gate: {gate_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] monthly: {monthly_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
