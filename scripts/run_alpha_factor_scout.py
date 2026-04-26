"""按 benchmark-first 口径快速筛查现有单因子，作为新 alpha 构造前的侦察层。"""

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

from scripts.run_backtest_eval import (
    BacktestConfig,
    _prepared_factors_cache_expected_meta,
    _rebalance_dates,
    _resolve_optional_path,
    build_market_ew_benchmark,
    build_open_to_open_returns,
    build_score,
    build_topk_weights,
    compare_full_vs_slices,
    contiguous_time_splits,
    load_config,
    load_daily_from_duckdb,
    normalize_weights,
    prepare_factors_for_backtest,
    resolve_industry_cap_and_map,
    rolling_walk_forward_windows,
    run_backtest,
    transaction_cost_params_from_mapping,
    walk_forward_backtest,
)
from scripts.light_strategy_proxy import (
    build_light_proxy_period_detail,
    infer_periods_per_year,
    summarize_light_strategy_proxy,
)
from scripts.research_identity import build_light_research_identity
from scripts.run_factor_admission_validation import (
    DEFAULT_BENCHMARK_KEY_YEARS,
    _json_sanitize,
    _summarize_oos_excess,
    _summarize_relative_to_benchmark,
    build_admission_table,
    compute_factor_gate_table,
)

IDENTITY_COLS = {"symbol", "trade_date"}
UTILITY_COLS = {
    "_universe_eligible",
    "announcement_date",
    "limit_move_hits_5d",
    "log_market_cap",
    "turnover_roll_mean",
    "price_position",
}
DEFAULT_BASELINE_FACTOR = "vol_to_turnover"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 benchmark-first 单因子 alpha scout")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="alpha_factor_scout_2026-04-20",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--config", default="config.yaml.backtest.r7_s2_prefilter_off_universe_on", help="研究基线配置")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--wf-train-window", type=int, default=252, help="滚动 WF 训练窗")
    p.add_argument("--wf-test-window", type=int, default=63, help="滚动 WF 测试窗")
    p.add_argument("--wf-step-window", type=int, default=63, help="滚动 WF 步长")
    p.add_argument("--wf-slice-splits", type=int, default=5, help="时间切片折数")
    p.add_argument("--wf-slice-min-train-days", type=int, default=252, help="时间切片最少训练窗")
    p.add_argument("--wf-slice-fixed-window", action="store_true", help="时间切片使用固定训练窗")
    p.add_argument("--top-k", type=int, default=20, help="alpha scout 统一使用的 Top-K")
    p.add_argument("--f1-min-ic", type=float, default=0.01, help="F1：T+21 IC 均值下限")
    p.add_argument("--f1-min-t", type=float, default=2.0, help="F1：T+21 IC t 值下限")
    p.add_argument(
        "--baseline-factor",
        default=DEFAULT_BASELINE_FACTOR,
        help="用于相对比较的基线单因子",
    )
    p.add_argument(
        "--factors",
        default="",
        help="只运行指定因子（逗号分隔）；为空时自动扫描可用 alpha 因子",
    )
    p.add_argument(
        "--exclude-factors",
        default="",
        help="额外排除的因子（逗号分隔）",
    )
    p.add_argument(
        "--benchmark-key-years",
        default="2021,2025,2026",
        help="benchmark-first 重点观察年份，逗号分隔",
    )
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


def classify_factor_family(factor_name: str, *, baseline_factor: str = DEFAULT_BASELINE_FACTOR) -> str:
    name = str(factor_name).strip()
    if name == baseline_factor:
        return "baseline"
    if name.startswith(("main_inflow_", "super_inflow_", "flow_divergence_", "proxy_main_inflow_", "proxy_up_down_")):
        return "fund_flow"
    if name.startswith(("holder_", "shareholder_")):
        return "shareholder"
    if name in {
        "pe_ttm",
        "pb",
        "ev_ebitda",
        "roe_ttm",
        "net_profit_yoy",
        "gross_margin_change",
        "debt_to_assets_change",
        "ocf_to_net_profit",
        "ocf_to_asset",
        "gross_margin_delta",
        "asset_turnover",
        "net_margin_stability",
        "northbound_net_inflow",
        "margin_buy_ratio",
    }:
        return "fundamental"
    if name in {"llm_sentiment", "llm_sentiment_score"} or name.startswith("llm_"):
        return "llm"
    return "price_volume"


def infer_candidate_factors(
    factor_df: pd.DataFrame,
    *,
    baseline_factor: str = DEFAULT_BASELINE_FACTOR,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    include_set = {str(item).strip() for item in (include or []) if str(item).strip()}
    exclude_set = {str(item).strip() for item in (exclude or []) if str(item).strip()}

    if include_set:
        ordered = [baseline_factor] + sorted(include_set - {baseline_factor})
        return [fac for fac in ordered if fac in factor_df.columns]

    candidates: list[str] = []
    for col in factor_df.columns:
        if col in IDENTITY_COLS or col in UTILITY_COLS or col in exclude_set:
            continue
        col_data = factor_df[col]
        if pd.api.types.is_datetime64_any_dtype(col_data):
            continue
        ser = pd.to_numeric(factor_df[col], errors="coerce")
        if ser.notna().sum() < 20:
            continue
        if not np.issubdtype(ser.dtype, np.number):
            continue
        candidates.append(str(col))

    deduped = sorted(dict.fromkeys(candidates))
    if baseline_factor in deduped:
        deduped.remove(baseline_factor)
        return [baseline_factor] + deduped
    return deduped


def find_missing_included_factors(
    factor_df: pd.DataFrame,
    *,
    baseline_factor: str = DEFAULT_BASELINE_FACTOR,
    include: list[str] | None = None,
) -> list[str]:
    requested = [baseline_factor] + [str(item).strip() for item in (include or []) if str(item).strip()]
    ordered = list(dict.fromkeys(requested))
    return [factor for factor in ordered if factor not in factor_df.columns]


def _build_scout_doc(
    summary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    scout_df: pd.DataFrame,
    output_prefix: str,
    *,
    baseline_factor: str,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    top_rows = scout_df.head(12).to_markdown(index=False) if not scout_df.empty else "_无候选结果_"
    family_counts = (
        summary_df.groupby("factor_family", dropna=False)["candidate_factor"]
        .count()
        .reset_index(name="candidate_count")
        .sort_values(["candidate_count", "factor_family"], ascending=[False, True])
    )
    family_rows = family_counts.to_markdown(index=False) if not family_counts.empty else "_无 family 统计_"
    return f"""# Alpha Scout 因子侦察

- 生成时间：`{generated_at}`
- 目标：在冻结 `V3` 执行口径的前提下，用 benchmark-first 规则快速扫描现有单因子
- 基线单因子：`{baseline_factor}`
- 固定执行：`Top-{int(summary_df['top_k'].iloc[0]) if not summary_df.empty else 20}` / `tplus1_open` / `prefilter=false` / `universe=true`
- 结果类型：`light_strategy_proxy`（用于单因子方向性和 benchmark-first proxy 判断，不等价于 full backtest）

## Scout 排名

{top_rows}

## 候选分布

{family_rows}

## 全量汇总

{summary_df.to_markdown(index=False)}

## IC 门槛

{gate_df.to_markdown(index=False)}

## 侦察结论

{scout_df.to_markdown(index=False)}

说明：

- `alpha scout` 的用途是给“新的 alpha 构造”提供起点，不直接替代正式准入结论。
- `scout_status=pass` 仍然表示该单因子同时满足 `IC gate + combo gate + benchmark-first gate`。

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_factor_gate.csv`
- `data/results/{output_prefix}_scout.csv`
- `data/results/{output_prefix}_*.json`
"""


def main() -> None:
    args = parse_args()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    baseline_factor = str(args.baseline_factor).strip()
    include = [item.strip() for item in str(args.factors).split(",") if item.strip()]
    exclude = [item.strip() for item in str(args.exclude_factors).split(",") if item.strip()]
    benchmark_key_years = [
        int(item.strip()) for item in str(args.benchmark_key_years).split(",") if str(item).strip()
    ] or list(DEFAULT_BENCHMARK_KEY_YEARS)

    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load base data: start={args.start} end={end_date}", flush=True)
    cfg, config_source = load_config(args.config)
    db_path = str(PROJECT_ROOT / str(cfg["paths"]["duckdb_path"]))
    prepared_factors_cache = _resolve_optional_path(args.prepared_factors_cache)
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
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
    base_cols = ["symbol", "trade_date", "turnover_roll_mean", "price_position", "limit_move_hits_5d", "log_market_cap"]
    factors = factors[[c for c in base_cols if c in factors.columns] + [c for c in factors.columns if c not in base_cols]].copy()
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    if prepared_factors_cache is not None:
        cache_state = "hit" if factors_cache_hit else "rebuilt"
        print(f"  prepared_factors_cache: {cache_state} -> {prepared_factors_cache}", flush=True)
    scenario_factors = factors
    missing_included = find_missing_included_factors(
        scenario_factors,
        baseline_factor=baseline_factor,
        include=include,
    )
    candidate_factors = infer_candidate_factors(
        scenario_factors,
        baseline_factor=baseline_factor,
        include=include,
        exclude=exclude,
    )
    if missing_included:
        warnings.warn(
            "以下指定因子当前不可用，已跳过："
            + ", ".join(missing_included)
            + "。这通常意味着对应数据表尚未落库，或 prepared factors 缓存仍基于旧列。",
            stacklevel=2,
        )
    if baseline_factor not in candidate_factors:
        raise SystemExit(f"baseline_factor={baseline_factor!r} 不在可用因子列中")
    print(f"  daily_df={daily_df.shape} factors={scenario_factors.shape} candidates={len(candidate_factors)}", flush=True)

    gate_df = compute_factor_gate_table(scenario_factors, daily_df, candidate_factors=candidate_factors)
    benchmark_min_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    market_benchmark = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_days).sort_index()

    signals = cfg.get("signals", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    rebalance_rule = str(backtest_cfg.get("eval_rebalance_rule", "M"))
    periods_per_year = infer_periods_per_year(rebalance_rule)
    research_identity = build_light_research_identity(
        topic="alpha_factor_scout",
        output_prefix=output_prefix,
        baseline_factor=baseline_factor,
        rebalance_rule=rebalance_rule,
        top_k=int(args.top_k),
        benchmark_key_years=benchmark_key_years,
        selector_parts={
            "pool": "explicit" if include else "auto",
            "count": len(candidate_factors),
            "factors": include[:4] if include else candidate_factors[:4],
        },
    )
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
    bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
    industry_cap_count, industry_map, industry_cap_state = resolve_industry_cap_and_map(
        int(portfolio_cfg.get("industry_cap_count", 5)),
        "data/cache/industry_map.csv",
    )

    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    open_returns.index = pd.to_datetime(open_returns.index)
    asset_returns = open_returns[
        (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
    ]
    summary_rows: list[dict[str, Any]] = []

    for idx, factor_name in enumerate(candidate_factors, start=1):
        print(f"[2/4] factor {idx}/{len(candidate_factors)}: {factor_name}", flush=True)
        score_df = build_score(scenario_factors, normalize_weights({factor_name: 1.0}))
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(args.top_k),
            rebalance_rule=rebalance_rule,
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 0.3)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]

        target_cols = sorted({str(col).zfill(6) for col in weights.columns})
        factor_asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        weights = weights.reindex(columns=target_cols, fill_value=0.0)
        first_reb = weights.index.min()
        if not factor_asset_returns.empty and first_reb > factor_asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([factor_asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        factor_asset_returns = factor_asset_returns[factor_asset_returns.index >= first_reb]

        res_wc = run_backtest(factor_asset_returns, weights, config=bt_cost)
        rolling = rolling_walk_forward_windows(
            factor_asset_returns.index,
            train_days=max(20, int(args.wf_train_window)),
            test_days=max(5, int(args.wf_test_window)),
            step_days=max(5, int(args.wf_step_window)),
        )
        _, wf_detail, wf_agg = walk_forward_backtest(
            factor_asset_returns,
            weights,
            rolling,
            config=bt_cost,
            use_test_only=True,
        )
        slices = contiguous_time_splits(
            factor_asset_returns.index,
            n_splits=max(2, int(args.wf_slice_splits)),
            min_train_days=max(20, int(args.wf_slice_min_train_days)),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )
        _, sp_detail, sp_agg = walk_forward_backtest(
            factor_asset_returns,
            weights,
            slices,
            config=bt_cost,
            use_test_only=True,
        )
        yearly_excess_df, benchmark_summary = _summarize_relative_to_benchmark(
            res_wc.daily_returns,
            market_benchmark,
            key_years=benchmark_key_years,
        )
        rolling_excess = _summarize_oos_excess(res_wc.daily_returns, market_benchmark, rolling)
        slice_excess = _summarize_oos_excess(res_wc.daily_returns, market_benchmark, slices)
        proxy_period_df = build_light_proxy_period_detail(
            res_wc.daily_returns,
            market_benchmark,
            rebalance_rule=rebalance_rule,
            scenario=f"single_{factor_name}",
        )
        proxy_summary = summarize_light_strategy_proxy(proxy_period_df, periods_per_year=periods_per_year)

        summary_rows.append(
            {
                "scenario": f"single_{factor_name}",
                "candidate_factor": factor_name,
                "family": "single_factor",
                "factor_family": classify_factor_family(factor_name, baseline_factor=baseline_factor),
                "is_baseline": factor_name == baseline_factor,
                "result_type": "light_strategy_proxy",
                "research_topic": research_identity["research_topic"],
                "research_config_id": research_identity["research_config_id"],
                "output_stem": research_identity["output_stem"],
                "rebalance_rule": rebalance_rule,
                "periods_per_year": float(periods_per_year),
                "proxy_periods": int(proxy_summary["n_periods"]),
                "top_k": int(args.top_k),
                "annualized_return": float(res_wc.panel.annualized_return),
                "proxy_annualized_return": float(proxy_summary["strategy_annualized_return"]),
                "proxy_benchmark_annualized_return": float(proxy_summary["benchmark_annualized_return"]),
                "sharpe_ratio": float(res_wc.panel.sharpe_ratio),
                "max_drawdown": float(res_wc.panel.max_drawdown),
                "turnover_mean": float(res_wc.panel.turnover_mean),
                "rolling_oos_median_ann_return": float(wf_agg.get("median_ann_return", np.nan)),
                "slice_oos_median_ann_return": float(sp_agg.get("median_ann_return", np.nan)),
                "annualized_excess_vs_market": float(proxy_summary["annualized_excess_vs_market"]),
                "full_backtest_annualized_excess_vs_market": float(
                    benchmark_summary.get("annualized_excess_return", np.nan)
                ),
                "yearly_excess_median_vs_market": float(benchmark_summary.get("yearly_excess_median", np.nan)),
                "key_year_excess_mean_vs_market": float(benchmark_summary.get("key_year_excess_mean", np.nan)),
                "key_year_excess_worst_vs_market": float(benchmark_summary.get("key_year_excess_worst", np.nan)),
                "rolling_oos_median_ann_excess_vs_market": float(
                    rolling_excess.get("median_ann_excess_return", np.nan)
                ),
                "slice_oos_median_ann_excess_vs_market": float(slice_excess.get("median_ann_excess_return", np.nan)),
            }
        )

        payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "result_type": "light_strategy_proxy",
            "research_topic": research_identity["research_topic"],
            "research_config_id": research_identity["research_config_id"],
            "output_stem": research_identity["output_stem"],
            "config_source": config_source,
            "factor_name": factor_name,
            "baseline_factor": baseline_factor,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "top_k": int(args.top_k),
                "prefilter": prefilter_cfg,
                "benchmark_key_years": benchmark_key_years,
                "industry_cap_state": industry_cap_state,
            },
            "full_sample": {"with_cost": _json_sanitize(res_wc.panel.to_dict())},
            "walk_forward_rolling": {
                "detail": _json_sanitize(wf_detail.to_dict(orient="records")),
                "agg": _json_sanitize(wf_agg),
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records")),
                "agg": _json_sanitize(sp_agg),
                "full_vs_slices": _json_sanitize(compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}),
            },
            "benchmark_first": {
                "benchmark_symbol": "market_ew_proxy",
                "benchmark_min_history_days": benchmark_min_days,
                "key_years": benchmark_key_years,
                "summary": _json_sanitize(benchmark_summary),
                "yearly_detail": _json_sanitize(yearly_excess_df.to_dict(orient="records")),
                "rolling_oos_excess": _json_sanitize(rolling_excess),
                "slice_oos_excess": _json_sanitize(slice_excess),
            },
        }
        report_path = results_dir / f"{output_prefix}_{factor_name}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["is_baseline", "annualized_excess_vs_market", "annualized_return"],
        ascending=[False, False, False],
    )
    baseline_label = f"single_{baseline_factor}"
    scout_df = build_admission_table(
        summary_df.drop(columns=["top_k"], errors="ignore"),
        gate_df,
        baseline_label=baseline_label,
        f1_min_ic=float(args.f1_min_ic),
        f1_min_t=float(args.f1_min_t),
        benchmark_key_years=benchmark_key_years,
    ).rename(columns={"admission_status": "scout_status"})
    if not gate_df.empty:
        gate_df.insert(0, "result_type", "factor_gate")
        gate_df.insert(1, "research_topic", research_identity["research_topic"])
        gate_df.insert(2, "research_config_id", research_identity["research_config_id"])
        gate_df.insert(3, "output_stem", research_identity["output_stem"])
    scout_df["result_type"] = "alpha_factor_scout"
    scout_df["research_topic"] = research_identity["research_topic"]
    scout_df["research_config_id"] = research_identity["research_config_id"]
    scout_df["output_stem"] = research_identity["output_stem"]
    scout_df = scout_df.sort_values(
        [
            "is_baseline",
            "pass_benchmark_gate",
            "delta_yearly_excess_median_vs_baseline",
            "annualized_excess_vs_market",
            "close21_ic_mean",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    gate_path = results_dir / f"{output_prefix}_factor_gate.csv"
    scout_path = results_dir / f"{output_prefix}_scout.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    scout_df.to_csv(scout_path, index=False, encoding="utf-8-sig")

    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text(
        _build_scout_doc(
            summary_df,
            gate_df,
            scout_df,
            output_prefix,
            baseline_factor=baseline_factor,
        ),
        encoding="utf-8",
    )

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "result_type": "research_manifest",
        **research_identity,
        "baseline_factor": baseline_factor,
        "candidate_factors": candidate_factors,
        "artifacts": [
            str(summary_path.relative_to(PROJECT_ROOT)),
            str(gate_path.relative_to(PROJECT_ROOT)),
            str(scout_path.relative_to(PROJECT_ROOT)),
            str(doc_path.relative_to(PROJECT_ROOT)),
        ]
        + [str((results_dir / f"{output_prefix}_{factor}.json").relative_to(PROJECT_ROOT)) for factor in candidate_factors],
    }
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] gate: {gate_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] scout: {scout_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
