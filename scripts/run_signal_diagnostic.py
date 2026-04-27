"""独立的 signal diagnostic 入口，用于快速输出 canonical 轻量诊断指标。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research_identity import build_light_research_identity

DEFAULT_BASELINE_FACTOR = "vol_to_turnover"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 benchmark-first signal diagnostic")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="signal_diagnostic",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--config", default="config.yaml.backtest.r7_s2_prefilter_off_universe_on", help="研究基线配置")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--top-k", type=int, default=20, help="统一使用的 Top-K")
    p.add_argument("--baseline-factor", default=DEFAULT_BASELINE_FACTOR, help="用于对照的基线单因子")
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
        "--rebalance-rule",
        default="",
        help="覆盖研究基线中的调仓频率，如 M/W/2W；为空时读取配置",
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


def build_signal_diagnostic_research_identity(
    *,
    output_prefix: str,
    baseline_factor: str,
    rebalance_rule: str,
    top_k: int,
    start_date: str,
    end_date: str,
    include: list[str],
    exclude: list[str],
    candidate_count: int,
) -> dict[str, str]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    benchmark_years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    return build_light_research_identity(
        topic="signal_diagnostic",
        output_prefix=output_prefix,
        baseline_factor=baseline_factor,
        rebalance_rule=rebalance_rule,
        top_k=top_k,
        benchmark_key_years=benchmark_years,
        selector_parts={
            "pool": "explicit" if include else "auto",
            "count": int(candidate_count),
            "include": include[:4],
            "exclude": exclude[:4],
        },
    )


def _build_doc(
    summary_df: pd.DataFrame,
    *,
    baseline_factor: str,
    research_identity: dict[str, str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    ranked = (
        summary_df.sort_values(
            ["is_baseline", "annualized_excess_vs_market", "strategy_sharpe_ratio"],
            ascending=[False, False, False],
        )
        if not summary_df.empty
        else summary_df
    )
    top_rows = ranked.head(12).to_markdown(index=False) if not ranked.empty else "_无诊断结果_"
    return f"""# Signal Diagnostic

- 生成时间：`{generated_at}`
- 目标：在 scout/full backtest 之前，先用统一口径输出单因子的频率感知轻量诊断
- 基线单因子：`{baseline_factor}`
- 结果类型：`signal_diagnostic`
- 研究主题：`{research_identity["research_topic"]}`
- 研究配置：`{research_identity["research_config_id"]}`
- 输出 stem：`{research_identity["output_stem"]}`

## Top Rows

{top_rows}

## 说明

- `signal_diagnostic` 只回答“这个信号在当前调仓频率下是否有方向性、波动和相对基准表现”。
- 这一层不替代 `scout / admission / full backtest`，只是把轻量诊断变成独立、稳定的 canonical 产物。

## 本轮产物

- `data/results/{research_identity["output_stem"]}_summary.csv`
- `data/results/{research_identity["output_stem"]}_period_detail.csv`
- `docs/{research_identity["output_stem"]}.md`
"""


def main() -> None:
    from scripts.light_strategy_proxy import (
        build_light_proxy_period_detail,
        infer_periods_per_year,
        summarize_signal_diagnostic,
    )
    from scripts.run_alpha_factor_scout import (
        classify_factor_family,
        find_missing_included_factors,
        infer_candidate_factors,
    )
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
        normalize_weights,
        prepare_factors_for_backtest,
        resolve_industry_cap_and_map,
        run_backtest,
        transaction_cost_params_from_mapping,
    )

    args = parse_args()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    baseline_factor = str(args.baseline_factor).strip()
    include = [item.strip() for item in str(args.factors).split(",") if item.strip()]
    exclude = [item.strip() for item in str(args.exclude_factors).split(",") if item.strip()]

    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] load base data: start={args.start} end={end_date}", flush=True)
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

    backtest_cfg = cfg.get("backtest", {}) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
    rebalance_rule = str(args.rebalance_rule).strip() or str(backtest_cfg.get("eval_rebalance_rule", "M"))
    periods_per_year = infer_periods_per_year(rebalance_rule)

    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), rebalance_rule))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    if prepared_factors_cache is not None:
        cache_state = "hit" if factors_cache_hit else "rebuilt"
        print(f"  prepared_factors_cache: {cache_state} -> {prepared_factors_cache}", flush=True)

    missing_included = find_missing_included_factors(
        factors,
        baseline_factor=baseline_factor,
        include=include,
    )
    candidate_factors = infer_candidate_factors(
        factors,
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
    print(f"  daily_df={daily_df.shape} factors={factors.shape} candidates={len(candidate_factors)}", flush=True)
    research_identity = build_signal_diagnostic_research_identity(
        output_prefix=output_prefix,
        baseline_factor=baseline_factor,
        rebalance_rule=rebalance_rule,
        top_k=int(args.top_k),
        start_date=args.start,
        end_date=end_date,
        include=include,
        exclude=exclude,
        candidate_count=len(candidate_factors),
    )

    benchmark_min_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    market_benchmark = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_days).sort_index()
    open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
    open_returns.index = pd.to_datetime(open_returns.index)
    asset_returns = open_returns[
        (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
    ]
    costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
    execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
    bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
    industry_cap_count, industry_map, _ = resolve_industry_cap_and_map(
        int(portfolio_cfg.get("industry_cap_count", 5)),
        "data/cache/industry_map.csv",
    )

    summary_rows: list[dict[str, Any]] = []
    period_detail_rows: list[dict[str, Any]] = []
    print("[2/3] run signal diagnostic", flush=True)
    for idx, factor_name in enumerate(candidate_factors, start=1):
        print(f"  factor {idx}/{len(candidate_factors)}: {factor_name}", flush=True)
        score_df = build_score(factors, normalize_weights({factor_name: 1.0}))
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=factors,
            daily_df=daily_df,
            top_k=int(args.top_k),
            rebalance_rule=rebalance_rule,
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 1.0)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]
        if weights.empty:
            continue

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
        if factor_asset_returns.empty:
            continue

        res_wc = run_backtest(factor_asset_returns, weights, config=bt_cost)
        period_df = build_light_proxy_period_detail(
            res_wc.daily_returns,
            market_benchmark,
            rebalance_rule=rebalance_rule,
            scenario=f"single_{factor_name}",
        )
        diag = summarize_signal_diagnostic(period_df, periods_per_year=periods_per_year)
        summary_rows.append(
            {
                "scenario": f"single_{factor_name}",
                "signal_name": factor_name,
                "factor_family": classify_factor_family(factor_name, baseline_factor=baseline_factor),
                "is_baseline": factor_name == baseline_factor,
                "result_type": "signal_diagnostic",
                "research_topic": research_identity["research_topic"],
                "research_config_id": research_identity["research_config_id"],
                "output_stem": research_identity["output_stem"],
                "rebalance_rule": rebalance_rule,
                "periods_per_year": float(periods_per_year),
                "top_k": int(args.top_k),
                "diagnostic_periods": int(diag["n_periods"]),
                **diag,
            }
        )
        if not period_df.empty:
            detail = period_df.copy()
            detail.insert(0, "signal_name", factor_name)
            detail.insert(1, "result_type", "signal_diagnostic")
            detail.insert(2, "research_topic", research_identity["research_topic"])
            detail.insert(3, "research_config_id", research_identity["research_config_id"])
            detail.insert(4, "output_stem", research_identity["output_stem"])
            period_detail_rows.extend(detail.to_dict(orient="records"))

        payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "config_source": config_source,
            "signal_name": factor_name,
            "result_type": "signal_diagnostic",
            **research_identity,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "top_k": int(args.top_k),
                "rebalance_rule": rebalance_rule,
            },
            "summary": diag,
        }
        out_json = results_dir / f"{research_identity['output_stem']}_{factor_name}.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(period_detail_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["is_baseline", "annualized_excess_vs_market", "strategy_sharpe_ratio"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    print("[3/3] write outputs", flush=True)
    output_stem = research_identity["output_stem"]
    summary_path = results_dir / f"{output_stem}_summary.csv"
    detail_path = results_dir / f"{output_stem}_period_detail.csv"
    doc_path = docs_dir / f"{output_stem}.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    doc_path.write_text(
        _build_doc(
            summary_df,
            baseline_factor=baseline_factor,
            research_identity=research_identity,
        ),
        encoding="utf-8",
    )
    print(f"  summary -> {summary_path}", flush=True)
    print(f"  period detail -> {detail_path}", flush=True)
    print(f"  doc -> {doc_path}", flush=True)


if __name__ == "__main__":
    main()
