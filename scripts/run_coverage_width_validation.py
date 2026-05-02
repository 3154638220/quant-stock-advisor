"""覆盖宽度 / 持有机制验证：B1/B2/B3/B4 对比 V3。"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from scripts.run_backtest_eval import (
    BacktestConfig,
    _attach_pit_fundamentals,
    attach_universe_filter,
    build_market_ew_benchmark,
    build_open_to_open_returns,
    build_score,
    build_topk_weights,
    compare_full_vs_slices,
    compute_factors,
    contiguous_time_splits,
    load_config,
    load_daily_from_duckdb,
    normalize_weights,
    rolling_walk_forward_windows,
    run_backtest,
    transaction_cost_params_from_mapping,
)
from scripts.run_p1_residual_diagnostics import summarize_rebalance_churn, summarize_weight_profile
from src.backtest.performance_panel import compute_performance_panel
from src.backtest.walk_forward import run_backtest_on_index, walk_forward_backtest
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    utc_now_iso,
    write_research_manifest,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    config_path: str
    is_baseline: bool = False
    signals_overrides: dict[str, Any] | None = None
    portfolio_overrides: dict[str, Any] | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("v3", "V3_baseline_top20", "config.yaml.backtest.r7_s2_prefilter_off_universe_on", is_baseline=True),
    Scenario("b1", "B1_top30", "config.yaml.backtest.b1_s2_top30"),
    Scenario("b2", "B2_top40", "config.yaml.backtest.b2_s2_top40"),
    Scenario("b3", "B3_buffer_20_40", "config.yaml.backtest.b3_s2_buffer_2040"),
    Scenario("b4", "B4_tiered_60_40_buffer_20_40", "config.yaml.backtest.b4_s2_tiered_buffer_2040"),
)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if str(item).strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if str(item).strip()]


def build_variant_scenarios(
    *,
    top_k: int,
    variant_mode: str,
    entry_top_k_values: list[int],
    hold_buffer_top_k_values: list[int],
    top_tier_count_values: list[int],
    top_tier_weight_share_values: list[float],
    base_config_path: str = "config.yaml.backtest.r7_s2_prefilter_off_universe_on",
) -> list[Scenario]:
    mode = str(variant_mode).strip().lower()
    if mode in ("", "none"):
        return []

    variants: list[Scenario] = []
    entry_values = sorted({int(v) for v in entry_top_k_values if int(v) > 0})
    hold_values = sorted({max(int(v), int(top_k)) for v in hold_buffer_top_k_values if int(v) > 0})
    tier_counts = sorted({int(v) for v in top_tier_count_values if 0 < int(v) < int(top_k)})
    tier_shares = sorted(
        {
            round(float(v), 6)
            for v in top_tier_weight_share_values
            if 0.0 < float(v) < 1.0
        }
    )

    if mode in ("b3", "both"):
        for entry_top_k in entry_values:
            for hold_buffer_top_k in hold_values:
                key = f"vb3_{entry_top_k}_{hold_buffer_top_k}"
                label = f"VB3_buffer_{entry_top_k}_{hold_buffer_top_k}"
                variants.append(
                    Scenario(
                        key=key,
                        label=label,
                        config_path=base_config_path,
                        signals_overrides={"top_k": int(top_k)},
                        portfolio_overrides={
                            "weight_method": "equal_weight",
                            "entry_top_k": int(entry_top_k),
                            "hold_buffer_top_k": int(hold_buffer_top_k),
                        },
                    )
                )

    if mode in ("b4", "both"):
        for entry_top_k in entry_values:
            for hold_buffer_top_k in hold_values:
                for top_tier_count in tier_counts:
                    for top_tier_weight_share in tier_shares:
                        share_pct = int(round(float(top_tier_weight_share) * 100))
                        key = f"vb4_{entry_top_k}_{hold_buffer_top_k}_t{top_tier_count}_s{share_pct}"
                        label = (
                            f"VB4_tiered_t{top_tier_count}_s{share_pct}_buffer_{entry_top_k}_{hold_buffer_top_k}"
                        )
                        variants.append(
                            Scenario(
                                key=key,
                                label=label,
                                config_path=base_config_path,
                                signals_overrides={"top_k": int(top_k)},
                                portfolio_overrides={
                                    "weight_method": "tiered_equal_weight",
                                    "entry_top_k": int(entry_top_k),
                                    "hold_buffer_top_k": int(hold_buffer_top_k),
                                    "top_tier_count": int(top_tier_count),
                                    "top_tier_weight_share": float(top_tier_weight_share),
                                },
                            )
                        )

    return variants


def select_scenarios(keys: list[str] | None = None, candidates: list[Scenario] | None = None) -> list[Scenario]:
    base_scenarios = list(candidates) if candidates is not None else list(SCENARIOS)
    selected = [scenario for scenario in base_scenarios if scenario.is_baseline]
    if not keys:
        selected.extend(scenario for scenario in base_scenarios if not scenario.is_baseline)
        return selected

    key_set = {str(key).strip().lower() for key in keys if str(key).strip()}
    selected.extend(
        scenario
        for scenario in base_scenarios
        if (not scenario.is_baseline) and scenario.key.lower() in key_set
    )
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行覆盖宽度 / 持有机制验证")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="coverage_width_validation_2026-04-20",
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
        help="只运行指定场景 key（逗号分隔，如 b3,b4）；为空则运行 V3/B1/B2/B3/B4 全部",
    )
    p.add_argument(
        "--variant-mode",
        default="none",
        choices=("none", "b3", "b4", "both"),
        help="附加生成 B3/B4 小范围变体；默认不生成",
    )
    p.add_argument(
        "--variant-entry-top-k-values",
        default="20",
        help="变体用 entry_top_k 列表，逗号分隔",
    )
    p.add_argument(
        "--variant-hold-buffer-top-k-values",
        default="35,40,50",
        help="变体用 hold_buffer_top_k 列表，逗号分隔",
    )
    p.add_argument(
        "--variant-top-tier-count-values",
        default="8,10,12",
        help="B4 变体用 top_tier_count 列表，逗号分隔",
    )
    p.add_argument(
        "--variant-top-tier-weight-share-values",
        default="0.55,0.60,0.65",
        help="B4 变体用 top_tier_weight_share 列表，逗号分隔",
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def materialize_scenario_config(
    scenario: Scenario,
    cfg: dict[str, Any],
    *,
    root_dir: Path = PROJECT_ROOT,
) -> str:
    if not scenario.signals_overrides and not scenario.portfolio_overrides:
        return scenario.config_path

    snapshot_name = f"config.yaml.backtest.{scenario.key}"
    snapshot_rel_path = Path("configs") / "backtests" / snapshot_name
    snapshot_path = root_dir / snapshot_rel_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _deep_merge(
        cfg,
        {
            "signals": scenario.signals_overrides or {},
            "portfolio": scenario.portfolio_overrides or {},
        },
    )
    snapshot_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return snapshot_rel_path.as_posix()


def summarize_yearly_excess(strategy_daily: pd.Series, benchmark_daily: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = strategy_daily.index.intersection(benchmark_daily.index)
    strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0).sort_index()
    bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0).sort_index()
    strat_y = strat.groupby(strat.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))
    bench_y = bench.groupby(bench.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))
    rows: list[dict[str, Any]] = []
    for year, value in strat_y.items():
        bval = float(bench_y.get(year, np.nan))
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(value),
                "benchmark_return": bval,
                "excess_return": float(value - bval) if np.isfinite(bval) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "median_yearly_return": float(out["strategy_return"].median()) if not out.empty else np.nan,
        "median_yearly_excess_return": float(out["excess_return"].median()) if not out.empty else np.nan,
        "best_year_excess_return": float(out["excess_return"].max()) if not out.empty else np.nan,
        "worst_year_excess_return": float(out["excess_return"].min()) if not out.empty else np.nan,
    }
    return out, summary


def summarize_oos_excess(
    *,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    benchmark: pd.Series,
    slices: list,
    bt_cfg: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sl in slices:
        try:
            res = run_backtest_on_index(asset_returns, weights, sl.test_index, config=bt_cfg)
        except ValueError:
            continue
        common = res.daily_returns.index.intersection(benchmark.index)
        strat = res.daily_returns.reindex(common).fillna(0.0)
        bench = benchmark.reindex(common).fillna(0.0)
        panel = compute_performance_panel((strat - bench).to_numpy(dtype=np.float64), periods_per_year=252.0)
        rows.append(
            {
                "fold_id": int(sl.fold_id),
                "n_test": int(len(sl.test_index)),
                "annualized_excess_return": float(panel.annualized_return),
                "excess_sharpe_ratio": float(panel.sharpe_ratio),
                "excess_max_drawdown": float(panel.max_drawdown),
                "excess_total_return": float(panel.total_return),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {"n_folds": 0}
    summary = {
        "n_folds": int(len(detail)),
        "median_ann_excess_return": float(detail["annualized_excess_return"].median()),
        "p25_ann_excess_return": float(detail["annualized_excess_return"].quantile(0.25)),
        "p75_ann_excess_return": float(detail["annualized_excess_return"].quantile(0.75)),
        "mean_ann_excess_return": float(detail["annualized_excess_return"].mean()),
    }
    return detail, summary


def _build_doc(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, output_prefix: str, scenarios: list[Scenario]) -> str:
    summary_md = summary_df.to_markdown(index=False)
    yearly_md = yearly_df.to_markdown(index=False) if not yearly_df.empty else "_无年度数据_"
    generated_at = pd.Timestamp.utcnow().isoformat()
    scenario_desc = "、".join(f"`{s.label}`" for s in scenarios)
    return f"""# 覆盖宽度 / 持有机制验证

- 生成时间：`{generated_at}`
- 固定主线：`S2 = vol_to_turnover` / `prefilter=false` / `universe=true` / `tplus1_open`
- 对照矩阵：{scenario_desc}

## 汇总

{summary_md}

## 年度超额

{yearly_md}

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_yearly.csv`
- `data/results/{output_prefix}_*.json`
"""


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    scenario_keys = [item.strip() for item in str(args.scenarios).split(",") if item.strip()]
    variant_scenarios = build_variant_scenarios(
        top_k=20,
        variant_mode=str(args.variant_mode),
        entry_top_k_values=_parse_csv_ints(args.variant_entry_top_k_values),
        hold_buffer_top_k_values=_parse_csv_ints(args.variant_hold_buffer_top_k_values),
        top_tier_count_values=_parse_csv_ints(args.variant_top_tier_count_values),
        top_tier_weight_share_values=_parse_csv_floats(args.variant_top_tier_weight_share_values),
    )
    available_scenarios = list(SCENARIOS) + variant_scenarios
    selected_scenarios = select_scenarios(scenario_keys, available_scenarios)
    if not selected_scenarios:
        raise SystemExit("未匹配到任何 coverage 场景；请检查 --scenarios 参数")
    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load base data: start={args.start} end={end_date}", flush=True)
    base_cfg, _ = load_config(selected_scenarios[0].config_path)
    db_path = str(PROJECT_ROOT / str(base_cfg["paths"]["duckdb_path"]))
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    print(f"  daily_df={daily_df.shape} factors={factors.shape}", flush=True)

    benchmark_min_history_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    benchmark_market = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_history_days)
    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for idx, scenario in enumerate(selected_scenarios, start=1):
        print(f"[2/4] scenario {idx}/{len(selected_scenarios)}: {scenario.key} ({scenario.label})", flush=True)
        cfg, config_source = load_config(scenario.config_path)
        signals = {**(cfg.get("signals", {}) or {}), **(scenario.signals_overrides or {})}
        portfolio_cfg = {**(cfg.get("portfolio", {}) or {}), **(scenario.portfolio_overrides or {})}
        effective_config_path = materialize_scenario_config(scenario, cfg)
        backtest_cfg = cfg.get("backtest", {}) or {}
        prefilter_cfg = cfg.get("prefilter", {}) or {}
        uf_cfg = cfg.get("universe_filter", {}) or {}
        risk_cfg = cfg.get("risk", {}) or {}

        scenario_factors = attach_universe_filter(
            factors,
            daily_df,
            enabled=bool(uf_cfg.get("enabled", False)),
            min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
            require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
        )
        ce_weights = normalize_weights(signals.get("composite_extended", {}))
        score_df = build_score(scenario_factors, ce_weights)
        score_df = score_df[score_df["trade_date"] >= pd.Timestamp(args.start)].copy()
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(signals.get("top_k", 20)),
            rebalance_rule=str(backtest_cfg.get("eval_rebalance_rule", "M")),
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 1.0)),
            entry_top_k=portfolio_cfg.get("entry_top_k"),
            hold_buffer_top_k=portfolio_cfg.get("hold_buffer_top_k"),
            top_tier_count=portfolio_cfg.get("top_tier_count"),
            top_tier_weight_share=portfolio_cfg.get("top_tier_weight_share"),
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
        )

        open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False).sort_index()
        asset_returns = open_returns.reindex(columns=sorted(set(weights.columns.astype(str)))).fillna(0.0)
        asset_returns = asset_returns[
            (asset_returns.index >= pd.Timestamp(args.start)) & (asset_returns.index <= pd.Timestamp(end_date))
        ]
        weights = weights.reindex(columns=asset_returns.columns, fill_value=0.0)
        if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        asset_returns = asset_returns[asset_returns.index >= weights.index.min()]
        rolling = rolling_walk_forward_windows(
            asset_returns.index,
            train_days=int(args.wf_train_window),
            test_days=int(args.wf_test_window),
            step_days=int(args.wf_step_window),
        )
        slices = contiguous_time_splits(
            asset_returns.index,
            n_splits=int(args.wf_slice_splits),
            min_train_days=int(args.wf_slice_min_train_days),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )

        bt_cfg = BacktestConfig(
            cost_params=transaction_cost_params_from_mapping(cfg.get("transaction_costs", {})),
            execution_mode="tplus1_open",
            execution_lag=1,
            limit_up_mode=str(backtest_cfg.get("limit_up_mode", "redistribute")),
        )
        res = run_backtest(asset_returns, weights, config=bt_cfg)
        yearly_df, yearly_summary = summarize_yearly_excess(res.daily_returns, benchmark_market)
        yearly_df.insert(0, "scenario", scenario.label)
        yearly_frames.append(yearly_df)

        _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cfg, use_test_only=True)
        _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cfg, use_test_only=True)
        wf_excess_detail, wf_excess_summary = summarize_oos_excess(
            asset_returns=asset_returns,
            weights=weights,
            benchmark=benchmark_market,
            slices=rolling,
            bt_cfg=bt_cfg,
        )
        sp_excess_detail, sp_excess_summary = summarize_oos_excess(
            asset_returns=asset_returns,
            weights=weights,
            benchmark=benchmark_market,
            slices=slices,
            bt_cfg=bt_cfg,
        )

        common = res.daily_returns.index.intersection(benchmark_market.index)
        excess_panel = compute_performance_panel(
            (res.daily_returns.reindex(common).fillna(0.0) - benchmark_market.reindex(common).fillna(0.0)).to_numpy(
                dtype=np.float64
            ),
            periods_per_year=252.0,
        )
        weight_profile = summarize_weight_profile(weights)
        churn = summarize_rebalance_churn(weights)
        out = {
            "config_source": config_source,
            "effective_config_path": effective_config_path,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "top_k": int(signals.get("top_k", 20)),
                "entry_top_k": portfolio_cfg.get("entry_top_k"),
                "hold_buffer_top_k": portfolio_cfg.get("hold_buffer_top_k"),
                "top_tier_count": portfolio_cfg.get("top_tier_count"),
                "top_tier_weight_share": portfolio_cfg.get("top_tier_weight_share"),
                "rebalance_rule": str(backtest_cfg.get("eval_rebalance_rule", "M")),
                "max_turnover": float(portfolio_cfg.get("max_turnover", 1.0)),
                "portfolio_method": str(portfolio_cfg.get("weight_method", "equal_weight")),
                "execution_mode": str(backtest_cfg.get("execution_mode", "tplus1_open")),
                "prefilter": prefilter_cfg,
                "universe_filter": uf_cfg,
                "benchmark_symbol": str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
                "benchmark_min_history_days": benchmark_min_history_days,
            },
            "full_sample": res.panel.to_dict(),
            "excess_vs_market": excess_panel.to_dict(),
            "yearly": _json_sanitize(yearly_df.to_dict(orient="records")),
            "walk_forward_rolling": {
                "detail": _json_sanitize(wf_detail.to_dict(orient="records")),
                "agg": _json_sanitize(dict(wf_agg)),
                "excess_detail": _json_sanitize(wf_excess_detail.to_dict(orient="records")),
                "excess_agg": _json_sanitize(wf_excess_summary),
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records")),
                "agg": _json_sanitize(dict(sp_agg)),
                "excess_detail": _json_sanitize(sp_excess_detail.to_dict(orient="records")),
                "excess_agg": _json_sanitize(sp_excess_summary),
                "full_vs_slices": _json_sanitize(compare_full_vs_slices(res.panel, sp_agg)),
            },
            "weight_profile": _json_sanitize(weight_profile),
            "rebalance_churn": _json_sanitize(churn),
        }
        out_path = results_dir / f"{output_prefix}_{scenario.key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_sanitize(out), f, ensure_ascii=False, indent=2)

        summary_rows.append(
            {
                "scenario": scenario.label,
                "annualized_return": float(res.panel.annualized_return),
                "sharpe_ratio": float(res.panel.sharpe_ratio),
                "max_drawdown": float(res.panel.max_drawdown),
                "turnover_mean": float(res.panel.turnover_mean),
                "excess_market_ew_ann": float(excess_panel.annualized_return),
                "median_yearly_excess_return": float(yearly_summary.get("median_yearly_excess_return", np.nan)),
                "rolling_oos_median_ann_return": float(wf_agg.get("median_ann_return", np.nan)),
                "rolling_oos_median_ann_excess_return": float(wf_excess_summary.get("median_ann_excess_return", np.nan)),
                "slice_oos_median_ann_return": float(sp_agg.get("median_ann_return", np.nan)),
                "slice_oos_median_ann_excess_return": float(sp_excess_summary.get("median_ann_excess_return", np.nan)),
                "mean_name_count": float(weight_profile.get("mean_name_count", np.nan)),
                "mean_prev_overlap_ratio": float(churn.get("mean_prev_overlap_ratio", np.nan)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df[summary_df["scenario"] == "V3_baseline_top20"].iloc[0]
    for col in [
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "excess_market_ew_ann",
        "median_yearly_excess_return",
        "rolling_oos_median_ann_excess_return",
        "slice_oos_median_ann_excess_return",
    ]:
        summary_df[f"delta_vs_v3__{col}"] = summary_df[col] - float(baseline_row[col])
    summary_df = summary_df.sort_values("scenario").reset_index(drop=True)
    yearly_all_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    yearly_path = results_dir / f"{output_prefix}_yearly.csv"
    doc_path = docs_dir / f"{output_prefix}.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly_all_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    doc_path.write_text(_build_doc(summary_df, yearly_all_df, output_prefix, selected_scenarios), encoding="utf-8")

    # --- standard research contract ---
    duration_sec = round(time.perf_counter() - started_at, 6)

    def _project_relative(p: str | Path) -> str:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT.resolve()))

    scenario_keys_list = [s.key for s in selected_scenarios]
    identity = make_research_identity(
        result_type="coverage_width_validation",
        research_topic="coverage_width_validation",
        research_config_id=f"scenarios_{slugify_token('-'.join(scenario_keys_list[:4]))}_variant_{slugify_token(args.variant_mode)}",
        output_stem=slugify_token(output_prefix),
    )
    data_slice = DataSlice(
        dataset_name="coverage_width_validation_backtest",
        source_tables=("a_share_daily",),
        date_start=args.start,
        date_end=end_date,
        asof_trade_date=end_date,
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="universe_filtered",
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=None,
        feature_columns=(),
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=str(selected_scenarios[0].config_path) if selected_scenarios else "",
        extra={
            "scenario_keys": scenario_keys_list,
            "variant_mode": args.variant_mode,
            "lookback_days": int(args.lookback_days),
        },
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(summary_path), "csv", False, "覆盖宽度汇总"),
        ArtifactRef("yearly_csv", _project_relative(yearly_path), "csv", False, "年度切片"),
        ArtifactRef("report_md", _project_relative(doc_path), "md", False, "验证报告"),
        ArtifactRef("manifest_json", _project_relative(results_dir / f"{output_prefix}_manifest.json"), "json", False),
    ) + tuple(
        ArtifactRef(
            f"scenario_{slugify_token(s.key)}_json",
            _project_relative(results_dir / f"{output_prefix}_{s.key}.json"),
            "json",
            False,
            f"场景 {s.label} 回测详情",
        )
        for s in selected_scenarios
    )
    metrics = {
        "scenario_count": len(selected_scenarios),
        "summary_rows": int(len(summary_df)),
    }
    gates = {
        "data_gate": {
            "passed": True,
        },
        "execution_gate": {
            "passed": bool(len(summary_rows) >= len(selected_scenarios)),
            "expected": len(selected_scenarios),
            "completed": len(summary_rows),
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=duration_sec,
        seed=None,
        data_slices=(data_slice,),
        config={"config_path": str(selected_scenarios[0].config_path) if selected_scenarios else ""},
        params={
            "cli": {k: str(v) for k, v in vars(args).items()},
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["coverage_width_validation_is_diagnostic_research_only"],
        },
        notes=f"Coverage width validation: {len(selected_scenarios)} scenarios.",
    )
    write_research_manifest(
        manifest_path,
        result,
        extra={
            "generated_at_utc": result.created_at,
            "legacy_artifacts": [
                _project_relative(summary_path),
                _project_relative(yearly_path),
                _project_relative(doc_path),
            ]
            + [_project_relative(results_dir / f"{output_prefix}_{s.key}.json") for s in selected_scenarios],
        },
    )
    append_experiment_result(results_dir.parent / "experiments", result)
    # --- end standard research contract ---

    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[4/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
