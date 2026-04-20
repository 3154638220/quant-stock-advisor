"""验证 alpha 新表达：小权重反向 + 残差化 / 条件化。"""

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
from scripts.run_factor_admission_validation import (
    DEFAULT_BENCHMARK_KEY_YEARS,
    _json_sanitize,
    _summarize_oos_excess,
    _summarize_relative_to_benchmark,
    build_admission_table,
    compute_factor_gate_table,
)

DEFAULT_CANDIDATES = ["realized_vol", "intraday_range", "tail_strength", "bias_short"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 alpha expression scout（反向小权重 + 残差化/条件化）")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="alpha_expression_scout_2026-04-20",
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
    p.add_argument("--top-k", type=int, default=20, help="统一使用的 Top-K")
    p.add_argument("--f1-min-ic", type=float, default=0.01, help="F1：T+21 IC 均值下限")
    p.add_argument("--f1-min-t", type=float, default=2.0, help="F1：T+21 IC t 值下限")
    p.add_argument("--baseline-factor", default=DEFAULT_BASELINE_FACTOR, help="用于相对比较的基线单因子")
    p.add_argument(
        "--candidate-source-scout",
        default="",
        help="可选：从既有 expression scout CSV 读取候选来源；默认会过滤掉 overlay_only",
    )
    p.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="候选反向因子，逗号分隔",
    )
    p.add_argument(
        "--blend-weights",
        default="0.10,0.20",
        help="相对基线的 blend 权重，逗号分隔",
    )
    p.add_argument(
        "--condition-z-thresholds",
        default="0.5,1.0",
        help="条件化启用阈值（候选因子横截面 z 分数下限），逗号分隔",
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
    p.add_argument(
        "--allow-overlay-candidates",
        action="store_true",
        help="允许从既有 scout 结果中继续纳入 promotion_decision=overlay_only 的来源候选",
    )
    return p.parse_args()


def _zscore_series(s: pd.Series) -> pd.Series:
    ser = pd.to_numeric(s, errors="coerce")
    mu = float(ser.mean()) if ser.notna().any() else np.nan
    sigma = float(ser.std(ddof=0)) if ser.notna().any() else np.nan
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return pd.Series(np.zeros(len(ser)), index=ser.index, dtype=float)
    out = (ser - mu) / sigma
    return out.replace([np.inf, -np.inf], np.nan)


def cross_sectional_residualize(
    factor_df: pd.DataFrame,
    *,
    target_col: str,
    control_cols: list[str],
    date_col: str = "trade_date",
    suffix: str = "_resid",
    min_obs: int = 8,
) -> pd.DataFrame:
    """按日期做横截面回归残差化。"""
    out = factor_df.copy()
    result_col = f"{target_col}{suffix}"
    residuals = pd.Series(index=out.index, dtype=float)

    for _, grp in out.groupby(date_col, sort=False):
        idx = grp.index
        y = pd.to_numeric(grp[target_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(y)
        design_cols: list[np.ndarray] = []
        for col in control_cols:
            if col not in grp.columns:
                continue
            x = pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=np.float64)
            x_valid = np.isfinite(x)
            valid &= x_valid
            design_cols.append(np.where(x_valid, x, 0.0))

        if not design_cols or int(valid.sum()) < max(int(min_obs), len(design_cols) + 2):
            residuals.loc[idx] = y - np.nanmean(y)
            continue

        X = np.column_stack(design_cols)
        X = np.hstack([np.ones((len(grp), 1), dtype=np.float64), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
            residuals.loc[idx] = y - (X @ beta)
        except np.linalg.LinAlgError:
            residuals.loc[idx] = y - np.nanmean(y[valid])

    out[result_col] = residuals
    return out


def build_conditioned_flip_factor(
    factor_df: pd.DataFrame,
    *,
    factor_col: str,
    z_threshold: float,
    date_col: str = "trade_date",
    suffix: str = "_cond_flip",
) -> pd.DataFrame:
    """只在横截面高位区间保留候选因子，供反向小权重惩罚使用。"""
    out = factor_df.copy()
    result_col = f"{factor_col}{suffix}"
    rows: list[pd.Series] = []
    for _, grp in out.groupby(date_col, sort=False):
        z = _zscore_series(grp[factor_col])
        conditioned = grp[factor_col].where(z >= float(z_threshold), 0.0)
        rows.append(pd.Series(conditioned.values, index=grp.index))
    if rows:
        out[result_col] = pd.concat(rows).sort_index()
    else:
        out[result_col] = np.nan
    return out


def build_expression_gate_table(
    gate_df: pd.DataFrame,
    *,
    baseline_factor: str,
    expression_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    spec_df = pd.DataFrame(expression_specs)
    if spec_df.empty:
        return gate_df.sort_values(["factor", "horizon_key"]).reset_index(drop=True)
    spec_df = spec_df.rename(columns={"expression_factor": "factor", "family": "expression_family"})
    gate_df = gate_df.merge(
        spec_df[["factor", "source_factor", "expression_family"]].drop_duplicates(),
        on="factor",
        how="left",
    )
    return gate_df.sort_values(["factor", "horizon_key"]).reset_index(drop=True)


def compute_expression_gate_table(
    factor_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    baseline_factor: str,
    expression_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    candidate_factors = [baseline_factor] + [str(spec["expression_factor"]) for spec in expression_specs]
    gate_df = compute_factor_gate_table(factor_df, daily_df, candidate_factors=candidate_factors)
    return build_expression_gate_table(
        gate_df,
        baseline_factor=baseline_factor,
        expression_specs=expression_specs,
    )


def build_expression_scenarios(
    *,
    baseline_factor: str,
    candidates: list[str],
    blend_weights: list[float],
    condition_z_thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_defs: list[dict[str, Any]] = [
        {
            "scenario": f"baseline_{baseline_factor}",
            "candidate_factor": baseline_factor,
            "base_factor": baseline_factor,
            "family": "baseline",
            "expression_type": "single",
            "factor_weight": 1.0,
            "score_weights": normalize_weights({baseline_factor: 1.0}),
            "is_baseline": True,
        }
    ]
    expression_specs: list[dict[str, Any]] = []

    for factor_name in candidates:
        resid_factor = f"flip_resid_{factor_name}"
        expression_specs.append(
            {
                "expression_factor": resid_factor,
                "source_factor": factor_name,
                "family": "flip_residual",
            }
        )
        for w in blend_weights:
            pct = int(round(float(w) * 100))
            scenario_defs.append(
                {
                    "scenario": f"flip_resid_blend_{pct}_{factor_name}",
                    "candidate_factor": resid_factor,
                    "base_factor": factor_name,
                    "family": "flip_residual",
                    "expression_type": "residual_blend",
                    "factor_weight": -float(w),
                    "score_weights": normalize_weights({baseline_factor: 1.0 - float(w), resid_factor: -float(w)}),
                    "is_baseline": False,
                }
            )
        for z_threshold in condition_z_thresholds:
            cond_factor = f"flip_cond_z{str(z_threshold).replace('.', 'p')}_{factor_name}"
            expression_specs.append(
                {
                    "expression_factor": cond_factor,
                    "source_factor": factor_name,
                    "family": "flip_conditioned",
                }
            )
            for w in blend_weights:
                pct = int(round(float(w) * 100))
                scenario_defs.append(
                    {
                        "scenario": f"flip_cond_z{str(z_threshold).replace('.', 'p')}_blend_{pct}_{factor_name}",
                        "candidate_factor": cond_factor,
                        "base_factor": factor_name,
                        "family": "flip_conditioned",
                        "expression_type": f"conditioned_z{z_threshold:g}_blend",
                        "factor_weight": -float(w),
                        "score_weights": normalize_weights({baseline_factor: 1.0 - float(w), cond_factor: -float(w)}),
                        "is_baseline": False,
                    }
                )
    return scenario_defs, expression_specs


def select_expression_source_candidates(
    scout_df: pd.DataFrame,
    *,
    baseline_factor: str,
    allow_overlay_candidates: bool,
) -> list[str]:
    if scout_df.empty:
        return []
    df = scout_df.copy()
    if "base_factor" not in df.columns or "promotion_decision" not in df.columns:
        return []
    decisions = {"mainline_eligible"}
    if allow_overlay_candidates:
        decisions.add("overlay_only")
    picked = (
        df.loc[df["promotion_decision"].astype(str).isin(decisions), "base_factor"]
        .dropna()
        .astype(str)
        .tolist()
    )
    out: list[str] = []
    for fac in picked:
        if fac == baseline_factor:
            continue
        if fac not in out:
            out.append(fac)
    return out


def resolve_expression_candidates(
    *,
    baseline_factor: str,
    candidates_arg: list[str],
    candidate_source_scout: str,
    allow_overlay_candidates: bool,
) -> tuple[list[str], str]:
    scout_path = _resolve_optional_path(candidate_source_scout)
    if scout_path is None:
        deduped = [c for c in dict.fromkeys(candidates_arg) if c and c != baseline_factor]
        return deduped, "cli"
    if not scout_path.exists():
        raise SystemExit(f"candidate_source_scout 不存在: {scout_path}")
    scout_df = pd.read_csv(scout_path, encoding="utf-8-sig")
    resolved = select_expression_source_candidates(
        scout_df,
        baseline_factor=baseline_factor,
        allow_overlay_candidates=allow_overlay_candidates,
    )
    if not resolved:
        gate_text = "mainline_eligible + overlay_only" if allow_overlay_candidates else "mainline_eligible"
        raise SystemExit(f"candidate_source_scout={scout_path} 中没有符合 {gate_text} 的 base_factor 候选")
    return resolved, f"scout:{scout_path}"


def classify_expression_promotion(scout_df: pd.DataFrame) -> pd.DataFrame:
    out = scout_df.copy()
    if out.empty:
        out["promotion_decision"] = pd.Series(dtype=object)
        out["promotion_note"] = pd.Series(dtype=object)
        return out

    out["promotion_decision"] = "reject"
    out["promotion_note"] = "未通过 benchmark-first 准入，且没有足够证据支持保留为 overlay。"
    baseline_mask = out["is_baseline"].fillna(False).astype(bool)
    pass_all_mask = (
        out["pass_f1_gate"].fillna(False).astype(bool)
        & out["pass_combo_gate"].fillna(False).astype(bool)
        & out["pass_benchmark_gate"].fillna(False).astype(bool)
    )
    overlay_mask = (
        (~baseline_mask)
        & out["family"].astype(str).str.startswith("flip_")
        & out["pass_combo_gate"].fillna(False).astype(bool)
        & (~out["pass_benchmark_gate"].fillna(False).astype(bool))
    )

    out.loc[baseline_mask, "promotion_decision"] = "baseline"
    out.loc[baseline_mask, "promotion_note"] = "当前默认研究基线。"
    out.loc[pass_all_mask, "promotion_decision"] = "mainline_eligible"
    out.loc[pass_all_mask, "promotion_note"] = "同时通过 IC、组合层与 benchmark-first，允许进入主比较矩阵。"
    out.loc[overlay_mask, "promotion_decision"] = "overlay_only"
    out.loc[overlay_mask, "promotion_note"] = "只允许作为组合层 overlay 对照保留，不得作为新主排序信号升级。"
    return out


def build_expression_promotion_summary(scout_df: pd.DataFrame) -> str:
    if scout_df.empty:
        return "- 本轮没有可总结的 expression 候选。"

    lines: list[str] = []
    baseline_rows = scout_df.loc[scout_df["promotion_decision"].astype(str) == "baseline", "scenario"].astype(str).tolist()
    if baseline_rows:
        lines.append(f"- 基线：`{baseline_rows[0]}` 继续作为当前默认研究基线。")

    mainline_rows = scout_df.loc[
        scout_df["promotion_decision"].astype(str) == "mainline_eligible",
        "scenario",
    ].astype(str).tolist()
    if mainline_rows:
        lines.append(
            "- 可升级主线："
            + "、".join(f"`{name}`" for name in mainline_rows)
            + " 已同时通过 IC、组合层与 benchmark-first，可进入主比较矩阵。"
        )
    else:
        lines.append("- 可升级主线：本轮没有 expression 候选同时通过 IC、组合层与 benchmark-first。")

    overlay_rows = scout_df.loc[
        scout_df["promotion_decision"].astype(str) == "overlay_only",
        "scenario",
    ].astype(str).tolist()
    if overlay_rows:
        lines.append(
            "- Overlay-only："
            + "、".join(f"`{name}`" for name in overlay_rows)
            + " 只允许作为组合层 overlay 对照保留，不得作为新主排序信号升级。"
        )

    reject_rows = scout_df.loc[
        scout_df["promotion_decision"].astype(str) == "reject",
        "scenario",
    ].astype(str).tolist()
    if reject_rows:
        preview = "、".join(f"`{name}`" for name in reject_rows[:5])
        suffix = " 等候选" if len(reject_rows) > 5 else ""
        lines.append(f"- Reject：{preview}{suffix} 当前未达到保留或升级条件。")
    return "\n".join(lines)


def _build_doc(
    summary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    scout_df: pd.DataFrame,
    output_prefix: str,
    *,
    baseline_factor: str,
    candidates: list[str],
    blend_weights: list[float],
    condition_z_thresholds: list[float],
    candidate_source_desc: str,
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    top_rows = scout_df.head(20).to_markdown(index=False) if not scout_df.empty else "_无候选结果_"
    promotion_summary = build_expression_promotion_summary(scout_df)
    return f"""# Alpha Expression Scout

- 生成时间：`{generated_at}`
- 目标：验证 `V3` 主干上的“小权重反向 + 残差化 / 条件化表达”
- 基线单因子：`{baseline_factor}`
- 候选来源：`{", ".join(candidates)}`
- 候选来源模式：`{candidate_source_desc}`
- 固定执行：`Top-{int(summary_df['top_k'].iloc[0]) if not summary_df.empty else 20}` / `tplus1_open` / `prefilter=false` / `universe=true`
- blend 权重：`{", ".join(f"{int(w * 100)}%" for w in blend_weights)}`
- 条件化阈值：`{", ".join(f"z>={v:g}" for v in condition_z_thresholds)}`

## Expression 排名

{top_rows}

## 全量汇总

{summary_df.to_markdown(index=False)}

## Scout Gate

{gate_df.to_markdown(index=False)}

## Expression 结论

{scout_df.to_markdown(index=False)}

## 决策摘要

{promotion_summary}

说明：

- `flip_residual` 表示先对候选因子相对 `V3` 主干做横截面残差化，再以小权重反向 blend。
- `flip_conditioned` 表示只在候选因子横截面高位区间启用惩罚，避免整列反向覆盖。
- 当前 `IC gate` 已直接基于 expression 列本身计算，不再沿用源因子翻转代理符号。
- `promotion_decision=overlay_only` 表示该表达最多只允许作为组合层 overlay 对照保留，不得进入新的主比较矩阵。

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
    candidate_arg_list = [item.strip() for item in str(args.candidates).split(",") if item.strip()]
    candidates, candidate_source_desc = resolve_expression_candidates(
        baseline_factor=baseline_factor,
        candidates_arg=candidate_arg_list,
        candidate_source_scout=str(args.candidate_source_scout),
        allow_overlay_candidates=bool(args.allow_overlay_candidates),
    )
    blend_weights = [float(item.strip()) for item in str(args.blend_weights).split(",") if item.strip()]
    condition_z_thresholds = [
        float(item.strip()) for item in str(args.condition_z_thresholds).split(",") if item.strip()
    ]
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
    print(f"  factors_raw={factors.shape}", flush=True)
    base_cols = ["symbol", "trade_date", "turnover_roll_mean", "price_position", "limit_move_hits_5d", "log_market_cap"]
    factors = factors[[c for c in base_cols if c in factors.columns] + [c for c in factors.columns if c not in base_cols]].copy()
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    print(f"  factors_rebalance={factors.shape}", flush=True)
    if prepared_factors_cache is not None:
        cache_state = "hit" if factors_cache_hit else "rebuilt"
        print(f"  prepared_factors_cache: {cache_state} -> {prepared_factors_cache}", flush=True)
    print(f"  factors_with_pit={factors.shape}", flush=True)
    scenario_factors = factors
    print(f"  scenario_factors={scenario_factors.shape}", flush=True)

    required = [baseline_factor] + candidates
    missing = [col for col in required if col not in scenario_factors.columns]
    if missing:
        raise SystemExit(f"缺少候选列: {missing}")

    gate_source_df = compute_factor_gate_table(scenario_factors, daily_df, candidate_factors=required)
    print(f"  gate_source_rows={len(gate_source_df)}", flush=True)
    scenario_factors = scenario_factors.copy()
    for factor_name in candidates:
        scenario_factors = cross_sectional_residualize(
            scenario_factors,
            target_col=factor_name,
            control_cols=[baseline_factor],
            suffix="_resid_against_v3",
        )
        scenario_factors[f"flip_resid_{factor_name}"] = scenario_factors[f"{factor_name}_resid_against_v3"]
        for z_threshold in condition_z_thresholds:
            scenario_factors = build_conditioned_flip_factor(
                scenario_factors,
                factor_col=factor_name,
                z_threshold=float(z_threshold),
                suffix=f"_cond_flip_z{str(z_threshold).replace('.', 'p')}",
            )
            scenario_factors[f"flip_cond_z{str(z_threshold).replace('.', 'p')}_{factor_name}"] = scenario_factors[
                f"{factor_name}_cond_flip_z{str(z_threshold).replace('.', 'p')}"
            ]
    print(f"  expression_columns_ready={len(scenario_factors.columns)}", flush=True)

    scenario_defs, expression_specs = build_expression_scenarios(
        baseline_factor=baseline_factor,
        candidates=candidates,
        blend_weights=blend_weights,
        condition_z_thresholds=condition_z_thresholds,
    )
    print(f"  scenarios={len(scenario_defs)} expressions={len(expression_specs)}", flush=True)

    benchmark_min_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    market_benchmark = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_days).sort_index()

    portfolio_cfg = cfg.get("portfolio", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    prefilter_cfg = cfg.get("prefilter", {}) or {}
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
    for idx, scenario in enumerate(scenario_defs, start=1):
        print(f"[2/4] scenario {idx}/{len(scenario_defs)}: {scenario['scenario']}", flush=True)
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

        res_wc = run_backtest(scenario_asset_returns, weights, config=bt_cost)
        rolling = rolling_walk_forward_windows(
            scenario_asset_returns.index,
            train_days=max(20, int(args.wf_train_window)),
            test_days=max(5, int(args.wf_test_window)),
            step_days=max(5, int(args.wf_step_window)),
        )
        _, wf_detail, wf_agg = walk_forward_backtest(
            scenario_asset_returns,
            weights,
            rolling,
            config=bt_cost,
            use_test_only=True,
        )
        slices = contiguous_time_splits(
            scenario_asset_returns.index,
            n_splits=max(2, int(args.wf_slice_splits)),
            min_train_days=max(20, int(args.wf_slice_min_train_days)),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )
        _, sp_detail, sp_agg = walk_forward_backtest(
            scenario_asset_returns,
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

        summary_rows.append(
            {
                "scenario": scenario["scenario"],
                "candidate_factor": scenario["candidate_factor"],
                "base_factor": scenario["base_factor"],
                "family": scenario["family"],
                "expression_type": scenario["expression_type"],
                "factor_weight": float(scenario["factor_weight"]),
                "is_baseline": bool(scenario["is_baseline"]),
                "top_k": int(args.top_k),
                "annualized_return": float(res_wc.panel.annualized_return),
                "sharpe_ratio": float(res_wc.panel.sharpe_ratio),
                "max_drawdown": float(res_wc.panel.max_drawdown),
                "turnover_mean": float(res_wc.panel.turnover_mean),
                "rolling_oos_median_ann_return": float(wf_agg.get("median_ann_return", np.nan)),
                "slice_oos_median_ann_return": float(sp_agg.get("median_ann_return", np.nan)),
                "annualized_excess_vs_market": float(benchmark_summary.get("annualized_excess_return", np.nan)),
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
            "config_source": config_source,
            "scenario": scenario,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "top_k": int(args.top_k),
                "candidates": candidates,
                "blend_weights": blend_weights,
                "condition_z_thresholds": condition_z_thresholds,
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
        report_path = results_dir / f"{output_prefix}_{scenario['scenario']}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["is_baseline", "annualized_excess_vs_market", "annualized_return"],
        ascending=[False, False, False],
    )
    gate_df = compute_expression_gate_table(
        scenario_factors,
        daily_df,
        baseline_factor=baseline_factor,
        expression_specs=expression_specs,
    )
    baseline_label = f"baseline_{baseline_factor}"
    scout_df = build_admission_table(
        summary_df.drop(columns=["top_k"], errors="ignore"),
        gate_df,
        baseline_label=baseline_label,
        f1_min_ic=float(args.f1_min_ic),
        f1_min_t=float(args.f1_min_t),
        benchmark_key_years=benchmark_key_years,
    ).rename(columns={"admission_status": "scout_status"})
    scout_df = classify_expression_promotion(scout_df)
    scout_df = scout_df.sort_values(
        [
            "is_baseline",
            "promotion_decision",
            "pass_benchmark_gate",
            "delta_yearly_excess_median_vs_baseline",
            "annualized_excess_vs_market",
            "close21_ic_mean",
        ],
        ascending=[False, True, False, False, False, False],
    ).reset_index(drop=True)

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    gate_path = results_dir / f"{output_prefix}_factor_gate.csv"
    scout_path = results_dir / f"{output_prefix}_scout.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    scout_df.to_csv(scout_path, index=False, encoding="utf-8-sig")

    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text(
        _build_doc(
            summary_df,
            gate_df,
            scout_df,
            output_prefix,
            baseline_factor=baseline_factor,
            candidates=candidates,
            blend_weights=blend_weights,
            condition_z_thresholds=condition_z_thresholds,
            candidate_source_desc=candidate_source_desc,
        ),
        encoding="utf-8",
    )

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "baseline_factor": baseline_factor,
        "candidates": candidates,
        "candidate_source_desc": candidate_source_desc,
        "artifacts": [
            str(summary_path.relative_to(PROJECT_ROOT)),
            str(gate_path.relative_to(PROJECT_ROOT)),
            str(scout_path.relative_to(PROJECT_ROOT)),
            str(doc_path.relative_to(PROJECT_ROOT)),
        ]
        + [
            str((results_dir / f"{output_prefix}_{scenario['scenario']}.json").relative_to(PROJECT_ROOT))
            for scenario in scenario_defs
        ],
    }
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] gate: {gate_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] scout: {scout_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
