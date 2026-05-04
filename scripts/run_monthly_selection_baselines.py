#!/usr/bin/env python3
"""M4: 月度选股 baseline ranker。

输入 M2 canonical dataset：
``data/cache/monthly_selection_features.parquet``。

输出 price-volume-only 的可解释 baseline、walk-forward ML baseline，以及
Rank IC、Top-K 超额、分桶 spread、年度/市场状态、行业暴露和换手诊断。

核心工具已迁移至 src/pipeline/monthly_baselines.py、src/reporting/markdown_report.py
和 src/research/gates.py。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    config_snapshot,
    utc_now_iso,
    write_research_manifest,
)
from src.pipeline.monthly_baselines import (
    BLEND_SPECS,
    FEATURE_SPECS,
    ML_FEATURE_COLS,
    BaselineRunConfig,
    build_leaderboard,
    build_market_benchmark_monthly,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
    build_realized_market_states,
    build_static_scores,
    build_summary_payload,
    build_walk_forward_scores,
    load_baseline_dataset,
    model_n_jobs_token,
    normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
    summarize_feature_importance,
    summarize_industry_exposure,
    summarize_regime_slice,
    summarize_year_slice,
    valid_pool_frame,
)
from src.reporting.markdown_report import (
    format_markdown_table,
    json_sanitize,
    project_relative,
)
from src.research.gates import (
    EXCESS_COL,
    INDUSTRY_EXCESS_COL,
    LABEL_COL,
    MARKET_COL,
    POOL_RULES,
    TOP20_COL,
)
from src.settings import config_path_candidates, load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M4 baseline ranker")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_baselines")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--candidate-pools", type=str, default="U1_liquid_tradable,U2_risk_sane")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--min-train-rows", type=int, default=500)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--model-n-jobs",
        type=int,
        default=0,
        help="模型训练线程数；0 表示使用全部 CPU 核心，1 保持旧的单线程行为。",
    )
    p.add_argument("--skip-xgboost", action="store_true", help="跳过 XGBoost baseline，便于快速烟雾测试")
    p.add_argument(
        "--rebalance-rule", type=str, default="M",
        choices=["W", "M", "BM", "Q", "W-FRI"],
        help="换仓频率：W=周 M=月 BM=双月 Q=季 (默认: M)",
    )
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _resolve_loaded_config_path(config_arg: Path | None) -> Path | None:
    if config_arg is not None:
        return resolve_config_path(config_arg)
    candidates: list[Path] = []
    env_path = os.environ.get("QUANT_CONFIG", "").strip()
    if env_path:
        candidates.extend(config_path_candidates(env_path))
    candidates.extend([ROOT / "config.yaml", ROOT / "config.yaml.example"])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted(set(vals))


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def build_doc(
    *,
    quality: dict[str, Any],
    leaderboard: pd.DataFrame,
    year_slice: pd.DataFrame,
    regime_slice: pd.DataFrame,
    industry_exposure: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    leader_view = leaderboard.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, True, False, False],
    )
    year_view = year_slice.sort_values(["candidate_pool_version", "model", "top_k", "year"]).head(40)
    regime_view = regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(40)
    industry_view = pd.DataFrame()
    if not industry_exposure.empty:
        industry_view = (
            industry_exposure.groupby(["candidate_pool_version", "model", "top_k", "industry_level1"], sort=True)
            .agg(mean_share=("industry_share", "mean"), months=("signal_date", "nunique"))
            .reset_index()
            .sort_values(["top_k", "candidate_pool_version", "model", "mean_share"], ascending=[True, True, True, False])
            .head(40)
        )
    best_u1_top20 = pd.DataFrame()
    best_u2_top20 = pd.DataFrame()
    if not leaderboard.empty:
        best_u1_top20 = leaderboard[
            (leaderboard["candidate_pool_version"] == "U1_liquid_tradable") & (leaderboard["top_k"] == 20)
        ].head(3)
        best_u2_top20 = leaderboard[
            (leaderboard["candidate_pool_version"] == "U2_risk_sane") & (leaderboard["top_k"] == 20)
        ].head(3)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection Baselines

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_baselines`
- 研究主题：`{quality.get('research_topic', '')}`
- 研究配置：`{quality.get('research_config_id', '')}`
- 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}`
- 训练/评估：静态 baseline 全样本打分；ML baseline 使用 walk-forward，只用测试月之前的数据训练。
- 有效标签月份：`{quality.get('valid_signal_months', 0)}`

## Leaderboard

{format_markdown_table(leader_view, max_rows=40)}

## Year Slice

{format_markdown_table(year_view, max_rows=40)}

## Realized Market State Slice

{format_markdown_table(regime_view, max_rows=40)}

## Industry Exposure

{format_markdown_table(industry_view, max_rows=40)}

## 口径

- 输入固定为 `data/cache/monthly_selection_features.parquet` 兼容的 M2 canonical dataset。
- 主训练池/主报告池为 `U1_liquid_tradable` 与 `U2_risk_sane`。
- 第一轮只使用 price-volume-only 特征：收益动量、低波、流动性、换手、价格位置和涨跌停路径特征。
- Top-K 并行报告 `20 / 30 / 50`；`B0_market_ew` 作为市场等权基准，非真实持仓模型。
- `realized_market_state` 使用同一持有期市场等权收益的全样本 20%/80% 分位切片，仅用于归因，不作为可交易信号。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 `cost_bps` 的简化成本敏感性。
- baseline overlap 不作为 M4 gate；M4 的核心是 Rank IC、Top-K 超额、Top-K vs next-K、分桶 spread、年度/状态稳定性、行业暴露和换手。

## 本轮结论

- 本轮新增：M4 price-volume-only baseline ranker，覆盖单因子、线性 blend、ElasticNet、Logistic top-bucket classifier、ExtraTrees sanity check、XGBoost regression/classifier。
- 数据质量：沿用 M2 canonical dataset 的 PIT 与候选池口径；本脚本只消费已落地特征，不引入新数据家族。
- `U1_liquid_tradable` Top20 当前领先模型：

{format_markdown_table(best_u1_top20, max_rows=3)}

- `U2_risk_sane` Top20 当前领先模型：

{format_markdown_table(best_u2_top20, max_rows=3)}

- 静态单因子/线性 blend 多数无法稳定跑赢市场，应保留为低门槛对照，不作为推荐候选。
- walk-forward ML baseline 有弱正向起点，但 strong-up / strong-down 切片仍不稳；M4 不进入生产。
- 下一步进入 M5，逐个验证 industry breadth、fund flow、fundamental、shareholder 等增量是否能稳定改善 Rank IC、Top-K 超额、分桶 spread 和强市参与度。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = _resolve_loaded_config_path(args.config)
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = project_relative(loaded_config_path) if loaded_config_path is not None else "default_config_lookup"
    dataset_path = _resolve_project_path(args.dataset)
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    experiments_dir = _resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    docs_dir = ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    top_ks = _parse_int_list(args.top_k)
    pools = _parse_str_list(args.candidate_pools)
    run_cfg = BaselineRunConfig(
        top_ks=tuple(top_ks),
        candidate_pools=tuple(pools),
        bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months),
        min_train_rows=int(args.min_train_rows),
        cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed),
        include_xgboost=not bool(args.skip_xgboost),
        model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_topic = "monthly_selection_baselines"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_buckets_{int(args.bucket_count)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m"
        f"_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_baselines",
        research_topic=research_topic,
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    research_topic = identity.research_topic
    research_config_id = identity.research_config_id
    output_stem = identity.output_stem

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    valid = valid_pool_frame(dataset)
    static_scores = build_static_scores(dataset)
    wf_scores, ml_importance = build_walk_forward_scores(dataset, run_cfg)
    scores = pd.concat([x for x in [static_scores, wf_scores] if not x.empty], ignore_index=True)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=run_cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=run_cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_feature_importance(static_scores, ml_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)

    quality = {
        "result_type": "monthly_selection_baselines",
        "research_topic": research_topic,
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "config_source": config_source,
        "dataset_path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": pools,
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
        "top_ks": top_ks,
        "bucket_count": int(args.bucket_count),
        "cost_assumption": f"{float(args.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_spec": "price_volume_only_v1",
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
        "pit_policy": "features are consumed from M2 PIT-safe monthly signal rows; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month",
        "hyperparameter_policy": "fixed conservative defaults; no random CV",
        "model_n_jobs": int(normalize_model_n_jobs(run_cfg.model_n_jobs)),
        "random_seed": int(args.random_seed),
        "rows": int(len(dataset)),
        "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": docs_dir / f"{output_stem}.md",
    }

    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = build_summary_payload(quality=quality, leaderboard=leaderboard)
    paths_out["summary_json"].write_text(
        json.dumps(json_sanitize(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = [
        project_relative(p)
        for key, p in paths_out.items()
        if key not in {"manifest", "doc"}
    ]
    paths_out["doc"].write_text(
        build_doc(
            quality=quality,
            leaderboard=leaderboard,
            year_slice=year_slice,
            regime_slice=regime_slice,
            industry_exposure=industry_exposure,
            artifacts=[*artifact_paths, project_relative(paths_out["manifest"])],
        ),
        encoding="utf-8",
    )

    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row: dict[str, Any] = {}
    if not leaderboard.empty:
        best_row = (
            leaderboard.sort_values(
                ["topk_excess_after_cost_mean", "rank_ic_mean"],
                ascending=[False, False],
            )
            .iloc[0]
            .to_dict()
        )
    rank_ic_observations = int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    best_after_cost_float = float(best_after_cost) if pd.notna(best_after_cost) else None
    data_slice = DataSlice(
        dataset_name="monthly_selection_baselines",
        source_tables=(project_relative(dataset_path),),
        date_start=min_signal_date,
        date_end=max_signal_date,
        asof_trade_date=max_signal_date or None,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(pools),
        rebalance_rule=args.rebalance_rule,
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="price_volume_only_v1",
        feature_columns=tuple(ML_FEATURE_COLS),
        label_columns=(LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL),
        pit_policy="features are consumed from M2 PIT-safe monthly signal rows; ML uses past months only",
        config_path=config_source,
        extra={
            "dataset_path": project_relative(dataset_path),
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "top_ks": top_ks,
            "bucket_count": int(args.bucket_count),
            "cv_policy": "walk_forward_by_signal_month",
        },
    )
    artifact_refs = (
        ArtifactRef("summary_json", project_relative(paths_out["summary_json"]), "json"),
        ArtifactRef("leaderboard_csv", project_relative(paths_out["leaderboard"]), "csv"),
        ArtifactRef("monthly_long_csv", project_relative(paths_out["monthly_long"]), "csv"),
        ArtifactRef("rank_ic_csv", project_relative(paths_out["rank_ic"]), "csv"),
        ArtifactRef("quantile_spread_csv", project_relative(paths_out["quantile_spread"]), "csv"),
        ArtifactRef("topk_holdings_csv", project_relative(paths_out["topk_holdings"]), "csv"),
        ArtifactRef("industry_exposure_csv", project_relative(paths_out["industry_exposure"]), "csv"),
        ArtifactRef("candidate_pool_width_csv", project_relative(paths_out["candidate_pool_width"]), "csv"),
        ArtifactRef(
            "candidate_pool_reject_reason_csv",
            project_relative(paths_out["candidate_pool_reject_reason"]),
            "csv",
        ),
        ArtifactRef("feature_importance_csv", project_relative(paths_out["feature_importance"]), "csv"),
        ArtifactRef("year_slice_csv", project_relative(paths_out["year_slice"]), "csv"),
        ArtifactRef("regime_slice_csv", project_relative(paths_out["regime_slice"]), "csv"),
        ArtifactRef("market_states_csv", project_relative(paths_out["market_states"]), "csv"),
        ArtifactRef("report_md", project_relative(paths_out["doc"]), "md"),
        ArtifactRef("manifest_json", project_relative(paths_out["manifest"]), "json"),
    )
    metrics = {
        "rows": int(quality["rows"]),
        "valid_rows": int(quality["valid_rows"]),
        "valid_signal_months": int(quality["valid_signal_months"]),
        "score_rows": int(len(scores)),
        "rank_ic_observations": rank_ic_observations,
        "monthly_long_rows": int(len(monthly_long)),
        "topk_holdings_rows": int(len(topk_holdings)),
        "model_count": int(len(quality["models"])),
        "best_model": str(best_row.get("model") or ""),
        "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
        "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
        "best_topk_excess_after_cost_mean": best_after_cost_float,
        "best_rank_ic_mean": float(best_row["rank_ic_mean"])
        if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean"))
        else None,
    }
    gates = {
        "data_gate": {
            "passed": bool(metrics["valid_rows"] > 0 and metrics["valid_signal_months"] > 0),
            "checks": {
                "has_valid_rows": metrics["valid_rows"] > 0,
                "has_valid_signal_months": metrics["valid_signal_months"] > 0,
            },
        },
        "rank_gate": {
            "passed": bool(rank_ic_observations > 0),
            "rank_ic_observations": rank_ic_observations,
        },
        "spread_gate": {
            "passed": bool(not monthly_long.empty and not quantile_spread.empty),
            "monthly_rows": int(len(monthly_long)),
            "quantile_spread_rows": int(len(quantile_spread)),
        },
        "year_gate": {
            "passed": bool(not year_slice.empty),
            "year_slice_rows": int(len(year_slice)),
        },
        "regime_gate": {
            "passed": bool(not regime_slice.empty),
            "regime_slice_rows": int(len(regime_slice)),
        },
        "baseline_gate": {
            "passed": bool(best_after_cost_float is not None and best_after_cost_float > 0.0),
            "best_topk_excess_after_cost_mean": best_after_cost_float,
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    config_info = config_snapshot(
        config_path=loaded_config_path,
        resolved_config=cfg_raw,
        sections=(
            "paths",
            "database",
            "signals",
            "portfolio",
            "backtest",
            "transaction_costs",
            "prefilter",
            "monthly_selection",
        ),
    )
    config_info["config_path"] = config_source
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=int(args.random_seed),
        data_slices=(data_slice,),
        config=config_info,
        params={
            "cli": vars(args),
            "run_config": {
                "top_ks": list(run_cfg.top_ks),
                "candidate_pools": list(run_cfg.candidate_pools),
                "bucket_count": run_cfg.bucket_count,
                "min_train_months": run_cfg.min_train_months,
                "min_train_rows": run_cfg.min_train_rows,
                "cost_bps": run_cfg.cost_bps,
                "include_xgboost": run_cfg.include_xgboost,
                "model_n_jobs": normalize_model_n_jobs(run_cfg.model_n_jobs),
            },
            "overrides": {
                key: value
                for key, value in {
                    "dataset": args.dataset,
                    "results_dir": args.results_dir.strip(),
                    "top_k": args.top_k,
                    "candidate_pools": args.candidate_pools,
                    "skip_xgboost": args.skip_xgboost,
                }.items()
                if value
            },
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["m4_baseline_research_only_not_promotion_candidate"],
        },
        notes="Monthly selection M4 baseline contract; ranking outputs are unchanged.",
    )
    write_research_manifest(
        paths_out["manifest"],
        result,
        extra={
            "generated_at_utc": result.created_at,
            "result_type": "monthly_selection_baselines_manifest",
            "research_topic": research_topic,
            "research_config_id": research_config_id,
            "output_stem": output_stem,
            "config_source": config_source,
            "dataset_path": project_relative(dataset_path),
            "dataset_version": "monthly_selection_features_v1",
            "candidate_pools": pools,
            "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
            "top_ks": top_ks,
            "feature_spec": "price_volume_only_v1",
            "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
            "pit_policy": data_slice.pit_policy,
            "legacy_artifacts": [*artifact_paths, project_relative(paths_out["doc"])],
        },
    )
    append_experiment_result(experiments_dir, result)

    print(f"[monthly-baselines] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-baselines] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-baselines] manifest={paths_out['manifest']}")
    print(f"[monthly-baselines] research_index={experiments_dir / 'research_results.jsonl'}")
    print(f"[monthly-baselines] doc={paths_out['doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
