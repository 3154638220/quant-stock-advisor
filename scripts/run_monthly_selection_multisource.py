#!/usr/bin/env python3
"""M5: 月度选股多源特征扩展。消费 M2 canonical dataset，逐个特征族做增量评估。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_identity import make_research_identity, slugify_token
from src.pipeline.cli_helpers import (
    parse_int_list,
    parse_str_list,
    project_relative,
    resolve_loaded_config_path,
    resolve_project_path,
)
from src.pipeline.monthly_baselines import (
    ML_FEATURE_COLS,
    build_leaderboard,
    build_monthly_long,
    build_quantile_spread,
    build_rank_ic,
    build_realized_market_states,
    load_baseline_dataset,
    model_n_jobs_token,
    normalize_model_n_jobs,
    summarize_candidate_pool_reject_reason,
    summarize_candidate_pool_width,
    summarize_industry_exposure,
    summarize_regime_slice,
    summarize_year_slice,
    valid_pool_frame,
)
from src.pipeline.monthly_multisource import (
    M5RunConfig,
    attach_enabled_families,
    build_all_m5_scores,
    build_feature_specs,
    build_incremental_delta,
    summarize_feature_coverage_by_spec,
    summarize_feature_importance,
)
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import format_markdown_table, json_sanitize
from src.research.gates import EXCESS_COL, LABEL_COL, MARKET_COL, POOL_RULES, TOP20_COL
from src.settings import load_config

PRICE_VOLUME_FEATURES: tuple[str, ...] = ML_FEATURE_COLS
_json_sanitize = json_sanitize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 M5 多源特征扩展")
    for flag, default, h in [
        ("--config", None, None), ("--dataset", "data/cache/monthly_selection_features.parquet", None),
        ("--duckdb-path", "", None), ("--output-prefix", "monthly_selection_m5_multisource", None),
        ("--results-dir", "", None), ("--top-k", "20,30,50", None), ("--bucket-count", 5, int),
        ("--candidate-pools", "U1_liquid_tradable,U2_risk_sane", None),
        ("--min-train-months", 24, int), ("--min-train-rows", 500, int),
        ("--max-fit-rows", 0, int), ("--cost-bps", 10.0, float), ("--random-seed", 42, int),
        ("--model-n-jobs", 0, int), ("--availability-lag-days", 30, int),
        ("--families", "industry_breadth,fund_flow,fundamental,shareholder", None),
        ("--ml-models", "elasticnet,logistic,extratrees,xgboost_excess,xgboost_top20", None),
        ("--rebalance-rule", "", None),
    ]:
        if h is int:
            p.add_argument(flag, type=int, default=default)
        elif h is float:
            p.add_argument(flag, type=float, default=default)
        else:
            p.add_argument(flag, type=str, default=default)
    p.add_argument("--skip-xgboost", action="store_true")
    return p.parse_args()


def _quality(dataset, scores, specs, cfg, dataset_path, db_path, output_stem, config_source, research_config_id):
    valid = valid_pool_frame(dataset)
    return {
        "result_type": "monthly_selection_m5_multisource", "research_topic": "monthly_selection_m5_multisource",
        "research_config_id": research_config_id, "output_stem": output_stem, "config_source": config_source,
        "dataset_path": project_relative(dataset_path), "duckdb_path": project_relative(db_path),
        "dataset_version": "monthly_selection_features_v1",
        "candidate_pools": list(cfg.candidate_pools), "top_ks": list(cfg.top_ks),
        "bucket_count": int(cfg.bucket_count),
        "cost_assumption": f"{float(cfg.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_specs": [{"name": s.name, "families": list(s.families), "feature_count": len(s.feature_cols)} for s in specs],
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
        "pit_policy": "monthly rows use signal-date-or-earlier features; fundamental uses announcement_date <= signal_date; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month", "hyperparameter_policy": "fixed conservative defaults; no random CV",
        "ml_models": list(cfg.ml_models), "max_fit_rows": int(cfg.max_fit_rows),
        "model_n_jobs": int(normalize_model_n_jobs(cfg.model_n_jobs)),
        "random_seed": int(cfg.random_seed), "rows": int(len(dataset)), "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }


def _doc(quality, leaderboard, incremental_delta, feature_coverage, year_slice, regime_slice, artifacts):
    generated_at = pd.Timestamp.utcnow().isoformat()
    leader_view = leaderboard.sort_values(
        ["top_k", "candidate_pool_version", "topk_excess_after_cost_mean", "rank_ic_mean"],
        ascending=[True, True, False, False],
    )
    delta_view = incremental_delta.head(40)
    cov_view = feature_coverage.copy()
    if not cov_view.empty:
        cov_view = cov_view[~cov_view["feature_spec"].eq("price_volume_only") | cov_view["feature"].isin(list(PRICE_VOLUME_FEATURES)[:3])].head(80)
    year_view = year_slice.sort_values(["candidate_pool_version", "model", "top_k", "year"]).head(40)
    regime_view = regime_slice.sort_values(["top_k", "candidate_pool_version", "model", "realized_market_state"]).head(40)
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection M5 Multisource

- 生成时间：`{generated_at}` · 输出 stem：`{quality.get('output_stem', '')}`
- 数据集：`{quality.get('dataset_path', '')}` · 数据库：`{quality.get('duckdb_path', '')}`
- 有效标签月份：`{quality.get('valid_signal_months', 0)}` · 单窗训练行上限：`{quality.get('max_fit_rows', 0)}`

## Leaderboard

{format_markdown_table(leader_view, max_rows=40)}

## Incremental Delta vs Price-Volume

{format_markdown_table(delta_view, max_rows=40)}

## Feature Coverage

{format_markdown_table(cov_view, max_rows=80)}

## Year Slice

{format_markdown_table(year_view, max_rows=40)}

## Realized Market State Slice

{format_markdown_table(regime_view, max_rows=40)}

## 口径

- `price_volume_only` 是 M5 内部 baseline；后续 spec 按 industry_breadth -> fund_flow -> fundamental -> shareholder 累积加入。
- `topk_excess_after_cost` 使用半 L1 月度换手乘以 cost_bps 的简化成本敏感性。

## 本轮产物

{artifact_lines}
"""


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    loaded_config_path = resolve_loaded_config_path(args.config)
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    config_source = project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup"
    dataset_path = resolve_project_path(args.dataset)
    db_path = resolve_project_path(args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb"))
    results_dir = resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    for d in [results_dir, experiments_dir, ROOT / "docs"]:
        d.mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = parse_str_list(args.candidate_pools)
    enabled_families = parse_str_list(args.families)
    cfg = M5RunConfig(
        top_ks=tuple(top_ks), candidate_pools=tuple(pools), bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months), min_train_rows=int(args.min_train_rows),
        max_fit_rows=int(args.max_fit_rows), cost_bps=float(args.cost_bps),
        random_seed=int(args.random_seed), include_xgboost=not bool(args.skip_xgboost),
        availability_lag_days=int(args.availability_lag_days),
        ml_models=tuple(parse_str_list(args.ml_models)), model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_families_{'-'.join(slugify_token(x) for x in ['price_volume_only', *enabled_families])}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_models_{'-'.join(slugify_token(x) for x in parse_str_list(args.ml_models))}"
        f"_maxfit_{int(args.max_fit_rows)}_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_m5_multisource",
        research_topic="monthly_selection_m5_multisource",
        research_config_id=research_config_id, output_stem=output_stem,
    )
    print(f"[monthly-m5] research_config_id={research_config_id}")

    dataset = load_baseline_dataset(dataset_path, candidate_pools=pools)
    dataset = attach_enabled_families(dataset, db_path, cfg, enabled_families)
    specs = build_feature_specs(enabled_families)
    feature_coverage = summarize_feature_coverage_by_spec(dataset, specs)
    scores, raw_importance = build_all_m5_scores(dataset, specs, cfg)
    if scores.empty:
        warnings.warn("M5 未生成任何 score；请检查训练窗、候选池或特征覆盖。", RuntimeWarning)
    rank_ic = build_rank_ic(scores)
    monthly_long, topk_holdings = build_monthly_long(scores, top_ks=top_ks, cost_bps=cfg.cost_bps)
    quantile_spread = build_quantile_spread(scores, bucket_count=cfg.bucket_count)
    market_states = build_realized_market_states(dataset)
    year_slice = summarize_year_slice(monthly_long)
    regime_slice = summarize_regime_slice(monthly_long, market_states)
    industry_exposure = summarize_industry_exposure(topk_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)
    reject_reason = summarize_candidate_pool_reject_reason(dataset)
    feature_importance = summarize_feature_importance(raw_importance)
    leaderboard = build_leaderboard(monthly_long, rank_ic, quantile_spread, regime_slice)
    incremental_delta = build_incremental_delta(leaderboard)
    quality = _quality(dataset, scores, specs, cfg, dataset_path, db_path, output_stem, config_source, research_config_id)

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "leaderboard": results_dir / f"{output_stem}_leaderboard.csv",
        "incremental_delta": results_dir / f"{output_stem}_incremental_delta.csv",
        "monthly_long": results_dir / f"{output_stem}_monthly_long.csv",
        "rank_ic": results_dir / f"{output_stem}_rank_ic.csv",
        "quantile_spread": results_dir / f"{output_stem}_quantile_spread.csv",
        "feature_coverage": results_dir / f"{output_stem}_feature_coverage.csv",
        "feature_importance": results_dir / f"{output_stem}_feature_importance.csv",
        "topk_holdings": results_dir / f"{output_stem}_topk_holdings.csv",
        "industry_exposure": results_dir / f"{output_stem}_industry_exposure.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "candidate_pool_reject_reason": results_dir / f"{output_stem}_candidate_pool_reject_reason.csv",
        "year_slice": results_dir / f"{output_stem}_year_slice.csv",
        "regime_slice": results_dir / f"{output_stem}_regime_slice.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": ROOT / "docs" / f"{output_stem}.md",
    }
    leaderboard.to_csv(paths_out["leaderboard"], index=False)
    incremental_delta.to_csv(paths_out["incremental_delta"], index=False)
    monthly_long.to_csv(paths_out["monthly_long"], index=False)
    rank_ic.to_csv(paths_out["rank_ic"], index=False)
    quantile_spread.to_csv(paths_out["quantile_spread"], index=False)
    feature_coverage.to_csv(paths_out["feature_coverage"], index=False)
    feature_importance.to_csv(paths_out["feature_importance"], index=False)
    topk_holdings.to_csv(paths_out["topk_holdings"], index=False)
    industry_exposure.to_csv(paths_out["industry_exposure"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    reject_reason.to_csv(paths_out["candidate_pool_reject_reason"], index=False)
    year_slice.to_csv(paths_out["year_slice"], index=False)
    regime_slice.to_csv(paths_out["regime_slice"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = {
        "quality": quality,
        "top_models_by_topk": leaderboard.sort_values(["top_k", "topk_excess_after_cost_mean", "rank_ic_mean"], ascending=[True, False, False]).groupby("top_k", as_index=False).head(5).to_dict(orient="records") if not leaderboard.empty else [],
        "best_incremental_by_topk": incremental_delta.groupby("top_k", as_index=False).head(5).to_dict(orient="records") if not incremental_delta.empty else [],
    }
    paths_out["summary_json"].write_text(json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_paths_raw = [project_relative(p) for key, p in paths_out.items() if key not in {"manifest", "doc"}]
    paths_out["doc"].write_text(
        _doc(quality, leaderboard, incremental_delta, feature_coverage, year_slice, regime_slice,
             [*artifact_paths_raw, project_relative(paths_out["manifest"])]),
        encoding="utf-8")

    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row = leaderboard.sort_values(["topk_excess_after_cost_mean", "rank_ic_mean"], ascending=[False, False]).iloc[0].to_dict() if not leaderboard.empty else {}
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    rank_ic_obs = int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    rebalance_rule = args.rebalance_rule or ("M" if dataset.empty or "rebalance_rule" not in dataset.columns else str(dataset["rebalance_rule"].iloc[0]).strip().upper() or "M")
    feature_columns = tuple(dict.fromkeys(col for spec in specs for col in spec.feature_cols))
    finalize_research_contract(
        identity=identity, script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at, config_source=config_source, config_raw=cfg_raw,
        loaded_config_path=loaded_config_path, experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_m5_multisource",
            source_tables=(project_relative(dataset_path), project_relative(db_path)),
            date_start=min_signal_date, date_end=max_signal_date, asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule=rebalance_rule, execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id="m5_" + "_".join(["price_volume_only", *enabled_families]),
            feature_columns=feature_columns,
            label_columns=(LABEL_COL, EXCESS_COL, MARKET_COL, TOP20_COL),
            pit_policy=quality["pit_policy"], config_path=config_source,
            extra={"dataset_path": project_relative(dataset_path), "duckdb_path": project_relative(db_path),
                   "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                   "enabled_families": enabled_families, "feature_specs": quality["feature_specs"],
                   "top_ks": top_ks, "bucket_count": int(args.bucket_count),
                   "availability_lag_days": int(args.availability_lag_days), "cv_policy": quality["cv_policy"]},
        ),
        metrics={
            "rows": int(quality["rows"]), "valid_rows": int(quality["valid_rows"]),
            "valid_signal_months": int(quality["valid_signal_months"]),
            "score_rows": int(len(scores)), "rank_ic_observations": rank_ic_obs,
            "monthly_long_rows": int(len(monthly_long)), "topk_holdings_rows": int(len(topk_holdings)),
            "feature_spec_count": int(len(specs)), "feature_coverage_rows": int(len(feature_coverage)),
            "incremental_delta_rows": int(len(incremental_delta)),
            "model_count": int(len(quality["models"])),
            "best_model": str(best_row.get("model") or ""),
            "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
            "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None,
            "best_rank_ic_mean": float(best_row["rank_ic_mean"]) if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean")) else None,
        },
        gates={
            "data_gate": {"passed": quality["valid_rows"] > 0 and quality["valid_signal_months"] > 0,
                          "checks": {"has_valid_rows": quality["valid_rows"] > 0, "has_valid_signal_months": quality["valid_signal_months"] > 0, "has_feature_coverage": len(feature_coverage) > 0}},
            "rank_gate": {"passed": rank_ic_obs > 0, "rank_ic_observations": rank_ic_obs},
            "spread_gate": {"passed": not monthly_long.empty and not quantile_spread.empty, "monthly_rows": len(monthly_long), "quantile_spread_rows": len(quantile_spread)},
            "baseline_gate": {"passed": not incremental_delta.empty, "incremental_delta_rows": len(incremental_delta), "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None},
            "year_gate": {"passed": not year_slice.empty, "year_slice_rows": len(year_slice)},
            "regime_gate": {"passed": not regime_slice.empty, "regime_slice_rows": len(regime_slice)},
            "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
        },
        seed=int(args.random_seed), promotion_blocking=["m5_multisource_research_only_not_promotion_candidate"],
        notes="Monthly selection M5 multisource contract; model outputs are unchanged.",
        artifact_paths_raw=artifact_paths_raw,
    )

    print(f"[monthly-m5] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-m5] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-m5] manifest={paths_out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
