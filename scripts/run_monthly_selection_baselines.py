#!/usr/bin/env python3
"""M4: 月度选股 baseline ranker。消费 M2 canonical dataset，输出 price-volume-only
可解释 baseline、walk-forward ML baseline，Rank IC、Top-K 超额、分桶 spread、
年度/市场状态、行业暴露和换手诊断。"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    BaselineRunConfig,
    build_baselines_doc,
    build_leaderboard,
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
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import json_sanitize
from src.research.gates import EXCESS_COL, INDUSTRY_EXCESS_COL, LABEL_COL, MARKET_COL, POOL_RULES, TOP20_COL
from src.settings import load_config


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
    p.add_argument("--model-n-jobs", type=int, default=0)
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--rebalance-rule", type=str, default="M", choices=["W", "M", "BM", "Q", "W-FRI"])
    return p.parse_args()


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    cfg_raw = load_config(args.config)
    paths = cfg_raw.get("paths", {}) or {}
    loaded_config_path = resolve_loaded_config_path(args.config)
    config_source = project_relative(loaded_config_path) if loaded_config_path else "default_config_lookup"
    dataset_path = resolve_project_path(args.dataset)
    results_dir = resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    for d in [results_dir, experiments_dir, ROOT / "docs"]:
        d.mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = parse_str_list(args.candidate_pools)
    run_cfg = BaselineRunConfig(
        top_ks=tuple(top_ks), candidate_pools=tuple(pools), bucket_count=int(args.bucket_count),
        min_train_months=int(args.min_train_months), min_train_rows=int(args.min_train_rows),
        cost_bps=float(args.cost_bps), random_seed=int(args.random_seed),
        include_xgboost=not bool(args.skip_xgboost), model_n_jobs=int(args.model_n_jobs),
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_topic = "monthly_selection_baselines"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}_buckets_{int(args.bucket_count)}"
        f"_jobs_{slugify_token(model_n_jobs_token(args.model_n_jobs))}"
        f"_wf_{int(args.min_train_months)}m_costbps_{slugify_token(args.cost_bps)}"
    )
    identity = make_research_identity(
        result_type="monthly_selection_baselines", research_topic=research_topic,
        research_config_id=research_config_id, output_stem=output_stem,
    )

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
        "result_type": "monthly_selection_baselines", "research_topic": research_topic,
        "research_config_id": research_config_id, "output_stem": output_stem,
        "config_source": config_source, "dataset_path": project_relative(dataset_path),
        "dataset_version": "monthly_selection_features_v1", "candidate_pools": pools,
        "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
        "top_ks": top_ks, "bucket_count": int(args.bucket_count),
        "cost_assumption": f"{float(args.cost_bps):.4g} bps per unit half-L1 turnover",
        "feature_spec": "price_volume_only_v1",
        "label_spec": "forward_1m_open_to_open_return + market-relative excess + top20 bucket",
        "pit_policy": "features are consumed from M2 PIT-safe monthly signal rows; ML uses past months only",
        "cv_policy": "walk_forward_by_signal_month", "hyperparameter_policy": "fixed conservative defaults; no random CV",
        "model_n_jobs": int(normalize_model_n_jobs(run_cfg.model_n_jobs)),
        "random_seed": int(args.random_seed), "rows": int(len(dataset)), "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
        "models": sorted(scores["model"].unique().tolist()) if not scores.empty else [],
    }

    paths_out = {f"{k}": results_dir / f"{output_stem}_{k}.csv" for k in [
        "leaderboard", "monthly_long", "rank_ic", "quantile_spread", "topk_holdings",
        "industry_exposure", "candidate_pool_width", "candidate_pool_reject_reason",
        "feature_importance", "year_slice", "regime_slice", "market_states",
    ]}
    paths_out["summary_json"] = results_dir / f"{output_stem}_summary.json"
    paths_out["manifest"] = results_dir / f"{output_stem}_manifest.json"
    paths_out["doc"] = ROOT / "docs" / f"{output_stem}.md"

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
        json.dumps(json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_paths_raw = [project_relative(p) for key, p in paths_out.items() if key not in {"manifest", "doc"}]
    paths_out["doc"].write_text(
        build_baselines_doc(quality=quality, leaderboard=leaderboard, year_slice=year_slice,
                            regime_slice=regime_slice, industry_exposure=industry_exposure,
                            artifacts=[*artifact_paths_raw, project_relative(paths_out["manifest"])]),
        encoding="utf-8")

    min_signal_date = str(quality.get("min_valid_signal_date") or "")
    max_signal_date = str(quality.get("max_valid_signal_date") or "")
    best_row = leaderboard.sort_values(["topk_excess_after_cost_mean", "rank_ic_mean"],
                                       ascending=[False, False]).iloc[0].to_dict() if not leaderboard.empty else {}
    best_after_cost = best_row.get("topk_excess_after_cost_mean")
    rank_ic_obs = int(pd.to_numeric(rank_ic.get("rank_ic"), errors="coerce").notna().sum()) if not rank_ic.empty else 0
    finalize_research_contract(
        identity=identity, script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at, config_source=config_source, config_raw=cfg_raw,
        loaded_config_path=loaded_config_path, experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_baselines",
            source_tables=(project_relative(dataset_path),),
            date_start=min_signal_date, date_end=max_signal_date,
            asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule=args.rebalance_rule, execution_mode="tplus1_open",
            label_return_mode="open_to_open", feature_set_id="price_volume_only_v1",
            feature_columns=tuple(ML_FEATURE_COLS),
            label_columns=(LABEL_COL, EXCESS_COL, INDUSTRY_EXCESS_COL, MARKET_COL, TOP20_COL),
            pit_policy="features are consumed from M2 PIT-safe monthly signal rows; ML uses past months only",
            config_path=config_source,
            extra={"dataset_path": project_relative(dataset_path),
                   "candidate_pool_rules": {p: POOL_RULES.get(p, "") for p in pools},
                   "top_ks": top_ks, "bucket_count": int(args.bucket_count),
                   "cv_policy": "walk_forward_by_signal_month"},
        ),
        metrics={
            "rows": int(quality["rows"]), "valid_rows": int(quality["valid_rows"]),
            "valid_signal_months": int(quality["valid_signal_months"]),
            "score_rows": int(len(scores)), "rank_ic_observations": rank_ic_obs,
            "monthly_long_rows": int(len(monthly_long)), "topk_holdings_rows": int(len(topk_holdings)),
            "model_count": int(len(quality["models"])),
            "best_model": str(best_row.get("model") or ""),
            "best_candidate_pool_version": str(best_row.get("candidate_pool_version") or ""),
            "best_top_k": int(best_row["top_k"]) if best_row.get("top_k") is not None and pd.notna(best_row.get("top_k")) else None,
            "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None,
            "best_rank_ic_mean": float(best_row["rank_ic_mean"]) if best_row.get("rank_ic_mean") is not None and pd.notna(best_row.get("rank_ic_mean")) else None,
        },
        gates={
            "data_gate": {"passed": len(valid) > 0 and (valid["signal_date"].nunique() > 0 if not valid.empty else False),
                          "checks": {"has_valid_rows": len(valid) > 0, "has_valid_signal_months": valid["signal_date"].nunique() > 0 if not valid.empty else False}},
            "rank_gate": {"passed": rank_ic_obs > 0, "rank_ic_observations": rank_ic_obs},
            "spread_gate": {"passed": not monthly_long.empty and not quantile_spread.empty,
                            "monthly_rows": len(monthly_long), "quantile_spread_rows": len(quantile_spread)},
            "year_gate": {"passed": not year_slice.empty, "year_slice_rows": len(year_slice)},
            "regime_gate": {"passed": not regime_slice.empty, "regime_slice_rows": len(regime_slice)},
            "baseline_gate": {"passed": best_after_cost is not None and pd.notna(best_after_cost) and float(best_after_cost) > 0.0,
                              "best_topk_excess_after_cost_mean": float(best_after_cost) if pd.notna(best_after_cost) else None},
            "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"},
        },
        seed=int(args.random_seed), promotion_blocking=["m4_baseline_research_only_not_promotion_candidate"],
        notes="Monthly selection M4 baseline contract; ranking outputs are unchanged.",
        artifact_paths_raw=artifact_paths_raw,
    )

    print(f"[monthly-baselines] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-baselines] leaderboard={paths_out['leaderboard']}")
    print(f"[monthly-baselines] manifest={paths_out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
