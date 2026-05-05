#!/usr/bin/env python3
"""M3：月度选股 oracle top-bucket 诊断。消费 M2 canonical dataset，输出候选池
oracle 上限、特征分桶单调性、简单 baseline 与 oracle overlap、regime 切片和行业分布。
"""

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
    resolve_project_path,
)
from src.pipeline.monthly_baselines import (
    build_realized_market_states,
    summarize_candidate_pool_width,
    valid_pool_frame,
)
from src.pipeline.monthly_oracle import (
    FEATURE_SPECS,
    build_oracle_doc,
    build_oracle_topk_tables,
    build_summary_payload,
    load_oracle_dataset,
    summarize_baseline_overlap,
    summarize_feature_bucket_monotonicity,
    summarize_industry_oracle_distribution,
    summarize_oracle_by_candidate_pool,
    summarize_regime_oracle_capacity,
)
from src.pipeline.research_runner import finalize_research_contract
from src.reporting.markdown_report import json_sanitize
from src.research.gates import LABEL_COL, MARKET_COL
from src.settings import load_config, resolve_config_path

_json_sanitize = json_sanitize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行月度选股 oracle top-bucket 诊断")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--output-prefix", type=str, default="monthly_selection_oracle")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--top-k", type=str, default="20,30,50")
    p.add_argument("--bucket-count", type=int, default=5)
    p.add_argument("--candidate-pools", type=str, default="U0_all_tradable,U1_liquid_tradable,U2_risk_sane")
    return p.parse_args()


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
    dataset_path = resolve_project_path(args.dataset)
    results_dir = resolve_project_path(args.results_dir.strip() or str(paths.get("results_dir") or "data/results"))
    experiments_dir = resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    results_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)

    top_ks = parse_int_list(args.top_k)
    pools = parse_str_list(args.candidate_pools)
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    research_topic = "monthly_selection_oracle"
    research_config_id = (
        f"dataset_{slugify_token(dataset_path.stem)}"
        f"_pools_{'-'.join(slugify_token(x) for x in pools)}"
        f"_topk_{'-'.join(str(x) for x in top_ks)}"
        f"_buckets_{int(args.bucket_count)}"
    )

    identity = make_research_identity(
        result_type="monthly_selection_oracle", research_topic=research_topic,
        research_config_id=research_config_id, output_stem=output_stem,
        canonical_config_name="monthly_selection_oracle_v1",
    )
    loaded_config_path = resolve_config_path(args.config) if args.config is not None else None

    dataset = load_oracle_dataset(dataset_path, candidate_pools=pools)
    valid = valid_pool_frame(dataset)
    monthly_oracle, oracle_holdings = build_oracle_topk_tables(dataset, top_ks=top_ks)
    oracle_summary = summarize_oracle_by_candidate_pool(monthly_oracle)
    feature_buckets = summarize_feature_bucket_monotonicity(dataset, bucket_count=int(args.bucket_count))
    baseline_overlap = summarize_baseline_overlap(dataset, top_ks=top_ks)
    market_states = build_realized_market_states(dataset)
    regime_oracle = summarize_regime_oracle_capacity(monthly_oracle, market_states)
    industry_distribution = summarize_industry_oracle_distribution(oracle_holdings)
    candidate_width = summarize_candidate_pool_width(dataset)

    quality = {
        "result_type": "monthly_selection_oracle", "research_topic": research_topic,
        "research_config_id": research_config_id, "output_stem": output_stem,
        "config_source": config_source, "dataset_path": project_relative(dataset_path),
        "candidate_pools": pools, "top_ks": top_ks, "bucket_count": int(args.bucket_count),
        "rows": int(len(dataset)), "valid_rows": int(len(valid)),
        "valid_signal_months": int(valid["signal_date"].nunique()) if not valid.empty else 0,
        "min_valid_signal_date": str(valid["signal_date"].min().date()) if not valid.empty else "",
        "max_valid_signal_date": str(valid["signal_date"].max().date()) if not valid.empty else "",
    }

    paths_out = {
        "summary_json": results_dir / f"{output_stem}_summary.json",
        "oracle_topk_return_by_month": results_dir / f"{output_stem}_oracle_topk_return_by_month.csv",
        "oracle_topk_holdings": results_dir / f"{output_stem}_oracle_topk_holdings.csv",
        "oracle_topk_by_candidate_pool": results_dir / f"{output_stem}_oracle_topk_by_candidate_pool.csv",
        "feature_bucket_monotonicity": results_dir / f"{output_stem}_feature_bucket_monotonicity.csv",
        "baseline_overlap": results_dir / f"{output_stem}_baseline_overlap.csv",
        "regime_oracle_capacity": results_dir / f"{output_stem}_regime_oracle_capacity.csv",
        "industry_oracle_distribution": results_dir / f"{output_stem}_industry_oracle_distribution.csv",
        "candidate_pool_width": results_dir / f"{output_stem}_candidate_pool_width.csv",
        "market_states": results_dir / f"{output_stem}_market_states.csv",
        "manifest": results_dir / f"{output_stem}_manifest.json",
        "doc": ROOT / "docs" / f"{output_stem}.md",
    }

    monthly_oracle.to_csv(paths_out["oracle_topk_return_by_month"], index=False)
    oracle_holdings.to_csv(paths_out["oracle_topk_holdings"], index=False)
    oracle_summary.to_csv(paths_out["oracle_topk_by_candidate_pool"], index=False)
    feature_buckets.to_csv(paths_out["feature_bucket_monotonicity"], index=False)
    baseline_overlap.to_csv(paths_out["baseline_overlap"], index=False)
    regime_oracle.to_csv(paths_out["regime_oracle_capacity"], index=False)
    industry_distribution.to_csv(paths_out["industry_oracle_distribution"], index=False)
    candidate_width.to_csv(paths_out["candidate_pool_width"], index=False)
    market_states.to_csv(paths_out["market_states"], index=False)

    summary_payload = build_summary_payload(
        oracle_summary=oracle_summary, feature_buckets=feature_buckets,
        baseline_overlap=baseline_overlap, quality=quality,
    )
    paths_out["summary_json"].write_text(
        json.dumps(_json_sanitize(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    artifact_paths_raw = [project_relative(p) for key, p in paths_out.items() if key not in {"manifest", "doc"}]
    paths_out["doc"].write_text(
        build_oracle_doc(quality=quality, oracle_summary=oracle_summary, feature_buckets=feature_buckets,
                         baseline_overlap=baseline_overlap, regime_oracle=regime_oracle,
                         industry_distribution=industry_distribution,
                         artifacts=[*artifact_paths_raw, project_relative(paths_out["manifest"])]),
        encoding="utf-8",
    )

    min_signal_date = str(dataset["signal_date"].min().date()) if not dataset.empty else ""
    max_signal_date = str(dataset["signal_date"].max().date()) if not dataset.empty else ""
    finalize_research_contract(
        identity=identity, script_path=project_relative(Path(__file__).resolve()),
        started_at=started_at, config_source=config_source, config_raw=cfg,
        loaded_config_path=loaded_config_path, experiments_dir=experiments_dir,
        paths_out=paths_out, dataset_path=dataset_path,
        data_slice_kwargs=dict(
            dataset_name="monthly_selection_oracle",
            source_tables=(project_relative(dataset_path),),
            date_start=min_signal_date, date_end=max_signal_date,
            asof_trade_date=max_signal_date or None,
            signal_date_col="signal_date", symbol_col="symbol",
            candidate_pool_version=",".join(pools),
            rebalance_rule="M", execution_mode="tplus1_open",
            label_return_mode="open_to_open",
            feature_set_id="oracle_v1",
            feature_columns=tuple(col for _, col, _ in FEATURE_SPECS),
            label_columns=(LABEL_COL, MARKET_COL),
            pit_policy="oracle uses ex-post label_forward_1m_o2o_return for upper-bound diagnosis only",
            config_path=config_source,
            extra={"dataset_path": project_relative(dataset_path), "top_ks": top_ks, "bucket_count": int(args.bucket_count)},
        ),
        metrics={"rows": int(quality["rows"]), "valid_rows": int(quality["valid_rows"]),
                 "valid_signal_months": int(quality["valid_signal_months"]),
                 "oracle_summary_rows": int(len(oracle_summary)),
                 "feature_bucket_rows": int(len(feature_buckets))},
        gates={"data_gate": {"passed": bool(len(valid) > 0 and valid["signal_date"].nunique() > 0),
                             "checks": {"has_valid_rows": len(valid) > 0, "has_valid_signal_months": valid["signal_date"].nunique() > 0}},
               "governance_gate": {"passed": True, "manifest_schema": "research_result_v1"}},
        promotion_blocking=["oracle_diagnostic_only_not_promotion_candidate"],
        notes="Monthly selection M3 oracle top-bucket diagnostic; uses ex-post labels for upper-bound estimation only.",
        artifact_paths_raw=artifact_paths_raw,
    )

    print(f"[monthly-oracle] valid_rows={quality['valid_rows']} valid_months={quality['valid_signal_months']}")
    print(f"[monthly-oracle] oracle_summary={paths_out['oracle_topk_by_candidate_pool']}")
    print(f"[monthly-oracle] doc={paths_out['doc']}")
    print(f"[monthly-oracle] manifest={paths_out['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
