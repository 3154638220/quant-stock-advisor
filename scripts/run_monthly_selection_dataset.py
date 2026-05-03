#!/usr/bin/env python3
"""构建月度选股 canonical dataset。

产物面向 docs/plan.md 的 M2：月末信号日截面、T+1 open 可买性、
候选池版本、open-to-open 月度标签，以及覆盖/宽度/过滤摘要。

核心逻辑已迁移至 src/pipeline/monthly_dataset.py；
本脚本仅保留 CLI 参数解析、文件 I/O 编排与 manifest 记录。
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
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
from src.pipeline.monthly_dataset import (
    FEATURE_COLS,
    LABEL_COLS,
    MonthlySelectionConfig,
    attach_buyability,
    attach_feature_transforms,
    attach_signal_features,
    build_candidate_pool_panel,
    build_monthly_labels,
    build_monthly_selection_dataset,
    build_quality_summary,
    build_research_config_id,
    load_industry_map,
    normalize_daily_frame,
    read_daily_from_duckdb,
    select_month_end_signal_dates,
    summarize_candidate_width,
    summarize_feature_coverage,
    summarize_label_distribution,
    summarize_reject_reasons,
)
from src.reporting.markdown_report import (
    format_markdown_table,
    project_relative,
)
from src.research.gates import POOL_RULES
from src.settings import config_path_candidates, load_config, resolve_config_path



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建月度选股 canonical dataset")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--duckdb-path", type=str, default="")
    p.add_argument("--daily-table", type=str, default="a_share_daily")
    p.add_argument("--industry-map", type=str, default="data/cache/industry_map.csv")
    p.add_argument("--start-date", type=str, default="2021-01-01")
    p.add_argument("--end-date", type=str, default="")
    p.add_argument("--min-history-days", type=int, default=120)
    p.add_argument("--min-amount-20d", type=float, default=50_000_000.0)
    p.add_argument("--limit-move-max", type=int, default=3)
    p.add_argument("--output-prefix", type=str, default="monthly_selection_dataset")
    p.add_argument("--cache-out", type=str, default="data/cache/monthly_selection_features.parquet")
    p.add_argument("--results-dir", type=str, default="")
    p.add_argument("--dry-run", action="store_true", help="只打印身份与输入范围，不写产物")
    return p.parse_args()


def _resolve_project_path(raw: str | Path, *, base: Path = ROOT) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else base / p


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


def _build_doc(
    *,
    quality: pd.DataFrame,
    width: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    label_distribution: pd.DataFrame,
    reject_reasons: pd.DataFrame,
    artifacts: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    width_summary = pd.DataFrame()
    if not width.empty:
        width_summary = (
            width.groupby("candidate_pool_version")
            .agg(
                months=("signal_date", "nunique"),
                median_width=("candidate_pool_width", "median"),
                min_width=("candidate_pool_width", "min"),
                max_width=("candidate_pool_width", "max"),
                median_pass_ratio=("candidate_pool_pass_ratio", "median"),
            )
            .reset_index()
        )
    label_tail = label_distribution.tail(12).copy() if not label_distribution.empty else label_distribution
    reject_summary = pd.DataFrame()
    if not reject_reasons.empty:
        reject_summary = (
            reject_reasons.groupby(["candidate_pool_version", "reject_reason"], sort=True)["count"]
            .sum()
            .reset_index()
            .sort_values(["candidate_pool_version", "count"], ascending=[True, False])
        )
    artifact_lines = "\n".join(f"- `{x}`" for x in artifacts)
    return f"""# Monthly Selection Dataset

- 生成时间：`{generated_at}`
- 结果类型：`monthly_selection_dataset`
- 研究主题：`{quality.iloc[0].get('research_topic', '') if not quality.empty else ''}`
- 研究配置：`{quality.iloc[0].get('research_config_id', '') if not quality.empty else ''}`
- 输出 stem：`{quality.iloc[0].get('output_stem', '') if not quality.empty else ''}`
- 配置来源：`{quality.iloc[0].get('config_source', '') if not quality.empty else ''}`

## Quality

{format_markdown_table(quality)}

## Candidate Pool Width

{format_markdown_table(width_summary)}

## Feature Coverage

{format_markdown_table(feature_coverage)}

## Label Distribution

{format_markdown_table(label_tail)}

## Reject Reasons

{format_markdown_table(reject_summary)}

## 口径

- 信号日：每月最后一个交易日。
- 执行口径：`tplus1_open`。
- 标签：从信号日后第一个 open-to-open 日收益开始，复利持有到下一次月末信号日开盘；不再包含下一信号日到下一月首个交易日开盘的隔夜区间。
- 候选池：`U0/U1/U2` 只做可交易、数据有效和极端风险过滤，不做 alpha 判断。
- 特征处理：按 `signal_date` 截面 winsorize 1%/99% 后 z-score，并保留缺失标记。

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
    db_path = args.duckdb_path.strip() or str(paths.get("duckdb_path") or "data/market.duckdb")
    db_path = str(_resolve_project_path(db_path))
    results_dir_raw = args.results_dir.strip() or str(paths.get("results_dir") or "data/results")
    results_dir = _resolve_project_path(results_dir_raw)
    experiments_dir = _resolve_project_path(str(paths.get("experiments_dir") or "data/experiments"))
    cache_out = _resolve_project_path(args.cache_out)
    industry_path = _resolve_project_path(args.industry_map)
    end_date = args.end_date.strip() or ""
    research_topic = "monthly_selection_dataset"
    research_config_id = build_research_config_id(
        start_date=args.start_date,
        end_date=end_date,
        min_history_days=args.min_history_days,
        min_amount_20d=args.min_amount_20d,
        limit_move_max=args.limit_move_max,
        daily_table=args.daily_table,
    )
    output_stem = f"{slugify_token(args.output_prefix)}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    identity = make_research_identity(
        result_type="monthly_selection_dataset",
        research_topic=research_topic,
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    research_topic = identity.research_topic
    research_config_id = identity.research_config_id
    output_stem = identity.output_stem

    print(f"[monthly-dataset] research_config_id={research_config_id}")
    print(f"[monthly-dataset] db_path={db_path}")
    print(f"[monthly-dataset] cache_out={cache_out}")
    if args.dry_run:
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    industry_map, industry_status = load_industry_map(industry_path)
    cfg = MonthlySelectionConfig(
        min_history_days=int(args.min_history_days),
        min_amount_20d=float(args.min_amount_20d),
        limit_move_max=int(args.limit_move_max),
    )
    with duckdb.connect(db_path, read_only=True) as con:
        daily = read_daily_from_duckdb(
            con,
            table=args.daily_table,
            start=args.start_date,
            end=end_date or None,
            min_history_days=cfg.min_history_days,
            price_position_lookback=cfg.price_position_lookback,
        )
    dataset = build_monthly_selection_dataset(
        daily,
        start_date=args.start_date,
        end_date=end_date or None,
        industry_map=industry_map,
        cfg=cfg,
    )
    dataset.to_parquet(cache_out, index=False)

    quality = build_quality_summary(
        dataset,
        research_topic=research_topic,
        research_config_id=research_config_id,
        output_stem=output_stem,
        config_source=config_source,
        industry_map_source_status=industry_status,
    )
    width = summarize_candidate_width(dataset)
    reject_reasons = summarize_reject_reasons(dataset)
    feature_coverage = summarize_feature_coverage(dataset)
    label_distribution = summarize_label_distribution(dataset)

    quality_path = results_dir / f"{output_stem}_quality.csv"
    width_path = results_dir / f"{output_stem}_candidate_pool_width.csv"
    reject_path = results_dir / f"{output_stem}_candidate_pool_reject_reason.csv"
    feature_path = results_dir / f"{output_stem}_feature_coverage.csv"
    label_path = results_dir / f"{output_stem}_label_distribution.csv"
    manifest_path = results_dir / f"{output_stem}_manifest.json"
    doc_path = docs_dir / f"{output_stem}.md"

    quality.to_csv(quality_path, index=False)
    width.to_csv(width_path, index=False)
    reject_reasons.to_csv(reject_path, index=False)
    feature_coverage.to_csv(feature_path, index=False)
    label_distribution.to_csv(label_path, index=False)
    artifacts = [
        str(cache_out.relative_to(ROOT)),
        str(quality_path.relative_to(ROOT)),
        str(width_path.relative_to(ROOT)),
        str(reject_path.relative_to(ROOT)),
        str(feature_path.relative_to(ROOT)),
        str(label_path.relative_to(ROOT)),
        str(doc_path.relative_to(ROOT)),
    ]
    doc_path.write_text(
        _build_doc(
            quality=quality,
            width=width,
            feature_coverage=feature_coverage,
            label_distribution=label_distribution,
            reject_reasons=reject_reasons,
            artifacts=[*artifacts, str(manifest_path.relative_to(ROOT))],
        ),
        encoding="utf-8",
    )

    quality_row = quality.iloc[0].to_dict() if not quality.empty else {}
    min_signal_date = str(quality_row.get("min_signal_date") or "")
    max_signal_date = str(quality_row.get("max_signal_date") or "")
    actual_date_end = end_date or (
        str(pd.to_datetime(daily["trade_date"]).max().date()) if not daily.empty else args.start_date
    )
    data_slice = DataSlice(
        dataset_name="monthly_selection_features",
        source_tables=(args.daily_table,),
        date_start=args.start_date,
        date_end=actual_date_end,
        asof_trade_date=max_signal_date or actual_date_end,
        signal_date_col="signal_date",
        symbol_col="symbol",
        candidate_pool_version=",".join(POOL_RULES),
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id="monthly_selection_features_v1",
        feature_columns=tuple(FEATURE_COLS),
        label_columns=tuple(LABEL_COLS),
        pit_policy="features use signal-date close-or-earlier daily bars; labels are output only for train/eval",
        config_path=config_source,
        extra={
            "candidate_pool_rules": POOL_RULES,
            "industry_map_source": project_relative(industry_path) if industry_path.exists() else str(industry_path),
            "industry_map_source_status": industry_status,
            "min_signal_date": min_signal_date,
            "max_signal_date": max_signal_date,
        },
    )
    artifact_refs = (
        ArtifactRef("dataset_parquet", project_relative(cache_out), "parquet"),
        ArtifactRef("quality_csv", project_relative(quality_path), "csv"),
        ArtifactRef("candidate_pool_width_csv", project_relative(width_path), "csv"),
        ArtifactRef("candidate_pool_reject_reason_csv", project_relative(reject_path), "csv"),
        ArtifactRef("feature_coverage_csv", project_relative(feature_path), "csv"),
        ArtifactRef("label_distribution_csv", project_relative(label_path), "csv"),
        ArtifactRef("report_md", project_relative(doc_path), "md"),
        ArtifactRef("manifest_json", project_relative(manifest_path), "json"),
    )
    metrics = {
        "rows": int(quality_row.get("rows") or 0),
        "symbols": int(quality_row.get("symbols") or 0),
        "signal_months": int(quality_row.get("signal_months") or 0),
        "label_valid_rows": int(quality_row.get("label_valid_rows") or 0),
        "min_signal_date": min_signal_date,
        "max_signal_date": max_signal_date,
    }
    gates = {
        "data_gate": {
            "passed": bool(
                metrics["rows"] > 0
                and metrics["symbols"] > 0
                and metrics["signal_months"] > 0
                and metrics["label_valid_rows"] > 0
            ),
            "checks": {
                "has_rows": metrics["rows"] > 0,
                "has_symbols": metrics["symbols"] > 0,
                "has_signal_months": metrics["signal_months"] > 0,
                "has_train_eval_labels": metrics["label_valid_rows"] > 0,
            },
        },
        "execution_gate": {
            "passed": True,
            "execution_mode": "tplus1_open",
            "sell_timing": "holding_month_last_trading_day_open",
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
        script_name=str(Path(__file__).resolve().relative_to(ROOT)),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=round(time.perf_counter() - started_at, 6),
        seed=None,
        data_slices=(data_slice,),
        config=config_info,
        params={
            "cli": vars(args),
            "dataset_config": {
                "min_history_days": cfg.min_history_days,
                "min_amount_20d": cfg.min_amount_20d,
                "limit_move_max": cfg.limit_move_max,
                "limit_move_lookback": cfg.limit_move_lookback,
                "price_position_lookback": cfg.price_position_lookback,
            },
            "overrides": {
                key: value
                for key, value in {
                    "duckdb_path": args.duckdb_path.strip(),
                    "results_dir": args.results_dir.strip(),
                    "end_date": end_date,
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
            "blocking_reasons": ["dataset_contract_only_not_promotion_candidate"],
        },
        notes="Monthly selection dataset contract; business outputs are unchanged.",
    )
    write_research_manifest(
        manifest_path,
        result,
        extra={
            "generated_at_utc": result.created_at,
            "result_type": "monthly_selection_dataset_manifest",
            "research_topic": research_topic,
            "research_config_id": research_config_id,
            "output_stem": output_stem,
            "config_source": config_source,
            "dataset_version": "monthly_selection_features_v1",
            "candidate_pool_versions": list(POOL_RULES),
            "feature_spec": FEATURE_COLS,
            "label_spec": LABEL_COLS,
            "pit_policy": data_slice.pit_policy,
            "industry_map_source": project_relative(industry_path) if industry_path.exists() else str(industry_path),
            "industry_map_source_status": industry_status,
            "legacy_artifacts": artifacts,
        },
    )
    append_experiment_result(experiments_dir, result)

    print(f"[monthly-dataset] rows={len(dataset)} symbols={dataset['symbol'].nunique() if not dataset.empty else 0}")
    print(f"[monthly-dataset] parquet={cache_out}")
    print(f"[monthly-dataset] quality={quality_path}")
    print(f"[monthly-dataset] manifest={manifest_path}")
    print(f"[monthly-dataset] research_index={experiments_dir / 'research_results.jsonl'}")
    print(f"[monthly-dataset] doc={doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
