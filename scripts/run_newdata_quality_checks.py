#!/usr/bin/env python3
"""运行资金流 / 股东人数数据质量检查并输出 canonical 研究产物。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.data_quality import (
    run_fund_flow_quality_checks,
    run_shareholder_quality_checks,
)
from scripts.research_identity import slugify_token
from src.settings import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="新数据家族质量检查：fund_flow / shareholder")
    p.add_argument("--config", type=Path, default=None, help="默认项目根 config.yaml")
    p.add_argument("--families", type=str, default="fund_flow,shareholder", help="逗号分隔")
    p.add_argument("--flow-table", type=str, default="a_share_fund_flow")
    p.add_argument("--shareholder-table", type=str, default="a_share_shareholder")
    p.add_argument("--daily-table", type=str, default="a_share_daily")
    p.add_argument("--shareholder-fallback-lag-days", type=int, default=30)
    p.add_argument("--min-effective-width", type=int, default=100)
    p.add_argument("--output-prefix", type=str, default="newdata_quality")
    p.add_argument("--results-dir", type=str, default="")
    return p.parse_args()


def _parse_families(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def build_newdata_quality_research_config_id(
    *,
    families: list[str],
    flow_table: str,
    shareholder_table: str,
    daily_table: str,
    shareholder_fallback_lag_days: int,
    min_effective_width: int,
) -> str:
    family_token = "-".join(slugify_token(item) for item in families) if families else "none"
    return (
        f"families_{family_token}"
        f"_flow_{slugify_token(flow_table)}"
        f"_holder_{slugify_token(shareholder_table)}"
        f"_daily_{slugify_token(daily_table)}"
        f"_lag_{int(shareholder_fallback_lag_days)}"
        f"_width_{int(min_effective_width)}"
    )


def build_newdata_quality_output_stem(
    *,
    output_prefix: str,
    research_config_id: str,
) -> str:
    return f"{slugify_token(output_prefix)}_{research_config_id}"


def _build_doc(
    report_df: pd.DataFrame,
    *,
    research_topic: str,
    research_config_id: str,
    output_stem: str,
    families: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    table = report_df.to_markdown(index=False) if not report_df.empty else "_无质量检查结果_"
    return f"""# Newdata Quality Checks

- 生成时间：`{generated_at}`
- 结果类型：`newdata_quality_summary`
- 研究主题：`{research_topic}`
- 研究配置：`{research_config_id}`
- 输出 stem：`{output_stem}`
- 数据家族：`{", ".join(families) if families else "none"}`

## Summary

{table}

## 说明

- 该产物用于在 scout / tree 实验之前，先确认新数据链路是否满足基本可解释性。
- `fund_flow` 重点检查覆盖率、主键重复、时间戳对齐、关键列空值率与可疑全零行。
- `shareholder` 重点检查 `notice_date` 覆盖率、fallback lag 使用比例、公告滞后异常与截面宽度。

## 本轮产物

- `data/results/{output_stem}_summary.csv`
- `data/results/{output_stem}.json`
- `data/results/{output_stem}_manifest.json`
- `docs/{output_stem}.md`
"""


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    db_path = str((paths.get("duckdb_path") or "data/market.duckdb")).strip()
    db_path = str((ROOT / db_path) if not Path(db_path).is_absolute() else Path(db_path))
    results_dir = str((paths.get("results_dir") or args.results_dir or "data/results")).strip()
    results_dir = str((ROOT / results_dir) if not Path(results_dir).is_absolute() else Path(results_dir))
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    families = _parse_families(args.families)
    allowed = {"fund_flow", "shareholder"}
    unknown = [x for x in families if x not in allowed]
    if unknown:
        raise ValueError(f"未知 families: {unknown}")

    research_topic = "newdata_quality_checks"
    research_config_id = build_newdata_quality_research_config_id(
        families=families,
        flow_table=args.flow_table,
        shareholder_table=args.shareholder_table,
        daily_table=args.daily_table,
        shareholder_fallback_lag_days=int(args.shareholder_fallback_lag_days),
        min_effective_width=int(args.min_effective_width),
    )
    output_stem = build_newdata_quality_output_stem(
        output_prefix=args.output_prefix,
        research_config_id=research_config_id,
    )
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "newdata_quality_report",
        "research_topic": research_topic,
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "db_path": db_path,
        "families": families,
    }
    with duckdb.connect(db_path, read_only=True) as con:
        if "fund_flow" in families:
            flow_report = run_fund_flow_quality_checks(
                con,
                table=args.flow_table,
                daily_table=args.daily_table,
            )
            payload["fund_flow"] = flow_report.to_dict()
            report_rows.append(
                {
                    "result_type": "newdata_quality_summary",
                    "research_topic": research_topic,
                    "research_config_id": research_config_id,
                    "output_stem": output_stem,
                    "family": "fund_flow",
                    "ok": flow_report.ok,
                    "table_exists": flow_report.table_exists,
                    "total_rows": flow_report.total_rows,
                    "distinct_symbols": flow_report.distinct_symbols,
                    "max_date": flow_report.max_trade_date,
                    "duplicate_pk_rows": flow_report.duplicate_pk_rows,
                    "coverage_ratio_vs_daily": flow_report.coverage_ratio_vs_daily,
                    "rows_without_daily_match": flow_report.rows_without_daily_match,
                    "all_zero_rows": flow_report.all_zero_flow_rows,
                    "notes": " | ".join(flow_report.notes),
                }
            )
        if "shareholder" in families:
            shareholder_report = run_shareholder_quality_checks(
                con,
                table=args.shareholder_table,
                fallback_lag_days=args.shareholder_fallback_lag_days,
                min_effective_width=args.min_effective_width,
            )
            payload["shareholder"] = shareholder_report.to_dict()
            report_rows.append(
                {
                    "result_type": "newdata_quality_summary",
                    "research_topic": research_topic,
                    "research_config_id": research_config_id,
                    "output_stem": output_stem,
                    "family": "shareholder",
                    "ok": shareholder_report.ok,
                    "table_exists": shareholder_report.table_exists,
                    "total_rows": shareholder_report.total_rows,
                    "distinct_symbols": shareholder_report.distinct_symbols,
                    "max_date": shareholder_report.max_end_date,
                    "duplicate_pk_rows": shareholder_report.duplicate_pk_rows,
                    "notice_date_coverage_ratio": shareholder_report.notice_date_coverage_ratio,
                    "fallback_lag_usage_ratio": shareholder_report.fallback_lag_usage_ratio,
                    "median_symbols_per_end_date": shareholder_report.median_symbols_per_end_date,
                    "effective_factor_dates_ge_min_width": shareholder_report.effective_factor_dates_ge_min_width,
                    "notes": " | ".join(shareholder_report.notes),
                }
            )

    report_df = pd.DataFrame(report_rows)
    summary_path = Path(results_dir) / f"{output_stem}_summary.csv"
    json_path = Path(results_dir) / f"{output_stem}.json"
    doc_path = docs_dir / f"{output_stem}.md"
    manifest_path = Path(results_dir) / f"{output_stem}_manifest.json"
    report_df.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    doc_path.write_text(
        _build_doc(
            report_df,
            research_topic=research_topic,
            research_config_id=research_config_id,
            output_stem=output_stem,
            families=families,
        ),
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "result_type": "newdata_quality_manifest",
        "research_topic": research_topic,
        "research_config_id": research_config_id,
        "output_stem": output_stem,
        "families": families,
        "artifacts": [
            str(summary_path.relative_to(ROOT)),
            str(json_path.relative_to(ROOT)),
            str(doc_path.relative_to(ROOT)),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[quality] summary_csv={summary_path}")
    print(f"[quality] json={json_path}")
    print(f"[quality] doc={doc_path}")
    print(f"[quality] manifest={manifest_path}")
    for row in report_rows:
        print(
            f"[quality] {row['family']}: ok={row['ok']} total_rows={row['total_rows']} "
            f"distinct_symbols={row['distinct_symbols']} notes={row['notes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
