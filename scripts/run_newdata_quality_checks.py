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
from src.settings import load_config, resolve_config_path


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


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无质量检查结果_"
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[col]) for col in df.columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _build_budget_decisions(report_df: pd.DataFrame) -> str:
    if report_df.empty:
        return "- 无可用质量检查结果；不继续投入 P2 研究预算。"

    lines: list[str] = []
    for _, row in report_df.iterrows():
        family = str(row.get("family", "unknown"))
        ok = bool(row.get("ok", False))
        notes = str(row.get("notes", "") or "")
        if family == "fund_flow":
            if ok:
                decision = "保留低优先级研究预算，只允许在明确机制假设下继续。"
                reason = "基础覆盖、重复、全零与日线对齐检查通过；但既有 G2/G4 回测未给出主线增量。"
            else:
                decision = "暂停新增研究预算，仅保留数据链路维护。"
                reason = notes or "基础质量检查未通过。"
        elif family == "shareholder":
            if ok:
                decision = "保留极低优先级研究预算，仅在 PIT/覆盖带来新证据时重启。"
                reason = "基础质量检查通过；但既有单因子与 G3 结论仍偏弱。"
            else:
                decision = "暂停新增研究预算，仅保留数据链路维护。"
                reason = notes or "基础质量检查未通过。"
        else:
            decision = "暂不投入研究预算。"
            reason = notes or "未知数据家族。"
        lines.append(f"- `{family}`：{decision}原因：{reason}")
    return "\n".join(lines)


def _build_alignment_breakdown(report_df: pd.DataFrame) -> str:
    if report_df.empty or "family" not in report_df.columns:
        return "_无可用断层拆解。_"

    lines: list[str] = []
    fund = report_df[report_df["family"] == "fund_flow"]
    if not fund.empty:
        row = fund.iloc[0]

        def _int(name: str) -> int:
            val = row.get(name, 0)
            return 0 if pd.isna(val) else int(float(val))

        flow_max = str(row.get("max_date", "") or "")
        daily_max = str(row.get("daily_max_date", "") or "")
        total_missing = _int("rows_without_daily_match")
        after_daily = _int("rows_after_daily_max_date")
        in_span = _int("rows_without_daily_match_within_daily_span")
        absent_rows = _int("rows_without_daily_match_absent_symbols")
        known_rows = _int("rows_without_daily_match_known_symbols")
        absent_symbols = _int("absent_symbol_count")
        lines.append(
            "- `fund_flow`："
            f"资金流最新 `{flow_max}`，日线最新 `{daily_max}`；"
            f"未匹配 `{total_missing}` 行，其中 `{after_daily}` 行晚于日线最新日期，"
            f"`{in_span}` 行位于日线覆盖区间内。覆盖区间内未匹配里，"
            f"`{absent_rows}` 行来自日线表完全没有的 `{absent_symbols}` 个标的，"
            f"`{known_rows}` 行是日线已覆盖标的但具体日期未匹配。"
        )

    holder = report_df[report_df["family"] == "shareholder"]
    if not holder.empty:
        row = holder.iloc[0]
        notes = str(row.get("notes", "") or "")
        lines.append(f"- `shareholder`：当前主要断点仍是 `{notes or '未发现额外拆解信息'}`。")
    return "\n".join(lines) if lines else "_无可用断层拆解。_"


def _build_doc(
    report_df: pd.DataFrame,
    *,
    research_topic: str,
    research_config_id: str,
    output_stem: str,
    config_source: str,
    families: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    table = _format_markdown_table(report_df)
    budget_decisions = _build_budget_decisions(report_df)
    alignment_breakdown = _build_alignment_breakdown(report_df)
    return f"""# Newdata Quality Checks

- 生成时间：`{generated_at}`
- 结果类型：`newdata_quality_summary`
- 研究主题：`{research_topic}`
- 研究配置：`{research_config_id}`
- 输出 stem：`{output_stem}`
- 配置来源：`{config_source}`
- 数据家族：`{", ".join(families) if families else "none"}`

## Summary

{table}

## Research Budget Decision

{budget_decisions}

## Alignment Breakdown

{alignment_breakdown}

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
    config_source = str(resolve_config_path(args.config)) if args.config is not None else "default_config_lookup"
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
        "config_source": config_source,
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
                    "config_source": config_source,
                    "family": "fund_flow",
                    "ok": flow_report.ok,
                    "table_exists": flow_report.table_exists,
                    "total_rows": flow_report.total_rows,
                    "distinct_symbols": flow_report.distinct_symbols,
                    "max_date": flow_report.max_trade_date,
                    "daily_max_date": flow_report.daily_max_trade_date,
                    "duplicate_pk_rows": flow_report.duplicate_pk_rows,
                    "coverage_ratio_vs_daily": flow_report.coverage_ratio_vs_daily,
                    "rows_without_daily_match": flow_report.rows_without_daily_match,
                    "rows_after_daily_max_date": flow_report.rows_after_daily_max_date,
                    "rows_without_daily_match_within_daily_span": (
                        flow_report.rows_without_daily_match_within_daily_span
                    ),
                    "rows_without_daily_match_absent_symbols": flow_report.rows_without_daily_match_absent_symbols,
                    "rows_without_daily_match_known_symbols": flow_report.rows_without_daily_match_known_symbols,
                    "absent_symbol_count": flow_report.absent_symbol_count,
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
                    "config_source": config_source,
                    "family": "shareholder",
                    "ok": shareholder_report.ok,
                    "table_exists": shareholder_report.table_exists,
                    "total_rows": shareholder_report.total_rows,
                    "distinct_symbols": shareholder_report.distinct_symbols,
                    "max_date": shareholder_report.max_end_date,
                    "duplicate_pk_rows": shareholder_report.duplicate_pk_rows,
                    "notice_date_coverage_ratio": shareholder_report.notice_date_coverage_ratio,
                    "fallback_lag_usage_ratio": shareholder_report.fallback_lag_usage_ratio,
                    "negative_notice_lag_rows": shareholder_report.negative_notice_lag_rows,
                    "median_notice_lag_days": shareholder_report.median_notice_lag_days,
                    "p90_notice_lag_days": shareholder_report.p90_notice_lag_days,
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
            config_source=config_source,
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
        "config_source": config_source,
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
