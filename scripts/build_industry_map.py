#!/usr/bin/env python3
"""Build and quality-check the canonical A-share industry map (thin CLI wrapper).

Core logic lives in src/data_fetcher/industry_map.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher.industry_map import (
    IndustryMapQuality,
    align_to_universe,
    build_quality_doc,
    deduplicate_mapping,
    fetch_mapping_by_source,
    load_current_universe,
    quality_summary,
)


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建真实行业映射并输出质量报告")
    p.add_argument("--output", default="data/cache/industry_map.csv")
    p.add_argument("--duckdb-path", default="data/market.duckdb")
    p.add_argument("--universe-json", default="data/cache/universe_symbols.json")
    p.add_argument("--sleep-sec", type=float, default=0.2)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-delay-sec", type=float, default=2.0)
    p.add_argument("--request-timeout-sec", type=float, default=12.0)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--source", default="auto", choices=["auto", "sw_official", "eastmoney", "sw", "shenwan"])
    p.add_argument("--asof-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--allow-low-coverage", action="store_true", help="覆盖率不足 90% 时仍返回 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asof_date = str(args.asof_date)
    output = _project_path(args.output)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    universe, universe_source = load_current_universe(
        duckdb_path=args.duckdb_path, universe_json=args.universe_json
    )
    raw = fetch_mapping_by_source(
        args.source,
        sleep_sec=max(float(args.sleep_sec), 0.0),
        asof_date=asof_date,
        retries=max(int(args.retries), 1),
        retry_delay_sec=max(float(args.retry_delay_sec), 0.0),
        request_timeout_sec=max(float(args.request_timeout_sec), 1.0),
        max_workers=max(int(args.max_workers), 1),
        symbols=universe,
    )
    deduped, raw_duplicate_count = deduplicate_mapping(raw)
    final = align_to_universe(deduped, universe, asof_date)
    quality = quality_summary(final, universe, universe_source, asof_date)
    if raw_duplicate_count and quality.duplicate_symbol_count == 0:
        quality = IndustryMapQuality(**{**quality.to_row(), "duplicate_symbol_count": 0})

    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False, encoding="utf-8-sig")

    summary_path = results_dir / f"industry_map_quality_{asof_date}_summary.csv"
    pd.DataFrame([quality.to_row()]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    width = (
        final[final["industry"].astype(str) != "unknown"]
        .groupby("industry", as_index=False)
        .agg(symbol_count=("symbol", "nunique"))
        .sort_values(["symbol_count", "industry"], ascending=[False, True])
    )
    doc_path = docs_dir / f"industry_map_quality_{asof_date}.md"
    doc_path.write_text(build_quality_doc(quality, width, output, PROJECT_ROOT), encoding="utf-8")

    print(f"industry_map -> {output} | rows={len(final)} | coverage={quality.coverage_ratio:.2%}")
    print(f"quality_summary -> {summary_path}")
    print(f"quality_doc -> {doc_path}")
    if (not quality.pass_coverage_90pct or not quality.pass_no_duplicate_symbols) and not args.allow_low_coverage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
