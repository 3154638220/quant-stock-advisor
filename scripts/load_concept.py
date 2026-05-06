#!/usr/bin/env python3
"""加载概念板块数据到 DuckDB（同花顺 THS 数据源）。

用法::

    # 首次全量回填（日线 OHLCV）
    python scripts/load_concept.py --backfill

    # 限制概念数 (测试用)
    python scripts/load_concept.py --backfill --concept-limit 20

    # M1 质量诊断
    python scripts/load_concept.py --diagnose

注意：当前仅回填概念板块日线 OHLCV（板块层面 breadth），不做个股-板块绑定。
THS 无成分股映射 API，东方财富 API 已被封。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher.concept_client import (
    backfill_all_concepts,
    diagnose_concept_quality,
)
from src.data_fetcher.migrations import apply_migrations

_LOG = logging.getLogger(__name__)


def _db_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "data" / "market.duckdb")


def cmd_backfill_daily(start_date: str, concept_limit: int | None) -> None:
    conn = duckdb.connect(_db_path())
    apply_migrations(conn)
    result = backfill_all_concepts(
        conn,
        start_date=start_date,
        concept_limit=concept_limit,
    )
    conn.close()
    print(f"Concepts: {result['concepts_success']}/{result['concepts_total']} success, "
          f"{result['concepts_empty']} empty, {result['total_rows']} total rows")


def cmd_diagnose() -> None:
    conn = duckdb.connect(_db_path(), read_only=True)
    report = diagnose_concept_quality(conn)
    conn.close()
    print(f"概念总数: {report.concept_count}")
    print(f"日线日期范围: {report.daily_date_min} ~ {report.daily_date_max}")
    print(f"日线总行数: {report.daily_total_rows}")
    print(f"有日线数据的概念数: {report.daily_concept_count}")
    print(f"pct_chg 空值率: {report.pct_chg_null_rate:.4f}")
    print(f"概念板块日线覆盖率: {report.coverage_pct:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="概念板块数据加载器（THS 同花顺）")
    p.add_argument("--backfill", action="store_true", help="回填概念日线 OHLCV")
    p.add_argument("--diagnose", action="store_true", help="M1 质量诊断")
    p.add_argument("--start-date", default="20200101", help="日线起始日期 YYYYMMDD")
    p.add_argument("--concept-limit", type=int, default=None, help="限制概念数（测试用）")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.diagnose:
        cmd_diagnose()
    elif args.backfill:
        cmd_backfill_daily(args.start_date, args.concept_limit)
        cmd_diagnose()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
