#!/usr/bin/env python3
"""加载概念板块数据到 DuckDB（THS 日线 + EM 成分股快照）。

用法::

    # 首次全量回填（日线 OHLCV）
    python scripts/load_concept.py --backfill

    # 限制概念数 (测试用)
    python scripts/load_concept.py --backfill --concept-limit 20

    # 拉取当前概念成分股快照（M13-B）
    python scripts/load_concept.py --backfill-membership

    # M1 质量诊断
    python scripts/load_concept.py --diagnose

注意：成分股数据只有当前快照，研究因子会按 snapshot_date 和 60 天滚动窗口使用，
不会把当前成分回填到历史信号日。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher.concept_client import (
    backfill_all_concepts,
    backfill_concept_membership_snapshot,
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


def cmd_backfill_membership(snapshot_date: str | None, concept_limit: int | None) -> None:
    conn = duckdb.connect(_db_path())
    apply_migrations(conn)
    snap = None
    if snapshot_date:
        snap = date.fromisoformat(snapshot_date)
    result = backfill_concept_membership_snapshot(
        conn,
        snapshot_date=snap,
        concept_limit=concept_limit,
    )
    conn.close()
    print(
        f"Membership: {result['membership_rows']} rows, "
        f"{result['membership_symbols']} symbols, "
        f"{result['concepts_success']}/{result['concepts_total']} concepts success, "
        f"{result['concepts_empty']} empty, {result['concepts_failed']} failed"
    )


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
    print(f"成分股快照行数: {report.membership_rows}")
    print(f"成分股覆盖股票数: {report.membership_symbol_count}")
    print(f"有成分股的概念数: {report.membership_concept_count}")
    print(f"成分股快照日期范围: {report.membership_snapshot_min} ~ {report.membership_snapshot_max}")


def main() -> None:
    p = argparse.ArgumentParser(description="概念板块数据加载器（THS 同花顺）")
    p.add_argument("--backfill", action="store_true", help="回填概念日线 OHLCV")
    p.add_argument("--backfill-membership", action="store_true", help="拉取当前概念成分股快照（M13-B）")
    p.add_argument("--diagnose", action="store_true", help="M1 质量诊断")
    p.add_argument("--start-date", default="20200101", help="日线起始日期 YYYYMMDD")
    p.add_argument("--snapshot-date", default="", help="成分股快照日期 YYYY-MM-DD；默认今天")
    p.add_argument("--concept-limit", type=int, default=None, help="限制概念数（测试用）")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.diagnose:
        cmd_diagnose()
    elif args.backfill_membership:
        cmd_backfill_membership(args.snapshot_date.strip() or None, args.concept_limit)
        cmd_diagnose()
    elif args.backfill:
        cmd_backfill_daily(args.start_date, args.concept_limit)
        cmd_diagnose()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
