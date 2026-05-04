#!/usr/bin/env python3
"""增量拉取 A 股基本面数据并写入 DuckDB `a_share_fundamental`。

用法:
    python scripts/fetch_fundamental.py --schema-only
    python scripts/fetch_fundamental.py --max-symbols 300
    python scripts/fetch_fundamental.py --symbols 600519,000001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import FundamentalClient, list_default_universe_symbols
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="抓取并更新基本面数据（PIT）")
    p.add_argument("--schema-only", action="store_true", help="仅创建/校验 DuckDB 表及索引，不发起网络抓取")
    p.add_argument("--max-symbols", type=int, default=None, help="仅抓取前 N 只（调试用）")
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔代码列表，优先级高于 --max-symbols")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径")
    p.add_argument("--disclosure-calendar", action="store_true", help="拉取财报实际披露日期写入 a_share_disclosure_date 表")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths") or {}
    logs_dir = Path((paths).get("logs_dir", "data/logs"))
    if not logs_dir.is_absolute():
        logs_dir = ROOT / logs_dir
    setup_app_logging(
        logs_dir,
        name="fetch_fundamental",
        log_format=str((cfg.get("logging") or {}).get("format", "json")),
    )
    log = get_logger("fetch_fundamental")

    if args.schema_only:
        with FundamentalClient(config_path=args.config) as fc:
            log.info("基本面表: %s @ %s", fc.cfg.table_name, fc.cfg.duckdb_path)
        log.info("已确保基本面表结构就绪（未抓取数据）。")
        return 0

    if args.disclosure_calendar:
        from src.cli.fetch_fundamental import run_disclosure_calendar

        db_path = Path((paths).get("duckdb_path", "data/market.duckdb"))
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return run_disclosure_calendar(
            db_path=db_path,
            symbols=args.symbols.split(",") if args.symbols else None,
            max_symbols=args.max_symbols,
            config_path=args.config,
            log=log,
        )

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(max_symbols=args.max_symbols, config_path=args.config)

    if not symbols:
        log.error("无可用代码，退出。")
        return 1

    with FundamentalClient(config_path=args.config) as fc:
        n = fc.update_symbols(symbols)
    log.info("基本面更新完成：symbols=%d, upsert_rows=%d", len(symbols), n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
