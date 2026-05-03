#!/usr/bin/env python3
"""
增量拉取 A 股基本面数据并写入 DuckDB `a_share_fundamental`。

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
    p.add_argument(
        "--schema-only",
        action="store_true",
        help="仅创建/校验 DuckDB 表 a_share_fundamental 及索引，不发起网络抓取",
    )
    p.add_argument("--max-symbols", type=int, default=None, help="仅抓取前 N 只（调试用）")
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔代码列表，优先级高于 --max-symbols")
    p.add_argument("--config", type=Path, default=None, help="配置文件路径")
    p.add_argument(
        "--disclosure-calendar",
        action="store_true",
        help="从 akshare 拉取财报实际披露日期，写入 a_share_disclosure_date 表（P2-3）。",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    logs_dir = Path((cfg.get("paths") or {}).get("logs_dir", "data/logs"))
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
        return _run_disclosure_calendar(args, cfg, log)

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


def _run_disclosure_calendar(args, cfg, log) -> int:
    """P2-3: 从 akshare 拉取财报实际披露日期并写入 DuckDB。"""
    import duckdb
    try:
        import akshare as ak
    except ModuleNotFoundError:
        log.error("缺少 akshare 依赖，无法拉取披露日历。")
        return 1

    db_path = Path((cfg.get("paths") or {}).get("duckdb_path", "data/market.duckdb"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(max_symbols=args.max_symbols, config_path=args.config)

    if not symbols:
        log.error("无可用代码，退出。")
        return 1

    con = duckdb.connect(str(db_path))
    try:
        # 建表
        con.execute("""
            CREATE TABLE IF NOT EXISTS a_share_disclosure_date (
                symbol VARCHAR NOT NULL,
                report_period DATE NOT NULL,
                disclosure_date DATE NOT NULL,
                report_type VARCHAR,
                PRIMARY KEY (symbol, report_period)
            )
        """)

        total_inserted = 0
        for symbol in symbols:
            try:
                em_sym = f"{symbol}.SH" if symbol.startswith(("60", "68")) else (
                    f"{symbol}.SZ" if symbol.startswith(("00", "30")) else f"{symbol}.BJ"
                )
                df = ak.stock_financial_report_disclosure_date_cninfo(symbol=em_sym)
                if df is None or df.empty:
                    log.debug("披露日历为空: %s", symbol)
                    continue

                # akshare 返回列名可能为中文，尝试多种映射
                period_col = next((c for c in df.columns if c in ("报告期", "财务报告报告期", "end_date")), None)
                disc_col = next((c for c in df.columns if c in ("实际披露日期", "实际披露日", "披露日期", "publish_date")), None)
                report_type_col = next((c for c in df.columns if c in ("报告类型", "报告类型", "report_type")), None)

                if period_col is None or disc_col is None:
                    log.debug("披露日日历列名不匹配: %s columns=%s", symbol, list(df.columns)[:10])
                    continue

                df["_symbol"] = _norm_symbol(symbol)
                df["_period"] = pd.to_datetime(df[period_col], errors="coerce").dt.normalize()
                df["_disc"] = pd.to_datetime(df[disc_col], errors="coerce").dt.normalize()
                df["_rtype"] = df[report_type_col].astype(str) if report_type_col else ""

                valid = df.dropna(subset=["_symbol", "_period", "_disc"])
                if valid.empty:
                    continue

                con.register("disc_in", valid[["_symbol", "_period", "_disc", "_rtype"]])
                con.execute("""
                    INSERT OR REPLACE INTO a_share_disclosure_date (symbol, report_period, disclosure_date, report_type)
                    SELECT _symbol, _period, _disc, _rtype FROM disc_in
                """)
                con.unregister("disc_in")
                total_inserted += len(valid)
                log.debug("披露日历: %s rows=%d", symbol, len(valid))
            except Exception as exc:
                log.debug("披露日历拉取失败: %s %s", symbol, exc)
            time.sleep(0.15)  # 控制频率

        log.info("披露日历更新完成：symbols=%d, upsert_rows=%d", len(symbols), total_inserted)
    finally:
        con.close()
    return 0


def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if s.isdigit():
        return s.zfill(6)
    import re
    m = re.search(r"(\d{6})", s)
    return m.group(1).zfill(6) if m else s


if __name__ == "__main__":
    raise SystemExit(main())
