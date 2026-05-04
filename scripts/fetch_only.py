#!/usr/bin/env python3
"""
仅增量更新 DuckDB 日线表，不跑因子与月度选股报告（适合网络不稳定时先拉数）。

主入口（项目根目录，须使用 conda 环境 quant-system）::

    conda activate quant-system
    python scripts/fetch_only.py --max-symbols 200
    python scripts/fetch_only.py --max-symbols 0    # 全量更新
    python scripts/fetch_only.py --symbols 600519,000001
    python scripts/fetch_only.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config, resolve_asof_trade_end

EPILOG = """
参数说明
  --max-symbols N
    仅拉取全市场列表前 N 只（与 AkShare 全市场快照顺序一致）；调试用。
    传 0 或负数表示不截断，即全量更新。
    与 --symbols 互斥：一旦指定 --symbols，则忽略本项。

  --symbols A,B,...
    逗号分隔 6 位 A 股代码（可不带前导零）。指定后只更新这些标的。

  --config PATH
    自定义 config.yaml；未指定时使用项目根目录下 config.yaml。
    其中的 paths.duckdb_path、akshare.*、features.lookback_trading_days 等均生效。

退出码
  0 成功；1 无标的列表或其它致命错误。

失败标的日志
  单标的多次重试仍失败时，会追加写入 logs_dir 下
  akshare_symbol_failed_YYYYMMDD.log（与日线增量逻辑一致）。
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="仅 AkShare → DuckDB 增量日线（不计算月度因子）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="全市场列表前 N 只；未指定 --symbols 时生效",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="逗号分隔 6 位代码；指定时忽略全市场列表",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.yaml 路径；默认项目根目录 config.yaml",
    )
    return p.parse_args()


# （A3 迁移：normalize_max_symbols 已迁入 src/data_fetcher/akshare_client.py，
#  此处从 src/ 重新导出以保持向后兼容。）
from src.data_fetcher.akshare_client import normalize_max_symbols  # noqa: F401


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    feat = cfg.get("features", {})

    lookback = int(feat.get("lookback_trading_days", 160))

    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = ROOT / logs_dir
    setup_app_logging(
        logs_dir,
        name="fetch_only",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("fetch_only")

    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_default_universe_symbols(
            max_symbols=normalize_max_symbols(args.max_symbols),
            config_path=args.config,
        )

    if not symbols:
        log.error("无可用标的列表，退出。")
        return 1

    end = resolve_asof_trade_end(paths)
    start = end - pd.offsets.BDay(lookback + 20)
    log.info("统一交易日上界: %s", end.date())

    with DuckDBManager(config_path=args.config) as db:
        counts = db.incremental_update_many(
            symbols,
            end_date=end.strftime("%Y%m%d"),
        )
        n_written = sum(r.rows_written for r in counts.values())
        n_fail = sum(1 for r in counts.values() if r.fetch_failed)
        log.info(
            "增量写入总行数: %s | 拉取失败标的数: %s | run_id=%s",
            n_written,
            n_fail,
            db.last_fetch_run_id,
        )

        qrep = db.quality_report()
        log.info("数据质量: %s", qrep.summary())
        if not qrep.ok:
            log.warning("质量未通过: %s", "; ".join(qrep.notes) if qrep.notes else qrep.summary())
        elif qrep.ohlc_invalid_rows or qrep.large_gap_rows:
            log.info(
                "数据特征统计（源站/长假/停牌等，非拉取失败）: ohlc_invalid=%s, large_gaps=%s",
                qrep.ohlc_invalid_rows,
                qrep.large_gap_rows,
            )

        df = db.read_daily_frame(symbols=symbols, start=start, end=end)
        log.info("库内查询行数（用于窗口校验）: %s", len(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
