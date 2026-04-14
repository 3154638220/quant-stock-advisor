#!/usr/bin/env python3
"""
仅对 DuckDB 日线表执行全表回填：amplitude_pct / pct_chg / change 中仍为 NULL 的行，
用前收与当日 OHLC 补算（与增量写入时逻辑一致）。不拉取网络数据。

用法（项目根目录，conda 环境 quant-system）::

    conda activate quant-system
    python scripts/backfill_derived_daily.py
    python scripts/backfill_derived_daily.py --config /path/to/config.yaml

退出码：0 成功。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher import DuckDBManager
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="全表回填日线衍生列（振幅、涨跌幅、涨跌额）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.yaml；默认项目根目录 config.yaml",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = ROOT / logs_dir
    setup_app_logging(
        logs_dir,
        name="backfill_derived_daily",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("backfill_derived_daily")

    with DuckDBManager(config_path=args.config) as db:
        fixed = db.backfill_derived_daily_columns()
        log.info("全表回填完成，本轮减少含 NULL 行数: %s", fixed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
