"""fetch_only 核心逻辑，从 scripts/fetch_only.py 迁入。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_fetcher import DuckDBManager, list_default_universe_symbols
from src.data_fetcher.akshare_client import normalize_max_symbols
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config, resolve_asof_trade_end


def run_fetch_only(
    *,
    symbols: str | None,
    max_symbols: int | None,
    config_path: Path | None,
    root: Path,
) -> int:
    """仅增量更新 DuckDB 日线表，不跑因子与月度选股报告。"""
    cfg = load_config(config_path)
    paths = cfg.get("paths", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    feat = cfg.get("features", {})

    lookback = int(feat.get("lookback_trading_days", 160))

    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = root / logs_dir
    setup_app_logging(
        logs_dir,
        name="fetch_only",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("fetch_only")

    if symbols:
        syms = [s.strip().zfill(6) for s in symbols.split(",") if s.strip()]
    else:
        syms = list_default_universe_symbols(
            max_symbols=normalize_max_symbols(max_symbols),
            config_path=config_path,
        )

    if not syms:
        log.error("无可用标的列表，退出。")
        return 1

    end = resolve_asof_trade_end(paths)
    start = end - pd.offsets.BDay(lookback + 20)
    log.info("统一交易日上界: %s", end.date())

    with DuckDBManager(config_path=config_path) as db:
        counts = db.incremental_update_many(
            syms,
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

        df = db.read_daily_frame(symbols=syms, start=start, end=end)
        log.info("库内查询行数（用于窗口校验）: %s", len(df))

    return 0
