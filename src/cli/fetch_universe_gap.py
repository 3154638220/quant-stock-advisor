"""fetch_universe_gap 核心逻辑，从 scripts/fetch_universe_gap.py 迁入。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_fetcher import DuckDBManager
from src.data_fetcher.akshare_client import (
    _fetch_akshare_universe_codes,
    _list_symbols_from_duckdb,
    _load_symbols_from_local_cache,
)
from src.logging_config import get_logger, setup_app_logging
from src.settings import load_config, resolve_asof_trade_end


def run_fetch_universe_gap(
    *,
    config_path: Path | None,
    dry_run: bool,
    use_cache: bool,
    root: Path,
) -> int:
    """补拉全市场代码表相对 DuckDB 尚未有线数据的标的。"""
    cfg = load_config(config_path)
    paths = cfg.get("paths", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}
    ak_cfg = cfg.get("akshare", {}) or {}

    logs_dir = paths.get("logs_dir", "data/logs")
    if not Path(logs_dir).is_absolute():
        logs_dir = root / logs_dir
    setup_app_logging(
        logs_dir,
        name="fetch_universe_gap",
        log_format=str(log_cfg.get("format", "json")),
    )
    log = get_logger("fetch_universe_gap")

    in_db = set(_list_symbols_from_duckdb(config_path=config_path))
    log.info("DuckDB  distinct symbols: %s", len(in_db))

    universe: list[str] = []
    source = ""
    if use_cache:
        universe = _load_symbols_from_local_cache(config_path=config_path)
        source = "local_universe_cache"
    else:
        codes = _fetch_akshare_universe_codes(
            config_path=config_path,
            source_timeout_sec=float(ak_cfg.get("universe_source_timeout_sec", 60.0)),
            source_retries=max(1, int(ak_cfg.get("universe_source_retries", 2))),
            retry_delay_sec=float(ak_cfg.get("universe_retry_delay_sec", 3.0)),
        )
        if codes:
            universe = codes
            source = "akshare_universe"
        else:
            universe = _load_symbols_from_local_cache(config_path=config_path)
            source = "local_universe_cache_fallback"

    if not universe:
        log.error("无可用全市场代码表（网络失败且本地无 universe 快照）。请先成功跑过一次全市场拉取或手动更新缓存。")
        return 1

    uni_set = {s.zfill(6) for s in universe}
    missing = sorted(uni_set - in_db)
    log.info(
        "代码表来源: %s | 表内标的数: %s | 库内已有: %s | 待补标的: %s",
        source,
        len(uni_set),
        len(in_db),
        len(missing),
    )
    if not missing:
        log.info("无缺口，退出。")
        return 0

    if dry_run:
        log.info("dry-run：前 20 个待补代码: %s", missing[:20])
        return 0

    feat = cfg.get("features", {}) or {}
    lookback = int(feat.get("lookback_trading_days", 160))
    end = resolve_asof_trade_end(paths)
    start = end - pd.offsets.BDay(lookback + 20)
    log.info("统一交易日上界: %s | 待补 %s 只", end.date(), len(missing))

    with DuckDBManager(config_path=config_path) as db:
        counts = db.incremental_update_many(missing, end_date=end.strftime("%Y%m%d"))
        n_written = sum(r.rows_written for r in counts.values())
        n_fail = sum(1 for r in counts.values() if r.fetch_failed)
        log.info(
            "补拉完成 | 写入总行数: %s | 拉取失败标的数: %s | run_id=%s",
            n_written,
            n_fail,
            db.last_fetch_run_id,
        )
        df = db.read_daily_frame(symbols=missing, start=start, end=end)
        log.info("库内查询行数（缺口标的窗口）: %s", len(df))

    return 0
