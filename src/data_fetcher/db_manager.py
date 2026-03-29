"""DuckDB 本地落盘与按交易日增量更新。"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import duckdb
import pandas as pd
import yaml

from .akshare_client import fetch_a_share_daily
from .data_quality import QualityConfig, QualityReport, run_quality_checks, validate_daily_frame

_LOG = logging.getLogger(__name__)


class SymbolUpdateResult(NamedTuple):
    """单标的增量结果：写入行数与是否拉取失败（与「无新数据」区分）。"""

    rows_written: int
    fetch_failed: bool


def _load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class DuckDBManager:
    """
    管理 DuckDB 连接、日线表结构，以及按 ``symbol`` 的增量拉取与写入。
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        *,
        duckdb_path: Optional[str] = None,
        table_daily: Optional[str] = None,
    ) -> None:
        root = _project_root()
        cfg_path = Path(config_path) if config_path else root / "config.yaml"
        cfg = _load_yaml_config(cfg_path) if cfg_path.exists() else {}

        paths = cfg.get("paths", {})
        db_cfg = cfg.get("database", {})
        ak_cfg = cfg.get("akshare", {})
        qual_cfg = cfg.get("quality", {})

        self._duckdb_path = duckdb_path or paths.get("duckdb_path", "data/market.duckdb")
        self._table = table_daily or db_cfg.get("table_daily", "a_share_daily")
        self._table_audit = db_cfg.get("table_audit", "data_fetch_audit")
        self._quality = QualityConfig.from_mapping(qual_cfg if isinstance(qual_cfg, dict) else None)
        self._adjust = ak_cfg.get("adjust", "qfq")
        self._sleep = float(ak_cfg.get("sleep_between_symbols_sec", 0.0))
        self._max_fetch_retries = max(1, int(ak_cfg.get("max_fetch_retries", 3)))
        self._retry_delay_sec = float(ak_cfg.get("retry_delay_sec", 2.0))
        self._timeout_sec = float(ak_cfg.get("request_timeout_sec", 10.0))

        logs_dir = paths.get("logs_dir", "data/logs")
        ld = Path(logs_dir)
        if not ld.is_absolute():
            ld = root / ld
        self._fetch_fail_log_dir = ld

        abs_db = Path(self._duckdb_path)
        if not abs_db.is_absolute():
            abs_db = root / abs_db
        self._duckdb_path_abs = abs_db
        abs_db.parent.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(str(abs_db))
        self._ensure_schema()
        self._ensure_audit_schema()
        self._last_fetch_run_id: Optional[str] = None

    def _append_fetch_failure_log(self, symbol: str, exc: BaseException) -> None:
        """失败标的写入单独日志（按日滚动文件名），便于事后补拉。"""
        self._fetch_fail_log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"akshare_symbol_failed_{datetime.now().strftime('%Y%m%d')}.log"
        path = self._fetch_fail_log_dir / stem
        line = (
            f"{datetime.now().isoformat(timespec='seconds')}\t{symbol}\t"
            f"{type(exc).__name__}: {exc}\n"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    @property
    def last_fetch_run_id(self) -> Optional[str]:
        """最近一次 ``incremental_update_many`` 使用的 ``run_id``（审计主键）。"""
        return self._last_fetch_run_id

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                symbol VARCHAR NOT NULL,
                trade_date DATE NOT NULL,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                amplitude_pct DOUBLE,
                pct_chg DOUBLE,
                change DOUBLE,
                turnover DOUBLE,
                PRIMARY KEY (symbol, trade_date)
            )
            """
        )

    def _ensure_audit_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_audit} (
                run_id VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP NOT NULL,
                symbol_count INTEGER NOT NULL,
                rows_written BIGINT NOT NULL,
                failures INTEGER NOT NULL,
                duration_ms BIGINT NOT NULL,
                PRIMARY KEY (run_id)
            )
            """
        )

    def record_fetch_audit(
        self,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        symbol_count: int,
        rows_written: int,
        failures: int,
        duration_ms: int,
    ) -> None:
        """写入一次拉取审计记录（幂等：同一 ``run_id`` 仅应插入一次）。"""
        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._table_audit}
            (run_id, started_at, finished_at, symbol_count, rows_written, failures, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                started_at,
                finished_at,
                symbol_count,
                rows_written,
                failures,
                duration_ms,
            ],
        )

    def last_trade_date(self, symbol: str) -> Optional[date]:
        row = self._conn.execute(
            f"""
            SELECT MAX(trade_date) FROM {self._table}
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        d = row[0]
        if isinstance(d, datetime):
            return d.date()
        return d if isinstance(d, date) else pd.Timestamp(d).date()

    def incremental_update_symbol(
        self,
        symbol: str,
        *,
        default_start: str = "20150101",
        end_date: Optional[str] = None,
    ) -> SymbolUpdateResult:
        """
        对单标的增量更新：从库中最后交易日的下一自然日拉到 ``end_date``（默认今天）。

        Returns
        -------
        SymbolUpdateResult
            ``rows_written`` 与 ``fetch_failed``（拉取经重试仍失败时为 True）。
        """
        today = datetime.now().date()
        end_s = end_date or today.strftime("%Y%m%d")
        end_dt = pd.to_datetime(end_s).date()

        last = self.last_trade_date(symbol)
        if last is None:
            start_s = default_start
        else:
            start_dt = last + timedelta(days=1)
            if start_dt > end_dt:
                return SymbolUpdateResult(0, False)
            start_s = start_dt.strftime("%Y%m%d")

        df: Optional[pd.DataFrame] = None
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_fetch_retries):
            try:
                df = fetch_a_share_daily(
                    symbol,
                    start_s,
                    end_s,
                    adjust=self._adjust,
                    timeout_sec=self._timeout_sec,
                )
                break
            except Exception as e:
                last_exc = e
                _LOG.warning(
                    "AkShare 拉取失败 symbol=%s 第 %s/%s 次: %s",
                    symbol,
                    attempt + 1,
                    self._max_fetch_retries,
                    e,
                )
                if attempt < self._max_fetch_retries - 1:
                    time.sleep(self._retry_delay_sec * (attempt + 1))
                else:
                    if last_exc is not None:
                        self._append_fetch_failure_log(symbol, last_exc)
                    return SymbolUpdateResult(0, True)
        if df is None:
            return SymbolUpdateResult(0, True)

        if df.empty:
            return SymbolUpdateResult(0, False)

        # 只保留不超过 end 的交易日
        df = df[df["trade_date"] <= pd.Timestamp(end_dt)]
        if df.empty:
            return SymbolUpdateResult(0, False)

        # 增量幂等：同批重复主键只保留最后一条，再与库内主键 INSERT OR REPLACE 合并
        df = df.sort_values("trade_date").drop_duplicates(
            subset=["symbol", "trade_date"],
            keep="last",
        )
        qrep = validate_daily_frame(df, cfg=self._quality)
        if not qrep.ok:
            _LOG.warning("写入前质量提示 symbol=%s: %s", symbol, qrep.summary())
            if qrep.notes:
                _LOG.warning("%s", "; ".join(qrep.notes))

        self._conn.register("df_incr", df)
        try:
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._table}
                SELECT * FROM df_incr
                """
            )
        finally:
            self._conn.unregister("df_incr")
        return SymbolUpdateResult(len(df), False)

    def incremental_update_many(
        self,
        symbols: List[str],
        *,
        default_start: str = "20150101",
        end_date: Optional[str] = None,
        record_audit: bool = True,
        run_id: Optional[str] = None,
    ) -> Dict[str, SymbolUpdateResult]:
        """批量增量更新；可选写入审计表（``run_id`` 默认新建 UUID）。"""
        rid = run_id or str(uuid.uuid4())
        self._last_fetch_run_id = rid
        started_at = datetime.now()
        t0 = time.perf_counter()
        counts: Dict[str, SymbolUpdateResult] = {}
        for sym in symbols:
            counts[sym] = self.incremental_update_symbol(
                sym,
                default_start=default_start,
                end_date=end_date,
            )
            if self._sleep > 0:
                time.sleep(self._sleep)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        rows_written = sum(r.rows_written for r in counts.values())
        failures = sum(1 for r in counts.values() if r.fetch_failed)
        if record_audit:
            finished_at = datetime.now()
            self.record_fetch_audit(
                rid,
                started_at,
                finished_at,
                len(symbols),
                rows_written,
                failures,
                duration_ms,
            )
        return counts

    def quality_report(self) -> QualityReport:
        """对当前日线表做只读质量检查。"""
        return run_quality_checks(self._conn, self._table, self._quality)

    def read_daily_frame(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[Union[str, date]] = None,
        end: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """读取日线为 DataFrame，供 Polars/张量因子使用。"""
        conds: List[str] = []
        params: List[Any] = []
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conds.append(f"symbol IN ({placeholders})")
            params.extend(symbols)
        if start is not None:
            conds.append("trade_date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            conds.append("trade_date <= ?")
            params.append(pd.Timestamp(end).date())
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        q = f"SELECT * FROM {self._table}{where} ORDER BY symbol, trade_date"
        return self._conn.execute(q, params).df()
