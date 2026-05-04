"""A 股股东人数数据落库与近似 PIT 查询。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Union

import pandas as pd

try:
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover
    def _missing_stock_zh_a_gdhs(*args, **kwargs):
        raise ModuleNotFoundError("缺少 akshare 依赖，无法抓取股东人数数据。")

    ak = SimpleNamespace(stock_zh_a_gdhs=_missing_stock_zh_a_gdhs)

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

from ..settings import load_config, project_root

_LOG = logging.getLogger(__name__)


def _require_akshare():
    return ak


def _require_duckdb():
    if duckdb is None:
        raise ModuleNotFoundError("缺少 duckdb 依赖，无法读写股东人数本地库。")
    return duckdb

SHAREHOLDER_TABLE_COLS: tuple[str, ...] = (
    "symbol",
    "end_date",
    "notice_date",
    "holder_count",
    "holder_change",
    "source",
    "fetched_at",
)

# P2-4: 除权日历表列定义
DIVIDEND_CALENDAR_TABLE_COLS: tuple[str, ...] = (
    "symbol",
    "ex_dividend_date",
    "dividend_plan",
    "source",
    "fetched_at",
)

DIVIDEND_CALENDAR_TABLE_NAME: str = "a_share_dividend_calendar"


@dataclass(frozen=True)
class ShareholderUpdateSummary:
    total_rows: int
    success_dates: int
    failed_dates: int


def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if s.isdigit():
        return s.zfill(6)
    m = pd.Series([s]).astype(str).str.extract(r"(\d{6})", expand=False).iloc[0]
    return str(m).zfill(6) if isinstance(m, str) and m else s


class ShareholderClient:
    """管理股东人数表的写入与按日期读取。"""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        *,
        duckdb_path: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        cfg = load_config(Path(config_path)) if config_path else load_config()
        root = project_root()
        db_path = duckdb_path or str((cfg.get("paths") or {}).get("duckdb_path", "data/market.duckdb"))
        tbl = table_name or str((cfg.get("database") or {}).get("table_shareholder", "a_share_shareholder"))
        p = Path(db_path)
        if not p.is_absolute():
            p = root / p
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = _require_duckdb().connect(str(p))
        self._table_name = tbl
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ShareholderClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        t = self._table_name
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                symbol VARCHAR NOT NULL,
                end_date DATE NOT NULL,
                notice_date DATE,
                holder_count BIGINT,
                holder_change BIGINT,
                source VARCHAR,
                fetched_at TIMESTAMP,
                PRIMARY KEY (symbol, end_date)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{t}_symbol_end_date
            ON {t}(symbol, end_date)
            """
        )
        existing = {str(r[1]) for r in self._conn.execute(f"PRAGMA table_info('{t}')").fetchall()}
        if "notice_date" not in existing:
            self._conn.execute(f"ALTER TABLE {t} ADD COLUMN notice_date DATE")

        # P2-4: 除权日历表，用于过滤送配股导致的 holder_change_rate 异常值
        self._ensure_dividend_calendar_schema()

    def _ensure_dividend_calendar_schema(self) -> None:
        """P2-4: 创建除权日历表（若不存在）。"""
        dt = DIVIDEND_CALENDAR_TABLE_NAME
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {dt} (
                symbol VARCHAR NOT NULL,
                ex_dividend_date DATE NOT NULL,
                dividend_plan VARCHAR,
                source VARCHAR,
                fetched_at TIMESTAMP,
                PRIMARY KEY (symbol, ex_dividend_date)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{dt}_symbol_exdiv
            ON {dt}(symbol, ex_dividend_date)
            """
        )

    @staticmethod
    def _prepare_shareholder_frame(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        required = {"代码", "股东户数统计截止日-本次", "股东户数-本次"}
        if not required.issubset(set(raw.columns)):
            return pd.DataFrame()

        out = pd.DataFrame(
            {
                "symbol": raw["代码"].map(_norm_symbol),
                "end_date": pd.to_datetime(raw["股东户数统计截止日-本次"], errors="coerce").dt.normalize(),
                "notice_date": pd.to_datetime(raw.get("公告日期"), errors="coerce").dt.normalize(),
                "holder_count": pd.to_numeric(raw["股东户数-本次"], errors="coerce").astype("Int64"),
                "holder_change": pd.to_numeric(raw.get("股东户数-增减"), errors="coerce").astype("Int64"),
                "source": "stock_zh_a_gdhs",
                "fetched_at": pd.Timestamp.now(),
            }
        )
        out = out.dropna(subset=["end_date"]).copy()
        if out.empty:
            return out
        return out[list(SHAREHOLDER_TABLE_COLS)].drop_duplicates(["symbol", "end_date"], keep="last")

    def fetch_end_date_shareholder(
        self,
        end_date: str,
        *,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> pd.DataFrame:
        key = str(end_date).replace("-", "")
        retries = max(1, int(max_retries))
        for attempt in range(retries):
            try:
                raw = _require_akshare().stock_zh_a_gdhs(symbol=key)
                return self._prepare_shareholder_frame(raw)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "股东人数拉取失败 end_date=%s 第 %d/%d 次: %s",
                    key,
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1 and retry_delay_sec > 0:
                    time.sleep(float(retry_delay_sec) * (attempt + 1))
        return pd.DataFrame()

    def upsert(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        aligned = df.copy()
        for c in SHAREHOLDER_TABLE_COLS:
            if c not in aligned.columns:
                aligned[c] = pd.NA
        aligned = aligned[list(SHAREHOLDER_TABLE_COLS)]
        aligned["source"] = aligned["source"].astype(object).where(aligned["source"].notna(), "")
        self._conn.register("sh_in", aligned)
        try:
            cols_sql = ", ".join(SHAREHOLDER_TABLE_COLS)
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._table_name} ({cols_sql})
                SELECT {cols_sql} FROM sh_in
                """
            )
        finally:
            self._conn.unregister("sh_in")
        return int(len(aligned))

    def update_end_dates_summary(
        self,
        end_dates: Iterable[str],
        *,
        sleep_sec: float = 0.5,
        log_every: int = 1,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> ShareholderUpdateSummary:
        total = 0
        success = 0
        fail = 0
        for i, end_date in enumerate(end_dates):
            df = self.fetch_end_date_shareholder(
                end_date,
                max_retries=max_retries,
                retry_delay_sec=retry_delay_sec,
            )
            if not df.empty:
                total += self.upsert(df)
                success += 1
                if success % log_every == 0:
                    _LOG.info("股东人数已处理 %d 个截止日（成功 %d，失败 %d）", i + 1, success, fail)
            else:
                fail += 1
                _LOG.error("股东人数截止日抓取失败或返回空数据: %s", str(end_date).replace("-", ""))
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        _LOG.info("股东人数完成：成功 %d，失败 %d，总行数 %d", success, fail, total)
        return ShareholderUpdateSummary(
            total_rows=total,
            success_dates=success,
            failed_dates=fail,
        )

    def update_end_dates(
        self,
        end_dates: Iterable[str],
        *,
        sleep_sec: float = 0.5,
        log_every: int = 1,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> int:
        summary = self.update_end_dates_summary(
            end_dates,
            sleep_sec=sleep_sec,
            log_every=log_every,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
        )
        return int(summary.total_rows)

    def load_by_date_range(
        self,
        *,
        start_date: Union[str, date, pd.Timestamp],
        end_date: Union[str, date, pd.Timestamp],
        symbols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        t = self._table_name
        s = pd.Timestamp(start_date).date()
        e = pd.Timestamp(end_date).date()
        sql = f"SELECT * FROM {t} WHERE end_date >= ? AND end_date <= ?"
        params: list[object] = [s, e]
        if symbols:
            sym_list = [str(_norm_symbol(sym)) for sym in symbols]
            if sym_list:
                placeholders = ", ".join(["?"] * len(sym_list))
                sql += f" AND symbol IN ({placeholders})"
                params.extend(sym_list)
        return self._conn.execute(sql, params).df()


# ── 季度末日期工具（A3: 从 scripts/fetch_shareholder.py 迁入 src/） ──


def _latest_completed_quarter_end(asof: str | pd.Timestamp | None = None) -> pd.Timestamp:
    """返回最近一个已完成的季度末日。"""
    latest = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof).normalize()
    quarter_end = latest.to_period("Q").end_time.normalize()
    if latest < quarter_end:
        quarter_end = (quarter_end - pd.offsets.QuarterEnd()).normalize()
    return quarter_end


def _recent_quarter_ends(latest_n: int) -> list[str]:
    """返回最近 N 个季度末日列表（YYYYMMDD 字符串）。"""
    quarter_end = _latest_completed_quarter_end()
    dates: list[str] = []
    cur = quarter_end
    for _ in range(max(1, int(latest_n))):
        dates.append(cur.strftime("%Y%m%d"))
        cur = (cur - pd.offsets.QuarterEnd()).normalize()
    return dates


def _quarter_ends_in_range(start_date: str, end_date: str = "") -> list[str]:
    """返回指定日期范围内的所有季度末日列表（YYYYMMDD 字符串）。"""
    start = pd.Timestamp(start_date).normalize()
    end = _latest_completed_quarter_end(end_date or None)
    if start > end:
        return []
    cur = start.to_period("Q").end_time.normalize()
    dates: list[str] = []
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur = (cur + pd.offsets.QuarterEnd()).normalize()
    return dates
