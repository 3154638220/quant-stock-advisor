"""A 股股东人数数据落库与近似 PIT 查询。"""

from __future__ import annotations

import logging
import time
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
    duckdb = None

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

    def fetch_end_date_shareholder(self, end_date: str) -> pd.DataFrame:
        key = str(end_date).replace("-", "")
        try:
            raw = _require_akshare().stock_zh_a_gdhs(symbol=key)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("股东人数拉取失败 end_date=%s: %s", key, exc)
            return pd.DataFrame()
        return self._prepare_shareholder_frame(raw)

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

    def update_end_dates(
        self,
        end_dates: Iterable[str],
        *,
        sleep_sec: float = 0.5,
        log_every: int = 1,
    ) -> int:
        total = 0
        success = 0
        fail = 0
        for i, end_date in enumerate(end_dates):
            df = self.fetch_end_date_shareholder(end_date)
            if not df.empty:
                total += self.upsert(df)
                success += 1
                if success % log_every == 0:
                    _LOG.info("股东人数已处理 %d 个截止日（成功 %d，失败 %d）", i + 1, success, fail)
            else:
                fail += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        _LOG.info("股东人数完成：成功 %d，失败 %d，总行数 %d", success, fail, total)
        return total

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
