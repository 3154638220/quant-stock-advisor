"""A 股个股资金流数据落库与查询（point-in-time）。"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover
    def _missing_stock_individual_fund_flow(*args, **kwargs):
        raise ModuleNotFoundError("缺少 akshare 依赖，无法使用 AkShare 资金流回退接口。")

    ak = SimpleNamespace(stock_individual_fund_flow=_missing_stock_individual_fund_flow)

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover
    duckdb = None

from ..settings import load_config, project_root

_LOG = logging.getLogger(__name__)
_H5_FUND_FLOW_HISTORY_URL = "https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData"


def _require_akshare():
    return ak


def _require_duckdb():
    if duckdb is None:
        raise ModuleNotFoundError("缺少 duckdb 依赖，无法读写资金流本地库。")
    return duckdb

FUND_FLOW_TABLE_COLS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "close",
    "pct_chg",
    "main_net_inflow",
    "main_net_inflow_pct",
    "super_large_net_inflow",
    "super_large_net_inflow_pct",
    "large_net_inflow",
    "large_net_inflow_pct",
    "medium_net_inflow",
    "medium_net_inflow_pct",
    "small_net_inflow",
    "small_net_inflow_pct",
    "source",
    "fetched_at",
)

_FUND_FLOW_IMPORT_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "code", "ticker", "ts_code", "股票代码", "证券代码", "代码"),
    "trade_date": ("trade_date", "date", "日期", "交易日期"),
    "close": ("close", "close_price", "收盘价"),
    "pct_chg": ("pct_chg", "pct_change", "change_pct", "涨跌幅"),
    "main_net_inflow": ("main_net_inflow", "主力净流入-净额", "主力净流入"),
    "main_net_inflow_pct": ("main_net_inflow_pct", "主力净流入-净占比", "主力净流入占比"),
    "super_large_net_inflow": ("super_large_net_inflow", "超大单净流入-净额", "超大单净流入"),
    "super_large_net_inflow_pct": (
        "super_large_net_inflow_pct",
        "超大单净流入-净占比",
        "超大单净流入占比",
    ),
    "large_net_inflow": ("large_net_inflow", "大单净流入-净额", "大单净流入"),
    "large_net_inflow_pct": ("large_net_inflow_pct", "大单净流入-净占比", "大单净流入占比"),
    "medium_net_inflow": ("medium_net_inflow", "中单净流入-净额", "中单净流入"),
    "medium_net_inflow_pct": ("medium_net_inflow_pct", "中单净流入-净占比", "中单净流入占比"),
    "small_net_inflow": ("small_net_inflow", "小单净流入-净额", "小单净流入"),
    "small_net_inflow_pct": ("small_net_inflow_pct", "小单净流入-净占比", "小单净流入占比"),
    "source": ("source", "数据源", "来源"),
    "fetched_at": ("fetched_at", "抓取时间", "更新时间"),
}


def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if s.isdigit():
        return s.zfill(6)
    m = pd.Series([s]).astype(str).str.extract(r"(\d{6})", expand=False).iloc[0]
    return str(m).zfill(6) if isinstance(m, str) and m else s


def _em_market(code6: str) -> str:
    c = _norm_symbol(code6)
    if c.startswith(("60", "68")):
        return "sh"
    if c.startswith(("00", "30")):
        return "sz"
    if c.startswith(("43", "83", "87", "82", "92")):
        return "bj"
    return "sz"


def _resolve_alias_columns(columns: Iterable[str]) -> dict[str, str]:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in _FUND_FLOW_IMPORT_ALIASES.items():
        for alias in aliases:
            hit = lookup.get(str(alias).strip().lower())
            if hit is not None:
                resolved[canonical] = hit
                break
    return resolved


class FundFlowClient:
    """管理个股资金流数据的写入与按日期查询。"""

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
        tbl = table_name or str((cfg.get("database") or {}).get("table_fund_flow", "a_share_fund_flow"))
        self._daily_table_name = str((cfg.get("database") or {}).get("table_daily", "a_share_daily"))
        p = Path(db_path)
        if not p.is_absolute():
            p = root / p
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = _require_duckdb().connect(str(p))
        self._table_name = tbl
        self._last_fetch_errors: dict[str, str] = {}
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FundFlowClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        t = self._table_name
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                symbol VARCHAR NOT NULL,
                trade_date DATE NOT NULL,
                close DOUBLE,
                pct_chg DOUBLE,
                main_net_inflow DOUBLE,
                main_net_inflow_pct DOUBLE,
                super_large_net_inflow DOUBLE,
                super_large_net_inflow_pct DOUBLE,
                large_net_inflow DOUBLE,
                large_net_inflow_pct DOUBLE,
                medium_net_inflow DOUBLE,
                medium_net_inflow_pct DOUBLE,
                small_net_inflow DOUBLE,
                small_net_inflow_pct DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                PRIMARY KEY (symbol, trade_date)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{t}_symbol_date
            ON {t}(symbol, trade_date)
            """
        )

    def fetch_symbol_fund_flow(
        self,
        symbol: str,
        *,
        timeout_sec: float = 15.0,
    ) -> pd.DataFrame:
        h5_df = self._fetch_symbol_fund_flow_h5(symbol, timeout_sec=timeout_sec)
        if not h5_df.empty:
            self._last_fetch_errors.pop(_norm_symbol(symbol), None)
            return h5_df

        code = _norm_symbol(symbol)
        market = _em_market(code)
        try:
            raw = _require_akshare().stock_individual_fund_flow(stock=code, market=market)
        except Exception as exc:
            self._last_fetch_errors[code] = f"{type(exc).__name__}: {exc}"
            _LOG.warning("资金流拉取失败 symbol=%s market=%s: %s", code, market, self._last_fetch_errors[code])
            return pd.DataFrame()

        if raw is None or raw.empty:
            self._last_fetch_errors[code] = "empty_response"
            return pd.DataFrame()

        col_map = {
            "日期": "trade_date",
            "收盘价": "close",
            "涨跌幅": "pct_chg",
            "主力净流入-净额": "main_net_inflow",
            "主力净流入-净占比": "main_net_inflow_pct",
            "超大单净流入-净额": "super_large_net_inflow",
            "超大单净流入-净占比": "super_large_net_inflow_pct",
            "大单净流入-净额": "large_net_inflow",
            "大单净流入-净占比": "large_net_inflow_pct",
            "中单净流入-净额": "medium_net_inflow",
            "中单净流入-净占比": "medium_net_inflow_pct",
            "小单净流入-净额": "small_net_inflow",
            "小单净流入-净占比": "small_net_inflow_pct",
        }
        df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        if "trade_date" not in df.columns:
            self._last_fetch_errors[code] = f"missing_trade_date_column: {sorted(df.columns.tolist())[:8]}"
            return pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        for c in [
            "close", "pct_chg", "main_net_inflow", "main_net_inflow_pct",
            "super_large_net_inflow", "super_large_net_inflow_pct",
            "large_net_inflow", "large_net_inflow_pct",
            "medium_net_inflow", "medium_net_inflow_pct",
            "small_net_inflow", "small_net_inflow_pct",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan

        df["symbol"] = code
        df["source"] = "stock_individual_fund_flow"
        df["fetched_at"] = pd.Timestamp.now()

        out = df[list(FUND_FLOW_TABLE_COLS)].drop_duplicates(["symbol", "trade_date"], keep="last")
        self._last_fetch_errors.pop(code, None)
        return out.sort_values("trade_date").reset_index(drop=True)

    def _fetch_symbol_fund_flow_h5(
        self,
        symbol: str,
        *,
        timeout_sec: float = 15.0,
    ) -> pd.DataFrame:
        code = _norm_symbol(symbol)
        market = _em_market(code)
        secid = f"{1 if market == 'sh' else 0}.{code}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://emdatah5.eastmoney.com/dc/zjlx/stock?fc={secid}&fn={code}",
            "Accept": "application/json,text/plain,*/*",
        }
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3",
            "fields2": "f51,f52,f53,f54,f55,f56,f62,f63",
            "ut": "",
        }
        try:
            resp = requests.get(_H5_FUND_FLOW_HISTORY_URL, params=params, headers=headers, timeout=timeout_sec)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            self._last_fetch_errors[code] = f"h5_{type(exc).__name__}: {exc}"
            _LOG.warning("H5 资金流拉取失败 symbol=%s secid=%s: %s", code, secid, self._last_fetch_errors[code])
            return pd.DataFrame()

        klines = (((payload or {}).get("data") or {}).get("klines")) or []
        if not klines:
            self._last_fetch_errors[code] = f"h5_empty_response_rc={payload.get('rc') if isinstance(payload, dict) else 'na'}"
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for item in klines:
            parts = str(item).split(",")
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "symbol": code,
                    "trade_date": pd.to_datetime(parts[0], errors="coerce"),
                    "main_net_inflow": pd.to_numeric(parts[1], errors="coerce"),
                    "small_net_inflow": pd.to_numeric(parts[2], errors="coerce"),
                    "medium_net_inflow": pd.to_numeric(parts[3], errors="coerce"),
                    "large_net_inflow": pd.to_numeric(parts[4], errors="coerce"),
                    "super_large_net_inflow": pd.to_numeric(parts[5], errors="coerce"),
                    "close": pd.to_numeric(parts[6], errors="coerce"),
                    "pct_chg": pd.to_numeric(parts[7], errors="coerce"),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            self._last_fetch_errors[code] = "h5_parse_empty"
            return pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["trade_date"]).copy()
        if df.empty:
            self._last_fetch_errors[code] = "h5_missing_trade_date"
            return pd.DataFrame()

        amount_map = self._load_daily_amount_map(
            code,
            start_date=df["trade_date"].min(),
            end_date=df["trade_date"].max(),
        )
        df["amount"] = df["trade_date"].map(amount_map)
        denom = pd.to_numeric(df["amount"], errors="coerce").replace(0, np.nan)
        df["main_net_inflow_pct"] = df["main_net_inflow"] / denom * 100.0
        df["super_large_net_inflow_pct"] = df["super_large_net_inflow"] / denom * 100.0
        df["large_net_inflow_pct"] = df["large_net_inflow"] / denom * 100.0
        df["medium_net_inflow_pct"] = df["medium_net_inflow"] / denom * 100.0
        df["small_net_inflow_pct"] = df["small_net_inflow"] / denom * 100.0
        df["source"] = "emdatah5_zjlx_history"
        df["fetched_at"] = pd.Timestamp.now()

        out = pd.DataFrame(
            {
                "symbol": df["symbol"],
                "trade_date": df["trade_date"],
                "close": df["close"],
                "pct_chg": df["pct_chg"],
                "main_net_inflow": df["main_net_inflow"],
                "main_net_inflow_pct": df["main_net_inflow_pct"],
                "super_large_net_inflow": df["super_large_net_inflow"],
                "super_large_net_inflow_pct": df["super_large_net_inflow_pct"],
                "large_net_inflow": df["large_net_inflow"],
                "large_net_inflow_pct": df["large_net_inflow_pct"],
                "medium_net_inflow": df["medium_net_inflow"],
                "medium_net_inflow_pct": df["medium_net_inflow_pct"],
                "small_net_inflow": df["small_net_inflow"],
                "small_net_inflow_pct": df["small_net_inflow_pct"],
                "source": df["source"],
                "fetched_at": df["fetched_at"],
            }
        )
        return out[list(FUND_FLOW_TABLE_COLS)].drop_duplicates(["symbol", "trade_date"], keep="last")

    def _load_daily_amount_map(
        self,
        symbol: str,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> dict[pd.Timestamp, float]:
        exists = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [self._daily_table_name],
        ).fetchone()
        if not exists or int(exists[0]) <= 0:
            _LOG.warning("缺少日线表 %s，fund_flow 百分比列将保留为空", self._daily_table_name)
            return {}
        sql = (
            f"SELECT trade_date, amount FROM {self._daily_table_name} "
            "WHERE symbol = ? AND trade_date >= ? AND trade_date <= ?"
        )
        rows = self._conn.execute(
            sql,
            [str(_norm_symbol(symbol)), pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()],
        ).fetchall()
        return {
            pd.Timestamp(trade_date).normalize(): float(amount)
            for trade_date, amount in rows
            if trade_date is not None and amount is not None and np.isfinite(float(amount))
        }

    def normalize_import_frame(
        self,
        raw: pd.DataFrame,
        *,
        source_label: str = "external_fund_flow_file",
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=list(FUND_FLOW_TABLE_COLS))

        resolved = _resolve_alias_columns(raw.columns)
        if "symbol" not in resolved or "trade_date" not in resolved:
            raise ValueError(
                "资金流导入至少需要 symbol/code 与 trade_date/date 列；"
                f"当前列: {list(raw.columns)[:12]}"
            )

        out = pd.DataFrame()
        for canonical in FUND_FLOW_TABLE_COLS:
            src = resolved.get(canonical)
            if src is None:
                out[canonical] = pd.NA
            else:
                out[canonical] = raw[src]

        out["symbol"] = out["symbol"].map(_norm_symbol)
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["symbol", "trade_date"]).copy()
        if out.empty:
            return pd.DataFrame(columns=list(FUND_FLOW_TABLE_COLS))

        for col in FUND_FLOW_TABLE_COLS:
            if col in {"symbol", "trade_date", "source", "fetched_at"}:
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce")

        if out["source"].isna().all():
            out["source"] = str(source_label)
        else:
            out["source"] = out["source"].astype(str).where(out["source"].notna(), str(source_label))
        if out["fetched_at"].isna().all():
            out["fetched_at"] = pd.Timestamp.now()
        else:
            out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
            out["fetched_at"] = out["fetched_at"].where(out["fetched_at"].notna(), pd.Timestamp.now())

        out = out[list(FUND_FLOW_TABLE_COLS)].drop_duplicates(["symbol", "trade_date"], keep="last")
        return out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def import_file(
        self,
        path: Union[str, Path],
        *,
        source_label: Optional[str] = None,
    ) -> int:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"资金流导入文件不存在: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(file_path)
        elif suffix in {".parquet", ".pq"}:
            raw = pd.read_parquet(file_path)
        else:
            raise ValueError(f"暂不支持的资金流导入格式: {suffix}（仅支持 .csv/.parquet）")

        normalized = self.normalize_import_frame(
            raw,
            source_label=str(source_label or f"external:{file_path.name}"),
        )
        return self.upsert(normalized)

    def upsert(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        aligned = df.copy()
        for c in FUND_FLOW_TABLE_COLS:
            if c not in aligned.columns:
                aligned[c] = pd.NA
        aligned = aligned[list(FUND_FLOW_TABLE_COLS)]
        aligned["source"] = aligned["source"].astype(object).where(aligned["source"].notna(), "")
        self._conn.register("ff_in", aligned)
        try:
            cols_sql = ", ".join(FUND_FLOW_TABLE_COLS)
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self._table_name} ({cols_sql})
                SELECT {cols_sql} FROM ff_in
                """
            )
        finally:
            self._conn.unregister("ff_in")
        return int(len(aligned))

    def update_symbols(
        self,
        symbols: Iterable[str],
        *,
        sleep_sec: float = 0.5,
        log_every: int = 50,
    ) -> int:
        total = 0
        success = 0
        fail = 0
        failed_examples: list[str] = []
        for i, sym in enumerate(symbols):
            df = self.fetch_symbol_fund_flow(sym)
            if not df.empty:
                total += self.upsert(df)
                success += 1
                if success % log_every == 0:
                    _LOG.info("资金流已处理 %d 只（成功 %d，失败 %d）", i + 1, success, fail)
            else:
                fail += 1
                code = _norm_symbol(sym)
                reason = self._last_fetch_errors.get(code, "empty_response")
                if len(failed_examples) < 5:
                    failed_examples.append(f"{code}: {reason}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        if failed_examples:
            _LOG.warning("资金流失败样例: %s", " | ".join(failed_examples))
        _LOG.info("资金流完成：成功 %d，失败 %d，总行数 %d", success, fail, total)
        return total

    def table_row_count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()
        return int(row[0]) if row else 0

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
        sql = f"SELECT * FROM {t} WHERE trade_date >= ? AND trade_date <= ?"
        params: list[object] = [s, e]
        if symbols:
            sym_list = list(symbols)
            if sym_list:
                placeholders = ", ".join(["?"] * len(sym_list))
                sql += f" AND symbol IN ({placeholders})"
                params.extend(sym_list)
        sql += " ORDER BY symbol, trade_date"
        return self._conn.execute(sql, params).fetchdf()
