"""A 股基本面数据落库与时点查询（point-in-time）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import akshare as ak
import duckdb
import numpy as np
import pandas as pd

from ..settings import load_config, project_root

_LOG = logging.getLogger(__name__)

# 与 `_ensure_schema` / `INSERT OR REPLACE` 列顺序一致（避免 DuckDB 按位置映射错位）
FUNDAMENTAL_TABLE_COLS: tuple[str, ...] = (
    "symbol",
    "report_period",
    "announcement_date",
    "pe_ttm",
    "pb",
    "ev_ebitda",
    "roe_ttm",
    "net_profit_yoy",
    "gross_margin",
    "gross_margin_change",
    "debt_to_assets",
    "debt_to_assets_change",
    "ocf_to_net_profit",
    "ocf_to_asset",
    "gross_margin_delta",
    "asset_turnover",
    "net_margin_stability",
    "northbound_net_inflow",
    "margin_buy_ratio",
    "source",
    "fetched_at",
)

# F1 盈利质量扩展：与 plan.md 诊断口径一致，需在 fetch 后落库
_FUNDAMENTAL_EXTRA_COLS: tuple[str, ...] = (
    "ocf_to_asset",
    "gross_margin_delta",
    "asset_turnover",
    "net_margin_stability",
)


def _ratio_series_as_decimal(s: pd.Series) -> pd.Series:
    """东财部分比率为百分数（如 15 表示 15%），乘周转率前转为小数。"""
    x = pd.to_numeric(s, errors="coerce")
    med = float(np.nanmedian(np.abs(x.dropna()))) if x.notna().any() else float("nan")
    if np.isfinite(med) and med > 1.0:
        return x / 100.0
    return x


def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if s.isdigit():
        return s.zfill(6)
    m = pd.Series([s]).astype(str).str.extract(r"(\d{6})", expand=False).iloc[0]
    return str(m).zfill(6) if isinstance(m, str) and m else s


def _em_symbol(code6: str) -> str:
    """东财财务指标接口使用的证券代码（如 600519.SH）。"""
    c = _norm_symbol(code6)
    if c.startswith(("60", "68")):
        return f"{c}.SH"
    if c.startswith(("00", "30")):
        return f"{c}.SZ"
    if c.startswith(("43", "83", "87", "82", "92")):
        return f"{c}.BJ"
    return f"{c}.SZ"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(np.float64)


def _pick_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    c = _pick_col(df, candidates)
    if c is None:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[c], errors="coerce")


@dataclass
class FundamentalClientConfig:
    duckdb_path: str = "data/market.duckdb"
    table_name: str = "a_share_fundamental"
    sleep_between_symbols_sec: float = 0.2


class FundamentalClient:
    """管理基本面表的写入与按公告日的 PIT 查询。"""

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
        tbl = table_name or str((cfg.get("database") or {}).get("table_fundamental", "a_share_fundamental"))
        p = Path(db_path)
        if not p.is_absolute():
            p = root / p
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(p))
        self.cfg = FundamentalClientConfig(
            duckdb_path=str(p),
            table_name=tbl,
            sleep_between_symbols_sec=float((cfg.get("akshare") or {}).get("sleep_between_symbols_sec", 0.2)),
        )
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FundamentalClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        t = self.cfg.table_name
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                symbol VARCHAR NOT NULL,
                report_period DATE,
                announcement_date DATE NOT NULL,
                pe_ttm DOUBLE,
                pb DOUBLE,
                ev_ebitda DOUBLE,
                roe_ttm DOUBLE,
                net_profit_yoy DOUBLE,
                gross_margin DOUBLE,
                gross_margin_change DOUBLE,
                debt_to_assets DOUBLE,
                debt_to_assets_change DOUBLE,
                ocf_to_net_profit DOUBLE,
                ocf_to_asset DOUBLE,
                gross_margin_delta DOUBLE,
                asset_turnover DOUBLE,
                net_margin_stability DOUBLE,
                northbound_net_inflow DOUBLE,
                margin_buy_ratio DOUBLE,
                source VARCHAR,
                fetched_at TIMESTAMP,
                PRIMARY KEY (symbol, announcement_date, report_period)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{t}_symbol_announce
            ON {t}(symbol, announcement_date)
            """
        )
        self._migrate_fundamental_extra_columns()

    def _migrate_fundamental_extra_columns(self) -> None:
        t = self.cfg.table_name
        try:
            existing = {
                str(r[1]) for r in self._conn.execute(f"PRAGMA table_info('{t}')").fetchall()
            }
        except Exception:  # noqa: BLE001
            return
        for col in _FUNDAMENTAL_EXTRA_COLS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE {t} ADD COLUMN {col} DOUBLE")

    def _prepare_financial_frame(self, raw: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        out = raw.copy()
        report_col = _pick_col(out, ["报告期", "报告日期", "日期", "截止日期", "statDate"])
        ann_col = _pick_col(out, ["公告日期", "公告日", "最新公告日期", "公告时间", "publishDate"])
        if report_col is None and ann_col is None:
            return pd.DataFrame()
        report_date = pd.to_datetime(out[report_col], errors="coerce") if report_col else pd.NaT
        ann_date = pd.to_datetime(out[ann_col], errors="coerce") if ann_col else report_date
        norm = pd.DataFrame(
            {
                "symbol": _norm_symbol(symbol),
                "report_period": report_date,
                "announcement_date": ann_date,
                "pe_ttm": _to_num(_pick_series(out, ["市盈率TTM", "市盈率(TTM)", "市盈率", "pe_ttm"])),
                "pb": _to_num(_pick_series(out, ["市净率", "市净率MRQ", "pb"])),
                "ev_ebitda": _to_num(_pick_series(out, ["EV/EBITDA", "企业倍数", "ev_ebitda"])),
                "roe_ttm": _to_num(_pick_series(out, ["净资产收益率", "净资产收益率(%)", "ROE", "roe_ttm"])),
                "net_profit_yoy": _to_num(
                    _pick_series(out, ["净利润同比增长率", "净利润同比", "归母净利润同比增长率", "net_profit_yoy"])
                ),
                "gross_margin": _to_num(_pick_series(out, ["销售毛利率", "毛利率", "gross_margin"])),
                "debt_to_assets": _to_num(_pick_series(out, ["资产负债率", "资产负债率(%)", "debt_to_assets"])),
                "ocf_to_net_profit": _to_num(
                    _pick_series(out, ["经营现金流净额/净利润", "经营现金流/净利润", "ocf_to_net_profit"])
                ),
                "northbound_net_inflow": _to_num(
                    _pick_series(out, ["北向资金净流入", "北向净流入", "northbound_net_inflow"])
                ),
                "margin_buy_ratio": _to_num(
                    _pick_series(out, ["融资买入额占成交比", "融资买入占比", "margin_buy_ratio"])
                ),
            }
        )
        norm["announcement_date"] = pd.to_datetime(norm["announcement_date"], errors="coerce").dt.normalize()
        norm["report_period"] = pd.to_datetime(norm["report_period"], errors="coerce").dt.normalize()
        norm = norm.dropna(subset=["announcement_date"]).copy()
        if norm.empty:
            return norm
        norm = norm.sort_values(["report_period", "announcement_date"])
        norm["gross_margin_change"] = norm["gross_margin"].diff()
        norm["debt_to_assets_change"] = norm["debt_to_assets"].diff()
        gm = pd.to_numeric(norm["gross_margin"], errors="coerce")
        norm["gross_margin_delta"] = gm.diff(4)
        norm["ocf_to_asset"] = np.nan
        norm["asset_turnover"] = np.nan
        norm["net_margin_stability"] = np.nan
        norm["source"] = source
        norm["fetched_at"] = pd.Timestamp.now()
        cols = [
            "symbol",
            "report_period",
            "announcement_date",
            "pe_ttm",
            "pb",
            "ev_ebitda",
            "roe_ttm",
            "net_profit_yoy",
            "gross_margin",
            "gross_margin_change",
            "debt_to_assets",
            "debt_to_assets_change",
            "ocf_to_net_profit",
            "ocf_to_asset",
            "gross_margin_delta",
            "asset_turnover",
            "net_margin_stability",
            "northbound_net_inflow",
            "margin_buy_ratio",
            "source",
            "fetched_at",
        ]
        return norm[cols].drop_duplicates(["symbol", "announcement_date", "report_period"], keep="last")

    @staticmethod
    def _load_valuation_history(code6: str) -> pd.DataFrame:
        """日频估值序列（PE-TTM、PB），用于与财报公告日 asof 对齐。"""
        try:
            raw = ak.stock_value_em(symbol=code6)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("拉取日频估值失败 symbol=%s: %s", code6, exc)
            return pd.DataFrame()
        if raw is None or raw.empty:
            return pd.DataFrame()
        dcol = _pick_col(raw, ["数据日期", "日期"])
        if dcol is None:
            return pd.DataFrame()
        pe_c = _pick_col(raw, ["PE(TTM)", "市盈率TTM", "市盈率(TTM)"])
        pb_c = _pick_col(raw, ["市净率", "PB", "pb"])
        pe_ser = _to_num(raw[pe_c]) if pe_c else pd.Series(np.nan, index=raw.index, dtype=np.float64)
        pb_ser = _to_num(raw[pb_c]) if pb_c else pd.Series(np.nan, index=raw.index, dtype=np.float64)
        out = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(raw[dcol], errors="coerce").dt.normalize(),
                "pe_ttm": pe_ser,
                "pb": pb_ser,
            }
        )
        out = out.dropna(subset=["trade_date"])
        if out.empty:
            return out
        return out.sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    @staticmethod
    def _attach_valuation_asof(norm: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
        if norm.empty or val.empty:
            return norm
        left = norm.sort_values("announcement_date").copy()
        left = left.drop(columns=["pe_ttm", "pb"], errors="ignore")
        right = val.sort_values("trade_date")
        merged = pd.merge_asof(
            left,
            right,
            left_on="announcement_date",
            right_on="trade_date",
            direction="backward",
            allow_exact_matches=True,
        )
        return merged.drop(columns=["trade_date"], errors="ignore")

    def _prepare_financial_frame_em(self, raw: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """东财 `stock_financial_analysis_indicator_em` 宽表 → 与 `a_share_fundamental` 对齐的长表。"""
        if raw is None or raw.empty:
            return pd.DataFrame()
        if "NOTICE_DATE" not in raw.columns or "REPORT_DATE" not in raw.columns:
            return pd.DataFrame()
        code = _norm_symbol(symbol)
        sub = raw.copy()
        report_date = pd.to_datetime(sub["REPORT_DATE"], errors="coerce")
        ann_date = pd.to_datetime(sub["NOTICE_DATE"], errors="coerce")
        ocf_col = "NCO_NETPROFIT" if "NCO_NETPROFIT" in sub.columns else None
        norm = pd.DataFrame(
            {
                "symbol": code,
                "report_period": report_date,
                "announcement_date": ann_date,
                "pe_ttm": np.nan,
                "pb": np.nan,
                "ev_ebitda": np.nan,
                "roe_ttm": _to_num(sub["ROEJQ"]) if "ROEJQ" in sub.columns else np.nan,
                "net_profit_yoy": _to_num(sub["PARENTNETPROFITTZ"]) if "PARENTNETPROFITTZ" in sub.columns else np.nan,
                "gross_margin": _to_num(sub["XSMLL"]) if "XSMLL" in sub.columns else np.nan,
                "debt_to_assets": _to_num(sub["ZCFZL"]) if "ZCFZL" in sub.columns else np.nan,
                "ocf_to_net_profit": _to_num(sub[ocf_col]) if ocf_col else np.nan,
                "northbound_net_inflow": np.nan,
                "margin_buy_ratio": np.nan,
                "_xsjll": _to_num(sub["XSJLL"]) if "XSJLL" in sub.columns else np.nan,
                "_zzc": _to_num(sub["ZZCZZTS"]) if "ZZCZZTS" in sub.columns else np.nan,
                "_jyx": _to_num(sub["JYXJLYYSR"]) if "JYXJLYYSR" in sub.columns else np.nan,
            }
        )
        norm["announcement_date"] = pd.to_datetime(norm["announcement_date"], errors="coerce").dt.normalize()
        norm["report_period"] = pd.to_datetime(norm["report_period"], errors="coerce").dt.normalize()
        norm = norm.dropna(subset=["announcement_date"]).copy()
        if norm.empty:
            return norm
        norm = norm.sort_values(["report_period", "announcement_date"])
        norm["gross_margin_change"] = pd.to_numeric(norm["gross_margin"], errors="coerce").diff()
        norm["debt_to_assets_change"] = pd.to_numeric(norm["debt_to_assets"], errors="coerce").diff()
        gm = pd.to_numeric(norm["gross_margin"], errors="coerce")
        norm["gross_margin_delta"] = gm.diff(4)
        xsjll = pd.to_numeric(norm["_xsjll"], errors="coerce")
        norm["net_margin_stability"] = -xsjll.rolling(4, min_periods=4).std()
        zzc = pd.to_numeric(norm["_zzc"], errors="coerce")
        j_dec = _ratio_series_as_decimal(norm["_jyx"])
        norm["asset_turnover"] = zzc
        norm["ocf_to_asset"] = j_dec * zzc
        norm = norm.drop(columns=["_xsjll", "_zzc", "_jyx"])
        norm["source"] = source
        norm["fetched_at"] = pd.Timestamp.now()
        cols = [
            "symbol",
            "report_period",
            "announcement_date",
            "pe_ttm",
            "pb",
            "ev_ebitda",
            "roe_ttm",
            "net_profit_yoy",
            "gross_margin",
            "gross_margin_change",
            "debt_to_assets",
            "debt_to_assets_change",
            "ocf_to_net_profit",
            "ocf_to_asset",
            "gross_margin_delta",
            "asset_turnover",
            "net_margin_stability",
            "northbound_net_inflow",
            "margin_buy_ratio",
            "source",
            "fetched_at",
        ]
        return norm[cols].drop_duplicates(["symbol", "announcement_date", "report_period"], keep="last")

    @staticmethod
    def _prepare_daily_valuation_only(val: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """无季报数据时，退化为日频估值落库（保证 PIT 下 PE/PB 可用）。"""
        if val is None or val.empty:
            return pd.DataFrame()
        code = _norm_symbol(symbol)
        out = pd.DataFrame(
            {
                "symbol": code,
                "report_period": val["trade_date"].values,
                "announcement_date": val["trade_date"].values,
                "pe_ttm": pd.to_numeric(val["pe_ttm"], errors="coerce"),
                "pb": pd.to_numeric(val["pb"], errors="coerce"),
                "ev_ebitda": np.nan,
                "roe_ttm": np.nan,
                "net_profit_yoy": np.nan,
                "gross_margin": np.nan,
                "gross_margin_change": np.nan,
                "debt_to_assets": np.nan,
                "debt_to_assets_change": np.nan,
                "ocf_to_net_profit": np.nan,
                "ocf_to_asset": np.nan,
                "gross_margin_delta": np.nan,
                "asset_turnover": np.nan,
                "net_margin_stability": np.nan,
                "northbound_net_inflow": np.nan,
                "margin_buy_ratio": np.nan,
                "source": source,
                "fetched_at": pd.Timestamp.now(),
            }
        )
        return out.drop_duplicates(["symbol", "announcement_date", "report_period"], keep="last")

    def fetch_symbol_fundamentals(self, symbol: str) -> pd.DataFrame:
        code = _norm_symbol(symbol)
        val_hist = self._load_valuation_history(code)

        frames: list[pd.DataFrame] = []

        try:
            em_raw = ak.stock_financial_analysis_indicator_em(symbol=_em_symbol(code))
            norm_em = self._prepare_financial_frame_em(em_raw, code, source="stock_financial_analysis_indicator_em")
            if not norm_em.empty and not val_hist.empty:
                norm_em = self._attach_valuation_asof(norm_em, val_hist)
            if not norm_em.empty:
                frames.append(norm_em)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("东财财务指标失败 symbol=%s: %s", code, exc)

        sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
            ("stock_financial_analysis_indicator", lambda: ak.stock_financial_analysis_indicator(symbol=code)),
            ("stock_financial_abstract", lambda: ak.stock_financial_abstract(symbol=code)),
        ]
        for name, fn in sources:
            try:
                raw = fn()
                norm = self._prepare_financial_frame(raw, code, source=name)
                if not norm.empty:
                    frames.append(norm)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("拉取基本面失败 symbol=%s source=%s: %s", code, name, exc)

        if not frames and not val_hist.empty:
            frames.append(self._prepare_daily_valuation_only(val_hist, code, source="stock_value_em"))

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["announcement_date", "report_period"]).drop_duplicates(
            ["symbol", "announcement_date", "report_period"],
            keep="last",
        )
        return out.reset_index(drop=True)

    def upsert(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        aligned = df.copy()
        for c in FUNDAMENTAL_TABLE_COLS:
            if c not in aligned.columns:
                aligned[c] = pd.NA
        aligned = aligned[list(FUNDAMENTAL_TABLE_COLS)]
        aligned["source"] = aligned["source"].astype(object).where(aligned["source"].notna(), "")
        self._conn.register("fund_in", aligned)
        try:
            cols_sql = ", ".join(FUNDAMENTAL_TABLE_COLS)
            self._conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.cfg.table_name} ({cols_sql})
                SELECT {cols_sql} FROM fund_in
                """
            )
        finally:
            self._conn.unregister("fund_in")
        return int(len(aligned))

    def update_symbols(self, symbols: Iterable[str]) -> int:
        total = 0
        for i, sym in enumerate(symbols):
            df = self.fetch_symbol_fundamentals(sym)
            total += self.upsert(df)
            if i >= 0 and self.cfg.sleep_between_symbols_sec > 0:
                time.sleep(self.cfg.sleep_between_symbols_sec)
        return total

    def load_point_in_time(
        self,
        *,
        asof_date: Union[str, date, pd.Timestamp],
        symbols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        t = self.cfg.table_name
        asof = pd.Timestamp(asof_date).date()
        conds = ["announcement_date <= ?"]
        params: list[object] = [asof]
        if symbols:
            sym_list = [str(_norm_symbol(s)) for s in symbols]
            if sym_list:
                conds.append(f"symbol IN ({','.join(['?'] * len(sym_list))})")
                params.extend(sym_list)
        where = " AND ".join(conds)
        q = f"""
            WITH cte AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY announcement_date DESC, report_period DESC NULLS LAST
                    ) AS rn
                FROM {t}
                WHERE {where}
            )
            SELECT * FROM cte
            WHERE rn = 1
        """
        out = self._conn.execute(q, params).df()
        out = out.drop(columns=["rn"], errors="ignore")
        if out.empty:
            return out
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        out["report_period"] = pd.to_datetime(out["report_period"], errors="coerce").dt.normalize()
        out["announcement_date"] = pd.to_datetime(out["announcement_date"], errors="coerce").dt.normalize()
        return out
