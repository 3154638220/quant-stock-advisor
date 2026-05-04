"""fetch_fundamental 核心逻辑，从 scripts/fetch_fundamental.py 迁入。"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from src.data_fetcher import list_default_universe_symbols


def run_disclosure_calendar(
    *,
    db_path: Path,
    symbols: list[str] | None,
    max_symbols: int | None,
    config_path: Path | None,
    log,
) -> int:
    """P2-3: 从 akshare 拉取财报实际披露日期并写入 DuckDB。"""
    try:
        import akshare as ak
    except ModuleNotFoundError:
        log.error("缺少 akshare 依赖，无法拉取披露日历。")
        return 1

    if symbols:
        syms = [s.strip().zfill(6) for s in symbols]
    else:
        syms = list_default_universe_symbols(max_symbols=max_symbols, config_path=config_path)

    if not syms:
        log.error("无可用代码，退出。")
        return 1

    con = duckdb.connect(str(db_path))
    try:
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
        for symbol in syms:
            try:
                em_sym = _build_eastmoney_symbol(symbol)
                df = ak.stock_financial_report_disclosure_date_cninfo(symbol=em_sym)
                if df is None or df.empty:
                    log.debug("披露日历为空: %s", symbol)
                    continue

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
            time.sleep(0.15)

        log.info("披露日历更新完成：symbols=%d, upsert_rows=%d", len(syms), total_inserted)
    finally:
        con.close()
    return 0


def _build_eastmoney_symbol(symbol: str) -> str:
    s = symbol.zfill(6)
    if s.startswith(("60", "68")):
        return f"{s}.SH"
    if s.startswith(("00", "30")):
        return f"{s}.SZ"
    return f"{s}.BJ"


def _norm_symbol(symbol: str) -> str:
    import re
    s = str(symbol).strip()
    if s.isdigit():
        return s.zfill(6)
    m = re.search(r"(\d{6})", s)
    return m.group(1).zfill(6) if m else s
