"""数据质量检查：日线表与新数据家族（资金流 / 股东人数）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd


@dataclass
class QualityConfig:
    """与 config.yaml 中 ``quality`` 段对齐。"""

    max_calendar_gap_days: int = 20
    null_ratio_max: float = 0.05
    ohlc_cols: tuple[str, ...] = ("open", "high", "low", "close")
    # 若为 False：仅统计，不把 OHLC/长假间隔判为「未通过」（常见于源站、停牌、长假）
    fail_on_ohlc_invalid: bool = True
    fail_on_large_gaps: bool = True

    @classmethod
    def from_mapping(cls, m: Optional[Dict[str, Any]]) -> "QualityConfig":
        if not m:
            return cls()
        return cls(
            max_calendar_gap_days=int(m.get("max_calendar_gap_days", 20)),
            null_ratio_max=float(m.get("null_ratio_max", 0.05)),
            ohlc_cols=tuple(
                m.get("ohlc_cols", ["open", "high", "low", "close"]),
            ),
            fail_on_ohlc_invalid=bool(m.get("fail_on_ohlc_invalid", True)),
            fail_on_large_gaps=bool(m.get("fail_on_large_gaps", True)),
        )


@dataclass
class QualityReport:
    ok: bool
    duplicate_pk_rows: int = 0
    ohlc_invalid_rows: int = 0
    large_gap_rows: int = 0
    null_ratio_violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"duplicate_pk={self.duplicate_pk_rows}",
            f"ohlc_invalid={self.ohlc_invalid_rows}",
            f"large_gaps={self.large_gap_rows}",
            f"null_violations={len(self.null_ratio_violations)}",
        ]
        return "; ".join(parts)


@dataclass
class FundFlowQualityReport:
    ok: bool
    table_exists: bool
    total_rows: int = 0
    distinct_symbols: int = 0
    min_trade_date: Optional[str] = None
    max_trade_date: Optional[str] = None
    duplicate_pk_rows: int = 0
    rows_without_daily_match: int = 0
    rows_after_daily_max_date: int = 0
    rows_without_daily_match_within_daily_span: int = 0
    rows_without_daily_match_absent_symbols: int = 0
    rows_without_daily_match_known_symbols: int = 0
    absent_symbol_count: int = 0
    daily_max_trade_date: Optional[str] = None
    all_zero_flow_rows: int = 0
    null_ratio_by_col: Dict[str, float] = field(default_factory=dict)
    coverage_ratio_vs_daily: Optional[float] = None
    median_symbols_per_day: float = 0.0
    p10_symbols_per_day: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShareholderQualityReport:
    ok: bool
    table_exists: bool
    total_rows: int = 0
    distinct_symbols: int = 0
    distinct_end_dates: int = 0
    min_end_date: Optional[str] = None
    max_end_date: Optional[str] = None
    duplicate_pk_rows: int = 0
    notice_date_coverage_ratio: float = 0.0
    fallback_lag_usage_ratio: float = 0.0
    negative_notice_lag_rows: int = 0
    median_notice_lag_days: Optional[float] = None
    p90_notice_lag_days: Optional[float] = None
    median_symbols_per_end_date: float = 0.0
    p10_symbols_per_end_date: float = 0.0
    effective_factor_dates_ge_min_width: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _duplicate_pk_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    q = f"""
    SELECT COALESCE(SUM(cnt - 1), 0)::BIGINT AS dup_extra
    FROM (
      SELECT symbol, trade_date, COUNT(*) AS cnt
      FROM {table}
      GROUP BY 1, 2
      HAVING COUNT(*) > 1
    ) t
    """
    row = conn.execute(q).fetchone()
    return int(row[0] or 0)


def _ohlc_invalid_count(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    cols: tuple[str, ...],
) -> int:
    """违反 high/low 与 open/close 关系的行数（四价均非空时检查）。"""
    o, h, lo, c = cols[0], cols[1], cols[2], cols[3]
    q = f"""
    SELECT COUNT(*)::BIGINT
    FROM {table}
    WHERE {o} IS NOT NULL AND {h} IS NOT NULL AND {lo} IS NOT NULL AND {c} IS NOT NULL
      AND (
        {h} < GREATEST({o}, {c})
        OR {lo} > LEAST({o}, {c})
        OR {h} < {lo}
        OR {o} <= 0 OR {h} <= 0 OR {lo} <= 0 OR {c} <= 0
      )
    """
    row = conn.execute(q).fetchone()
    return int(row[0] or 0)


def _large_gap_count(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    max_gap_days: int,
) -> int:
    """相邻交易日历间隔超过阈值的条数（按 symbol 分区）。"""
    q = f"""
    WITH w AS (
      SELECT
        symbol,
        trade_date,
        lag(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_d
      FROM {table}
    )
    SELECT COUNT(*)::BIGINT
    FROM w
    WHERE prev_d IS NOT NULL
      AND date_diff('day', prev_d, trade_date) > {int(max_gap_days)}
    """
    row = conn.execute(q).fetchone()
    return int(row[0] or 0)


def _describe_column_names(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = conn.execute(f"DESCRIBE {table}").fetchall()
    return {str(r[0]) for r in rows}


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and int(row[0] or 0) > 0)


def _null_ratio_checks(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    threshold: float,
    columns: List[str],
) -> List[str]:
    violations: List[str] = []
    total = conn.execute(f"SELECT COUNT(*)::BIGINT FROM {table}").fetchone()
    n = int(total[0] or 0)
    if n == 0:
        return violations
    existing = _describe_column_names(conn, table)
    for col in columns:
        if col not in existing:
            continue
        row = conn.execute(
            f"""
            SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)::BIGINT
            FROM {table}
            """,
        ).fetchone()
        nulls = int(row[0] or 0)
        ratio = nulls / n
        if ratio > threshold:
            violations.append(f"{col}: {ratio:.4f} > {threshold}")
    return violations


def run_quality_checks(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    cfg: Optional[QualityConfig] = None,
) -> QualityReport:
    """
    对日线表执行质量检查（只读）。

    - 主键重复：``(symbol, trade_date)`` 出现次数 > 1 的额外行数之和
    - OHLC：四价齐全时校验 high/low 与开收关系及正数
    - 连续性：同 symbol 相邻交易日历间隔 > ``max_calendar_gap_days``
    - 空值：各列空值比例超过 ``null_ratio_max`` 则记入 violations
    """
    qc = cfg or QualityConfig()
    notes: List[str] = []

    dup = _duplicate_pk_count(conn, table)
    ohlc_n = _ohlc_invalid_count(conn, table, qc.ohlc_cols)
    gaps = _large_gap_count(conn, table, qc.max_calendar_gap_days)

    price_cols = list(qc.ohlc_cols) + [
        c
        for c in ("volume", "amount", "amplitude_pct", "pct_chg", "change", "turnover")
        if c not in qc.ohlc_cols
    ]
    null_v = _null_ratio_checks(conn, table, qc.null_ratio_max, price_cols)

    ohlc_fail = qc.fail_on_ohlc_invalid and ohlc_n > 0
    gap_fail = qc.fail_on_large_gaps and gaps > 0
    ok = dup == 0 and not ohlc_fail and not gap_fail and len(null_v) == 0
    if not ok:
        if dup:
            notes.append("存在重复主键，请修复后再依赖该表。")
        if ohlc_fail:
            notes.append("存在 OHLC 逻辑非法或非正价格行。")
        if gap_fail:
            notes.append(
                f"存在相邻交易日历间隔 > {qc.max_calendar_gap_days} 天的记录（长假或缺数）。",
            )
        if null_v:
            notes.append("部分列空值比例超过阈值：" + "; ".join(null_v))

    return QualityReport(
        ok=ok,
        duplicate_pk_rows=dup,
        ohlc_invalid_rows=ohlc_n,
        large_gap_rows=gaps,
        null_ratio_violations=null_v,
        notes=notes,
    )


def validate_daily_frame(df: pd.DataFrame, *, cfg: Optional[QualityConfig] = None) -> QualityReport:
    """
    对尚未落库的 DataFrame 做与库内一致的 OHLC / 主键检查（用于写入前校验）。
    不含「大间隔」检查（需全历史排序上下文）。
    """
    qc = cfg or QualityConfig()
    notes: List[str] = []
    if df.empty:
        return QualityReport(ok=True, notes=["empty frame"])

    dup = 0
    if "symbol" in df.columns and "trade_date" in df.columns:
        dup = len(df) - len(df.drop_duplicates(subset=["symbol", "trade_date"]))

    o, h, lo, c = qc.ohlc_cols[0], qc.ohlc_cols[1], qc.ohlc_cols[2], qc.ohlc_cols[3]
    sub = df[[o, h, lo, c]].dropna(how="any")
    ohlc_n = 0
    if not sub.empty:
        bad = (
            (sub[h] < sub[[o, c]].max(axis=1))
            | (sub[lo] > sub[[o, c]].min(axis=1))
            | (sub[h] < sub[lo])
            | (sub[o] <= 0)
            | (sub[h] <= 0)
            | (sub[lo] <= 0)
            | (sub[c] <= 0)
        )
        ohlc_n = int(bad.sum())

    null_v: List[str] = []
    price_cols = list(qc.ohlc_cols) + [
        x
        for x in ("volume", "amount", "amplitude_pct", "pct_chg", "change", "turnover")
        if x in df.columns and x not in qc.ohlc_cols
    ]
    for col in price_cols:
        if col not in df.columns:
            continue
        ratio = float(df[col].isna().mean())
        if ratio > qc.null_ratio_max:
            null_v.append(f"{col}: {ratio:.4f} > {qc.null_ratio_max}")

    ohlc_fail = qc.fail_on_ohlc_invalid and ohlc_n > 0
    ok = dup == 0 and not ohlc_fail and len(null_v) == 0
    if not ok:
        if dup:
            notes.append("DataFrame 内存在重复 (symbol, trade_date)。")
        if ohlc_fail:
            notes.append("存在 OHLC 非法行。")
        if null_v:
            notes.extend(null_v)

    return QualityReport(
        ok=ok,
        duplicate_pk_rows=dup,
        ohlc_invalid_rows=ohlc_n,
        large_gap_rows=0,
        null_ratio_violations=null_v,
        notes=notes,
    )


def run_fund_flow_quality_checks(
    conn: duckdb.DuckDBPyConnection,
    *,
    table: str = "a_share_fund_flow",
    daily_table: str = "a_share_daily",
) -> FundFlowQualityReport:
    """
    资金流数据质量检查。

    关注项：
    - 表覆盖率与截面宽度
    - 关键列空值率
    - 与日线表的时间戳错位
    - 重复主键与可疑全零行
    """
    if not _table_exists(conn, table):
        return FundFlowQualityReport(ok=False, table_exists=False, notes=[f"表不存在: {table}"])

    raw = conn.execute(
        f"""
        SELECT
            symbol,
            trade_date,
            main_net_inflow_pct,
            super_large_net_inflow_pct,
            small_net_inflow_pct
        FROM {table}
        """
    ).df()
    if raw.empty:
        return FundFlowQualityReport(ok=False, table_exists=True, notes=[f"表为空: {table}"])

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    for col in ("main_net_inflow_pct", "super_large_net_inflow_pct", "small_net_inflow_pct"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    dup = int(len(raw) - len(raw.drop_duplicates(["symbol", "trade_date"])))
    dedup = raw.drop_duplicates(["symbol", "trade_date"], keep="last").copy()
    per_day = dedup.groupby("trade_date", dropna=True)["symbol"].nunique()
    null_ratio_by_col = {
        col: float(dedup[col].isna().mean())
        for col in ("main_net_inflow_pct", "super_large_net_inflow_pct", "small_net_inflow_pct")
    }
    all_zero_rows = int(
        (
            dedup[["main_net_inflow_pct", "super_large_net_inflow_pct", "small_net_inflow_pct"]]
            .fillna(np.nan)
            .eq(0.0)
            .all(axis=1)
        ).sum()
    )

    rows_without_daily_match = 0
    rows_after_daily_max_date = 0
    rows_without_daily_match_within_daily_span = 0
    rows_without_daily_match_absent_symbols = 0
    rows_without_daily_match_known_symbols = 0
    absent_symbol_count = 0
    daily_max_trade_date: Optional[str] = None
    coverage_ratio_vs_daily: Optional[float] = None
    if _table_exists(conn, daily_table):
        daily_pairs = conn.execute(
            f"""
            SELECT DISTINCT
                CAST(symbol AS VARCHAR) AS symbol,
                CAST(trade_date AS DATE) AS trade_date
            FROM {daily_table}
            """
        ).df()
        if not daily_pairs.empty:
            daily_pairs["symbol"] = daily_pairs["symbol"].astype(str).str.zfill(6)
            daily_pairs["trade_date"] = pd.to_datetime(daily_pairs["trade_date"], errors="coerce").dt.normalize()
            max_daily_ts = daily_pairs["trade_date"].max()
            daily_max_trade_date = str(max_daily_ts.date()) if pd.notna(max_daily_ts) else None
            merged = dedup.merge(daily_pairs, on=["symbol", "trade_date"], how="left", indicator=True)
            rows_without_daily_match = int((merged["_merge"] == "left_only").sum())
            if pd.notna(max_daily_ts):
                missing = merged[merged["_merge"] == "left_only"].copy()
                rows_after_daily_max_date = int((missing["trade_date"] > max_daily_ts).sum())
                in_span_missing = missing[missing["trade_date"] <= max_daily_ts].copy()
                rows_without_daily_match_within_daily_span = int(len(in_span_missing))
                daily_symbols = set(daily_pairs["symbol"].dropna().astype(str))
                absent_symbol_mask = ~in_span_missing["symbol"].astype(str).isin(daily_symbols)
                rows_without_daily_match_absent_symbols = int(absent_symbol_mask.sum())
                rows_without_daily_match_known_symbols = int((~absent_symbol_mask).sum())
                absent_symbol_count = int(in_span_missing.loc[absent_symbol_mask, "symbol"].nunique())
            overlap_days = daily_pairs["trade_date"].isin(dedup["trade_date"].dropna().unique())
            daily_overlap = daily_pairs.loc[overlap_days]
            if not daily_overlap.empty:
                coverage_ratio_vs_daily = float(len(merged) - rows_without_daily_match) / float(len(daily_overlap))

    notes: List[str] = []
    ok = True
    if dup > 0:
        ok = False
        notes.append("存在重复 (symbol, trade_date) 主键。")
    if rows_without_daily_match_known_symbols > 0:
        ok = False
        notes.append("存在日线已覆盖标的的资金流 trade_date 找不到对应日线行，可能有日期错位。")
    if rows_without_daily_match_absent_symbols > 0:
        notes.append("部分资金流标的完全不在日线表中，通常是数据源市场范围宽于当前日线 universe。")
    elif rows_after_daily_max_date > 0:
        ok = False
        notes.append("资金流日期晚于日线表最新日期；请先补齐日线表再判断资金流对齐质量。")
    if rows_without_daily_match_absent_symbols > 0 and rows_after_daily_max_date > 0:
        ok = False
        notes.append("资金流日期晚于日线表最新日期；请先补齐日线表再判断剩余对齐质量。")
    if all_zero_rows > 0 and all_zero_rows / max(len(dedup), 1) > 0.2:
        ok = False
        notes.append("全零资金流行占比偏高，需排查是否有静默失败或异常回填。")
    high_null_cols = [c for c, ratio in null_ratio_by_col.items() if ratio > 0.3]
    if high_null_cols:
        ok = False
        notes.append("关键资金流列空值率偏高：" + ", ".join(sorted(high_null_cols)))

    return FundFlowQualityReport(
        ok=ok,
        table_exists=True,
        total_rows=int(len(dedup)),
        distinct_symbols=int(dedup["symbol"].nunique()),
        min_trade_date=str(dedup["trade_date"].min().date()) if dedup["trade_date"].notna().any() else None,
        max_trade_date=str(dedup["trade_date"].max().date()) if dedup["trade_date"].notna().any() else None,
        duplicate_pk_rows=dup,
        rows_without_daily_match=rows_without_daily_match,
        rows_after_daily_max_date=rows_after_daily_max_date,
        rows_without_daily_match_within_daily_span=rows_without_daily_match_within_daily_span,
        rows_without_daily_match_absent_symbols=rows_without_daily_match_absent_symbols,
        rows_without_daily_match_known_symbols=rows_without_daily_match_known_symbols,
        absent_symbol_count=absent_symbol_count,
        daily_max_trade_date=daily_max_trade_date,
        all_zero_flow_rows=all_zero_rows,
        null_ratio_by_col=null_ratio_by_col,
        coverage_ratio_vs_daily=coverage_ratio_vs_daily,
        median_symbols_per_day=float(per_day.median()) if not per_day.empty else 0.0,
        p10_symbols_per_day=float(per_day.quantile(0.1)) if not per_day.empty else 0.0,
        notes=notes,
    )


def run_shareholder_quality_checks(
    conn: duckdb.DuckDBPyConnection,
    *,
    table: str = "a_share_shareholder",
    fallback_lag_days: int = 30,
    min_effective_width: int = 100,
) -> ShareholderQualityReport:
    """
    股东人数数据质量检查。

    关注项：
    - notice_date 覆盖率
    - fallback lag 使用比例
    - 公告滞后是否异常
    - 截面覆盖宽度与可用日期数
    """
    if not _table_exists(conn, table):
        return ShareholderQualityReport(ok=False, table_exists=False, notes=[f"表不存在: {table}"])

    raw = conn.execute(
        f"""
        SELECT symbol, end_date, notice_date, holder_count, holder_change
        FROM {table}
        """
    ).df()
    if raw.empty:
        return ShareholderQualityReport(ok=False, table_exists=True, notes=[f"表为空: {table}"])

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["end_date"] = pd.to_datetime(raw["end_date"], errors="coerce").dt.normalize()
    raw["notice_date"] = pd.to_datetime(raw["notice_date"], errors="coerce").dt.normalize()
    raw["holder_count"] = pd.to_numeric(raw["holder_count"], errors="coerce")
    raw["holder_change"] = pd.to_numeric(raw["holder_change"], errors="coerce")

    dup = int(len(raw) - len(raw.drop_duplicates(["symbol", "end_date"])))
    dedup = raw.drop_duplicates(["symbol", "end_date"], keep="last").copy()
    notice_present = raw["notice_date"].notna()
    notice_coverage = float(notice_present.mean()) if len(raw) else 0.0
    fallback_usage = float((~notice_present).mean()) if len(raw) else 0.0
    lag_days = (raw["notice_date"] - raw["end_date"]).dt.days
    negative_notice_lag_rows = int((lag_days < 0).fillna(False).sum())
    per_end_date = dedup.groupby("end_date", dropna=True)["symbol"].nunique()

    availability_date = raw["notice_date"].copy()
    invalid_notice_mask = availability_date.notna() & (availability_date < raw["end_date"])
    fallback_mask = availability_date.isna() | invalid_notice_mask
    availability_date = availability_date.where(
        ~fallback_mask,
        raw["end_date"] + pd.to_timedelta(int(fallback_lag_days), unit="D"),
    )
    effective_dates = availability_date.value_counts(dropna=True)
    effective_factor_dates_ge_min_width = int((effective_dates >= int(min_effective_width)).sum())

    notes: List[str] = []
    ok = True
    if dup > 0:
        ok = False
        notes.append("存在重复 (symbol, end_date) 主键。")
    if notice_coverage < 0.5:
        ok = False
        notes.append("notice_date 覆盖率过低，PIT 可解释性不足。")
    if negative_notice_lag_rows > 0:
        notes.append("存在 notice_date 早于 end_date 的记录，PIT 可用日已按 end_date + fallback lag 保守处理。")
    if float(per_end_date.median()) < float(min_effective_width) if not per_end_date.empty else True:
        ok = False
        notes.append("股东人数截面覆盖宽度不足。")
    if effective_factor_dates_ge_min_width == 0:
        ok = False
        notes.append("没有足够宽度的有效可用日期，单因子研究样本偏弱。")

    return ShareholderQualityReport(
        ok=ok,
        table_exists=True,
        total_rows=int(len(dedup)),
        distinct_symbols=int(dedup["symbol"].nunique()),
        distinct_end_dates=int(dedup["end_date"].nunique()),
        min_end_date=str(dedup["end_date"].min().date()) if dedup["end_date"].notna().any() else None,
        max_end_date=str(dedup["end_date"].max().date()) if dedup["end_date"].notna().any() else None,
        duplicate_pk_rows=dup,
        notice_date_coverage_ratio=notice_coverage,
        fallback_lag_usage_ratio=fallback_usage,
        negative_notice_lag_rows=negative_notice_lag_rows,
        median_notice_lag_days=float(lag_days[lag_days.notna()].median()) if lag_days.notna().any() else None,
        p90_notice_lag_days=float(lag_days[lag_days.notna()].quantile(0.9)) if lag_days.notna().any() else None,
        median_symbols_per_end_date=float(per_end_date.median()) if not per_end_date.empty else 0.0,
        p10_symbols_per_end_date=float(per_end_date.quantile(0.1)) if not per_end_date.empty else 0.0,
        effective_factor_dates_ge_min_width=effective_factor_dates_ge_min_width,
        notes=notes,
    )
