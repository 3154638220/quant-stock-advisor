"""
DuckDB Schema Migration 机制。

提供轻量级 schema 版本管理与自动演进：
- 在 DuckDB 中维护 ``schema_migrations`` 元数据表
- 定义有序 migration，启动时自动检测并执行未完成的 migration
- 支持 idempotent（幂等）DDL（CREATE TABLE IF NOT EXISTS, ALTER TABLE ADD COLUMN IF NOT EXISTS）

用法::

    from src.data_fetcher.migrations import apply_migrations

    conn = duckdb.connect("data/market.duckdb")
    applied = apply_migrations(conn)
    print(f"Applied {len(applied)} migrations")

典型集成: 在 ``DuckDBManager.__init__`` 结尾调用::

    from src.data_fetcher.migrations import apply_migrations
    apply_migrations(self._conn)
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb

_LOG = logging.getLogger(__name__)

# ── Migration 定义 ───────────────────────────────────────────────────────
# 每个 migration 是一个 (version: int, label: str, sql: str) 三元组。
# version 必须递增，已执行过的 migration 不会重复执行。
# SQL 应当幂等（使用 IF NOT EXISTS / IF EXISTS）。

MIGRATIONS: list[tuple[int, str, str]] = [
    # ── v1: 初始 schema（已由 DuckDBManager._ensure_schema 创建，此处仅做元数据记录）──
    (1, "initial_core_tables", """
        CREATE TABLE IF NOT EXISTS a_share_daily (
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
        );
        CREATE INDEX IF NOT EXISTS idx_a_share_daily_trade_date_symbol
            ON a_share_daily(trade_date, symbol);
    """),
    # ── v2: 新增衍生列（change, pct_chg, amplitude_pct 可能在旧版本中缺失）──
    (2, "add_derived_daily_columns", """
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS change DOUBLE;
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS pct_chg DOUBLE;
        ALTER TABLE a_share_daily ADD COLUMN IF NOT EXISTS amplitude_pct DOUBLE;
    """),
    # ── v3: 审计表 ──
    (3, "data_fetch_audit_table", """
        CREATE TABLE IF NOT EXISTS data_fetch_audit (
            run_id VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP NOT NULL,
            symbol_count INTEGER NOT NULL,
            rows_written BIGINT NOT NULL,
            failures INTEGER NOT NULL,
            duration_ms BIGINT NOT NULL,
            PRIMARY KEY (run_id)
        );
    """),
    # ── v4: 资金流表（可能由其他模块创建，此处确保存在）──
    (4, "fund_flow_table", """
        CREATE TABLE IF NOT EXISTS a_share_fund_flow (
            symbol VARCHAR NOT NULL,
            trade_date DATE NOT NULL,
            main_net_inflow_pct DOUBLE,
            super_large_net_inflow_pct DOUBLE,
            small_net_inflow_pct DOUBLE,
            PRIMARY KEY (symbol, trade_date)
        );
    """),
    # ── v5: IC 监控表 ──
    (5, "ic_monitor_table", """
        CREATE TABLE IF NOT EXISTS ic_monitor (
            factor VARCHAR,
            trade_date DATE,
            ic DOUBLE,
            recorded_at TIMESTAMP,
            PRIMARY KEY (factor, trade_date)
        );
    """),
    # ── v6: 股东数据表 ──
    (6, "shareholder_table", """
        CREATE TABLE IF NOT EXISTS a_share_shareholder (
            symbol VARCHAR NOT NULL,
            end_date DATE NOT NULL,
            holder_count DOUBLE,
            PRIMARY KEY (symbol, end_date)
        );
    """),
    # ── v7: 除权日历表 ──
    (7, "dividend_calendar_table", """
        CREATE TABLE IF NOT EXISTS a_share_dividend_calendar (
            symbol VARCHAR NOT NULL,
            ex_dividend_date DATE NOT NULL,
            PRIMARY KEY (symbol, ex_dividend_date)
        );
    """),
    # ── v8: 运行事件日志表（P2-9: 结构化事件日志，供 SQL 查询分析）──
    (8, "run_events_log", """
        CREATE SEQUENCE IF NOT EXISTS seq_run_events_id START 1;
        CREATE TABLE IF NOT EXISTS run_events (
            event_id BIGINT PRIMARY KEY DEFAULT nextval('seq_run_events_id'),
            run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_type VARCHAR NOT NULL,
            event_payload JSON,
            run_id VARCHAR,
            symbol VARCHAR,
            signal_date DATE
        );
        CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_run_events_ts ON run_events(run_ts);
    """),
    # ── v9: 基本面 raw 数据表（fundamental_client 使用）──
    (9, "fundamental_raw_table", """
        CREATE TABLE IF NOT EXISTS a_share_fundamental_raw (
            symbol VARCHAR NOT NULL,
            report_period DATE NOT NULL,
            notice_date DATE,
            pe_ttm DOUBLE,
            pb DOUBLE,
            ev_ebitda DOUBLE,
            roe_ttm DOUBLE,
            net_profit_yoy DOUBLE,
            gross_margin_change DOUBLE,
            gross_margin_delta DOUBLE,
            debt_to_assets_change DOUBLE,
            ocf_to_net_profit DOUBLE,
            ocf_to_asset DOUBLE,
            asset_turnover DOUBLE,
            net_margin_stability DOUBLE,
            northbound_net_inflow DOUBLE,
            margin_buy_ratio DOUBLE,
            PRIMARY KEY (symbol, report_period)
        );
    """),
    # ── v10: OOS 追踪表（P2-10: 样本外收益记录）──
    (10, "oos_tracking", """
        CREATE TABLE IF NOT EXISTS oos_tracking (
            config_id VARCHAR NOT NULL,
            signal_date DATE NOT NULL,
            run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            top_k INTEGER NOT NULL,
            candidate_pool VARCHAR NOT NULL,
            cost_bps DOUBLE NOT NULL,
            predicted_excess_monthly DOUBLE,
            realized_excess_monthly DOUBLE,
            holding_returns_json VARCHAR,
            benchmark_return DOUBLE,
            num_holdings INTEGER,
            limit_up_excluded INTEGER,
            PRIMARY KEY (config_id, signal_date, top_k, candidate_pool, cost_bps)
        );
        CREATE INDEX IF NOT EXISTS idx_oos_tracking_date ON oos_tracking(signal_date);
    """),
]

# ── Migration 引擎 ───────────────────────────────────────────────────────

_META_TABLE = "schema_migrations"
_META_DDL = f"""
CREATE TABLE IF NOT EXISTS {_META_TABLE} (
    version INTEGER PRIMARY KEY,
    label VARCHAR NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _current_version(conn: duckdb.DuckDBPyConnection) -> int:
    """返回已应用的最大 migration 版本号；未初始化时返回 0。"""
    try:
        row = conn.execute(
            f"SELECT MAX(version) FROM {_META_TABLE}"
        ).fetchone()
        if row is None or row[0] is None:
            return 0
        return int(row[0])
    except Exception:
        return 0


def apply_migrations(
    conn: duckdb.DuckDBPyConnection,
    *,
    target_version: int | None = None,
) -> list[int]:
    """
    执行所有未完成的 migration。

    Parameters
    ----------
    conn
        DuckDB 连接。调用方负责管理连接生命周期。
    target_version : int | None
        目标版本号。若为 None，应用所有待处理 migration。
        可用于回滚测试（仅向前应用，不支持回滚）。

    Returns
    -------
    list[int]
        本次新应用的 migration 版本号列表。
    """
    conn.execute(_META_DDL)

    current = _current_version(conn)
    if target_version is None:
        target = max(v for v, _, _ in MIGRATIONS) if MIGRATIONS else 0
    else:
        target = max(0, int(target_version))

    applied: list[int] = []
    for ver, label, sql in MIGRATIONS:
        if ver <= current:
            continue
        if ver > target:
            break
        _LOG.info("Applying migration v%d: %s", ver, label)
        try:
            conn.execute("BEGIN TRANSACTION;")
            # DuckDB 不支持多语句 execute，需逐条分割
            for stmt in _split_sql(sql):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.execute(
                f"INSERT INTO {_META_TABLE} (version, label) VALUES (?, ?);",
                [ver, label],
            )
            conn.execute("COMMIT;")
            applied.append(ver)
            _LOG.info("Migration v%d applied successfully.", ver)
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            _LOG.error("Migration v%d FAILED: %s", ver, exc)
            raise

    if applied:
        _LOG.info("Applied %d migrations: %s", len(applied), applied)
    return applied


def get_migration_status(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """
    返回 migration 状态摘要：当前版本、已应用列表、待处理列表。

    Returns
    -------
    dict
        ``{"current_version": int, "applied": [...], "pending": [...]}``
    """
    current = _current_version(conn)
    applied_list = []
    pending_list = []
    for ver, label, _ in MIGRATIONS:
        if ver <= current:
            applied_list.append({"version": ver, "label": label})
        else:
            pending_list.append({"version": ver, "label": label})
    return {
        "current_version": current,
        "applied": applied_list,
        "pending": pending_list,
    }


def _split_sql(text: str) -> list[str]:
    """将多语句 SQL 按分号分割为独立语句列表。"""
    statements: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.rstrip().endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    # 最后一个语句可能不以分号结尾
    if buf:
        remaining = "\n".join(buf).strip()
        if remaining:
            statements.append(remaining)
    return statements
