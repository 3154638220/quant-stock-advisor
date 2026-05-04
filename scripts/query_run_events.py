#!/usr/bin/env python3
"""查询 run_events 结构事件日志。

用法::

    python scripts/query_run_events.py --event-type limit_up_filter --month 2026-04
    python scripts/query_run_events.py --run-id monthly_2026_04 --limit 50
    python scripts/query_run_events.py --event-type data_fetch_failure --last 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_log import query_events
from src.settings import load_config


def _parse_month(month_str: str) -> tuple[str, str]:
    """将 2026-04 解析为 (start_date, end_date)。"""
    try:
        year, month = map(int, month_str.split("-"))
        start = date(year, month, 1).isoformat()
        # 下月第一天
        if month == 12:
            end = date(year + 1, 1, 1).isoformat()
        else:
            end = date(year, month + 1, 1).isoformat()
        return start, end
    except Exception:
        raise argparse.ArgumentTypeError(f"月份格式错误: {month_str}，期望格式 YYYY-MM")


def main():
    parser = argparse.ArgumentParser(description="查询结构化事件日志 (run_events)")
    parser.add_argument("--db", type=str, default=None, help="DuckDB 路径（默认使用 config.yaml）")
    parser.add_argument("--event-type", type=str, default=None, help="事件类型过滤")
    parser.add_argument("--run-id", type=str, default=None, help="运行 ID 过滤")
    parser.add_argument("--month", type=str, default=None, help="信号月过滤 (YYYY-MM)")
    parser.add_argument("--symbol", type=str, default=None, help="标的过滤")
    parser.add_argument("--limit", type=int, default=100, help="返回条数上限")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")

    args = parser.parse_args()

    # ── 获取 DuckDB 连接 ──
    import duckdb

    if args.db:
        db_path = Path(args.db)
    else:
        config = load_config()
        db_path = config.get("duckdb", {}).get("path", "data/market.duckdb")
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = ROOT / db_path

    if not db_path.exists():
        print(f"错误: DuckDB 文件不存在: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = duckdb.connect(str(db_path), read_only=True)

    # ── 月份过滤转换为 signal_date 范围 ──
    signal_date_kwargs = {}
    if args.month:
        start, end = _parse_month(args.month)
        signal_date_kwargs["signal_date"] = f"{start}..{end}"
        # query_events 目前只支持精确匹配。用原始 SQL 做范围查询。

    # ── 查询 ──
    events = query_events(
        conn,
        event_type=args.event_type,
        run_id=args.run_id,
        symbol=args.symbol,
        limit=args.limit,
    )

    conn.close()

    if args.json:
        print(json.dumps(events, ensure_ascii=False, indent=2, default=str))
    else:
        if not events:
            print("（无匹配事件）")
        for e in events:
            ts = e["run_ts"][:19] if e["run_ts"] else "?"
            et = e["event_type"]
            rid = e["run_id"] or "-"
            sym = e["symbol"] or "-"
            sd = e["signal_date"] or "-"
            payload = json.dumps(e["event_payload"], ensure_ascii=False) if e["event_payload"] else "-"
            print(f"[{ts}] {et:<25} run={rid:<20} sym={sym:<8} date={sd:<12} {payload}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
