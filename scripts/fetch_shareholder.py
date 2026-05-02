"""拉取全市场个股股东人数历史数据并落库。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher.shareholder_client import ShareholderClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _latest_completed_quarter_end(asof: str | pd.Timestamp | None = None) -> pd.Timestamp:
    latest = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof).normalize()
    quarter_end = latest.to_period("Q").end_time.normalize()
    if latest < quarter_end:
        quarter_end = (quarter_end - pd.offsets.QuarterEnd()).normalize()
    return quarter_end


def _recent_quarter_ends(latest_n: int) -> list[str]:
    quarter_end = _latest_completed_quarter_end()
    dates: list[str] = []
    cur = quarter_end
    for _ in range(max(1, int(latest_n))):
        dates.append(cur.strftime("%Y%m%d"))
        cur = (cur - pd.offsets.QuarterEnd()).normalize()
    return dates


def _quarter_ends_in_range(start_date: str, end_date: str = "") -> list[str]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拉取股东人数季度截面并落库")
    p.add_argument(
        "--end-dates",
        default="",
        help="要抓取的截止日，逗号分隔，格式如 20251231；为空时按 --latest-n 自动生成",
    )
    p.add_argument(
        "--duckdb-path",
        default="",
        help="可选 DuckDB 路径；为空时使用默认配置",
    )
    p.add_argument(
        "--latest-n",
        type=int,
        default=4,
        help="当 --end-dates 为空时，自动抓取最近多少个季度末",
    )
    p.add_argument(
        "--start-date",
        default="",
        help="可选：按日期范围自动生成季度末；例如 2015-01-01",
    )
    p.add_argument(
        "--end-date",
        default="",
        help="可选：日期范围上界；默认使用最近已完成季度末",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=0.5,
        help="每个截止日之间的 sleep 秒数",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="每处理多少个截止日打印一次进度",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="单个季度末抓取失败时最多重试次数",
    )
    p.add_argument(
        "--retry-delay-sec",
        type=float,
        default=1.0,
        help="单个季度末重试等待秒数（后续尝试线性递增）",
    )
    p.add_argument(
        "--require-success",
        action="store_true",
        help="若任一季度末抓取失败，则以非 0 退出码结束",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    end_dates = [item.strip() for item in str(args.end_dates).split(",") if item.strip()]
    if not end_dates and str(args.start_date).strip():
        end_dates = _quarter_ends_in_range(str(args.start_date).strip(), str(args.end_date).strip())
    elif not end_dates:
        end_dates = _recent_quarter_ends(int(args.latest_n))
    if not end_dates:
        logger.error("未生成任何季度末，请检查 --start-date/--end-date 是否有效。")
        return 1
    logger.info("准备抓取 %d 个截止日: %s", len(end_dates), ", ".join(end_dates))

    with ShareholderClient(duckdb_path=args.duckdb_path or None) as client:
        summary = client.update_end_dates_summary(
            end_dates,
            sleep_sec=float(args.sleep_sec),
            log_every=max(1, int(args.log_every)),
            max_retries=max(1, int(args.max_retries)),
            retry_delay_sec=max(0.0, float(args.retry_delay_sec)),
        )
        logger.info(
            "股东人数历史数据拉取完成，总行数: %d，成功截止日: %d，失败截止日: %d",
            summary.total_rows,
            summary.success_dates,
            summary.failed_dates,
        )
        if args.require_success and summary.failed_dates > 0:
            logger.error("存在失败截止日，按 --require-success 要求返回非 0 退出码。")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
