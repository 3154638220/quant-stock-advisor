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


def _recent_quarter_ends(latest_n: int) -> list[str]:
    latest = pd.Timestamp.today().normalize()
    quarter_end = latest.to_period("Q").end_time.normalize()
    dates: list[str] = []
    cur = quarter_end
    for _ in range(max(1, int(latest_n))):
        dates.append(cur.strftime("%Y%m%d"))
        cur = (cur - pd.offsets.QuarterEnd()).normalize()
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end_dates = [item.strip() for item in str(args.end_dates).split(",") if item.strip()]
    if not end_dates:
        end_dates = _recent_quarter_ends(int(args.latest_n))
    logger.info("准备抓取 %d 个截止日: %s", len(end_dates), ", ".join(end_dates))

    with ShareholderClient(duckdb_path=args.duckdb_path or None) as client:
        total = client.update_end_dates(
            end_dates,
            sleep_sec=float(args.sleep_sec),
            log_every=max(1, int(args.log_every)),
        )
        logger.info("股东人数历史数据拉取完成，总行数: %d", total)


if __name__ == "__main__":
    main()
