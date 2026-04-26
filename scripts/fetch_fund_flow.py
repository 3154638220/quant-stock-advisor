"""拉取全市场个股资金流历史数据并落库。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetcher.fund_flow_client import FundFlowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="拉取个股资金流历史数据并落库")
    p.add_argument(
        "--input-file",
        default="",
        help="可选：从外部 CSV/Parquet 导入资金流数据，而不是在线抓取",
    )
    p.add_argument(
        "--source-label",
        default="",
        help="导入模式下可选 source 标签；默认 external:<文件名>",
    )
    p.add_argument(
        "--cache-path",
        default="data/cache/universe_symbols.json",
        help="universe symbols 缓存路径",
    )
    p.add_argument(
        "--duckdb-path",
        default="",
        help="可选 DuckDB 路径；为空时使用默认配置",
    )
    p.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="最多处理多少只股票；0 表示全量",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=0.5,
        help="每只股票之间的 sleep 秒数",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="每处理多少只股票打印一次进度",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with FundFlowClient(duckdb_path=args.duckdb_path or None) as client:
        if str(args.input_file).strip():
            input_path = Path(str(args.input_file).strip())
            before_rows = client.table_row_count()
            imported = client.import_file(input_path, source_label=args.source_label or None)
            after_rows = client.table_row_count()
            logger.info(
                "资金流文件导入完成: file=%s imported_rows=%d table_rows_before=%d table_rows_after=%d",
                input_path,
                imported,
                before_rows,
                after_rows,
            )
            return

        cache_path = Path(args.cache_path)
        if not cache_path.exists():
            logger.error("无法读取 universe 缓存: %s", cache_path)
            return

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        symbols = payload.get("symbols", [])
        if int(args.max_symbols) > 0:
            symbols = symbols[: int(args.max_symbols)]
        logger.info("从缓存读取 %d 只股票", len(symbols))

        total = client.update_symbols(
            symbols,
            sleep_sec=float(args.sleep_sec),
            log_every=max(1, int(args.log_every)),
        )
        logger.info("资金流历史数据拉取完成，总行数: %d", total)


if __name__ == "__main__":
    main()
