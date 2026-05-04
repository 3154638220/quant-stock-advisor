#!/usr/bin/env python3
"""仅增量更新 DuckDB 日线表，不跑因子与月度选股报告（适合网络不稳定时先拉数）。

用法:
    python scripts/fetch_only.py --max-symbols 200
    python scripts/fetch_only.py --max-symbols 0    # 全量更新
    python scripts/fetch_only.py --symbols 600519,000001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.fetch_only import run_fetch_only

EPILOG = """
参数说明
  --max-symbols N   仅拉取全市场列表前 N 只；0 或负数=全量。与 --symbols 互斥。
  --symbols A,B,... 逗号分隔 6 位 A 股代码。
  --config PATH     自定义 config.yaml。

退出码: 0 成功；1 致命错误。
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="仅 AkShare → DuckDB 增量日线（不计算月度因子）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument("--max-symbols", type=int, default=None, help="全市场列表前 N 只")
    p.add_argument("--symbols", type=str, default=None, help="逗号分隔 6 位代码")
    p.add_argument("--config", type=Path, default=None, help="config.yaml 路径")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_fetch_only(
        symbols=args.symbols,
        max_symbols=args.max_symbols,
        config_path=args.config,
        root=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
