#!/usr/bin/env python3
"""补拉「全市场代码表」相对 DuckDB 尚未有线数据的标的。

用法:
    python scripts/fetch_universe_gap.py --dry-run
    python scripts/fetch_universe_gap.py
    python scripts/fetch_universe_gap.py --use-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.fetch_universe_gap import run_fetch_universe_gap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="补拉代码表相对 DuckDB 的缺口标的")
    p.add_argument("--config", type=Path, default=None, help="config.yaml 路径")
    p.add_argument("--dry-run", action="store_true", help="只打印数量与差集规模，不写库")
    p.add_argument("--use-cache", action="store_true", help="不请求网络，仅用本地 universe_symbols.json 与库做差")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_fetch_universe_gap(
        config_path=args.config,
        dry_run=args.dry_run,
        use_cache=args.use_cache,
        root=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
