#!/usr/bin/env python3
"""
抓取 A 股 symbol -> industry 映射，供回测行业上限约束使用。

默认输出: data/cache/industry_map.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"缺少预期列，候选={candidates}, 实际={list(df.columns)}")


def fetch_industry_mapping(sleep_sec: float = 0.2) -> pd.DataFrame:
    import akshare as ak

    boards = ak.stock_board_industry_name_em()
    board_col = _pick_col(boards, ["板块名称", "name"])
    out_rows: list[dict[str, str]] = []
    for board in boards[board_col].astype(str).tolist():
        try:
            cons = ak.stock_board_industry_cons_em(symbol=board)
        except Exception:
            continue
        code_col = _pick_col(cons, ["代码", "symbol"])
        for sym in cons[code_col].astype(str).tolist():
            s6 = str(sym).strip().zfill(6)
            if len(s6) == 6 and s6.isdigit():
                out_rows.append({"symbol": s6, "industry": str(board).strip()})
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if not out_rows:
        return pd.DataFrame(columns=["symbol", "industry"])
    tab = pd.DataFrame(out_rows).drop_duplicates(subset=["symbol"], keep="first")
    return tab.sort_values(["industry", "symbol"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="抓取 A 股行业映射 CSV")
    p.add_argument("--output", default="data/cache/industry_map.csv", help="输出 CSV 路径")
    p.add_argument("--sleep-sec", type=float, default=0.2, help="行业板块请求间隔秒数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = fetch_industry_mapping(sleep_sec=max(args.sleep_sec, 0.0))
    out = Path(args.output)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已写入 {out} | rows={len(table)}")


if __name__ == "__main__":
    main()
