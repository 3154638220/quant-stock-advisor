#!/usr/bin/env python3
"""Compatibility wrapper for the canonical industry map builder."""

from __future__ import annotations

import argparse

from scripts.build_industry_map import main as build_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="抓取 A 股行业映射 CSV")
    p.add_argument("--output", default="data/cache/industry_map.csv", help="输出 CSV 路径")
    p.add_argument("--sleep-sec", type=float, default=0.2, help="行业板块请求间隔秒数")
    p.add_argument("--asof-date", default="", help="映射日期；默认使用当天")
    return p.parse_args()


def main() -> None:
    build_main()


if __name__ == "__main__":
    main()
