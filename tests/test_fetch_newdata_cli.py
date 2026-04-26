from __future__ import annotations

import sys

from scripts.fetch_fund_flow import parse_args as parse_fund_flow_args
from scripts.fetch_shareholder import _recent_quarter_ends, parse_args as parse_shareholder_args


def test_fetch_fund_flow_parse_args_accepts_limits(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_fund_flow.py",
            "--input-file",
            "data/raw/fund_flow_sample.csv",
            "--source-label",
            "vendor_x",
            "--cache-path",
            "data/cache/custom.json",
            "--duckdb-path",
            "data/tmp/fund_flow.duckdb",
            "--max-symbols",
            "25",
            "--sleep-sec",
            "0.1",
            "--log-every",
            "5",
        ],
    )

    args = parse_fund_flow_args()

    assert args.input_file == "data/raw/fund_flow_sample.csv"
    assert args.source_label == "vendor_x"
    assert args.cache_path == "data/cache/custom.json"
    assert args.duckdb_path == "data/tmp/fund_flow.duckdb"
    assert args.max_symbols == 25
    assert args.sleep_sec == 0.1
    assert args.log_every == 5


def test_fetch_shareholder_parse_args_accepts_limits(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_shareholder.py",
            "--end-dates",
            "20251231,20250930",
            "--duckdb-path",
            "data/tmp/shareholder.duckdb",
            "--latest-n",
            "2",
            "--sleep-sec",
            "0.2",
            "--log-every",
            "7",
        ],
    )

    args = parse_shareholder_args()

    assert args.end_dates == "20251231,20250930"
    assert args.duckdb_path == "data/tmp/shareholder.duckdb"
    assert args.latest_n == 2
    assert args.sleep_sec == 0.2
    assert args.log_every == 7


def test_recent_quarter_ends_returns_compact_date_keys():
    out = _recent_quarter_ends(3)

    assert len(out) == 3
    assert all(len(item) == 8 and item.isdigit() for item in out)
