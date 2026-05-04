from __future__ import annotations

import sys

import pandas as pd

# A3: 迁移到 src/ 导入（消除 tests → scripts 硬依赖）
from src.data_fetcher.akshare_client import normalize_max_symbols
from src.data_fetcher.index_benchmarks import IndexFetchSpec, parse_index_specs, standardize_index_daily
from src.data_fetcher.shareholder_client import (
    _latest_completed_quarter_end,
    _quarter_ends_in_range,
    _recent_quarter_ends,
)

# parse_args 函数仍在 scripts/ 中（薄层 CLI），测试它们需要 scripts 导入
# （这些是 CLI 参数解析测试，属于合理依赖）
try:
    from scripts.fetch_fund_flow import parse_args as parse_fund_flow_args
except ImportError:
    parse_fund_flow_args = None  # type: ignore[assignment]
try:
    from scripts.fetch_shareholder import parse_args as parse_shareholder_args
except ImportError:
    parse_shareholder_args = None  # type: ignore[assignment]


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


def test_fetch_only_max_symbols_zero_means_full_universe():
    assert normalize_max_symbols(None) is None
    assert normalize_max_symbols(0) is None
    assert normalize_max_symbols(-1) is None
    assert normalize_max_symbols(25) == 25


def test_fetch_index_benchmarks_standardizes_akshare_daily():
    spec = IndexFetchSpec("csi1000", "000852", "sh000852")
    raw = pd.DataFrame(
        {
            "date": ["2026-04-29", "2026-04-30"],
            "open": ["8163.35", "8345.25"],
            "close": [8343.07, 8381.95],
        }
    )

    out = standardize_index_daily(raw, spec)

    assert out[["trade_date", "open", "symbol"]].to_dict(orient="records") == [
        {"trade_date": pd.Timestamp("2026-04-29"), "open": 8163.35, "symbol": "000852"},
        {"trade_date": pd.Timestamp("2026-04-30"), "open": 8345.25, "symbol": "000852"},
    ]
    assert parse_index_specs([])[0].name == "csi1000"


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
            "--start-date",
            "2015-01-01",
            "--end-date",
            "2025-12-31",
            "--sleep-sec",
            "0.2",
            "--log-every",
            "7",
            "--max-retries",
            "5",
            "--retry-delay-sec",
            "1.5",
            "--require-success",
        ],
    )

    args = parse_shareholder_args()

    assert args.end_dates == "20251231,20250930"
    assert args.duckdb_path == "data/tmp/shareholder.duckdb"
    assert args.latest_n == 2
    assert args.start_date == "2015-01-01"
    assert args.end_date == "2025-12-31"
    assert args.sleep_sec == 0.2
    assert args.log_every == 7
    assert args.max_retries == 5
    assert args.retry_delay_sec == 1.5
    assert args.require_success is True


def test_recent_quarter_ends_returns_compact_date_keys():
    out = _recent_quarter_ends(3)

    assert len(out) == 3
    assert all(len(item) == 8 and item.isdigit() for item in out)


def test_latest_completed_quarter_end_avoids_future_quarter():
    out = _latest_completed_quarter_end("2026-05-01")

    assert out == pd.Timestamp("2026-03-31")


def test_quarter_ends_in_range_covers_2015_history():
    out = _quarter_ends_in_range("2015-01-01", "2015-12-31")

    assert out == ["20150331", "20150630", "20150930", "20151231"]
