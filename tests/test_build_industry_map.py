from __future__ import annotations

import json

import pandas as pd

from src.data_fetcher.industry_map import (
    FALLBACK_SOURCE,
    align_to_universe,
    deduplicate_mapping,
    load_current_universe,
    normalize_symbol,
    quality_summary,
)


def test_normalize_symbol_handles_suffix_and_numeric_text():
    assert normalize_symbol("sh600519") == "600519"
    assert normalize_symbol("1.0") == "000001"
    assert normalize_symbol("000002") == "000002"


def test_deduplicate_mapping_outputs_required_schema_and_no_final_duplicates():
    raw = pd.DataFrame(
        [
            {
                "symbol": "1",
                "industry": "银行",
                "industry_level1": "银行",
                "industry_level2": "银行",
                "source": "ak",
                "asof_date": "2026-04-28",
            },
            {
                "symbol": "000001",
                "industry": "金融",
                "industry_level1": "金融",
                "industry_level2": "银行",
                "source": "ak",
                "asof_date": "2026-04-28",
            },
        ]
    )

    out, raw_dupes = deduplicate_mapping(raw)

    assert raw_dupes == 2
    assert list(out.columns) == ["symbol", "industry", "industry_level1", "industry_level2", "source", "asof_date"]
    assert out["symbol"].tolist() == ["000001"]


def test_align_to_universe_marks_missing_symbols_as_diagnostic_fallback():
    mapping = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "industry": "银行",
                "industry_level1": "银行",
                "industry_level2": "银行",
                "source": "akshare.stock_board_industry_cons_em",
                "asof_date": "2026-04-28",
            }
        ]
    )

    out = align_to_universe(mapping, ["000001", "000002"], "2026-04-28")
    missing = out[out["symbol"] == "000002"].iloc[0]

    assert missing["industry"] == "unknown"
    assert missing["source"] == FALLBACK_SOURCE
    q = quality_summary(out, ["000001", "000002"], "unit", "2026-04-28")
    assert q.coverage_ratio == 0.5
    assert q.unknown_ratio == 0.5
    assert q.fallback_used is True


def test_load_current_universe_falls_back_to_json(tmp_path):
    cache = tmp_path / "universe_symbols.json"
    cache.write_text(json.dumps({"symbols": ["600519", "1"]}), encoding="utf-8")

    symbols, source = load_current_universe(duckdb_path=tmp_path / "missing.duckdb", universe_json=cache)

    assert symbols == ["000001", "600519"]
    assert source.startswith("universe_json:")
