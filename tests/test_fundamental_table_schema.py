"""a_share_fundamental 表结构由 FundamentalClient 初始化时创建。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

from src.data_fetcher.fundamental_client import FundamentalClient


def test_fundamental_client_creates_table_and_index():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.duckdb"
        with FundamentalClient(duckdb_path=str(db)) as fc:
            tbl = fc.cfg.table_name
        con = duckdb.connect(str(db), read_only=True)
        try:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                [tbl],
            ).fetchall()
            assert len(rows) == 1
            cols = {r[1] for r in con.execute(f"PRAGMA table_info('{tbl}')").fetchall()}
            assert "pe_ttm" in cols and "pb" in cols and "announcement_date" in cols
            for c in ("ocf_to_asset", "gross_margin_delta", "asset_turnover", "net_margin_stability"):
                assert c in cols
        finally:
            con.close()
