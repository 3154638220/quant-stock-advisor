from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

from src.data_fetcher.shareholder_client import ShareholderClient


def test_shareholder_client_creates_table_and_index():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.duckdb"
        with ShareholderClient(duckdb_path=str(db)) as client:
            pass
        con = duckdb.connect(str(db), read_only=True)
        try:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
                ["a_share_shareholder"],
            ).fetchall()
            assert len(rows) == 1
            cols = {r[1] for r in con.execute("PRAGMA table_info('a_share_shareholder')").fetchall()}
            assert {"symbol", "end_date", "notice_date", "holder_count", "holder_change", "source", "fetched_at"} <= cols
        finally:
            con.close()
