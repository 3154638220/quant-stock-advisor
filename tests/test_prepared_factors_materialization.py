from datetime import datetime

import duckdb
import pandas as pd

from scripts.materialize_prepared_factors import check_freshness, write_to_duckdb


def _write_dataset(path) -> None:
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "signal_date": [pd.Timestamp("2026-04-30")],
            "feature_ret_20d": [0.01],
        }
    ).to_parquet(path)


def test_check_freshness_reports_missing_when_table_absent(tmp_path) -> None:
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    db_path = tmp_path / "market.duckdb"
    _write_dataset(dataset_path)

    assert check_freshness(str(db_path), str(dataset_path)) == "missing"


def test_check_freshness_reports_stale_when_dataset_is_newer_than_meta(tmp_path) -> None:
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    db_path = tmp_path / "market.duckdb"
    _write_dataset(dataset_path)

    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE prepared_factors AS SELECT 1 AS marker")
        con.execute(
            """
            CREATE TABLE _materialization_meta (
                table_name VARCHAR,
                materialized_at TIMESTAMP,
                source_file VARCHAR,
                row_count BIGINT,
                col_count BIGINT
            )
            """
        )
        con.execute(
            "INSERT INTO _materialization_meta VALUES (?, ?, ?, ?, ?)",
            [
                "prepared_factors",
                datetime(2000, 1, 1),
                str(dataset_path),
                1,
                1,
            ],
        )

    assert check_freshness(str(db_path), str(dataset_path)) == "stale"


def test_write_to_duckdb_records_materialization_meta_and_freshness(tmp_path) -> None:
    dataset_path = tmp_path / "monthly_selection_features.parquet"
    db_path = tmp_path / "market.duckdb"
    _write_dataset(dataset_path)

    df = pd.read_parquet(dataset_path)
    write_to_duckdb(df, str(db_path), str(dataset_path))

    with duckdb.connect(str(db_path), read_only=True) as con:
        row_count = con.execute("SELECT COUNT(*) FROM prepared_factors").fetchone()[0]
        meta = con.execute(
            """
            SELECT table_name, source_file, row_count, col_count
            FROM _materialization_meta
            WHERE table_name = 'prepared_factors'
            """
        ).fetchone()

    assert row_count == 1
    assert meta[0] == "prepared_factors"
    assert meta[1] == str(dataset_path.resolve())
    assert meta[2] == 1
    assert meta[3] == len(df.columns)
    assert check_freshness(str(db_path), str(dataset_path)) == "fresh"
