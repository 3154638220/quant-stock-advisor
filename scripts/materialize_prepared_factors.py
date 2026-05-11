#!/usr/bin/env python3
"""Materialize the full prepared_factors table into duckdb for W5/W6 analysis.

Loads the base monthly-selection dataset and attaches all 13 non-price-volume
feature families via DataLoader, then writes the result to duckdb as the
``prepared_factors`` table.  Existing rows are replaced.

Usage::

    python scripts/materialize_prepared_factors.py
    python scripts/materialize_prepared_factors.py --families quality,reversal_volume,liquidity_position
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.shared_loaders import (
    CANONICAL_FAMILY_ORDER,
    DataLoader,
    DataLoaderConfig,
)

_LOG = logging.getLogger(__name__)

ALL_FAMILIES: list[str] = [f for f in CANONICAL_FAMILY_ORDER if f != "price_volume"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize prepared_factors into duckdb")
    p.add_argument("--db", default="data/market.duckdb", help="DuckDB path")
    p.add_argument(
        "--dataset",
        default="data/cache/monthly_selection_features.parquet",
        help="Base monthly-selection parquet",
    )
    p.add_argument(
        "--families",
        default=",".join(ALL_FAMILIES),
        help="Comma-separated family names to attach",
    )
    p.add_argument("--skip-families", default="", help="Comma-separated families to skip")
    p.add_argument(
        "--batch-size", type=int, default=0,
        help="Process N months per batch (0 = all at once)",
    )
    p.add_argument(
        "--skip-freshness-check", action="store_true",
        help="Skip the staleness check against source dataset mtime",
    )
    return p.parse_args()


def load_base_dataset(path: str) -> pd.DataFrame:
    """Load the base monthly-selection parquet with price-volume features + labels."""
    df = pd.read_parquet(path)
    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.normalize()
    # Keep label columns needed by W5/W6
    _LOG.info("Base dataset: %s rows, %s cols, %s unique dates", len(df), len(df.columns), df["signal_date"].nunique())
    return df


def attach_families(
    dataset: pd.DataFrame,
    db_path: str,
    families: list[str],
) -> pd.DataFrame:
    """Attach feature families via DataLoader, skipping any that fail."""
    loader = DataLoader(db_path, config=DataLoaderConfig(strict_unknown_families=False))
    result = dataset.copy()
    succeeded: list[str] = []
    failed: list[str] = []

    for family in families:
        if family not in loader.registry:
            _LOG.warning("Skipping unknown family: %s", family)
            continue
        t0 = time.monotonic()
        try:
            result = loader.attach(result, [family])
            elapsed = time.monotonic() - t0
            new_cols = [c for c in result.columns if c not in dataset.columns]
            _LOG.info("  %s: attached, %d new cols in %.1fs", family, len(new_cols), elapsed)
            succeeded.append(family)
        except Exception:
            elapsed = time.monotonic() - t0
            _LOG.exception("  %s: FAILED after %.1fs, skipping", family, elapsed)
            failed.append(family)

    _LOG.info("Succeeded: %s", succeeded)
    if failed:
        _LOG.warning("Failed: %s", failed)
    return result


def check_freshness(db_path: str, dataset_path: str) -> str:
    """Compare prepared_factors last-write time against dataset mtime.

    Returns one of 'fresh', 'stale', 'missing', 'unknown'.
    """
    try:
        dataset_mtime = os.path.getmtime(dataset_path)
    except OSError:
        _LOG.warning("Dataset file not found, skipping freshness check: %s", dataset_path)
        return "missing"

    con = duckdb.connect(db_path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='prepared_factors'"
        ).fetchall()]
        if "prepared_factors" not in tables:
            return "missing"

        meta_exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='_materialization_meta'"
        ).fetchone()[0] > 0
        if not meta_exists:
            return "unknown"

        row = con.execute(
            "SELECT materialized_at FROM _materialization_meta "
            "WHERE table_name='prepared_factors' ORDER BY materialized_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return "unknown"

        last_write_ts = pd.Timestamp(row[0]).timestamp()
        if dataset_mtime > last_write_ts + 1:  # 1s tolerance for filesystem timestamp granularity
            return "stale"
        return "fresh"
    finally:
        con.close()


def write_to_duckdb(df: pd.DataFrame, db_path: str, dataset_path: str) -> None:
    """Write (or replace) the prepared_factors table in duckdb."""
    con = duckdb.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS prepared_factors")
        # Register the dataframe and create table from it
        con.execute("CREATE TABLE prepared_factors AS SELECT * FROM df")
        row_count = con.execute("SELECT COUNT(*) FROM prepared_factors").fetchone()[0]
        col_count = len(con.execute("DESCRIBE prepared_factors").fetchall())
        _LOG.info("prepared_factors written: %d rows, %d columns", row_count, col_count)

        # Record materialization metadata for freshness checks
        con.execute("""
            CREATE TABLE IF NOT EXISTS _materialization_meta (
                table_name VARCHAR,
                materialized_at TIMESTAMP,
                source_file VARCHAR,
                row_count BIGINT,
                col_count BIGINT
            )
        """)
        con.execute(
            "INSERT INTO _materialization_meta VALUES (?, NOW(), ?, ?, ?)",
            ["prepared_factors", str(Path(dataset_path).resolve()), row_count, col_count],
        )
        _LOG.info("_materialization_meta updated")
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    skip = {f.strip() for f in args.skip_families.split(",") if f.strip()}
    families = [f for f in families if f not in skip]

    # Freshness check: warn if parquet was updated after last materialization
    if not args.skip_freshness_check:
        status = check_freshness(args.db, args.dataset)
        if status == "stale":
            _LOG.warning(
                "Dataset %s is newer than prepared_factors table in %s. "
                "Consider re-running materialization to keep W5/W6 analysis in sync.",
                args.dataset, args.db,
            )
        elif status == "missing":
            _LOG.info("prepared_factors table not yet created — first run")
        elif status == "unknown":
            _LOG.info("No materialization metadata found — will record after this run")
        else:
            _LOG.info("prepared_factors is up to date with dataset")

    _LOG.info("Loading base dataset from %s", args.dataset)
    df = load_base_dataset(args.dataset)

    _LOG.info("Attaching %d families: %s", len(families), families)
    df = attach_families(df, args.db, families)

    _LOG.info("Writing prepared_factors to %s", args.db)
    write_to_duckdb(df, args.db, args.dataset)

    # Print factor column summary
    factor_cols = [c for c in df.columns if c.startswith("feature_") and not c.endswith("_z")]
    z_cols = [c for c in df.columns if c.endswith("_z")]
    missing_cols = [c for c in df.columns if c.startswith("is_missing_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]
    _LOG.info(
        "Factor summary: %d raw, %d z-score, %d missing flags, %d labels",
        len(factor_cols), len(z_cols), len(missing_cols), len(label_cols),
    )


if __name__ == "__main__":
    main()
