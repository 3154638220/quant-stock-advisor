#!/usr/bin/env python3
"""Fetch and cache A-share symbol/name mapping for research reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.akshare_client import _require_akshare
from src.data_fetcher.akshare_resilience import call_with_timeout, install_akshare_requests_resilience
from src.settings import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch A-share stock names into a local CSV cache")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=str, default="data/cache/a_share_stock_names.csv")
    p.add_argument("--timeout-sec", type=float, default=60.0)
    return p.parse_args()


def _resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _normalize_names(raw: pd.DataFrame) -> pd.DataFrame:
    symbol_col = next((c for c in ["代码", "证券代码", "股票代码", "symbol", "code"] if c in raw.columns), "")
    name_col = next((c for c in ["名称", "股票名称", "证券简称", "name", "stock_name"] if c in raw.columns), "")
    if not symbol_col or not name_col:
        raise ValueError(f"AkShare name table missing symbol/name columns: {list(raw.columns)}")
    out = raw[[symbol_col, name_col]].rename(columns={symbol_col: "symbol", name_col: "name"}).copy()
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out = out[(out["symbol"].str.len() == 6) & out["name"].ne("")]
    return out.drop_duplicates("symbol", keep="last").sort_values("symbol").reset_index(drop=True)


def fetch_stock_names(*, config_path: Path | None, timeout_sec: float) -> pd.DataFrame:
    cfg = load_config(config_path)
    install_akshare_requests_resilience(cfg)
    ak = _require_akshare()

    def _fetch() -> pd.DataFrame:
        return ak.stock_info_a_code_name()

    raw = call_with_timeout(_fetch, timeout_sec=timeout_sec, label="stock_info_a_code_name")
    return _normalize_names(raw)


def main() -> int:
    args = parse_args()
    out_path = _resolve_project_path(args.out)
    names = fetch_stock_names(config_path=args.config, timeout_sec=float(args.timeout_sec))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names.to_csv(out_path, index=False)
    print(f"[stock-names] rows={len(names)} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
