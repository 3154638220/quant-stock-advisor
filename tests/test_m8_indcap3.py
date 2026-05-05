"""Tests for src/reporting/m8_indcap3.py - M8 monthly report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting.m8_indcap3 import (
    CANDIDATE_POOL,
    MODEL_NAME,
    TOP_K,
    generate_monthly_reports,
    load_stock_names,
)

# ── load_stock_names ───────────────────────────────────────────────────────


def test_load_stock_names(tmp_path):
    csv_path = tmp_path / "names.csv"
    csv_path.write_text("symbol,name\n000001,平安银行\n000002,万科A\n600000,浦发银行\n")
    df = load_stock_names(csv_path)
    assert len(df) == 3
    assert list(df.columns) == ["symbol", "name"]
    assert df.iloc[0]["symbol"] == "000001"
    assert df.iloc[0]["name"] == "平安银行"


def test_load_stock_names_dedup():
    """Duplicate symbols should be deduplicated, keeping first."""
    csv_path = Path(tempfile.mkdtemp()) / "names.csv"
    csv_path.write_text("symbol,name\n000001,名字A\n000001,名字B\n")
    df = load_stock_names(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "名字A"


def test_load_stock_names_pads_to_6():
    csv_path = Path(tempfile.mkdtemp()) / "names.csv"
    csv_path.write_text("symbol,name\n1,test\n")
    df = load_stock_names(csv_path)
    assert df.iloc[0]["symbol"] == "000001"


# ── generate_monthly_reports ───────────────────────────────────────────────


def _make_holdings_csv(path: Path, months: list[str]) -> Path:
    rows = []
    for m in months:
        signal_date = pd.Timestamp(f"{m}-15")
        for rank in range(1, 21):
            rows.append({
                "model": MODEL_NAME,
                "candidate_pool_version": CANDIDATE_POOL,
                "top_k": TOP_K,
                "signal_date": signal_date,
                "symbol": f"{rank:06d}",
                "selected_rank": rank,
                "score": 0.5 + 0.01 * rank,
                "industry_level1": "银行" if rank <= 5 else "科技",
                "label_forward_1m_o2o_return": 0.01 * (21 - rank) / 20,
                "buy_trade_date": signal_date + pd.Timedelta(days=1),
                "sell_trade_date": signal_date + pd.Timedelta(days=30),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_generate_monthly_reports_creates_files(tmp_path):
    names = pd.DataFrame({"symbol": [f"{i:06d}" for i in range(1, 21)],
                          "name": [f"股票{i}" for i in range(1, 21)]})
    holdings_path = tmp_path / "holdings.csv"
    _make_holdings_csv(holdings_path, ["2023-01", "2023-02"])
    output_dir = tmp_path / "reports"

    generate_monthly_reports(holdings_path, names, output_dir)

    assert output_dir.exists()
    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 3  # 2 monthly + 1 summary
    summary = output_dir / "m8_regime_aware_indcap3_summary.md"
    assert summary.exists()
    content = summary.read_text()
    assert "统计摘要" in content
    assert "月均等权收益" in content


def test_generate_monthly_reports_empty_strategy(tmp_path):
    """When strategy has no matching rows, it should not crash."""
    names = pd.DataFrame({"symbol": [], "name": []})
    df = pd.DataFrame({"model": ["other"], "candidate_pool_version": ["other"],
                       "top_k": [99], "signal_date": [pd.Timestamp("2023-01-01")],
                       "symbol": ["000001"]})
    holdings_path = tmp_path / "holdings.csv"
    df.to_csv(holdings_path, index=False)
    output_dir = tmp_path / "reports"

    generate_monthly_reports(holdings_path, names, output_dir)
    # Should not create any md files since strategy doesn't match
    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 0


def test_generate_monthly_reports_handles_nan_returns(tmp_path):
    names = pd.DataFrame({"symbol": [f"{i:06d}" for i in range(1, 6)],
                          "name": [f"股票{i}" for i in range(1, 6)]})
    rows = []
    for rank in range(1, 6):
        rows.append({
            "model": MODEL_NAME, "candidate_pool_version": CANDIDATE_POOL,
            "top_k": TOP_K, "signal_date": pd.Timestamp("2023-06-15"),
            "symbol": f"{rank:06d}", "selected_rank": rank,
            "score": float(rank), "industry_level1": "金融",
            "label_forward_1m_o2o_return": np.nan,
            "buy_trade_date": pd.Timestamp("2023-06-16"),
            "sell_trade_date": pd.Timestamp("2023-07-16"),
        })
    df = pd.DataFrame(rows)
    holdings_path = tmp_path / "holdings.csv"
    df.to_csv(holdings_path, index=False)
    output_dir = tmp_path / "reports"
    generate_monthly_reports(holdings_path, names, output_dir)
    # Should handle NaN returns gracefully
    assert output_dir.exists()


# ── Constants ──────────────────────────────────────────────────────────────


def test_constants_defined():
    assert isinstance(MODEL_NAME, str)
    assert isinstance(CANDIDATE_POOL, str)
    assert TOP_K == 20
