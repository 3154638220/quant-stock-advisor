import json
import sys

import pandas as pd
import pytest

import scripts.run_monthly_benchmark_suite as suite
from scripts.run_monthly_benchmark_suite import (
    BenchmarkSpec,
    annualized_return,
    build_benchmark_suite,
    compounded_return,
    load_index_csv_monthly_returns,
    summarize_relative,
)


def test_compounded_and_annualized_return_use_monthly_compounding():
    returns = pd.Series([0.10, -0.05, 0.02])

    assert compounded_return(returns) == pytest.approx((1.10 * 0.95 * 1.02) - 1.0)
    assert annualized_return(returns) == pytest.approx((1.10 * 0.95 * 1.02) ** 4 - 1.0)


def test_summarize_relative_reports_hit_rate_and_ir():
    strategy = pd.Series([0.03, 0.01, -0.02])
    benchmark = pd.Series([0.01, 0.02, -0.03])

    out = summarize_relative("s", strategy, "b", benchmark)

    assert out["months"] == 3
    assert out["win_months"] == 2
    assert out["monthly_hit_rate"] == pytest.approx(2 / 3)
    assert out["mean_monthly_excess"] == pytest.approx(((0.02) + (-0.01) + 0.01) / 3)


def test_load_index_csv_monthly_returns_aligns_open_to_open(tmp_path):
    csv = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"],
            "symbol": ["000852", "000852", "000852", "000852"],
            "open": [1000.0, 1100.0, 1200.0, 1140.0],
        }
    ).to_csv(csv, index=False)
    schedule = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2023-12-29", "2024-01-31"]),
            "buy_trade_date": pd.to_datetime(["2024-01-02", "2024-02-01"]),
            "sell_trade_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        }
    )

    out, meta = load_index_csv_monthly_returns(BenchmarkSpec("csi1000", csv, "000852"), schedule)

    assert meta["status"] == "ok"
    assert meta["covered_months"] == 2
    assert out.tolist() == pytest.approx([0.10, -0.05])


def test_build_benchmark_suite_includes_internal_alpha_and_broad_ew():
    monthly = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "buy_trade_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "sell_trade_date": pd.to_datetime(["2024-02-29", "2024-03-29"]),
            "topk_return": [0.05, -0.01],
            "cost_drag": [None, 0.001],
            "candidate_pool_mean_return": [0.02, -0.02],
            "market_ew_return": [0.01, -0.03],
        }
    )

    summary, relative, series, meta = build_benchmark_suite(monthly, [])

    assert set(summary["benchmark"]) == {
        "model_top20_net",
        "model_top20_gross",
        "u1_candidate_pool_ew",
        "all_a_market_ew",
    }
    assert set(relative["benchmark"]) == {"u1_candidate_pool_ew", "all_a_market_ew"}
    assert "model_top20_net" in series.columns
    assert meta == []


def test_main_writes_standard_research_manifest(tmp_path, monkeypatch):
    monthly_path = tmp_path / "monthly.csv"
    pd.DataFrame(
        {
            "signal_date": ["2024-01-31", "2024-02-29"],
            "buy_trade_date": ["2024-02-01", "2024-03-01"],
            "sell_trade_date": ["2024-02-29", "2024-03-29"],
            "candidate_pool_version": ["U1_liquid_tradable", "U1_liquid_tradable"],
            "model": ["M8_regime_aware_fixed_policy__indcap3", "M8_regime_aware_fixed_policy__indcap3"],
            "top_k": [20, 20],
            "topk_return": [0.05, -0.01],
            "market_ew_return": [0.01, -0.03],
            "candidate_pool_mean_return": [0.02, -0.02],
            "cost_drag": [None, 0.001],
        }
    ).to_csv(monthly_path, index=False)
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monthly_benchmark_suite.py",
            "--monthly-long",
            "monthly.csv",
            "--results-dir",
            "data/results",
            "--as-of-date",
            "2026-05-02",
        ],
    )

    assert suite.main() == 0

    manifest_path = tmp_path / "data/results/monthly_selection_benchmark_suite_2026-05-02_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research_result_v1"
    assert payload["identity"]["research_topic"] == "monthly_selection_benchmark_suite"
    assert payload["data_slices"][0]["date_start"] == "2024-01-31"
    assert payload["metrics"]["months"] == 2
    assert {item["name"] for item in payload["artifacts"]} >= {"summary_csv", "relative_csv", "report_md"}
    assert (tmp_path / "data/experiments/research_results.jsonl").exists()
