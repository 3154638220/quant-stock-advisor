from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_backtest_eval import (
    _prepared_factors_cache_expected_meta,
    _safe_float_or_nan,
    load_prepared_factors_cache,
    prepare_factors_for_backtest,
    write_prepared_factors_cache,
)


def test_prepared_factors_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "prepared_factors.parquet"
    factors = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "vol_to_turnover": [1.23],
            "_universe_eligible": [True],
        }
    )
    meta = _prepared_factors_cache_expected_meta(
        start_date="2024-01-01",
        end_date="2024-01-31",
        lookback_days=260,
        min_hist_days=130,
        db_path="/tmp/mock.duckdb",
        results_dir="/tmp/results",
        universe_filter_cfg={"enabled": True, "min_amount_20d": 1},
    )

    write_prepared_factors_cache(cache_path, factors, meta)
    loaded = load_prepared_factors_cache(cache_path, meta)

    assert loaded is not None
    assert loaded.equals(factors)


def test_prepared_factors_cache_meta_mismatch_returns_none(tmp_path: Path):
    cache_path = tmp_path / "prepared_factors.parquet"
    factors = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "vol_to_turnover": [1.23],
        }
    )
    meta = _prepared_factors_cache_expected_meta(
        start_date="2024-01-01",
        end_date="2024-01-31",
        lookback_days=260,
        min_hist_days=130,
        db_path="/tmp/mock.duckdb",
        results_dir="/tmp/results",
        universe_filter_cfg={"enabled": False},
    )
    write_prepared_factors_cache(cache_path, factors, meta)

    mismatched = dict(meta)
    mismatched["min_hist_days"] = 252

    assert load_prepared_factors_cache(cache_path, mismatched) is None


def test_prepare_factors_for_backtest_uses_cache_hit(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "prepared_factors.parquet"
    factors = pd.DataFrame(
        {
            "symbol": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "vol_to_turnover": [1.23],
            "_universe_eligible": [True],
        }
    )
    meta = _prepared_factors_cache_expected_meta(
        start_date="2024-01-01",
        end_date="2024-01-31",
        lookback_days=260,
        min_hist_days=130,
        db_path="/tmp/mock.duckdb",
        results_dir=str(tmp_path / "results"),
        universe_filter_cfg={"enabled": True},
    )
    write_prepared_factors_cache(cache_path, factors, meta)

    def _should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("cache hit should bypass factor preparation")

    monkeypatch.setattr("scripts.run_backtest_eval.compute_factors", _should_not_run)

    loaded, cache_hit = prepare_factors_for_backtest(
        pd.DataFrame(),
        min_hist_days=130,
        db_path="/tmp/mock.duckdb",
        results_dir=tmp_path / "results",
        universe_filter_cfg={"enabled": True},
        cache_path=cache_path,
        refresh_cache=False,
        cache_meta=meta,
    )

    assert cache_hit is True
    assert loaded.equals(factors)


def test_safe_float_or_nan_handles_none():
    assert _safe_float_or_nan(None) != _safe_float_or_nan(None)  # nan
    assert _safe_float_or_nan("1.25") == 1.25
