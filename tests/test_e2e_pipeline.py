"""端到端集成测试：fetch -> factor -> rank -> backtest。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.data_fetcher.db_manager as dbm
from src.backtest.engine import BacktestConfig, run_backtest
from src.data_fetcher import DuckDBManager
from src.features.tensor_alpha import compute_momentum_rsi_torch
from src.models.rank_score import sort_key_for_dataframe


def _write_test_config(tmp_path: Path) -> Path:
    cfg = {
        "paths": {
            "duckdb_path": str(tmp_path / "market.duckdb"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "database": {
            "table_daily": "a_share_daily",
            "table_audit": "data_fetch_audit",
            "auto_backfill_derived_on_init": False,
        },
        "akshare": {
            "fetch_workers": 1,
            "request_timeout_sec": 1,
            "max_fetch_retries": 1,
            "retry_delay_sec": 0,
            "daily_allow_fallback": False,
            "backfill_derived_after_fetch": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _fake_daily(symbol: str, start_date: str, end_date: str, **_: object) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    base = 10.0 + (1.0 if symbol.endswith("1") else 2.0)
    trend = np.linspace(0.0, 0.9, len(dates))
    close = base + trend
    open_ = close * 0.998
    high = close * 1.01
    low = close * 0.99
    volume = np.full(len(dates), 1_000_000.0)
    amount = close * volume
    turnover = np.full(len(dates), 0.02)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "turnover": turnover,
        }
    )


def test_fetch_factor_rank_backtest_e2e(tmp_path, monkeypatch) -> None:
    cfg_path = _write_test_config(tmp_path)
    monkeypatch.setattr(dbm, "fetch_a_share_daily", _fake_daily)

    symbols = ["000001", "000002"]
    with DuckDBManager(config_path=cfg_path) as db:
        counts = db.incremental_update_many(symbols, end_date="20240131")
        assert all(not x.fetch_failed for x in counts.values())
        assert sum(x.rows_written for x in counts.values()) > 0
        daily = db.read_daily_frame(symbols=symbols)

    close_wide = (
        daily.pivot(index="trade_date", columns="symbol", values="close")
        .sort_index()
        .astype(float)
    )
    close_np = close_wide.T.to_numpy(dtype=np.float64)
    mom, rsi = compute_momentum_rsi_torch(
        close_np,
        momentum_window=3,
        rsi_period=3,
        device="cpu",
    )
    snapshot = pd.DataFrame(
        {
            "symbol": close_wide.columns.tolist(),
            "momentum": mom[:, -1].detach().cpu().numpy(),
            "rsi": rsi[:, -1].detach().cpu().numpy(),
        }
    )
    ranked = sort_key_for_dataframe(
        snapshot,
        sort_by="composite",
        w_momentum=0.6,
        w_rsi=0.4,
        rsi_mode="level",
    )
    assert ranked.iloc[0]["rank"] == 1
    assert "composite_score" in ranked.columns

    asset_returns = close_wide.pct_change().fillna(0.0)
    rebalance_date = asset_returns.index[3]
    top_symbols = ranked["symbol"].head(2).tolist()
    row = {c: 0.0 for c in asset_returns.columns}
    for sym in top_symbols:
        row[sym] = 0.5
    weights_signal = pd.DataFrame([row], index=[rebalance_date]).reindex(columns=asset_returns.columns)

    res = run_backtest(
        asset_returns.loc[rebalance_date:],
        weights_signal,
        config=BacktestConfig(execution_mode="close_to_close"),
    )
    assert len(res.daily_returns) > 0
    assert np.isfinite(res.panel.annualized_return)
