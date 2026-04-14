from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pandas as pd
import pytest

from src.data_fetcher.akshare_resilience import call_with_timeout, fetch_dataframe_with_cache


def _cfg(tmp_path: Path) -> dict:
    return {
        "akshare": {
            "cache_dir": str(tmp_path / "akshare-cache"),
            "request_timeout_sec": 1.0,
            "http_connect_timeout_sec": 0.5,
            "http_read_timeout_sec": 1.0,
            "http_transport_retries": 0,
            "http_retry_backoff_sec": 0.0,
            "api_retry_delay_sec": 0.0,
            "stale_cache_on_error": True,
        }
    }


def test_fetch_dataframe_with_cache_falls_back_to_secondary_fetcher(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    expected = pd.DataFrame({"code": ["600519", "000001"]})

    def _fail():
        raise RuntimeError("primary down")

    got = fetch_dataframe_with_cache(
        [
            ("primary", _fail),
            ("secondary", lambda: expected),
        ],
        cache_key="universe",
        cache_ttl_sec=3600,
        retries=1,
        cfg=cfg,
    )

    assert got.equals(expected)
    payload = json.loads((tmp_path / "akshare-cache" / "universe.json").read_text(encoding="utf-8"))
    assert payload["source"] == "secondary"


def test_fetch_dataframe_with_cache_uses_stale_cache_on_error(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    expected = pd.DataFrame({"title": ["cached news"], "datetime": ["2026-03-31 09:30:00"]})

    warm = fetch_dataframe_with_cache(
        [("news", lambda: expected)],
        cache_key="stock_news_600519",
        cache_ttl_sec=3600,
        retries=1,
        cfg=cfg,
    )
    assert warm.equals(expected)

    cached = fetch_dataframe_with_cache(
        [("news", lambda: (_ for _ in ()).throw(RuntimeError("network down")))],
        cache_key="stock_news_600519",
        cache_ttl_sec=3600,
        retries=1,
        cfg=cfg,
    )

    assert list(cached["title"]) == ["cached news"]
    assert str(pd.to_datetime(cached.iloc[0]["datetime"])) == "2026-03-31 09:30:00"


def test_call_with_timeout_uses_future_timeout_outside_main_thread(monkeypatch) -> None:
    monkeypatch.delattr(signal, "SIGALRM", raising=False)

    def _slow():
        time.sleep(0.2)
        return "ok"

    with pytest.raises(TimeoutError, match="超时"):
        call_with_timeout(_slow, timeout_sec=0.01, label="slow-fetch")
