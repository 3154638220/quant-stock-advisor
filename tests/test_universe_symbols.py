"""股票池获取回退逻辑测试。"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from src.data_fetcher.akshare_client import fetch_a_share_daily, list_default_universe_symbols


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {
        "paths": {
            "duckdb_path": str(db_path),
            "universe_cache_path": str(tmp_path / "cache" / "universe_symbols.json"),
        },
        "database": {"table_daily": "a_share_daily"},
        "akshare": {
            "min_cached_universe_symbols": 1000,
            "universe_source_retries": 1,
            "universe_retry_delay_sec": 0.0,
            "universe_source_timeout_sec": 1.0,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path


def test_list_default_universe_symbols_falls_back_to_secondary_api(tmp_path, monkeypatch) -> None:
    def _fail_primary():
        raise RuntimeError("primary unavailable")

    def _secondary():
        return pd.DataFrame({"code": ["600519", "000001", "600519"]})

    monkeypatch.setattr("src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em", _fail_primary)
    monkeypatch.setattr("src.data_fetcher.akshare_client.ak.stock_info_a_code_name", _secondary)
    cfg_path = _write_config(tmp_path, tmp_path / "missing.duckdb")

    got = list_default_universe_symbols(
        max_symbols=2,
        config_path=cfg_path,
    )

    assert got == ["600519", "000001"]


def test_list_default_universe_symbols_prefers_akshare_when_config(tmp_path, monkeypatch) -> None:
    """universe_prefer_akshare 时先走接口，不因 DuckDB 已有数据而跳过。"""
    db_path = tmp_path / "market.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE a_share_daily(symbol VARCHAR, trade_date DATE)")
    con.execute("INSERT INTO a_share_daily VALUES ('999999', '2026-03-24')")
    con.close()

    cfg_path = _write_config(tmp_path, db_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["akshare"]["universe_prefer_akshare"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: pd.DataFrame({"code": ["600519", "000001"]}),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: (_ for _ in ()).throw(RuntimeError("should not be reached if primary ok")),
    )

    got = list_default_universe_symbols(max_symbols=2, config_path=cfg_path)

    assert got == ["600519", "000001"]


def test_list_default_universe_symbols_falls_back_to_duckdb(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "market.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE a_share_daily(symbol VARCHAR, trade_date DATE)")
    con.execute(
        """
        INSERT INTO a_share_daily VALUES
        ('000001', '2026-03-24'),
        ('600519', '2026-03-24'),
        ('000001', '2026-03-25')
        """
    )
    con.close()

    cfg_path = _write_config(tmp_path, db_path)

    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: (_ for _ in ()).throw(RuntimeError("primary unavailable")),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: (_ for _ in ()).throw(RuntimeError("secondary unavailable")),
    )

    got = list_default_universe_symbols(max_symbols=1, config_path=cfg_path)

    # 限制只取 1 只时不再用「字典序第一只」，避免与全 000xxx 截取问题同源；种子 42 下为 600519
    assert got == ["600519"]


def test_list_default_universe_symbols_uses_local_cache_when_apis_fail(tmp_path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path, tmp_path / "missing.duckdb")
    cache_path = tmp_path / "cache" / "universe_symbols.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-25T09:00:00",
                "source": "stock_info_a_code_name",
                "count": 1200,
                "symbols": ["600519", "000001", "300750"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: (_ for _ in ()).throw(RuntimeError("primary unavailable")),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: (_ for _ in ()).throw(RuntimeError("secondary unavailable")),
    )

    got = list_default_universe_symbols(max_symbols=2, config_path=cfg_path)

    assert got == ["600519", "000001"]


def test_list_default_universe_symbols_persists_cache_after_external_success(tmp_path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path, tmp_path / "missing.duckdb")
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: pd.DataFrame({"code": ["600519", "000001", "300750"]}),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    got = list_default_universe_symbols(max_symbols=2, config_path=cfg_path)

    cache_path = tmp_path / "cache" / "universe_symbols.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert got == ["600519", "000001"]
    assert payload["source"] == "stock_info_a_code_name"
    assert payload["count"] == 3
    assert payload["symbols"] == ["600519", "000001", "300750"]


def test_fetch_a_share_daily_falls_back_to_stock_zh_a_hist(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_daily",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("daily unavailable")),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_hist",
        lambda symbol, **kwargs: pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-20", "2026-03-25"]),
                "开盘": [10.0, 10.5],
                "最高": [10.2, 10.8],
                "最低": [9.9, 10.4],
                "收盘": [10.1, 10.7],
                "成交量": [1000, 1200],
                "成交额": [10000, 12840],
                "换手率": [0.01, 0.02],
                "涨跌额": [0.1, 0.6],
                "涨跌幅": [1.0, 5.94],
                "振幅": [3.0, 3.96],
            }
        ),
    )

    df = fetch_a_share_daily("000001", "20260321", "20260325")

    assert list(df["symbol"]) == ["000001", "000001"]
    assert list(df["trade_date"].dt.strftime("%Y-%m-%d")) == ["2026-03-20", "2026-03-25"]
    assert float(df.iloc[1]["close"]) == 10.7
    assert {"amplitude_pct", "pct_chg", "change"}.issubset(df.columns)
    assert round(float(df.iloc[1]["change"]), 6) == 0.6
