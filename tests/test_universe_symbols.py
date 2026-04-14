"""股票池获取回退逻辑测试。"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from src.data_fetcher.akshare_client import fetch_a_share_daily, list_default_universe_symbols
from src.data_fetcher.db_manager import DuckDBManager


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
            "universe_merge_duckdb_and_cache": True,
            "universe_target_min_symbols": 0,
            "incremental_universe_duckdb_only": True,
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
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: pd.DataFrame({"code": ["600519", "000001"]}),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: (_ for _ in ()).throw(RuntimeError("secondary should not run if em ok")),
    )

    got = list_default_universe_symbols(max_symbols=2, config_path=cfg_path)

    assert got == ["600519", "000001"]


def test_incremental_universe_duckdb_only_skips_local_cache_union(tmp_path, monkeypatch) -> None:
    """库内标的达阈值时仅用 DuckDB，不与 universe_symbols.json 并集（增量拉取只数与库内一致）。"""
    db_path = tmp_path / "market.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE a_share_daily(symbol VARCHAR, trade_date DATE)")
    for sym in ("000001", "600519", "300750"):
        con.execute("INSERT INTO a_share_daily VALUES (?, '2026-03-24')", [sym])
    con.close()

    cfg_path = _write_config(tmp_path, db_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["akshare"]["min_cached_universe_symbols"] = 2
    cfg["akshare"]["incremental_universe_duckdb_only"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    cache_path = tmp_path / "cache" / "universe_symbols.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "symbols": ["000001", "600519", "300750", "688001", "000002"],
                "count": 5,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _should_not_call_universe_api():
        raise AssertionError("全市场代码表不应在 duckdb_incremental_only 路径被请求")

    monkeypatch.setattr(
        "src.data_fetcher.akshare_client._fetch_akshare_universe_codes",
        _should_not_call_universe_api,
    )

    got = list_default_universe_symbols(config_path=cfg_path)
    assert got == ["000001", "300750", "600519"]


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

    assert len(got) == 2
    assert set(got).issubset({"600519", "000001", "300750"})


def test_list_default_universe_symbols_persists_cache_after_external_success(tmp_path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path, tmp_path / "missing.duckdb")
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_spot_em",
        lambda: pd.DataFrame({"code": ["600519", "000001", "300750"]}),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_info_a_code_name",
        lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    got = list_default_universe_symbols(max_symbols=2, config_path=cfg_path)

    cache_path = tmp_path / "cache" / "universe_symbols.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert got == ["600519", "000001"]
    assert payload["source"] == "stock_zh_a_spot_em"
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
                "日期": pd.to_datetime(["2026-03-24", "2026-03-25"]),
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
    assert list(df["trade_date"].dt.strftime("%Y-%m-%d")) == ["2026-03-24", "2026-03-25"]
    assert float(df.iloc[1]["close"]) == 10.7
    assert {"amplitude_pct", "pct_chg", "change"}.issubset(df.columns)
    assert round(float(df.iloc[1]["change"]), 6) == 0.6


def test_fetch_a_share_daily_passes_date_range_to_sina(monkeypatch) -> None:
    seen = {}

    def _mock_sina(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-24", "2026-03-25"]),
                "open": [10.0, 10.5],
                "high": [10.2, 10.7],
                "low": [9.9, 10.3],
                "close": [10.1, 10.6],
                "volume": [1000, 1100],
            }
        )

    monkeypatch.setattr("src.data_fetcher.akshare_client.ak.stock_zh_a_daily", _mock_sina)

    df = fetch_a_share_daily("600519", "20260324", "20260325", adjust="qfq")

    assert seen["symbol"] == "sh600519"
    assert seen["start_date"] == "20260324"
    assert seen["end_date"] == "20260325"
    assert seen["adjust"] == "qfq"
    assert len(df) == 2


def test_fetch_a_share_daily_supports_em_preference(monkeypatch) -> None:
    called = {"sina": 0, "em": 0}

    def _mock_sina(**kwargs):
        called["sina"] += 1
        raise AssertionError("sina should not be called first")

    def _mock_em(symbol, **kwargs):
        called["em"] += 1
        return pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-25"]),
                "开盘": [10.0],
                "最高": [10.2],
                "最低": [9.9],
                "收盘": [10.1],
                "成交量": [1000],
                "成交额": [10000],
                "换手率": [0.01],
                "涨跌额": [0.1],
                "涨跌幅": [1.0],
                "振幅": [3.0],
            }
        )

    monkeypatch.setattr("src.data_fetcher.akshare_client.ak.stock_zh_a_daily", _mock_sina)
    monkeypatch.setattr("src.data_fetcher.akshare_client.ak.stock_zh_a_hist", _mock_em)

    df = fetch_a_share_daily("000001", "20260325", "20260325", source_preference="em")

    assert called["em"] == 1
    assert called["sina"] == 0
    assert list(df["symbol"]) == ["000001"]


def test_fetch_em_fills_missing_change_pct_amplitude(monkeypatch) -> None:
    """东财接口未返回涨跌额/涨跌幅/振幅时，由前收与 OHLC 推导。"""
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_daily",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("daily unavailable")),
    )
    monkeypatch.setattr(
        "src.data_fetcher.akshare_client.ak.stock_zh_a_hist",
        lambda symbol, **kwargs: pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-24", "2026-03-25"]),
                "开盘": [10.0, 10.5],
                "最高": [10.2, 10.7],
                "最低": [9.9, 10.3],
                "收盘": [10.1, 10.6],
                "成交量": [1000, 1100],
                "成交额": [10000, 11000],
                "换手率": [0.01, 0.02],
            }
        ),
    )

    df = fetch_a_share_daily("000001", "20260324", "20260325", source_preference="em")

    assert len(df) == 2
    assert pd.isna(df.iloc[0]["change"])
    assert pd.isna(df.iloc[0]["pct_chg"])
    assert pd.isna(df.iloc[0]["amplitude_pct"])
    assert abs(float(df.iloc[1]["change"]) - 0.5) < 1e-9
    assert abs(float(df.iloc[1]["pct_chg"]) - (0.5 / 10.1 * 100.0)) < 1e-6
    assert abs(float(df.iloc[1]["amplitude_pct"]) - ((10.7 - 10.3) / 10.1 * 100.0)) < 1e-6


def test_backfill_derived_daily_columns_in_duckdb(tmp_path) -> None:
    """库内 NULL 行由前收与 OHLC 回填。"""
    db_path = tmp_path / "market.duckdb"
    cfg_path = _write_config(tmp_path, db_path)
    with DuckDBManager(config_path=cfg_path) as db:
        db.connection.execute(
            """
            INSERT INTO a_share_daily VALUES
              ('000001', '2020-01-02', 10, 10.1, 10.2, 9.9, 1000, 10000, NULL, NULL, NULL, 0.01),
              ('000001', '2020-01-03', 10.1, 10.5, 10.8, 10.4, 1200, 12000, NULL, NULL, NULL, 0.02);
            """
        )
        fixed = db.backfill_derived_daily_columns()
        assert fixed == 1
        row = db.connection.execute(
            """
            SELECT change, pct_chg, amplitude_pct FROM a_share_daily
            WHERE trade_date = '2020-01-03'
            """
        ).fetchone()
        assert row is not None
        assert abs(float(row[0]) - 0.4) < 1e-9
        assert abs(float(row[1]) - (0.4 / 10.1 * 100.0)) < 1e-6
        assert abs(float(row[2]) - ((10.8 - 10.4) / 10.1 * 100.0)) < 1e-6


def test_duckdb_manager_auto_backfills_derived_on_init(tmp_path) -> None:
    db_path = tmp_path / "market.duckdb"
    cfg_path = _write_config(tmp_path, db_path)
    with DuckDBManager(config_path=cfg_path) as db:
        db.connection.execute(
            """
            INSERT INTO a_share_daily VALUES
              ('000001', '2020-01-02', 10, 10.0, 10.2, 9.8, 1000, 10000, NULL, NULL, NULL, 0.01),
              ('000001', '2020-01-03', 10.1, 10.4, 10.6, 10.0, 1200, 12000, NULL, NULL, NULL, 0.02);
            """
        )

    with DuckDBManager(config_path=cfg_path) as db2:
        row = db2.connection.execute(
            """
            SELECT change, pct_chg, amplitude_pct FROM a_share_daily
            WHERE symbol='000001' AND trade_date='2020-01-03'
            """
        ).fetchone()
        assert row is not None
        assert abs(float(row[0]) - 0.4) < 1e-9
        assert abs(float(row[1]) - (0.4 / 10.0 * 100.0)) < 1e-6
        assert abs(float(row[2]) - ((10.6 - 10.0) / 10.0 * 100.0)) < 1e-6


def test_duckdb_manager_creates_trade_date_symbol_index(tmp_path) -> None:
    db_path = tmp_path / "market.duckdb"
    cfg_path = _write_config(tmp_path, db_path)
    with DuckDBManager(config_path=cfg_path) as db:
        rows = db.connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name='a_share_daily'"
        ).fetchall()
    names = {str(r[0]) for r in rows}
    assert "idx_a_share_daily_trade_date_symbol" in names
