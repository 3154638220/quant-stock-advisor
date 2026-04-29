from __future__ import annotations

import logging

import pandas as pd

from src.data_fetcher.fund_flow_client import FundFlowClient


def test_fetch_symbol_fund_flow_records_exception_reason(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise ConnectionError("upstream closed")

    monkeypatch.setattr("src.data_fetcher.fund_flow_client.requests.get", _boom)
    monkeypatch.setattr("src.data_fetcher.fund_flow_client.ak.stock_individual_fund_flow", _boom)

    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        out = client.fetch_symbol_fund_flow("600519")

        assert out.empty
        assert "600519" in client._last_fetch_errors
        assert "ConnectionError" in client._last_fetch_errors["600519"]


def test_update_symbols_logs_failed_examples(monkeypatch, tmp_path, caplog):
    def _empty(*args, **kwargs):
        return pd.DataFrame()

    def _h5_boom(*args, **kwargs):
        raise ConnectionError("h5 upstream closed")

    monkeypatch.setattr("src.data_fetcher.fund_flow_client.requests.get", _h5_boom)
    monkeypatch.setattr("src.data_fetcher.fund_flow_client.ak.stock_individual_fund_flow", _empty)

    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        with caplog.at_level(logging.WARNING):
            total = client.update_symbols(["600519", "000001"], sleep_sec=0.0, log_every=1)

        assert total == 0
        assert "资金流失败样例" in caplog.text
        assert "empty_response" in caplog.text


def test_table_row_count_reflects_upserted_rows(tmp_path):
    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        assert client.table_row_count() == 0
        df = pd.DataFrame(
            {
                "symbol": ["600519"],
                "trade_date": [pd.Timestamp("2026-01-02")],
                "close": [1500.0],
                "pct_chg": [1.2],
                "main_net_inflow": [1.0],
                "main_net_inflow_pct": [0.1],
                "super_large_net_inflow": [0.2],
                "super_large_net_inflow_pct": [0.02],
                "large_net_inflow": [0.3],
                "large_net_inflow_pct": [0.03],
                "medium_net_inflow": [0.4],
                "medium_net_inflow_pct": [0.04],
                "small_net_inflow": [-0.5],
                "small_net_inflow_pct": [-0.05],
                "source": ["test"],
                "fetched_at": [pd.Timestamp("2026-01-03 10:00:00")],
            }
        )
        client.upsert(df)
        assert client.table_row_count() == 1


def test_normalize_import_frame_accepts_common_aliases(tmp_path):
    raw = pd.DataFrame(
        {
            "code": ["600519", "000001"],
            "date": ["2026-01-02", "2026-01-03"],
            "收盘价": [1500.0, 12.3],
            "涨跌幅": [1.2, -0.5],
            "主力净流入-净额": [10.0, -5.0],
            "主力净流入-净占比": [0.11, -0.02],
            "超大单净流入-净额": [2.0, -1.0],
            "超大单净流入-净占比": [0.02, -0.01],
        }
    )

    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        out = client.normalize_import_frame(raw, source_label="vendor_x")

        assert list(out["symbol"]) == ["000001", "600519"]
        assert out["trade_date"].notna().all()
        assert out["source"].eq("vendor_x").all()
        assert {"main_net_inflow", "main_net_inflow_pct", "super_large_net_inflow"}.issubset(out.columns)


def test_import_file_csv_writes_rows(tmp_path):
    csv_path = tmp_path / "fund_flow_sample.csv"
    pd.DataFrame(
        {
            "代码": ["600519"],
            "日期": ["2026-01-02"],
            "主力净流入-净额": [10.0],
            "主力净流入-净占比": [0.11],
            "超大单净流入-净额": [2.0],
            "超大单净流入-净占比": [0.02],
        }
    ).to_csv(csv_path, index=False)

    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        count = client.import_file(csv_path, source_label="csv_vendor")
        saved = client.load_by_date_range(start_date="2026-01-01", end_date="2026-01-31")

        assert count == 1
        assert len(saved) == 1
        assert saved["symbol"].iloc[0] == "600519"
        assert saved["source"].iloc[0] == "csv_vendor"


def test_fetch_symbol_fund_flow_prefers_h5_history_and_backfills_pct(monkeypatch, tmp_path):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "rc": 0,
                "data": {
                    "klines": [
                        "2026-01-02,100.0,-20.0,-30.0,10.0,40.0,15.2,1.5",
                        "2026-01-03,50.0,-10.0,-20.0,5.0,25.0,15.6,2.0",
                    ]
                },
            }

    monkeypatch.setattr("src.data_fetcher.fund_flow_client.requests.get", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(
        "src.data_fetcher.fund_flow_client.ak.stock_individual_fund_flow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("AkShare fallback should not be used")),
    )

    with FundFlowClient(duckdb_path=str(tmp_path / "fund_flow.duckdb")) as client:
        client._conn.execute(
            """
            CREATE TABLE a_share_daily (
                symbol VARCHAR,
                trade_date DATE,
                amount DOUBLE
            )
            """
        )
        client._conn.execute(
            """
            INSERT INTO a_share_daily(symbol, trade_date, amount)
            VALUES
              ('600519', DATE '2026-01-02', 1000.0),
              ('600519', DATE '2026-01-03', 500.0)
            """
        )
        out = client.fetch_symbol_fund_flow("600519")

        assert len(out) == 2
        assert out["source"].eq("emdatah5_zjlx_history").all()
        first = out.loc[out["trade_date"] == pd.Timestamp("2026-01-02")].iloc[0]
        assert first["main_net_inflow"] == 100.0
        assert first["main_net_inflow_pct"] == 10.0
        assert first["super_large_net_inflow_pct"] == 4.0
        assert first["small_net_inflow_pct"] == -2.0
