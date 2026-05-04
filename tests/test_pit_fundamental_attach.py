from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.features.fundamental_factors import pit_safe_fundamental_rows

# A3: 核心函数已迁入 src/，消除对已删除脚本的依赖
from src.pipeline.backtest_runner import _attach_pit_fundamentals

# _attach_pit_roe_ttm 随 scripts/run_p2_regime_aware_dual_sleeve_v1.py 删除而退役；
# 其功能已被 _attach_pit_fundamentals（通用 PIT 基本面附着）覆盖。
# 对应测试已改写为使用 _attach_pit_fundamentals（见下方 test_attach_pit_roe_ttm_*）。
_attach_pit_roe_ttm = None  # deprecated, replaced by _attach_pit_fundamentals


def test_pit_safe_fundamental_rows_requires_real_notice_lag_for_statements():
    raw = pd.DataFrame(
        {
            "report_period": pd.to_datetime(["2025-12-31", "2025-12-31", "2025-12-31"]),
            "announcement_date": pd.to_datetime(["2025-12-31", "2026-03-15", "2025-12-31"]),
            "source": [
                "stock_financial_analysis_indicator",
                "stock_financial_analysis_indicator_em",
                "stock_value_em",
            ],
        }
    )

    assert pit_safe_fundamental_rows(raw).tolist() == [False, True, True]


def test_attach_pit_fundamentals_uses_latest_announcement_by_trade_date(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "pit.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_fundamental (
                symbol VARCHAR,
                report_period DATE,
                announcement_date DATE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                ev_ebitda DOUBLE,
                roe_ttm DOUBLE,
                net_profit_yoy DOUBLE,
                gross_margin_change DOUBLE,
                debt_to_assets_change DOUBLE,
                ocf_to_net_profit DOUBLE,
                ocf_to_asset DOUBLE,
                gross_margin_delta DOUBLE,
                asset_turnover DOUBLE,
                net_margin_stability DOUBLE,
                northbound_net_inflow DOUBLE,
                margin_buy_ratio DOUBLE
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_fundamental VALUES
            ('000001', DATE '2025-09-30', DATE '2025-10-31', 10, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
            ('000001', DATE '2025-12-31', DATE '2025-12-31', 999, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
            ('000001', DATE '2025-12-31', DATE '2026-03-15', 30, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
            ('000002', DATE '2025-09-30', DATE '2025-11-05', 20, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000001", "000002"],
            "trade_date": pd.to_datetime(["2026-02-28", "2026-02-28", "2026-03-31", "2026-03-31"]).tz_localize(None),
            "log_market_cap": [10.0, 11.0, 10.5, 11.5],
        }
    )
    # 确保 trade_date dtype 与 DuckDB 返回一致（ns → us 避免 merge_asof 不兼容）
    factors["trade_date"] = factors["trade_date"].astype("datetime64[us]")

    # A3: preprocess_fundamental_cross_section 已迁入 src.pipeline.backtest_runner
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.preprocess_fundamental_cross_section",
        lambda df, **_: df,
    )

    out = (
        _attach_pit_fundamentals(factors, str(db_path))
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )

    assert out["pe_ttm"].tolist() == [10.0, 20.0, 30.0, 20.0]


def test_attach_pit_roe_ttm_filters_statement_rows_without_real_notice_lag(tmp_path, monkeypatch):
    """原 _attach_pit_roe_ttm 测试，改写为使用通用 _attach_pit_fundamentals。

    _attach_pit_roe_ttm 随 scripts/run_p2_regime_aware_dual_sleeve_v1.py 删除而退役，
    但其 PIT 过滤逻辑已被 _attach_pit_fundamentals → pit_safe_fundamental_rows 覆盖。
    """
    db_path = Path(tmp_path) / "pit_roe.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE a_share_fundamental (
                symbol VARCHAR,
                report_period DATE,
                announcement_date DATE,
                roe_ttm DOUBLE,
                source VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO a_share_fundamental VALUES
            ('000001', DATE '2025-09-30', DATE '2025-10-31', 1, 'stock_financial_analysis_indicator_em'),
            ('000001', DATE '2025-12-31', DATE '2025-12-31', 99, 'stock_financial_analysis_indicator'),
            ('000001', DATE '2025-12-31', DATE '2026-03-15', 3, 'stock_financial_analysis_indicator_em')
            """
        )
    finally:
        con.close()

    factors = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "trade_date": pd.to_datetime(["2026-02-28", "2026-03-31"]).tz_localize(None),
            "log_market_cap": [10.0, 10.5],
        }
    )
    # 确保 trade_date dtype 与 DuckDB 兼容
    factors["trade_date"] = factors["trade_date"].astype("datetime64[us]")

    # Mock preprocessing to pass raw values through for assertion
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.preprocess_fundamental_cross_section",
        lambda df, **_: df,
    )

    out = _attach_pit_fundamentals(factors, str(db_path)).sort_values("trade_date").reset_index(drop=True)

    # PIT: 2026-02-28 应取 2025-10-31 公告的 roe_ttm=1（2025-12-31 的 stock_financial_analysis_indicator
    # 被 pit_safe_fundamental_rows 过滤掉）。2026-03-31 应取 2026-03-15 公告的 roe_ttm=3。
    assert out["roe_ttm"].tolist() == [1.0, 3.0]
