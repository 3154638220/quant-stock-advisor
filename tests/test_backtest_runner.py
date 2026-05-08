"""Tests for src/pipeline/backtest_runner.py — 编排层边界与 fallback 路径。

覆盖重点（P3-1）：
- 因子缓存命中/失效/刷新逻辑
- _attach_pit_fundamentals 的空数据与缺失表 fallback
- load_industry_map 的文件不存在/列名变体
- build_market_ew_benchmark 与 build_regime_weight_overrides 的空输入
- build_topk_weights 的空截面/单标的/行业上限边界
- _pick_topk_with_industry_cap 的行业约束边界
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.pipeline.backtest_runner import (
    _attach_pit_fundamentals,
    _pick_topk_with_industry_cap,
    _select_topk_with_holding_buffer,
    build_asset_returns,
    build_market_ew_benchmark,
    build_market_ew_open_to_open_benchmark,
    build_regime_weight_overrides,
    build_symbol_benchmark,
    build_topk_weights,
    load_daily_from_duckdb,
    load_industry_map,
    prepare_factors_for_backtest,
)
from src.pipeline.factor_cache import (
    load_prepared_factors_cache,
    write_prepared_factors_cache,
)

# ── 辅助工厂函数 ─────────────────────────────────────────────────────────


def _make_daily_db(db_path: str, rows: list[dict] | None = None) -> None:
    """在 DuckDB 中创建 a_share_daily 表并写入测试数据。"""
    con = duckdb.connect(db_path)
    try:
        con.execute(
            """CREATE TABLE a_share_daily (
                symbol VARCHAR, trade_date DATE,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                volume DOUBLE, turnover DOUBLE, pct_chg DOUBLE, amount DOUBLE
            )"""
        )
        if rows:
            for r in rows:
                con.execute(
                    "INSERT INTO a_share_daily VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [str(r["symbol"]), r["trade_date"], r["open"], r["high"],
                     r["low"], r["close"], r.get("volume", 1e6), r.get("turnover", 5e7),
                     r.get("pct_chg", 0.0), r.get("amount", r["close"] * r.get("volume", 1e6))],
                )
    finally:
        con.close()


def _make_mini_daily_df(n_symbols: int = 3, n_days: int = 200) -> pd.DataFrame:
    """生成最小日线 DataFrame（供不需要 DB 的函数使用）。"""
    np.random.seed(42)
    syms = [f"{i:06d}" for i in range(1, n_symbols + 1)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rows = []
    for sym in syms:
        close = 10.0 + np.cumsum(np.random.randn(n_days) * 0.1)
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": sym, "trade_date": dt,
                "open": close[i] * 0.99, "high": close[i] * 1.02,
                "low": close[i] * 0.98, "close": close[i],
                "volume": 1e6, "turnover": 5e7, "pct_chg": 0.0,
                "amount": close[i] * 1e6,
            })
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _make_mini_factors(n_symbols: int = 5, n_dates: int = 3) -> pd.DataFrame:
    """生成含基本因子列的最小 DataFrame。"""
    syms = [f"{i:06d}" for i in range(1, n_symbols + 1)]
    dates = pd.date_range("2023-06-01", periods=n_dates, freq="MS")
    rows = []
    for dt in dates:
        for sym in syms:
            rows.append({
                "symbol": sym, "trade_date": dt,
                "momentum": np.random.randn() * 0.1,
                "rsi": 50 + np.random.randn() * 10,
                "realized_vol": 0.2 + np.random.random() * 0.1,
                "log_market_cap": 10 + np.random.random(),
                "turnover_roll_mean": 1e7,
                "price_position": np.random.random(),
                "limit_move_hits_5d": 0.0,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# load_daily_from_duckdb
# ═══════════════════════════════════════════════════════════════════════════


def test_load_daily_from_duckdb_basic(tmp_path):
    db_path = str(tmp_path / "test.db")
    dates = pd.date_range("2023-01-03", "2023-01-20", freq="B")
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"symbol": "000001", "trade_date": dt.date(),
                     "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.0 + i * 0.1})
    _make_daily_db(db_path, rows)
    df = load_daily_from_duckdb(db_path, "2023-01-03", "2023-01-20", lookback_days=60)
    assert not df.empty
    assert "close" in df.columns
    assert df["symbol"].iloc[0] == "000001"


def test_load_daily_from_duckdb_empty_range(tmp_path):
    db_path = str(tmp_path / "test_empty.db")
    _make_daily_db(db_path, [
        {"symbol": "000001", "trade_date": "2022-01-03", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2},
    ])
    # 查询范围在数据范围之外 → 返回空 DataFrame（不抛异常）
    df = load_daily_from_duckdb(db_path, "2024-01-01", "2024-01-10", lookback_days=60)
    # 空结果或只有列头
    assert isinstance(df, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════
# _attach_pit_fundamentals — 空数据 & 缺失表 fallback
# ═══════════════════════════════════════════════════════════════════════════


def test_attach_pit_fundamentals_no_fundamental_table(tmp_path, monkeypatch):
    """DuckDB 中没有 a_share_fundamental 表时应返回原 factors 不变。"""
    db_path = str(tmp_path / "nofund.db")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE dummy (x INT)")
    con.close()

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.preprocess_fundamental_cross_section",
        lambda df, **_: df,
    )
    factors = _make_mini_factors(3, 2)
    result = _attach_pit_fundamentals(factors, db_path)
    # 应返回原数据（列不变）
    for col in factors.columns:
        assert col in result.columns


def test_attach_pit_fundamentals_empty_fund_table(tmp_path, monkeypatch):
    """a_share_fundamental 存在但为空。"""
    db_path = str(tmp_path / "emptyfund.db")
    con = duckdb.connect(db_path)
    con.execute(
        """CREATE TABLE a_share_fundamental (
            symbol VARCHAR, report_period DATE, announcement_date DATE,
            pe_ttm DOUBLE, pb DOUBLE, roe_ttm DOUBLE
        )"""
    )
    con.close()

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.preprocess_fundamental_cross_section",
        lambda df, **_: df,
    )
    factors = _make_mini_factors(3, 2)
    result = _attach_pit_fundamentals(factors, db_path)
    assert "pe_ttm" in result.columns or len(result) == len(factors)


def test_attach_pit_fundamentals_all_fund_rows_filtered(tmp_path, monkeypatch):
    """所有 fundamental 行被 pit_safe_fundamental_rows 过滤后不崩溃。"""
    db_path = str(tmp_path / "allfiltered.db")
    con = duckdb.connect(db_path)
    con.execute(
        """CREATE TABLE a_share_fundamental (
            symbol VARCHAR, report_period DATE, announcement_date DATE,
            pe_ttm DOUBLE, pb DOUBLE, roe_ttm DOUBLE
        )"""
    )
    # 插入一个会被过滤的行（source=stock_financial_analysis_indicator 且 report_period=announcement_date）
    con.execute(
        """INSERT INTO a_share_fundamental VALUES
           ('000001', DATE '2025-12-31', DATE '2025-12-31', 10, 1, NULL)"""
    )
    con.close()

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.preprocess_fundamental_cross_section",
        lambda df, **_: df,
    )
    factors = _make_mini_factors(2, 1)
    result = _attach_pit_fundamentals(factors, db_path)
    assert not result.empty


# ═══════════════════════════════════════════════════════════════════════════
# prepare_factors_for_backtest — 缓存命中/失效/刷新
# ═══════════════════════════════════════════════════════════════════════════


def test_prepare_factors_cache_write_and_hit(tmp_path, monkeypatch):
    """写缓存后可命中，第二次调用返回 cache_hit=True。"""
    cache_path = tmp_path / "factors.parquet"
    daily = _make_mini_daily_df(3, 200)
    db_path = str(tmp_path / "cachetest.db")
    _make_daily_db(db_path)

    # Mock 重量级依赖 让 prepare_factors_for_backtest 只测缓存逻辑
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.compute_factors",
        lambda daily_df, min_hist_days: _make_mini_factors(3, 5),
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner._attach_pit_fundamentals",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_fund_flow",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_shareholder_factors",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_universe_filter",
        lambda factors, daily_df, enabled, min_amount_20d, require_roe_ttm_positive: factors,
    )

    meta = {"start_date": "2023-01-01", "end_date": "2023-12-31", "version": 1}

    # 第一次：冷启动，应写入缓存
    result1, hit1 = prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=False, cache_meta=meta,
    )
    assert not hit1
    assert cache_path.exists()
    assert cache_path.with_suffix(".parquet.meta.json").exists()

    # 第二次：应命中缓存
    result2, hit2 = prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=False, cache_meta=meta,
    )
    assert hit2
    assert len(result2) == len(result1)


def test_prepare_factors_cache_refresh_flag(tmp_path, monkeypatch):
    """refresh_cache=True 时应跳过缓存读取。"""
    cache_path = tmp_path / "refresh.parquet"
    daily = _make_mini_daily_df(3, 200)
    db_path = str(tmp_path / "refreshtest.db")
    _make_daily_db(db_path)

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.compute_factors",
        lambda daily_df, min_hist_days: _make_mini_factors(3, 5),
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner._attach_pit_fundamentals",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_fund_flow",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_shareholder_factors",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_universe_filter",
        lambda factors, daily_df, enabled, min_amount_20d, require_roe_ttm_positive: factors,
    )

    meta = {"start_date": "2023-01-01", "end_date": "2023-12-31"}

    # 先写入缓存
    prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=False, cache_meta=meta,
    )

    # refresh_cache=True 应穿透缓存
    call_count = [0]
    orig = prepare_factors_for_backtest

    _, hit = prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=True, cache_meta=meta,
    )
    assert not hit


def test_prepare_factors_cache_meta_mismatch(tmp_path, monkeypatch):
    """缓存 meta 不匹配时应判定为 cache miss。"""
    cache_path = tmp_path / "mismatch.parquet"
    daily = _make_mini_daily_df(3, 200)
    db_path = str(tmp_path / "mismatchtest.db")
    _make_daily_db(db_path)

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.compute_factors",
        lambda daily_df, min_hist_days: _make_mini_factors(3, 5),
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner._attach_pit_fundamentals",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_fund_flow",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_shareholder_factors",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_universe_filter",
        lambda factors, daily_df, enabled, min_amount_20d, require_roe_ttm_positive: factors,
    )

    # 写缓存时用 meta_v1
    prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=False,
        cache_meta={"start_date": "2023-01-01", "version": 1},
    )
    assert cache_path.exists()

    # 用不同的 meta 去读 → 应 miss
    result, hit = prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=cache_path, refresh_cache=False,
        cache_meta={"start_date": "2023-06-01", "version": 2},
    )
    assert not hit


def test_prepare_factors_no_cache_path(tmp_path, monkeypatch):
    """cache_path=None 时不读写缓存直接计算。"""
    daily = _make_mini_daily_df(3, 200)
    db_path = str(tmp_path / "nocachetest.db")
    _make_daily_db(db_path)

    monkeypatch.setattr(
        "src.pipeline.backtest_runner.compute_factors",
        lambda daily_df, min_hist_days: _make_mini_factors(3, 5),
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner._attach_pit_fundamentals",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_fund_flow",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_shareholder_factors",
        lambda factors, db_path: factors,
    )
    monkeypatch.setattr(
        "src.pipeline.backtest_runner.attach_universe_filter",
        lambda factors, daily_df, enabled, min_amount_20d, require_roe_ttm_positive: factors,
    )

    result, hit = prepare_factors_for_backtest(
        daily, min_hist_days=130, db_path=db_path, results_dir=tmp_path,
        universe_filter_cfg={"enabled": False},
        cache_path=None, refresh_cache=False, cache_meta={},
    )
    assert not hit
    assert not result.empty


# ═══════════════════════════════════════════════════════════════════════════
# load_industry_map
# ═══════════════════════════════════════════════════════════════════════════


def test_load_industry_map_valid_csv(tmp_path, monkeypatch):
    p = tmp_path / "ind.csv"
    p.write_text("symbol,industry\n000001,银行\n000002,地产\n", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.backtest_runner.PROJECT_ROOT", tmp_path)
    result = load_industry_map(str(p))
    assert result == {"000001": "银行", "000002": "地产"}


def test_load_industry_map_nonexistent_file():
    result = load_industry_map("/nonexistent/path/ind.csv")
    assert result == {}


def test_load_industry_map_chinese_column_names(tmp_path, monkeypatch):
    p = tmp_path / "ind_cn.csv"
    p.write_text("代码,行业\n000001,银行\n000002,地产\n", encoding="utf-8-sig")
    monkeypatch.setattr("src.pipeline.backtest_runner.PROJECT_ROOT", tmp_path)
    result = load_industry_map(str(p))
    assert result == {"000001": "银行", "000002": "地产"}


def test_load_industry_map_dedup_keeps_last(tmp_path, monkeypatch):
    p = tmp_path / "ind_dup.csv"
    p.write_text("symbol,industry\n000001,银行\n000001,金融\n", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.backtest_runner.PROJECT_ROOT", tmp_path)
    result = load_industry_map(str(p))
    assert result["000001"] == "金融"


def test_load_industry_map_filters_invalid(tmp_path, monkeypatch):
    p = tmp_path / "ind_bad.csv"
    p.write_text("symbol,industry\n12345,银行\n,地产\n000002,\n", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.backtest_runner.PROJECT_ROOT", tmp_path)
    result = load_industry_map(str(p))
    # 12345 len=5 被过滤；空 symbol 被过滤；空 industry 被过滤
    assert "12345" not in result
    assert "" not in result
    assert "000002" not in result


# ═══════════════════════════════════════════════════════════════════════════
# build_market_ew_benchmark
# ═══════════════════════════════════════════════════════════════════════════


def test_build_market_ew_benchmark_empty_df():
    empty = pd.DataFrame(columns=["symbol", "trade_date", "close"])
    result = build_market_ew_benchmark(empty, "2023-01-01", "2023-12-31")
    assert result.empty


def test_build_market_ew_benchmark_basic():
    daily = _make_mini_daily_df(3, 300)
    result = build_market_ew_benchmark(daily, "2023-03-01", "2023-10-31", min_days=100)
    assert not result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_build_market_ew_benchmark_min_days_fallback():
    """当所有股票都不满足 min_days 时，应回退到全部股票。"""
    daily = _make_mini_daily_df(2, 50)
    result = build_market_ew_benchmark(daily, "2023-01-01", "2023-03-15", min_days=500)
    # 50 < 500 但仍应有输出（fallback 到全部）
    assert not result.empty


# ═══════════════════════════════════════════════════════════════════════════
# build_market_ew_open_to_open_benchmark
# ═══════════════════════════════════════════════════════════════════════════


def test_build_market_ew_o2o_empty():
    empty = pd.DataFrame(columns=["symbol", "trade_date", "open"])
    result = build_market_ew_open_to_open_benchmark(empty, "2023-01-01", "2023-12-31")
    assert result.empty


def test_build_market_ew_o2o_basic():
    daily = _make_mini_daily_df(3, 250)
    result = build_market_ew_open_to_open_benchmark(daily, "2023-03-01", "2023-09-30", min_days=100)
    assert not result.empty


# ═══════════════════════════════════════════════════════════════════════════
# build_symbol_benchmark
# ═══════════════════════════════════════════════════════════════════════════


def test_build_symbol_benchmark_nonexistent():
    daily = _make_mini_daily_df(2, 100)
    result = build_symbol_benchmark(daily, "999999", "2023-01-01", "2023-06-30")
    assert result.empty


def test_build_symbol_benchmark_basic():
    daily = _make_mini_daily_df(2, 100)
    result = build_symbol_benchmark(daily, "000001", "2023-01-01", "2023-06-30")
    assert not result.empty


# ═══════════════════════════════════════════════════════════════════════════
# build_asset_returns
# ═══════════════════════════════════════════════════════════════════════════


def test_build_asset_returns_empty_symbols():
    daily = _make_mini_daily_df(3, 100)
    result = build_asset_returns(daily, [], "2023-01-01", "2023-06-30")
    assert result.empty


def test_build_asset_returns_basic():
    daily = _make_mini_daily_df(3, 200)
    result = build_asset_returns(daily, ["000001", "000002"], "2023-01-01", "2023-06-30")
    assert not result.empty
    assert "000001" in result.columns or result.shape[1] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# build_regime_weight_overrides
# ═══════════════════════════════════════════════════════════════════════════


def test_regime_overrides_empty_factors():
    empty = pd.DataFrame(columns=["trade_date", "symbol"])
    daily = _make_mini_daily_df(3, 300)
    from src.market.regime import MARKET_EW_PROXY
    config = {"bull_return_threshold": 0.05, "bear_return_threshold": 0.04,
              "vol_threshold_ann": 0.25, "short_window": 20, "long_window": 60}
    overrides, df = build_regime_weight_overrides(
        empty, daily, {"A": 0.5, "B": 0.5},
        benchmark_symbol=MARKET_EW_PROXY, regime_cfg_raw=config,
    )
    assert overrides == {}
    assert df.empty


def test_regime_overrides_market_ew_proxy():
    from src.market.regime import MARKET_EW_PROXY
    daily = _make_mini_daily_df(5, 500)
    factors = daily[["symbol", "trade_date"]].drop_duplicates().copy()
    config = {"bull_return_threshold": 0.05, "bear_return_threshold": 0.04,
              "vol_threshold_ann": 0.25, "short_window": 20, "long_window": 60}
    overrides, df = build_regime_weight_overrides(
        factors, daily, {"A": 0.5, "B": 0.5},
        benchmark_symbol=MARKET_EW_PROXY, regime_cfg_raw=config,
    )
    # 函数不应崩溃，返回合法类型
    assert isinstance(overrides, dict)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "trade_date" in df.columns
        assert "regime" in df.columns


# ═══════════════════════════════════════════════════════════════════════════
# _pick_topk_with_industry_cap
# ═══════════════════════════════════════════════════════════════════════════


def test_pick_topk_no_industry_map():
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 21)],
        "score": np.linspace(0.1, 1.0, 20),
    })
    result = _pick_topk_with_industry_cap(df, top_k=10, industry_map=None, industry_cap_count=None)
    assert len(result) == 10
    # 应返回最高分的前 10 只
    assert result["score"].max() == 1.0


def test_pick_topk_with_industry_cap():
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 21)],
        "score": np.linspace(0.1, 1.0, 20),
    })
    industry_map = {f"{i:06d}": "银行" if i <= 5 else "科技" for i in range(1, 21)}
    # cap=3 → 每个行业最多 3 只
    result = _pick_topk_with_industry_cap(df, top_k=10, industry_map=industry_map, industry_cap_count=3)
    assert len(result) <= 10
    # 银行最多 3 只
    bank_count = sum(1 for s in result["symbol"] if industry_map.get(str(s)) == "银行")
    assert bank_count <= 3


def test_pick_topk_single_stock():
    df = pd.DataFrame({"symbol": ["000001"], "score": [0.8]})
    result = _pick_topk_with_industry_cap(df, top_k=10, industry_map=None, industry_cap_count=None)
    assert len(result) == 1


def test_pick_topk_fewer_than_k():
    """候选不足 top_k 时返回所有候选。"""
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 6)],
        "score": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    result = _pick_topk_with_industry_cap(df, top_k=10, industry_map=None, industry_cap_count=None)
    assert len(result) == 5


def test_pick_topk_unknown_industry_no_cap():
    """_UNKNOWN_ 行业不受 cap 限制。"""
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 15)],
        "score": np.linspace(0.1, 1.0, 14),
    })
    # 所有股票都映射到未知行业
    industry_map = {f"{i:06d}": "_UNKNOWN_" for i in range(1, 15)}
    result = _pick_topk_with_industry_cap(df, top_k=10, industry_map=industry_map, industry_cap_count=2)
    # _UNKNOWN_ 不设上限
    assert len(result) == 10


# ═══════════════════════════════════════════════════════════════════════════
# _select_topk_with_holding_buffer
# ═══════════════════════════════════════════════════════════════════════════


def test_select_topk_buffer_no_prev_holdings():
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 21)],
        "score": np.linspace(0.1, 1.0, 20),
    })
    result = _select_topk_with_holding_buffer(
        df, top_k=10, entry_top_k=15, hold_buffer_top_k=15,
        prev_holdings=set(), industry_map=None, industry_cap_count=None,
    )
    assert len(result) == 10


def test_select_topk_buffer_with_prev_holdings():
    df = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 31)],
        "score": sorted(np.random.randn(30) * 0.1 + 0.5, reverse=True),
    })
    # 前 5 只是 hold
    prev = {f"{i:06d}" for i in range(1, 6)}
    result = _select_topk_with_holding_buffer(
        df, top_k=10, entry_top_k=15, hold_buffer_top_k=15,
        prev_holdings=prev, industry_map=None, industry_cap_count=None,
    )
    assert 1 <= len(result) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# build_topk_weights — 空数据 / 边界
# ═══════════════════════════════════════════════════════════════════════════


def test_build_topk_weights_empty_score():
    empty_score = pd.DataFrame(columns=["symbol", "trade_date", "score"])
    factors = _make_mini_factors(5, 3)
    daily = _make_mini_daily_df(5, 200)
    with pytest.raises(RuntimeError, match="权重"):
        build_topk_weights(empty_score, factors, daily, top_k=10, rebalance_rule="M",
                          prefilter_cfg={"enabled": False}, max_turnover=1.0)


def test_build_topk_weights_single_date():
    """只有一天 score 数据 → 应生成一组权重。"""
    np.random.seed(42)
    syms = [f"{i:06d}" for i in range(1, 31)]
    score_df = pd.DataFrame({
        "symbol": syms,
        "trade_date": pd.Timestamp("2023-06-30"),
        "score": np.linspace(0.1, 1.0, len(syms)),
    })
    factors = _make_mini_factors(30, 5)
    daily = _make_mini_daily_df(30, 300)
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": True, "limit_move_max": 2,
                       "turnover_low_pct": 0.10, "turnover_high_pct": 0.98,
                       "price_position_high_pct": 0.90},
        max_turnover=1.0, portfolio_method="equal_weight",
    )
    assert not result.empty
    assert result.shape[1] <= 10  # top_k=10


def test_build_topk_weights_tiered_equal():
    syms = [f"{i:06d}" for i in range(1, 31)]
    score_df = pd.DataFrame({
        "symbol": syms,
        "trade_date": pd.Timestamp("2023-06-30"),
        "score": np.linspace(0.1, 1.0, len(syms)),
    })
    factors = _make_mini_factors(30, 5)
    daily = _make_mini_daily_df(30, 300)
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": False}, max_turnover=1.0,
        portfolio_method="tiered_equal_weight",
        top_tier_count=5, top_tier_weight_share=0.6,
    )
    assert not result.empty


def test_build_topk_weights_with_holding_buffer():
    syms = [f"{i:06d}" for i in range(1, 31)]
    dates = [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-31")]
    rows = []
    for dt in dates:
        for s in syms:
            rows.append({"symbol": s, "trade_date": dt, "score": np.random.random()})
    score_df = pd.DataFrame(rows)
    factors = _make_mini_factors(30, 5)
    daily = _make_mini_daily_df(30, 300)
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": False}, max_turnover=1.0,
        entry_top_k=15, hold_buffer_top_k=15,
        portfolio_method="equal_weight",
    )
    assert not result.empty


def test_build_topk_weights_with_industry_cap():
    syms = [f"{i:06d}" for i in range(1, 31)]
    score_df = pd.DataFrame({
        "symbol": syms,
        "trade_date": pd.Timestamp("2023-06-30"),
        "score": np.linspace(0.1, 1.0, len(syms)),
    })
    factors = _make_mini_factors(30, 5)
    daily = _make_mini_daily_df(30, 300)
    industry_map = {f"{i:06d}": "银行" if i <= 10 else ("科技" if i <= 20 else "消费") for i in range(1, 31)}
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": False}, max_turnover=1.0,
        industry_map=industry_map, industry_cap_count=3,
        portfolio_method="equal_weight",
    )
    assert not result.empty


def test_build_topk_weights_with_turnover_constraint():
    """max_turnover=0.5 时，月度间换手应受限。"""
    syms = [f"{i:06d}" for i in range(1, 31)]
    dates = [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-31")]
    rows = []
    for dt in dates:
        for s in syms:
            rows.append({"symbol": s, "trade_date": dt, "score": np.random.random()})
    score_df = pd.DataFrame(rows)
    factors = _make_mini_factors(30, 5)
    daily = _make_mini_daily_df(30, 300)
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": False}, max_turnover=0.5,
        portfolio_method="equal_weight",
    )
    assert not result.empty
    # 两个月调仓，权重矩阵应有 2 行（或 1 行若日期过滤）
    assert 1 <= len(result) <= 2


def test_build_topk_weights_return_details():
    syms = [f"{i:06d}" for i in range(1, 21)]
    score_df = pd.DataFrame({
        "symbol": syms,
        "trade_date": pd.Timestamp("2023-06-30"),
        "score": np.linspace(0.1, 1.0, len(syms)),
    })
    factors = _make_mini_factors(20, 5)
    daily = _make_mini_daily_df(20, 200)
    result = build_topk_weights(
        score_df, factors, daily, top_k=10, rebalance_rule="M",
        prefilter_cfg={"enabled": False}, max_turnover=1.0,
        portfolio_method="equal_weight", return_details=True,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    w_wide, diag_detail, diag_summary = result
    assert not w_wide.empty
    assert not diag_detail.empty
    assert isinstance(diag_summary, dict)
    assert diag_summary["portfolio_method"] == "equal_weight"
