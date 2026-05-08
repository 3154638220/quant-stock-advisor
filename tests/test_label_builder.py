"""Tests for src/pipeline/label_builder.py - label construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.label_builder import (
    _daily_asset_return_matrix,
    _slugify_token,
    build_investable_period_return_panel,
    build_p1_monthly_investable_label,
    build_p1_training_label,
    select_rebalance_dates,
)

# ── _slugify_token ─────────────────────────────────────────────────────────


def test_slugify_basic():
    assert _slugify_token("Rank Fusion") == "rank_fusion"


def test_slugify_special_chars():
    assert _slugify_token("hello! world?") == "hello_world"


def test_slugify_numeric_input():
    result = _slugify_token(123)
    assert result == "123"


# ── build_p1_training_label ───────────────────────────────────────────────


def _make_label_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04", "2023-01-04"])
    return pd.DataFrame({
        "trade_date": dates,
        "symbol": ["000001", "000002", "000001", "000002"],
        "ret_5d": [0.02, -0.01, 0.03, 0.01],
        "ret_20d": [0.05, -0.02, 0.06, 0.02],
    })


def test_build_rank_fusion_label():
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel,
        label_columns=["ret_5d", "ret_20d"],
        label_weights=[0.5, 0.5],
        label_mode="rank_fusion",
        date_col="trade_date",
    )
    assert col == "forward_ret_fused"
    assert meta["label_mode"] == "rank_fusion"
    assert len(out) == len(panel)
    assert out[col].notna().all()


def test_build_top_bucket_rank_fusion():
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel,
        label_columns=["ret_5d", "ret_20d"],
        label_weights=[0.5, 0.5],
        label_mode="top_bucket_rank_fusion",
    )
    assert meta["label_mode"] == "top_bucket_rank_fusion"
    assert meta["label_top_bucket_quantile"] == 0.2
    assert not out.empty


def test_build_market_relative_label():
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel,
        label_columns=["ret_5d", "ret_20d"],
        label_weights=[0.5, 0.5],
        label_mode="market_relative",
    )
    assert meta["label_mode"] == "market_relative"
    assert meta["label_market_proxy"] == "same_date_cross_section_equal_weight"


def test_build_up_capture_market_relative():
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel,
        label_columns=["ret_5d", "ret_20d"],
        label_weights=[0.5, 0.5],
        label_mode="up_capture_market_relative",
    )
    assert meta["label_mode"] == "up_capture_market_relative"
    assert meta["label_up_capture_multiplier"] == 2.0


def test_build_raw_fusion_label():
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel,
        label_columns=["ret_5d", "ret_20d"],
        label_weights=[0.5, 0.5],
        label_mode="raw_fusion",
    )
    assert meta["label_mode"] == "raw_fusion"


def test_build_label_weights_normalized():
    """Weights should be normalized internally."""
    panel = _make_label_panel()
    out, col, meta = build_p1_training_label(
        panel, label_columns=["ret_5d", "ret_20d"],
        label_weights=[10.0, 10.0], label_mode="rank_fusion",
    )
    # Both equal weight -> should produce score near 0 (centered rank)
    assert meta["label_weights_normalized"] == "0.5,0.5"


def test_label_mode_invalid_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="label_mode"):
        build_p1_training_label(
            panel, label_columns=["ret_5d"], label_weights=[1.0],
            label_mode="invalid_mode",
        )


def test_label_columns_missing_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="缺少标签列"):
        build_p1_training_label(
            panel, label_columns=["nonexistent_col"], label_weights=[1.0],
        )


def test_label_weights_mismatch_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="长度不一致"):
        build_p1_training_label(
            panel, label_columns=["ret_5d", "ret_20d"],
            label_weights=[1.0],
        )


def test_label_weights_zero_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="非法"):
        build_p1_training_label(
            panel, label_columns=["ret_5d"], label_weights=[0.0],
        )


def test_label_weights_nan_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="非法"):
        build_p1_training_label(
            panel, label_columns=["ret_5d"], label_weights=[np.nan],
        )


def test_label_columns_empty_raises():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="不能为空"):
        build_p1_training_label(
            panel, label_columns=[], label_weights=[],
        )


def test_label_missing_date_col_raises():
    panel = pd.DataFrame({"ret_5d": [0.01], "symbol": ["000001"]})
    with pytest.raises(ValueError, match="缺少日期列"):
        build_p1_training_label(
            panel, label_columns=["ret_5d"], label_weights=[1.0],
            date_col="trade_date",
        )


# ── select_rebalance_dates ─────────────────────────────────────────────────


def test_select_rebalance_dates_monthly():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    result = select_rebalance_dates(dates, rebalance_rule="M")
    assert not result.empty
    assert "trade_date" in result.columns
    assert len(result) >= 11  # at least 11 month-end dates in a year


# ── _daily_asset_return_matrix ─────────────────────────────────────────────


def test_daily_asset_return_matrix_empty():
    result = _daily_asset_return_matrix(pd.DataFrame(), execution_mode="tplus1_open")
    assert result.empty


def test_daily_asset_return_matrix_close_to_close():
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-03", "2023-01-04"]),
        "symbol": ["000001", "000001", "000002", "000002"],
        "close": [10.0, 10.5, 20.0, 19.5],
    })
    result = _daily_asset_return_matrix(daily, execution_mode="close_to_close")
    assert not result.empty


def test_daily_asset_return_matrix_invalid_mode():
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03"]),
        "symbol": ["000001"],
        "close": [10.0],
    })
    with pytest.raises(ValueError, match="仅支持"):
        _daily_asset_return_matrix(daily, execution_mode="invalid")


# ── build_investable_period_return_panel ───────────────────────────────────


def test_investable_panel_missing_columns():
    panel = pd.DataFrame({"symbol": ["000001"]})
    with pytest.raises(ValueError, match="缺少列"):
        build_investable_period_return_panel(panel, pd.DataFrame(), rebalance_rule="M")


def test_investable_panel_empty_daily():
    panel = _make_label_panel()
    result = build_investable_period_return_panel(
        panel, pd.DataFrame(), rebalance_rule="M",
    )
    assert result.empty


def test_investable_panel_too_few_dates():
    panel = _make_label_panel()
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03"]),
        "symbol": ["000001"],
        "close": [10.0],
    })
    result = build_investable_period_return_panel(
        panel, daily, rebalance_rule="M",
    )
    assert result.empty


# ── build_p1_monthly_investable_label ─────────────────────────────────────


def test_monthly_investable_label_invalid_mode():
    panel = _make_label_panel()
    with pytest.raises(ValueError, match="monthly label_mode"):
        build_p1_monthly_investable_label(
            panel, pd.DataFrame(), rebalance_rule="M", label_mode="invalid",
        )


def test_monthly_investable_label_basic():
    panel = _make_label_panel()
    panel = panel.rename(columns={"trade_date": "trade_date"})
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-02-01"]),
        "symbol": ["000001", "000001", "000001"],
        "close": [10.0, 10.2, 10.5],
    })
    result, col, meta = build_p1_monthly_investable_label(
        panel, daily, rebalance_rule="M",
    )
    # May be empty due to date window matching, but should not error
    assert isinstance(col, str)
    assert meta["label_mode"] in (
        "monthly_investable", "monthly_investable_market_relative",
        "monthly_investable_up_capture_market_relative",
    )


# ═══════════════════════════════════════════════════════════════════════════
# P3-1 补充: market_relative 数值正确性 与 边界
# ═══════════════════════════════════════════════════════════════════════════


def test_market_relative_label_numeric_correctness():
    """market_relative 应减去当日截面等权均值。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "symbol": ["000001", "000002"],
        "ret_5d": [0.10, -0.04],
    })
    out, col, meta = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    # 截面均值 = (0.10 + (-0.04)) / 2 = 0.03
    # 000001: 0.10 - 0.03 = 0.07
    # 000002: -0.04 - 0.03 = -0.07
    mean = (0.10 + (-0.04)) / 2
    vals = out.set_index("symbol").sort_index()[col]
    assert vals["000001"] == pytest.approx(0.10 - mean)
    assert vals["000002"] == pytest.approx(-0.04 - mean)


def test_market_relative_single_stock():
    """单股票截面: market_ret = 自身值 → 结果应为 0。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03"]),
        "symbol": ["000001"],
        "ret_5d": [0.05],
    })
    out, col, meta = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    assert out[col].iloc[0] == pytest.approx(0.0)


def test_market_relative_balanced():
    """对称正负收益 → 截面均值为 0 → market_relative 等于原始值。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "symbol": ["000001", "000002"],
        "ret_5d": [0.08, -0.08],
    })
    out, col, meta = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    vals = out.set_index("symbol").sort_index()[col]
    assert vals["000001"] == pytest.approx(0.08)
    assert vals["000002"] == pytest.approx(-0.08)


def test_benchmark_relative_same_as_market_relative():
    """benchmark_relative 与 market_relative 在当前实现中行为一致（均用截面均值）。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "symbol": ["000001", "000002"],
        "ret_5d": [0.06, 0.02],
    })
    out_m, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    out_b, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="benchmark_relative",
    )
    pd.testing.assert_series_equal(
        out_m.sort_values("symbol")["forward_ret_fused"].reset_index(drop=True),
        out_b.sort_values("symbol")["forward_ret_fused"].reset_index(drop=True),
    )


def test_up_capture_multiplier_applied_when_market_up():
    """当截面均值为正时, up_capture 模式应放大 market_relative 信号。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "symbol": ["000001", "000002"],
        "ret_5d": [0.10, 0.02],  # 均值为正 (0.06)
    })
    out_up, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="up_capture_market_relative",
    )
    out_mr, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    mr_vals = out_mr.set_index("symbol").sort_index()["forward_ret_fused"]
    up_vals = out_up.set_index("symbol").sort_index()["forward_ret_fused"]
    # market_ret=0.06 > 0, 所以 up_capture = market_relative * 2
    for sym in ["000001", "000002"]:
        assert up_vals[sym] == pytest.approx(mr_vals[sym] * 2.0)


def test_up_capture_no_multiplier_when_market_down():
    """当截面均值为负时, up_capture 应等于 market_relative。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "symbol": ["000001", "000002"],
        "ret_5d": [-0.02, -0.08],  # 均值为负 (-0.05)
    })
    out_up, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="up_capture_market_relative",
    )
    out_mr, _, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    mr_vals = out_mr.set_index("symbol").sort_index()["forward_ret_fused"]
    up_vals = out_up.set_index("symbol").sort_index()["forward_ret_fused"]
    for sym in ["000001", "000002"]:
        assert up_vals[sym] == pytest.approx(mr_vals[sym])


def test_market_relative_multi_date():
    """多日期截面: 每日独立计算 market_relative。"""
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2023-01-03",
                                       "2023-01-04", "2023-01-04"]),
        "symbol": ["000001", "000002", "000001", "000002"],
        "ret_5d": [0.10, -0.02, 0.04, 0.06],
    })
    out, col, _ = build_p1_training_label(
        panel, label_columns=["ret_5d"], label_weights=[1.0],
        label_mode="market_relative",
    )
    # Day 1: mean=(0.10-0.02)/2=0.04 → [0.06, -0.06]
    # Day 2: mean=(0.04+0.06)/2=0.05 → [-0.01, 0.01]
    day1 = out[out["trade_date"] == pd.Timestamp("2023-01-03")].set_index("symbol").sort_index()[col]
    day2 = out[out["trade_date"] == pd.Timestamp("2023-01-04")].set_index("symbol").sort_index()[col]
    assert day1["000001"] == pytest.approx(0.10 - 0.04)
    assert day1["000002"] == pytest.approx(-0.02 - 0.04)
    assert day2["000001"] == pytest.approx(0.04 - 0.05)
    assert day2["000002"] == pytest.approx(0.06 - 0.05)
