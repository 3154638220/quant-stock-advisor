"""Walk-forward / 时间切片验证：避免仅依赖全样本绩效。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtest.performance_panel import PerformancePanel, aggregate_walk_forward_panels


@dataclass(frozen=True)
class TimeSlice:
    """单个切片：训练窗与测试窗（含端点在内的交易日索引）。"""

    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    fold_id: int


def contiguous_time_splits(
    trading_index: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    min_train_days: int = 20,
) -> List[TimeSlice]:
    """
    时间顺序 K 段切片：将样本按时间均分为 ``n_splits`` 段测试窗；
    第 k 折的训练集为**该测试段之前**的所有交易日（扩展窗），测试集为第 k 段。

    首段若训练样本不足 ``min_train_days`` 则跳过。
    """
    if n_splits < 2:
        raise ValueError("n_splits 须 >= 2")
    idx = pd.DatetimeIndex(pd.to_datetime(trading_index).normalize()).sort_values().unique()
    n = len(idx)
    if n < n_splits + min_train_days:
        raise ValueError("交易日过少，无法切片")

    bounds = [int(round(i)) for i in np.linspace(0, n, n_splits + 1)]
    out: List[TimeSlice] = []
    fid = 0
    for k in range(n_splits):
        a, b = bounds[k], bounds[k + 1]
        if b <= a:
            continue
        train_ix = idx[:a]
        test_ix = idx[a:b]
        if len(train_ix) < min_train_days or len(test_ix) < 1:
            continue
        out.append(
            TimeSlice(
                train_index=pd.DatetimeIndex(train_ix),
                test_index=pd.DatetimeIndex(test_ix),
                fold_id=fid,
            )
        )
        fid += 1
    return out


def rolling_walk_forward_windows(
    trading_index: pd.DatetimeIndex,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[TimeSlice]:
    """
    滚动 walk-forward：固定训练长度 ``train_days``、测试 ``test_days``，每次向前滑动 ``step_days``。

    所有长度均以**交易日个数**计（非日历日）。
    """
    if train_days < 5 or test_days < 1 or step_days < 1:
        raise ValueError("train_days/test_days/step_days 不合法")
    idx = pd.DatetimeIndex(pd.to_datetime(trading_index).normalize()).sort_values().unique()
    n = len(idx)
    out: List[TimeSlice] = []
    start = 0
    fid = 0
    while True:
        tr_end = start + train_days
        te_end = tr_end + test_days
        if te_end > n:
            break
        train_ix = idx[start:tr_end]
        test_ix = idx[tr_end:te_end]
        out.append(
            TimeSlice(
                train_index=pd.DatetimeIndex(train_ix),
                test_index=pd.DatetimeIndex(test_ix),
                fold_id=fid,
            )
        )
        fid += 1
        start += step_days
    return out


def run_backtest_on_index(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    *,
    config: Optional[BacktestConfig] = None,
    rebalance_rule: Optional[str] = None,
) -> BacktestResult:
    """在 ``date_index`` 子区间上裁剪数据后调用 ``run_backtest``。"""
    ix = pd.DatetimeIndex(pd.to_datetime(date_index).normalize())
    ar = asset_returns.reindex(ix).dropna(how="all")
    # 权重：仅保留落在区间内的调仓行
    ws = weights_signal.copy()
    ws.index = pd.to_datetime(ws.index).normalize()
    ws = ws[ws.index.isin(ar.index)]
    if ar.empty or ws.empty:
        raise ValueError("切片内无重叠的收益或权重")
    return run_backtest(ar, ws, config=config, rebalance_rule=rebalance_rule)


def walk_forward_backtest(
    asset_returns: pd.DataFrame,
    weights_signal: pd.DataFrame,
    slices: Sequence[TimeSlice],
    *,
    config: Optional[BacktestConfig] = None,
    rebalance_rule: Optional[str] = None,
    use_test_only: bool = True,
) -> Tuple[List[PerformancePanel], pd.DataFrame, Dict[str, Any]]:
    """
    在多个时间切片上分别回测，返回各折 ``PerformancePanel``、明细表与聚合摘要。

    use_test_only
        为 True 时仅在每折的 **测试窗** 上回测（典型 walk-forward OOS）；
        为 False 时在 **训练+测试** 全段上回测（用于对照）。
    """
    rows: List[Dict[str, Any]] = []
    panels: List[PerformancePanel] = []

    for sl in slices:
        if use_test_only:
            sub_ix = sl.test_index
        else:
            sub_ix = pd.DatetimeIndex(
                np.unique(np.concatenate([sl.train_index.values, sl.test_index.values]))
            ).sort_values()
        try:
            res = run_backtest_on_index(
                asset_returns,
                weights_signal,
                sub_ix,
                config=config,
                rebalance_rule=rebalance_rule,
            )
        except ValueError:
            continue
        panels.append(res.panel)
        d = res.panel.to_dict()
        d["fold_id"] = sl.fold_id
        d["n_train"] = len(sl.train_index)
        d["n_test"] = len(sl.test_index)
        rows.append(d)

    detail = pd.DataFrame(rows)
    agg = aggregate_walk_forward_panels(panels, method="mean") if panels else {"n_folds": 0}
    return panels, detail, agg


def compare_full_vs_slices(
    full_panel: PerformancePanel,
    slice_agg: Mapping[str, Any],
) -> Dict[str, Any]:
    """将全样本面板与切片聚合对比，突出过拟合风险（仅数值对比，无判断逻辑）。"""
    out: Dict[str, Any] = {"full_sample": full_panel.to_dict(), "slices_agg": dict(slice_agg)}
    for k in (
        "annualized_return",
        "sharpe_ratio",
        "calmar_ratio",
        "max_drawdown",
        "win_rate",
    ):
        fk = f"{k}_agg"
        if fk in slice_agg and np.isfinite(slice_agg[fk]):
            out[f"delta_{k}"] = float(full_panel.to_dict()[k]) - float(slice_agg[fk])
    return out
