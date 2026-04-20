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
    min_train_days: int = 252,
    expanding_window: bool = True,
) -> List[TimeSlice]:
    """
    时间顺序 K 段切片：将样本按时间均分为 n_splits 段测试窗。
    - expanding_window=True: 第 k 折训练集为测试段之前全部交易日（扩展窗）
    - expanding_window=False: 第 k 折训练集固定为测试段前 min_train_days 个交易日
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
        if bool(expanding_window):
            train_ix = idx[:a]
        else:
            train_start = max(0, a - int(min_train_days))
            train_ix = idx[train_start:a]
        test_ix = idx[a:b]
        if len(train_ix) < min_train_days or len(test_ix) < 1:
            continue
        out.append(TimeSlice(train_index=pd.DatetimeIndex(train_ix), test_index=pd.DatetimeIndex(test_ix), fold_id=fid))
        fid += 1
    return out


def rolling_walk_forward_windows(
    trading_index: pd.DatetimeIndex,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[TimeSlice]:
    """滚动 walk-forward：固定训练长度/测试长度，每次向前滑动 step_days。"""
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
        out.append(
            TimeSlice(
                train_index=pd.DatetimeIndex(idx[start:tr_end]),
                test_index=pd.DatetimeIndex(idx[tr_end:te_end]),
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
    """在 date_index 子区间上裁剪数据后调用 run_backtest。"""
    ix = pd.DatetimeIndex(pd.to_datetime(date_index).normalize())
    ar = asset_returns.reindex(ix).dropna(how="all")

    ws = weights_signal.copy()
    ws.index = pd.to_datetime(ws.index).normalize()
    ws_in = ws[ws.index.isin(ar.index)]
    if not ar.empty:
        start_dt = ar.index.min()
        ws_prev = ws[ws.index < start_dt]
        if not ws_prev.empty:
            seed = ws_prev.iloc[[-1]].copy()
            seed.index = pd.DatetimeIndex([start_dt])
            ws = pd.concat([seed, ws_in], axis=0)
            ws = ws[~ws.index.duplicated(keep="last")].sort_index()
        else:
            ws = ws_in
    else:
        ws = ws_in

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
    """在多个时间切片上分别回测，返回各折面板、明细表与聚合摘要。"""
    rows: List[Dict[str, Any]] = []
    panels: List[PerformancePanel] = []
    for sl in slices:
        sub_ix = sl.test_index if use_test_only else pd.DatetimeIndex(
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
    if panels:
        ann = np.array(
            [p.annualized_return for p in panels if np.isfinite(p.annualized_return)],
            dtype=np.float64,
        )
        if ann.size > 0:
            agg["median_ann_return"] = float(np.median(ann))
            agg["p25_ann_return"] = float(np.quantile(ann, 0.25))
            agg["p75_ann_return"] = float(np.quantile(ann, 0.75))
        else:
            agg["median_ann_return"] = float("nan")
            agg["p25_ann_return"] = float("nan")
            agg["p75_ann_return"] = float("nan")
    return panels, detail, agg


def compare_full_vs_slices(full_panel: PerformancePanel, slice_agg: Mapping[str, Any]) -> Dict[str, Any]:
    """将全样本面板与切片聚合对比，输出差值。"""
    out: Dict[str, Any] = {"full_sample": full_panel.to_dict(), "slices_agg": dict(slice_agg)}
    for k in ("annualized_return", "sharpe_ratio", "calmar_ratio", "max_drawdown", "win_rate"):
        fk = f"{k}_agg"
        if fk in slice_agg and np.isfinite(slice_agg[fk]):
            out[f"delta_{k}"] = float(full_panel.to_dict()[k]) - float(slice_agg[fk])
    return out
