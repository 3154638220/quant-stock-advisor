"""OOS 自动记录：在 M7 月度报告生成后，将预测超额写入 oos_tracking 表，
并尝试回填上月预测的实现值。

Usage (in run_monthly_selection_report.py main, after report generation):
    from src.monitoring.oos_auto_writer import record_oos_from_m7_report
    record_oos_from_m7_report(
        db_path=db_path,
        config_id="monthly_selection_u1_top20_m8_natural",
        signal_date="2026-05-05",
        predicted_excess_monthly=0.005,
        candidate_pool="U1_liquid_tradable",
        cost_bps=30.0,
        holdings=["000001", "000002", ...],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.monitoring.oos_tracker import OOSTracker

_LOG = logging.getLogger(__name__)


@dataclass
class OOSWriteResult:
    """OOS 自动写入结果。"""

    config_id: str
    signal_date: str
    predicted_written: bool
    realized_backfilled: bool = False
    backfilled_date: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    previous_prediction: Optional[float] = None
    current_prediction: Optional[float] = None


def record_oos_from_m7_report(
    db_path: str | Path,
    config_id: str,
    signal_date: str | date,
    *,
    predicted_excess_monthly: float,
    candidate_pool: str = "U1_liquid_tradable",
    top_k: int = 20,
    cost_bps: float = 30.0,
    holdings: Optional[Sequence[str]] = None,
    num_holdings: Optional[int] = None,
    limit_up_excluded: int = 0,
) -> OOSWriteResult:
    """从 M7 报告指标自动写入 OOS 预测记录，并尝试回填上月实现值。

    应在 M7 报告生成完成后调用，执行两个操作：
    1. 写入当前月的预测超额（predicted_excess_monthly）
    2. 若上月存在预测记录且尚无实现值，尝试用当前可用的数据回填

    Parameters
    ----------
    db_path: DuckDB 数据库路径
    config_id: Promoted 配置标识（如 "monthly_selection_u1_top20_m8_natural_soft_gamma0_20"）
    signal_date: 信号日期（本月截面）
    predicted_excess_monthly: 回测预测的月均超额（如 walk-forward 均值，以小数表示，0.005 = 0.5%）
    candidate_pool: 候选池标识
    top_k: 持仓数量
    cost_bps: 假设成本 (bps)
    holdings: 持仓标的列表
    num_holdings: 实际持仓数
    limit_up_excluded: 因涨停被排除的标的数

    Returns
    -------
    OOSWriteResult: 写入结果摘要。
    """
    result = OOSWriteResult(
        config_id=str(config_id),
        signal_date=str(signal_date),
        predicted_written=False,
    )

    try:
        tracker = OOSTracker(db_path)
    except Exception as e:
        result.errors.append(f"Failed to connect to OOS tracker: {e}")
        return result

    try:
        # 1. 写入当前月预测
        tracker.record_oos(
            config_id=config_id,
            signal_date=signal_date,
            top_k=top_k,
            candidate_pool=candidate_pool,
            cost_bps=cost_bps,
            predicted_excess_monthly=float(predicted_excess_monthly),
            realized_excess_monthly=None,  # 下月才可知
            holdings=list(holdings) if holdings else None,
            num_holdings=num_holdings,
            limit_up_excluded=limit_up_excluded,
        )
        result.predicted_written = True
        result.current_prediction = float(predicted_excess_monthly)

        # 2. 尝试回填上月预测的实现值
        result = _try_backfill_previous(tracker, config_id, result)

    except Exception as e:
        result.errors.append(f"Write failed: {e}")
    finally:
        tracker.close()

    return result


def _try_backfill_previous(
    tracker: OOSTracker,
    config_id: str,
    result: OOSWriteResult,
) -> OOSWriteResult:
    """查找上月尚未填写 realized_excess 的记录，尝试回填。

    回填需要数据库中存在该月的 forward return 数据，
    通过 holdings 中的标的 label_forward_1m_o2o_return 计算截面均值。
    """
    try:
        # 查找最近一条仅有预测、无实现的记录（且非本月）
        rows = tracker._conn.execute(
            f"""
            SELECT signal_date, predicted_excess_monthly, holding_returns_json
            FROM {tracker._table}
            WHERE config_id = ?
              AND predicted_excess_monthly IS NOT NULL
              AND realized_excess_monthly IS NULL
              AND signal_date < ?
            ORDER BY signal_date DESC
            LIMIT 1
            """,
            [str(config_id), str(result.signal_date)],
        ).fetchall()

        if not rows:
            return result

        prev_date, prev_pred, holdings_json = rows[0]
        result.previous_prediction = float(prev_pred) if prev_pred is not None else None

        # 尝试从 holdings 计算实现超额
        realized = _compute_realized_excess_from_holdings(
            tracker, str(prev_date), holdings_json
        )

        if realized is not None:
            tracker.update_realized(
                config_id=config_id,
                signal_date=str(prev_date),
                realized_excess_monthly=realized,
            )
            result.realized_backfilled = True
            result.backfilled_date = str(prev_date)
            _LOG.info(
                "OOS backfill: config=%s date=%s realized=%.4f%%",
                config_id, prev_date, realized * 100,
            )

    except Exception as e:
        result.errors.append(f"Backfill failed: {e}")

    return result


def _compute_realized_excess_from_holdings(
    tracker: OOSTracker,
    signal_date: str,
    holdings_json: Optional[str],
) -> Optional[float]:
    """从 a_share_daily 计算持仓的 T+1 开盘到次月开盘实现超额。

    计算逻辑：
    - buy_open = signal_date 之后第一个交易日的开盘价
    - sell_open = 次月第一个交易日的开盘价（即下次调仓买入日）
    - realized_return = sell_open / buy_open - 1
    - realized_excess = mean(holdings_returns) - market_ew_return
    """
    if not holdings_json:
        return None

    import json
    try:
        holdings_data = json.loads(holdings_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not holdings_data:
        return None

    if isinstance(holdings_data, dict):
        symbols = list(holdings_data.keys())
    elif isinstance(holdings_data, list):
        symbols = holdings_data
    else:
        return None

    symbols = [str(s).strip() for s in symbols if str(s).strip()]
    if not symbols:
        return None

    try:
        # 零填充为 6 位代码以匹配数据库格式
        sym_list = "', '".join(s.zfill(6) for s in symbols[:50])

        # signal_date 之后第一个交易日 = 买入日（开盘价买入）
        # 次月第一个交易日 = 卖出日（开盘价卖出）
        # sell_after = signal_date 所在月的下下个月 1 号
        sd = pd.Timestamp(signal_date)
        sell_cutoff = (sd + pd.DateOffset(months=2)).replace(day=1).strftime("%Y-%m-%d")

        df = tracker._conn.execute(
            f"""
            WITH
            ranked AS (
                SELECT symbol, trade_date, open,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) AS rn
                FROM a_share_daily
                WHERE symbol IN ('{sym_list}')
                  AND trade_date > '{signal_date}'
            ),
            buy_prices AS (
                SELECT symbol, open AS buy_open FROM ranked WHERE rn = 1
            ),
            sell_ranked AS (
                SELECT symbol, trade_date, open,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) AS rn
                FROM a_share_daily
                WHERE symbol IN ('{sym_list}')
                  AND trade_date >= '{sell_cutoff}'
            ),
            sell_prices AS (
                SELECT symbol, open AS sell_open FROM sell_ranked WHERE rn = 1
            )
            SELECT b.symbol, b.buy_open, s.sell_open,
                   (s.sell_open / NULLIF(b.buy_open, 0) - 1.0) AS realized_return
            FROM buy_prices b
            LEFT JOIN sell_prices s ON b.symbol = s.symbol
            WHERE b.buy_open > 0
            """,
        ).df()

        if df.empty:
            return None

        rets = pd.to_numeric(df["realized_return"], errors="coerce").dropna()
        if len(rets) < max(3, len(symbols) * 0.5):
            return None

        return float(rets.mean())

    except Exception:
        return None


def record_oos_batch_from_history(
    db_path: str | Path,
    config_id: str,
    history_df: pd.DataFrame,
    *,
    candidate_pool: str = "U1_liquid_tradable",
    top_k: int = 20,
    cost_bps: float = 30.0,
) -> list[OOSWriteResult]:
    """从历史回测结果批量写入 OOS 记录（用于建立 OOS 基线）。

    history_df 需含列: signal_date, predicted_excess_monthly, [realized_excess_monthly]

    用于 M10 完成后手工补录历史月份的预测 vs 实现超额。
    """
    results: list[OOSWriteResult] = []
    try:
        tracker = OOSTracker(db_path)
    except Exception as e:
        return [OOSWriteResult(
            config_id=config_id, signal_date="batch", predicted_written=False,
            errors=[f"Connection failed: {e}"],
        )]

    try:
        for _, row in history_df.iterrows():
            sd = str(row.get("signal_date", ""))
            if not sd:
                continue
            pred = float(row.get("predicted_excess_monthly", 0.0))
            real = row.get("realized_excess_monthly")
            real_val = float(real) if real is not None and not (
                isinstance(real, float) and np.isnan(real)
            ) else None

            r = OOSWriteResult(config_id=config_id, signal_date=sd, predicted_written=False)
            try:
                tracker.record_oos(
                    config_id=config_id,
                    signal_date=sd,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                    cost_bps=cost_bps,
                    predicted_excess_monthly=pred,
                    realized_excess_monthly=real_val,
                )
                r.predicted_written = True
                r.current_prediction = pred
            except Exception as e:
                r.errors.append(str(e))
            results.append(r)
    finally:
        tracker.close()

    return results
