"""
样本外 (OOS) 追踪器。

每月换仓后记录实际 Top20 持有收益，与历史回测预测超额对比，
计算"预测-实现超额差异"的滚动统计，并在连续不达标时触发降级告警。

用法::

    from src.monitoring.oos_tracker import OOSTracker

    tracker = OOSTracker(db_path="data/market.duckdb")
    tracker.record_oos(
        config_id="monthly_selection_u1_top20_m8_natural",
        signal_date="2026-04-28",
        top_k=20,
        candidate_pool="U1_liquid_tradable",
        cost_bps=30.0,
        predicted_excess_monthly=0.005,
        realized_excess_monthly=0.003,
        holdings=["600001", "000002"],
    )
    degradation = tracker.check_degradation(
        config_id="monthly_selection_u1_top20_m8_natural",
        min_months=3,
        threshold_ratio=0.5,
    )
    if degradation["degraded"]:
        print("WARNING: OOS degradation detected!")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

# ── duckdb 初始化 SQL（若未经过 migration 则创建表）──

OOS_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS oos_tracking (
    config_id VARCHAR NOT NULL,
    signal_date DATE NOT NULL,
    run_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    top_k INTEGER NOT NULL,
    candidate_pool VARCHAR NOT NULL,
    cost_bps DOUBLE NOT NULL,
    predicted_excess_monthly DOUBLE,
    realized_excess_monthly DOUBLE,
    holding_returns_json VARCHAR,
    benchmark_return DOUBLE,
    num_holdings INTEGER,
    limit_up_excluded INTEGER,
    PRIMARY KEY (config_id, signal_date, top_k, candidate_pool, cost_bps)
);
CREATE INDEX IF NOT EXISTS idx_oos_tracking_date ON oos_tracking(signal_date);
"""


# ── 数据结构 ─────────────────────────────────────────────────────────────

@dataclass
class OOSRecord:
    """单月 OOS 记录。"""

    config_id: str
    signal_date: str
    top_k: int
    candidate_pool: str
    cost_bps: float
    predicted_excess_monthly: Optional[float] = None
    realized_excess_monthly: Optional[float] = None
    benchmark_return: Optional[float] = None
    num_holdings: Optional[int] = None
    limit_up_excluded: Optional[int] = None
    holding_returns: Optional[dict[str, float]] = None


@dataclass
class OOSDegradationResult:
    """OOS 降级检测结果。"""

    config_id: str
    degraded: bool
    consecutive_degraded_months: int
    recent_realized_mean: float
    backtest_predicted_mean: float
    ratio: float
    threshold_ratio: float
    message: str


# ── 核心类 ───────────────────────────────────────────────────────────────

class OOSTracker:
    """
    样本外表现追踪器。

    Parameters
    ----------
    db_path
        DuckDB 数据库路径（与主库共用）。
    table_name
        OOS 追踪表名，默认 ``oos_tracking``。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        table_name: str = "oos_tracking",
    ) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute(OOS_TRACKING_DDL)
        self._table = table_name

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ── 写入 ──────────────────────────────────────────────────────────

    def record_oos(
        self,
        config_id: str,
        signal_date: str | date,
        *,
        top_k: int = 20,
        candidate_pool: str = "U1_liquid_tradable",
        cost_bps: float = 10.0,
        predicted_excess_monthly: Optional[float] = None,
        realized_excess_monthly: Optional[float] = None,
        holdings: Optional[Sequence[str]] = None,
        holding_returns: Optional[dict[str, float]] = None,
        benchmark_return: Optional[float] = None,
        num_holdings: Optional[int] = None,
        limit_up_excluded: int = 0,
    ) -> None:
        """
        记录一次月度换仓后的 OOS 实际表现。

        若 realized_excess 尚未可知（T+1 开盘换仓后需等 1 个月），
        可先写入 predicted_excess，待下月再补充 realized_excess。

        Parameters
        ----------
        config_id
            Promoted 配置标识。
        signal_date
            信号日（换仓决策日）。
        predicted_excess_monthly
            回测预测的月均超额（如 walk-forward 均值）。
        realized_excess_monthly
            实际持有的月均超额（label_forward_1m_o2o_return 的截面均值）。
        holdings
            持仓标的列表（可选，用于事后审计）。
        holding_returns
            各标的实际收益（可选）。
        benchmark_return
            基准同期收益（如 U1 候选池等权）。
        """
        holding_json = None
        if holding_returns:
            holding_json = json.dumps(holding_returns, ensure_ascii=False)
        elif holdings:
            holding_json = json.dumps(list(holdings), ensure_ascii=False)

        n_holdings = num_holdings or (len(holdings) if holdings else None)

        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._table}
                (config_id, signal_date, top_k, candidate_pool, cost_bps,
                 predicted_excess_monthly, realized_excess_monthly,
                 holding_returns_json, benchmark_return,
                 num_holdings, limit_up_excluded, run_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                str(config_id),
                str(signal_date),
                int(top_k),
                str(candidate_pool),
                float(cost_bps),
                float(predicted_excess_monthly) if predicted_excess_monthly is not None else None,
                float(realized_excess_monthly) if realized_excess_monthly is not None else None,
                holding_json,
                float(benchmark_return) if benchmark_return is not None else None,
                n_holdings,
                int(limit_up_excluded),
            ],
        )
        _LOG.info(
            "OOS record: config=%s date=%s pred=%.4f%% real=%.4f%%",
            config_id,
            signal_date,
            (predicted_excess_monthly or 0) * 100,
            (realized_excess_monthly or 0) * 100,
        )

    # ── 查询 ──────────────────────────────────────────────────────────

    def get_history(
        self,
        config_id: str,
        *,
        lookback_months: int | None = None,
    ) -> pd.DataFrame:
        """
        获取指定配置的 OOS 历史记录。

        Parameters
        ----------
        config_id
            配置标识。
        lookback_months
            仅返回最近 N 个月（None = 全部）。

        Returns
        -------
        DataFrame
            列：config_id, signal_date, predicted_excess_monthly,
            realized_excess_monthly, benchmark_return, ...
        """
        query = f"""
            SELECT config_id, signal_date, top_k, candidate_pool, cost_bps,
                   predicted_excess_monthly, realized_excess_monthly,
                   benchmark_return, num_holdings, limit_up_excluded,
                   run_ts
            FROM {self._table}
            WHERE config_id = ?
              AND realized_excess_monthly IS NOT NULL
            ORDER BY signal_date ASC
        """
        df = self._conn.execute(query, [str(config_id)]).df()
        if df.empty:
            return df

        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df = df.sort_values("signal_date")

        if lookback_months is not None and lookback_months > 0:
            cutoff = df["signal_date"].max() - pd.DateOffset(months=int(lookback_months))
            df = df[df["signal_date"] >= cutoff]

        return df

    def get_latest_predicted_excess(self, config_id: str) -> Optional[float]:
        """获取最近一个月换仓的预测超额。"""
        row = self._conn.execute(
            f"""
            SELECT predicted_excess_monthly
            FROM {self._table}
            WHERE config_id = ?
              AND predicted_excess_monthly IS NOT NULL
            ORDER BY signal_date DESC
            LIMIT 1
            """,
            [str(config_id)],
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def update_realized(
        self,
        config_id: str,
        signal_date: str | date,
        *,
        realized_excess_monthly: float,
        holding_returns: Optional[dict[str, float]] = None,
        benchmark_return: Optional[float] = None,
    ) -> bool:
        """
        补充更新某次换仓的实际实现超额（换仓后 1 个月可知）。

        Returns
        -------
        bool
            是否成功更新（找到对应记录）。
        """
        holding_json = (
            json.dumps(holding_returns, ensure_ascii=False)
            if holding_returns else None
        )
        result = self._conn.execute(
            f"""
            UPDATE {self._table}
            SET realized_excess_monthly = ?,
                holding_returns_json = COALESCE(?, holding_returns_json),
                benchmark_return = COALESCE(?, benchmark_return),
                run_ts = CURRENT_TIMESTAMP
            WHERE config_id = ? AND signal_date = ?
            """,
            [
                float(realized_excess_monthly),
                holding_json,
                float(benchmark_return) if benchmark_return is not None else None,
                str(config_id),
                str(signal_date),
            ],
        )
        updated = result.fetchall()[0][0] if hasattr(result, 'fetchall') else 0
        return bool(updated)

    # ── 降级检测 ──────────────────────────────────────────────────────

    def check_degradation(
        self,
        config_id: str,
        *,
        min_months: int = 3,
        threshold_ratio: float = 0.5,
        lookback_months: int | None = None,
    ) -> OOSDegradationResult:
        """
        检测 OOS 降级：连续 min_months 个月实现超额低于回测预测均值的 threshold_ratio。

        这是 Promotion Gate 的 OOS 监控触发器。

        Parameters
        ----------
        config_id
            要检查的配置。
        min_months
            连续不达标月数的触发阈值（默认 3）。
        threshold_ratio
            实现超额 / 预测超额的比率下限（默认 0.5，即低于预测的 50% 触发）。
        lookback_months
            用于计算"回测预测均值"的历史窗口（None = 全部）。

        Returns
        -------
        OOSDegradationResult
        """
        df = self.get_history(config_id, lookback_months=lookback_months)
        if df.empty or len(df) < min_months:
            return OOSDegradationResult(
                config_id=config_id,
                degraded=False,
                consecutive_degraded_months=0,
                recent_realized_mean=0.0,
                backtest_predicted_mean=0.0,
                ratio=1.0,
                threshold_ratio=threshold_ratio,
                message=f"数据不足（需要至少 {min_months} 个月，当前 {len(df)} 个月）",
            )

        pred_mean = float(df["predicted_excess_monthly"].mean())
        realized = df["realized_excess_monthly"].values

        # 检查最近 min_months 个月是否全部低于阈值
        recent_realized = realized[-min_months:]
        ratio = float(np.mean(recent_realized) / pred_mean) if abs(pred_mean) > 1e-8 else 0.0

        # 同时检查连续不达标月数
        consecutive = 0
        for r in reversed(realized):
            if abs(pred_mean) > 1e-8 and (r / pred_mean) < threshold_ratio:
                consecutive += 1
            else:
                break

        degraded = consecutive >= min_months

        message = (
            f"OOS DEGRADED: {config_id} 连续 {consecutive} 个月 "
            f"实现超额 / 预测超额 < {threshold_ratio:.0%} "
            f"(预测均值={pred_mean*100:.2f}%, 近期实现均值={np.mean(recent_realized)*100:.2f}%, "
            f"比率={ratio:.2%})"
        ) if degraded else (
            f"OOS OK: {config_id} 连续不达标 {consecutive}/{min_months} 个月 "
            f"(预测均值={pred_mean*100:.2f}%, 近期实现均值={np.mean(recent_realized)*100:.2f}%)"
        )

        _LOG.info(message)
        return OOSDegradationResult(
            config_id=config_id,
            degraded=degraded,
            consecutive_degraded_months=consecutive,
            recent_realized_mean=float(np.mean(recent_realized)),
            backtest_predicted_mean=float(pred_mean),
            ratio=ratio,
            threshold_ratio=threshold_ratio,
            message=message,
        )

    def summary(
        self,
        config_id: str | None = None,
        *,
        last_n_months: int = 6,
    ) -> pd.DataFrame:
        """
        生成 OOS 汇总统计表。

        Parameters
        ----------
        config_id
            配置标识。None = 所有配置汇总。
        last_n_months
            近期窗口月数。

        Returns
        -------
        DataFrame
        """
        params: list = []
        if config_id:
            params.append(str(config_id))

        df = self._conn.execute(
            f"""
            SELECT config_id,
                   COUNT(*) as n_months,
                   MIN(signal_date) as first_date,
                   MAX(signal_date) as last_date,
                   AVG(predicted_excess_monthly) as avg_predicted,
                   AVG(realized_excess_monthly) as avg_realized,
                   AVG(realized_excess_monthly - predicted_excess_monthly) as avg_excess_diff,
                   AVG(benchmark_return) as avg_benchmark
            FROM {self._table}
            WHERE realized_excess_monthly IS NOT NULL
              {('AND config_id = ?' if config_id else '')}
            GROUP BY config_id
            ORDER BY avg_realized DESC
            """,
            params,
        ).df()

        if df.empty:
            return df

        # 近期统计
        for col in ["avg_predicted", "avg_realized", "avg_excess_diff", "avg_benchmark"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ── 便捷函数 ─────────────────────────────────────────────────────────────

def record_monthly_oos(
    db_path: str | Path,
    config_id: str,
    signal_date: str | date,
    *,
    predicted_excess_monthly: Optional[float] = None,
    realized_excess_monthly: Optional[float] = None,
    holdings: Optional[Sequence[str]] = None,
    holding_returns: Optional[dict[str, float]] = None,
    benchmark_return: Optional[float] = None,
    top_k: int = 20,
    candidate_pool: str = "U1_liquid_tradable",
    cost_bps: float = 10.0,
    num_holdings: Optional[int] = None,
    limit_up_excluded: int = 0,
) -> None:
    """
    便捷函数：记录一次月度 OOS 表现，自动管理 Tracker 生命周期。

    适合在月度报告脚本末尾调用。
    """
    tracker = OOSTracker(db_path)
    try:
        tracker.record_oos(
            config_id=config_id,
            signal_date=signal_date,
            top_k=top_k,
            candidate_pool=candidate_pool,
            cost_bps=cost_bps,
            predicted_excess_monthly=predicted_excess_monthly,
            realized_excess_monthly=realized_excess_monthly,
            holdings=holdings,
            holding_returns=holding_returns,
            benchmark_return=benchmark_return,
            num_holdings=num_holdings,
            limit_up_excluded=limit_up_excluded,
        )
    finally:
        tracker.close()
