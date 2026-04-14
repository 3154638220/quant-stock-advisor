"""
因子 IC 衰减滚动监控：持久化记录与告警。

功能：
- 将每次计算得到的 IC 序列（逐日截面 IC）追加写入 JSON 存储文件。
- 按滚动窗口统计近期 IC 均值，当均值绝对值跌破阈值时触发告警。
- JSON 格式便于离线分析，也可直接读入 DuckDB。

典型用法（每日运行结束后调用）::

    from src.features.ic_monitor import ICMonitor

    monitor = ICMonitor(store_path="data/logs/ic_monitor.json")
    monitor.append(factor_name="momentum", ic_series=ic_ser)
    alerts = monitor.check_decay_alerts(window=20, threshold=0.03)
    for a in alerts:
        print(a)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ICRecord:
    """单条 IC 记录：某因子在某交易日的截面 IC 值。"""

    factor: str
    trade_date: str   # ISO-8601，如 "2026-01-02"
    ic: float
    recorded_at: str  # 写入时间戳（ISO-8601）


@dataclass
class ICDecayAlert:
    """IC 衰减告警。"""

    factor: str
    window: int
    threshold: float
    rolling_mean_ic: float
    check_date: str
    message: str

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# 存储 I/O
# ---------------------------------------------------------------------------

def _load_store(store_path: Path) -> List[Dict]:
    if not store_path.exists():
        return []
    try:
        with store_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("IC 监控存储读取失败（将重建）: %s", e)
    return []


def _save_store(records: List[Dict], store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = store_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    tmp.replace(store_path)


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

class ICMonitor:
    """
    因子 IC 持久化监控器。

    Parameters
    ----------
    store_path
        JSON 存储文件路径，不存在时自动创建。
    """

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def append(
        self,
        factor_name: str,
        ic_series: pd.Series,
        *,
        overwrite_dates: bool = False,
    ) -> int:
        """
        将 IC 序列追加到存储文件。

        Parameters
        ----------
        factor_name
            因子名称，如 ``"momentum"``。
        ic_series
            逐日 IC 序列，索引为交易日（``str`` 或 ``datetime``）。
        overwrite_dates
            若为 True，则覆盖已存在的同因子同日期记录；默认追加跳过重复。

        Returns
        -------
        int
            实际写入的新记录数。
        """
        records = _load_store(self.store_path)
        now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        existing: set[tuple[str, str]] = set()
        if not overwrite_dates:
            for r in records:
                existing.add((r.get("factor", ""), r.get("trade_date", "")))

        new_rows: list[Dict] = []
        for dt_raw, ic_val in ic_series.items():
            if pd.isna(ic_val):
                continue
            dt_str = (
                dt_raw.isoformat()
                if isinstance(dt_raw, (date, datetime))
                else str(dt_raw)
            )
            key = (factor_name, dt_str)
            if not overwrite_dates and key in existing:
                continue
            new_rows.append(
                asdict(
                    ICRecord(
                        factor=factor_name,
                        trade_date=dt_str,
                        ic=float(ic_val),
                        recorded_at=now_str,
                    )
                )
            )

        if new_rows:
            if overwrite_dates:
                keep = [
                    r
                    for r in records
                    if (r.get("factor"), r.get("trade_date"))
                    not in {(x["factor"], x["trade_date"]) for x in new_rows}
                ]
                records = keep + new_rows
            else:
                records.extend(new_rows)
            _save_store(records, self.store_path)
            logger.debug(
                "IC 监控：因子 %s 写入 %d 条新记录 -> %s",
                factor_name,
                len(new_rows),
                self.store_path,
            )
        return len(new_rows)

    def append_many(
        self,
        factor_ic_map: Dict[str, pd.Series],
        *,
        overwrite_dates: bool = False,
    ) -> Dict[str, int]:
        """批量写入多个因子的 IC 序列。"""
        return {
            name: self.append(name, ser, overwrite_dates=overwrite_dates)
            for name, ser in factor_ic_map.items()
        }

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        factors: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        读取存储为 DataFrame，列：``factor``、``trade_date``、``ic``、``recorded_at``。

        Parameters
        ----------
        factors
            若指定，仅返回对应因子的记录；``None`` 表示全部。
        """
        records = _load_store(self.store_path)
        if not records:
            return pd.DataFrame(
                columns=["factor", "trade_date", "ic", "recorded_at"]
            )
        df = pd.DataFrame(records)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["ic"] = pd.to_numeric(df["ic"], errors="coerce")
        if factors is not None:
            df = df[df["factor"].isin(factors)]
        return df.sort_values(["factor", "trade_date"]).reset_index(drop=True)

    def rolling_ic_stats(
        self,
        window: int = 20,
        *,
        factors: Optional[Sequence[str]] = None,
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        对每个因子计算近 ``window`` 条记录的滚动均值/标准差/IR。

        Returns
        -------
        DataFrame
            列：``factor``、``trade_date``、``ic``、
            ``roll_mean``、``roll_std``、``roll_ir``。
        """
        if window < 2:
            raise ValueError("window 须 >= 2")
        mp = min_periods or max(2, window // 2)
        df = self.load_dataframe(factors=factors)
        if df.empty:
            return df

        parts = []
        for fname, grp in df.groupby("factor"):
            g = grp.sort_values("trade_date").copy()
            g["roll_mean"] = (
                g["ic"].rolling(window=window, min_periods=mp).mean()
            )
            g["roll_std"] = (
                g["ic"].rolling(window=window, min_periods=mp).std(ddof=0)
            )
            g["roll_ir"] = g["roll_mean"] / g["roll_std"].replace(0, np.nan)
            parts.append(g)

        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # 告警
    # ------------------------------------------------------------------

    def check_decay_alerts(
        self,
        *,
        window: int = 20,
        threshold: float = 0.03,
        factors: Optional[Sequence[str]] = None,
        check_date: Optional[str] = None,
    ) -> List[ICDecayAlert]:
        """
        检查最新滚动 IC 均值是否低于阈值，若低则生成告警。

        Parameters
        ----------
        window
            用于计算滚动均值的观测数（交易日数）。
        threshold
            |滚动 IC 均值| 低于此值时告警（通常建议 0.02~0.05）。
        factors
            指定要检查的因子子集；默认检查全部。
        check_date
            用于日志显示的检查日期；``None`` 则用今日。

        Returns
        -------
        告警列表；无问题时为空列表。
        """
        today = check_date or date.today().isoformat()
        stats = self.rolling_ic_stats(window=window, factors=factors)
        if stats.empty:
            return []

        alerts: list[ICDecayAlert] = []
        for fname, grp in stats.groupby("factor"):
            latest = grp.sort_values("trade_date").iloc[-1]
            roll_mean = latest.get("roll_mean")
            if roll_mean is None or np.isnan(roll_mean):
                continue
            if abs(float(roll_mean)) < threshold:
                msg = (
                    f"[IC告警] 因子 {fname!r} 近 {window} 期滚动 IC 均值 "
                    f"{roll_mean:.4f} 低于阈值 {threshold}（检查日: {today}）"
                )
                logger.warning(msg)
                alerts.append(
                    ICDecayAlert(
                        factor=str(fname),
                        window=window,
                        threshold=threshold,
                        rolling_mean_ic=float(roll_mean),
                        check_date=today,
                        message=msg,
                    )
                )
        return alerts

    def summary(
        self,
        *,
        factors: Optional[Sequence[str]] = None,
        tail_window: int = 20,
    ) -> pd.DataFrame:
        """
        各因子 IC 的全局与近期统计汇总表。

        Returns
        -------
        DataFrame
            列：``factor``、``n_obs``、``ic_mean``、``ic_std``、``ic_ir``、
            ``ic_hit_rate``（正 IC 比例）、
            ``recent_mean``（近 tail_window 期均值）。
        """
        df = self.load_dataframe(factors=factors)
        if df.empty:
            return pd.DataFrame()

        rows = []
        for fname, grp in df.groupby("factor"):
            ic = grp["ic"].dropna()
            n = int(len(ic))
            m = float(ic.mean()) if n else np.nan
            s = float(ic.std(ddof=0)) if n > 1 else np.nan
            ir = m / s if s and not np.isnan(s) and abs(s) > 1e-15 else np.nan
            hit = float((ic > 0).mean()) if n else np.nan
            recent = float(ic.tail(tail_window).mean()) if n else np.nan
            rows.append(
                {
                    "factor": fname,
                    "n_obs": n,
                    "ic_mean": round(m, 6) if not np.isnan(m) else np.nan,
                    "ic_std": round(s, 6) if not np.isnan(s) else np.nan,
                    "ic_ir": round(ir, 4) if not np.isnan(ir) else np.nan,
                    "ic_hit_rate": round(hit, 4) if not np.isnan(hit) else np.nan,
                    f"recent_{tail_window}_mean": (
                        round(recent, 6) if not np.isnan(recent) else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def compute_and_persist_factor_ic(
    df: pd.DataFrame,
    factor_col: str,
    forward_ret_col: str,
    store_path: str | Path,
    *,
    date_col: str = "trade_date",
    method: str = "spearman",
    alert_window: int = 20,
    alert_threshold: float = 0.03,
    overwrite_dates: bool = False,
) -> tuple[pd.Series, List[ICDecayAlert]]:
    """
    一步完成「计算截面 IC → 持久化 → 告警检查」。

    Parameters
    ----------
    df
        含 ``date_col``、``factor_col``、``forward_ret_col`` 的长表。
    factor_col
        因子列名。
    forward_ret_col
        前瞻收益列名。
    store_path
        IC 监控 JSON 文件路径。
    method
        ``"spearman"``（RankIC）或 ``"pearson"``（IC）。
    alert_window
        滚动告警窗口（观测数）。
    alert_threshold
        告警阈值（|滚动 IC 均值| < threshold 时告警）。

    Returns
    -------
    (ic_series, alerts)
    """
    from .factor_eval import information_coefficient

    ic_ser = information_coefficient(
        df,
        factor_col=factor_col,
        forward_ret_col=forward_ret_col,
        date_col=date_col,
        method=method,  # type: ignore[arg-type]
    )

    monitor = ICMonitor(store_path)
    monitor.append(factor_col, ic_ser, overwrite_dates=overwrite_dates)
    alerts = monitor.check_decay_alerts(
        window=alert_window,
        threshold=alert_threshold,
        factors=[factor_col],
    )
    return ic_ser, alerts
