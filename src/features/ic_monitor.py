"""
因子 IC 衰减滚动监控：持久化记录与告警。

功能：
- 将每次计算得到的 IC 序列（逐日截面 IC）追加写入 DuckDB 或 JSON 存储文件。
- 按滚动窗口统计近期 IC 均值，当均值绝对值跌破阈值时触发告警。
- P2-8: DuckDB 主存储（事务保证并发安全），JSONL 保留为可选的向后兼容导出。

典型用法（每日运行结束后调用）::

    from src.features.ic_monitor import ICMonitor

    monitor = ICMonitor(store_path="data/logs/ic_monitor.json", db_path="data/market.duckdb")
    monitor.append(factor_name="momentum", ic_series=ic_ser)
    alerts = monitor.check_decay_alerts(window=20, threshold=0.03,
                                        alert_handler=my_webhook_handler)
    for a in alerts:
        print(a)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DuckDB 建表 DDL
# ---------------------------------------------------------------------------

IC_MONITOR_DDL = """
CREATE TABLE IF NOT EXISTS ic_monitor (
    factor       VARCHAR,
    trade_date   DATE,
    ic           DOUBLE,
    recorded_at  TIMESTAMP,
    PRIMARY KEY (factor, trade_date)
);
"""


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
# 存储 I/O (JSONL，向后兼容)
# ---------------------------------------------------------------------------

def _load_store(store_path: Path) -> List[Dict]:
    """JSONL 格式：每行一条 JSON 记录，流式解析。"""
    if not store_path.exists():
        return []
    records: List[Dict] = []
    try:
        for line in store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("IC 监控存储读取失败（将重建）: %s", e)
        return []
    return records


def _save_store(records: List[Dict], store_path: Path) -> None:
    """JSONL 格式追加写入：每条记录一行，原子写入。"""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = store_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(store_path)


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

class ICMonitor:
    """
    因子 IC 持久化监控器。

    P2-8: 支持 DuckDB 主存储（事务保证并发安全），JSONL 作为向后兼容的
    可选导出格式。支持告警回调（如企业微信 Webhook）。

    Parameters
    ----------
    store_path
        JSON 存储文件路径（向后兼容），不存在时自动创建。
        若仅使用 DuckDB 后端可设为 ``None``。
    db_path
        DuckDB 数据库路径。若提供，优先使用 DuckDB 存储。
        默认为 ``None``，仅使用 JSONL。
    """

    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        *,
        db_path: Optional[str | Path] = None,
    ) -> None:
        self.store_path = Path(store_path) if store_path else None
        self._db_path = Path(db_path) if db_path else None
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))
            self._conn.execute(IC_MONITOR_DDL)

    def _ensure_db(self) -> Optional[duckdb.DuckDBPyConnection]:
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return self._conn
            except Exception:
                # 连接断开，重连
                self._conn = duckdb.connect(str(self._db_path))
                self._conn.execute(IC_MONITOR_DDL)
                return self._conn
        return None

    def close(self) -> None:
        """关闭 DuckDB 连接。"""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

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
        将 IC 序列追加到存储（DuckDB 优先，JSONL 作为兼容导出）。

        P2-8: DuckDB 事务保证并发安全，并发写入不丢失数据。

        Parameters
        ----------
        factor_name
            因子名称，如 ``"momentum"``。
        ic_series
            逐日 IC 序列，索引为交易日（``str`` 或 ``datetime``）。
        overwrite_dates
            若为 True，则覆盖已存在的同因子同日期记录；默认追加。

        Returns
        -------
        int
            实际写入的新记录数。
        """
        now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        new_rows: list[Dict] = []
        for dt_raw, ic_val in ic_series.items():
            if pd.isna(ic_val):
                continue
            dt_str = (
                dt_raw.isoformat()
                if isinstance(dt_raw, (date, datetime))
                else str(dt_raw)
            )
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

        if not new_rows:
            return 0

        # P2-8: DuckDB 优先写入（事务保证原子性）
        db = self._ensure_db()
        if db is not None:
            try:
                df_new = pd.DataFrame(new_rows)
                df_new["trade_date"] = pd.to_datetime(df_new["trade_date"], errors="coerce")
                df_new["recorded_at"] = pd.to_datetime(df_new["recorded_at"], errors="coerce")
                if overwrite_dates:
                    # 删除旧记录后写入
                    dates_to_del = df_new["trade_date"].dropna().dt.date.unique().tolist()
                    if dates_to_del:
                        placeholders = ",".join(["?"] * len(dates_to_del))
                        db.execute(
                            f"DELETE FROM ic_monitor WHERE factor = ? AND trade_date IN ({placeholders})",
                            [factor_name] + dates_to_del,
                        )
                db.execute("INSERT OR REPLACE INTO ic_monitor SELECT * FROM df_new")
            except Exception as exc:
                logger.warning("DuckDB IC 写入失败，回退 JSONL: %s", exc)
                db = None  # 回退 JSONL

        # JSONL 兼容写入（DuckDB 不可用时或作为额外导出）
        if db is None and self.store_path is not None:
            if overwrite_dates:
                records = _load_store(self.store_path)
                new_keys = {(x["factor"], x["trade_date"]) for x in new_rows}
                keep = [
                    r
                    for r in records
                    if (r.get("factor"), r.get("trade_date")) not in new_keys
                ]
                records = keep + new_rows
                _save_store(records, self.store_path)
            else:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                with self.store_path.open("a", encoding="utf-8") as fh:
                    for row in new_rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.debug(
            "IC 监控：因子 %s 写入 %d 条新记录",
            factor_name,
            len(new_rows),
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

        P2-8: 优先从 DuckDB 读取；若不可用则回退 JSONL。

        Parameters
        ----------
        factors
            若指定，仅返回对应因子的记录；``None`` 表示全部。
        """
        # P2-8: 优先 DuckDB
        db = self._ensure_db()
        if db is not None:
            try:
                if factors:
                    placeholders = ",".join(["?"] * len(list(factors)))
                    df = db.execute(
                        f"SELECT * FROM ic_monitor WHERE factor IN ({placeholders}) ORDER BY factor, trade_date",
                        list(factors),
                    ).df()
                else:
                    df = db.execute("SELECT * FROM ic_monitor ORDER BY factor, trade_date").df()
                if not df.empty:
                    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                    df["ic"] = pd.to_numeric(df["ic"], errors="coerce")
                    return df.reset_index(drop=True)
            except Exception:
                pass  # 回退 JSONL

        # JSONL 回退
        if self.store_path is None:
            return pd.DataFrame(columns=["factor", "trade_date", "ic", "recorded_at"])
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
                g["ic"].rolling(window=window, min_periods=mp).std(ddof=1)
            )
            g["roll_ir"] = g["roll_mean"] / g["roll_std"].replace(0, np.nan)
            parts.append(g)

        return pd.concat(parts, ignore_index=True)

    def rolling_icir(
        self,
        factor_names: list[str],
        *,
        window: int = 20,
        min_periods: int = 12,
    ) -> dict[str, float]:
        """P1-3: 返回各因子最近 window 期的 ICIR = mean(IC) / std(IC)。

        用于动态因子权重：ICIR 越高的因子在合成分中权重越大。
        若某因子数据不足，返回 0.0。

        Returns
        -------
        dict : factor_name -> latest_icir (float)
        """
        stats = self.rolling_ic_stats(window=window, factors=factor_names, min_periods=min_periods)
        if stats.empty:
            return {f: 0.0 for f in factor_names}

        result: dict[str, float] = {}
        for fname in factor_names:
            grp = stats[stats["factor"] == fname]
            if grp.empty:
                result[fname] = 0.0
                continue
            latest = grp.sort_values("trade_date").iloc[-1]
            ir = latest.get("roll_ir")
            result[fname] = float(ir) if ir is not None and np.isfinite(ir) else 0.0
        return result

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
        alert_handler: Optional[Callable[[ICDecayAlert], None]] = None,
    ) -> List[ICDecayAlert]:
        """
        检查最新滚动 IC 均值是否低于阈值，若低则生成告警。

        P2-8: 支持 alert_handler 回调（如企业微信 Webhook），触发告警时调用。

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
        alert_handler
            可选回调，接收 ``ICDecayAlert`` 对象。可用于发送企业微信、
            钉钉通知等。若未提供，仅记录日志。

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
                alert = ICDecayAlert(
                    factor=str(fname),
                    window=window,
                    threshold=threshold,
                    rolling_mean_ic=float(roll_mean),
                    check_date=today,
                    message=msg,
                )
                alerts.append(alert)
                # P2-8: 调用告警回调（如企业微信 Webhook）
                if alert_handler is not None:
                    try:
                        alert_handler(alert)
                    except Exception as exc:
                        logger.error("IC 告警回调执行失败: %s", exc)
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
            s = float(ic.std(ddof=1)) if n > 1 else np.nan
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
