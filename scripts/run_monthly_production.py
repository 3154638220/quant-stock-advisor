#!/usr/bin/env python3
"""M15: 月度生产一键脚本。

用法：
    python scripts/run_monthly_production.py --month 2026-05
    python scripts/run_monthly_production.py --month 2026-05 --skip-fetch
    python scripts/run_monthly_production.py --month 2026-05 --dry-run

内部调用顺序：
    1. fetch_only.py（增量更新日线）
    2. run_monthly_selection_report.py（生成 Top-20 推荐，自动选信号日；
       脚本内部自动调用 record_oos_from_m7_report 写入预测 + 回填上月实现值）
    3. run_monthly_benchmark_suite.py（更新成本压力表）
    4. 告警检查（OOS 连续失效、候选池收窄、next_trade_date 缺失、LHB 数据陈旧）
    5. 输出摘要

告警机制：
    - OOS 连续 N 个月超额 < -1%：打印 WARNING
    - 候选池通过标的 < 2000：打印 WARNING
    - 新信号日 next_trade_date 为空：抛出 ValueError
    - LHB 数据最新日期落后今日 > 30 天：打印 WARNING
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROMOTED_CONFIG = "configs/promoted/monthly_selection_u1_top20_indcap3_hardcap_baseline.json"
DUCKDB_PATH = "data/market.duckdb"
OOS_WARNING_CONSECUTIVE = 3
OOS_WARNING_THRESHOLD = -0.01
POOL_WIDTH_WARNING = 2000
LHB_STALE_DAYS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M15 月度生产一键脚本")
    p.add_argument("--month", required=True, help="目标月份，格式 YYYY-MM（如 2026-05）")
    p.add_argument("--skip-fetch", action="store_true", help="跳过数据拉取")
    p.add_argument("--config", default=PROMOTED_CONFIG, help="Promoted config 路径")
    p.add_argument("--dry-run", action="store_true", help="仅打印即将执行的命令，不实际运行")
    return p.parse_args()


def resolve_month_dates(month_str: str) -> dict:
    year, month = int(month_str[:4]), int(month_str[5:7])
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return {
        "month_str": month_str,
        "approx_signal_month_end": last_day.strftime("%Y-%m-%d"),
        "next_month_first": next_month.strftime("%Y-%m-%d"),
    }


def load_promoted_config(config_path: str) -> dict:
    with open(ROOT / config_path) as f:
        return json.load(f)


def run_cmd(cmd: list[str], step_name: str, dry_run: bool = False) -> int:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}[M15] Step: {step_name}")
    print(f"  {'CMD' if not dry_run else 'WOULD RUN'}: {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERROR] Step failed with code {result.returncode}")
    return result.returncode


def check_alerts(
    results_dir: Path,
    month_str: str,
    config_id: str,
    db_path: str,
):
    """告警检查：OOS 连续失效、候选池收窄、next_trade_date 缺失、LHB 陈旧。"""
    print("\n[M15] Running alert checks ...")

    # 1. OOS tracking（DuckDB）
    _check_oos_degradation(db_path, config_id)

    # 2. 候选池宽度检查
    _check_pool_width(results_dir)

    # 3. next_trade_date 检查
    _check_next_trade_date(results_dir)

    # 4. LHB 数据陈旧检查
    _check_lhb_freshness(db_path)


def _check_oos_degradation(db_path: str, config_id: str):
    """从 DuckDB oos_tracking 表读取 OOS 记录，检测连续负超额。"""
    try:
        conn = duckdb.connect(str(ROOT / db_path), read_only=True)
        df = conn.execute(
            """
            SELECT signal_date, realized_excess_monthly
            FROM oos_tracking
            WHERE config_id = ?
              AND realized_excess_monthly IS NOT NULL
            ORDER BY signal_date ASC
            """,
            [config_id],
        ).df()
        conn.close()

        if df.empty:
            print("  OOS tracking 无已实现记录，跳过检查")
            return

        consecutive_bad = 0
        for _, row in df[::-1].iterrows():
            if row["realized_excess_monthly"] < OOS_WARNING_THRESHOLD:
                consecutive_bad += 1
            else:
                break

        if consecutive_bad >= OOS_WARNING_CONSECUTIVE:
            print(
                f"  [WARNING] OOS 连续 {consecutive_bad} 个月超额 < -1%，请人工复核"
            )
            print(
                f"           触发阈值：连续 {OOS_WARNING_CONSECUTIVE} 个月 < {OOS_WARNING_THRESHOLD:%}"
            )
        else:
            print(
                f"  OOS 连续失效计数：{consecutive_bad}/{OOS_WARNING_CONSECUTIVE}（触发阈值）"
            )
            # 打印最近 3 条记录
            for _, row in df.tail(3).iterrows():
                re = row["realized_excess_monthly"]
                flag = "⚠️ " if re < OOS_WARNING_THRESHOLD else "  "
                print(f"    {flag}{row['signal_date']} → {re:+.2%}")
    except Exception as e:
        print(f"  [WARNING] OOS 检查失败: {e}")


def _check_pool_width(results_dir: Path):
    pool_files = sorted(results_dir.glob("*candidate_pool_width*.csv"))
    if not pool_files:
        print("  候选池宽度文件不存在，跳过检查")
        return
    df = pd.read_csv(pool_files[-1])
    if "candidate_pool_pass_rows" in df.columns:
        recent = int(df["candidate_pool_pass_rows"].iloc[-1])
        if recent < POOL_WIDTH_WARNING:
            print(
                f"  [WARNING] 候选池异常收窄：最近月通过标的 {recent} < {POOL_WIDTH_WARNING}"
            )
        else:
            print(f"  候选池宽度：{recent}（阈值 {POOL_WIDTH_WARNING}）")


def _check_next_trade_date(results_dir: Path):
    rec_files = sorted(results_dir.glob("*recommendations*.csv"))
    if not rec_files:
        print("  推荐文件不存在，跳过 next_trade_date 检查")
        return
    df = pd.read_csv(rec_files[-1])
    col = next(
        (c for c in ["next_trade_date_expected", "next_trade_date"] if c in df.columns),
        None,
    )
    if col is None:
        return
    missing = int(df[col].isna().sum())
    if missing > 0:
        raise ValueError(
            f"新信号日 next_trade_date 缺失 {missing}/{len(df)} 条，不允许静默失败"
        )
    print(f"  next_trade_date 检查通过（0/{len(df)} 缺失）")


def _check_lhb_freshness(db_path: str):
    """检查 LHB 数据最新日期是否落后今日超过 LHB_STALE_DAYS 天。"""
    try:
        conn = duckdb.connect(str(ROOT / db_path), read_only=True)
        row = conn.execute(
            "SELECT MAX(data_date) FROM a_share_lhb_daily"
        ).fetchone()
        conn.close()

        if row is None or row[0] is None:
            print("  LHB 数据表为空或不存在，跳过检查")
            return

        latest = pd.Timestamp(row[0])
        days_behind = (pd.Timestamp.now() - latest).days
        if days_behind > LHB_STALE_DAYS:
            print(
                f"  [WARNING] LHB 数据最新日期 {latest.date()} 落后 {days_behind} 天"
                f"（阈值 {LHB_STALE_DAYS} 天），可能需要更新"
            )
        else:
            print(f"  LHB 数据新鲜度 OK（最新 {latest.date()}，{days_behind} 天前）")
    except Exception as e:
        print(f"  [WARNING] LHB 检查失败: {e}")


def print_summary(
    results_dir: Path,
    month_str: str,
    config_id: str,
    db_path: str,
):
    """输出本月摘要。"""
    print(f"\n{'='*60}")
    print(f"  M15 月度生产摘要 — {month_str}")
    print(f"  Config: {config_id}")
    print(f"{'='*60}")

    # 查找最新的推荐文件
    rec_files = sorted(results_dir.glob(f"*{month_str[:4]}*top20*.csv"))
    if not rec_files:
        rec_files = sorted(results_dir.glob("*recommendations*.csv"))

    if rec_files:
        df = pd.read_csv(rec_files[-1])
        signal_date = (
            str(df["signal_date"].iloc[0]) if "signal_date" in df.columns else "N/A"
        )
        print(f"  信号日：{signal_date}")
        print(f"  Top-20 标的数：{len(df)}")
        if "symbol" in df.columns:
            print(f"  持仓：{', '.join(str(s) for s in df['symbol'].head(10))}{' ...' if len(df) > 10 else ''}")
        if "industry_level1" in df.columns:
            industries = df["industry_level1"].value_counts().head(5)
            print(f"  行业分布（Top-5）：")
            for ind, cnt in industries.items():
                print(f"    {ind}: {cnt}")

    # OOS 历史表（DuckDB）
    _print_oos_history(db_path, config_id)

    print(f"{'='*60}\n")


def _print_oos_history(db_path: str, config_id: str):
    try:
        conn = duckdb.connect(str(ROOT / db_path), read_only=True)
        df = conn.execute(
            """
            SELECT signal_date, predicted_excess_monthly, realized_excess_monthly
            FROM oos_tracking
            WHERE config_id = ?
            ORDER BY signal_date ASC
            """,
            [config_id],
        ).df()
        conn.close()

        if df.empty:
            print("\n  OOS 历史：无记录")
            return

        print(f"\n  OOS 历史（{len(df)} 条）：")
        for _, row in df.iterrows():
            sd = str(row["signal_date"])[:10]
            pred = row["predicted_excess_monthly"]
            real = row["realized_excess_monthly"]
            pred_s = f"{pred:+.2%}" if pd.notna(pred) else "N/A"
            real_s = f"{real:+.2%}" if pd.notna(real) else "待补录"
            flag = (
                "⚠️ "
                if pd.notna(real) and real < OOS_WARNING_THRESHOLD
                else "  "
            )
            print(f"    {flag}{sd}  预测 {pred_s}  实现 {real_s}")
    except Exception as e:
        print(f"\n  OOS 历史查询失败: {e}")


def main() -> int:
    args = parse_args()
    dates = resolve_month_dates(args.month)
    month_str = dates["month_str"]
    month_stem = month_str.replace("-", "_")

    # 读取 promoted config
    promoted = load_promoted_config(args.config)
    config_id = promoted["config_id"]
    method = promoted.get("method", {})
    cost_bps = method.get("cost_bps", 10.0)
    candidate_pool = method.get("candidate_pool_version", "U1_liquid_tradable")
    top_k = method.get("top_k", 20)
    model = method.get("model", "M8_regime_aware_fixed_policy__indcap3")

    results_dir = ROOT / "data" / "results"
    docs_dir = ROOT / "docs" / "reports" / month_str
    for d in [results_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[M15] 月度生产开始 — {month_str}")
    print(f"  Config: {args.config}")
    print(f"  Config ID: {config_id}")
    print(f"  Model: {model}")
    print(f"  Cost: {cost_bps} bps")
    print(f"  结果目录: {results_dir}")
    print(f"  报告目录: {docs_dir}")

    # ── Step 1: 数据更新 ──
    if not args.skip_fetch:
        fetch_script = ROOT / "scripts" / "fetch_only.py"
        if fetch_script.exists():
            rc = run_cmd(
                ["python3", str(fetch_script)],
                "fetch_only.py（增量更新日线）",
                dry_run=args.dry_run,
            )
            if rc != 0:
                print(
                    "[WARNING] 数据拉取返回非零，后续步骤可能缺少最新日线"
                )
        else:
            print("[M15] fetch_only.py 不存在，跳过数据拉取（使用已有数据）")
    else:
        print("[M15] --skip-fetch，跳过数据拉取")

    # ── Step 2: 生成 Top-20 推荐 ──
    # 报告脚本不传 --signal-date 时会自动选择数据集中最新信号日；
    # 脚本内部末尾自动调用 record_oos_from_m7_report 写入预测 + 回填上月实现值。
    output_stem = f"monthly_selection_{month_stem}_top{top_k}_promoted"
    cmd_step2 = [
        "python3",
        str(ROOT / "scripts" / "run_monthly_selection_report.py"),
        "--config", args.config,
        "--output-prefix", output_stem,
        "--top-k", str(top_k),
        "--candidate-pools", candidate_pool,
        "--families", "industry_breadth,fund_flow,fundamental",
        "--cost-bps", str(int(cost_bps)),
    ]
    rc = run_cmd(cmd_step2, "run_monthly_selection_report.py", dry_run=args.dry_run)

    # ── Step 3: Benchmark suite ──
    monthly_long_candidates = sorted(
        results_dir.glob("*concentration_regime*monthly_long.csv")
    )
    if monthly_long_candidates:
        monthly_long = str(monthly_long_candidates[-1])
        rc3 = run_cmd(
            [
                "python3",
                str(ROOT / "scripts" / "run_monthly_benchmark_suite.py"),
                "--monthly-long", monthly_long,
                "--model", model,
                "--candidate-pool", candidate_pool,
                "--top-k", str(top_k),
                "--output-prefix",
                f"monthly_selection_benchmark_suite_{month_str}",
            ],
            "run_monthly_benchmark_suite.py",
            dry_run=args.dry_run,
        )
    else:
        print(
            "\n[M15] 未找到 concentration_regime monthly_long.csv，跳过 benchmark suite"
        )

    # ── Step 4: 告警检查 ──
    check_alerts(results_dir, month_str, config_id, DUCKDB_PATH)

    # ── Step 5: 摘要 ──
    print_summary(results_dir, month_str, config_id, DUCKDB_PATH)

    print(f"[M15] 月度生产完成 — {month_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
