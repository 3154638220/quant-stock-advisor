#!/usr/bin/env python3
"""M15: 月度生产一键脚本。

用法：
    python scripts/run_monthly_production.py --month 2026-05
    python scripts/run_monthly_production.py --month 2026-05 --skip-fetch

内部调用顺序：
    1. fetch_only.py（更新日线至月末最后交易日）
    2. run_monthly_selection_report.py（生成 Top-20 推荐，自动选信号日）
    3. run_monthly_benchmark_suite.py（更新成本压力表）
    4. 输出摘要：信号日、Top-20 名单、OOS 历史表

告警机制：
    - OOS 连续 N 个月超额 < -1%：打印 WARNING
    - 候选池通过标的 < 2000：打印 WARNING
    - 新信号日 next_trade_date 为空：抛出 ValueError
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMOTED_CONFIG = "configs/promoted/monthly_selection_u1_top20_indcap3_hardcap_baseline.json"
OOS_TRACKING_FILE = "data/results/oos_tracking.json"
OOS_WARNING_CONSECUTIVE = 3
OOS_WARNING_THRESHOLD = -0.01
POOL_WIDTH_WARNING = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M15 月度生产一键脚本")
    p.add_argument("--month", required=True, help="目标月份，格式 YYYY-MM（如 2026-05）")
    p.add_argument("--skip-fetch", action="store_true", help="跳过数据拉取")
    p.add_argument("--config", default=PROMOTED_CONFIG, help="Promoted config 路径")
    p.add_argument("--dry-run", action="store_true", help="仅打印即将执行的命令，不实际运行")
    return p.parse_args()


def resolve_month_dates(month_str: str) -> dict:
    """根据月份解析信号日和交易日期。"""
    year, month = int(month_str[:4]), int(month_str[5:7])
    # 信号日 = 当月最后一个交易日（近似为最后一天，脚本内部会处理）
    # 下个月第一天作为参考
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    # 实际信号日由 run_monthly_selection_report 的 --auto-signal-date 内部确定
    return {
        "month_str": month_str,
        "approx_signal_month_end": last_day.strftime("%Y-%m-%d"),
        "next_month_first": next_month.strftime("%Y-%m-%d"),
    }


def run_cmd(cmd: list[str], step_name: str, dry_run: bool = False) -> int:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}[M15] Step: {step_name}")
    print(f"  {'CMD' if not dry_run else 'WOULD RUN'}: {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERROR] Step failed with code {result.returncode}")
    return result.returncode


def check_alerts(results_dir: Path, month_str: str):
    """告警检查：OOS 连续失效、候选池收窄、next_trade_date 缺失。"""
    print("\n[M15] Running alert checks ...")

    # 1. OOS tracking
    oos_path = ROOT / OOS_TRACKING_FILE
    if oos_path.exists():
        with open(oos_path) as f:
            oos = json.load(f)
        excesses = [
            entry.get("realized_excess", 0)
            for entry in oos.get("monthly", [])
            if isinstance(entry, dict)
        ]
        consecutive_bad = 0
        for ex in reversed(excesses):
            if ex < OOS_WARNING_THRESHOLD:
                consecutive_bad += 1
            else:
                break
        if consecutive_bad >= OOS_WARNING_CONSECUTIVE:
            print(f"  [WARNING] OOS 连续 {consecutive_bad} 个月超额 < -1%，请人工复核")
            print(f"           触发阈值：连续 {OOS_WARNING_CONSECUTIVE} 个月 < {OOS_WARNING_THRESHOLD:%}")
        else:
            print(f"  OOS 连续失效计数：{consecutive_bad}（阈值 {OOS_WARNING_CONSECUTIVE}）")
    else:
        print(f"  OOS tracking 文件不存在：{oos_path}，跳过检查")

    # 2. 候选池宽度检查
    pool_width_files = sorted(results_dir.glob("*candidate_pool_width*.csv"))
    if pool_width_files:
        import pandas as pd
        df = pd.read_csv(pool_width_files[-1])
        if "candidate_pool_pass_rows" in df.columns:
            recent = df["candidate_pool_pass_rows"].iloc[-1]
            if recent < POOL_WIDTH_WARNING:
                print(f"  [WARNING] 候选池异常收窄：最近月通过标的 {recent} < {POOL_WIDTH_WARNING}")
            else:
                print(f"  候选池宽度：{recent}（阈值 {POOL_WIDTH_WARNING}）")

    # 3. next_trade_date 检查
    rec_files = sorted(results_dir.glob("*recommendations*.csv"))
    if rec_files:
        import pandas as pd
        df = pd.read_csv(rec_files[-1])
        if "next_trade_date_expected" in df.columns:
            missing = df["next_trade_date_expected"].isna().sum()
            if missing > 0:
                raise ValueError(
                    f"新信号日 next_trade_date 缺失 {missing}/{len(df)} 条，不允许静默失败"
                )
            print(f"  next_trade_date 检查通过（0/{len(df)} 缺失）")


def print_summary(results_dir: Path, month_str: str):
    """输出本月摘要。"""
    print(f"\n{'='*60}")
    print(f"  M15 月度生产摘要 — {month_str}")
    print(f"{'='*60}")

    # 查找最新的推荐文件
    rec_files = sorted(results_dir.glob(f"*{month_str[:4]}*top20*.csv"))
    if rec_files:
        import pandas as pd
        df = pd.read_csv(rec_files[-1])
        signal_date = df["signal_date"].iloc[0] if "signal_date" in df.columns else "N/A"
        print(f"  信号日：{signal_date}")
        print(f"  Top-20 标的数：{len(df)}")
        if "industry_level1" in df.columns:
            industries = df["industry_level1"].value_counts().head(5)
            print(f"  行业分布（Top-5）：")
            for ind, cnt in industries.items():
                print(f"    {ind}: {cnt}")

    # OOS 历史表
    oos_path = ROOT / OOS_TRACKING_FILE
    if oos_path.exists():
        with open(oos_path) as f:
            oos = json.load(f)
        monthly = oos.get("monthly", [])
        if monthly:
            print(f"\n  OOS 历史（{len(monthly)} 个月）：")
            for entry in monthly:
                sd = entry.get("signal_date", "?")
                re = entry.get("realized_excess", 0)
                flag = "⚠️ " if re < OOS_WARNING_THRESHOLD else "  "
                print(f"    {flag}{sd} → {re:+.2%}")

    print(f"{'='*60}\n")


def main() -> int:
    args = parse_args()
    dates = resolve_month_dates(args.month)
    month_str = dates["month_str"]
    results_dir = ROOT / "data" / "results"
    docs_dir = ROOT / "docs" / "reports" / month_str
    for d in [results_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[M15] 月度生产开始 — {month_str}")
    print(f"  Config: {args.config}")
    print(f"  结果目录: {results_dir}")
    print(f"  报告目录: {docs_dir}")

    # Step 1: 数据更新
    if not args.skip_fetch:
        # 尝试 fetch_only.py；如不存在则跳过
        fetch_script = ROOT / "scripts" / "fetch_only.py"
        if fetch_script.exists():
            rc = run_cmd(
                ["python3", str(fetch_script), "--end-date", dates["next_month_first"]],
                "fetch_only.py",
                dry_run=args.dry_run,
            )
            if rc != 0:
                print("[WARNING] 数据拉取返回非零，后续步骤可能缺少最新日线")
        else:
            print("[M15] fetch_only.py 不存在，跳过数据拉取（使用已有数据）")
    else:
        print("[M15] --skip-fetch，跳过数据拉取")

    # Step 2: 生成 Top-20 推荐
    output_stem = f"monthly_selection_{month_str.replace('-', '_')}_top20_promoted"
    rc = run_cmd(
        [
            "python3", str(ROOT / "scripts" / "run_monthly_selection_report.py"),
            "--config", args.config,
            "--output-prefix", output_stem,
            "--top-k", "20",
            "--candidate-pools", "U1_liquid_tradable",
            "--families", "industry_breadth,fund_flow,fundamental",
            "--cost-bps", "10",
            "--auto-signal-date" if False else "",  # placeholder for future flag
        ],
        "run_monthly_selection_report.py",
        dry_run=args.dry_run,
    )

    # Step 3: Benchmark suite (更新成本压力表)
    monthly_long_candidates = sorted(results_dir.glob("*concentration_regime*monthly_long.csv"))
    if monthly_long_candidates:
        monthly_long = str(monthly_long_candidates[-1])
        rc = run_cmd(
            [
                "python3", str(ROOT / "scripts" / "run_monthly_benchmark_suite.py"),
                "--monthly-long", monthly_long,
                "--model", "M8_regime_aware_fixed_policy__indcap3",
                "--candidate-pool", "U1_liquid_tradable",
                "--top-k", "20",
                "--output-prefix", f"monthly_selection_benchmark_suite_{month_str}",
            ],
            "run_monthly_benchmark_suite.py",
            dry_run=args.dry_run,
        )

    # Step 4: 告警检查
    check_alerts(results_dir, month_str)

    # Step 5: 摘要
    print_summary(results_dir, month_str)

    print(f"[M15] 月度生产完成 — {month_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
