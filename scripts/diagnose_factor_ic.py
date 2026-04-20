#!/usr/bin/env python3
"""
P0-A：因子 Rank IC 诊断脚本（独立于回测引擎）。

输出：
1) 因子 x horizon 的 Rank IC 汇总（mean/std/icir/hit_rate/t-value）
2) 基于 P0-A 规则的因子处置建议
3) （可选）把指定 horizon 的逐日 IC 写入 ic_monitor.json，供后续 ICIR 动态权重使用
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用已有的数据读取与因子计算逻辑，确保诊断口径与当前策略一致。
from run_backtest_eval import (  # type: ignore[import-not-found]
    _attach_pit_fundamentals,
    attach_universe_filter,
    compute_factors,
    load_config,
    load_daily_from_duckdb,
)
from src.features.factor_eval import rank_ic
from src.features.ic_f1_gate import build_f1_gate_rows
from src.features.ic_monitor import ICMonitor


@dataclass
class HorizonSpec:
    key: str
    settlement: str
    horizon_days: int
    column: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P0-A 因子 Rank IC 诊断（tplus1_open + close horizons）")
    p.add_argument("--config", default="", help="配置文件路径；默认按 config.yaml.backtest -> config.yaml")
    p.add_argument("--start", default="2021-01-01", help="起始日期")
    p.add_argument("--end", default="", help="结束日期（为空则取配置 asof_trade_date 或今日）")
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="标的最少历史交易日")
    p.add_argument(
        "--horizon-days",
        type=int,
        nargs="+",
        default=[1, 5, 21],
        help="close_to_close 前瞻窗口（交易日），例如: 1 5 21",
    )
    p.add_argument(
        "--factors",
        default="",
        help="逗号分隔因子名；为空则取 config.signals.composite_extended 的键",
    )
    p.add_argument("--out-csv", default="data/results/factor_ic_report.csv", help="汇总 CSV 输出路径")
    p.add_argument("--out-json", default="data/results/factor_ic_report.json", help="汇总 JSON 输出路径")
    p.add_argument(
        "--out-daily-ic-csv",
        default="",
        help="可选：输出逐日 IC 明细（长表）CSV",
    )
    p.add_argument(
        "--monitor-path",
        default="",
        help="可选：输出到 ic_monitor.json 的路径；为空则不写",
    )
    p.add_argument(
        "--monitor-settlement",
        choices=("close_to_close", "tplus1_open"),
        default="close_to_close",
        help="写入 ic_monitor 的收益口径",
    )
    p.add_argument("--monitor-horizon", type=int, default=21, help="写入 ic_monitor 的 horizon")
    p.add_argument("--overwrite-monitor", action="store_true", help="写入 ic_monitor 时覆盖同日期记录")
    p.add_argument(
        "--tplus1-invalid-threshold",
        type=float,
        default=0.01,
        help="P0-A：T+1 开盘 IC 低于该阈值视为当前执行口径无效",
    )
    p.add_argument(
        "--t21-strong-threshold",
        type=float,
        default=0.02,
        help="P0-A：T+21 收盘 IC 高于该阈值视为中期有效",
    )
    p.add_argument(
        "--tplus1-zero-threshold",
        type=float,
        default=0.005,
        help="P1-B：|T+1 开盘 IC| 小于该阈值时建议权重清零",
    )
    p.add_argument(
        "--apply-universe-m2",
        action="store_true",
        help="M2.4：20 日日均成交额 + ROE_TTM>0 过滤后再算 IC（与回测 universe_filter 口径一致时可读 config）",
    )
    p.add_argument(
        "--universe-min-amount-20d",
        type=float,
        default=None,
        help="流动性阈值（元）；默认5e7 或 config.universe_filter.min_amount_20d",
    )
    p.add_argument(
        "--universe-skip-roe",
        action="store_true",
        help="M2.4 过滤时不强制 ROE_TTM>0（仅测流动性）",
    )
    p.add_argument(
        "--f1-validate",
        action="store_true",
        help="F1：对指定因子检查 T+21(close_21d) Rank IC 与 t 值是否达到进组合门槛",
    )
    p.add_argument(
        "--f1-factors",
        default="ocf_to_asset,gross_margin_delta,net_margin_stability,asset_turnover",
        help="F1 验收因子列表（逗号分隔）；须含于 --factors 或 composite_extended",
    )
    p.add_argument("--f1-min-ic", type=float, default=0.01, help="F1：T+21 IC 均值下限")
    p.add_argument("--f1-min-t", type=float, default=2.0, help="F1：T+21 IC t 值下限")
    return p.parse_args()


def _resolve_end_date(cfg: dict[str, Any], end_arg: str) -> str:
    if str(end_arg).strip():
        return str(end_arg).strip()
    x = str(cfg.get("paths", {}).get("asof_trade_date", "")).strip()
    if x:
        return x
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _select_factors(cfg: dict[str, Any], factor_arg: str, factor_df: pd.DataFrame) -> list[str]:
    if factor_arg.strip():
        cand = [x.strip() for x in factor_arg.split(",") if x.strip()]
    else:
        cand = list((cfg.get("signals", {}) or {}).get("composite_extended", {}).keys())
    factors = [f for f in cand if f in factor_df.columns]
    missing = [f for f in cand if f not in factor_df.columns]
    if missing:
        print(f"[WARN] 以下因子在因子表中不存在，已跳过: {missing}")
    if not factors:
        raise RuntimeError("无可用因子可评估")
    return factors


def _attach_forward_returns(
    daily_df: pd.DataFrame,
    close_horizons: list[int],
) -> tuple[pd.DataFrame, list[HorizonSpec]]:
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    g = d.groupby("symbol", sort=False)
    specs: list[HorizonSpec] = []

    # tplus1_open 真实可交易口径：open(T+1+h)/open(T+1)-1。这里固定 h=1。
    open_col = "fwd_tplus1_open_1d"
    d[open_col] = g["open"].shift(-2) / g["open"].shift(-1) - 1.0
    specs.append(HorizonSpec(key="tplus1_open_1d", settlement="tplus1_open", horizon_days=1, column=open_col))

    for h in sorted(set(int(x) for x in close_horizons if int(x) > 0)):
        c = f"fwd_close_{h}d"
        d[c] = g["close"].shift(-h) / d["close"] - 1.0
        specs.append(HorizonSpec(key=f"close_{h}d", settlement="close_to_close", horizon_days=h, column=c))
    return d, specs


def _ic_stats(ic_ser: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(ic_ser, errors="coerce").dropna()
    n = int(len(x))
    if n <= 0:
        return {
            "n_dates": 0,
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "ic_hit_rate": float("nan"),
            "ic_t_value": float("nan"),
        }
    mean_v = float(x.mean())
    std_v = float(x.std(ddof=0))
    ir_v = mean_v / std_v if std_v > 1e-15 else float("nan")
    hit_v = float((x > 0).mean())
    t_v = mean_v / (std_v / math.sqrt(n)) if std_v > 1e-15 and n > 1 else float("nan")
    return {
        "n_dates": n,
        "ic_mean": mean_v,
        "ic_std": std_v,
        "ic_ir": ir_v,
        "ic_hit_rate": hit_v,
        "ic_t_value": t_v,
    }


def _build_decision_table(
    summary_df: pd.DataFrame,
    *,
    tplus1_invalid_threshold: float,
    t21_strong_threshold: float,
    tplus1_zero_threshold: float,
) -> pd.DataFrame:
    pivot = (
        summary_df.pivot_table(index="factor", columns="horizon_key", values="ic_mean", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "tplus1_open_1d" not in pivot.columns:
        pivot["tplus1_open_1d"] = np.nan
    if "close_21d" not in pivot.columns:
        pivot["close_21d"] = np.nan
    pivot = pivot.rename(
        columns={
            "tplus1_open_1d": "ic_mean_tplus1_open_1d",
            "close_21d": "ic_mean_close_21d",
        }
    )
    pivot["flag_invalid_tplus1"] = pivot["ic_mean_tplus1_open_1d"] < float(tplus1_invalid_threshold)
    pivot["flag_strong_t21"] = pivot["ic_mean_close_21d"] > float(t21_strong_threshold)
    pivot["flag_remove_both_negative"] = (
        (pivot["ic_mean_tplus1_open_1d"] < 0.0) & (pivot["ic_mean_close_21d"] < 0.0)
    )
    pivot["flag_zero_weak_tplus1"] = pivot["ic_mean_tplus1_open_1d"].abs() < float(tplus1_zero_threshold)
    pivot["flag_flip_tplus1_negative_t21_positive"] = (
        (pivot["ic_mean_tplus1_open_1d"] < 0.0) & (pivot["ic_mean_close_21d"] > float(t21_strong_threshold))
    )
    decisions: list[str] = []
    p1_actions: list[str] = []
    for _, r in pivot.iterrows():
        tags: list[str] = []
        if bool(r["flag_invalid_tplus1"]):
            tags.append("remove_or_flip")
        if bool(r["flag_strong_t21"]):
            tags.append("keep_and_upweight")
        if not tags:
            tags.append("observe")
        decisions.append(";".join(tags))
        if bool(r["flag_remove_both_negative"]):
            p1_actions.append("remove")
        elif bool(r["flag_zero_weak_tplus1"]):
            p1_actions.append("zero")
        elif bool(r["flag_flip_tplus1_negative_t21_positive"]):
            p1_actions.append("flip")
        else:
            p1_actions.append("keep")
    pivot["decision"] = decisions
    pivot["p1_action"] = p1_actions
    return pivot.sort_values("factor").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    cfg, cfg_source = load_config(args.config)
    end_date = _resolve_end_date(cfg, args.end)
    start_date = str(args.start)
    _db = Path(str(cfg.get("paths", {}).get("duckdb_path", "data/market.duckdb")))
    db_path = str(_db if _db.is_absolute() else ROOT / _db)

    print("=" * 70)
    print("P0-A 因子 IC 诊断")
    print(f"区间: {start_date} ~ {end_date} | 配置: {cfg_source}")
    print("=" * 70)

    print("[1/4] 读取日线数据...")
    daily_df = load_daily_from_duckdb(db_path, start_date, end_date, int(args.lookback_days))
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"]).dt.normalize()
    print(f"  rows={len(daily_df):,} | symbols={daily_df['symbol'].nunique():,}")

    print("[2/4] 计算因子与前瞻收益标签...")
    factor_df = compute_factors(daily_df, min_hist_days=int(args.min_hist_days))
    factor_df = _attach_pit_fundamentals(factor_df, db_path)
    uf = cfg.get("universe_filter", {}) or {}
    u_enabled = bool(args.apply_universe_m2) or bool(uf.get("enabled", False))
    u_min = (
        float(args.universe_min_amount_20d)
        if args.universe_min_amount_20d is not None
        else float(uf.get("min_amount_20d", 50_000_000))
    )
    u_roe = bool(uf.get("require_roe_ttm_positive", True)) and not bool(args.universe_skip_roe)
    factor_df = attach_universe_filter(
        factor_df,
        daily_df,
        enabled=u_enabled,
        min_amount_20d=u_min,
        require_roe_ttm_positive=u_roe,
    )
    factors = _select_factors(cfg, str(args.factors), factor_df)
    daily_with_fwd, horizon_specs = _attach_forward_returns(daily_df, close_horizons=[int(x) for x in args.horizon_days])
    merge_cols = ["symbol", "trade_date", *[h.column for h in horizon_specs]]
    panel = factor_df[["symbol", "trade_date", *factors, "_universe_eligible"]].merge(
        daily_with_fwd[merge_cols],
        on=["symbol", "trade_date"],
        how="left",
    )
    panel = panel[(panel["trade_date"] >= pd.Timestamp(start_date)) & (panel["trade_date"] <= pd.Timestamp(end_date))]
    if u_enabled:
        panel = panel.loc[panel["_universe_eligible"].to_numpy(dtype=bool)]
    panel = panel.drop(columns=["_universe_eligible"], errors="ignore")
    print(f"  panel_rows={len(panel):,} | factors={len(factors)} | horizons={len(horizon_specs)} | universe_m2={'on' if u_enabled else 'off'}")

    print("[3/4] 计算截面 Rank IC...")
    summary_rows: list[dict[str, Any]] = []
    daily_ic_rows: list[dict[str, Any]] = []
    ic_series_map: dict[tuple[str, str], pd.Series] = {}
    for hs in horizon_specs:
        for fac in factors:
            sub = panel[["trade_date", "symbol", fac, hs.column]].rename(columns={hs.column: "forward_ret"})
            ic_ser = rank_ic(sub, factor_col=fac, forward_ret_col="forward_ret", date_col="trade_date")
            ic_series_map[(fac, hs.key)] = ic_ser
            st = _ic_stats(ic_ser)
            summary_rows.append(
                {
                    "factor": fac,
                    "horizon_key": hs.key,
                    "settlement": hs.settlement,
                    "horizon_days": int(hs.horizon_days),
                    **st,
                }
            )
            if str(args.out_daily_ic_csv).strip():
                for dt, v in ic_ser.items():
                    daily_ic_rows.append(
                        {
                            "factor": fac,
                            "horizon_key": hs.key,
                            "trade_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                            "ic": float(v) if pd.notna(v) else np.nan,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["factor", "horizon_days", "settlement"]).reset_index(drop=True)
    decision_df = _build_decision_table(
        summary_df,
        tplus1_invalid_threshold=float(args.tplus1_invalid_threshold),
        t21_strong_threshold=float(args.t21_strong_threshold),
        tplus1_zero_threshold=float(args.tplus1_zero_threshold),
    )

    print("[4/4] 写出报告...")
    out_csv = Path(args.out_csv).expanduser()
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = Path(args.out_json).expanduser()
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    f1_rows: list[dict[str, Any]] = []
    if bool(args.f1_validate):
        f1_cand = [x.strip() for x in str(args.f1_factors).split(",") if x.strip()]
        f1_rows = build_f1_gate_rows(
            summary_df,
            f1_cand,
            ic_min=float(args.f1_min_ic),
            t_min=float(args.f1_min_t),
        )
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config_source": str(cfg_source),
        "range": {"start": start_date, "end": end_date},
        "factors": factors,
        "horizons": [asdict(h) for h in horizon_specs],
        "universe_m2": {
            "enabled": bool(u_enabled),
            "min_amount_20d": float(u_min),
            "require_roe_ttm_positive": bool(u_roe),
        },
        "decision_rules": {
            "tplus1_invalid_threshold": float(args.tplus1_invalid_threshold),
            "t21_strong_threshold": float(args.t21_strong_threshold),
            "tplus1_zero_threshold": float(args.tplus1_zero_threshold),
        },
        "summary": summary_df.to_dict(orient="records"),
        "factor_decisions": decision_df.to_dict(orient="records"),
        "f1_gate": f1_rows,
        "f1_rules": (
            {
                "min_ic_close_21d": float(args.f1_min_ic),
                "min_t_close_21d": float(args.f1_min_t),
            }
            if bool(args.f1_validate)
            else {}
        ),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    daily_ic_path = ""
    if str(args.out_daily_ic_csv).strip():
        out_daily = Path(str(args.out_daily_ic_csv)).expanduser()
        if not out_daily.is_absolute():
            out_daily = ROOT / out_daily
        out_daily.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(daily_ic_rows).to_csv(out_daily, index=False, encoding="utf-8-sig")
        daily_ic_path = str(out_daily)

    if str(args.monitor_path).strip():
        monitor_path = Path(str(args.monitor_path)).expanduser()
        if not monitor_path.is_absolute():
            monitor_path = ROOT / monitor_path
        target_key = (
            f"close_{int(args.monitor_horizon)}d"
            if str(args.monitor_settlement) == "close_to_close"
            else f"tplus1_open_{int(args.monitor_horizon)}d"
        )
        factor_map: dict[str, pd.Series] = {}
        for fac in factors:
            ser = ic_series_map.get((fac, target_key))
            if ser is not None:
                factor_map[fac] = ser
        if not factor_map:
            print(f"[WARN] monitor 目标 horizon 不存在：{target_key}，跳过写入")
        else:
            mon = ICMonitor(monitor_path)
            write_stat = mon.append_many(factor_map, overwrite_dates=bool(args.overwrite_monitor))
            total_new = int(sum(int(v) for v in write_stat.values()))
            print(f"[OK] ic_monitor 已更新: {monitor_path} | horizon={target_key} | new_rows={total_new}")

    invalid_cnt = int(decision_df["flag_invalid_tplus1"].sum())
    strong_cnt = int(decision_df["flag_strong_t21"].sum())
    action_cnt = decision_df["p1_action"].value_counts().to_dict()
    print(f"[OK] 汇总 CSV: {out_csv}")
    print(f"[OK] 汇总 JSON: {out_json}")
    if daily_ic_path:
        print(f"[OK] 逐日 IC CSV: {daily_ic_path}")
    print(
        "[RESULT] 因子结论："
        f" tplus1 无效={invalid_cnt}/{len(decision_df)},"
        f" t21 强有效={strong_cnt}/{len(decision_df)}"
    )
    print(
        "[RESULT] P1-B 建议："
        f" keep={int(action_cnt.get('keep', 0))},"
        f" zero={int(action_cnt.get('zero', 0))},"
        f" flip={int(action_cnt.get('flip', 0))},"
        f" remove={int(action_cnt.get('remove', 0))}"
    )
    if bool(args.f1_validate) and f1_rows:
        print("[RESULT] F1进组合门槛（T+21 close Rank IC）：")
        for r in f1_rows:
            tag = "PASS" if r["pass_f1"] else "FAIL"
            extra = f" | {r['reason']}" if r.get("reason") and not r["pass_f1"] else ""
            print(
                f"  [{tag}] {r['factor']}: IC_mean={r['ic_mean']:.6g}, t={r['ic_t_value']:.6g}, n={r['n_dates']}{extra}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
