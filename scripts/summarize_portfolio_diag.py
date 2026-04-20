from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def load_row(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = _get(obj, "portfolio_diagnostics", "summary") or _get(obj, "meta", "portfolio_diagnostics_summary") or {}
    return {
        "label": path.stem,
        "config_source": obj.get("config_source"),
        "portfolio_method": _get(obj, "parameters", "portfolio_method"),
        "annualized_return": _get(obj, "full_sample", "with_cost", "annualized_return"),
        "sharpe_ratio": _get(obj, "full_sample", "with_cost", "sharpe_ratio"),
        "max_drawdown": _get(obj, "full_sample", "with_cost", "max_drawdown"),
        "turnover_mean": _get(obj, "full_sample", "with_cost", "turnover_mean"),
        "rolling_oos_median_ann_return": _get(obj, "walk_forward_rolling", "agg", "median_ann_return"),
        "slice_oos_median_ann_return": _get(obj, "walk_forward_slices", "agg", "median_ann_return"),
        "n_rebalances": summary.get("n_rebalances"),
        "mean_weight_std": summary.get("mean_weight_std"),
        "median_effective_n": summary.get("median_effective_n"),
        "mean_diag_share": summary.get("mean_diag_share"),
        "median_condition_number": summary.get("median_condition_number"),
        "mean_l1_diff_vs_equal": summary.get("mean_l1_diff_vs_equal"),
        "max_l1_diff_vs_equal": summary.get("max_l1_diff_vs_equal"),
        "equal_like_ratio": summary.get("equal_like_ratio"),
        "solver_success_ratio": summary.get("solver_success_ratio"),
        "fallback_counts": json.dumps(summary.get("fallback_counts", {}), ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="汇总 run_backtest_eval 的组合优化诊断 JSON")
    p.add_argument("--output", required=True, help="输出 CSV")
    p.add_argument("reports", nargs="+", help="输入 JSON 报告")
    args = p.parse_args()

    rows = [load_row(Path(x)) for x in args.reports]
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out)


if __name__ == "__main__":
    main()
