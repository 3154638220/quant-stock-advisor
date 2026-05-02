"""在当前 S2 基线下验证 F1 候选因子是否具备组合层准入资格。"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def _missing_runtime_dependency(name: str, exc: ModuleNotFoundError):
    def _raiser(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"{name} 依赖缺失：{exc.name}。当前仅可使用无需回测运行时的汇总/表格函数。"
        ) from exc

    return _raiser


try:
    from scripts.run_backtest_eval import (
        BacktestConfig,
        _attach_pit_fundamentals,
        _rebalance_dates,
        attach_universe_filter,
        build_asset_returns,
        build_market_ew_benchmark,
        build_open_to_open_returns,
        build_regime_weight_overrides,
        build_score,
        build_topk_weights,
        compare_full_vs_slices,
        compute_factors,
        contiguous_time_splits,
        load_config,
        load_daily_from_duckdb,
        normalize_weights,
        resolve_industry_cap_and_map,
        rolling_walk_forward_windows,
        run_backtest,
        transaction_cost_params_from_mapping,
        walk_forward_backtest,
    )
except ModuleNotFoundError as exc:
    BacktestConfig = Any  # type: ignore[assignment]
    _attach_pit_fundamentals = _missing_runtime_dependency("_attach_pit_fundamentals", exc)
    _rebalance_dates = _missing_runtime_dependency("_rebalance_dates", exc)
    attach_universe_filter = _missing_runtime_dependency("attach_universe_filter", exc)
    build_market_ew_benchmark = _missing_runtime_dependency("build_market_ew_benchmark", exc)
    build_asset_returns = _missing_runtime_dependency("build_asset_returns", exc)
    build_open_to_open_returns = _missing_runtime_dependency("build_open_to_open_returns", exc)
    build_regime_weight_overrides = _missing_runtime_dependency("build_regime_weight_overrides", exc)
    build_score = _missing_runtime_dependency("build_score", exc)
    build_topk_weights = _missing_runtime_dependency("build_topk_weights", exc)
    compare_full_vs_slices = _missing_runtime_dependency("compare_full_vs_slices", exc)
    compute_factors = _missing_runtime_dependency("compute_factors", exc)
    contiguous_time_splits = _missing_runtime_dependency("contiguous_time_splits", exc)
    load_config = _missing_runtime_dependency("load_config", exc)
    load_daily_from_duckdb = _missing_runtime_dependency("load_daily_from_duckdb", exc)
    normalize_weights = _missing_runtime_dependency("normalize_weights", exc)
    resolve_industry_cap_and_map = _missing_runtime_dependency("resolve_industry_cap_and_map", exc)
    rolling_walk_forward_windows = _missing_runtime_dependency("rolling_walk_forward_windows", exc)
    run_backtest = _missing_runtime_dependency("run_backtest", exc)
    transaction_cost_params_from_mapping = _missing_runtime_dependency("transaction_cost_params_from_mapping", exc)
    walk_forward_backtest = _missing_runtime_dependency("walk_forward_backtest", exc)
from scripts.research_identity import make_research_identity, slugify_token
from src.features.factor_eval import rank_ic
from src.models.experiment import append_experiment_result
from src.models.research_contract import (
    ArtifactRef,
    DataSlice,
    ExperimentResult,
    build_result_id,
    utc_now_iso,
    write_research_manifest,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    config_path: str
    candidate_factor: str
    family: str
    is_baseline: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("a0", "A0_baseline_s2", "config.yaml.backtest.f1_baseline_s2", "vol_to_turnover", "baseline", True),
    Scenario("a1", "A1_gross_single", "config.yaml.backtest.f1_gross_single", "gross_margin_delta", "gross"),
    Scenario("a2", "A2_gross_blend_10", "config.yaml.backtest.f1_gross_blend_10", "gross_margin_delta", "gross"),
    Scenario("a3", "A3_gross_blend_20", "config.yaml.backtest.f1_gross_blend_20", "gross_margin_delta", "gross"),
    Scenario("a4", "A4_ocf_asset_single", "config.yaml.backtest.f1_ocf_asset_single", "ocf_to_asset", "ocf_asset"),
    Scenario("a5", "A5_ocf_asset_blend_10", "config.yaml.backtest.f1_ocf_asset_blend_10", "ocf_to_asset", "ocf_asset"),
    Scenario("a6", "A6_ocf_asset_blend_20", "config.yaml.backtest.f1_ocf_asset_blend_20", "ocf_to_asset", "ocf_asset"),
    Scenario(
        "a7",
        "A7_net_margin_single",
        "config.yaml.backtest.f1_net_margin_single",
        "net_margin_stability",
        "net_margin",
    ),
    Scenario(
        "a8",
        "A8_net_margin_blend_10",
        "config.yaml.backtest.f1_net_margin_blend_10",
        "net_margin_stability",
        "net_margin",
    ),
    Scenario(
        "a9",
        "A9_net_margin_blend_20",
        "config.yaml.backtest.f1_net_margin_blend_20",
        "net_margin_stability",
        "net_margin",
    ),
    Scenario(
        "a10",
        "A10_asset_turnover_single",
        "config.yaml.backtest.f1_asset_turnover_single",
        "asset_turnover",
        "asset_turnover",
    ),
    Scenario(
        "a11",
        "A11_asset_turnover_blend_10",
        "config.yaml.backtest.f1_asset_turnover_blend_10",
        "asset_turnover",
        "asset_turnover",
    ),
    Scenario(
        "a12",
        "A12_asset_turnover_blend_20",
        "config.yaml.backtest.f1_asset_turnover_blend_20",
        "asset_turnover",
        "asset_turnover",
    ),
)

DEFAULT_BENCHMARK_KEY_YEARS: tuple[int, ...] = (2021, 2025, 2026)


def _slugify_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def build_admission_research_config_id(
    *,
    selected_families: list[str],
    benchmark_key_years: list[int],
    f1_min_ic: float,
    f1_min_t: float,
) -> str:
    family_token = "-".join(_slugify_token(item) for item in selected_families) if selected_families else "all"
    years_token = "-".join(str(int(item)) for item in benchmark_key_years) if benchmark_key_years else "none"
    ic_bp = int(round(float(f1_min_ic) * 10_000))
    t_bp = int(round(float(f1_min_t) * 100))
    return f"families_{family_token}_yrs_{years_token}_ic{ic_bp}_t{t_bp}"


def build_admission_output_stem(
    *,
    output_prefix: str,
    research_config_id: str,
) -> str:
    return f"{_slugify_token(output_prefix)}_{research_config_id}"


def select_scenarios(families: list[str] | None = None) -> list[Scenario]:
    selected = [scenario for scenario in SCENARIOS if scenario.is_baseline]
    if not families:
        selected.extend(scenario for scenario in SCENARIOS if not scenario.is_baseline)
        return selected

    family_set = {family.strip().lower() for family in families if family.strip()}
    selected.extend(
        scenario
        for scenario in SCENARIOS
        if (not scenario.is_baseline) and scenario.family.lower() in family_set
    )
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 F1 候选因子准入验证")
    p.add_argument("--start", default="2021-01-01", help="回测起始日期")
    p.add_argument("--end", default="", help="回测结束日期；为空时取今天")
    p.add_argument(
        "--output-prefix",
        default="factor_admission_validation_2026-04-19",
        help="输出文件前缀（写入 data/results 与 docs）",
    )
    p.add_argument("--lookback-days", type=int, default=260, help="因子热身回看交易日")
    p.add_argument("--min-hist-days", type=int, default=130, help="最少历史交易日")
    p.add_argument("--wf-train-window", type=int, default=252, help="滚动 WF 训练窗")
    p.add_argument("--wf-test-window", type=int, default=63, help="滚动 WF 测试窗")
    p.add_argument("--wf-step-window", type=int, default=63, help="滚动 WF 步长")
    p.add_argument("--wf-slice-splits", type=int, default=5, help="时间切片折数")
    p.add_argument("--wf-slice-min-train-days", type=int, default=252, help="时间切片最少训练窗")
    p.add_argument("--wf-slice-fixed-window", action="store_true", help="时间切片使用固定训练窗")
    p.add_argument("--f1-min-ic", type=float, default=0.01, help="F1：T+21 IC 均值下限")
    p.add_argument("--f1-min-t", type=float, default=2.0, help="F1：T+21 IC t 值下限")
    p.add_argument(
        "--families",
        default="",
        help="只运行指定候选族（逗号分隔），如 net_margin,asset_turnover；为空则运行全部候选",
    )
    p.add_argument(
        "--benchmark-key-years",
        default="2021,2025,2026",
        help="benchmark-first 重点观察年份，逗号分隔",
    )
    return p.parse_args()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return _json_sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _forward_returns_with_specs(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    d = daily_df.sort_values(["symbol", "trade_date"]).copy()
    g = d.groupby("symbol", sort=False)
    d["fwd_tplus1_open_1d"] = g["open"].shift(-2) / g["open"].shift(-1) - 1.0
    d["fwd_close_21d"] = g["close"].shift(-21) / d["close"] - 1.0
    specs = [
        {"key": "tplus1_open_1d", "column": "fwd_tplus1_open_1d"},
        {"key": "close_21d", "column": "fwd_close_21d"},
    ]
    return d, specs


def _ic_stats(ic_ser: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(ic_ser, errors="coerce").dropna()
    n = int(len(x))
    if n <= 0:
        return {"n_dates": 0, "ic_mean": np.nan, "ic_t_value": np.nan}
    mean_v = float(x.mean())
    std_v = float(x.std(ddof=0))
    t_v = mean_v / (std_v / np.sqrt(n)) if std_v > 1e-15 and n > 1 else np.nan
    return {"n_dates": n, "ic_mean": mean_v, "ic_t_value": t_v}


def _annualize_daily_returns(daily_returns: pd.Series) -> float:
    arr = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    total = float((1.0 + arr).prod())
    if total <= 0:
        return float("nan")
    return float(total ** (252.0 / len(arr)) - 1.0)


def _summarize_relative_to_benchmark(
    strategy_daily: pd.Series,
    benchmark_daily: pd.Series,
    *,
    key_years: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    common = pd.DatetimeIndex(strategy_daily.index).intersection(pd.DatetimeIndex(benchmark_daily.index)).sort_values()
    if len(common) == 0:
        empty = pd.DataFrame(columns=["year", "strategy_return", "benchmark_return", "excess_return"])
        return empty, {
            "annualized_excess_return": np.nan,
            "yearly_excess_median": np.nan,
            "key_year_excess_mean": np.nan,
            "key_year_excess_worst": np.nan,
        }

    strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0)
    bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0)
    excess = strat - bench
    yearly = pd.DataFrame(
        {
            "strategy_return": strat.groupby(strat.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0)),
            "benchmark_return": bench.groupby(bench.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0)),
        }
    ).reset_index(names="year")
    yearly["year"] = yearly["year"].astype(int)
    yearly["excess_return"] = yearly["strategy_return"] - yearly["benchmark_return"]

    key_year_df = yearly[yearly["year"].isin(key_years)].copy()
    return yearly, {
        "annualized_excess_return": _annualize_daily_returns(excess),
        "yearly_excess_median": float(pd.to_numeric(yearly["excess_return"], errors="coerce").median()),
        "key_year_excess_mean": (
            float(pd.to_numeric(key_year_df["excess_return"], errors="coerce").mean()) if not key_year_df.empty else np.nan
        ),
        "key_year_excess_worst": (
            float(pd.to_numeric(key_year_df["excess_return"], errors="coerce").min()) if not key_year_df.empty else np.nan
        ),
    }


def _summarize_oos_excess(
    strategy_daily: pd.Series,
    benchmark_daily: pd.Series,
    slices: list[Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sl in slices:
        common = (
            pd.DatetimeIndex(sl.test_index)
            .intersection(pd.DatetimeIndex(strategy_daily.index))
            .intersection(pd.DatetimeIndex(benchmark_daily.index))
            .sort_values()
        )
        if len(common) == 0:
            continue
        strat = pd.to_numeric(strategy_daily.reindex(common), errors="coerce").fillna(0.0)
        bench = pd.to_numeric(benchmark_daily.reindex(common), errors="coerce").fillna(0.0)
        rows.append(
            {
                "fold_id": int(getattr(sl, "fold_id", len(rows))),
                "annualized_excess_return": _annualize_daily_returns(strat - bench),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"median_ann_excess_return": np.nan, "detail": []}
    return {
        "median_ann_excess_return": float(pd.to_numeric(detail["annualized_excess_return"], errors="coerce").median()),
        "detail": detail.to_dict(orient="records"),
    }


def compute_factor_gate_table(
    factor_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    candidate_factors: list[str],
) -> pd.DataFrame:
    d, specs = _forward_returns_with_specs(daily_df)
    merged = factor_df.merge(
        d[["symbol", "trade_date"] + [s["column"] for s in specs]],
        on=["symbol", "trade_date"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for fac in candidate_factors:
        if fac not in merged.columns:
            continue
        for spec in specs:
            ic_ser = rank_ic(
                merged[["trade_date", fac, spec["column"]]].rename(columns={fac: "factor", spec["column"]: "forward_ret"}),
                factor_col="factor",
                forward_ret_col="forward_ret",
                date_col="trade_date",
            )
            stats = _ic_stats(ic_ser) if not ic_ser.empty else _ic_stats(pd.Series(dtype=float))
            rows.append(
                {
                    "factor": fac,
                    "horizon_key": spec["key"],
                    "n_dates": stats["n_dates"],
                    "ic_mean": stats["ic_mean"],
                    "ic_t_value": stats["ic_t_value"],
                }
            )
    return pd.DataFrame(rows).sort_values(["factor", "horizon_key"]).reset_index(drop=True)


def build_admission_table(
    summary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    *,
    baseline_label: str,
    f1_min_ic: float,
    f1_min_t: float,
    benchmark_key_years: list[int] | None = None,
) -> pd.DataFrame:
    benchmark_key_years = list(benchmark_key_years or DEFAULT_BENCHMARK_KEY_YEARS)
    baseline = summary_df.loc[summary_df["scenario"] == baseline_label].iloc[0]
    benchmark_excess_col = (
        "full_backtest_annualized_excess_vs_market"
        if "full_backtest_annualized_excess_vs_market" in summary_df.columns
        else "annualized_excess_vs_market"
    )
    gate_close = gate_df[gate_df["horizon_key"] == "close_21d"][["factor", "ic_mean", "ic_t_value"]].copy()
    gate_close = gate_close.rename(columns={"ic_mean": "close21_ic_mean", "ic_t_value": "close21_ic_t"})
    gate_open = gate_df[gate_df["horizon_key"] == "tplus1_open_1d"][["factor", "ic_mean", "ic_t_value"]].copy()
    gate_open = gate_open.rename(columns={"ic_mean": "open1_ic_mean", "ic_t_value": "open1_ic_t"})
    gate = gate_close.merge(gate_open, on="factor", how="outer")
    gate["pass_f1_gate"] = (
        (pd.to_numeric(gate["close21_ic_mean"], errors="coerce") > float(f1_min_ic))
        & (pd.to_numeric(gate["close21_ic_t"], errors="coerce") > float(f1_min_t))
    )

    out = summary_df.merge(gate, left_on="candidate_factor", right_on="factor", how="left")
    out["benchmark_excess_metric"] = str(benchmark_excess_col)
    out["delta_ann_vs_baseline"] = pd.to_numeric(out["annualized_return"], errors="coerce") - float(baseline["annualized_return"])
    out["delta_sharpe_vs_baseline"] = pd.to_numeric(out["sharpe_ratio"], errors="coerce") - float(baseline["sharpe_ratio"])
    out["delta_rolling_vs_baseline"] = (
        pd.to_numeric(out["rolling_oos_median_ann_return"], errors="coerce")
        - float(baseline["rolling_oos_median_ann_return"])
    )
    out["delta_slice_vs_baseline"] = (
        pd.to_numeric(out["slice_oos_median_ann_return"], errors="coerce")
        - float(baseline["slice_oos_median_ann_return"])
    )
    out["delta_ann_excess_vs_baseline"] = (
        pd.to_numeric(out[benchmark_excess_col], errors="coerce")
        - float(pd.to_numeric(baseline[benchmark_excess_col], errors="coerce"))
    )
    out["delta_yearly_excess_median_vs_baseline"] = (
        pd.to_numeric(out["yearly_excess_median_vs_market"], errors="coerce")
        - float(baseline["yearly_excess_median_vs_market"])
    )
    out["delta_rolling_excess_vs_baseline"] = (
        pd.to_numeric(out["rolling_oos_median_ann_excess_vs_market"], errors="coerce")
        - float(baseline["rolling_oos_median_ann_excess_vs_market"])
    )
    out["delta_slice_excess_vs_baseline"] = (
        pd.to_numeric(out["slice_oos_median_ann_excess_vs_market"], errors="coerce")
        - float(baseline["slice_oos_median_ann_excess_vs_market"])
    )
    out["delta_key_year_excess_mean_vs_baseline"] = (
        pd.to_numeric(out["key_year_excess_mean_vs_market"], errors="coerce")
        - float(baseline["key_year_excess_mean_vs_market"])
    )
    out["pass_combo_gate"] = (
        (pd.to_numeric(out["delta_ann_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(out["delta_rolling_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(out["delta_slice_vs_baseline"], errors="coerce") > 0.0)
    )
    out["pass_benchmark_gate"] = (
        (pd.to_numeric(out[benchmark_excess_col], errors="coerce") >= 0.0)
        & (pd.to_numeric(out["rolling_oos_median_ann_excess_vs_market"], errors="coerce") >= 0.0)
        & (pd.to_numeric(out["slice_oos_median_ann_excess_vs_market"], errors="coerce") >= 0.0)
        & (pd.to_numeric(out["delta_yearly_excess_median_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(out["delta_key_year_excess_mean_vs_baseline"], errors="coerce") > 0.0)
    )
    out["admission_status"] = np.where(
        out["is_baseline"],
        "baseline",
        np.where(out["pass_f1_gate"] & out["pass_combo_gate"] & out["pass_benchmark_gate"], "pass", "fail"),
    )
    out["benchmark_key_years"] = ",".join(str(int(y)) for y in benchmark_key_years)
    if "research_topic" in summary_df.columns:
        out["research_topic"] = summary_df["research_topic"]
    if "research_config_id" in summary_df.columns:
        out["research_config_id"] = summary_df["research_config_id"]
    if "output_stem" in summary_df.columns:
        out["output_stem"] = summary_df["output_stem"]
    out["result_type"] = "factor_admission"
    return out


def _build_doc(
    summary_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    admission_df: pd.DataFrame,
    output_prefix: str,
    *,
    candidate_factors: list[str],
) -> str:
    generated_at = pd.Timestamp.utcnow().isoformat()
    candidate_text = "、".join(f"`{factor}`" for factor in candidate_factors)
    benchmark_excess_metric = (
        summary_df.get("benchmark_excess_metric", pd.Series(["annualized_excess_vs_market"]))
        .dropna()
        .astype(str)
        .iloc[0]
    )
    return f"""# F1 候选因子准入验证

- 生成时间：`{generated_at}`
- 基线：`S2 = vol_to_turnover`，并沿用当前默认研究口径 `prefilter=false + universe=true`
- 候选：{candidate_text}
- 比较矩阵：基线、候选单因子、候选 `10% / 20%` 叠加版
- benchmark gate 年化超额口径：`{benchmark_excess_metric}`

## 组合汇总

{summary_df.to_markdown(index=False)}

## IC 门槛

{gate_df.to_markdown(index=False)}

## 准入结论

{admission_df.to_markdown(index=False)}

说明：`pass` 现在必须同时通过 `IC gate + combo gate + benchmark-first gate`。其中 benchmark-first gate 至少要求：

- `{benchmark_excess_metric} >= 0`
- `rolling / slice OOS` 的超额年化中位数均不为负
- 年度超额中位数相对基线改善
- 关键落后年份的平均超额相对基线改善

## 本轮产物

- `data/results/{output_prefix}_summary.csv`
- `data/results/{output_prefix}_factor_gate.csv`
- `data/results/{output_prefix}_admission.csv`
- `data/results/{output_prefix}_a*.json`
"""


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_prefix = str(args.output_prefix).strip()
    selected_families = [item.strip() for item in str(args.families).split(",") if item.strip()]
    selected_scenarios = select_scenarios(selected_families)
    if not selected_scenarios:
        raise SystemExit("未匹配到任何场景；请检查 --families 参数")
    baseline_label = next(s.label for s in selected_scenarios if s.is_baseline)
    results_dir = PROJECT_ROOT / "data/results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load base data: start={args.start} end={end_date}", flush=True)
    base_cfg, _ = load_config("config.yaml.backtest.f1_baseline_s2")
    db_path = str(PROJECT_ROOT / str(base_cfg["paths"]["duckdb_path"]))
    daily_df = load_daily_from_duckdb(db_path, args.start, end_date, args.lookback_days)
    print(f"  daily_df={daily_df.shape}", flush=True)
    factors = compute_factors(daily_df, min_hist_days=args.min_hist_days)
    print(f"  factors={factors.shape}", flush=True)
    needed_base_cols = [
        "symbol",
        "trade_date",
        "vol_to_turnover",
        "turnover_roll_mean",
        "price_position",
        "limit_move_hits_5d",
        "log_market_cap",
    ]
    factors = factors[[c for c in needed_base_cols if c in factors.columns]].copy()
    rebalance_dates = set(_rebalance_dates(factors["trade_date"].unique(), "M"))
    factors = factors[factors["trade_date"].isin(rebalance_dates)].copy()
    print(f"  factors@rebalance={factors.shape}", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        factors = _attach_pit_fundamentals(factors, db_path)
    print(f"  factors+pit={factors.shape}", flush=True)

    candidate_factors = sorted({s.candidate_factor for s in selected_scenarios if not s.is_baseline})
    benchmark_key_years = [
        int(item.strip()) for item in str(args.benchmark_key_years).split(",") if str(item).strip()
    ] or list(DEFAULT_BENCHMARK_KEY_YEARS)
    research_topic = "factor_admission_validation"
    research_config_id = build_admission_research_config_id(
        selected_families=selected_families,
        benchmark_key_years=benchmark_key_years,
        f1_min_ic=float(args.f1_min_ic),
        f1_min_t=float(args.f1_min_t),
    )
    output_stem = build_admission_output_stem(
        output_prefix=output_prefix,
        research_config_id=research_config_id,
    )
    gate_factor_df = attach_universe_filter(
        factors,
        daily_df,
        enabled=True,
        min_amount_20d=50_000_000.0,
        require_roe_ttm_positive=True,
    )
    gate_df = compute_factor_gate_table(gate_factor_df, daily_df, candidate_factors=candidate_factors)
    if not gate_df.empty:
        gate_df.insert(0, "result_type", "factor_gate")
        gate_df.insert(1, "research_topic", research_topic)
        gate_df.insert(2, "research_config_id", research_config_id)
        gate_df.insert(3, "output_stem", output_stem)
    benchmark_min_days = max(60, int(0.35 * max(daily_df["trade_date"].nunique(), 1)))
    market_benchmark = build_market_ew_benchmark(daily_df, args.start, end_date, min_days=benchmark_min_days).sort_index()

    summary_rows: list[dict[str, Any]] = []

    for idx, scenario in enumerate(selected_scenarios, start=1):
        print(f"[2/4] scenario {idx}/{len(selected_scenarios)}: {scenario.key} ({scenario.label})", flush=True)
        cfg, config_source = load_config(scenario.config_path)
        signals = cfg.get("signals", {})
        portfolio_cfg = cfg.get("portfolio", {})
        backtest_cfg = cfg.get("backtest", {})
        risk_cfg = cfg.get("risk", {}) or {}
        prefilter_cfg = cfg.get("prefilter", {}) or {}
        regime_cfg = cfg.get("regime", {}) or {}
        uf_cfg = cfg.get("universe_filter", {}) or {}

        ce_weights = normalize_weights(signals.get("composite_extended", {}))
        scenario_factors = attach_universe_filter(
            factors,
            daily_df,
            enabled=bool(uf_cfg.get("enabled", False)),
            min_amount_20d=float(uf_cfg.get("min_amount_20d", 50_000_000)),
            require_roe_ttm_positive=bool(uf_cfg.get("require_roe_ttm_positive", True)),
        )

        regime_overrides: dict[pd.Timestamp, dict[str, float]] = {}
        if bool(regime_cfg.get("enabled", True)):
            regime_overrides, _ = build_regime_weight_overrides(
                scenario_factors,
                daily_df,
                ce_weights,
                benchmark_symbol=str(risk_cfg.get("benchmark_symbol", "market_ew_proxy")),
                regime_cfg_raw=regime_cfg,
                market_ew_min_days=439,
            )

        score_df = build_score(
            scenario_factors,
            ce_weights,
            weights_by_date=regime_overrides if regime_overrides else None,
        )
        print(f"  score_df={score_df.shape}", flush=True)

        industry_cap_count, industry_map, _ = resolve_industry_cap_and_map(
            int(portfolio_cfg.get("industry_cap_count", 5)),
            "data/cache/industry_map.csv",
        )
        weights = build_topk_weights(
            score_df=score_df,
            factor_df=scenario_factors,
            daily_df=daily_df,
            top_k=int(signals.get("top_k", 20)),
            rebalance_rule=str(backtest_cfg.get("eval_rebalance_rule", "M")),
            prefilter_cfg=prefilter_cfg,
            max_turnover=float(portfolio_cfg.get("max_turnover", 1.0)),
            industry_map=industry_map,
            industry_cap_count=industry_cap_count,
            portfolio_method=str(portfolio_cfg.get("weight_method", "equal_weight")),
            cov_lookback_days=int(portfolio_cfg.get("cov_lookback_days", 252)),
            cov_ridge=float(portfolio_cfg.get("cov_ridge", 1e-6)),
            cov_shrinkage=str(portfolio_cfg.get("cov_shrinkage", "ledoit_wolf")).lower(),
            cov_ewma_halflife=float(portfolio_cfg.get("cov_ewma_halflife", 20.0)),
            risk_aversion=float(portfolio_cfg.get("risk_aversion", 1.0)),
        )
        weights = weights[weights.index >= pd.Timestamp(args.start)]
        print(f"  weights={weights.shape}", flush=True)

        target_cols = sorted({str(col).zfill(6) for col in weights.columns})
        execution_mode = str(backtest_cfg.get("execution_mode", "tplus1_open")).lower().strip()
        if execution_mode == "tplus1_open":
            open_returns = build_open_to_open_returns(daily_df, zero_if_limit_up_open=False)
            open_returns = open_returns.sort_index()
            open_returns.index = pd.to_datetime(open_returns.index)
            asset_returns = open_returns[
                (open_returns.index >= pd.Timestamp(args.start)) & (open_returns.index <= pd.Timestamp(end_date))
            ]
        else:
            asset_returns = build_asset_returns(daily_df, target_cols, args.start, end_date)
        asset_returns = asset_returns.reindex(columns=target_cols).fillna(0.0)
        weights = weights.reindex(columns=target_cols, fill_value=0.0)

        first_reb = weights.index.min()
        if not asset_returns.empty and weights.index.min() > asset_returns.index.min():
            seed = weights.iloc[[0]].copy()
            seed.index = pd.DatetimeIndex([asset_returns.index.min()])
            weights = pd.concat([seed, weights], axis=0)
            weights = weights[~weights.index.duplicated(keep="last")].sort_index()
        asset_returns = asset_returns[asset_returns.index >= first_reb]

        costs = transaction_cost_params_from_mapping(cfg.get("transaction_costs", {}))
        bt_no_cost = BacktestConfig(cost_params=None, execution_mode=execution_mode, execution_lag=1)
        bt_cost = BacktestConfig(cost_params=costs, execution_mode=execution_mode, execution_lag=1)
        res_nc = run_backtest(asset_returns, weights, config=bt_no_cost)
        res_wc = run_backtest(asset_returns, weights, config=bt_cost)

        rolling = rolling_walk_forward_windows(
            asset_returns.index,
            train_days=max(20, int(args.wf_train_window)),
            test_days=max(5, int(args.wf_test_window)),
            step_days=max(5, int(args.wf_step_window)),
        )
        _, wf_detail, wf_agg = walk_forward_backtest(asset_returns, weights, rolling, config=bt_cost, use_test_only=True)
        slices = contiguous_time_splits(
            asset_returns.index,
            n_splits=max(2, int(args.wf_slice_splits)),
            min_train_days=max(20, int(args.wf_slice_min_train_days)),
            expanding_window=not bool(args.wf_slice_fixed_window),
        )
        _, sp_detail, sp_agg = walk_forward_backtest(asset_returns, weights, slices, config=bt_cost, use_test_only=True)
        yearly_excess_df, benchmark_summary = _summarize_relative_to_benchmark(
            res_wc.daily_returns,
            market_benchmark,
            key_years=benchmark_key_years,
        )
        rolling_excess = _summarize_oos_excess(res_wc.daily_returns, market_benchmark, rolling)
        slice_excess = _summarize_oos_excess(res_wc.daily_returns, market_benchmark, slices)

        summary_rows.append(
            {
                "scenario": scenario.label,
                "scenario_key": scenario.key,
                "candidate_factor": scenario.candidate_factor,
                "family": scenario.family,
                "is_baseline": scenario.is_baseline,
                "result_type": "factor_admission_summary",
                "research_topic": research_topic,
                "research_config_id": research_config_id,
                "output_stem": output_stem,
                "config_source": config_source,
                "annualized_return": float(res_wc.panel.annualized_return),
                "sharpe_ratio": float(res_wc.panel.sharpe_ratio),
                "max_drawdown": float(res_wc.panel.max_drawdown),
                "turnover_mean": float(res_wc.panel.turnover_mean),
                "rolling_oos_median_ann_return": float(wf_agg.get("median_ann_return", np.nan)),
                "slice_oos_median_ann_return": float(sp_agg.get("median_ann_return", np.nan)),
                "annualized_excess_vs_market": float(benchmark_summary.get("annualized_excess_return", np.nan)),
                "yearly_excess_median_vs_market": float(benchmark_summary.get("yearly_excess_median", np.nan)),
                "key_year_excess_mean_vs_market": float(benchmark_summary.get("key_year_excess_mean", np.nan)),
                "key_year_excess_worst_vs_market": float(benchmark_summary.get("key_year_excess_worst", np.nan)),
                "rolling_oos_median_ann_excess_vs_market": float(
                    rolling_excess.get("median_ann_excess_return", np.nan)
                ),
                "slice_oos_median_ann_excess_vs_market": float(slice_excess.get("median_ann_excess_return", np.nan)),
            }
        )

        payload = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "result_type": "factor_admission_scenario",
            "research_topic": research_topic,
            "research_config_id": research_config_id,
            "output_stem": output_stem,
            "scenario_key": scenario.key,
            "scenario_label": scenario.label,
            "config_source": config_source,
            "parameters": {
                "start": args.start,
                "end": end_date,
                "weights": ce_weights,
                "prefilter": prefilter_cfg,
                "universe_filter": uf_cfg,
            },
            "full_sample": {
                "no_cost": _json_sanitize(res_nc.panel.to_dict()),
                "with_cost": _json_sanitize(res_wc.panel.to_dict()),
            },
            "walk_forward_rolling": {
                "detail": _json_sanitize(wf_detail.to_dict(orient="records")),
                "agg": _json_sanitize(wf_agg),
            },
            "walk_forward_slices": {
                "detail": _json_sanitize(sp_detail.to_dict(orient="records")),
                "agg": _json_sanitize(sp_agg),
                "full_vs_slices": _json_sanitize(compare_full_vs_slices(res_wc.panel, sp_agg) if sp_agg else {}),
            },
            "benchmark_first": {
                "benchmark_symbol": "market_ew_proxy",
                "benchmark_min_history_days": benchmark_min_days,
                "key_years": benchmark_key_years,
                "summary": _json_sanitize(benchmark_summary),
                "yearly_detail": _json_sanitize(yearly_excess_df.to_dict(orient="records")),
                "rolling_oos_excess": _json_sanitize(rolling_excess),
                "slice_oos_excess": _json_sanitize(slice_excess),
            },
        }
        report_path = results_dir / f"{output_prefix}_{scenario.key}.json"
        report_path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {scenario.label}: {report_path.relative_to(PROJECT_ROOT)}", flush=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("scenario").reset_index(drop=True)
    admission_df = build_admission_table(
        summary_df,
        gate_df,
        baseline_label=baseline_label,
        f1_min_ic=float(args.f1_min_ic),
        f1_min_t=float(args.f1_min_t),
        benchmark_key_years=benchmark_key_years,
    )

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    gate_path = results_dir / f"{output_prefix}_factor_gate.csv"
    admission_path = results_dir / f"{output_prefix}_admission.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    admission_df.to_csv(admission_path, index=False, encoding="utf-8-sig")

    doc_path = docs_dir / f"{output_prefix}.md"
    doc_path.write_text(
        _build_doc(
            summary_df,
            gate_df,
            admission_df,
            output_prefix,
            candidate_factors=candidate_factors,
        ),
        encoding="utf-8",
    )
    print(f"[3/4] summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] gate: {gate_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] admission: {admission_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"[3/4] doc: {doc_path.relative_to(PROJECT_ROOT)}", flush=True)

    # --- standard research contract ---
    duration_sec = round(time.perf_counter() - started_at, 6)

    def _project_relative(p: str | Path) -> str:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT.resolve()))

    manifest_path = results_dir / f"{output_prefix}_manifest.json"
    identity = make_research_identity(
        result_type="factor_admission_validation",
        research_topic=research_topic,
        research_config_id=research_config_id,
        output_stem=output_stem,
    )
    data_slice = DataSlice(
        dataset_name="factor_admission_validation_backtest",
        source_tables=("a_share_daily",),
        date_start=args.start,
        date_end=end_date,
        asof_trade_date=end_date,
        signal_date_col="trade_date",
        symbol_col="symbol",
        candidate_pool_version="universe_filtered",
        rebalance_rule="M",
        execution_mode="tplus1_open",
        label_return_mode="open_to_open",
        feature_set_id=None,
        feature_columns=tuple(candidate_factors),
        label_columns=(),
        pit_policy="signal_date_close_visible_only",
        config_path=str(selected_scenarios[0].config_path) if selected_scenarios else "",
        extra={
            "selected_families": selected_families,
            "lookback_days": int(args.lookback_days),
            "min_hist_days": int(args.min_hist_days),
        },
    )
    artifact_refs = (
        ArtifactRef("summary_csv", _project_relative(summary_path), "csv", False, "准入验证汇总"),
        ArtifactRef("gate_csv", _project_relative(gate_path), "csv", False, "准入 gate 表"),
        ArtifactRef("admission_csv", _project_relative(admission_path), "csv", False, "准入判定表"),
        ArtifactRef("report_md", _project_relative(doc_path), "md", False, "准入验证报告"),
        ArtifactRef("manifest_json", _project_relative(manifest_path), "json", False),
    ) + tuple(
        ArtifactRef(
            f"scenario_{slugify_token(s.key)}_json",
            _project_relative(results_dir / f"{output_prefix}_{s.key}.json"),
            "json",
            False,
            f"场景 {s.label} 回测详情",
        )
        for s in selected_scenarios
    )
    pass_count = int((admission_df["admission_status"] == "pass").sum()) if "admission_status" in admission_df.columns else 0
    metrics = {
        "scenario_count": len(selected_scenarios),
        "candidate_count": len(candidate_factors),
        "admission_pass_count": pass_count,
        "selected_families": selected_families,
    }
    gates = {
        "data_gate": {
            "passed": bool(daily_df is not None and len(daily_df) > 0),
            "daily_rows": int(len(daily_df)) if daily_df is not None else 0,
        },
        "execution_gate": {
            "passed": bool(len(summary_rows) == len(selected_scenarios)),
            "expected": len(selected_scenarios),
            "completed": len(summary_rows),
        },
        "governance_gate": {
            "passed": True,
            "manifest_schema": "research_result_v1",
        },
    }
    result = ExperimentResult(
        result_id=build_result_id(identity, [data_slice], metrics),
        identity=identity,
        script_name=_project_relative(Path(__file__).resolve()),
        command=shlex.join([sys.executable, *sys.argv]),
        created_at=utc_now_iso(),
        duration_sec=duration_sec,
        seed=None,
        data_slices=(data_slice,),
        config={"config_path": str(selected_scenarios[0].config_path) if selected_scenarios else ""},
        params={
            "cli": {k: str(v) for k, v in vars(args).items()},
            "scenarios": [s.__dict__ for s in selected_scenarios],
        },
        metrics=metrics,
        gates=gates,
        artifacts=artifact_refs,
        promotion={
            "production_eligible": False,
            "registry_status": "not_registered",
            "blocking_reasons": ["factor_admission_validation_is_diagnostic_research_only"],
        },
        notes=f"Factor admission validation: {len(selected_scenarios)} scenarios, {len(candidate_factors)} candidates.",
    )
    write_research_manifest(
        manifest_path,
        result,
        extra={
            "generated_at_utc": result.created_at,
            "research_topic": research_topic,
            "research_config_id": research_config_id,
            "output_stem": output_stem,
            "selected_families": selected_families,
            "scenarios": [s.__dict__ for s in selected_scenarios],
            "legacy_artifacts": [
                _project_relative(summary_path),
                _project_relative(gate_path),
                _project_relative(admission_path),
                _project_relative(doc_path),
            ]
            + [_project_relative(results_dir / f"{output_prefix}_{s.key}.json") for s in selected_scenarios],
        },
    )
    append_experiment_result(results_dir.parent / "experiments", result)
    # --- end standard research contract ---

    print(f"[4/4] manifest: {manifest_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
