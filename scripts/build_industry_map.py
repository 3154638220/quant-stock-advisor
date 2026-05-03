#!/usr/bin/env python3
"""Build and quality-check the canonical A-share industry map.

Outputs:
- data/cache/industry_map.csv
- data/results/industry_map_quality_YYYY-MM-DD_summary.csv
- docs/industry_map_quality_YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:  # AkShare is only needed for live fetching; unit tests use local helpers.
    from akshare.utils.cons import headers
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime package
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLUMNS = ["symbol", "industry", "industry_level1", "industry_level2", "source", "asof_date"]
REAL_SOURCE = "akshare.stock_board_industry_cons_em"
SW_SOURCE = "akshare.sw_index_third_cons"
SW_OFFICIAL_SOURCE = "akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo"
FALLBACK_SOURCE = "fallback_only_for_diagnostic"


@dataclass(frozen=True)
class IndustryMapQuality:
    asof_date: str
    source: str
    universe_source: str
    universe_count: int
    mapped_universe_count: int
    coverage_ratio: float
    unknown_count: int
    unknown_ratio: float
    duplicate_symbol_count: int
    industry_count: int
    min_industry_size: int
    median_industry_size: float
    max_industry_size: int
    fallback_used: bool
    pass_coverage_90pct: bool
    pass_no_duplicate_symbols: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "asof_date": self.asof_date,
            "source": self.source,
            "universe_source": self.universe_source,
            "universe_count": self.universe_count,
            "mapped_universe_count": self.mapped_universe_count,
            "coverage_ratio": self.coverage_ratio,
            "unknown_count": self.unknown_count,
            "unknown_ratio": self.unknown_ratio,
            "duplicate_symbol_count": self.duplicate_symbol_count,
            "industry_count": self.industry_count,
            "min_industry_size": self.min_industry_size,
            "median_industry_size": self.median_industry_size,
            "max_industry_size": self.max_industry_size,
            "fallback_used": self.fallback_used,
            "pass_coverage_90pct": self.pass_coverage_90pct,
            "pass_no_duplicate_symbols": self.pass_no_duplicate_symbols,
        }


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_symbol(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
    return digits.zfill(6) if digits else ""


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"缺少预期列，候选={candidates}, 实际={list(df.columns)}")


def _valid_symbol_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.fullmatch(r"\d{6}").fillna(False)


def load_current_universe(*, duckdb_path: str | Path, universe_json: str | Path) -> tuple[list[str], str]:
    """Load current universe from DuckDB latest date, falling back to local universe cache."""
    db = _project_path(duckdb_path)
    if db.exists():
        try:
            import duckdb

            with duckdb.connect(str(db), read_only=True) as con:
                tab = con.execute(
                    """
                    SELECT DISTINCT CAST(symbol AS VARCHAR) AS symbol
                    FROM a_share_daily
                    WHERE trade_date = (SELECT max(trade_date) FROM a_share_daily)
                    """
                ).fetchdf()
            symbols = sorted({normalize_symbol(s) for s in tab["symbol"].tolist() if normalize_symbol(s)})
            if symbols:
                return symbols, f"duckdb_latest:{db}"
        except Exception:
            pass

    cache = _project_path(universe_json)
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        raw = payload.get("symbols", []) if isinstance(payload, dict) else []
        symbols = sorted({normalize_symbol(s) for s in raw if normalize_symbol(s)})
        if symbols:
            return symbols, f"universe_json:{cache}"

    return [], "empty"


def _call_with_retry(fn: Any, *, retries: int, delay_sec: float) -> Any:
    last_exc: Exception | None = None
    for i in range(max(1, int(retries))):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i + 1 < max(1, int(retries)) and delay_sec > 0:
                time.sleep(delay_sec)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry helper called without attempts")


def fetch_industry_mapping(
    sleep_sec: float = 0.2,
    asof_date: str | None = None,
    retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> pd.DataFrame:
    """Fetch real industry membership from Eastmoney industry boards via akshare."""
    import akshare as ak

    asof = asof_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    boards = _call_with_retry(
        ak.stock_board_industry_name_em,
        retries=retries,
        delay_sec=retry_delay_sec,
    )
    board_col = _pick_col(boards, ["板块名称", "name"])
    rows: list[dict[str, str]] = []
    for board in boards[board_col].dropna().astype(str).tolist():
        industry = board.strip()
        if not industry:
            continue
        try:
            cons = _call_with_retry(
                lambda: ak.stock_board_industry_cons_em(symbol=industry),
                retries=retries,
                delay_sec=retry_delay_sec,
            )
        except Exception:
            continue
        code_col = _pick_col(cons, ["代码", "symbol"])
        for sym in cons[code_col].dropna().tolist():
            s6 = normalize_symbol(sym)
            if s6 and len(s6) == 6 and s6.isdigit():
                rows.append(
                    {
                        "symbol": s6,
                        "industry": industry,
                        "industry_level1": industry,
                        "industry_level2": industry,
                        "source": REAL_SOURCE,
                        "asof_date": asof,
                    }
                )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


def fetch_sw_industry_mapping(
    sleep_sec: float = 0.2,
    asof_date: str | None = None,
    retries: int = 3,
    retry_delay_sec: float = 2.0,
    request_timeout_sec: float = 12.0,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch real Shenwan level-1/2/3 industry membership via akshare."""
    import akshare as ak

    asof = asof_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    third = _call_with_retry(
        ak.sw_index_third_info,
        retries=retries,
        delay_sec=retry_delay_sec,
    )
    second = _call_with_retry(
        ak.sw_index_second_info,
        retries=retries,
        delay_sec=retry_delay_sec,
    )
    second_to_first = {}
    if {"行业名称", "上级行业"}.issubset(second.columns):
        second_to_first = dict(zip(second["行业名称"].astype(str), second["上级行业"].astype(str)))

    tasks: list[tuple[str, str, str, str, int]] = []
    for _, r in third.iterrows():
        code = str(r.get("行业代码", "")).strip()
        level3 = str(r.get("行业名称", "")).strip()
        level2 = str(r.get("上级行业", "")).strip()
        level1 = second_to_first.get(level2, level2)
        expected_count = int(pd.to_numeric(r.get("成份个数", 0), errors="coerce") or 0)
        if not code or not level3:
            continue
        tasks.append((code, level1, level2, level3, expected_count))

    def _fetch_one(task: tuple[str, str, str, str, int]) -> list[dict[str, str]]:
        code, level1, level2, level3, expected_count = task
        url = f"https://legulegu.com/stockdata/index-composition?industryCode={code}"
        local_rows: list[dict[str, str]] = []
        last_exc: Exception | None = None
        for i in range(max(1, int(retries))):
            try:
                r = requests.get(url, headers=headers, timeout=float(request_timeout_sec))
                r.raise_for_status()
                tables = pd.read_html(StringIO(r.text))
                if not tables:
                    raise RuntimeError("no html tables found")
                cons = tables[0].copy()
                if len(cons.columns) >= 3:
                    cons.columns = ["序号", "股票代码", "股票简称", *[f"extra_{j}" for j in range(len(cons.columns) - 3)]]
                if "股票代码" not in cons.columns:
                    raise RuntimeError(f"missing 股票代码 column: {list(cons.columns)}")
                for sym in cons["股票代码"].dropna().tolist():
                    s6 = normalize_symbol(sym)
                    if s6 and len(s6) == 6 and s6.isdigit():
                        local_rows.append(
                            {
                                "symbol": s6,
                                "industry": level1,
                                "industry_level1": level1,
                                "industry_level2": level2,
                                "source": SW_SOURCE,
                                "asof_date": asof,
                            }
                        )
                if expected_count > 0 and len(local_rows) == 0:
                    raise RuntimeError(f"empty constituents, expected {expected_count}")
                return local_rows
            except Exception as exc:
                last_exc = exc
                if i + 1 < max(1, int(retries)) and retry_delay_sec > 0:
                    time.sleep(retry_delay_sec)
        raise RuntimeError(f"sw cons fetch failed after retries: {code} {level3}: {last_exc}")

    rows: list[dict[str, str]] = []
    workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_fetch_one, task): task for task in tasks}
        done = 0
        for future in concurrent.futures.as_completed(future_map):
            rows.extend(future.result())
            done += 1
            if done == 1 or done % 25 == 0 or done == len(tasks):
                print(f"[sw] fetched industries {done}/{len(tasks)} rows={len(rows)}", flush=True)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    if not rows:
        raise RuntimeError("Shenwan industry fetch returned zero rows")
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


def _build_industry_name_lookup(category: pd.DataFrame) -> dict[str, dict[str, str]]:
    if not {"类目编码", "类目名称", "父类编码", "分级"}.issubset(category.columns):
        raise ValueError(f"unexpected category columns: {list(category.columns)}")
    cat = category.copy()
    cat["类目编码"] = cat["类目编码"].astype(str).str.strip()
    cat["父类编码"] = cat["父类编码"].astype(str).str.strip()
    cat["类目名称"] = cat["类目名称"].astype(str).str.strip()
    code_to_name = dict(zip(cat["类目编码"], cat["类目名称"]))
    code_to_parent = dict(zip(cat["类目编码"], cat["父类编码"]))
    code_to_level = dict(zip(cat["类目编码"], pd.to_numeric(cat["分级"], errors="coerce").fillna(-1).astype(int)))

    out: dict[str, dict[str, str]] = {}
    for code, level in code_to_level.items():
        if level < 1:
            continue
        chain = [code]
        cur = code
        while cur in code_to_parent and code_to_parent[cur] and code_to_parent[cur] != cur:
            cur = code_to_parent[cur]
            chain.append(cur)
            if cur == "S":
                break
        by_level = {code_to_level.get(c, -1): code_to_name.get(c, "") for c in chain}
        out[code] = {
            "industry_level1": by_level.get(1, ""),
            "industry_level2": by_level.get(2, ""),
            "industry_level3": by_level.get(3, ""),
        }
    return out


def fetch_sw_official_industry_mapping(
    asof_date: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch official Shenwan stock industry history and map it to readable levels."""
    import akshare as ak

    asof = asof_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    asof_ts = pd.Timestamp(asof).normalize()
    hist = ak.stock_industry_clf_hist_sw()
    category = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
    lookup = _build_industry_name_lookup(category)

    required = {"symbol", "start_date", "industry_code"}
    if not required.issubset(hist.columns):
        raise ValueError(f"unexpected SW official columns: {list(hist.columns)}")
    tab = hist.copy()
    tab["symbol"] = tab["symbol"].map(normalize_symbol)
    tab["start_date"] = pd.to_datetime(tab["start_date"], errors="coerce")
    tab["industry_code"] = tab["industry_code"].astype(str).str.strip()
    tab = tab[_valid_symbol_mask(tab["symbol"]) & tab["start_date"].notna() & (tab["start_date"] <= asof_ts)].copy()
    if symbols is not None:
        symbol_set = {normalize_symbol(s) for s in symbols if normalize_symbol(s)}
        tab = tab[tab["symbol"].isin(symbol_set)].copy()
    tab = tab.sort_values(["symbol", "start_date"]).drop_duplicates("symbol", keep="last")
    if tab.empty:
        raise RuntimeError("official Shenwan industry history returned no usable rows")

    rows: list[dict[str, str]] = []
    missing_codes: set[str] = set()
    for r in tab.itertuples(index=False):
        code = f"S{str(r.industry_code).strip()}"
        info = lookup.get(code)
        if not info:
            missing_codes.add(code)
            continue
        level1 = info.get("industry_level1", "")
        level2 = info.get("industry_level2", "") or level1
        level3 = info.get("industry_level3", "") or level2
        if not level1:
            missing_codes.add(code)
            continue
        rows.append(
            {
                "symbol": str(r.symbol).zfill(6),
                "industry": level1,
                "industry_level1": level1,
                "industry_level2": level2,
                "source": SW_OFFICIAL_SOURCE,
                "asof_date": asof,
            }
        )
    if missing_codes:
        sample = ", ".join(sorted(missing_codes)[:10])
        raise RuntimeError(f"missing Shenwan category names for {len(missing_codes)} codes: {sample}")
    if not rows:
        raise RuntimeError("official Shenwan industry mapping returned zero rows")
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


def fetch_mapping_by_source(
    source: str,
    *,
    sleep_sec: float,
    asof_date: str,
    retries: int,
    retry_delay_sec: float,
    request_timeout_sec: float = 12.0,
    max_workers: int = 8,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    source_norm = source.strip().lower()
    if source_norm in {"sw_official", "shenwan_official", "official_sw"}:
        return fetch_sw_official_industry_mapping(asof_date=asof_date, symbols=symbols)
    if source_norm == "eastmoney":
        return fetch_industry_mapping(
            sleep_sec=sleep_sec,
            asof_date=asof_date,
            retries=retries,
            retry_delay_sec=retry_delay_sec,
        )
    if source_norm in {"sw", "shenwan"}:
        return fetch_sw_industry_mapping(
            sleep_sec=sleep_sec,
            asof_date=asof_date,
            retries=retries,
            retry_delay_sec=retry_delay_sec,
            request_timeout_sec=request_timeout_sec,
            max_workers=max_workers,
        )
    if source_norm != "auto":
        raise ValueError("--source must be one of: auto, sw_official, eastmoney, sw")
    try:
        return fetch_sw_official_industry_mapping(asof_date=asof_date, symbols=symbols)
    except Exception as exc:
        print(f"[warn] official Shenwan industry fetch failed, fallback to Eastmoney source: {exc}")
        return fetch_industry_mapping(
            sleep_sec=sleep_sec,
            asof_date=asof_date,
            retries=retries,
            retry_delay_sec=retry_delay_sec,
        )


def deduplicate_mapping(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), 0
    tab = raw.copy()
    tab["symbol"] = tab["symbol"].map(normalize_symbol)
    for col in REQUIRED_COLUMNS:
        if col not in tab.columns:
            tab[col] = ""
        tab[col] = tab[col].astype(str).str.strip()
    tab = tab[_valid_symbol_mask(tab["symbol"]) & (tab["industry"] != "")].copy()
    duplicate_count = int(tab.duplicated("symbol", keep=False).sum())
    tab = tab.sort_values(["symbol", "industry"]).drop_duplicates("symbol", keep="first")
    return tab[REQUIRED_COLUMNS].sort_values("symbol").reset_index(drop=True), duplicate_count


def align_to_universe(mapping: pd.DataFrame, universe: list[str], asof_date: str) -> pd.DataFrame:
    """Add explicit unknown rows for current-universe symbols absent from the real map."""
    if not universe:
        return mapping.copy()
    base = pd.DataFrame({"symbol": sorted({normalize_symbol(s) for s in universe if normalize_symbol(s)})})
    out = base.merge(mapping, on="symbol", how="left")
    missing = out["industry"].isna() | (out["industry"].astype(str).str.strip() == "")
    out.loc[missing, "industry"] = "unknown"
    out.loc[missing, "industry_level1"] = "unknown"
    out.loc[missing, "industry_level2"] = "unknown"
    out.loc[missing, "source"] = FALLBACK_SOURCE
    out.loc[missing, "asof_date"] = asof_date
    for col in REQUIRED_COLUMNS:
        out[col] = out[col].astype(str).str.strip()
    return out[REQUIRED_COLUMNS].sort_values("symbol").reset_index(drop=True)


def quality_summary(mapping: pd.DataFrame, universe: list[str], universe_source: str, asof_date: str) -> IndustryMapQuality:
    symbols = sorted({normalize_symbol(s) for s in universe if normalize_symbol(s)})
    final = mapping.copy()
    final["symbol"] = final["symbol"].map(normalize_symbol)
    duplicate_symbol_count = int(final.duplicated("symbol", keep=False).sum())
    universe_set = set(symbols)
    in_universe = final[final["symbol"].isin(universe_set)].copy() if symbols else final.copy()
    known = in_universe["industry"].astype(str).str.strip().ne("unknown") & in_universe["industry"].astype(str).str.strip().ne("")
    mapped_count = int(known.sum())
    universe_count = int(len(symbols))
    unknown_count = int((~known).sum()) if universe_count else 0
    coverage = float(mapped_count / universe_count) if universe_count else 0.0
    unknown_ratio = float(unknown_count / universe_count) if universe_count else 0.0
    industry_sizes = final[final["industry"].astype(str) != "unknown"].groupby("industry")["symbol"].nunique()
    sources = sorted(set(final["source"].astype(str).str.strip()))
    source = ",".join(sources)
    return IndustryMapQuality(
        asof_date=asof_date,
        source=source,
        universe_source=universe_source,
        universe_count=universe_count,
        mapped_universe_count=mapped_count,
        coverage_ratio=coverage,
        unknown_count=unknown_count,
        unknown_ratio=unknown_ratio,
        duplicate_symbol_count=duplicate_symbol_count,
        industry_count=int(industry_sizes.shape[0]),
        min_industry_size=int(industry_sizes.min()) if not industry_sizes.empty else 0,
        median_industry_size=float(industry_sizes.median()) if not industry_sizes.empty else 0.0,
        max_industry_size=int(industry_sizes.max()) if not industry_sizes.empty else 0,
        fallback_used=bool((final["source"].astype(str) == FALLBACK_SOURCE).any()),
        pass_coverage_90pct=bool(coverage >= 0.90),
        pass_no_duplicate_symbols=bool(duplicate_symbol_count == 0),
    )


def build_quality_doc(quality: IndustryMapQuality, industry_sizes: pd.DataFrame, output_csv: Path) -> str:
    row = quality.to_row()
    lines = [
        f"# Industry Map Quality {quality.asof_date}",
        "",
        "## Summary",
        "",
        pd.DataFrame([row]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Checks",
        "",
        f"- 当前 universe 覆盖率：`{quality.coverage_ratio:.2%}`（要求 `>= 90%`）。",
        f"- unknown 比例：`{quality.unknown_ratio:.2%}`，unknown rows 使用 `{FALLBACK_SOURCE}` 标记，仅供诊断。",
        f"- 重复 symbol：`{quality.duplicate_symbol_count}`。",
        f"- PIT 说明：`source={quality.source}`，`asof_date={quality.asof_date}`。",
        f"- 输出文件：`{output_csv.relative_to(PROJECT_ROOT) if output_csv.is_relative_to(PROJECT_ROOT) else output_csv}`。",
        "",
        "## Industry Width",
        "",
    ]
    if industry_sizes.empty:
        lines.append("_无行业宽度统计_")
    else:
        lines.append(industry_sizes.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建真实行业映射并输出质量报告")
    p.add_argument("--output", default="data/cache/industry_map.csv")
    p.add_argument("--duckdb-path", default="data/market.duckdb")
    p.add_argument("--universe-json", default="data/cache/universe_symbols.json")
    p.add_argument("--sleep-sec", type=float, default=0.2)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-delay-sec", type=float, default=2.0)
    p.add_argument("--request-timeout-sec", type=float, default=12.0)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--source", default="auto", choices=["auto", "sw_official", "eastmoney", "sw", "shenwan"])
    p.add_argument("--asof-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--allow-low-coverage", action="store_true", help="覆盖率不足 90% 时仍返回 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asof_date = str(args.asof_date)
    output = _project_path(args.output)
    results_dir = PROJECT_ROOT / "data" / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    universe, universe_source = load_current_universe(duckdb_path=args.duckdb_path, universe_json=args.universe_json)
    raw = fetch_mapping_by_source(
        args.source,
        sleep_sec=max(float(args.sleep_sec), 0.0),
        asof_date=asof_date,
        retries=max(int(args.retries), 1),
        retry_delay_sec=max(float(args.retry_delay_sec), 0.0),
        request_timeout_sec=max(float(args.request_timeout_sec), 1.0),
        max_workers=max(int(args.max_workers), 1),
        symbols=universe,
    )
    deduped, raw_duplicate_count = deduplicate_mapping(raw)
    final = align_to_universe(deduped, universe, asof_date)
    quality = quality_summary(final, universe, universe_source, asof_date)
    if raw_duplicate_count and quality.duplicate_symbol_count == 0:
        quality = IndustryMapQuality(**{**quality.to_row(), "duplicate_symbol_count": 0})

    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False, encoding="utf-8-sig")

    summary_path = results_dir / f"industry_map_quality_{asof_date}_summary.csv"
    pd.DataFrame([quality.to_row()]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    width = (
        final[final["industry"].astype(str) != "unknown"]
        .groupby("industry", as_index=False)
        .agg(symbol_count=("symbol", "nunique"))
        .sort_values(["symbol_count", "industry"], ascending=[False, True])
    )
    doc_path = docs_dir / f"industry_map_quality_{asof_date}.md"
    doc_path.write_text(build_quality_doc(quality, width, output), encoding="utf-8")

    print(f"industry_map -> {output} | rows={len(final)} | coverage={quality.coverage_ratio:.2%}")
    print(f"quality_summary -> {summary_path}")
    print(f"quality_doc -> {doc_path}")
    if (not quality.pass_coverage_90pct or not quality.pass_no_duplicate_symbols) and not args.allow_low_coverage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
