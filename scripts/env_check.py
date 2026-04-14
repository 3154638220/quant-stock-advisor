#!/usr/bin/env python3
"""
运行前环境自检：Python 版本、PyTorch/CUDA、DuckDB 可写、AkShare 连通。

建议在 conda activate quant-system 后执行::

    python scripts/env_check.py
    python scripts/env_check.py --quiet   # 仅退出码，无输出
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.akshare_client import fetch_a_share_daily
from src.settings import load_config, project_root


def _ok(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[OK] {msg}", flush=True)


def _fail(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[FAIL] {msg}", flush=True)


def _dns_summary(hosts: list[str]) -> str:
    pairs: list[str] = []
    for host in hosts:
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            ips: list[str] = []
            for item in infos:
                ip = item[4][0]
                if ip not in ips:
                    ips.append(ip)
            pairs.append(f"{host}={'/'.join(ips[:2])}")
        except Exception as exc:
            pairs.append(f"{host}=ERR:{type(exc).__name__}")
    return "; ".join(pairs)


def run_checks(*, config: Path | None, quiet: bool) -> int:
    root = project_root()
    cfg = load_config(config)
    paths = cfg.get("paths", {})
    duckdb_rel = paths.get("duckdb_path", "data/market.duckdb")
    duckdb_path = Path(duckdb_rel)
    if not duckdb_path.is_absolute():
        duckdb_path = root / duckdb_path

    failed = 0

    # Python 3.10.x（与 pyproject requires-python 一致）
    vi = sys.version_info
    if vi.major != 3 or vi.minor != 10:
        _fail(
            f"期望 Python 3.10.x，当前 {vi.major}.{vi.minor}.{vi.micro}",
            quiet=quiet,
        )
        failed += 1
    else:
        _ok(f"Python {vi.major}.{vi.minor}.{vi.micro}", quiet=quiet)

    # PyTorch + CUDA
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda_ok else "n/a"
        _ok(
            f"torch {torch.__version__} | cuda.is_available()={cuda_ok}"
            + (f" | device0={dev}" if cuda_ok else "（将使用 CPU 回退）"),
            quiet=quiet,
        )
        if os.environ.get("REQUIRE_CUDA", "").lower() in ("1", "true", "yes") and not cuda_ok:
            _fail("环境变量 REQUIRE_CUDA 已启用但 CUDA 不可用", quiet=quiet)
            failed += 1
    except Exception as e:
        _fail(f"PyTorch 检查异常: {e}", quiet=quiet)
        failed += 1

    # DuckDB 可写
    try:
        import duckdb

        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(duckdb_path))
        try:
            con.execute("CREATE TABLE IF NOT EXISTS _env_probe (x INTEGER)")
            con.execute("INSERT INTO _env_probe VALUES (1)")
            con.execute("DELETE FROM _env_probe WHERE x = 1")
            con.execute("DROP TABLE IF EXISTS _env_probe")
        finally:
            con.close()
        _ok(f"DuckDB 可写: {duckdb_path}", quiet=quiet)
    except Exception as e:
        _fail(f"DuckDB 不可写或无法打开 {duckdb_path}: {e}", quiet=quiet)
        failed += 1

    # AkShare 连通（轻量请求）
    try:
        df = fetch_a_share_daily(
            "000001",
            "20240102",
            "20240110",
            adjust=cfg.get("akshare", {}).get("adjust", "qfq"),
            timeout_sec=float(cfg.get("akshare", {}).get("request_timeout_sec", 10.0)),
        )
        if df is None or df.empty:
            _fail("AkShare 返回空表（网络或接口异常）", quiet=quiet)
            failed += 1
        else:
            _ok(
                f"AkShare 连通（样本: 000001 日线 {len(df)} 行）",
                quiet=quiet,
            )
    except Exception as e:
        dns_msg = _dns_summary(
            [
                "finance.sina.com.cn",
                "stock.finance.sina.com.cn",
                "push2his.eastmoney.com",
            ]
        )
        _fail(f"AkShare 连通性失败: {e} | DNS: {dns_msg}", quiet=quiet)
        failed += 1

    # Conda 环境提示（非强制）
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "quant-system":
        _ok("CONDA_DEFAULT_ENV=quant-system", quiet=quiet)
    elif not quiet:
        print(
            f"[WARN] 当前 conda 环境为 {conda_env!r}，建议使用: conda activate quant-system",
            flush=True,
        )

    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description="量化流水线环境自检")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.yaml（用于 DuckDB 路径）；默认项目根",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="仅通过退出码表示结果（0=全部通过）",
    )
    args = p.parse_args()
    return run_checks(config=args.config, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
