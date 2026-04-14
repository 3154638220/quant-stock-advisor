#!/usr/bin/env python3
"""AkShare 网络诊断与可选 DNS 固化。"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_DNS = ["223.5.5.5", "119.29.29.29", "1.1.1.1"]
DEFAULT_FALLBACK_DNS = ["8.8.8.8", "1.0.0.1"]
DEFAULT_HOSTS = [
    "finance.sina.com.cn",
    "stock.finance.sina.com.cn",
    "push2his.eastmoney.com",
    "quote.eastmoney.com",
]


def _read_nameservers() -> list[str]:
    path = Path("/etc/resolv.conf")
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("nameserver "):
            out.append(line.split()[1])
    return out


def _resolve_hosts(hosts: list[str]) -> list[str]:
    rows: list[str] = []
    for host in hosts:
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            ips: list[str] = []
            for item in infos:
                ip = item[4][0]
                if ip not in ips:
                    ips.append(ip)
            rows.append(f"[OK] {host}: {', '.join(ips[:4])}")
        except Exception as exc:
            rows.append(f"[FAIL] {host}: {type(exc).__name__}: {exc}")
    return rows


def _apply_systemd_resolved(dns: list[str], fallback_dns: list[str]) -> None:
    if os.geteuid() != 0:
        raise PermissionError("应用系统 DNS 需要 root 权限")
    conf_dir = Path("/etc/systemd/resolved.conf.d")
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / "quant-system-akshare.conf"
    content = (
        "[Resolve]\n"
        f"DNS={' '.join(dns)}\n"
        f"FallbackDNS={' '.join(fallback_dns)}\n"
        "DNSStubListener=yes\n"
    )
    conf_path.write_text(content, encoding="utf-8")
    subprocess.run(["systemctl", "restart", "systemd-resolved"], check=True)
    subprocess.run(["resolvectl", "flush-caches"], check=False)
    print(f"[OK] 已写入 {conf_path}")


def _apply_docker_dns(dns: list[str]) -> None:
    if os.geteuid() != 0:
        raise PermissionError("更新 Docker DNS 需要 root 权限")
    path = Path("/etc/docker/daemon.json")
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"/etc/docker/daemon.json 不是合法 JSON: {exc}") from exc
    payload["dns"] = dns
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.run(["systemctl", "restart", "docker"], check=False)
    print(f"[OK] 已更新 {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="AkShare 网络诊断与 DNS 固化")
    p.add_argument("--apply-dns", action="store_true", help="写入 systemd-resolved 与 Docker DNS")
    p.add_argument(
        "--dns",
        default=",".join(DEFAULT_DNS),
        help="主 DNS，逗号分隔；默认 223.5.5.5,119.29.29.29,1.1.1.1",
    )
    p.add_argument(
        "--fallback-dns",
        default=",".join(DEFAULT_FALLBACK_DNS),
        help="Fallback DNS，逗号分隔",
    )
    p.add_argument(
        "--skip-docker",
        action="store_true",
        help="应用 DNS 时跳过 /etc/docker/daemon.json",
    )
    args = p.parse_args()

    dns = [x.strip() for x in args.dns.split(",") if x.strip()]
    fallback_dns = [x.strip() for x in args.fallback_dns.split(",") if x.strip()]

    print("[INFO] 当前 resolv.conf nameserver:", ", ".join(_read_nameservers()) or "(无)")
    for row in _resolve_hosts(DEFAULT_HOSTS):
        print(row)

    if not args.apply_dns:
        return 0

    try:
        _apply_systemd_resolved(dns, fallback_dns)
        if not args.skip_docker:
            _apply_docker_dns(dns)
    except Exception as exc:
        print(f"[FAIL] 应用 DNS 失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
