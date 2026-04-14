"""按配置初始化日志（控制台 + 按日文件，支持 JSON 结构化输出）。"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class JsonLineFormatter(logging.Formatter):
    """单行 JSON 日志格式，便于 ELK / DuckDB 等系统直接消费。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_log_format(log_format: Optional[str]) -> str:
    env = os.environ.get("QUANT_LOG_FORMAT", "").strip().lower()
    if env in ("json", "text"):
        return env
    fmt = str(log_format or "json").strip().lower()
    return fmt if fmt in ("json", "text") else "json"


def setup_app_logging(
    logs_dir: Union[str, Path],
    *,
    name: str = "quant",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_format: str = "json",
) -> logging.Logger:
    """
    初始化应用日志：控制台 UTF-8 + 可选 ``{name}_YYYYMMDD.log``。

    多次调用时若根 logger 已有 handler，则不再重复添加（便于脚本测试）。
    """
    root = logging.getLogger(name)
    root.setLevel(level)
    if root.handlers:
        return root

    text_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter: logging.Formatter = (
        JsonLineFormatter() if _resolve_log_format(log_format) == "json" else text_fmt
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    if log_to_file:
        log_path = Path(logs_dir)
        if not log_path.is_absolute():
            from .settings import project_root

            log_path = project_root() / log_path
        log_path.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_path / stem, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "quant")
