"""按 config 的 paths.logs_dir 配置根日志（控制台 + 按日滚动文件）。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_app_logging(
    logs_dir: Union[str, Path],
    *,
    name: str = "quant",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    初始化应用日志：控制台 UTF-8 + 可选 ``{name}_YYYYMMDD.log``。

    多次调用时若根 logger 已有 handler，则不再重复添加（便于脚本测试）。
    """
    root = logging.getLogger(name)
    root.setLevel(level)
    if root.handlers:
        return root

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)
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
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "quant")
