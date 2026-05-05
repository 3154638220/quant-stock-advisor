"""Shared CLI path-resolution helpers used by scripts/ runners.

Extracted from duplicated definitions across scripts/run_monthly_selection_*.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# Root is two levels up from src/pipeline/
ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def project_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def resolve_loaded_config_path(config_arg: Path | None) -> Path | None:
    from src.settings import config_path_candidates, resolve_config_path

    if config_arg is not None:
        return resolve_config_path(config_arg)
    candidates: list[Path] = []
    env_path = os.environ.get("QUANT_CONFIG", "").strip()
    if env_path:
        candidates.extend(config_path_candidates(env_path))
    candidates.extend([ROOT / "config.yaml", ROOT / "config.yaml.example"])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def parse_float_list(raw: str) -> list[float]:
    return sorted({float(x.strip()) for x in str(raw).split(",") if x.strip()})


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]
