"""保证从项目根目录以 ``src.*`` 导入。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.registry import FACTOR_REGISTRY  # noqa: E402


_INITIAL_FACTOR_ACTIVE_STATE = {
    name: spec.active for name, spec in FACTOR_REGISTRY.items()
}


def _restore_factor_registry_active_state() -> None:
    for name, active in _INITIAL_FACTOR_ACTIVE_STATE.items():
        if name in FACTOR_REGISTRY:
            FACTOR_REGISTRY[name].active = active


@pytest.fixture(autouse=True)
def isolate_factor_registry_active_state():
    """Prevent factor-governance tests from leaking global registry state."""

    _restore_factor_registry_active_state()
    yield
    _restore_factor_registry_active_state()
