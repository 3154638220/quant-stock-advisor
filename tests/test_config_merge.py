"""配置文件合并语义（composite_extended 须整表覆盖）。"""

from __future__ import annotations

import pytest

try:
    from scripts.run_backtest_eval import DEFAULT_CONFIG, _deep_merge, load_config
except ImportError:
    DEFAULT_CONFIG = {}
    _deep_merge = None  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]
from src.settings import resolve_config_path


def test_deep_merge_signals_composite_extended_full_replace():
    if _deep_merge is None:
        pytest.skip("scripts.run_backtest_eval 不可用")
    loaded = {"signals": {"composite_extended": {"ocf_to_net_profit": 0.5, "vol_to_turnover": 0.5}}}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    ce = merged["signals"]["composite_extended"]
    assert set(ce.keys()) == {"ocf_to_net_profit", "vol_to_turnover"}
    assert ce["ocf_to_net_profit"] == 0.5
    assert "momentum" not in ce


def test_default_config_exposes_weekly_kdj_weights():
    if not DEFAULT_CONFIG:
        pytest.skip("scripts.run_backtest_eval 不可用")
    ce = DEFAULT_CONFIG["signals"]["composite_extended"]
    assert "weekly_kdj_j" not in ce
    assert "weekly_kdj_oversold_depth" not in ce
    assert "weekly_kdj_rebound" not in ce


def test_legacy_backtest_snapshot_name_resolves_to_configs_dir():
    if load_config is None:
        pytest.skip("scripts.run_backtest_eval 不可用")
    legacy_name = "config.yaml.backtest.r7_s2_prefilter_off_universe_on"

    resolved = resolve_config_path(legacy_name)
    cfg, source = load_config(legacy_name)

    assert resolved.as_posix().endswith(f"configs/backtests/{legacy_name}")
    assert source.endswith(f"configs/backtests/{legacy_name}")
    assert cfg["signals"]["top_k"] == 20
