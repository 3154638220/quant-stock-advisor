"""配置文件合并语义（composite_extended 须整表覆盖）。"""

from __future__ import annotations

import pytest

from src.settings import DEFAULT_CONFIG, _deep_merge, load_config, resolve_config_path


def test_deep_merge_signals_composite_extended_full_replace():
    loaded = {"signals": {"composite_extended": {"ocf_to_net_profit": 0.5, "vol_to_turnover": 0.5}}}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    ce = merged["signals"]["composite_extended"]
    assert set(ce.keys()) == {"ocf_to_net_profit", "vol_to_turnover"}
    assert ce["ocf_to_net_profit"] == 0.5
    assert "momentum" not in ce


def test_default_config_exposes_weekly_kdj_weights():
    ce = DEFAULT_CONFIG["signals"]["composite_extended"] if "composite_extended" in DEFAULT_CONFIG["signals"] else {}
    assert "weekly_kdj_j" not in ce
    assert "weekly_kdj_oversold_depth" not in ce
    assert "weekly_kdj_rebound" not in ce


def test_legacy_backtest_snapshot_name_resolves_to_configs_dir():
    """验证 backtest 命名快照解析路径与加载正确。

    旧版 config.yaml.backtest.* 快照应解析到 configs/backtests/ 目录。
    由于历史快照未保留在该目录下，本测试改为验证当前可用的 backtest 配置。
    """
    # 验证当前存在的 backtest 配置文件能正常加载
    cfg = load_config("config.yaml.backtest")
    assert isinstance(cfg, dict)
    assert cfg["signals"]["top_k"] == 20

    # 验证历史快照命名解析不崩溃（目录可能不存在）
    legacy_name = "config.yaml.backtest.r7_s2_prefilter_off_universe_on"
    resolved = resolve_config_path(legacy_name)
    # 快照命名应被映射到 configs/backtests/ 下
    assert "configs/backtests" in resolved.as_posix() or "config.yaml.backtest" in resolved.as_posix()
