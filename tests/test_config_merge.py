"""配置文件合并语义（composite_extended 须整表覆盖）。"""

from __future__ import annotations

from scripts.run_backtest_eval import DEFAULT_CONFIG, _deep_merge


def test_deep_merge_signals_composite_extended_full_replace():
    loaded = {"signals": {"composite_extended": {"ocf_to_net_profit": 0.5, "vol_to_turnover": 0.5}}}
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    ce = merged["signals"]["composite_extended"]
    assert set(ce.keys()) == {"ocf_to_net_profit", "vol_to_turnover"}
    assert ce["ocf_to_net_profit"] == 0.5
    assert "momentum" not in ce


def test_default_config_exposes_weekly_kdj_weights():
    ce = DEFAULT_CONFIG["signals"]["composite_extended"]
    assert "weekly_kdj_j" not in ce
    assert "weekly_kdj_oversold_depth" not in ce
    assert "weekly_kdj_rebound" not in ce
