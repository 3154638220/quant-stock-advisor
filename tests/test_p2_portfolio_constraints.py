from __future__ import annotations

import pandas as pd

from scripts.run_backtest_eval import _pick_topk_with_industry_cap, resolve_industry_cap_and_map


def test_resolve_industry_cap_and_map_silent_downgrade_when_map_missing(tmp_path):
    missing_csv = tmp_path / "not_found.csv"
    cap, industry_map, status = resolve_industry_cap_and_map(5, str(missing_csv))
    assert cap == 0
    assert industry_map == {}
    assert status == "disabled_missing_map"


def test_resolve_industry_cap_and_map_loads_map_when_available(tmp_path):
    csv_path = tmp_path / "industry_map.csv"
    csv_path.write_text("symbol,industry\n000001,银行\n000002,银行\n000003,有色\n", encoding="utf-8")
    cap, industry_map, status = resolve_industry_cap_and_map(5, str(csv_path))
    assert cap == 5
    assert industry_map["000001"] == "银行"
    assert industry_map["000003"] == "有色"
    assert status == "enabled"


def test_pick_topk_with_industry_cap_respects_per_industry_count():
    day_df = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
            "score": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        }
    )
    industry_map = {
        "000001": "A",
        "000002": "A",
        "000003": "A",
        "000004": "B",
        "000005": "B",
        "000006": "C",
    }
    topk = _pick_topk_with_industry_cap(
        day_df,
        top_k=4,
        industry_map=industry_map,
        industry_cap_count=1,
    )
    picked = topk["symbol"].astype(str).tolist()
    inds = [industry_map[s] for s in picked]
    assert len(picked) == 3
    assert inds.count("A") <= 1
    assert inds.count("B") <= 1
