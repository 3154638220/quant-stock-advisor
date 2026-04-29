# Industry Map Quality 2026-04-28

## Summary

| asof_date   | source                                                            | universe_source                                      |   universe_count |   mapped_universe_count |   coverage_ratio |   unknown_count |   unknown_ratio |   duplicate_symbol_count |   industry_count |   min_industry_size |   median_industry_size |   max_industry_size | fallback_used   | pass_coverage_90pct   | pass_no_duplicate_symbols   |
|:------------|:------------------------------------------------------------------|:-----------------------------------------------------|-----------------:|------------------------:|-----------------:|----------------:|----------------:|-------------------------:|-----------------:|--------------------:|-----------------------:|--------------------:|:----------------|:----------------------|:----------------------------|
| 2026-04-28  | akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo | duckdb_latest:/home/x12dpg/hjx/lh/data/market.duckdb |             5184 |                    5184 |           1.0000 |               0 |          0.0000 |                        0 |               31 |                  15 |               123.0000 |                 533 | False           | True                  | True                        |

## Checks

- 当前 universe 覆盖率：`100.00%`（要求 `>= 90%`）。
- unknown 比例：`0.00%`，unknown rows 使用 `fallback_only_for_diagnostic` 标记，仅供诊断。
- 重复 symbol：`0`。
- PIT 说明：`source=akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`，`asof_date=2026-04-28`。
- 输出文件：`data/cache/industry_map.csv`。

## Industry Width

| industry   |   symbol_count |
|:-----------|---------------:|
| 机械设备   |            533 |
| 医药生物   |            479 |
| 电子       |            478 |
| 基础化工   |            408 |
| 电力设备   |            365 |
| 计算机     |            335 |
| 汽车       |            283 |
| 轻工制造   |            158 |
| 建筑装饰   |            156 |
| 国防军工   |            140 |
| 有色金属   |            139 |
| 环保       |            132 |
| 公用事业   |            131 |
| 传媒       |            130 |
| 交通运输   |            126 |
| 食品饮料   |            123 |
| 通信       |            122 |
| 农林牧渔   |            104 |
| 纺织服饰   |            104 |
| 商贸零售   |             99 |
| 房地产     |             98 |
| 家用电器   |             94 |
| 社会服务   |             81 |
| 非银金融   |             81 |
| 建筑材料   |             72 |
| 石油石化   |             47 |
| 钢铁       |             44 |
| 银行       |             42 |
| 煤炭       |             36 |
| 美容护理   |             29 |
| 综合       |             15 |
