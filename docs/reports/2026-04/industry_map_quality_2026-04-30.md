# Industry Map Quality 2026-04-30

## Summary

| asof_date   | source                                                            | universe_source                                      |   universe_count |   mapped_universe_count |   coverage_ratio |   unknown_count |   unknown_ratio |   duplicate_symbol_count |   industry_count |   min_industry_size |   median_industry_size |   max_industry_size | fallback_used   | pass_coverage_90pct   | pass_no_duplicate_symbols   |
|:------------|:------------------------------------------------------------------|:-----------------------------------------------------|-----------------:|------------------------:|-----------------:|----------------:|----------------:|-------------------------:|-----------------:|--------------------:|-----------------------:|--------------------:|:----------------|:----------------------|:----------------------------|
| 2026-04-30  | akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo | duckdb_latest:/home/x12dpg/hjx/lh/data/market.duckdb |             5143 |                    5143 |           1.0000 |               0 |          0.0000 |                        0 |               31 |                  15 |               122.0000 |                 526 | False           | True                  | True                        |

## Checks

- 当前 universe 覆盖率：`100.00%`（要求 `>= 90%`）。
- unknown 比例：`0.00%`，unknown rows 使用 `fallback_only_for_diagnostic` 标记，仅供诊断。
- 重复 symbol：`0`。
- PIT 说明：`source=akshare.stock_industry_clf_hist_sw+stock_industry_category_cninfo`，`asof_date=2026-04-30`。
- 输出文件：`data/cache/industry_map.csv`。

## Industry Width

| industry   |   symbol_count |
|:-----------|---------------:|
| 机械设备   |            526 |
| 电子       |            476 |
| 医药生物   |            475 |
| 基础化工   |            407 |
| 电力设备   |            364 |
| 计算机     |            329 |
| 汽车       |            284 |
| 轻工制造   |            156 |
| 建筑装饰   |            152 |
| 有色金属   |            139 |
| 国防军工   |            136 |
| 公用事业   |            131 |
| 传媒       |            130 |
| 环保       |            130 |
| 交通运输   |            126 |
| 通信       |            122 |
| 食品饮料   |            122 |
| 农林牧渔   |            103 |
| 纺织服饰   |            102 |
| 商贸零售   |             98 |
| 房地产     |             98 |
| 家用电器   |             94 |
| 非银金融   |             80 |
| 社会服务   |             79 |
| 建筑材料   |             71 |
| 石油石化   |             47 |
| 钢铁       |             43 |
| 银行       |             42 |
| 煤炭       |             37 |
| 美容护理   |             29 |
| 综合       |             15 |
