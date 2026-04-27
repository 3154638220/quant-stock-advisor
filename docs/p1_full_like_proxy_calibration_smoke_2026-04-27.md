# P1 Full-Like Proxy Calibration Smoke

- 日期：`2026-04-27`
- 结果类型：`light_strategy_proxy`
- 研究主题：`p1_tree_groups`
- 目的：给 P1 light proxy 增加 `Top-20 + max_turnover=0.3` 的 full-like 诊断层，检查它是否比旧版无约束 Top-K proxy 更保守。

## 改动

1. 新增 `build_tree_turnover_aware_proxy_detail`，在验证集上按调仓期保留上一期持仓，只允许 `max_turnover` 对应比例换入新 Top-K。
2. `scripts/run_p1_tree_groups.py` 新增 `--proxy-max-turnover`，默认 `0.3`。
3. summary 新增 `full_like_proxy_*` 字段和 `proxy_gap_full_like_minus_unconstrained`。
4. detail 同时输出 `proxy_variant=topk_unconstrained` 与 `proxy_variant=full_like_turnover_aware`。

## Smoke 结果

命令：

```bash
python scripts/run_p1_tree_groups.py \
  --config config.yaml.backtest \
  --groups G0,G1 \
  --label-horizons 5,10,20 \
  --label-mode market_relative \
  --xgboost-objective rank \
  --proxy-horizon 5 \
  --rebalance-rule M \
  --top-k 20 \
  --proxy-max-turnover 0.3 \
  --val-frac 0.2 \
  --out-tag p1_proxy_calibration_marketrel
```

| group | old proxy excess | full-like proxy excess | gap | periods |
| --- | ---: | ---: | ---: | ---: |
| G0 | +3.04% | +1.42% | -1.62% | 2 |
| G1 | +16.42% | +9.74% | -6.68% | 2 |

## 判读

新版 full-like proxy 能把旧 proxy 明显压低，尤其对 `G1` 的压低更大，方向正确。但本轮验证窗口只有 `2` 个调仓期，只能视为实现 smoke，不能作为最终校准结论。

当前仍不能说明 full-like proxy 已经足以过滤 `market_relative + G1` 这类失败样本；下一步需要扩大验证窗口，或直接复用历史 bundle/score 面板做跨失败样本校准。

## 产物

- `data/results/p1_proxy_calibration_marketrel_rb_m_top20_lh_5-10-20_px_5_val20_lbl_market_relative_obj_rank_20260427_044634_summary.csv`
- `data/results/p1_proxy_calibration_marketrel_rb_m_top20_lh_5-10-20_px_5_val20_lbl_market_relative_obj_rank_20260427_044634_detail.csv`
- `data/results/p1_proxy_calibration_marketrel_rb_m_top20_lh_5-10-20_px_5_val20_lbl_market_relative_obj_rank_20260427_044634.json`

## 是否改变主计划

改变执行队列，但不改变研究结论：P1 仍不 promotion。下一步从“设计 full-like proxy”转为“扩大校准窗口，验证 full-like proxy 能否提前识别已知失败样本”。
