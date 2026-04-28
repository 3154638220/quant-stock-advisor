# R5 配置治理结论

日期：`2026-04-28`

## 结论

R5 已建立 promoted registry：

```text
configs/promoted/README.md
configs/promoted/promoted_registry.json
```

当前没有任何 P1/R2/R3 研究候选进入生产：

| 候选 | 当前状态 | 是否 production eligible |
| --- | --- | --- |
| `P1_label_neighborhood_family` | rejected / frozen | 否 |
| `R2_legacy_dual_sleeve_family` | rejected | 否 |
| `R2B_v1_replacement_family` | rejected | 否 |
| `U3_A_real_industry_leadership__EDGE_GATED` | gray zone 已被 Day 8 weight audit 否定 | 否 |
| `U3_B_buyable_leadership_persistence__EDGE_GATED` | rejected | 否 |
| `U3_C_pairwise_residual_edge__EDGE_GATED` | rejected | 否 |

`promoted_registry.json` 中：

```text
promoted_configs = []
active_promoted_config_id = null
has_promoted_research_candidates = false
```

## 生产边界

日更推荐的生产配置只能来自 `configs/promoted/promoted_registry.json` 中的 `promoted_configs`。
`daily proxy` 不是 promotion 终点；`gray zone` 不是 production candidate。

`config.yaml.backtest` 与 `configs/backtests/` 继续作为研究回测入口和历史配置快照；未 promotion 的研究配置不得写入 `config.yaml.example` 或本地生产 `config.yaml` 的默认主线。

## 验收

1. promoted registry 骨架已新增。
2. registry 显式记录当前无 active promoted config。
3. registry 显式记录 R2B/R3 replacement 主线暂停，`U3_A` 不能进入生产。
4. 配置目录说明已更新，明确 `configs/promoted/` 与 `configs/backtests/` 的边界。
5. 新增测试确认 registry 为空、字段齐全，并检查生产模板没有写入 R2B/R3 候选 ID。
