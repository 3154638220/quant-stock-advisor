# M2.5-C 诊断口径一致性检查

日期：2026-04-20

目的：对齐 `factor_ic_f1_pipeline.json`、`config.yaml.backtest`、`universe_m24_topk_pipeline.json` 与 `scripts/run_backtest_eval.py`，确认“诊断结论”和“实际回测口径”是否一致。

## 1. 结论摘要

本次检查确认，当前研发链路中至少存在 4 个会影响判断的口径差异：

1. F1 / IC 诊断已经表明 `lower_shadow_ratio` 不应继续作为核心因子，但多个回测配置仍给它较高权重。
2. `config.yaml.backtest` 的默认研究口径仍停留在 `top_k=10 + 3M + 三因子`，与 04-20 计划要求的 `top_k=20 + M + 双因子` 不一致。
3. 回测的“权重方法”只影响 Top-K 内部持仓分配，不影响 Top-K 的选股结果；因此它无法纠正前端打分阶段被弱因子污染的问题。
4. 配置中声明了行业约束，但运行时因 `data/cache/industry_map.csv` 缺失而静默降级，最终结果并未真正应用行业约束。

综合来看，“IC 没有转成收益”不只是持仓优化问题，也包含了明显的“诊断口径与回测输入未同步”问题。

## 2. 差异清单

| 编号 | 项目 | 诊断口径 | 回测口径 | 影响 |
| --- | --- | --- | --- | --- |
| C1 | 核心因子组成 | `ocf_to_net_profit`、`vol_to_turnover` 最可靠；`lower_shadow_ratio` 应降级 | 仍在多个主回测中保留 `lower_shadow_ratio: 0.3` | 弱因子仍在前端打分阶段稀释主信号 |
| C2 | 默认研究参数 | 04-20 计划要求 `top_k=20`、`rebalance_rule=M`、双因子 | 默认配置仍是 `top_k=10`、`rebalance_rule=3M`、三因子 | 后续实验容易基线漂移 |
| C3 | 权重方法实验含义 | 计划希望验证组合构建是否浪费信号 | 实现上只在 Top-K 选完后分配仓位 | “权重法无效”不能直接推导为“组合构建无关” |
| C4 | 行业约束 | 文档与配置中默认保留 | 实际运行已降级为关闭 | 回测结果与计划中的约束假设不一致 |

## 3. 证据与解释

### C1. `lower_shadow_ratio` 已不适合作为核心因子，但回测仍给高权重

`factor_ic_f1_pipeline.json` 的 `close_21d` 结果显示：

| 因子 | IC | t 值 |
| --- | ---: | ---: |
| `vol_to_turnover` | 0.030534 | 5.422 |
| `ocf_to_net_profit` | 0.018419 | 11.459 |
| `gross_margin_delta` | 0.007959 | 7.639 |
| `net_margin_stability` | 0.005386 | 3.706 |
| `ocf_to_asset` | 0.005177 | 5.011 |
| `lower_shadow_ratio` | 0.004018 | 1.726 |

同时 `factor_decisions` 中 `lower_shadow_ratio` 的 `p1_action` 已是 `zero`，说明按诊断规则它不应继续作为主线因子。

但 [config.yaml.backtest](/mnt/ssd/lh/config.yaml.backtest#L19) 里默认仍配置：

```yaml
signals:
  top_k: 10
  composite_extended:
    ocf_to_net_profit: 0.50
    lower_shadow_ratio: 0.30
    vol_to_turnover: 0.20
```

Universe 回测快照 [universe_m24_topk_pipeline.json](/mnt/ssd/lh/data/results/universe_m24_topk_pipeline.json) 也记录了相同的三因子权重：

```json
"composite_extended_weights": {
  "ocf_to_net_profit": 0.5,
  "lower_shadow_ratio": 0.3,
  "vol_to_turnover": 0.2
}
```

这意味着即便后端持仓做了等权、风险平价或最小方差，前端选股时仍然有 30% 的分数来自一个诊断上已经应降级的弱因子。

### C2. 默认研究配置仍停留在旧口径

04-20 计划要求先建立受控基线：

- `top_k=20`
- `rebalance_rule=M`
- `max_turnover=0.3`
- 双因子：`ocf_to_net_profit: 0.7`、`vol_to_turnover: 0.3`

但 [config.yaml.backtest](/mnt/ssd/lh/config.yaml.backtest#L19) 的默认研究口径仍是：

- `top_k: 10`
- `rebalance_rule: 3M`
- 三因子组合

这会带来两个问题：

1. 后续如果直接复用默认配置，实验会不知不觉回到 `M2.2` 时代的口径。
2. 文档里讨论的是 04-20 的新问题，但默认代码入口仍指向旧研究结论。

### C3. 回测“权重方法”只影响持仓分配，不影响选股源头

[scripts/run_backtest_eval.py](/mnt/ssd/lh/scripts/run_backtest_eval.py#L641) 中，`build_score()` 会先按 `composite_extended` 权重把各因子做截面 `z-score` 后线性合成为单一 `score`。随后 [scripts/run_backtest_eval.py](/mnt/ssd/lh/scripts/run_backtest_eval.py#L884) 才基于这个 `score` 选出 Top-K。

直到 [scripts/run_backtest_eval.py](/mnt/ssd/lh/scripts/run_backtest_eval.py#L915)，`portfolio_method` 才开始介入，它只会对已经选中的 Top-K 做仓位分配：

- `equal_weight`
- `risk_parity`
- `min_variance`
- `mean_variance`

因此：

1. 如果前端 `score` 被弱因子稀释，优化型权重方法并不能把“错选进来的股票”再剔除掉。
2. `M2.5-B` 里 `min_variance` 优于等权，只能说明持仓分配有帮助；不能说明“选股输入已经合理”。

### C4. 配置写了行业约束，但实际运行没有启用

[config.yaml.backtest](/mnt/ssd/lh/config.yaml.backtest#L45) 里声明：

```yaml
portfolio:
  industry_cap_count: 5
```

但 `data/cache/industry_map.csv` 当前不存在，`load_industry_map()` / `resolve_industry_cap_and_map()` 会在映射缺失时直接返回 `0`，即静默关闭行业约束，逻辑见：

- [scripts/run_backtest_eval.py](/mnt/ssd/lh/scripts/run_backtest_eval.py#L684)
- [scripts/run_backtest_eval.py](/mnt/ssd/lh/scripts/run_backtest_eval.py#L712)

这也和结果快照一致：`universe_m24_topk_pipeline.json` 的 `parameters.industry_cap_count` 为 `0`。

这类静默降级虽然是工程上可接受的 fallback，但会让“配置口径”和“结果口径”不一致，容易误判约束是否真的产生了作用。

## 4. 需要同步修正的地方

建议最少同步以下 4 项：

1. 将默认研究配置从三因子旧口径切换为 04-20 双因子研究配置，或明确拆分出新的默认研究配置文件。
2. 在默认回测配置中移除 `lower_shadow_ratio` 的核心地位，避免它继续参与主分数合成。
3. 在权重方法对照结论里明确注明：当前 `portfolio_method` 不改变 Top-K 选股集合，只改变 Top-K 内部仓位。
4. 对行业约束增加显式提示或失败标记，避免配置写着启用、结果实际关闭却不易察觉。

## 5. 当前判断

M2.5-C 的一致性检查结论是：

- “IC 没有变成收益”并非单一的持仓优化问题。
- 更直接的问题，是诊断已经建议降权/移除的弱因子，仍然在多个核心回测中参与前端打分。
- 在继续讨论新因子扩展前，应该先统一默认研究口径，否则 04-21 的实验仍可能在旧口径上打转。
