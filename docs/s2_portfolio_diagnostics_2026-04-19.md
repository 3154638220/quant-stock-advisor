# S2 配权可观测性诊断

- 生成日期：`2026-04-19`
- 主线来源：`docs/plan.md` 第 `2.3` 节
- 固定口径：`S2 = vol_to_turnover` 单因子 / `tplus1_open` / `M` / `top_k=20` / `max_turnover=0.3` / `universe_filter=false`
- 对照方法：`equal_weight`、`risk_parity`、`min_variance`

## 1. 结果汇总

| 方法 | 含成本 CAGR | 含成本夏普 | MaxDD | 半 L1 换手 | rolling OOS 中位年化 | slice OOS 中位年化 | mean L1(diff vs equal) | equal-like ratio | 中位有效持仓数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `equal_weight` | `-8.95%` | `-0.380` | `48.68%` | `26.94%` | `-3.12%` | `-3.63%` | `0.000` | `100%` | `20.00` |
| `risk_parity` | `-8.95%` | `-0.380` | `48.68%` | `26.94%` | `-3.12%` | `-3.63%` | `0.000` | `100%` | `20.00` |
| `min_variance` | `-5.60%` | `-0.267` | `35.14%` | `51.41%` | `-12.38%` | `-8.31%` | `1.128` | `0%` | `6.09` |

对应结果文件：

- `data/results/s2_vol_equal_weight_diag_2026-04-19.json`
- `data/results/s2_vol_risk_parity_diag_2026-04-19.json`
- `data/results/s2_vol_min_variance_diag_2026-04-19.json`
- `data/results/s2_portfolio_diag_compare_2026-04-19.csv`

## 2. 已回答的问题

### 2.1 `risk_parity` 为什么和等权完全一致

结论：当前不是“优化器没有生效”，而是“优化器稳定求出等权型解”。

证据：

- `risk_parity` 全样本 `75` 次调仓全部 `solver_success=true`
- `fallback_counts = {"equal_like_solution": 75}`
- `mean_l1_diff_vs_equal = 0.0`
- `equal_like_ratio = 1.0`
- `median_effective_n = 20.0`

解释：

- 当前 `S2` 候选集合下，`risk_parity` 使用的协方差矩阵没有退化到不可求解；中位条件数约 `50.30`
- 但在该协方差结构下，等风险贡献解与等权几乎重合，因此回测结果与等权完全一致
- 所以主计划里的问题已经可以收敛为：`risk_parity` 目前更像“解释协方差几何结构的诊断工具”，而不是能稳定带来额外收益的默认配权方法

### 2.2 `min_variance` 有没有真正改变持仓

结论：有，而且改动幅度很大。

证据：

- `mean_l1_diff_vs_equal = 1.128`
- `max_l1_diff_vs_equal = 1.440`
- `equal_like_ratio = 0`
- `mean_weight_std = 0.0787`
- `median_effective_n = 6.09`
- `fallback_counts = {"_none": 75}`

解释：

- `min_variance` 并没有退化成等权或 fallback
- 它把原本 `20` 只近似等权的组合压缩成“有效持仓数约 6 只”的集中组合
- 这种集中化降低了全样本回撤，并改善了全样本夏普
- 但代价是换手显著升高，且 rolling / slice OOS 的中位年化明显弱于等权

## 3. 主线含义

可以更新的判断：

1. `risk_parity == equal_weight` 的原因已经解释清楚：不是链路坏了，而是当前输入下的最优解本身接近等权。
2. `min_variance` 是“真实改变持仓”的方法，不应与 `risk_parity` 一起归类为“没有起作用”。
3. 但 `min_variance` 暂时还不能直接升级为主线默认配权，因为它只改善了全样本，却没有改善 OOS 中位数，而且显著放大了换手。

因此，本轮结论与 `docs/plan.md` 保持一致：

- `risk_parity` 降级为诊断工具，而非默认候选方案
- `min_variance` 保留为可选研究分支，但仍未达到“全样本与 OOS 同时稳定优于等权”的放行标准
- `S2 + equal_weight` 仍是当前最稳的研究基线

## 4. 下一步建议

按主计划优先级，下一步应转向：

1. 解释为什么 `P1 静态` 仍优于当前 `S2` 基线
2. 在不改变 `S2` 候选集合的前提下，判断 `prefilter` 是否误伤有效候选
3. 将 `Universe` 与 `prefilter` 影响从 score / 配权问题里继续拆开
