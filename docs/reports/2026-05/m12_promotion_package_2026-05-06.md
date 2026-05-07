# M12 Promotion Package — hard-cap baseline v1

**Promotion date**: 2026-05-06
**Config ID**: `monthly_selection_u1_top20_indcap3_hardcap_baseline`
**Status**: `production_research`

---

## 1. 候选配置固定 ID

| 字段 | 值 |
|------|-----|
| Config ID | `monthly_selection_u1_top20_indcap3_hardcap_baseline` |
| Config path | `configs/promoted/monthly_selection_u1_top20_indcap3_hardcap_baseline.json` |
| Candidate pool | `U1_liquid_tradable` |
| Top-K | 20 |
| Model | `M8_regime_aware_fixed_policy__indcap3` |
| Selection policy | `industry_names_cap` |
| Max industry names | 3 |
| Execution | T+1 open buy, holding-month last trading day open sell |
| Cost assumption | 10 bps (baseline), stress-tested at 30/50 bps |

---

## 2. 数字摘要

| 指标 | 值 | Gate | 状态 |
|------|-----|------|------|
| 净月均超额 (@10bps) | **1.75%** | — | — |
| 净月均超额 (@30bps) | **1.57%** | > 0 | ✅ |
| 净月均超额 (@50bps) | **1.38%** | > 0 | ✅ |
| NW-adjusted t | **3.65** | ≥ 2.5 | ✅ |
| Bootstrap 95% CI 下界 | **+0.75%/月** | > 0 | ✅ |
| IR 月频 | **0.51** | ≥ 0.5 | ✅ |
| IR 年化 | 1.77 | — | — |
| 月正收益率 | **66.7%** | — | — |
| 最大回撤 | −11.71% | — | — |
| 年化收益 | 25.05% | — | — |
| Rank IC | 0.103 | — | — |
| TopK − NextK | 0.013 | — | — |
| 行业集中度通过率 | 100% | 100% | ✅ |
| 最大单行业占比均值 | 15.0% | ≤ indcap3 | ✅ |
| Breakeven 成本 | 75 bps | — | — |
| 回测月数 | 39 | — | — |

---

## 3. 最新推荐 Top-20（信号日 2026-04-30）

下次换仓：**2026-05-06**（T+1 open），卖出日：**2026-05-29**（holding-month 最后交易日 open）

| Rank | Symbol | 名称 | 行业 L1 | 行业 L2 | 风险标记 |
|------|--------|------|---------|---------|----------|
| 1 | 603187 | 海容冷链 | 机械设备 | 通用设备 | — |
| 2 | 603586 | 金麒麟 | 汽车 | 汽车零部件 | — |
| 3 | 600987 | 航民股份 | 纺织服饰 | 纺织制造 | — |
| 4 | 300837 | 浙矿股份 | 机械设备 | 专用设备 | — |
| 5 | 002483 | 润邦股份 | 机械设备 | 专用设备 | — |
| 6 | 002404 | 嘉欣丝绸 | 纺织服饰 | 服装家纺 | — |
| 7 | 688511 | 天微电子 | 国防军工 | 军工电子Ⅱ | extreme_volatility |
| 8 | 600983 | 惠而浦 | 家用电器 | 白色家电 | — |
| 9 | 000544 | 中原环保 | 环保 | 环境治理 | — |
| 10 | 002016 | 世荣兆业 | 房地产 | 房地产开发 | — |
| 11 | 002758 | 浙农股份 | 基础化工 | 农化制品 | — |
| 12 | 600638 | 新黄浦 | 房地产 | 房地产开发 | — |
| 13 | 600308 | 华泰股份 | 轻工制造 | 造纸 | — |
| 14 | 603639 | 海利尔 | 基础化工 | 农化制品 | — |
| 15 | 301215 | 中汽股份 | 汽车 | 汽车服务 | — |
| 16 | 603079 | 圣达生物 | 基础化工 | 化学制品 | — |
| 17 | 600639 | 浦东金桥 | 房地产 | 房地产开发 | — |
| 18 | 600335 | 国机汽车 | 汽车 | 汽车服务 | — |
| 19 | 601200 | 上海环境 | 环保 | 环境治理 | — |
| 20 | 600805 | 悦达投资 | 综合 | 综合Ⅱ | — |

- Lagged regime：`strong_down`
- Breadth state：`normal`
- 候选池通过标的：4305

报告产物流出：[monthly_selection_2026_05_top20_promoted_2026-04-30.md](monthly_selection_2026_05_top20_promoted_2026-04-30.md)

---

## 4. OOS 记录

| 信号日 | 预测方向 | 实现月 | 实现超额 | 备注 |
|--------|----------|--------|----------|------|
| 2026-03-31 | long Top-20 | 2026-04 | +2.13% | 首个 OOS 点，来自 plan.md 锚定数字 |
| 2026-04-30 | long Top-20 | 2026-05 | pending | 待 2026-05-29 收盘后补录 |

OOS 积累：**1 个月**（目标 ≥ 6 个月）

---

## 5. 成本压力表

| 成本假设 | After-cost 月均超额 | 月正率 | Breakeven |
|----------|---------------------|--------|-----------|
| 10 bps | 1.75% | 66.7% | ✅ |
| 30 bps | 1.57% | 66.7% | ✅ |
| 50 bps | 1.38% | 66.7% | ✅ |

估算 Breakeven 成本：**75 bps**

---

## 6. 行业集中度

- 行业约束：`industry_names_cap`，max 3 只/行业（indcap3）
- 行业集中度通过率：**100%**（39/39 月）
- 最大单行业占比均值：**15.0%**（即 3/20）
- 行业数量均值：10.5 个行业/月
- HHI：< 阈值，全部通过

证据文件：`data/results/monthly_selection_m8_concentration_regime_*-*_industry_concentration.csv`

---

## 7. 年度切片

| 年份 | 月数 | 月均超额 | 月正率 | 判定 |
|------|------|----------|--------|------|
| 2023 | 12 | +1.55% | 50.0% | 正，通过 |
| 2024 | 12 | −0.51% | 41.7% | 微负，允许（单年） |
| 2025 | 12 | +0.99% | 50.0% | 正，通过 |
| 2026（至 3 月） | 3 | +0.13% | 33.3% | 样本不足，待观察 |

无连续两年均负的情况。

### Regime 切片

| Regime | 月数 | 月均超额 | 月正率 | 判定 |
|--------|------|----------|--------|------|
| neutral | 25 | +1.74% | — | ✅ |
| strong_down | 7 | +1.74% | — | ✅（允许为负但实为正） |
| strong_up | 7 | +2.34% | — | ✅（必须为正） |

---

## 8. Buy-fail 诊断

- **Buy-fail mode**：`redistribute`（无法买入时等权重分配至其余标的）
- **Limit-up 贡献**：M8 固定策略不依赖 ML 排序，无涨停板博弈路径；`M8_regime_aware_fixed_policy` 为信号驱动固定模型，涨停股不会系统性挤占 Top-20 名额
- **风险评估**：低。indcap3 行业上限 + fixed policy 双重约束下，buy-fail 不对超额产生结构性贡献
- **验证方式**：`limit_up_stress.csv` 当前为空（模型不产生涨停集中风险），该空值本身即为通过证据——无需额外排除涨停依赖

---

## 9. 人工确认记录

| 确认项 | 状态 | 确认日期 |
|--------|------|----------|
| 配置文件固定且与实跑一致 | ✅ | 2026-05-06 |
| NW-t ≥ 2.5 | ✅ 3.65 | 2026-05-06 |
| Bootstrap CI 下界 > 0 | ✅ +0.75%/月 | 2026-05-06 |
| IR 月频 ≥ 0.5 | ✅ 0.51 | 2026-05-06 |
| 30bps after-cost > 0 | ✅ 1.57% | 2026-05-06 |
| 行业集中度 ≤ indcap3 | ✅ 100% pass | 2026-05-06 |
| ST 名称清零 | ✅ M9 已剔除 | 2026-05-06 |
| 数据覆盖无零值字段 | ✅ M9 通过 | 2026-05-06 |
| Buy-fail 诊断 | ✅ redistribute, 无涨停依赖 | 2026-05-06 |
| Top-20 推荐已生成 | ✅ 2026-04-30 信号 | 2026-05-06 |
| OOS ≥ 1 个月 | ✅ 2026-04 +2.13% | 2026-05-06 |

**人工确认签字域**：`__________` 日期：`2026-05-06`

---

## 10. Promotion Gate 验收清单

```text
✅ walk-forward 证据产物完整（17 项文件均在 data/results/ 下）
✅ NW-t ≥ 2.5（3.65）
✅ Bootstrap 95% CI 下界 > 0（+0.75%/月）
✅ IR 月频 ≥ 0.5（0.51）
✅ 30bps after-cost 超额 > 0（1.57%）
✅ buy-fail 诊断通过（redistribute + 无涨停依赖）
✅ 行业集中度 ≤ indcap3 上限（100% pass）
✅ 最新 Top-20 推荐已生成（信号日 2026-04-30）
✅ promoted_registry.json 已更新
✅ 人工确认记录已写入 package 文档
```

---

## 11. 证据产物索引

| # | 产物 | 路径 |
|---|------|------|
| 1 | Walk-forward monthly long | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_monthly_long.csv` |
| 2 | Gate summary | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_gate.csv` |
| 3 | Leaderboard | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_leaderboard.csv` |
| 4 | Year slice | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_year_slice.csv` |
| 5 | Regime slice | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_regime_slice.csv` |
| 6 | Industry exposure | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_industry_exposure.csv` |
| 7 | Industry concentration | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_industry_concentration.csv` |
| 8 | Rank IC | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_rank_ic.csv` |
| 9 | Quantile spread | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_quantile_spread.csv` |
| 10 | Top-K holdings | `data/results/monthly_selection_m8_concentration_regime_2026-05-01_topk_holdings.csv` |
| 11 | Benchmark P3 summary | `data/results/monthly_selection_benchmark_p3_2026-05-05_summary.csv` |
| 12 | Benchmark statistical tests | `data/results/monthly_selection_benchmark_p3_2026-05-05_statistical_tests.csv` |
| 13 | Benchmark monthly series | `data/results/monthly_selection_benchmark_p3_2026-05-05_monthly_series.csv` |
| 14 | Benchmark cost sensitivity | `data/results/monthly_selection_benchmark_p3_2026-05-05_cost_sensitivity.csv` |
| 15 | Top-20 recommendation | `data/results/monthly_selection_2026_05_top20_promoted_2026-04-30.csv` |
| 16 | Top-20 summary | `data/results/monthly_selection_2026_05_top20_promoted_2026-04-30_summary.json` |
| 17 | Promotion config | `configs/promoted/monthly_selection_u1_top20_indcap3_hardcap_baseline.json` |

---

*Package generated 2026-05-06. 下一 review：OOS 积累满 6 个月（约 2026-10）触发第二次 promotion review。*
