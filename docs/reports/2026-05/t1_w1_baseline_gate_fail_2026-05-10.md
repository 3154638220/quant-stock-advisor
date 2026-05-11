# T1 W1 因子 M8 Baseline Gate 失败报告

**日期**：2026-05-10  
**来源计划**：`docs/plan-05-09.md` Section 2  
**结论**：T1 不通过，W1 因子不入生产模型

---

## 1. 实验设计

4 组 M8 浓度管线运行 @16bps，Top20，U1 pool：

| 组 | --families | output-prefix |
|----|-----------|---------------|
| A (baseline) | industry_breadth,fund_flow,fundamental | monthly_selection_m8_baseline_rerun_2026_05_09 |
| B (+quality) | ...fundamental,quality | monthly_selection_m8_plus_quality_2026_05_09 |
| C (+quality+rv) | ...fundamental,quality,reversal_volume | monthly_selection_m8_plus_quality_rv_2026_05_09 |
| D (+all W1) | ...fundamental,quality,reversal_volume,liquidity_position | monthly_selection_m8_plus_w1_all_2026_05_09 |

所有结果文件位于 `data/results/`。

---

## 2. 结果对比

| 配置 | after-cost/月 | Rank IC | 模型 | vs M8 baseline |
|------|--------------|---------|------|----------------|
| M8 baseline (regime-aware indcap3) | **1.93%** | 0.0973 | M8 regime-aware | — |
| +quality M5 ET indcap3 | 1.88% | 0.1066 | M5 ExtraTrees | −0.05% |
| +quality+rv M5 EN indcap3 | **1.95%** | 0.1009 | M5 ElasticNet | +0.02% |
| +all W1 M5 ET indcap3 | 1.82% | 0.1091 | M5 ExtraTrees | −0.11% |
| +all W1 M6 XGB indcap4 | 1.99% | 0.0164 | M6 XGBoost | +0.06% ⚠️ |

---

## 3. 三大失败原因

### 3.1 M8 Regime-Aware Policy 对新因子不可用（架构限制）

M8 的核心优势是 `lagged_regime_fixed_score_blend`（regime-aware 选股策略 + indcap3 行业分散）。但添加 `--families` 后，管线仅输出 M5（ExtraTrees/ElasticNet）和 M6（XGBoost/calibrated）模型，不输出 M8 regime-aware 模型。

**含义**：T1 实际对比的是 M5/M6+indcap3 vs M8 regime-aware baseline，而非计划设计的 M8+W1 vs M8 baseline。

### 3.2 最佳 M5 组合增量微弱

+quality+rv M5 EN indcap3 达到 1.95%/月，仅比 M8 baseline 1.93% 高 0.02%/月。无实践意义。

### 3.3 全量叠加有损

+all W1 M5 ET indcap3 = 1.82%，低于 M8 baseline 0.11%/月。Rank IC 虽从 0.097 提升至 0.109，但 after-cost 下降——新因子引入的噪声/冗余超过信号增益。

---

## 4. W1 因子个体质量（正面信号）

| 因子族 | 对 M5 的提升 | 说明 |
|--------|-------------|------|
| quality | M5 ET 1.02%→1.88%（+0.86%） | 强信号，最大的单族贡献 |
| reversal_volume | M5 EN 1.56%→1.95%（+0.39%） | 在 ElasticNet 中有增量 |
| liquidity_position | 全量叠加 M5 ET 下降至 1.82% | 边际负贡献 |

---

## 5. 下一步

- **不入模**：不更新 `promoted_registry.json`
- **I9**：修复 M8 regime-aware policy 因子族扩展问题后重跑 T1
- **T4**：quality + reversal_volume 子集始终优于全量，可考虑作为 M5 候选
- **OOS**：继续积累（OOS#2 待 2026-05-29）

---

## 6. 证据文件索引

- `data/results/monthly_selection_m8_baseline_rerun_2026_05_09_2026_05_09_*`
- `data/results/monthly_selection_m8_plus_quality_2026_05_09_2026_05_09_*`
- `data/results/monthly_selection_m8_plus_quality_rv_2026_05_09_2026_05_09_*`
- `data/results/monthly_selection_m8_plus_w1_all_2026_05_09_2026_05_09_*`
