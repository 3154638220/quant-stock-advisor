# M6 LTR 失败归因分析 (2026-05-06)

生成时间：2026-05-06  
数据来源：M6 LTR indzneutral (2026-05-04) × M5 multisource pitfix (2026-05-03)

## 逐年 After-Cost Excess (XGBoost Rank NDCG)

| Year | Pool | Months | Mean Excess/Mo | Total Excess | Hit Rate | Mean Turnover |
|------|------|--------|---------------|-------------|----------|---------------|
| 2023 | U1 | 12 | -0.72% | -9.38% | 16.7% | 0.93 |
| 2023 | U2 | 12 | -0.05% | -2.67% | 33.3% | 1.00 |
| 2024 | U1 | 12 | -0.56% | -7.12% | 33.3% | 0.95 |
| 2024 | U2 | 12 | -1.20% | -14.34% | 33.3% | 0.96 |
| 2025 | U1 | 12 | -1.74% | -20.77% | 33.3% | 0.95 |
| 2025 | U2 | 12 | -1.37% | -16.07% | 41.7% | 0.93 |
| 2026 | U1 | 3 | -0.75% | -2.47% | 33.3% | 0.70 |
| 2026 | U2 | 3 | -1.22% | -3.61% | 0.0% | 0.85 |

**结论**：M6 LTR 在所有年份、两个候选池上均为负超额（或接近零）。月均换手率极高（0.85–1.00），turnover 接近 100%/月。

## 候选池结构分析

| Pool | Pass/Month | Total Symbols | Signal Months |
|------|-----------|---------------|---------------|
| U0 (all) | 4,762 | 5,197 | 64 |
| U1 (liquid) | 3,320 | 5,197 | 64 |
| U2 (risk_sane) | 2,976 | 5,197 | 64 |

U1/U2 标的层面 overlap = 100%（所有 symbol 在所有 pool 中出现，区别在于 pass 过滤条件）。

## 关键发现

### 1. 原假设不成立

原假设「U1 池噪声过多导致 LTR 失效」与证据矛盾：
- U2 池（经过 risk filtering）同样负超额
- U1/U2 的候选池重叠度为 100%，仅有 filtering pass 率的差异（U1: 3,320 vs U2: 2,976）
- 2023 年 U2 表现略好于 U1，但 2024–2026 年 U2 甚至更差

### 2. 真正的失败模式：换手与排序失效

- 月均 turnover 高达 0.85–1.00，说明 XGBoost NDCG 模型几乎每月都在完全更换持仓
- 高换手 + 成本拖累 = 净超额为负（即使 gross 信号有微弱正收益也被成本吞噬）
- Rank IC 约 0.007（from M6 contract test），接近零

### 3. 根本原因推断

| 假设 | 证据 | 可能性 |
|------|------|--------|
| NDCG lambda-rank 在极低 label 信噪比下退化 | Rank IC ~0.007 | 高 |
| XGBoost 超参不适合月度低频场景 | 高 turnover | 中 |
| Label 定义（future 1M return）在线性模型下有效但 LTR 需要更优定义 | ElasticNet 有效但 LTR 无效 | 中 |
| 特征非线性未被 XGBoost 有效捕获 | M6 特征重要性几乎均匀 | 低 |

## 下一步行动

1. **冻结 M6 LTR 作为主线候选**：当前证据不支持 promote
2. **测试 ElasticNet × ExtraTrees 对比**：确认是否是 XGBoost-LTR 的模型选择问题
3. **降低 M6 研究优先级**：将资源集中到 P4（融资融券）和 P5（主题/概念）
4. **若需重试 LTR**：先解决 label 设计（分段 label？未来 3M 收益？）和 turnover 约束

## 产物

- `data/results/m6_ltr_failure_attribution_20260506.md`
- `data/results/m6_contract_test_2026-05-05_*.csv` (已有)
