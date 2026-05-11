# 实验配置

这里收纳临时或探索性的配置快照，避免把试验文件放在项目根目录或 `tmp/`。

- `weekly_kdj/`：周线 KDJ 开关对照配置。

月度选股当前 promoted 主线仍是 `Top20`。如果只是想先验证是否值得缩到 `Top10` 或 `Top5`，优先走研究脚本而不是改生产配置，例如：

```bash
python scripts/run_monthly_selection_concentration_regime.py --topk-preset narrow
```

这会按 `Top5,Top10,Top20` 自动生成对应的行业 cap 网格，先产出研究证据，再决定是否 promotion。

约定：实验配置可以被报告引用，但不代表生产候选；进入主线前需要迁移到 `configs/backtests/` 或 `configs/promoted/` 并补齐证据链。

---

## 因子新增两阶段 Gate 规程

新因子进入生产主线需通过两个独立阶段。**Stage A 通过≠Stage B 通过**，两者验证不同层次的问题。

### Stage A：IC Gate（单因子信号质量）

验证新因子在截面上的独立预测能力。

| 维度 | 说明 |
|------|------|
| **基准** | `price_volume_only`（8 个价量特征 + ExtraTrees） |
| **模型** | ExtraTrees / ElasticNet 原始输出 |
| **门槛** | delta IC IR ≥ 0.3 且 coverage ≥ 60% |
| **通过后状态** | 「验证候选」— 因子数据链路可用，信号方向明确 |

执行脚本：

```bash
# 个体因子 IC 验收
python scripts/run_oracle_ic_gate.py \
  --baseline data/results/monthly_selection_m5_pv_only_*.json \
  --candidate data/results/monthly_selection_m5_plus_<family>_*.json \
  --family <family_name> \
  --output-dir data/results/oracle_ic_gate/
```

### Stage B：M8 Baseline Gate（生产模型增量收益）

验证因子加入 M8 生产模型后，整体 after-cost 超额是否提升。

| 维度 | 说明 |
|------|------|
| **基准** | 当前 `promoted_registry.json` 中的生产 config（当前为 `M8_regime_aware_fixed_policy__indcap3`） |
| **模型** | M8 regime-aware policy + indcap3 行业分散 |
| **门槛** | after-cost 月超额 delta vs production baseline ≥ 0，NW-t ≥ 2.5，Bootstrap CI 下界 > 0 |
| **通过后状态** | 「promotion 候选」— 可创建研究配置，等待 OOS 积累后正式 promote |

执行脚本：

```bash
# baseline 重跑
python scripts/run_monthly_selection_concentration_regime.py \
  --families industry_breadth,fund_flow,fundamental \
  --output-prefix monthly_selection_m8_baseline_rerun

# 实验组：加入新因子族
python scripts/run_monthly_selection_concentration_regime.py \
  --families industry_breadth,fund_flow,fundamental,<new_family> \
  --output-prefix monthly_selection_m8_plus_<new_family>
```

对比 `topk_excess_after_cost_mean` 与 baseline 的差值。

### 状态流转

```
新因子代码交付 → Stage A (IC Gate) → 验证候选 → Stage B (M8 Baseline Gate) → promotion 候选 → OOS 积累 → Promoted
                       ↓ 不通过                    ↓ 不通过
                    停止/修复因子              观察/子集精选(T4)
```

### 记录要求

- Stage A 结果写入 `docs/reports/YYYY-MM/ic_gate_<family>_YYYY-MM-DD.md`
- Stage B 结果写入 `docs/reports/YYYY-MM/baseline_gate_<family>_YYYY-MM-DD.md`
- 通过 Stage B 后在 `configs/experiments/` 创建研究候选 JSON
- T1 结论（通过/不通过 + 原因 + 证据路径）更新到当前主计划

---

## 当前研究候选

### monthly_selection_m8_indcap3_plus_quality

- **Config**: [`configs/experiments/monthly_selection_m8_indcap3_plus_quality.json`](monthly_selection_m8_indcap3_plus_quality.json)
- **状态**: `research_candidate` (2026-05-11 创建)
- **Stage A (IC Gate)**: ✅ 通过 — quality 族 13 个因子 IC delta +0.0027 vs PV-only, coverage ≥ 60%
- **Stage B (M8 Baseline Gate)**: ✅ 通过 (2026-05-10 重跑) — M8+quality indcap3 = 2.15%/月, delta +0.16% vs baseline 1.99%, 全 7 gate 通过
- **证据文件**:
  - Stage B leaderboard: `docs/reports/2026-05/monthly_selection_m8_plus_quality_2026_05_10_2026_05_10.md`
  - Baseline rerun: `docs/reports/2026-05/monthly_selection_m8_baseline_rerun_2026_05_10_2026_05_10.md`
  - Leaderboard CSV: `data/results/monthly_selection_m8_plus_quality_2026_05_10_leaderboard.csv`
  - Gate CSV: `data/results/monthly_selection_m8_plus_quality_2026_05_10_gate.csv`
- **Promotion 阻塞**: OOS gate — OOS #2 待 2026-05-29 录入
- **排除的因子族**:
  - `reversal_volume`: 毒性, 在 quality 基础上 −0.81%/月
  - `liquidity_position`: 中性偏负, 部分修复 rv 损害但仍低于 baseline

---

## 弱因子治理日志

### 2026-05-11: W5 审计结果应用

- **脚本**: `scripts/apply_factor_audit_results.py`
- **审计源**: `data/results/factor_audit_2026_05_09_full/weak_factors.csv`
- **动作**: 15 个唯一因子 `active` → `False`（IC IR < 0.2）
- **跳过的 `is_missing_*` flags**: 21 个（已在管线层面通过 `--exclude-missing-flags` 排除）
- **治理日志**: `data/results/factor_audit_2026_05_09_full/factor_governance_log.jsonl`
- **摘要**: `data/results/factor_audit_2026_05_09_full/governance_summary.md`

各家族降级数量：

| Family | Demoted |
|--------|---------|
| concept | 1 |
| fundamental | 5 |
| lhb | 1 |
| liquidity_position | 2 |
| margin_trading | 1 |
| northbound | 2 |
| quality | 1 |
| shareholder | 2 |

> 降级因子在下次 `get_factor_cols(only_active=True)` 调用时自动排除。
