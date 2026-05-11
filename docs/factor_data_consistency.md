# 因子数据双轨一致性

## 背景

项目中存在两份因子数据来源：

| 数据源 | 用途 | 更新方式 |
|--------|------|---------|
| `data/cache/monthly_selection_features.parquet` | M5/M8 pipeline 训练（价量基础特征） | `run_monthly_selection_dataset.py` 按需重建 |
| `data/market.duckdb → prepared_factors` 表 | W5 因子审计、W6 Oracle gap 分析（全量因子） | `materialize_prepared_factors.py` 手动运行 |

M5/M8 管线通过 `--families` 参数在运行时从 DuckDB 动态加载因子族，不依赖 `prepared_factors`。但 W5/W6 分析工具直接从 `prepared_factors` 读取，如果新增因子族或 parquet 重建后未重新物化，分析结论会漏掉新因子。

## 何时需要手动重跑物化

以下任一情况发生后，**必须**重跑 `materialize_prepared_factors.py`：

1. **新增因子族**（如 W1 quality 族入库后）：新因子不会自动出现在 `prepared_factors` 中
2. **`monthly_selection_features.parquet` 重建**（如扩展历史区间或修正 PIT 逻辑）：parquet 列或行变化后物化表过期
3. **因子族表结构变更**（如 DuckDB 中某族新增/删除列）：物化表仍保留旧 schema

## 自动 freshness check

`materialize_prepared_factors.py` 内置了 staleness 检测：

- 每次物化后在 DuckDB 写入 `_materialization_meta` 表（记录时间戳、源文件、行列数）
- 下次运行时自动对比 parquet 文件 mtime，若 parquet 更新则打印 WARNING
- 可通过 `--skip-freshness-check` 跳过检测

## 月度 SOP 集成

`run_monthly_production.py` 支持 `--rematerialize-prepared-factors` 选项，在月度流程末尾自动重跑物化。建议在新月数据接入后使用：

```bash
python scripts/run_monthly_production.py --month 2026-06 --rematerialize-prepared-factors
```

## 验证一致性

运行以下命令确认物化表包含所有预期因子族：

```bash
python scripts/materialize_prepared_factors.py --families quality,reversal_volume  # 指定新增族
```

物化完成后检查输出中的 "Factor summary" 行，确认 raw/z-score/missing 列数符合预期。
