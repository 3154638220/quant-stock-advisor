# 文档目录

长期维护的说明文档保留在 `docs/` 根目录；按日期生成的研究报告归档到 `docs/reports/<YYYY-MM>/`；历史计划文档归档到 `docs/archive/plans/`。

## 活跃文档

- `plan.md` — 当前唯一主计划（canonical），所有研究决策的权威来源。
- `plan-05-05` — 2026-05-05 改进计划（Phase A-C 行动路线）。
- `项目算法建模详解.md` — 算法、统计口径与模型说明。
- `月度选股流程说明.md` — 月频选股主线、候选池、脚本/模块边界、产物与 promotion 纪律。
- `研究脚本契约化与配置治理.md` — 研究 manifest、配置治理和脚本模块化治理记录。
- `config_reference.md` — 配置文件参考。

## 归档

- `archive/plans/` — 历史计划文档（plan-05-03, plan-05-04）。
- `reports/<YYYY-MM>/` — 月度研究报告、诊断报告、证据链。

## 规则

- 新报告若是一次性实验产物，放入 `docs/reports/<YYYY-MM>/`。
- 历史计划文档移入 `docs/archive/plans/`，只保留当前活跃版本在根目录。
- 只有会被 README 或月度主线持续引用的文档才留在根目录。
