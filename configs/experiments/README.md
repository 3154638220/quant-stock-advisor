# 实验配置

这里收纳临时或探索性的配置快照，避免把试验文件放在项目根目录或 `tmp/`。

- `weekly_kdj/`：周线 KDJ 开关对照配置。

月度选股当前 promoted 主线仍是 `Top20`。如果只是想先验证是否值得缩到 `Top10` 或 `Top5`，优先走研究脚本而不是改生产配置，例如：

```bash
python scripts/run_monthly_selection_concentration_regime.py --topk-preset narrow
```

这会按 `Top5,Top10,Top20` 自动生成对应的行业 cap 网格，先产出研究证据，再决定是否 promotion。

约定：实验配置可以被报告引用，但不代表生产候选；进入主线前需要迁移到 `configs/backtests/` 或 `configs/promoted/` 并补齐证据链。
