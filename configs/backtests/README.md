# Backtest 配置快照

这里保存历史研究配置快照和场景变体。根目录只保留当前 canonical 研究入口 `config.yaml.backtest`。

命名约定：

- `r*`：prefilter / universe / turnover 等基线复核场景。
- `s*`：score 消融场景。
- `f1_*`：基本面单因子准入场景。
- `b*`、`vb*`：覆盖宽度、缓冲带和分层等权场景。
- `p1*`：P1 树模型或早期 P1/V3 研究快照。

这些文件是研究证据链的一部分，原则上不要就地覆写。新实验若需要临时动态快照，优先写入本目录并在产物里记录 `config_source`。
