# 配置目录

本目录收纳非默认入口的配置快照，避免把研究变体平铺在项目根目录。

- `../config.yaml.example`：月度选股生产模板。
- `../config.yaml`：本机私有运行配置，通常不入库。
- `../config.yaml.backtest`：当前 canonical 研究回测快照。
- `backtests/`：历史 backtest 场景、消融和验证快照。
- `experiments/`：临时或探索性配置，按主题归档；用于保留实验上下文，不作为 canonical 入口。
- `promoted/`：生产 promotion registry；月度选股生产候选只能来自该 registry。当前 active promoted 方法是 `monthly_selection_u1_top20_indcap3_hardcap_baseline`。

配置加载兼容旧命令：传入 `config.yaml.backtest.r7_s2_prefilter_off_universe_on` 这类旧快照名时，加载器会先查根目录，再自动查 `configs/backtests/`。

R5 配置治理约束：`gray zone` 不是 production candidate；未写入 `configs/promoted/promoted_registry.json` 的研究配置不得进入生产默认配置。
