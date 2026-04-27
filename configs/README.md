# 配置目录

本目录收纳非默认入口的配置快照，避免把研究变体平铺在项目根目录。

- `../config.yaml.example`：生产和日更链路的可运行模板。
- `../config.yaml`：本机私有运行配置，通常不入库。
- `../config.yaml.backtest`：当前 canonical 研究回测快照。
- `backtests/`：历史 backtest 场景、消融和验证快照。

配置加载兼容旧命令：传入 `config.yaml.backtest.r7_s2_prefilter_off_universe_on` 这类旧快照名时，加载器会先查根目录，再自动查 `configs/backtests/`。
