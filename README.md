# GRL

本仓库用于研究影响最大化（Influence Maximization）中的 GRL 方法，并逐步从历史实验脚本整理为一个可审计、可复现、可扩展的实验代码库。

## 当前整理状态

本次整理完成了第一版结构化改造：

- 将历史实现迁移到 `legacy/`
- 新增 `configs/`、`docs/`、`scripts/`、`src/grl/`、`tests/`、`outputs/`、`checkpoints/`
- 建立论文—代码映射与实验协议文档
- 提供配置驱动的最小实验入口 `scripts/run_experiment.py`

## 目录结构

```text
GRL/
├── baselines.py
├── gnn.py
├── gnn_celf.py
├── configs/
├── docs/
├── scripts/
│   └── debug/
├── src/
│   └── grl/
├── tests/
├── outputs/
├── checkpoints/
└── legacy/
```

## 历史实现说明

以下文件已移入 `legacy/`，作为论文审计与对照实验的历史版本：

- `legacy/grl.py`
- `legacy/grl-v2.py`
- `legacy/grl-v3.py`
- `legacy/previous_code/`

调试脚本已移入：

- `scripts/debug/check_norm.py`
- `scripts/debug/debug_gnn.py`

## 最小实验入口

可以通过配置文件运行第一版 baseline 实验：

```bash
python scripts/run_experiment.py --config configs/jazz.yaml
```

输出将保存在：

```text
outputs/<dataset>/<timestamp>/
```

其中包含：

- `config.yaml`
- `metrics.json`
- `selected_seeds.json`
- `run.log`

## 文档

- `docs/DEVELOPMENT_PLAN.md`
- `docs/PAPER_CODE_MAPPING.md`
- `docs/EXPERIMENT_PROTOCOL.md`

## 下一步

后续将继续推进：

1. 固定论文实验配置；
2. 建立 GNN 单独评估流程；
3. 补充 Oracle 诊断实验；
4. 重构为直接预测边际增益的集合条件化模型。
