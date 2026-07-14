# GRL

本仓库用于研究影响最大化（Influence Maximization）中的 GRL 方法，并逐步从历史实验脚本整理为一个可审计、可复现、可扩展的实验代码库。

## 当前整理状态

本次整理完成了第一版结构化改造，并开始推进第一阶段“可复现传统基线”建设：

- 将历史实现迁移到 `legacy/`
- 建立并保留 `configs/`、`docs/`、`scripts/`、`src/grl/`、`tests/` 等主线目录
- 建立论文—代码映射与实验协议文档
- 提供配置驱动的最小实验入口 `scripts/run_experiment.py`
- 将图加载、IC 扩散与传统 baseline 迁入 `src/grl/`
- 增加数据集检查脚本与基础测试骨架

## 目录结构

```text
GRL/
├── configs/
├── data/
├── docs/
├── legacy/
├── param/
├── scripts/
│   └── debug/
├── src/
│   └── grl/
├── tests/
├── README.md
└── requirements.txt
```

## 历史实现说明

以下文件保留在 `legacy/`，作为论文审计与对照实验的历史版本：

- `legacy/grl.py`
- `legacy/grl-v2.py`
- `legacy/grl-v3.py`
- `legacy/previous_code/`

此前仓库根目录中存在的旧脚本 `baselines.py`、`gnn.py`、`gnn_celf.py` 已删除；其功能要么已经迁移至 `src/grl/` 模块化实现，要么仅保留在历史版本与文档记录中。

调试脚本已移入：

- `scripts/debug/check_norm.py`
- `scripts/debug/debug_gnn.py`

## 安装

```bash
pip install -r requirements.txt
```

## 最小实验入口

可以通过配置文件运行第一版 baseline 实验：

```bash
python scripts/run_experiment.py --config configs/nethept.yaml
```

运行结果默认会输出到：

```text
outputs/<dataset>/<timestamp>/
```

其中包含：

- `config.yaml`
- `metrics.json`
- `selected_seeds.json`
- `run.log`

## 数据集检查

可以先检查配置与图统计是否一致：

```bash
python scripts/inspect_dataset.py --config configs/nethept.yaml
```

输出将包含数据集名称、图路径、有向性、节点数、边数、自环、重复边和连通分量等信息。

## 运行测试

```bash
pytest
```

## 当前基线范围

当前统一入口已接入：

- Degree
- Degree Discount

当前仍**未**接入：

- GNN-CELF
- GRL / DQN
- FeatureDQN

## 第二阶段：独立 GNN

可以使用以下命令训练和评估独立 GNN：

```bash
PYTHONPATH=src python scripts/train_gnn.py --config configs/gnn_nethept.yaml
PYTHONPATH=src python scripts/evaluate_gnn.py --config configs/gnn_nethept.yaml
```

当前评估会输出：

- MAE / RMSE
- Spearman / Kendall 排名相关
- Top-K 召回率
- GNN 选种后的真实 spread
- 与 Degree / Degree Discount 的对比

## 第三阶段：Oracle 诊断实验

可以运行：

```bash
PYTHONPATH=src python scripts/run_oracle_diagnostics.py --config configs/gnn_nethept.yaml
```

当前 Oracle 诊断会记录：

- 每一步的 oracle 最优节点与真实边际增益
- Degree 候选池是否召回 oracle 最优节点
- GNN 候选池是否召回 oracle 最优节点
- 选中节点的 oracle rank
- 选中增益与 oracle 最优增益的比值

当前仓库默认提供的是 **smoke-test 级 Oracle 配置**，会限制候选池、节点子集和步数，避免在大图上直接做全图暴力 oracle 导致运行时间过长。

## 文档

- `docs/DEVELOPMENT_PLAN.md`
- `docs/PAPER_CODE_MAPPING.md`
- `docs/EXPERIMENT_PROTOCOL.md`

## 下一步

后续将继续推进：

1. 建立 GNN 单独评估流程；
2. 补充 Oracle 诊断实验；
3. 重构并对齐模块化 GRL v3；
4. 进一步探索集合条件化边际增益模型。
