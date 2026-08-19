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

## 第二阶段：边际增益预测器

当前主线模型不是预测总 spread，而是直接预测给定 seed set `S` 下候选节点 `v` 的条件边际增益：

```text
Delta(v | S) = sigma(S union {v}) - sigma(S)
```

可以使用以下命令训练和评估边际增益预测器：

```bash
PYTHONPATH=src python scripts/train_gnn.py --config configs/gnn_nethept.yaml
PYTHONPATH=src python scripts/evaluate_gnn.py --config configs/gnn_nethept.yaml
```

当前模型直接学习 `Delta(v | S)`，评估会输出：

- MAE / RMSE
- Spearman / Kendall 排名相关
- Top-K 召回率
- pairwise ranking accuracy
- Top-K 召回率
- 条件候选排序指标

边际增益标签使用 common live-edge sampling 生成，使 `sigma(S)` 和 `sigma(S union {v})` 使用同一组扩散样本，避免由于 Monte Carlo 随机噪声产生负边际增益标签。

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

边际增益数据集可以单独生成：

```bash
PYTHONPATH=src python scripts/build_marginal_dataset.py \
  --config configs/smoke/nethept_gnn.yaml \
  --output /tmp/nethept_marginal_dataset.json
```

`configs/smoke/` 用于快速验证，`configs/paper/` 提供多 budget、多随机种子的实验字段。

## 首轮小规模实验

首轮流程验证使用小型 `network_science` 图、单个随机种子和低 MC 次数。运行：

```bash
PYTHONPATH=src python scripts/run_first_smoke.py \
  --config configs/smoke/network_science_first_round.yaml
```

该入口依次完成：

1. Degree / Degree Discount baseline；
2. 边际增益数据集生成；
3. `MarginalGainPredictor` 训练与测试集评估；
4. sequential regret 评估；
5. candidate loss / ranking loss oracle 诊断。

结果保存在：

```text
outputs/first_smoke/network_science/first_round_results.json
```

`outputs/` 默认被 `.gitignore` 忽略，因此实验结果不会自动进入 Git 提交。首轮实际运行结果如下：

| 指标 | 结果 |
|---|---:|
| Degree spread | 20.73 |
| Degree Discount spread | 21.60 |
| MarginalGain MAE | 1.440 |
| MarginalGain RMSE | 1.916 |
| Spearman | -0.377 |
| Kendall | -0.267 |
| Pairwise accuracy | 0.357 |
| Top-1 recall | 0.000 |
| Top-5 recall | 0.800 |
| Sequential cumulative regret | 31.50 |

本轮结果的主要用途是验证代码链路，而不是证明模型已经优于 baseline。结果显示数据集生成、模型训练、sequential 评估和 oracle loss 分解均正常，但在极少样本、低 MC、仅 2 个 epoch 的设置下，边际增益模型排序能力较弱。下一步应增加样本量、MC 次数和训练轮数后，再进行 NetHEPT 单 seed 实验。

## 文档

- `docs/DEVELOPMENT_PLAN.md`
- `docs/PAPER_CODE_MAPPING.md`
- `docs/EXPERIMENT_PROTOCOL.md`

## 下一步

后续将继续推进：

1. 增加边际增益训练样本和 MC 次数；
2. 在 NetHEPT 上进行单 seed 验证；
3. 比较候选召回误差与模型排序误差；
4. 重构并对齐模块化 GRL v3；
5. 进一步接入 FeatureDQN、CELF/IMM 和完整 GRL pipeline。
