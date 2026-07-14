# CODEBASE GUIDE

本文件用于快速说明当前仓库的主线结构、运行入口与各目录职责，帮助后续开发时快速判断：

- 代码从哪里进；
- 不同功能改哪里；
- 哪些目录是主线，哪些目录是历史或资源；
- 复现实验时优先看哪些文件。

## 一句话总览

当前仓库的正式主线是：

- **入口脚本**：`scripts/`
- **核心实现**：`src/grl/`
- **实验配置**：`configs/`
- **回归验证**：`tests/`

历史实现与论文追溯内容保留在：

- `legacy/`

运行时依赖的数据与模型资产位于：

- `data/`
- `param/`

---

## 顶层目录职责

### `scripts/`

这里放的是**可直接运行的实验入口脚本**。如果你想知道“这个项目真正怎么跑”，优先看这里。

当前主要入口包括：

- `scripts/run_experiment.py`
  - 传统 baseline 统一入口
  - 负责读取配置、加载图、运行 baseline、保存 `metrics.json` 等结果
- `scripts/inspect_dataset.py`
  - 数据集检查入口
  - 用于确认图路径、节点数、边数、连通分量等统计是否与预期一致
- `scripts/train_gnn.py`
  - 独立 GNN 训练入口
  - 负责训练传播范围预测模型，并保存模型与训练指标
- `scripts/evaluate_gnn.py`
  - 独立 GNN 评估入口
  - 输出 MAE / RMSE / 排名相关 / Top-K 召回等指标
- `scripts/run_oracle_diagnostics.py`
  - Oracle 诊断入口
  - 用于分析候选池是否覆盖 oracle 最优点、选择损失来自哪里

`scripts/debug/` 中的脚本属于**调试辅助工具**，不是日常主线实验入口。

### `src/grl/`

这里是**当前正式维护的核心代码实现**。原则上，后续功能开发、重构和测试都应优先围绕这里进行。

子模块职责如下：

- `src/grl/baselines/`
  - 当前主线 baseline 实现
  - 已接入：`Degree`、`Degree Discount`
- `src/grl/data/`
  - 图数据加载与校验
  - 关键入口：`load_graph_from_config`
- `src/grl/diffusion/`
  - 扩散仿真逻辑
  - 当前主线主要是 IC（Independent Cascade）
- `src/grl/evaluation/`
  - 评估逻辑
  - 包含 baseline spread 评估，以及 GNN 指标评估
- `src/grl/models/`
  - 模型定义与特征构造
  - 当前主线包含 GNN 预测器、Node2Vec 嵌入加载/构造等能力
- `src/grl/training/`
  - 训练流程封装
  - 当前主要是 `GNNTrainer`
- `src/grl/diagnostics/`
  - Oracle 诊断相关逻辑
- `src/grl/utils/`
  - 配置加载、随机种子设置等公共工具

### `configs/`

这里放实验配置文件。主线脚本几乎都通过 `--config` 从这里读参数。

例如：

- `configs/nethept.yaml`
- `configs/network_science.yaml`
- `configs/gnn_nethept.yaml`

如果你要改：

- 数据集路径
- seed budget
- 扩散概率
- Monte Carlo 次数
- GNN 训练/评估参数

通常先改这里，而不是先改脚本。

### `tests/`

这里是当前主线代码的回归测试。

作用包括：

- 验证 baseline 输出是否合法；
- 验证扩散逻辑是否基本正确；
- 验证图加载与可复现性；
- 验证 GNN 评估与 Oracle 诊断的关键行为。

如果你修改了 `src/grl/` 中的逻辑，应优先考虑是否需要同步更新这里。

### `legacy/`

这里保存的是**历史实现**，主要用于：

- 论文审计；
- 对照实验；
- 追溯 GRL / DQN / GNN-CELF 等历史方案的实现细节。

这个目录**不是当前主线执行链路的一部分**。一般情况下：

- 想跑当前实验，不需要改这里；
- 想追溯论文历史实现，可以看这里。

### `data/`

原始图数据与数据集文本文件。

这部分是运行输入，不是主要逻辑代码。

### `param/`

模型参数与预训练/缓存资产，例如：

- GNN 权重
- Node2Vec 嵌入
- 历史模型文件

这部分通常会被训练脚本、评估脚本和 Oracle 诊断脚本间接使用。

### `docs/`

仓库说明文档。

当前较重要的文档包括：

- `docs/DEVELOPMENT_PLAN.md`
- `docs/PAPER_CODE_MAPPING.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/CODEBASE_GUIDE.md`（本文件）

---

## 常见任务应该从哪里开始看

### 1. 我只想跑一个 baseline 实验

先看：

- `scripts/run_experiment.py`
- `configs/nethept.yaml`（或其他数据集配置）
- `src/grl/baselines/`
- `src/grl/evaluation/spread.py`

### 2. 我想检查数据集是否读对了

先看：

- `scripts/inspect_dataset.py`
- `src/grl/data/graph_loader.py`

### 3. 我想训练或评估当前 GNN

先看：

- `scripts/train_gnn.py`
- `scripts/evaluate_gnn.py`
- `src/grl/training/gnn_trainer.py`
- `src/grl/models/gnn.py`
- `src/grl/evaluation/gnn_metrics.py`

### 4. 我想分析为什么 GNN 选点不如 oracle

先看：

- `scripts/run_oracle_diagnostics.py`
- `src/grl/diagnostics/oracle.py`

### 5. 我想追溯论文里 GRL / DQN / GNN-CELF 的旧实现

先看：

- `legacy/grl.py`
- `legacy/grl-v2.py`
- `legacy/grl-v3.py`
- `docs/PAPER_CODE_MAPPING.md`

---

## 运行入口与执行链路

### Baseline 实验

```bash
python scripts/run_experiment.py --config configs/nethept.yaml
```

典型链路：

- `scripts/run_experiment.py`
- `src/grl/utils/config.py`
- `src/grl/data/graph_loader.py`
- `src/grl/baselines/`
- `src/grl/evaluation/spread.py`
- `src/grl/diffusion/independent_cascade.py`

### 数据集检查

```bash
python scripts/inspect_dataset.py --config configs/nethept.yaml
```

### GNN 训练

```bash
PYTHONPATH=src python scripts/train_gnn.py --config configs/gnn_nethept.yaml
```

### GNN 评估

```bash
PYTHONPATH=src python scripts/evaluate_gnn.py --config configs/gnn_nethept.yaml
```

### Oracle 诊断

```bash
PYTHONPATH=src python scripts/run_oracle_diagnostics.py --config configs/gnn_nethept.yaml
```

---

## 当前边界：哪些不属于主线

以下内容目前**不属于当前主线执行代码**：

- `legacy/` 中的历史实验脚本
- `scripts/debug/` 中的调试脚本
- 文档中提及、但已从根目录删除的旧脚本：
  - `baselines.py`
  - `gnn.py`
  - `gnn_celf.py`

这些内容可能仍有研究价值，但默认不应作为当前实验入口。

---

## 建议的阅读顺序

如果是第一次接手这个仓库，推荐按下面顺序阅读：

1. `README.md`
2. `docs/CODEBASE_GUIDE.md`
3. `scripts/run_experiment.py`
4. `src/grl/data/graph_loader.py`
5. `src/grl/baselines/`
6. `src/grl/diffusion/independent_cascade.py`
7. `scripts/train_gnn.py` / `scripts/evaluate_gnn.py`
8. `src/grl/models/gnn.py`
9. `src/grl/diagnostics/oracle.py`
10. `docs/PAPER_CODE_MAPPING.md` 与 `legacy/`

这样可以先理解当前能跑的主线，再回头理解历史实现与论文映射。