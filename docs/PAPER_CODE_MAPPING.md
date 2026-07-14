# PAPER — CODE MAPPING

> 目的：建立论文描述与当前代码实现之间的一一对应关系，明确一致项、不一致项和后续处理方向。

| 论文模块 | 论文描述 | 当前代码文件 | 当前类或函数 | 是否一致 | 后续处理 |
|---|---|---|---|---|---|
| Node2Vec 节点表示 | 节点嵌入 | `legacy/grl.py` / `legacy/grl-v2.py` / `legacy/grl-v3.py` / `gnn.py` | `train_node2vec`, `get_features` | 部分一致 | 保留并统一接口 |
| GNN 影响估计 | 预测种子集合影响范围 | `legacy/grl-v3.py`, `gnn.py`, `gnn_celf.py` | `ISEstimatorGNN`, `GCNEstimator` | 部分一致 | 后续审计输入/输出定义 |
| Rainbow DQN | RL 主体 | `legacy/grl.py`, `legacy/grl-v2.py`, `legacy/grl-v3.py` | `DuelingDQN`, `FeatureDQN` | 不一致 | 当前仅为简化 DQN/FeatureDQN，未完整实现 Rainbow |
| GNN 奖励 | IS 增量作为 reward | `legacy/grl-v2.py`, `legacy/grl-v3.py` | `train_rl_agent` | 基本一致 | 需验证奖励是否完全对应论文定义 |
| Top-k 候选 | Q 值最高候选 | `legacy/grl-v2.py`, `legacy/grl-v3.py` | `joint_inference` | 一致 | 后续加入候选召回评估 |
| Gated Fusion | 融合 Q 与 IS 增量 | `legacy/grl-v2.py`, `legacy/grl-v3.py` | `joint_inference` | 不一致 | 当前为固定权重融合，不是可学习门控 |
| IC/LT 扩散 | 传播仿真 | `legacy/grl.py`, `legacy/grl-v2.py`, `legacy/grl-v3.py`, `gnn.py`, `baselines.py` | `run_ic_simulation`, `fast_ic_simulation` | 部分一致 | 当前主要是 IC，需要统一接口并预留 LT |
| 传统基线 | Degree / Degree Discount / CELF / IMM | `baselines.py` | `IMBaselines` | 基本一致 | 可作为统一实验入口的 baseline 模块 |
| `grl.py`/`grl-v2.py`/`grl-v3.py` 关系 | 对应哪一阶段实验 | `legacy/` | 脚本级 | 待确认 | 需要结合论文表格与历史提交继续审计 |

## 初步结论

1. 当前仓库**没有完整 Rainbow DQN** 实现；
2. 当前仓库**没有可学习 gated fusion**，只有手工加权；
3. 当前 GNN 主要学习 **总 Influence Spread**，边际增益通过差分得到；
4. 当前代码结构尚未形成统一配置系统与标准实验输出流程。

## 后续补充项

- 核查论文中数据集名称与当前 `data/` 目录是否一致；
- 核查论文主表结果由哪个脚本、配置和随机种子得到；
- 补全 `grl.py`、`grl-v2.py`、`grl-v3.py` 的历史定位说明。
