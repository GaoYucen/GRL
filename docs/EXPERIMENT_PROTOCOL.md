# EXPERIMENT PROTOCOL

## 目标

建立统一、可复现的实验执行与结果保存规范，避免参数散落在脚本中。

## 统一运行方式

推荐入口：

```bash
python scripts/run_experiment.py --config configs/nethept.yaml
```

## 输出目录规范

```text
outputs/
└── <dataset_name>/
    └── <timestamp>/
        ├── config.yaml
        ├── metrics.json
        ├── selected_seeds.json
        └── run.log
```

## 第一版最小实验内容

当前第一版最小入口优先支持：

- 读取 YAML 配置；
- 固化随机种子；
- 保存本次配置副本；
- 调用 `src/grl/baselines/` 中的 Degree / Degree Discount 基线；
- 输出 seed 集合、spread 均值/标准差与运行耗时。

## 第一版约束

- 先不强制接入旧 GRL 训练流程；
- 先保证“能跑、能留档、能比较”；
- 后续再将 Oracle 诊断、GNN 评估、DQN 召回诊断逐步纳入统一入口。

## 第二阶段补充入口

```bash
PYTHONPATH=src python scripts/train_gnn.py --config configs/gnn_nethept.yaml
PYTHONPATH=src python scripts/evaluate_gnn.py --config configs/gnn_nethept.yaml
```

## 第三阶段补充入口

```bash
PYTHONPATH=src python scripts/run_oracle_diagnostics.py --config configs/gnn_nethept.yaml
```

说明：当前默认 Oracle 配置是 smoke-test 级别，会限制 `candidate_pool_size`、`max_nodes` 和 `step_limit`，用于快速验证诊断流程而不是直接执行全图精细诊断。
