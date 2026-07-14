import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import argparse
import os
from gnn_celf import GNNCELFSolver, fast_ic_simulation

def debug_gnn(data_path, k_check=1):
    print(f"=== GNN 诊断模式 (检查 k={k_check} 时的排序能力) ===")
    
    # 1. 初始化 Solver
    solver = GNNCELFSolver(data_path)
    solver.gnn.eval()
    
    # 2. 获取真实度数最高的 Top 20 节点 (作为"优等生"基准)
    print("正在寻找 Degree Top-20 节点...")
    degrees = dict(solver.G.out_degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    
    # 3. 随机选 20 个普通节点 (作为"差生"对照)
    random_nodes = list(np.random.choice(list(solver.G.nodes()), 20, replace=False))
    
    test_nodes = top_nodes + random_nodes
    
    print(f"\n正在对比 {len(test_nodes)} 个节点的 GNN 预测值 vs 真实 MC 值...")
    print(f"{'NodeID':<8} | {'Degree':<8} | {'GNN Pred':<10} | {'True MC':<10} | {'Error':<10}")
    print("-" * 60)
    
    results = []
    
    for node in test_nodes:
        # GNN 预测
        pred = solver.predict_spread([node])
        
        # 真实 MC (跑 200 次保证准)
        seeds_arr = np.array([node], dtype=np.int32)
        true_val = fast_ic_simulation(solver.indptr, solver.indices, solver.data, seeds_arr, 200, solver.num_nodes)
        
        results.append((node, degrees[node], pred, true_val))
        
        print(f"{node:<8} | {degrees[node]:<8} | {pred:<10.4f} | {true_val:<10.4f} | {pred-true_val:<10.4f}")

    # 分析相关性
    preds = np.array([r[2] for r in results])
    trues = np.array([r[3] for r in results])
    correlation = np.corrcoef(preds, trues)[0, 1]
    
    print("-" * 60)
    print(f"GNN与真实值的相关系数: {correlation:.4f} (越接近 1.0 越好)")
    
    if correlation < 0.5:
        print("\n[严重警告] GNN 几乎无法区分好节点和坏节点！")
        print("建议：重新训练 GNN，并在训练数据中强制加入大量 k=1 的样本。")
    else:
        print("\n[通过] GNN 具备基本的排序能力。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt')
    args = parser.parse_args()
    debug_gnn(args.data)