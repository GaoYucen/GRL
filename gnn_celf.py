import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import heapq
import time
import os
import argparse
from tqdm import tqdm
from numba import njit

# ==========================================
# Part 1: Numba 加速验证内核 (用于最终评估)
# ==========================================
@njit(parallel=False)
def fast_ic_simulation(indptr, indices, data, seeds, mc, num_nodes):
    total_spread = 0.0
    for _ in range(mc):
        visited = np.zeros(num_nodes, dtype=np.bool_)
        queue = np.empty(num_nodes, dtype=np.int32)
        q_start = 0
        q_end = 0
        for s in seeds:
            if not visited[s]:
                visited[s] = True
                queue[q_end] = s
                q_end += 1
        current_spread = q_end
        while q_start < q_end:
            u = queue[q_start]
            q_start += 1
            start_idx = indptr[u]
            end_idx = indptr[u+1]
            for i in range(start_idx, end_idx):
                v = indices[i]
                if visited[v]: continue
                if np.random.random() < data[i]:
                    visited[v] = True
                    queue[q_end] = v
                    q_end += 1
                    current_spread += 1
        total_spread += current_spread
    return total_spread / mc

# ==========================================
# Part 2: GNN 模型定义 (必须与训练时一致)
# ==========================================
class ISEstimatorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ISEstimatorGNN, self).__init__()
        # 输入: Embedding(64) + Degree(1) + Mask(1)
        self.fc1 = nn.Linear(input_dim + 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings, norm_degrees, seed_mask):
        # 支持 Batch [B, N, D] 或 Single [N, D]
        if embeddings.dim() == 3:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=2) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=1) 
        else:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=1) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=0) 
        return self.fc_out(graph_rep)

# ==========================================
# Part 3: GNN-CELF 求解器
# ==========================================
class GNNCELFSolver:
    def __init__(self, graph_path, model_dir="param"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. 加载图
        print(f"Loading graph from {graph_path}...")
        self.G = self.load_graph(graph_path)
        self.nodes = list(self.G.nodes())
        self.num_nodes = len(self.nodes)
        
        # 2. 准备特征 (Embedding & Degree)
        print("Preparing features...")
        emb_path = os.path.join(model_dir, f"node2vec_{os.path.basename(graph_path)}.pth")
        if os.path.exists(emb_path):
            self.embeddings = torch.load(emb_path, map_location=self.device)
            # 确保归一化 (匹配训练时的预处理)
            self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
        else:
            raise FileNotFoundError(f"Embeddings not found at {emb_path}. Please run grl.py --train_gnn first.")
            
        degrees = dict(self.G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        self.norm_degrees = torch.zeros((self.num_nodes, 1)).to(self.device)
        for i in range(self.num_nodes):
            self.norm_degrees[i] = degrees.get(i, 0) / max_degree
            
        self.feat_dim = self.embeddings.shape[1]
        
        # 3. 加载 GNN 模型
        print("Loading GNN model...")
        self.gnn = ISEstimatorGNN(self.feat_dim).to(self.device)
        gnn_path = os.path.join(model_dir, "gnn_model.pth")
        if os.path.exists(gnn_path):
            self.gnn.load_state_dict(torch.load(gnn_path, map_location=self.device))
            self.gnn.eval()
        else:
            raise FileNotFoundError(f"GNN model not found at {gnn_path}")

        # 4. 准备 Numba 验证数据
        adj = nx.to_scipy_sparse_array(self.G, nodelist=range(self.num_nodes), format='csr', weight='weight')
        self.indptr = adj.indptr.astype(np.int32)
        self.indices = adj.indices.astype(np.int32)
        self.data = adj.data.astype(np.float64)
        # 预热 Numba
        fast_ic_simulation(self.indptr, self.indices, self.data, np.array([0], dtype=np.int32), 1, self.num_nodes)

    def load_graph(self, path):
        G = nx.DiGraph()
        with open(path, 'r') as f:
            header = f.readline().split()
            if len(header) == 3: f.seek(0)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    G.add_edge(u, v, weight=p)
        G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
        return G

    def predict_spread(self, current_seeds, candidates=None):
        """
        使用 GNN 预测 Spread。
        如果 candidates 为 None，预测 current_seeds 的 Spread。
        如果 candidates 不为空，批量预测 current_seeds + cand 的 Spread。
        """
        with torch.no_grad():
            # 基础 Mask
            base_mask = torch.zeros((self.num_nodes, 1)).to(self.device)
            if current_seeds:
                base_mask[current_seeds] = 1.0
            
            if candidates is None:
                # 单次预测
                return self.gnn(self.embeddings, self.norm_degrees, base_mask).item()
            else:
                # 批量预测 (Batch Processing)
                batch_size = 128 # 显存允许可调大
                results = []
                
                # 扩展静态特征以匹配 Batch
                # 注意：这里我们分批处理，避免一次性扩展太大爆显存
                num_cands = len(candidates)
                
                for i in range(0, num_cands, batch_size):
                    batch_cands = candidates[i:i+batch_size]
                    curr_batch_size = len(batch_cands)
                    
                    # 构造 Batch Mask
                    # [B, N, 1]
                    batch_masks = base_mask.unsqueeze(0).expand(curr_batch_size, -1, -1).clone()
                    
                    # 在对应的 Batch 维度设置候选点为 1
                    for idx, cand_node in enumerate(batch_cands):
                        batch_masks[idx, cand_node, 0] = 1.0
                    
                    # 扩展 Embeddings 和 Degrees
                    batch_embs = self.embeddings.unsqueeze(0).expand(curr_batch_size, -1, -1)
                    batch_degs = self.norm_degrees.unsqueeze(0).expand(curr_batch_size, -1, -1)
                    
                    # GNN 推理
                    preds = self.gnn(batch_embs, batch_degs, batch_masks)
                    results.extend(preds.cpu().numpy().flatten())
                    
                return results

    def run_celf(self, k):
        print(f"\n[GNN-CELF] Selecting {k} seeds...")
        start_time = time.time()
        
        # --- 1. 初始化: 计算所有节点的第一轮边际收益 ---
        print("  - Initialization Phase (Batch GNN Inference)...")
        # 初始 Spread (空集)
        base_spread = 0.0 # self.predict_spread([]) 理论上是0
        
        # 批量预测所有节点单独作为种子的 Spread
        all_nodes = list(range(self.num_nodes))
        # 使用 GNN 批量预测
        spreads = self.predict_spread([], candidates=all_nodes)
        
        # 构建最大堆 (-gain, node)
        marg_gain_heap = []
        for node, spread in zip(all_nodes, spreads):
            heapq.heappush(marg_gain_heap, (-spread, node))
            
        S = []
        current_gnn_spread = 0.0
        
        # 记录节点上次更新时的 S 长度
        last_update_idx = {node: -1 for node in self.nodes}
        
        # --- 2. 选择阶段 ---
        print("  - Selection Phase...")
        pbar = tqdm(total=k)
        
        while len(S) < k:
            neg_gain, best_node = heapq.heappop(marg_gain_heap)
            gain = -neg_gain
            
            # Lazy Check: 如果是最新数据，直接选中
            if last_update_idx[best_node] == len(S):
                S.append(best_node)
                current_gnn_spread += gain
                pbar.update(1)
                continue
            
            # Re-compute: 只需要调用一次 GNN
            # Gain = GNN(S + {u}) - GNN(S)
            new_spread = self.predict_spread(S + [best_node]) # 这里只预测一个，不用 Batch
            new_gain = new_spread - current_gnn_spread
            
            # 更新堆
            last_update_idx[best_node] = len(S)
            heapq.heappush(marg_gain_heap, (-new_gain, best_node))
            
        pbar.close()
        total_time = time.time() - start_time
        return S, total_time

    def verify_true_spread(self, seeds, mc=2000):
        print(f"\nVerifying True Spread with MC={mc}...")
        seeds_arr = np.array(list(seeds), dtype=np.int32)
        return fast_ic_simulation(self.indptr, self.indices, self.data, seeds_arr, mc, self.num_nodes)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    # 1. 初始化求解器
    solver = GNNCELFSolver(args.data)
    
    # 2. 运行 GNN-CELF
    seeds, runtime = solver.run_celf(args.k)
    
    print("-" * 50)
    print(f"GNN-CELF Selected Seeds: {seeds}")
    print(f"Algorithm Runtime: {runtime:.4f}s")
    print("-" * 50)
    
    # 3. 验证真实效果
    true_spread = solver.verify_true_spread(seeds, mc=2000)
    print(f"[Metric] True Final Spread: {true_spread:.4f}")