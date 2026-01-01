import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import numpy as np
import random
import os
import argparse
import time
import scipy.sparse as sp
from tqdm import tqdm
from numba import njit
from node2vec import Node2Vec

# ==========================================
# 1. Numba 加速模拟器 (用于生成 Ground Truth)
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
                # 只有掷骰子成功才传播
                if np.random.random() < data[i]:
                    visited[v] = True
                    queue[q_end] = v
                    q_end += 1
                    current_spread += 1
                    
        total_spread += current_spread
    return total_spread / mc

# ==========================================
# 2. 图环境 (包含核心特征工程修复)
# ==========================================
class GraphEnv:
    def __init__(self, graph_path, device):
        self.device = device
        print(f"Loading graph from {graph_path}...")
        self.G = self.load_graph(graph_path)
        self.nodes = list(self.G.nodes())
        self.num_nodes = len(self.nodes)
        
        # 准备 Numba 数据结构
        adj = nx.to_scipy_sparse_array(self.G, nodelist=range(self.num_nodes), format='csr', weight='weight')
        self.indptr = adj.indptr.astype(np.int32)
        self.indices = adj.indices.astype(np.int32)
        self.data = adj.data.astype(np.float64)
        
        # 预热 Numba
        fast_ic_simulation(self.indptr, self.indices, self.data, np.array([0], dtype=np.int32), 1, self.num_nodes)
        
        # [修复 1] 必须使用出度 (Out-Degree)！
        print("Calculating node OUT-degrees...")
        degrees = dict(self.G.out_degree())
        max_degree = max(degrees.values()) if degrees else 1
        self.norm_degrees = torch.zeros((self.num_nodes, 1)).to(self.device)
        for i in range(self.num_nodes):
            self.norm_degrees[i] = degrees.get(i, 0) / max_degree

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

    def get_features(self, save_dir="param"):
        # 加载或训练 Node2Vec
        emb_path = os.path.join(save_dir, "node2vec_emb.pth")
        if os.path.exists(emb_path):
            print("Loading existing Node2Vec embeddings...")
            embeddings = torch.load(emb_path, map_location=self.device)
        else:
            print("Training Node2Vec...")
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            # 这里的参数可以根据需要调优
            node2vec = Node2Vec(self.G, dimensions=64, walk_length=10, num_walks=10, workers=4, quiet=True)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            emb_matrix = np.zeros((self.num_nodes, 64))
            for i in range(self.num_nodes):
                if str(i) in model.wv:
                    emb_matrix[i] = model.wv[str(i)]
            embeddings = torch.FloatTensor(emb_matrix).to(self.device)
            torch.save(embeddings, emb_path)
            
        # [关键] Embedding 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_weighted_adj(self):
        """
        [修复] 构建加权稀疏邻接矩阵
        确保将 CSR 格式转换为 COO 格式以获取 row 和 col 属性
        """
        print("Building Weighted Adjacency Matrix...")
        import scipy.sparse as sp
        
        # 1. 获取加权矩阵 (weight=传播概率)
        # 注意：to_scipy_sparse_array 在新版 networkx 中返回 csr_array
        adj_scipy = nx.to_scipy_sparse_array(self.G, nodelist=range(self.num_nodes), weight='weight', format='csr')
        
        # 2. 添加自环 (节点肯定激活自己，概率=1.0)
        eyes = sp.eye(self.num_nodes, format='csr')
        adj_scipy = adj_scipy + eyes 
        
        # 3. 【关键修复】强制转换为 COO 格式，这样才有 .row 和 .col 属性
        adj_coo = adj_scipy.tocoo()
        
        # 4. 转为 PyTorch Sparse Tensor
        indices = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64))
        values = torch.from_numpy(adj_coo.data.astype(np.float32))
        shape = torch.Size(adj_coo.shape)
        
        adj = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        return adj
    
    def run_degree_discount(self, k):
        # 辅助函数：生成高质量贪心样本
        d = dict(self.G.out_degree())
        dd = d.copy()
        t = {n: 0 for n in self.G.nodes()}
        S = []
        p = 0.01 # 简化平均概率
        for _ in range(k):
            if not dd: break
            u = max(dd, key=dd.get)
            S.append(u)
            dd.pop(u)
            for v in self.G.neighbors(u):
                if v in dd:
                    t[v] += 1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v]) * t[v] * p
        return S

# ==========================================
# 3. 改进版 GCN 模型 (支持稀疏矩阵传播)
# ==========================================
class GCNEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GCNEstimator, self).__init__()
        # Layer 1: 输入(Emb+Deg+Mask) -> Hidden
        self.conv1 = nn.Linear(input_dim + 2, hidden_dim) 
        # Layer 2: Hidden -> Hidden (模拟多跳传播)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        # Layer 3: Readout
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings, norm_degrees, seed_mask, adj):
        # 构建输入特征 [N, D+2]
        x = torch.cat([embeddings, norm_degrees, seed_mask], dim=1)

        # --- GCN Layer 1 ---
        x = self.conv1(x) # Linear Transform
        x = torch.sparse.mm(adj, x) # Message Passing (Propagation)
        x = F.relu(x)

        # --- GCN Layer 2 ---
        x = self.conv2(x)
        x = torch.sparse.mm(adj, x) # Second Hop
        x = F.relu(x)
        
        # --- Readout (Sum Pooling) ---
        # 汇总全图被激活的能量
        graph_rep = torch.sum(x, dim=0)
        
        return self.fc_out(graph_rep)

# ==========================================
# 4. 训练与评估流程
# ==========================================
def generate_dataset(env, num_samples=5000):
    print(f"Generating {num_samples} training samples...")
    data = []
    
    # 比例设置
    n_random_k1 = int(num_samples * 0.25) # [策略] 25% 单节点样本，教它认人
    n_random = int(num_samples * 0.25)    # 25% 随机组合
    n_high_deg = int(num_samples * 0.3)   # 30% 高出度组合
    n_greedy = num_samples - n_random_k1 - n_random - n_high_deg # 20% 贪心序列
    
    # 准备高出度节点池
    degrees = dict(env.G.out_degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
    
    # 准备贪心序列
    greedy_seq = env.run_degree_discount(50)
    
    pbar = tqdm(total=num_samples)
    
    # 1. 纯随机 k=1 (基础能力)
    nodes = list(env.G.nodes())
    for _ in range(n_random_k1):
        seeds = [random.choice(nodes)]
        # 训练时 MC 次数少一点以加快速度，但太少会噪声大，建议 50
        true_val = fast_ic_simulation(env.indptr, env.indices, env.data, np.array(seeds, dtype=np.int32), 50, env.num_nodes)
        
        mask = torch.zeros((env.num_nodes, 1))
        mask[seeds] = 1.0
        data.append((mask, torch.tensor(true_val).float()))
        pbar.update(1)
        
    # 2. 纯随机 k=2~20
    for _ in range(n_random):
        k = random.randint(2, 20)
        seeds = random.sample(nodes, k)
        true_val = fast_ic_simulation(env.indptr, env.indices, env.data, np.array(seeds, dtype=np.int32), 50, env.num_nodes)
        mask = torch.zeros((env.num_nodes, 1))
        mask[seeds] = 1.0
        data.append((mask, torch.tensor(true_val).float()))
        pbar.update(1)

    # 3. 高出度随机
    for _ in range(n_high_deg):
        k = random.randint(1, 20)
        seeds = random.sample(top_nodes, k)
        true_val = fast_ic_simulation(env.indptr, env.indices, env.data, np.array(seeds, dtype=np.int32), 50, env.num_nodes)
        mask = torch.zeros((env.num_nodes, 1))
        mask[seeds] = 1.0
        data.append((mask, torch.tensor(true_val).float()))
        pbar.update(1)
        
    # 4. 贪心子集
    for _ in range(n_greedy):
        k = random.randint(1, 20)
        seeds = greedy_seq[:k]
        true_val = fast_ic_simulation(env.indptr, env.indices, env.data, np.array(seeds, dtype=np.int32), 50, env.num_nodes)
        mask = torch.zeros((env.num_nodes, 1))
        mask[seeds] = 1.0
        data.append((mask, torch.tensor(true_val).float()))
        pbar.update(1)
        
    pbar.close()
    return data

def train_gnn(args):
    # 1. 初始化
    # device = torch.device("cuda" if torch.cuda.is_available() else 
    #                       "mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")
    
    env = GraphEnv(args.data, device)
    embeddings = env.get_features()
    adj = env.get_weighted_adj()
    
    # 2. 数据集
    train_data = generate_dataset(env, num_samples=args.samples)
    
    # 3. 模型
    model = GCNEstimator(input_dim=embeddings.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # 4. 训练
    print("\n>>> Start Training...")
    # 这里的 batch_size 指的是多少个样本更新一次梯度
    # 但由于 GCN 需要全图 adj，我们没法简单地把 N 个节点切分。
    # 这里的做法是：每个样本调用一次 forward (针对全图)，累积 loss，然后 step
    accum_steps = 16 # 相当于 Batch Size
    
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train_data)
        total_loss = 0
        optimizer.zero_grad()
        
        # 进度条
        pbar = tqdm(total=len(train_data), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for i, (mask, label) in enumerate(train_data):
            mask = mask.to(device)
            label = label.to(device)
            
            # Forward
            pred = model(embeddings, env.norm_degrees, mask, adj)
            
            loss = loss_fn(pred.flatten(), label.flatten())
            
            # Backward
            loss.backward()
            total_loss += loss.item()
            
            # Gradient Accumulation
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.2f}"})
            pbar.update(1)
            
        pbar.close()
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_data):.4f}")
        
    # 5. 保存
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "gnn_model.pth"))
    print("Model saved.")

    # 6. 自动评估
    evaluate_model(env, model, embeddings, adj)

def evaluate_model(env, model, embeddings, adj):
    print("\n=== Final Evaluation: k=1 Ranking Ability ===")
    model.eval()
    
    # 选取 Out-Degree 最高的 20 个 和 随机 20 个
    degrees = dict(env.G.out_degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    random_nodes = random.sample(list(env.G.nodes()), 20)
    test_nodes = list(set(top_nodes + random_nodes))
    
    preds = []
    trues = []
    
    print(f"{'Node':<6} | {'OutDeg':<6} | {'Pred':<8} | {'True':<8} | {'Diff':<8}")
    print("-" * 50)
    
    with torch.no_grad():
        for node in test_nodes:
            # 预测
            mask = torch.zeros((env.num_nodes, 1)).to(env.device)
            mask[node] = 1.0
            pred = model(embeddings, env.norm_degrees, mask, adj).item()
            
            # 真实 (高精度 MC)
            true_val = fast_ic_simulation(env.indptr, env.indices, env.data, np.array([node], dtype=np.int32), 1000, env.num_nodes)
            
            preds.append(pred)
            trues.append(true_val)
            print(f"{node:<6} | {degrees[node]:<6} | {pred:<8.2f} | {true_val:<8.2f} | {pred-true_val:<8.2f}")
            
    # 计算相关系数
    corr = np.corrcoef(preds, trues)[0, 1]
    mse = np.mean((np.array(preds) - np.array(trues))**2)
    print("-" * 50)
    print(f"Correlation Coefficient: {corr:.4f}")
    print(f"MSE on Test Set: {mse:.4f}")
    
    if corr > 0.8:
        print("✅ GCN 训练成功！排序能力优秀。")
    elif corr > 0.5:
        print("⚠️ GCN 效果一般，能分清好坏但不够准。")
    else:
        print("❌ GCN 训练失败，无法区分节点优劣。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt')
    parser.add_argument('--samples', type=int, default=3000, help="Training set size")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='param')
    args = parser.parse_args()
    
    train_gnn(args)