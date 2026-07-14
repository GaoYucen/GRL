import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from node2vec import Node2Vec
from collections import deque
from tqdm import tqdm
import os
import argparse
import time
import heapq

# ==========================================
# 1. 图环境与数据加载 (Environment)
# ==========================================
class GraphEnv:
    def __init__(self, graph_path, save_dir="param"):
        # 先cuda，再mps，再cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.G = self.load_graph(graph_path)
        self.nodes = list(self.G.nodes())
        self.num_nodes = len(self.nodes)
        
        # [优化 1] 特征工程：计算归一化度数
        # Degree 也是最重要的先验知识，显式输入给 GNN 可以大幅提升保底效果
        print("Calculating node degrees...")
        degrees = dict(self.G.degree())
        max_degree = max(degrees.values())
        self.norm_degrees = torch.zeros((self.num_nodes, 1)).to(self.device)
        for i in range(self.num_nodes):
            self.norm_degrees[i] = degrees.get(i, 0) / max_degree
        
        # 预计算 Node2Vec Embedding
        emb_filename = os.path.join(save_dir, f"node2vec_{os.path.basename(graph_path)}.pth")
        if os.path.exists(emb_filename):
            print(f"Loading Node2Vec embeddings from {emb_filename}...")
            self.embeddings = torch.load(emb_filename, map_location=self.device)
        else:
            print("Training Node2Vec embeddings...")
            self.embeddings = self.train_node2vec()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.embeddings, emb_filename)
            print(f"Node2Vec embeddings saved to {emb_filename}")
            
        self.embedding_dim = self.embeddings.shape[1]
        print(f"Node embeddings ready. Dim: {self.embedding_dim}")

    def load_graph(self, path):
        """加载 NetHEPT 格式: u v weight"""
        G = nx.DiGraph()
        with open(path, 'r') as f:
            header = f.readline().split() # Skip header N M
            if len(header) == 3: # Handle case where no header exists or different format
                 f.seek(0)
                 
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    G.add_edge(u, v, weight=p)
        # 映射 ID 确保从 0 到 N-1
        G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
        return G

    def train_node2vec(self):
        # 适当增加参数以获得更好的 Embedding 质量
        node2vec = Node2Vec(self.G, dimensions=64, walk_length=10, num_walks=20, workers=4, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        emb_matrix = np.zeros((self.num_nodes, 64))
        for i in range(self.num_nodes):
            if str(i) in model.wv:
                emb_matrix[i] = model.wv[str(i)]
            else:
                emb_matrix[i] = np.random.normal(0, 0.1, 64)
        return torch.FloatTensor(emb_matrix).to(self.device)

    def run_ic_simulation(self, seeds, mc=50):
        """蒙特卡洛模拟 (Ground Truth)"""
        spread_sum = 0
        # 转换为 list 提高遍历速度
        seed_list = list(seeds)
        for _ in range(mc):
            active = set(seed_list)
            new_active = list(seed_list)
            while new_active:
                next_new_active = []
                for u in new_active:
                    if self.G.has_node(u):
                        for v in self.G.neighbors(u):
                            if v not in active:
                                prob = self.G[u][v]['weight']
                                if random.random() < prob:
                                    active.add(v)
                                    next_new_active.append(v)
                new_active = next_new_active
            spread_sum += len(active)
        return spread_sum / mc

    # [优化 2] 辅助函数：Degree Discount 启发式算法
    # 用于生成高质量的训练样本，让 GNN 见过“好种子”
    def run_degree_discount(self, k):
        d = dict(self.G.degree())
        dd = d.copy()
        t = {n: 0 for n in self.G.nodes()}
        S = []
        for _ in range(k):
            u = max(dd, key=dd.get)
            S.append(u)
            dd.pop(u)
            for v in self.G.neighbors(u):
                if v in dd:
                    t[v] += 1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v]) * t[v] * 0.01 # 简化的 p=0.01
        return S

# ==========================================
# 2. GNN 预估模型 (GNN for Estimation)
# ==========================================
class ISEstimatorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ISEstimatorGNN, self).__init__()
        # [优化 1] 输入维度增加：Embedding + Degree(1) + Mask(1)
        self.fc1 = nn.Linear(input_dim + 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings, norm_degrees, seed_mask):
        """
        支持 Batch 或 Single 输入
        embeddings: [B, N, D] 或 [N, D]
        norm_degrees: [B, N, 1] 或 [N, 1]
        seed_mask: [B, N, 1] 或 [N, 1]
        """
        if embeddings.dim() == 3:
            # Batch 模式
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=2) # [B, N, D+2]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # Sum Pooling 聚合全图信息 (对节点维度 N 求和)
            graph_rep = torch.sum(x, dim=1) # [B, hidden_dim]
            output = self.fc_out(graph_rep) # [B, 1]
        else:
            # Single 模式
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=1) # [N, D+2]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # Sum Pooling 聚合全图信息
            graph_rep = torch.sum(x, dim=0) # [hidden_dim]
            output = self.fc_out(graph_rep) # [1]
        return output

# ==========================================
# 3. 强化学习模型 (Rainbow DQN - Simplified)
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Linear(state_dim, 128)
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        x = F.relu(self.feature_layer(state))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean())
        return q_vals

# ==========================================
# 4. 核心逻辑控制器 (GRL Framework)
# ==========================================
class GRLModel:
    def __init__(self, graph_path, save_dir="param"):
        self.env = GraphEnv(graph_path, save_dir=save_dir)
        self.device = self.env.device
        
        self.gnn = ISEstimatorGNN(input_dim=self.env.embedding_dim).to(self.device)
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=0.001)
        
        self.dqn = DuelingDQN(state_dim=self.env.embedding_dim, action_dim=self.env.num_nodes).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0005) # 稍微调低 LR
        self.loss_fn = nn.MSELoss()

    def save_gnn(self, save_dir="param"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.gnn.state_dict(), os.path.join(save_dir, "gnn_model.pth"))
        print(f"GNN model saved to {save_dir}/gnn_model.pth")

    def save_rl(self, save_dir="param"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.dqn.state_dict(), os.path.join(save_dir, "dqn_model.pth"))
        print(f"RL model saved to {save_dir}/dqn_model.pth")

    def load_gnn(self, save_dir="param"):
        path = os.path.join(save_dir, "gnn_model.pth")
        if os.path.exists(path):
            self.gnn.load_state_dict(torch.load(path, map_location=self.device))
            print(f"GNN model loaded from {path}")
            return True
        print(f"Warning: GNN model not found at {path}")
        return False

    def load_rl(self, save_dir="param"):
        path = os.path.join(save_dir, "dqn_model.pth")
        if os.path.exists(path):
            self.dqn.load_state_dict(torch.load(path, map_location=self.device))
            print(f"RL model loaded from {path}")
            return True
        print(f"Warning: RL model not found at {path}")
        return False

    def save_models(self, save_dir="param"):
        self.save_gnn(save_dir)
        self.save_rl(save_dir)

    def load_models(self, save_dir="param"):
        success_gnn = self.load_gnn(save_dir)
        success_rl = self.load_rl(save_dir)
        if not (success_gnn and success_rl):
            print("Warning: Some models could not be loaded.")
        
    # --- 阶段 1: 训练 GNN 预估器 (包含核心优化) ---
    def train_gnn_estimator(self, total_samples=5000, batch_size=64):
        """
        [优化 2] 混合采样策略 (Mixed Sampling Strategy)
        不再只用 random，而是混合 Random + HighDegree + Greedy
        """
        print(f"Training GNN Estimator with Mixed Sampling (N={total_samples})...")
        self.gnn.train()
        
        data_buffer = []
        
        # 1. 准备采样比例
        n_random = int(total_samples * 0.4) # 40% 随机 (见识差样本)
        n_high_deg = int(total_samples * 0.4) # 40% 高度数 (见识好样本)
        n_greedy = total_samples - n_random - n_high_deg # 20% 贪心 (见识极好样本)
        
        # 预计算 Top Degree Nodes
        print("   - Preparing High Degree Candidates...")
        degrees = sorted(dict(self.env.G.degree()).items(), key=lambda x: x[1], reverse=True)
        top_1000_nodes = [x[0] for x in degrees[:1000]]
        
        # 预计算 Greedy Sequence
        print("   - Running Degree Discount for Greedy Samples...")
        greedy_seq = self.env.run_degree_discount(50)
        
        # --- 生成数据 ---
        pbar = tqdm(total=total_samples, desc="Generating Data")
        
        # A. Random Samples
        for _ in range(n_random):
            k = random.randint(1, 50)
            seeds = random.sample(range(self.env.num_nodes), k)
            true_is = self.env.run_ic_simulation(seeds, mc=30) # 训练时 MC 可以少一点加快速度
            mask = torch.zeros((self.env.num_nodes, 1))
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float()))
            pbar.update(1)
            
        # B. High Degree Samples
        for _ in range(n_high_deg):
            k = random.randint(1, 50)
            seeds = random.sample(top_1000_nodes, k)
            true_is = self.env.run_ic_simulation(seeds, mc=30)
            mask = torch.zeros((self.env.num_nodes, 1))
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float()))
            pbar.update(1)
            
        # C. Greedy Samples (Subsets of the greedy sequence)
        for _ in range(n_greedy):
            k = random.randint(1, 50)
            seeds = greedy_seq[:k]
            true_is = self.env.run_ic_simulation(seeds, mc=30)
            mask = torch.zeros((self.env.num_nodes, 1))
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float()))
            pbar.update(1)
            
        pbar.close()

        # --- 训练循环 ---
        epochs = 50
        print(f"   - Starting GNN Training Loop ({epochs} epochs)...")
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(data_buffer)
            
            # Batch Training
            num_batches = len(data_buffer) // batch_size
            batch_pbar = tqdm(range(num_batches), desc=f"      Epoch {epoch+1}/{epochs}", leave=False)
            for i in batch_pbar:
                batch = data_buffer[i*batch_size : (i+1)*batch_size]
                
                # 堆叠 Batch 数据
                masks = torch.stack([x[0] for x in batch]).to(self.device) # [B, N, 1]
                labels = torch.stack([x[1] for x in batch]).to(self.device).unsqueeze(1) # [B, 1]
                
                # Expand embeddings and degrees to batch size
                # Embs: [B, N, D], Degrees: [B, N, 1]
                batch_embs = self.env.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                batch_degs = self.env.norm_degrees.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Forward (Modified GNN with Degree Input)
                preds = self.gnn(batch_embs, batch_degs, masks)
                
                loss = F.mse_loss(preds, labels)
                
                self.gnn_optimizer.zero_grad()
                loss.backward()
                self.gnn_optimizer.step()
                total_loss += loss.item()
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if (epoch+1) % 5 == 0:
                print(f"      [GNN] Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/num_batches:.4f}")

    # --- 阶段 2: 训练 RL Agent ---
    def train_rl_agent(self, episodes=500, k=50):
        print(f"Training RL Agent (Episodes={episodes})...")
        self.gnn.eval() 
        replay_buffer = deque(maxlen=5000)
        epsilon = 1.0
        min_epsilon = 0.05
        
        for episode in tqdm(range(episodes), desc="RL Training"):
            current_seeds = []
            state = torch.zeros(self.env.embedding_dim).to(self.device)
            current_mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
            
            # 初始 IS 预估
            with torch.no_grad():
                # 注意维度匹配：Input 需要增加 batch 维度
                current_is_est = self.gnn(
                    self.env.embeddings.unsqueeze(0), 
                    self.env.norm_degrees.unsqueeze(0), 
                    current_mask.unsqueeze(0)
                ).item()
            
            total_reward = 0
            
            for step in range(k):
                # Epsilon-Greedy
                if random.random() < epsilon:
                    # 简单优化：随机时，偶尔也偏向选 Degree 大的，加快探索
                    if random.random() < 0.5:
                        valid_actions = list(set(range(self.env.num_nodes)) - set(current_seeds))
                        action = random.choice(valid_actions)
                    else:
                         # 随机选 Top 2000 的节点
                         cand = random.randint(0, min(2000, self.env.num_nodes-1))
                         action = cand if cand not in current_seeds else random.choice(list(set(range(self.env.num_nodes)) - set(current_seeds)))

                else:
                    with torch.no_grad():
                        q_values = self.dqn(state)
                        q_values[current_seeds] = -float('inf')
                        action = q_values.argmax().item()
                
                # 执行 Action
                new_seeds = current_seeds + [action]
                
                new_mask = current_mask.clone()
                new_mask[action] = 1.0
                
                with torch.no_grad():
                    new_is_est = self.gnn(
                        self.env.embeddings.unsqueeze(0), 
                        self.env.norm_degrees.unsqueeze(0), 
                        new_mask.unsqueeze(0)
                    ).item()
                
                # [优化 3] Reward Shaping: 归一化奖励
                # 原始增量可能很大 (e.g. 50)，除以 10 让其落在较小区间，利于 NN 收敛
                raw_reward = new_is_est - current_is_est
                reward = raw_reward / 10.0 
                
                action_emb = self.env.embeddings[action]
                next_state = state + action_emb
                
                replay_buffer.append((state.clone(), action, reward, next_state.clone()))
                
                state = next_state
                current_seeds = new_seeds
                current_mask = new_mask
                current_is_est = new_is_est
                total_reward += reward
                
                if len(replay_buffer) > 64:
                    self.update_dqn(replay_buffer)
            
            # Epsilon Decay
            epsilon = max(min_epsilon, epsilon * 0.99)
            
            if (episode+1) % 10 == 0:
                pass # 可以打印 total_reward 观察收敛

    def update_dqn(self, buffer, batch_size=64):
        batch = random.sample(buffer, batch_size)
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch]).to(self.device)
        rewards = torch.tensor([x[2] for x in batch]).float().to(self.device)
        next_states = torch.stack([x[3] for x in batch])
        
        curr_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.dqn(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q # Gamma = 0.99
            
        loss = F.smooth_l1_loss(curr_q, target_q) # Hubber Loss 比 MSE 更稳
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --- 阶段 3: 联合推理 ---
    def joint_inference(self, k=50, top_k_candidates=10):
        print(f"Running Joint Inference (k={k})...")
        self.dqn.eval()
        self.gnn.eval()
        
        selected_seeds = []
        state = torch.zeros(self.env.embedding_dim).to(self.device)
        current_mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
        
        # 预加载数据进 Batch 以加速 inference
        batch_embs = self.env.embeddings.unsqueeze(0)
        batch_degs = self.env.norm_degrees.unsqueeze(0)
        
        # 门控权重 alpha
        alpha = 0.6 
        
        with torch.no_grad():
            base_is = self.gnn(batch_embs, batch_degs, current_mask.unsqueeze(0)).item()
        
        for _ in tqdm(range(k), desc="Inferring"):
            # 1. DQN 选 Top-K
            with torch.no_grad():
                q_values = self.dqn(state)
                q_values[selected_seeds] = -float('inf')
                vals, candidates = torch.topk(q_values, top_k_candidates)
                candidates = candidates.cpu().numpy()
            
            best_cand = -1
            best_score = -float('inf')
            best_is_inc = 0
            
            # 2. GNN 精确预估
            for cand in candidates:
                cand_mask = current_mask.clone()
                cand_mask[cand] = 1.0
                
                with torch.no_grad():
                    cand_is = self.gnn(batch_embs, batch_degs, cand_mask.unsqueeze(0)).item()
                
                is_inc = cand_is - base_is
                q_val = q_values[cand].item()
                
                # 简单归一化: 假设 Q 值和 IS Inc 都在类似量级，或者手动调节 alpha
                # 由于我们 Reward 归一化过，Q 值大概率较小，IS Inc 较大
                # 这里我们主要信赖 GNN (真实增量)
                score = 0.3 * q_val + 0.7 * is_inc 
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_is_inc = is_inc
            
            selected_seeds.append(best_cand)
            
            # Update
            state = state + self.env.embeddings[best_cand]
            current_mask[best_cand] = 1.0
            base_is += best_is_inc
            
        return selected_seeds

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRL Influence Maximization Framework")
    # 模式控制
    parser.add_argument('--train_gnn', action='store_true', help="Train GNN Estimator (Slow)")
    parser.add_argument('--train_rl', action='store_true', help="Train RL Agent (Fast)")
    parser.add_argument('--eval', action='store_true', help="Run evaluation on test set")
    
    # 数据与超参
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt', help="Graph data path")
    parser.add_argument('--k', type=int, default=10, help="Number of seeds to select")
    parser.add_argument('--gnn_samples', type=int, default=3000, help="Number of samples for GNN training")
    parser.add_argument('--rl_episodes', type=int, default=100, help="Number of episodes for RL training")
    parser.add_argument('--save_dir', type=str, default='param', help="Directory to save/load models")
    
    args = parser.parse_args()

    # 如果没有任何 flag，默认进入 eval 模式
    if not (args.train_gnn or args.train_rl or args.eval):
        args.eval = True

    grl = GRLModel(args.data, save_dir=args.save_dir)
    
    # 1. GNN 阶段
    if args.train_gnn:
        print(f"\n>>> Phase 1: Training GNN Estimator (Samples: {args.gnn_samples})")
        start_gnn = time.time()
        grl.train_gnn_estimator(total_samples=args.gnn_samples, batch_size=64)
        grl.save_gnn(args.save_dir)
        print(f">>> GNN Training Finished: {time.time() - start_gnn:.2f}s")
    else:
        # 如果不训练 GNN，但要训练 RL 或 Eval，则必须加载 GNN
        if args.train_rl or args.eval:
            if not grl.load_gnn(args.save_dir):
                print("Error: GNN model is required. Please run with --train_gnn first.")
                exit()

    # 2. RL 阶段
    if args.train_rl:
        print(f"\n>>> Phase 2: Training RL Agent (Episodes: {args.rl_episodes})")
        start_rl = time.time()
        grl.train_rl_agent(episodes=args.rl_episodes, k=args.k)
        grl.save_rl(args.save_dir)
        print(f">>> RL Training Finished: {time.time() - start_rl:.2f}s")
    else:
        # 如果不训练 RL，但要 Eval，则必须加载 RL
        if args.eval:
            if not grl.load_rl(args.save_dir):
                print("Error: RL model is required. Please run with --train_rl first.")
                exit()

    # 3. 评估阶段
    if args.eval:
        print(f"\n>>> Phase 3: Running Evaluation (k={args.k})...")
        start_infer = time.time()
        seeds = grl.joint_inference(k=args.k)
        infer_time = time.time() - start_infer
    
        print("-" * 50)
        print(f"Selected Seeds: {seeds}")
        print("-" * 50)
        print(f"[Metric] Inference Time: {infer_time:.4f}s")
    
        print("Verifying Final Spread (MC=1000)...")
        spread = grl.env.run_ic_simulation(seeds, mc=1000)
        print(f"[Metric] Final Spread: {spread:.4f}")