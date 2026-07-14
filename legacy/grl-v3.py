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
        print("Calculating node OUT-degrees...")
        degrees = dict(self.G.out_degree())
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
        
        # [新增关键点] 预拼接所有节点的特征矩阵，供 DQN 直接查询
        # Shape: [N, 65] (64 dim embedding + 1 dim degree)
        self.node_features = torch.cat([self.embeddings, self.norm_degrees], dim=1)
        self.feat_dim = self.node_features.shape[1]
        
        print(f"Node features ready. Dim: {self.feat_dim}")

    def load_graph(self, path):
        """加载 NetHEPT 格式: u v weight"""
        G = nx.DiGraph()
        with open(path, 'r') as f:
            header = f.readline().split() 
            if len(header) == 3: 
                 f.seek(0)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    G.add_edge(u, v, weight=p)
        G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
        return G

    def train_node2vec(self):
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
        spread_sum = 0
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

    def run_degree_discount(self, k):
        d = dict(self.G.out_degree())
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
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v]) * t[v] * 0.01 
        return S

# ==========================================
# 2. GNN 预估模型 (保持不变)
# ==========================================
class ISEstimatorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ISEstimatorGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim + 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings, norm_degrees, seed_mask):
        if embeddings.dim() == 3:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=2) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=1) 
            output = self.fc_out(graph_rep) 
        else:
            x = torch.cat([embeddings, norm_degrees, seed_mask], dim=1) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            graph_rep = torch.sum(x, dim=0) 
            output = self.fc_out(graph_rep) 
        return output

# ==========================================
# 3. [重构] 特征导向型 RL 模型 (Feature-based DQN)
# ==========================================
class FeatureDQN(nn.Module):
    def __init__(self, state_dim, node_feat_dim):
        super(FeatureDQN, self).__init__()
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Value Stream: 评估当前状态的好坏 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream: 不再输出 15000 个值，而是输出一个"偏好向量"
        # 这个向量代表：当前状态下，我想要什么样的节点特征？
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, node_feat_dim) # 输出维度 = 节点特征维度 (65)
        )

    def forward(self, state, node_features):
        """
        state: [B, state_dim]
        node_features: [N, feat_dim] (所有节点的特征矩阵)
        """
        # 1. 提取状态特征
        x = self.feature_layer(state)
        
        # 2. 计算 V(s) [B, 1]
        val = self.value_stream(x)
        
        # 3. 计算偏好权重 W(s) [B, feat_dim]
        weights = self.advantage_stream(x)
        
        # 4. 计算 Q 值 = W(s) * Node_Feats^T
        # 这一步实现了泛化：只要特征匹配，Q值就高
        # weights: [B, D], node_features: [N, D] -> [B, N]
        if weights.dim() == 1: weights = weights.unsqueeze(0)
        
        # 矩阵乘法计算所有节点的 Advantage
        adv = torch.mm(weights, node_features.t()) 
        
        # 5. Dueling 组合: Q = V + (A - mean(A))
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_vals

# ==========================================
# 4. 核心逻辑控制器
# ==========================================
class GRLModel:
    def __init__(self, graph_path, save_dir="param"):
        self.env = GraphEnv(graph_path, save_dir=save_dir)
        self.device = self.env.device
        
        self.gnn = ISEstimatorGNN(input_dim=self.env.embedding_dim).to(self.device)
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=0.001)
        
        # [修改] 使用 FeatureDQN
        self.dqn = FeatureDQN(state_dim=self.env.embedding_dim, node_feat_dim=self.env.feat_dim).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0005)
        self.loss_fn = nn.MSELoss()

    def save_gnn(self, save_dir="param"):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        torch.save(self.gnn.state_dict(), os.path.join(save_dir, "gnn_model.pth"))

    def save_rl(self, save_dir="param"):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        torch.save(self.dqn.state_dict(), os.path.join(save_dir, "dqn_model.pth"))

    def load_gnn(self, save_dir="param"):
        path = os.path.join(save_dir, "gnn_model.pth")
        if os.path.exists(path):
            self.gnn.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False

    def load_rl(self, save_dir="param"):
        path = os.path.join(save_dir, "dqn_model.pth")
        if os.path.exists(path):
            self.dqn.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False

    def train_gnn_estimator(self, total_samples=5000, batch_size=64):
        # ... (此处代码保持不变，请使用上一版 Mixed Sampling 逻辑) ...
        # 为节省篇幅，这里假设已包含之前的 train_gnn_estimator 实现
        # 请直接复制之前优化过的代码块
        print(f"Training GNN Estimator with Mixed Sampling (N={total_samples})...")
        self.gnn.train()
        
        data_buffer = []
        n_random = int(total_samples * 0.4) 
        n_high_deg = int(total_samples * 0.4) 
        n_greedy = total_samples - n_random - n_high_deg
        
        degrees = sorted(dict(self.env.G.out_degree()).items(), key=lambda x: x[1], reverse=True)
        top_1000_nodes = [x[0] for x in degrees[:1000]]
        greedy_seq = self.env.run_degree_discount(50)
        
        pbar = tqdm(total=total_samples, desc="Generating Data")
        
        # 混合采样逻辑同前...
        for _ in range(n_random):
            k = random.randint(1, 20) # 缩小 k 范围适配小样本
            seeds = random.sample(range(self.env.num_nodes), k)
            true_is = self.env.run_ic_simulation(seeds, mc=30)
            mask = torch.zeros((self.env.num_nodes, 1))
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float()))
            pbar.update(1)
            
        for _ in range(n_high_deg):
            k = random.randint(1, 20)
            seeds = random.sample(top_1000_nodes, k)
            true_is = self.env.run_ic_simulation(seeds, mc=30)
            mask = torch.zeros((self.env.num_nodes, 1))
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float()))
            pbar.update(1)

        for _ in range(n_greedy):
            k = random.randint(1, 20)
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
    def train_rl_agent(self, episodes=500, k=10):
        print(f"Training RL Agent (Episodes={episodes})...")
        self.gnn.eval() 
        replay_buffer = deque(maxlen=5000)
        epsilon = 1.0
        min_epsilon = 0.05
        
        # 记录每个 episode 的总奖励
        episode_rewards = []
        
        for episode in tqdm(range(episodes), desc="RL Training"):
            current_seeds = []
            state = torch.zeros(self.env.embedding_dim).to(self.device)
            current_mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
            
            # 【修正点1】这里必须初始化 total_reward
            total_reward = 0
            
            with torch.no_grad():
                current_is_est = self.gnn(
                    self.env.embeddings.unsqueeze(0), 
                    self.env.norm_degrees.unsqueeze(0), 
                    current_mask.unsqueeze(0)
                ).item()
            
            for step in range(k):
                # Epsilon-Greedy
                if random.random() < epsilon:
                    # 混合随机策略
                    if random.random() < 0.5:
                        valid_actions = list(set(range(self.env.num_nodes)) - set(current_seeds))
                        action = random.choice(valid_actions)
                    else:
                        cand = random.randint(0, min(2000, self.env.num_nodes-1))
                        action = cand if cand not in current_seeds else random.choice(list(set(range(self.env.num_nodes)) - set(current_seeds)))
                else:
                    with torch.no_grad():
                        # [关键] 传入 node_features
                        q_values = self.dqn(state.unsqueeze(0), self.env.node_features)
                        q_values = q_values.squeeze(0)
                        q_values[current_seeds] = -float('inf')
                        action = q_values.argmax().item()
                
                new_seeds = current_seeds + [action]
                new_mask = current_mask.clone()
                new_mask[action] = 1.0
                
                with torch.no_grad():
                    new_is_est = self.gnn(
                        self.env.embeddings.unsqueeze(0), 
                        self.env.norm_degrees.unsqueeze(0), 
                        new_mask.unsqueeze(0)
                    ).item()
                
                raw_reward = new_is_est - current_is_est
                reward = raw_reward / 10.0 # Reward Shaping
                
                action_emb = self.env.embeddings[action]
                next_state = state + action_emb
                
                replay_buffer.append((state.clone(), action, reward, next_state.clone()))
                
                # 【修正点2】这里必须累加 reward
                total_reward += reward
                
                state = next_state
                current_seeds = new_seeds
                current_mask = new_mask
                current_is_est = new_is_est
                
                if len(replay_buffer) > 64:
                    self.update_dqn(replay_buffer)
            
            epsilon = max(min_epsilon, epsilon * 0.98)

            # 现在这里才能正确添加 total_reward
            episode_rewards.append(total_reward)
            
            # 每 10 个 Episode 打印一次详细日志
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                log_str = f"\n[Episode {episode+1}] Avg Reward: {avg_reward:.4f} | Epsilon: {epsilon:.4f}\n"
                print(log_str)
                with open("log/rl_train.log", "a") as logf:
                    logf.write(log_str)
                # 监控 Q 值
                with torch.no_grad():
                    sample_q = self.dqn(state.unsqueeze(0), self.env.node_features).max().item()
                log_str = f"    -> Max Q-Value: {sample_q:.4f}\n"
                print(log_str)
                with open("log/rl_train.log", "a") as logf:
                    logf.write(log_str)
                # 监控 GNN 预估值 (k个种子后的最终预估)
                with torch.no_grad():
                    test_est = self.gnn(
                        self.env.embeddings.unsqueeze(0), 
                        self.env.norm_degrees.unsqueeze(0), 
                        current_mask.unsqueeze(0)
                    ).item()
                log_str = f"    -> Final GNN Est Spread: {test_est:.4f}\n"
                print(log_str)
                with open("log/rl_train.log", "a") as logf:
                    logf.write(log_str)

    def update_dqn(self, buffer, batch_size=64):
        batch = random.sample(buffer, batch_size)
        states = torch.stack([x[0] for x in batch]).to(self.device)
        actions = torch.tensor([x[1] for x in batch]).to(self.device)
        rewards = torch.tensor([x[2] for x in batch]).float().to(self.device)
        next_states = torch.stack([x[3] for x in batch]).to(self.device)
        
        # [关键] 传入 node_features
        # curr_q: [B, N] -> gather -> [B]
        all_q_curr = self.dqn(states, self.env.node_features)
        curr_q = all_q_curr.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            all_q_next = self.dqn(next_states, self.env.node_features)
            next_q = all_q_next.max(1)[0]
            target_q = rewards + 0.99 * next_q 
            
        loss = F.smooth_l1_loss(curr_q, target_q)
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
        
        batch_embs = self.env.embeddings.unsqueeze(0)
        batch_degs = self.env.norm_degrees.unsqueeze(0)
        
        with torch.no_grad():
            base_is = self.gnn(batch_embs, batch_degs, current_mask.unsqueeze(0)).item()
        
        for _ in tqdm(range(k), desc="Inferring"):
            with torch.no_grad():
                # [关键] 传入 node_features
                q_values = self.dqn(state.unsqueeze(0), self.env.node_features).squeeze(0)
                q_values[selected_seeds] = -float('inf')
                vals, candidates = torch.topk(q_values, top_k_candidates)
                candidates = candidates.cpu().numpy()
            
            best_cand = -1
            best_score = -float('inf')
            best_is_inc = 0
            
            for cand in candidates:
                cand_mask = current_mask.clone()
                cand_mask[cand] = 1.0
                
                with torch.no_grad():
                    cand_is = self.gnn(batch_embs, batch_degs, cand_mask.unsqueeze(0)).item()
                
                is_inc = cand_is - base_is
                q_val = q_values[cand].item()
                
                # 组合分数
                score = 0.3 * q_val + 0.7 * is_inc 
                
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_is_inc = is_inc
            
            selected_seeds.append(best_cand)
            state = state + self.env.embeddings[best_cand]
            current_mask[best_cand] = 1.0
            base_is += best_is_inc
            
        return selected_seeds

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_gnn', action='store_true')
    parser.add_argument('--train_rl', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--gnn_samples', type=int, default=3000)
    parser.add_argument('--rl_episodes', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='param')
    
    args = parser.parse_args()
    if not (args.train_gnn or args.train_rl or args.eval): args.eval = True

    grl = GRLModel(args.data, save_dir=args.save_dir)
    
    if args.train_gnn:
        print(f"\n>>> Phase 1: Training GNN Estimator")
        start = time.time()
        grl.train_gnn_estimator(total_samples=args.gnn_samples, batch_size=64)
        grl.save_gnn(args.save_dir)
        print(f"Time: {time.time()-start:.2f}s")
    
    if args.train_rl:
        if not args.train_gnn: grl.load_gnn(args.save_dir)
        print(f"\n>>> Phase 2: Training RL Agent")
        start = time.time()
        grl.train_rl_agent(episodes=args.rl_episodes, k=args.k)
        grl.save_rl(args.save_dir)
        print(f"Time: {time.time()-start:.2f}s")

    if args.eval:
        if not args.train_rl: grl.load_rl(args.save_dir)
        if not args.train_gnn: grl.load_gnn(args.save_dir)
        print(f"\n>>> Phase 3: Evaluation (k={args.k})")
        start = time.time()
        seeds = grl.joint_inference(k=args.k)
        print(f"Time: {time.time()-start:.4f}s")
        print(f"Selected Seeds: {seeds}")
        
        print("Verifying Final Spread (MC=1000)...")
        spread = grl.env.run_ic_simulation(seeds, mc=1000)
        print(f"[Metric] Final Spread: {spread:.4f}")