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
import os  # 需要导入 os 库
import argparse
import time 

# ==========================================
# 1. 图环境与数据加载 (Environment)
# ==========================================
class GraphEnv:
    def __init__(self, graph_path, save_dir="param"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = self.load_graph(graph_path)
        self.nodes = list(self.G.nodes())
        self.num_nodes = len(self.nodes)
        self.node_map = {node: i for i, node in enumerate(self.nodes)} # 原始ID -> 索引
        self.idx_map = {i: node for i, node in enumerate(self.nodes)}  # 索引 -> 原始ID
        
        # 预计算 Node2Vec Embedding [cite: 678]
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
            # 跳过第一行 (N M)
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    G.add_edge(u, v, weight=p)
        # 重新映射节点ID为 0 ~ N-1
        G = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
        return G

    def train_node2vec(self):
        """论文 4.2.1: 使用 Node2Vec 获取用户嵌入 """
        # 为了演示速度，这里参数设得较小，实际使用请调大
        node2vec = Node2Vec(self.G, dimensions=64, walk_length=10, num_walks=10, workers=4, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # 按节点索引顺序提取 embedding
        emb_matrix = np.zeros((self.num_nodes, 64))
        for i in range(self.num_nodes):
            if str(i) in model.wv:
                emb_matrix[i] = model.wv[str(i)]
            else:
                emb_matrix[i] = np.random.normal(0, 0.1, 64)
        return torch.FloatTensor(emb_matrix).to(self.device)

    def run_ic_simulation(self, seeds, mc=50):
        """蒙特卡洛模拟 (作为 GNN 训练的 Ground Truth)"""
        spread_sum = 0
        for _ in range(mc):
            active = set(seeds)
            new_active = set(seeds)
            while new_active:
                next_new_active = set()
                for u in new_active:
                    for v in self.G.neighbors(u):
                        if v not in active:
                            prob = self.G[u][v]['weight']
                            if random.random() < prob:
                                active.add(v)
                                next_new_active.add(v)
                new_active = next_new_active
                active.update(new_active)
            spread_sum += len(active)
        return spread_sum / mc

# ==========================================
# 2. GNN 预估模型 (GNN for Estimation)
# ==========================================
class ISEstimatorGNN(nn.Module):
    """
    论文 4.2.1: 用于快速预估给定种子集的 Influence Spread 
    输入: Node Embedding + Seed Status (Binary)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(ISEstimatorGNN, self).__init__()
        # 论文提到使用 GNN + FC Layer [cite: 781]
        # 这里实现一个简化的 GCN 变体，输入特征为 (Embedding + is_seed)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1) # 输出标量 IS
        
    def forward(self, embeddings, seed_mask, adj_matrix=None):
        """
        embeddings: [N, D]
        seed_mask: [N, 1] (0 or 1 indicating if node is seed)
        """
        # 拼接静态 Embedding 和 动态 Seed Mask
        x = torch.cat([embeddings, seed_mask], dim=1) # [N, D+1]
        
        # 简单的图聚合模拟 (这里用 MLP 替代复杂 GCN 以简化实现，实际应用可用 torch_geometric)
        # 论文提到单调性约束 (Monotonicity) [cite: 783]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 聚合所有节点信息预测全图 IS
        # Global Pooling (Sum or Mean)
        graph_rep = torch.sum(x, dim=0) 
        
        # 论文提到使用 Sigmoid 确保单调增长特性 
        # 但 IS 是数值，这里可能指内部激活，最终输出通常经过 Scaling
        output = self.fc_out(graph_rep)
        return output

# ==========================================
# 3. 强化学习模型 (Rainbow DQN - Simplified)
# ==========================================
class DuelingDQN(nn.Module):
    """
    论文 4.3: Rainbow DQN (此处简化为 Dueling Double DQN) [cite: 788, 1101]
    State: Sum of node embedding of selected seeds [cite: 838, 1103]
    Action: Select a new node
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Linear(state_dim, 128)
        
        # Dueling Network Architecture
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
        # Q = V + (A - mean(A))
        q_vals = value + (advantage - advantage.mean())
        return q_vals

# ==========================================
# 4. 核心逻辑控制器 (GRL Framework)
# ==========================================
class GRLModel:
    def __init__(self, graph_path, save_dir="param"):
        self.env = GraphEnv(graph_path, save_dir=save_dir)
        self.device = self.env.device
        
        # 初始化 GNN
        self.gnn = ISEstimatorGNN(input_dim=self.env.embedding_dim).to(self.device)
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=0.001)
        
        # 初始化 DQN
        self.dqn = DuelingDQN(state_dim=self.env.embedding_dim, action_dim=self.env.num_nodes).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def save_models(self, save_dir="param"):
        """保存训练好的 GNN 和 DQN 模型参数"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(self.gnn.state_dict(), os.path.join(save_dir, "gnn_model.pth"))
        torch.save(self.dqn.state_dict(), os.path.join(save_dir, "dqn_model.pth"))
        print(f"Models saved to {save_dir}/")

    def load_models(self, save_dir="param"):
        """加载模型参数 (用于推理阶段)"""
        gnn_path = os.path.join(save_dir, "gnn_model.pth")
        dqn_path = os.path.join(save_dir, "dqn_model.pth")
        
        if os.path.exists(gnn_path) and os.path.exists(dqn_path):
            self.gnn.load_state_dict(torch.load(gnn_path))
            self.dqn.load_state_dict(torch.load(dqn_path))
            print(f"Models loaded from {save_dir}/")
        else:
            print("No saved models found. Please train first.")
        
    # --- 阶段 1: 训练 GNN 预估器 ---
    def train_gnn_estimator(self, num_samples=1000, batch_size=32):
        """
        论文 4.2.1: Supervised Learning for GNN [cite: 774]
        Label: MC Simulation result [cite: 777]
        """
        print("Training GNN Estimator...")
        self.gnn.train()
        
        # 生成训练数据
        data_buffer = []
        for _ in tqdm(range(num_samples), desc="Generating GNN Data"):
            # 随机生成大小为 1~50 的种子集
            k = random.randint(1, 20) # [cite: 1007] Paper uses 20 for GNN training
            seeds = random.sample(range(self.env.num_nodes), k)
            
            # Ground Truth IS
            true_is = self.env.run_ic_simulation(seeds, mc=50)
            
            # 构建 Seed Mask
            mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
            mask[seeds] = 1.0
            data_buffer.append((mask, torch.tensor(true_is).float().to(self.device)))

        # 训练循环
        for epoch in range(100): # [cite: 1026] Paper uses 100 epochs
            total_loss = 0
            random.shuffle(data_buffer)
            
            for i in range(0, len(data_buffer), batch_size):
                batch = data_buffer[i:i+batch_size]
                masks = torch.stack([x[0] for x in batch]) # [B, N, 1]
                labels = torch.stack([x[1] for x in batch]) # [B]
                
                # 由于 GNN 实现简单，这里逐个 mask loop (实际应 batch 处理)
                preds = []
                for j in range(len(batch)):
                    pred = self.gnn(self.env.embeddings, batch[j][0])
                    preds.append(pred)
                preds = torch.cat(preds)
                
                loss = F.mse_loss(preds, labels) # MSE Loss 
                
                self.gnn_optimizer.zero_grad()
                loss.backward()
                self.gnn_optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, GNN Loss: {total_loss:.4f}")

    # --- 阶段 2: 训练 RL Agent ---
    def train_rl_agent(self, episodes=1000, k=50):
        """
        论文 4.3: Training of RL model [cite: 787]
        Reward: IS increment estimated by GNN [cite: 792, 1104]
        """
        print("Training RL Agent...")
        self.gnn.eval() # GNN 固定，用于提供 Reward
        replay_buffer = deque(maxlen=2000)
        epsilon = 1.0
        
        for episode in tqdm(range(episodes), desc="RL Training"):
            current_seeds = []
            # 初始 State: 种子集 Embedding 之和 (空集为0)
            state = torch.zeros(self.env.embedding_dim).to(self.device)
            
            # 当前的 IS 预估值 (Baseline)
            current_mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
            with torch.no_grad():
                current_is_est = self.gnn(self.env.embeddings, current_mask).item()
            
            total_reward = 0
            
            for step in range(k): # 逐步选 k 个点
                # Epsilon-Greedy Action [cite: 840, 1105]
                if random.random() < epsilon:
                    # 随机选一个不在 current_seeds 里的点
                    valid_actions = list(set(range(self.env.num_nodes)) - set(current_seeds))
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = self.dqn(state)
                        # Mask 掉已选的点
                        q_values[current_seeds] = -float('inf')
                        action = q_values.argmax().item()
                
                # 执行 Action
                new_seeds = current_seeds + [action]
                
                # 计算 Reward (由 GNN 提供)
                new_mask = current_mask.clone()
                new_mask[action] = 1.0
                
                with torch.no_grad():
                    new_is_est = self.gnn(self.env.embeddings, new_mask).item()
                
                # Reward = IS Increment 
                reward = new_is_est - current_is_est
                
                # 下一个 State
                action_emb = self.env.embeddings[action]
                next_state = state + action_emb # Sum of embeddings 
                
                # 存入 Buffer
                replay_buffer.append((state.clone(), action, reward, next_state.clone()))
                
                # 更新状态
                state = next_state
                current_seeds = new_seeds
                current_mask = new_mask
                current_is_est = new_is_est
                total_reward += reward
                
                # 训练 DQN
                if len(replay_buffer) > 64:
                    self.update_dqn(replay_buffer)
            
            # 衰减 Epsilon
            epsilon = max(0.01, epsilon * 0.995)

    def update_dqn(self, buffer, batch_size=32):
        batch = random.sample(buffer, batch_size)
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch]).to(self.device)
        rewards = torch.tensor([x[2] for x in batch]).float().to(self.device)
        next_states = torch.stack([x[3] for x in batch])
        
        # 计算当前 Q
        curr_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q (Gamma=0.99)
        with torch.no_grad():
            next_q = self.dqn(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q
            
        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --- 阶段 3: 联合推理 ---
    def joint_inference(self, k=50, top_k_candidates=5):
        """
        论文 4.4: Joint Inference Module [cite: 858, 932]
        结合 Q-Value 和 GNN 预估增量进行选点
        """
        print("Running Joint Inference...")
        self.dqn.eval()
        self.gnn.eval()
        
        selected_seeds = []
        state = torch.zeros(self.env.embedding_dim).to(self.device)
        current_mask = torch.zeros((self.env.num_nodes, 1)).to(self.device)
        
        # 门控单元参数 (论文中称为 Gated Unit, 这里简化为加权平均) [cite: 936]
        # 实际论文可能使用可学习的 Gate，这里设为固定权重或简单的 Linear 层
        # 假设 alpha 控制权重
        alpha = 0.5 
        
        for _ in tqdm(range(k), desc="Inferring Seeds"):
            # 1. 获取 DQN 的 Top-K Q-value 候选人 
            with torch.no_grad():
                q_values = self.dqn(state)
                # Mask 已选
                q_values[selected_seeds] = -float('inf')
                # 选出 Top-K 候选者
                vals, candidates = torch.topk(q_values, top_k_candidates)
                candidates = candidates.cpu().numpy()
            
            best_candidate = -1
            best_score = -float('inf')
            
            # 当前 GNN 基础值
            with torch.no_grad():
                base_is = self.gnn(self.env.embeddings, current_mask).item()
            
            # 2. 对每个候选人，用 GNN 算真实增量 
            for cand in candidates:
                # 计算 Q-value (Normalized or Raw)
                q_val = q_values[cand].item()
                
                # 计算 GNN IS Increment
                cand_mask = current_mask.clone()
                cand_mask[cand] = 1.0
                with torch.no_grad():
                    cand_is = self.gnn(self.env.embeddings, cand_mask).item()
                is_increment = cand_is - base_is
                
                # 3. 联合打分 
                # 注意：Q值和IS增量量级不同，实际需归一化。这里直接加权演示。
                score = alpha * q_val + (1 - alpha) * is_increment
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            # 选定
            selected_seeds.append(best_candidate)
            
            # 更新状态
            action_emb = self.env.embeddings[best_candidate]
            state = state + action_emb
            current_mask[best_candidate] = 1.0
            
        return selected_seeds

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 配置命令行参数
    parser = argparse.ArgumentParser(description="GRL Influence Maximization Model")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                        help="运行模式: 'train' (重新训练) 或 'test' (加载 param/ 中的模型进行测试)")
    parser.add_argument('--data', type=str, default='data/NetHEPT.txt', help="数据集路径")
    parser.add_argument('--k', type=int, default=50, help="种子节点数量 k")
    args = parser.parse_args()

    # 2. 初始化环境与模型
    grl = GRLModel(args.data)
    
    if args.mode == 'train':
        # ==========================
        # 模式 A: 训练模式
        # ==========================
        print(f">>> 开始训练模式 (数据集: {args.data})")
        start_train = time.time()
        
        # 预训练 GNN 预估器 [cite: 774, 1026]
        grl.train_gnn_estimator(num_samples=200, batch_size=16) 
        
        # 训练 RL Agent [cite: 787, 1106]
        grl.train_rl_agent(episodes=50, k=args.k) 
        
        print(f">>> 训练完成，总耗时: {time.time() - start_train:.2f}s")
        
        # 自动保存模型到 param/ 文件夹
        grl.save_models("param")
        
    else:
        # ==========================
        # 模式 B: 测试模式 (默认)
        # ==========================
        print(f">>> 开始测试模式 (正在从 param/ 加载模型...)")
        grl.load_models("param")

    # 3. 执行评估 (无论训练还是测试，最后都会运行推理并计算结果)
    print(f"\n>>> 执行联合推理 (k={args.k})...")
    
    start_infer = time.time()
    # 联合推理模块结合 Q值与 GNN 增量 [cite: 858, 932, 1107]
    final_seeds = grl.joint_inference(k=args.k, top_k_candidates=5) 
    infer_time = time.time() - start_infer

    print("-" * 50)
    print(f"最终选定的种子节点: {final_seeds}")
    print("-" * 50)
    
    # 输出论文关注的核心指标 [cite: 1010, 1028]
    print(f"[指标] 推理时间 (Inference Time): {infer_time:.4f} 秒")
    
    # 使用高精度蒙特卡洛模拟验证最终影响力范围 [cite: 777, 1026]
    print("正在进行高精度验证 (MC=1000)...")
    final_spread = grl.env.run_ic_simulation(final_seeds, mc=1000)
    print(f"[指标] 最终影响力范围 (Final Spread): {final_spread:.4f}")