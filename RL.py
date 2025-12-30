import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gnn import MLP
from tqdm import tqdm
from diffusion_model import ic

# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # 返回x中最大的维度
        return x

def step_dqn(G, state, action, IS, count, k):
    # 找到state中为1的位置，即已经选择的种子节点
    seeds = [[] for _ in range(state.shape[0])]
    for i, sublist in enumerate(state):
        for j, value in enumerate(sublist):
            if value == 1:
                seeds[i].append(j)
    for i, node in enumerate(action):
        seeds[i].append(node)
    IS_new = [[] for _ in range(state.shape[0])]
    for i, sublist in enumerate(seeds):
        IS_new[i] = ic(G, seeds[i], 1000)
    reward = [IS_new[i] - IS[i] for i in range(len(IS))]
    # seed_vec是表示出seeds位置的列表
    for i in range(len(seeds)):
        state[i][action] = 1
    if count < k:
        done = False
    else:
        done = True

    return state, IS_new, reward, done

# DQN 训练函数
def train_dqn(G, num_episodes, batch_size, gamma, target_update_freq, k):
    state_dim = len(G.nodes())
    action_dim = 1
    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    memory = []
    total_rewards = []

    for episode in range(num_episodes):
        # state为长度为len(G.nodes())的全0
        state = [0 for i in range(len(G.nodes()))]
        state = torch.FloatTensor(state).unsqueeze(0)
        done = False
        episode_reward = [0]
        IS = [0]
        count = 0
        max_reward = 0

        while not done:
            # 根据 Q 网络选择动作
            with torch.no_grad():
                q_values = q_net(state)
                action = q_values.argmax(dim=1).tolist()
            next_state, IS, reward, done = step_dqn(G, state, action, IS, count, k)
            count += 1
            next_state = torch.FloatTensor(next_state)
            episode_reward += reward

            # 将经验存入记忆
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= batch_size:
                # 从记忆中采样
                batch = np.random.choice(memory, batch_size, replace=False)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.cat(next_states)
                dones = torch.FloatTensor(dones)

                # 计算目标 Q 值
                with torch.no_grad():
                    target_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * target_q_values * (1 - dones)

                # 计算当前 Q 值
                current_q_values = q_net(states).gather(1, actions).squeeze()

                # 计算损失并更新网络
                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_rewards.append(episode_reward)
        # 定期更新目标网络
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode_reward > max_reward:
            max_reward = episode_reward
            torch.save(q_net.state_dict(), 'param/dqn.pth')
            print('save model')

    return total_rewards

# 使用示例
num_episodes = 1000
batch_size = 32
gamma = 0.99
target_update_freq = 100

# 打开一个csv文件
# 使用numpy.genfromtxt读取文件
# data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
data_name = 'musae_PTBR_edges_new'
data = np.genfromtxt('data/'+data_name+'.csv', delimiter=',')
# 取第三列为weights
weights = data[:,2]
# 将前两列变为int
data = data[:,:2].astype(int)

# 对data中的序号进行重新编码
# 先找到所有的节点
nodes = list(set(data.flatten()))
# 重新编码
new_nodes = {node: i for i, node in enumerate(nodes)}
# 重新编码data
data = np.array([[new_nodes[node] for node in edge] for edge in data])

# 从data和weights创建字典
edges = []
for i, edge in enumerate(data):
    edges.append((edge[0], edge[1], weights[i]))

# 做成Graph对象
import networkx as nx
G = nx.DiGraph()
G.add_weighted_edges_from(edges)

#%%
k = 20

rewards = train_dqn(G, num_episodes, batch_size, gamma, target_update_freq, k)

# test
q_net = DQN(len(G.nodes()), 1)
q_net.load_state_dict(torch.load('param/dqn.pth'))
state = [0 for i in range(len(G.nodes()))]
state = torch.FloatTensor(state).unsqueeze(0)
done = False

while not done:
    with torch.no_grad():
        q_values = q_net(state)
        action = q_values.max(1)[1].item()
    next_state, IS, reward, done = step_dqn(G, state, action, 0, 0, k)
    state = next_state

print(IS)
