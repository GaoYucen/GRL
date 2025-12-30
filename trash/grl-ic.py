#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
# 取第三列为weights
weights = data[:,2]
# 将前两列转换为整数类型
data = data[:, :2].astype(int)
# 从data和weights创建字典
edges = []
for i, edge in enumerate(data):
    edges.append((edge[0], edge[1], weights[i]))
#%%
# 创建图
import networkx as nx
# 使用字典创建图
G = nx.DiGraph()
G.add_weighted_edges_from(edges)

#%% 确认节点和边集合，1912个节点，31299条边
nodes = list(G.nodes)
edges = list(G.edges)

#%%
import torch
from gnn import MLP
from tqdm import tqdm
# 选择种子节点
import random
# 计算影响力
from diffusion_model import ic
threshold = 0.9

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = MLP(input_dim=len(nodes), output_dim=1, num_layers=3, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.L1Loss()

num_epoches = 100
batch_size = 64

#%% 训练模型
for epoch in tqdm(range(num_epoches)):
    seeds_list = []
    IS_list = []
    for i in range(0, batch_size):
        seeds = random.sample(nodes, 20)
        # 根据seeds计算vector
        seed_vec = [1 if node in seeds else 0 for node in nodes]
        for node in seeds:
            for succ in G.successors(node):
                    seed_vec[succ] += 0.5
        IS = ic(G, seeds, 1000)
        seeds_list.append(seed_vec)
        IS_list.append(IS)
    optimizer.zero_grad()
    seeds_list = torch.tensor(seeds_list).to(dtype=torch.float32).to(device)
    pred = model(seeds_list)
    IS_list = torch.tensor(IS_list).to(dtype=torch.float32).to(device)
    loss = criterion(pred, IS_list)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch:', epoch)
        print('loss:', loss)
#存储模型数据
torch.save(model.state_dict(), 'model-ic.pth')

#%% 测试模型
# 选择种子节点
import random
import time
# 加载保存的状态字典
model_params = torch.load('model-ic.pth')
# 将参数加载到模型中
model.load_state_dict(model_params)
model.eval()
seeds = random.sample(nodes, 20)
start_time_mlp = time.time()
# 根据seeds计算vector
seed_vec = [1 if node in seeds else 0 for node in nodes]
for node in seeds:
    for succ in G.successors(node):
            seed_vec[succ] += 0.5
# 转换为tensor
seed_vec = torch.tensor(seed_vec).to(dtype=torch.float32).to(device)
# 预测
pred = model(seed_vec)
print('pred:', pred)
end_time_mlp = time.time()
print('MLP Time:', end_time_mlp - start_time_mlp)
start_time_influence = time.time()
# 计算影响力
IS = ic(G, seeds, 1000)
print('IS:', IS)
end_time_influence = time.time()
print('Influence Time:', end_time_influence - start_time_influence)
