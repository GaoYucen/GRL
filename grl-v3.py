#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
# data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
data = np.genfromtxt('data/twitch_gamers/small_twitch_edges.csv', delimiter=',')
# 取第三列为weights
weights = data[:,2]
# 将前两列变为int
data = data[:,:2].astype(int)

#%% 做成Graph对象
import networkx as nx
G = nx.DiGraph()
G.add_edges_from(data)

#%% 确认节点和边集合，1912个节点，31299条边
nodes = list(G.nodes)
edges = list(G.edges)

#%% 使用Node2Vec算法获取点特征
from node2vec import Node2Vec
# 加载Word2Vec库
from gensim.models import Word2Vec
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
# model_node = node2vec.fit(window=10, min_count=1, batch_words=4)
# # 存储模型参数
# model_node.save('param/node2vec.model')
# 读取模型参数
model_node = Word2Vec.load('param/node2vec.model')
# # 获取节点特征
# node_features = np.array([model_node.wv[str(node)] for node in nodes])
# # 存储节点特征
# np.save('data/node_features.npy', node_features)

#%%
import torch
from gnn import MLP
from tqdm import tqdm
# 选择种子节点
import random
# 计算影响力
from diffusion_model import influence_count
threshold = 0.9

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = MLP(input_dim=64, output_dim=1, num_layers=3, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.L1Loss()

num_epoches = 100
batch_size = 64

# #%% 训练模型
# for epoch in tqdm(range(num_epoches)):
#     seeds_list = []
#     IS_list = []
#     for i in range(0, batch_size):
#         seeds = random.sample(nodes, 20)
#         seed_vec_list = []
#         # 设置一个64维的array
#         for node in seeds:
#             seed_vec = np.zeros(64)
#             seed_vec += model_node.wv[str(node)]
#             for succ in G.successors(node):
#                     seed_vec += 0.5 * model_node.wv[str(succ)]
#             seed_vec_list.append(seed_vec)
#         IS = influence_count(nodes, edges, seeds, threshold)
#         seeds_list.append(seed_vec_list)
#         IS_list.append(IS)
#     optimizer.zero_grad()
#     seeds_list = torch.tensor(seeds_list).to(dtype=torch.float32).to(device)
#     pred = model(seeds_list)
#     # print('pred:', pred)
#     # 将[64, 20, 1]的pred转换为[64, 1]
#     pred = torch.sum(pred, dim=1)
#     # print(pred.size())
#     IS_list = torch.tensor(IS_list).to(dtype=torch.float32).to(device)
#     loss = criterion(pred, IS_list)
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print('epoch:', epoch)
#         # print('pred:', pred)
#         # print('IS_list:', IS_list)
#         print('loss:', loss)
#         # 将loss存到log
#         with open('log/log-v3.txt', 'a') as f:
#             f.write('epoch:' + str(epoch) + '\n')
#             f.write('loss:' + str(loss) + '\n')
# #存储模型数据
# torch.save(model.state_dict(), 'model-v3.pth')

#%% 测试模型
import time
# 加载保存的状态字典
model_params = torch.load('model-v3.pth')
# 将参数加载到模型中
model.load_state_dict(model_params)
model.eval()
seeds = random.sample(nodes, 50)
start_time_mlp = time.time()
seed_vec_list = []
# 设置一个64维的array
for node in seeds:
    seed_vec = np.zeros(64)
    seed_vec += model_node.wv[str(node)]
    for succ in G.successors(node):
            seed_vec += 0.5 * model_node.wv[str(succ)]
    seed_vec_list.append(seed_vec)
seeds_list = torch.tensor(seed_vec_list).to(dtype=torch.float32).to(device)
pred = model(seeds_list)
# 将[64, 20, 1]的pred转换为[64, 1]
pred = torch.sum(pred, dim=0)
print('pred:', pred)
end_time_mlp = time.time()
print('mlp time:', end_time_mlp - start_time_mlp)
start_time_ic = time.time()
IS = influence_count(nodes, edges, seeds, threshold)
print('IS:', IS)
end_time_ic = time.time()
print('ic time:', end_time_ic - start_time_ic)
