from baseline_model import celf
from diffusion_model import ic
from gnn import MLP
import torch

# 选择数据集
#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
# data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
data_list = ['jazz', 'coraml', 'networkscience', 'musae_PTBR_edges_new']
# for dataname in data_list:
#     data_name = dataname
data_name = 'musae_PTBR_edges_new'
data = np.genfromtxt('data/'+data_name+'.csv', delimiter=',')
# 取第三列为weights
weights = data[:,2]
# 将前两列变为int
data = data[:,:2].astype(int)

#%% 对data中的序号进行重新编码
# 先找到所有的节点
nodes = list(set(data.flatten()))
# 重新编码
new_nodes = {node: i for i, node in enumerate(nodes)}
# 重新编码data
data = np.array([[new_nodes[node] for node in edge] for edge in data])

#%%
# 从data和weights创建字典
edges = []
for i, edge in enumerate(data):
    edges.append((edge[0], edge[1], weights[i]))

#%% 做成Graph对象
import networkx as nx
G = nx.DiGraph()
G.add_weighted_edges_from(edges)

#%% 确认节点和边集合，1912个节点，31299条边
nodes = list(G.nodes)
edges = list(G.edges)

num_simulations = 1

#%%
# 选择种子节点并汇报运行时间
import random
import time
print(data_name)
k_list = [10, 20, 50, 100]
for k in k_list:
    print('k:', k)
    start_time = time.time()
    seeds, is_celf = celf(G, k, num_simulations)
    end_time = time.time()
    print('CELF:')
    print('time:', end_time-start_time)
    print('seeds:', seeds)
    print('is_celf:', is_celf)