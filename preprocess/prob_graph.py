#%% 读取data/digg/digg_friends.csv的数据
import pandas as pd
import random
import numpy as np

# 使用pandas读取CSV文件
df = pd.read_csv('data/digg/digg_friends.csv', header=None)

#%% 转成np.array格式，仅保留后两列
data = df.values[:, 2:]

#%% 创建图
import networkx as nx
# 创建有向图
G = nx.DiGraph()
# 添加节点
for row in data:
    G.add_edge(row[0], row[1])

#%% 为图的每条边添加传播概率
def add_probabilities_to_graph(graph):
    """
    为图的每条边添加传播概率。
    :param graph: 一个有向图对象。
    """
    num_edges = graph.number_of_edges()
    # init_pro = [random.uniform(0, 0.15) for _ in range(num_edges)]
    init_pro = [(random.random() ** 2) * 0.15 for _ in range(num_edges)]
    edge_index = 0
    for u, v in graph.edges():
        graph[u][v]['weight'] = init_pro[edge_index]
        edge_index += 1

#%% 为图的每条边添加传播概率
add_probabilities_to_graph(G)

#%% 存储图
import pickle
with open('data/digg/digg.pkl', 'wb') as f:
    pickle.dump(G, f)
