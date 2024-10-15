import random
from diffusion_model import ic
import torch
from gnn import MLP

def celf(G, k, num_simulation):
    seeds = []
    # 计算每个节点的边际收益
    marginal_gains = {node: ic(G, [node], num_simulation) for node in G}
    for _ in range(k):
        max_marginal_gain = -1
        best_node = None
        for node in marginal_gains:
            if node not in seeds:
                gain = ic(G, seeds + [node], num_simulation) - ic(G, seeds, num_simulation)
                if gain > max_marginal_gain:
                    max_marginal_gain = gain
                    best_node = node
        seeds.append(best_node)
        # 更新边际收益（因为节点的加入可能会改变其他节点的边际收益）
        for node in marginal_gains:
            if node not in seeds:
                gain = ic(G, seeds + [node], num_simulation) - ic(G, seeds, num_simulation)
                marginal_gains[node] = gain
    return seeds, ic(G, seeds, num_simulation)