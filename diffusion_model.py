import random
import time
import numpy as np
import networkx as nx
import pandas as pd
import pickle

#%% 用蒙特卡洛模拟计算影响力
def ic(graph, seeds, num_simulations):
    total_spread = 0

    for _ in range(num_simulations):
        """
                模拟一次信息传播过程
                seeds: 种子节点列表
                """
        activated = set(seeds)
        spread = 0

        while activated:
            # 使用列表推导结合条件判断简化新激活节点的查找逻辑
            # new_activated = {neighbor for node in activated for neighbor in graph.successors(node)
            #                  if neighbor not in activated and random.random() < graph.get_edge_data(node, neighbor)[
            #                      'weight']}
            new_activated = set()
            for node in activated:
                neighbors = graph.successors(node)
                for neighbor in neighbors:
                    if neighbor not in activated:
                        if random.random() < graph.get_edge_data(node, neighbor)['weight']:
                            new_activated.add(neighbor)
            # 使用update方法直接更新激活节点集合，更高效
            activated = new_activated
            print('activated:', activated, '\n')
            spread += len(new_activated)

        total_spread += spread

    return total_spread / num_simulations

# 如果是主函数
if __name__ == '__main__':
    # 读取digg.pkl图文件

    with open('data/digg/digg.pkl', 'rb') as f:
        G = pickle.load(f)

    nodes = list(G.nodes())

    # Monte_Carlo
    num_simulations = 1
    seeds = random.sample(nodes, 30)
    start_time = time.time()
    print(ic(G, seeds, num_simulations))
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


