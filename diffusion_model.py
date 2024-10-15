import random
import time
import numpy as np

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
            new_activated = set()
            for node in activated:
                neighbors = graph.successors(node)
                for neighbor in neighbors:
                    if neighbor not in activated:
                        if random.random() < graph.get_edge_data(node, neighbor)['weight']:
                            new_activated.add(neighbor)
            # activated去掉原先已经激活的节点，加上新激活的节点
            activated = new_activated
            spread += len(new_activated)

        total_spread += spread

    return total_spread / num_simulations

# 如果是主函数
if __name__ == '__main__':
    #%% 打开一个csv文件
    import numpy as np
    # 使用numpy.genfromtxt读取文件
    data = np.genfromtxt('data/jazz.csv', delimiter=',')
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

    nodes = list(G.nodes())

    #%%
    num_simulations = 1000
    seeds = random.sample(nodes, 20)
    start_time = time.time()
    print(ic(G, seeds, num_simulations))
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


