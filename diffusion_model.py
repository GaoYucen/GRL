import random
import time

def influence_count(nodes, edges, seeds, threshold):
    ''' Calculate influent result
    Args:
        nodes (list) [#node]: nodes list of the graph;
        edges (list of list) [#edge, 2]: edges list of the graph;
        seeds (list) [#seed]: selected seeds;
        threshold (float): influent threshold, between 0 and 1;
    Return:
        final_actived_node (list): list of influent nodes;
    '''
    in_degree = {}
    inactive_nodes = []
    active_nodes = []
    nodes_status = {}

    for edge in edges:
        if edge[0] in seeds:
            active_nodes.append(edge[0])
        else:
            inactive_nodes.append(edge[0])
        if edge[1] in seeds:
            active_nodes.append(edge[1])
        else:
            inactive_nodes.append(edge[1])
        if edge[1] in in_degree:
            in_degree[edge[1]] += 1
        else:
            in_degree[edge[1]] = 1

    active_nodes = list(set(active_nodes))
    inactive_nodes = list(set(inactive_nodes))

    for node in nodes:
        nodes_status[node] = 0
    for node in active_nodes:
        nodes_status[node] = 1

    while (active_nodes):
        new_actived_nodes = []
        for edge in edges:
            if nodes_status[edge[0]] == 1:
                if nodes_status[edge[1]] == 0:
                    p = np.array([1 - threshold / in_degree[edge[1]], threshold / in_degree[edge[1]]])
                    flag = np.random.choice([0, 1], p=p.ravel())
                    if flag:
                        new_actived_nodes.append(edge[1])
        for node in active_nodes:
            nodes_status[node] = 2
        for node in new_actived_nodes:
            nodes_status[node] = 1
        active_nodes = new_actived_nodes

    final_actived_node = 0
    for node in nodes:
        if nodes_status[node] == 2:
            final_actived_node += 1
    return final_actived_node

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

    nodes = list(G.nodes())

    #%%
    num_simulations = 1000
    seeds = random.sample(nodes, 20)
    start_time = time.time()
    print(ic(G, seeds, num_simulations))
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


