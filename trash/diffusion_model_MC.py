import numpy as np
import random
import time

class InfluenceModel:
    def __init__(self, graph, p):
        """
        graph: 邻接列表表示的图
        p: 每条边传播概率的字典
        """
        self.graph = graph
        self.p = p

    def simulate(self, seeds):
        """
        模拟一次信息传播过程
        seeds: 种子节点列表
        """
        activated = set(seeds)
        spread = 0

        while activated:
            new_activated = set()
            for node in activated:
                neighbors = self.graph[node]
                for neighbor in neighbors:
                    if neighbor not in activated:
                        if (node, neighbor) in self.p and random.random() < self.p[(node, neighbor)]:
                            new_activated.add(neighbor)
            # activated去掉原先已经激活的节点，加上新激活的节点
            activated = new_activated
            spread += len(new_activated)

        return spread

def influence_spread(graph, p, seeds, num_simulations):
    """
    计算种子节点集合的影响力
    graph: 邻接列表表示的图
    p: 边的传播概率字典
    seeds: 种子节点集合
    num_simulations: 蒙特卡洛模拟的次数
    """
    model = InfluenceModel(graph, p)
    total_spread = 0

    for _ in range(num_simulations):
        total_spread += model.simulate(seeds)

    return total_spread / num_simulations

# 示例图和传播概率
graph = {
    1: {2, 3},
    2: {1, 4},
    3: {1, 4},
    4: {2, 3}
}

# 传播概率，这里简化为每个节点对每个邻居的传播概率相同
p = {
    (1, 2): 0.1,
    (1, 3): 0.2,
    (2, 1): 0.1,
    (2, 4): 0.3,
    (3, 1): 0.2,
    (3, 4): 0.4,
    (4, 2): 0.3,
    (4, 3): 0.4
}

# 种子节点集合
seeds = [1]

# 蒙特卡洛模拟次数
num_simulations = 1000

# 计算影响力
start_time = time.time()
spread = influence_spread(graph, p, seeds, num_simulations)
print(f"Estimated influence spread: {spread}")
end_time = time.time()
print(f"Time: {end_time - start_time}")