import networkx as nx
import random
import heapq
import time
import math
import numpy as np
from tqdm import tqdm

class IMBaselines:
    def __init__(self, graph_path):
        """
        读取 NetHEPT 格式数据: 
        Line 1: N M
        Line 2+: u v weight
        """
        self.G = nx.DiGraph()
        self.load_graph(graph_path)
        print(f"Graph Loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

    def load_graph(self, path):
        with open(path, 'r') as f:
            # 跳过第一行元数据 (N M)
            next(f) 
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    self.G.add_edge(u, v, weight=p)

    def run_IC(self, seeds, mc=1000):
        """
        蒙特卡洛模拟计算 Spread (用于评估最终效果)
        """
        spread = 0
        for _ in range(mc):
            active = set(seeds)
            new_active = set(seeds)
            while new_active:
                next_new_active = set()
                for u in new_active:
                    if self.G.has_node(u):
                        for v in self.G.neighbors(u):
                            if v not in active:
                                # 掷骰子
                                prob = self.G[u][v]['weight']
                                if random.random() < prob:
                                    active.add(v)
                                    next_new_active.add(v)
                new_active = next_new_active
            spread += len(active)
        return spread / mc

    # ==========================
    # 1. High Degree (基于度中心性)
    # ==========================
    def high_degree(self, k):
        print(f"Running High Degree (k={k})...")
        # 按出度排序
        sorted_nodes = sorted(self.G.out_degree, key=lambda x: x[1], reverse=True)
        S = [n for n, d in sorted_nodes[:k]]
        return S

    # ==========================
    # 2. Degree Discount (IC模型下 PMIA 的高效替代/启发式基线)
    # ==========================
    def degree_discount_ic(self, k, p=0.01):
        """
        Chen et al. KDD 2009. 
        假设边权重统一或平均概率 p，如果是不同权重，这只是一个近似启发式。
        """
        print(f"Running Degree Discount (k={k})...")
        d = dict(self.G.out_degree())
        dd = d.copy() # degree discount
        t = {n: 0 for n in self.G.nodes()} # number of selected neighbors
        S = []
        
        for _ in range(k):
            # 选折扣度最大的节点
            u = max(dd, key=dd.get)
            S.append(u)
            dd.pop(u) # 移除已选节点
            
            # 更新邻居的折扣度
            for v in self.G.neighbors(u):
                if v in dd:
                    t[v] += 1
                    # 更新公式: d_v - 2*t_v - (d_v - t_v)*t_v*p (简化版)
                    # 这里使用原始论文的近似逻辑: dd[v] = d[v] - 2*t[v] - (d[v]-t[v])*t[v]*p
                    # 注意：如果您的图权重差异很大，这个Heuristic效果会打折
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v]) * t[v] * p
        return S



    # ==========================
    # 3. CELF (完整实现)
    # ==========================
    def celf(self, k, mc=1000):
        """
        CELF: Cost-Effective Lazy Forward Optimization
        利用 Submodularity 性质加速 Greedy。
        核心思想：如果节点 u 在上一轮的边际收益已经小于节点 v 当前的收益，
        那 u 这一轮肯定不是最优，不用重算 u，直接跳过。
        """
        print(f"Running CELF (k={k}, mc={mc})...")
        
        # --- 第一步：初始化 (计算所有节点的初始边际收益) ---
        # 这是一个耗时操作，必须计算 N 次 MC
        print("  - CELF Initialization (Calculating 1st round marginal gains)...")
        marg_gain_heap = [] # 最大堆 (存负值): (-gain, node)
        
        # 遍历所有节点计算第一轮收益（即空集时的收益）
        for node in tqdm(self.G.nodes()):
            # 这里的 seeds 是 [node]
            spread = self.run_IC([node], mc)
            heapq.heappush(marg_gain_heap, (-spread, node))
            
        S = [] # 选中的种子集合
        current_spread = 0 # 当前种子集的覆盖范围
        
        # 记录每个节点最后一次计算收益是在“选中第几个种子”之后
        # 用于判断堆顶元素的数据是否“过时”
        last_update_idx = {node: -1 for node in self.G.nodes()}
        
        print(f"  - CELF Selection Phase (Target k={k})...")
        
        # 进度条
        pbar = tqdm(total=k)
        
        while len(S) < k:
            # 1. 取出堆顶（目前看起来收益最大的节点）
            neg_gain, best_node = heapq.heappop(marg_gain_heap)
            gain = -neg_gain
            
            # 2. 检查是否满足 Lazy 条件
            # 如果 best_node 的收益是基于当前 S 计算的（即没有过时），那它就是全局最优
            # 因为子模性保证了收益只会递减，其他节点的真实收益不可能比它们在旧状态下的收益还高
            if last_update_idx[best_node] == len(S):
                S.append(best_node)
                current_spread += gain
                pbar.update(1)
                # print(f"    Selected: {best_node}, Gain: {gain:.2f}, Total: {current_spread:.2f}")
                continue
            
            # 3. 如果是过时的数据，需要重新计算该节点的边际收益
            # Marginal Gain = Spread(S + {u}) - Spread(S)
            new_spread = self.run_IC(S + [best_node], mc)
            new_gain = new_spread - current_spread
            
            # 4. 更新该节点的状态，并放回堆中
            last_update_idx[best_node] = len(S)
            heapq.heappush(marg_gain_heap, (-new_gain, best_node))
            
            # 这里不选入 S，而是进入下一次循环，
            # 下一次循环会再次对比由于由堆排序带来的新堆顶
            
        pbar.close()
        return S

    # ==========================
    # 4. IMM (Influence Maximization via Martingales) - Simplified
    # ==========================
    def imm(self, k, epsilon=0.5, l=1):
        """
        简化版 IMM: 省略了复杂的 theta 估算，直接生成固定数量的 RR Sets。
        为了达到理论保证，theta 需要动态计算 (Tang et al. SIGMOD 2015)。
        这里为了代码可读性，我们生成一个较大的固定数量 RR sets。
        """
        n = self.G.number_of_nodes()
        # 估算需要的 RR set 数量 (这里简化处理，实际需要公式)
        # 实际使用中，NetHEPT 这种规模建议至少 100,000 个 RR sets
        theta = int(100000) 
        print(f"Running IMM (k={k}, RR_Sets={theta})...")
        
        R = [] # 存储所有 RR set
        
        # 构建反向图 (用于从 v 反向遍历)
        # 注意：NetworkX 的 reverse() 会复制图，内存消耗大。
        # 对于只读操作，可以直接遍历 predecessors (虽然慢一点) 或预处理 transpose
        G_rev = self.G.reverse() 
        
        # 1. Generate RR Sets
        print("  - Generating RR Sets...")
        nodes = list(self.G.nodes())
        for _ in tqdm(range(theta)):
            v = random.choice(nodes)
            rr_set = set([v])
            q = [v]
            while q:
                curr = q.pop(0)
                # 在反向图中找邻居 = 原图中的入边
                for neighbor in G_rev.neighbors(curr):
                    if neighbor not in rr_set:
                        # 检查边是否激活 (原图 neighbor -> curr)
                        weight = G_rev[curr][neighbor]['weight'] # 注意 reverse 后属性保留
                        if random.random() < weight:
                            rr_set.add(neighbor)
                            q.append(neighbor)
            R.append(rr_set)
            
        # 2. Node Selection (Maximum Coverage Task)
        print("  - Selecting Nodes...")
        S = []
        # 记录每个节点出现在多少个 RR set 中
        node_rr_count = {}
        for idx, rr in enumerate(R):
            for node in rr:
                if node not in node_rr_count:
                    node_rr_count[node] = set()
                node_rr_count[node].add(idx)
        
        # 贪心选择覆盖最多 RR set 的节点
        covered_indices = set()
        for _ in range(k):
            best_node = -1
            max_gain = -1
            
            # 寻找当前覆盖增益最大的节点
            # (优化：这里可以用堆，但为了代码清晰用循环)
            for node, indices in node_rr_count.items():
                if node in S:
                    continue
                gain = len(indices - covered_indices)
                if gain > max_gain:
                    max_gain = gain
                    best_node = node
            
            if best_node != -1:
                S.append(best_node)
                covered_indices.update(node_rr_count[best_node])
                
        return S

# ==========================
# 主程序调用示例
# ==========================
if __name__ == "__main__":
    # 假设你的数据文件路径
    data_path = "data/NetHEPT.txt" 
    
    # 初始化
    im = IMBaselines(data_path)
    
    k = 50
    
    # 1. High Degree
    s_degree = im.high_degree(k)
    print(f"High Degree Seeds: {s_degree[:5]}...")
    
    # 2. Degree Discount (推荐作为 Heuristic Baseline)
    s_dd = im.degree_discount_ic(k)
    print(f"Degree Discount Seeds: {s_dd[:5]}...")
    
    # 3. IMM (推荐作为 Efficiency/Approximation Baseline)
    s_imm = im.imm(k)
    print(f"IMM Seeds: {s_imm[:5]}...")
    
    # 4. CELF (推荐作为 Effectiveness Upper Bound - 跑得慢，慎用)
    s_celf = im.celf(k, mc=100) # 这里mc设小点以便测试
    print(f"CELF Seeds: {s_celf[:5]}...")
    
    # --- 最终评估 ---
    print("\nEvaluating Spread (MC=1000)...")
    print(f"High Degree Spread: {im.run_IC(s_degree, mc=1000):.4f}")
    print(f"Degree Discount Spread: {im.run_IC(s_dd, mc=1000):.4f}")
    print(f"IMM Spread: {im.run_IC(s_imm, mc=1000):.4f}")
    print(f"CELF Spread: {im.run_IC(s_celf, mc=1000):.4f}")