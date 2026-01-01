import networkx as nx
import random
import heapq
import time
import numpy as np
from tqdm import tqdm
from numba import njit

# ==========================================
# Part 1: Numba 加速内核 (用于快速蒙特卡洛评估)
# ==========================================
@njit(parallel=False)
def fast_ic_simulation(indptr, indices, data, seeds, mc, num_nodes):
    """
    Numba 加速的 IC 传播模拟。
    比纯 Python 快 10-100 倍，用于最终评估 Spread。
    """
    total_spread = 0.0
    
    for _ in range(mc):
        visited = np.zeros(num_nodes, dtype=np.bool_)
        queue = np.empty(num_nodes, dtype=np.int32)
        q_start = 0
        q_end = 0
        
        # 初始种子
        for s in seeds:
            if not visited[s]:
                visited[s] = True
                queue[q_end] = s
                q_end += 1
        
        current_spread = q_end
        
        # BFS
        while q_start < q_end:
            u = queue[q_start]
            q_start += 1
            
            start_idx = indptr[u]
            end_idx = indptr[u+1]
            
            for i in range(start_idx, end_idx):
                v = indices[i]
                if visited[v]:
                    continue
                
                prob = data[i]
                if np.random.random() < prob:
                    visited[v] = True
                    queue[q_end] = v
                    q_end += 1
                    current_spread += 1
                    
        total_spread += current_spread
        
    return total_spread / mc

# ==========================================
# Part 2: 主类实现
# ==========================================
class IMBaselines:
    def __init__(self, graph_path):
        """
        初始化：加载图并准备 Numba 数据结构
        """
        print(f"Loading graph from {graph_path}...")
        self.G = nx.DiGraph()
        self.load_graph(graph_path)
        
        self.nodes = list(self.G.nodes())
        self.num_nodes = max(self.nodes) + 1
        self.num_edges = self.G.number_of_edges()
        print(f"Graph Loaded: {len(self.nodes)} nodes, {self.num_edges} edges.")

        # --- 准备 Numba 需要的 CSR 格式 ---
        print("Preparing Numba CSR structures...")
        # 确保节点 ID 覆盖 0 到 max_id
        adj = nx.to_scipy_sparse_array(self.G, nodelist=range(self.num_nodes), format='csr', weight='weight')
        self.indptr = adj.indptr.astype(np.int32)
        self.indices = adj.indices.astype(np.int32)
        self.data = adj.data.astype(np.float64)
        
        # 预编译 Numba 函数
        print("JIT Compiling...")
        fast_ic_simulation(self.indptr, self.indices, self.data, np.array([0], dtype=np.int32), 1, self.num_nodes)

    def load_graph(self, path):
        with open(path, 'r') as f:
            # 尝试跳过第一行 (N M)
            first_line = f.readline().split()
            if len(first_line) == 2:
                pass # N M header
            else:
                f.seek(0) # 如果不是 header，回退

            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
                    self.G.add_edge(u, v, weight=p)

    def evaluate(self, seeds, mc=2000):
        """
        高精度评估函数 (使用 Numba)
        论文画图时，所有方法选出的 seeds 都要扔到这里跑一遍
        """
        seeds_arr = np.array(list(seeds), dtype=np.int32)
        return fast_ic_simulation(self.indptr, self.indices, self.data, seeds_arr, mc, self.num_nodes)

    # ==========================
    # Alg 1: High Degree
    # ==========================
    def high_degree(self, k):
        print(f"[Running High Degree] k={k}")
        # 按出度排序
        sorted_nodes = sorted(self.G.out_degree, key=lambda x: x[1], reverse=True)
        return [n for n, d in sorted_nodes[:k]]

    # ==========================
    # Alg 2: Degree Discount (IC Model Optimized)
    # ==========================
    def degree_discount_ic(self, k, p=0.01):
        print(f"[Running Degree Discount] k={k}")
        d = dict(self.G.out_degree())
        dd = d.copy()
        t = {n: 0 for n in self.G.nodes()} 
        S = []
        
        # 如果数据中有真实概率，最好计算一下平均概率作为 p
        # p = np.mean(self.data) 
        
        for _ in tqdm(range(k)):
            u = max(dd, key=dd.get)
            S.append(u)
            dd.pop(u)
            
            for v in self.G.neighbors(u):
                if v in dd:
                    t[v] += 1
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v]) * t[v] * p
        return S

    # ==========================
    # Alg 3: CELF (Snapshot Based - Fast & Accurate)
    # ==========================
    def celf(self, k, mc=100):
        """
        基于快照的 CELF。
        mc: 生成多少个快照用于贪心选择 (通常 50-100 足够区分优劣)
        """
        print(f"[Running CELF] k={k}, snapshots={mc}")
        
        # 1. 生成快照 (预先掷骰子)
        print("  - Generating Snapshots...")
        snapshots = []
        for _ in range(mc):
            # 仅保留激活边的邻接表
            snapshot = {u: [] for u in self.nodes}
            for u, v, data in self.G.edges(data=True):
                if random.random() < data['weight']:
                    snapshot[u].append(v)
            snapshots.append(snapshot)
            
        # 辅助函数：在快照上计算 Spread
        def compute_spread_fast(S_set):
            total_reach = 0
            for snap in snapshots:
                # BFS on snapshot
                visited = set(S_set)
                q = list(S_set)
                count = 0
                while q:
                    u = q.pop(0)
                    count += 1 # Count reachable
                    # 快照里的边都是已激活的
                    for v in snap[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                total_reach += count
            return total_reach / mc

        # 2. CELF Initialization
        print("  - Initialization Phase...")
        marg_gain_heap = []
        for node in tqdm(self.nodes):
            spread = compute_spread_fast([node])
            heapq.heappush(marg_gain_heap, (-spread, node))
            
        S = []
        spread_S = 0
        last_update_idx = {node: -1 for node in self.nodes}
        
        # 3. Selection Phase
        print("  - Selection Phase...")
        pbar = tqdm(total=k)
        while len(S) < k:
            neg_gain, best_node = heapq.heappop(marg_gain_heap)
            gain = -neg_gain
            
            if last_update_idx[best_node] == len(S):
                S.append(best_node)
                spread_S += gain
                pbar.update(1)
                continue
            
            # Re-compute marginal gain
            current_spread = compute_spread_fast(S + [best_node])
            new_gain = current_spread - spread_S
            
            last_update_idx[best_node] = len(S)
            heapq.heappush(marg_gain_heap, (-new_gain, best_node))
            
        pbar.close()
        return S

    # ==========================
    # Alg 4: IMM (Simplified)
    # ==========================
    def imm(self, k, theta=50000):
        """
        使用反向可达集 (RR Sets)
        theta: 采样数量 (NetHEPT 建议 50k-100k)
        """
        print(f"[Running IMM] k={k}, RR_Sets={theta}")
        R = []
        G_rev = self.G.reverse()
        nodes = list(self.G.nodes())
        
        # 1. Generate RR Sets
        for _ in tqdm(range(theta)):
            v = random.choice(nodes)
            rr_set = {v}
            q = [v]
            while q:
                curr = q.pop(0)
                for neighbor in G_rev.neighbors(curr):
                    if neighbor not in rr_set:
                        weight = G_rev[curr][neighbor]['weight']
                        if random.random() < weight:
                            rr_set.add(neighbor)
                            q.append(neighbor)
            R.append(rr_set)
            
        # 2. Greedy Selection
        S = []
        node_rr_count = {}
        # 建立倒排索引: node -> list of rr_set_ids
        for idx, rr in enumerate(R):
            for node in rr:
                if node not in node_rr_count:
                    node_rr_count[node] = set()
                node_rr_count[node].add(idx)
        
        covered_indices = set()
        for _ in range(k):
            best_node = -1
            max_gain = -1
            
            # 寻找当前能覆盖最多"未覆盖RR Set"的节点
            for node, indices in node_rr_count.items():
                if node in S: continue
                gain = len(indices - covered_indices)
                if gain > max_gain:
                    max_gain = gain
                    best_node = node
            
            if best_node != -1:
                S.append(best_node)
                covered_indices.update(node_rr_count[best_node])
                
        return S

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # 配置
    data_path = "data/NetHEPT.txt" # 请修改为您的路径
    k_seeds = 10
    eval_mc = 2000 # 最终画图用的评估次数，越大越准
    
    # 初始化
    im = IMBaselines(data_path)
    
    results = {}
    
    # 1. High Degree
    start = time.time()
    s_deg = im.high_degree(k_seeds)
    results['HighDegree'] = (s_deg, time.time() - start)
    
    # 2. Degree Discount
    start = time.time()
    s_dd = im.degree_discount_ic(k_seeds, p=0.01) # NetHEPT 也可以不传p，算法内未动态计算p，建议p设为平均权重
    results['DegreeDiscount'] = (s_dd, time.time() - start)

    # 3. IMM
    start = time.time()
    s_imm = im.imm(k_seeds, theta=50000)
    results['IMM'] = (s_imm, time.time() - start)
    
    # 4. CELF (Snapshot)
    start = time.time()
    # mc=50 表示生成50张图来做贪心选择，足以区分优劣
    s_celf = im.celf(k_seeds, mc=100) 
    results['CELF'] = (s_celf, time.time() - start)
    
    print("\n" + "="*50)
    print(f"FINAL EVALUATION (MC={eval_mc} with Numba)")
    print("="*50)
    print(f"{'Method':<20} | {'Time(s)':<10} | {'Spread':<10}")
    print("-" * 46)
    
    for name, (seeds, runtime) in results.items():
        # 统一使用 Numba 内核进行公正评估
        spread = im.evaluate(seeds, mc=eval_mc)
        print(f"{name:<20} | {runtime:<10.4f} | {spread:<10.4f}")