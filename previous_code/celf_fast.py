import networkx as nx
import random
import heapq
import time
from tqdm import tqdm

class IMBaselinesFast:
    def __init__(self, graph_path):
        self.G = nx.DiGraph()
        self.load_graph(graph_path)
        self.nodes = list(self.G.nodes())
        print(f"Graph Loaded: {self.G.number_of_nodes()} nodes.")

    def load_graph(self, path):
        with open(path, 'r') as f:
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

    # ==========================================
    # 核心优化：预生成 Monte-Carlo 快照
    # ==========================================
    def generate_snapshots(self, mc=100):
        """
        预先生成 mc 个具体的图实例。
        后续计算 Spread 变为在这些图上的确定性可达性分析。
        """
        print(f"Generating {mc} snapshots for acceleration...")
        self.snapshots = []
        for _ in tqdm(range(mc)):
            # 创建一个仅包含激活边的轻量级图（用邻接表表示）
            # 格式: {u: [v1, v2, ...]}
            snapshot = {node: [] for node in self.nodes}
            for u, v, data in self.G.edges(data=True):
                if random.random() < data['weight']:
                    snapshot[u].append(v)
            self.snapshots.append(snapshot)
        return self.snapshots

    def compute_spread_on_snapshots(self, S):
        """
        在预生成的快照上计算平均覆盖范围。
        使用 BFS。
        """
        total_reach = 0
        for snapshot in self.snapshots:
            # 标准 BFS
            visited = set(S)
            queue = list(S)
            count = 0
            while queue:
                u = queue.pop(0)
                count += 1
                # 在当前快照中找邻居
                if u in snapshot:
                    for v in snapshot[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
            total_reach += count
        return total_reach / len(self.snapshots)

    # ==========================================
    # CELF (适配 Snapshot 版)
    # ==========================================
    def celf(self, k, mc=100):
        # 1. 先生成快照 (这是提速的关键！)
        if not hasattr(self, 'snapshots'):
            self.generate_snapshots(mc)
            
        print(f"Running CELF (Snapshot-based, k={k})...")
        
        # --- Initialization ---
        marg_gain_heap = []
        print("  - Initialization phase...")
        for node in tqdm(self.nodes):
            # 针对单个节点的 Spread，不用全图BFS，只跑连通分量即可，这里简化调用统一接口
            spread = self.compute_spread_on_snapshots([node])
            heapq.heappush(marg_gain_heap, (-spread, node))
            
        S = []
        spread_S = 0
        
        # 记录该节点上次更新是在 S 长度为多少时
        last_update_idx = {node: -1 for node in self.nodes}
        
        print("  - Selection phase...")
        pbar = tqdm(total=k)
        
        while len(S) < k:
            neg_gain, best_node = heapq.heappop(marg_gain_heap)
            gain = -neg_gain
            
            # Lazy check
            if last_update_idx[best_node] == len(S):
                S.append(best_node)
                spread_S += gain
                pbar.update(1)
                continue
            
            # Re-compute using snapshots
            # New Spread - Old Spread
            new_spread = self.compute_spread_on_snapshots(S + [best_node])
            new_gain = new_spread - spread_S
            
            last_update_idx[best_node] = len(S)
            heapq.heappush(marg_gain_heap, (-new_gain, best_node))
            
        pbar.close()
        return S

# 使用示例
if __name__ == "__main__":
    im = IMBaselinesFast("data/NetHEPT.txt")
    # 注意：Snapshot 方式下，mc=100 意味着生成 100 张静态图。
    # 之后所有的计算都在这 100 张图上查表，不用再掷骰子了，速度极快。
    seeds = im.celf(k=50, mc=1000) # 可以适当减小mc，因为快照比较占内存
    print(seeds)
    print(f"CELF Spread: {im.run_IC(seeds, mc=1000):.4f}")