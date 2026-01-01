import torch
import os
import numpy as np

# 假设您的模型保存在 param/ 目录下
save_dir = "param"
emb_path = os.path.join(save_dir, "node2vec_NetHEPT.txt.pth")

if os.path.exists(emb_path):
    print(f"Loading embeddings from {emb_path}...")
    # 加载到 CPU
    embeddings = torch.load(emb_path, map_location='cpu')
    
    # 1. 打印基本统计信息
    print(f"Embedding Shape: {embeddings.shape}")
    print(f"Min Value: {embeddings.min().item():.4f}")
    print(f"Max Value: {embeddings.max().item():.4f}")
    print(f"Mean Value: {embeddings.mean().item():.4f}")
    
    # 2. 关键：计算模长 (L2 Norm)
    # 对于 FeatureDQN 的点积操作，这个指标最关键
    norms = torch.norm(embeddings, p=2, dim=1)
    
    print("-" * 30)
    print(f"【关键指标】Embedding L2 Norms:")
    print(f"Min Norm: {norms.min().item():.4f}")
    print(f"Max Norm: {norms.max().item():.4f}")
    print(f"Avg Norm: {norms.mean().item():.4f}")
    print("-" * 30)
    
    if norms.mean() > 5.0:
        print("结论：Embedding 模长过大！")
        print("原因：未经归一化的特征在进行点积运算(dot product)时会导致数值爆炸。")
        print("这就是为什么 FeatureDQN 的 Q 值会飙升到几十亿。")
    else:
        print("结论：Embedding 模长正常。可能需要检查其他原因。")

else:
    print(f"文件 {emb_path} 不存在，请先运行 GNN 训练生成 Embedding。")