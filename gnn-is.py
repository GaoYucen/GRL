#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
# data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
data_list = ['jazz', 'coraml', 'networkscience', 'musae_PTBR_edges_new']
for dataname in data_list[0:3]:
    data_name = dataname
    print(data_name)
    data = np.genfromtxt('data/'+data_name+'.csv', delimiter=',')
    # 取第三列为weights
    weights = data[:,2]
    # 将前两列变为int
    data = data[:,:2].astype(int)

    #%% 对data中的序号进行重新编码
    # 先找到所有的节点
    nodes = list(set(data.flatten()))
    # 重新编码
    new_nodes = {node: i for i, node in enumerate(nodes)}
    # 重新编码data
    data = np.array([[new_nodes[node] for node in edge] for edge in data])

    #%%
    # 从data和weights创建字典
    edges = []
    for i, edge in enumerate(data):
        edges.append((edge[0], edge[1], weights[i]))

    #%% 做成Graph对象
    import networkx as nx
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)

    #%% 确认节点和边集合，1912个节点，31299条边
    nodes = list(G.nodes)
    edges = list(G.edges)

    #%%
    import torch
    from gnn import MLP
    from tqdm import tqdm
    # 选择种子节点
    import random
    # 计算影响力
    from diffusion_model import ic

    # 指定mps为device
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    model = MLP(input_dim=len(nodes), output_dim=1, num_layers=3, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    num_epoches = 10000
    batch_size = 256
    num_simulation = 1000

    min_loss= 1000
    early_stop = 0

    #%% 训练模型
    for epoch in tqdm(range(num_epoches)):
        seeds_list = []
        IS_list = []
        for i in range(0, batch_size):
            seeds = random.sample(nodes, 20)
            # 根据seeds计算vector
            seed_vec = [1 if node in seeds else 0 for node in nodes]
            for node in seeds:
                for succ in G.successors(node):
                        seed_vec[succ] += 0.5
            IS = ic(G, seeds, num_simulation)
            seeds_list.append(seed_vec)
            IS_list.append(IS)
        optimizer.zero_grad()
        seeds_list = torch.tensor(seeds_list).to(dtype=torch.float32).to(device)
        pred = model(seeds_list)
        IS_list = torch.tensor(IS_list).to(dtype=torch.float32).unsqueeze(-1).to(device)
        loss = criterion(pred, IS_list)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('epoch:', epoch)
            print('loss:', loss)
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), 'model_'+data_name+'.pth')
            print('save model')
            early_stop = 0
        # 如果连续多轮没有下降，提前结束
        else:
            early_stop += 1
            if early_stop > 30:
                break

    # #%% 测试模型
    # # 选择种子节点
    # import random
    # import time
    # # 加载保存的状态字典
    # model_params = torch.load('model_'+data_name+'.pth')
    # # 将参数加载到模型中
    # model.load_state_dict(model_params)
    #
    # # test超参数
    # seeds_list = []
    #
    # seeds_vec_list = []
    # ic_list = []
    #
    # model.eval()
    #
    # for i in range(0, batch_size):
    #     seeds = random.sample(nodes, 20)
    #     seeds_list.append(seeds)
    #
    # start_time_mlp = time.time()
    # for i in range(0, batch_size):
    #     seeds = seeds_list[i]
    #     # 根据seeds计算vector
    #     seed_vec = [1 if node in seeds else 0 for node in nodes]
    #     for node in seeds:
    #         for succ in G.successors(node):
    #                 seed_vec[succ] += 0.5
    #     seeds_vec_list.append(seed_vec)
    # # 转换为tensor
    # seeds_vec_list = torch.tensor(seeds_vec_list).to(dtype=torch.float32).to(device)
    # # 预测
    # pred_list = model(seeds_vec_list).cpu().detach().numpy()
    # # print('pred:', pred)
    # end_time_mlp = time.time()
    # model_time = end_time_mlp - start_time_mlp
    #
    # start_time_influence = time.time()
    # for i in range(0, batch_size):
    #     seeds = seeds_list[i]
    #     # 计算影响力
    #     IS = ic(G, seeds, num_simulation)
    #     ic_list.append(IS)
    # end_time_influence = time.time()
    # ic_time = end_time_influence - start_time_influence
    #
    # # 对比预测值和真实值，取绝对值的均值
    # pred_list = np.array(pred_list)
    # ic_list = np.array(ic_list)
    # mae = np.mean(np.abs(pred_list - ic_list))
    # # 方差
    # var = np.var(np.abs(pred_list - ic_list))
    # print('mean pred:', np.mean(pred_list))
    # print('mean ic:', np.mean(ic_list))
    # print('MAE:', mae)
    # print('Var:', var)
    #
    # # 对比时间
    # print('MLP Time:', model_time)
    # print('Influence Time:', ic_time)

    #
    # print('seeds_list:', seeds_list)
