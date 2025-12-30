#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')

#%% 检查data前两列有多少个不同的数值
nodes = list(set(data[:,:2].flatten()))
print(len(nodes))

#%%
data_name_list = ['jazz', 'coraml', 'networkscience']
node_num_list = [198, 1098, 1565]
edge_num_list = [2742, 7981, 13532]

# 从data中随机选取198个节点，并找到与这198个节点相关的2742条边形成新的边数据
for i, data_name in enumerate(data_name_list):
    nodes = np.random.choice(np.unique(data[:,:2].flatten()), node_num_list[i], replace=False)
    # print(nodes)
    new_data = []
    for edge in data:
        if edge[0] in nodes or edge[1] in nodes:
            new_data.append(edge)
        if len(new_data) >= edge_num_list[i]:
            break
    new_data = np.array(new_data)
    print('new_data:', new_data.shape)
    np.savetxt('data/'+data_name+'.csv', new_data, delimiter=',', fmt='%d,%d,%f')

#%%

for data_name in data_name_list:
    data = np.genfromtxt('data/'+data_name+'.csv', delimiter=',')
    # 取第三列为weights
    weights = data[:,2]
    # 将前两列变为int
    data = data[:,:2].astype(int)
    #对data中的序号进行重新编码
    # 先找到所有的节点
    nodes = list(set(data.flatten()))
    # 重新编码
    new_nodes = {node: i for i, node in enumerate(nodes)}
    # 重新编码data
    data = np.array([[new_nodes[node] for node in edge] for edge in data])
    # 打印data中的最大值，最小值和唯一值的数量
    print(data_name, data.max(), data.min(), len(np.unique(data)))