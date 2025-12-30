#%%
import argparse
import pickle

parser = argparse.ArgumentParser(description="GenIM")
datasets = ['jazz', 'cora_ml', 'power_grid', 'netscience', 'random5']
parser.add_argument("-d", "--dataset", default="jazz", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [1, 5, 10, 20]
parser.add_argument("-sp", "--seed_rate", default=1, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
mode = ['Normal', 'Budget Constraint']
parser.add_argument("-m", "--mode", default="normal", type=str,
                    help="one of: {}".format(", ".join(sorted(mode))))
args = parser.parse_args(args=[])

#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
data = np.genfromtxt('../data/Influence-Maximization-on-Graph-Data-main/musae_PTBR_edges.csv', delimiter=',', dtype='int', skip_header=1)

#%%
weights = np.random.lognormal(mean=0.3, sigma=1, size=len(data))  # 假设对数分布的均值为0，标准差为1

#%% 放缩到(0,1)
weights = weights/max(weights)

#%% 绘制weights的分布图
import matplotlib.pyplot as plt
plt.hist(weights, bins=100)
plt.show()

#%% 将weights添加到data后
data = np.c_[data, weights]

#%% 保存到文件
np.savetxt('../data/musae_PTBR_edges_new.csv', data, delimiter=',', fmt='%d,%d,%f')

