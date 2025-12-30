import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random
import time
import torch
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
import torch.nn as nn
import torch.optim as optim
from gnn import MLP
from diffusion_model import influence_count
class DeepQNetwork(nn.Module):
    def __init__(self, n_actions=1, n_features=100, n_l1=20, n_l2=20,
                 learning_rate=0.001, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=10, memory_size=500, batch_size=32,
                 e_greedy_increment=None, top_k=5, grl=None, weight_IS=1):
        super(DeepQNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.n_l1 = n_l1
        self.n_l2 = n_l2

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 4 + 1))

        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

        self.top_k = top_k
        self.grl = grl
        self.weight_IS = weight_IS

        self.cost_his = []

    def _build_net(self):
        return nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_l1),
            nn.ReLU(),
            nn.Linear(self.n_l1, self.n_l2),
            nn.ReLU(),
            nn.Linear(self.n_l2, self.n_actions)
        )

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, action_list, action_embedding, env_embedding, **kwargs):
        env_embedding_batch = np.repeat(env_embedding[np.newaxis, :], action_embedding.shape[0], axis=0)
        # print("action_embedding.size():", action_embedding.size())
        # print("env_embedding_batch.size():", env_embedding_batch.size())
        observation = np.concatenate([env_embedding_batch, action_embedding], axis=1)
        observation_tensor = torch.FloatTensor(observation)

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation_tensor).detach().numpy().flatten()
            top_k_indices = np.argsort(actions_value)[-self.top_k:]
            top_k_actions = [action_list[i] for i in top_k_indices]
            is_increments = np.array([self.grl(new_seed=top_k_action, old_seeds=kwargs['old_seeds'], model=kwargs['model'], model_node=kwargs['model_node'], G=kwargs['G']).detach().numpy() for top_k_action in top_k_actions]).squeeze()
            weighted_values = actions_value[top_k_indices] + self.weight_IS * np.array(is_increments)
            action_ind = np.argmax(weighted_values)
            action = top_k_actions[action_ind]
        else:
            action = random.choice(action_list)
        # print(action)
        return action

    def choose_action_no_ran(self, action_list, action_embedding, env_embedding, **kwargs):
        env_embedding_batch = np.repeat(env_embedding[np.newaxis, :], action_embedding.shape[0], axis=0)
        # print(action_embedding.size())
        # print(env_embedding_batch.size())
        observation = np.concatenate([env_embedding_batch, action_embedding], axis=1)
        observation_tensor = torch.FloatTensor(observation)
        actions_value = self.eval_net(observation_tensor).detach().numpy().flatten()
        top_k_indices = np.argsort(actions_value)[-self.top_k:]
        top_k_actions = [action_list[i] for i in top_k_indices]
        is_increments = np.array([self.grl(new_seed=top_k_action, old_seeds=kwargs['old_seeds'], model=kwargs['model'], model_node=kwargs['model_node'], G=kwargs['G']).detach().numpy() for top_k_action in top_k_actions]).squeeze()
        weighted_values = actions_value[top_k_indices] + self.weight_IS * np.array(is_increments)
        action_ind = np.argmax(weighted_values)
        action = top_k_actions[action_ind]
        return action

    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)

        batch_memory = self.memory[sample_index, :]
        s_ = torch.FloatTensor(batch_memory[:, -2 * self.n_features:])  # next states
        s = torch.FloatTensor(batch_memory[:, :2 * self.n_features])  # current states

        q_next = self.target_net(s_).detach().numpy()
        q_eval = self.eval_net(s).detach().numpy()

        q_target = q_eval.copy()
        reward = batch_memory[:, 2 * self.n_features][:, np.newaxis]
        q_target = reward + self.gamma * q_next

        # Train eval network
        q_target_tensor = torch.FloatTensor(q_target)
        loss = nn.MSELoss()(self.eval_net(s), q_target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def get_cost_history(self):
        return self.cost_his

    def save_model(self, save_path):
        torch.save(self.eval_net.state_dict(), save_path)

    def load_model(self, load_path):
        self.eval_net.load_state_dict(torch.load(load_path))


def seed_embedding(model_node, seeds, G):
    if seeds is None or len(seeds) == 0:
        return torch.zeros(1, 64).to(device)
    seed_vec_list = []
    for node in seeds:
        seed_vec = np.zeros(64)
        seed_vec += model_node.wv[str(node)]
        for succ in G.successors(node):
                seed_vec += 0.5 * model_node.wv[str(succ)]
        seed_vec_list.append(seed_vec)
    seeds_list = torch.tensor(seed_vec_list).to(dtype=torch.float32).to(device)
    return seeds_list

def grl_influence_count(model, model_node, seeds, G):
    seeds_list = seed_embedding(model_node, seeds, G)
    # print(seeds_list.size())
    pred = model(seeds_list)
    pred = torch.sum(pred, dim=0)
    return pred

def grl_incrementIS(model, model_node, old_seeds, new_seed, G):
    new_seeds = old_seeds.add(new_seed)
    return grl_influence_count(model, model_node, new_seeds, G) - grl_influence_count(model, model_node, old_seeds, G)

def train(RL, G, model,cmodel_node, nodes, edges,threshold = 0.9,
          batchsize=32, n_l1=10, n_l2=20, episode=100, seed_number_training=4, word2vec_embedding_dim=20, n=10):
    print('========training========')
    # all_nodes = set(node2vec_embedding.keys()) - seed_initial
    for i in range(0, episode):
        seed_set = set()
        seed_set_embedding = torch.zeros(word2vec_embedding_dim)
        all_nodes = set(nodes)
        for t in range(1, seed_number_training + 1):
            candidate_seed = list(all_nodes - seed_set)
            candidate_seed_embedding = np.array([model_node.wv[str(node)] for node in candidate_seed])
            candidate_seed_embedding = torch.from_numpy(candidate_seed_embedding).to(device)

            selected_seed = RL.choose_action(candidate_seed, candidate_seed_embedding, seed_set_embedding, old_seeds=seed_set, model=model, model_node=model_node, G=G)
            
            reward = grl_influence_count(model, model_node, seed_set, G)
            # print("seed_set_embedding.size():", seed_set_embedding.size())
            seed_set_embedding_ = seed_set_embedding + seed_embedding(model_node, [selected_seed], G).squeeze(0)
            # print("seed_set_embedding_.size():", seed_set_embedding_.size())
            next_seed_set = seed_set | {selected_seed}
            next_candidate_seed = list(all_nodes - next_seed_set)
            next_candidate_seed_embedding = np.array([model_node.wv[str(node)] for node in next_candidate_seed])
            next_candidate_seed_embedding = torch.from_numpy(next_candidate_seed_embedding).to(device)
            next_selected_seed = RL.choose_action_no_ran(next_candidate_seed,
                                                         next_candidate_seed_embedding, seed_set_embedding_, old_seeds=next_seed_set, model=model, model_node=model_node, G=G)
            RL.store_transition(seed_set_embedding.detach().numpy(), model_node.wv[str(selected_seed)], reward.squeeze(0).detach().numpy(),
                                np.hstack((seed_set_embedding_.detach().numpy(), model_node.wv[str(next_selected_seed)])))
            seed_set.add(selected_seed)
            seed_set_embedding = seed_set_embedding + seed_embedding(model_node, [selected_seed], G).squeeze(0)
            if t % 50 == 0:
                print('Episode %d/%d, step %d/%d' % (i + 1, episode, t, seed_number_training))
                print(' Selected_seed: ', selected_seed, ' reward: ', reward, ' influence count: ',
                      influence_count(nodes, edges, seed_set, threshold))
            if t % n == 0:
                RL.learn()
        model_filename = f'train_rl/episode_{i}_ic.ckpt'
        RL.save_model(model_filename)
    return RL


def test(G, model, model_node, nodes, RL, k,
         n_subarea=10, word2vec_embedding_dim=10):
    seed_set = set()
    seed_set_embedding = seed_embedding(model_node, seed_set, G)
    all_nodes = set(nodes)
    while len(seed_set) < k:
        candidate_seed = list(all_nodes - seed_set)
        candidate_seed_embedding = torch.tensor([model_node.wv[str(node)] for node in candidate_seed]).to(device)
        selected_seed = RL.choose_action(candidate_seed, candidate_seed_embedding, seed_set_embedding, old_seeds=seed_set, model=model, model_node=model_node, G=G)
        seed_set.add(selected_seed)
        seed_set_embedding = seed_set_embedding + seed_embedding(model_node, [selected_seed], G)
    return seed_set

word2vec_embedding_dim = 64
batchsize = 128
n_l1 = 50
n_l2 = 20
episode = 50
seed_number_training = 50
n = 10 # 每10步学习一次

RL = DeepQNetwork(n_actions=1, n_features=word2vec_embedding_dim,
                  n_l1=n_l1,
                  n_l2=n_l2,
                  batch_size=batchsize,
                  grl = grl_incrementIS)

#%% 打开一个csv文件
import numpy as np
# 使用numpy.genfromtxt读取文件
data = np.genfromtxt('data/musae_PTBR_edges_new.csv', delimiter=',')
# data = np.genfromtxt('data/twitch_gamers/small_twitch_edges.csv', delimiter=',')
# 取第三列为weights
weights = data[:,2]
# 将前两列变为int
data = data[:,:2].astype(int)

#%% 做成Graph对象
import networkx as nx
G = nx.DiGraph()
G.add_edges_from(data)

#%% 确认节点和边集合，1912个节点，31299条边
nodes = list(G.nodes)
edges = list(G.edges)
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
# model_node = node2vec.fit(window=10, min_count=1, batch_words=4)
# # 存储模型参数
# model_node.save('param/node2vec.model')
# 读取模型参数
model_node = Word2Vec.load('param/node2vec.model')

model = MLP(input_dim=64, output_dim=1, num_layers=3, hidden_dim=128).to(device)
model_params = torch.load('model-v3.pth')
# 将参数加载到模型中
model.load_state_dict(model_params)
model.eval()

RL = train(RL, G, model, model_node, nodes, edges,
           batchsize=batchsize, n_l1=n_l1, n_l2=n_l2, episode=episode, seed_number_training=seed_number_training, n=n,
           word2vec_embedding_dim=word2vec_embedding_dim)

