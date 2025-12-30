#定义一个MLP
import torch
import torch.nn as nn

#%% 定义一个MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        # self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)