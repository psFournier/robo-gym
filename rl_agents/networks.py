import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class QNetwork(nn.Module):
	def __init__(self, env):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(
			np.array(env.observation_space.shape).prod()+np.prod(env.action_space.shape), 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x, a, device):
		x = torch.Tensor(x).to(device)
		x = torch.cat([x, a], 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Actor(nn.Module):
	def __init__(self, env):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))

	def forward(self, x, device):
		x = torch.Tensor(x).to(device)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return torch.tanh(self.fc_mu(x))
