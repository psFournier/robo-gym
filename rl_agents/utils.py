import random
import numpy as np
import torch

def seed(env, seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	env.seed(seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
	slope =  (end_sigma - start_sigma) / duration
	return max(slope * t + start_sigma, end_sigma)