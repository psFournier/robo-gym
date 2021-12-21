from gym.envs.registration import register
from rl_agents import ddpg
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
import pybullet_envs
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from rl_agents import utils

register(
	id='ReacherPyBulletEnv-v1',
	entry_point='pybulletgym.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
	reward_threshold=1000.,
)

env = gym.make('ReacherPyBulletEnv-v1')

parser = argparse.ArgumentParser(description='DDPG agent')
# Common arguments
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
					help='the name of this experiment')
parser.add_argument('--learning-rate', type=float, default=3e-4,
					help='the learning rate of the optimizer')
parser.add_argument('--seed', type=int, default=1,
					help='seed of the experiment')
parser.add_argument('--episode-length', type=int, default=100,
					help='the maximum length of each episode')
parser.add_argument('--total-timesteps', type=int, default=100000,
					help='total timesteps of the experiments')
parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
					help='if toggled, `torch.backends.cudnn.deterministic=False`')
parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
					help='if toggled, cuda will not be enabled by default')
parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
					help='run the script in production mode and use wandb to log outputs')
parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
					help='weather to capture videos of the agent performances (check out `videos` folder)')
parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
					help="the wandb's project name")
parser.add_argument('--wandb-entity', type=str, default=None,
					help="the entity (team) of wandb's project")

# Algorithm specific arguments
parser.add_argument('--buffer-size', type=int, default=int(2e4),
					help='the replay memory buffer size')
parser.add_argument('--gamma', type=float, default=0.99,
					help='the discount factor gamma')
parser.add_argument('--tau', type=float, default=0.005,
					help="target smoothing coefficient (default: 0.005)")
parser.add_argument('--max-grad-norm', type=float, default=0.5,
					help='the maximum norm for the gradient clipping')
parser.add_argument('--batch-size', type=int, default=128,
					help="the batch size of sample from the reply memory")
parser.add_argument('--exploration-noise', type=float, default=0.1,
					help='the scale of exploration noise')
parser.add_argument('--learning-starts', type=int, default=10e3,
					help="timestep to start learning")
parser.add_argument('--policy-frequency', type=int, default=2,
					help="the frequency of training policy (delayed)")
parser.add_argument('--noise-clip', type=float, default=0.5,
					help='noise clip parameter of the Target Policy Smoothing Regularization')

args = parser.parse_args()
if not args.seed:
	args.seed = int(time.time())

experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
	'\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))


torch.backends.cudnn.deterministic = args.torch_deterministic

utils.seed(env, args.seed)

if args.capture_video:
	env = Monitor(env, f'videos/{experiment_name}')

ddpg.run(
	env,
	args,
	writer
)