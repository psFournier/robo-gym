from gym.envs.registration import register
from rl_agents import ddpg, ddpg_her
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.wrappers import Monitor
import time
from rl_agents import utils
from docopt import docopt
from schema import Schema, Use, Or

help = """

Usage: 
  [scipt].py [options]

Options:
  --exp_name VAL
  --learning_rate VAL  [default: 0.001]
  --seed VAL
  --episode_length VAL  [default: 100]
  --total_timesteps VAL  [default: 200000]
  --torch_deterministic
  --cuda
  --capture_video
  --buffer_size VAL  [default: 500000] 
  --gamma VAL  [default: 0.99]
  --tau VAL  [default: 0.005]
  --batch_size VAL  [default: 128]
  --exploration_noise VAL  [default: 0.1]
  --learning_starts VAL  [default: 10000]
  --policy_frequency VAL  [default: 4]
  
"""

float_params = [
	'--learning_rate',
	'--gamma',
	'--tau',
	'--exploration_noise'
]

int_params = [
	'--seed',
	'--episode_length',
	'--total_timesteps',
	'--buffer_size',
	'--batch_size',
	'--learning_starts',
	'--policy_frequency'
]

args = Schema({
	lambda x: x in float_params: Or(None, Use(float)),
	lambda x: x in int_params: Or(None, Use(int)),
	str: object
}).validate(docopt(help))

register(
	id='ReacherPyBulletEnv-v1',
	entry_point='pybulletgym.envs.roboschool.envs.manipulation.reacher_env:ReacherBulletEnv',
	reward_threshold=1000.,
)

env = gym.make('ReacherPyBulletEnv-v1')

if not args['--seed']:
	args['--seed'] = int(time.time())

experiment_name = f"{args['--exp_name']}__{args['--seed']}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
	'\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))


torch.backends.cudnn.deterministic = args['--torch_deterministic']

utils.seed(env, args['--seed'])

if args['--capture_video']:
	env = Monitor(env, f'videos/{experiment_name}')

ddpg_her.run(
	env,
	args,
	writer
)