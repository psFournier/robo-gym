import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

# initialize environment
env = gym.make('ObstacleAvoidanceWifibotSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)

# choose and run appropriate algorithm provided by stable-baselines
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3(MlpPolicy, env, verbose=1, action_noise=action_noise)
model.learn(total_timesteps=50000)

# save model
model.save('td3_wifibot_basic')