import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

# initialize environment
env = gym.make('NoObstacleNavigationWifibotSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)

# choose and run appropriate algorithm provided by stable-baselines
model = TD3.load("../td3_mir_basic")
for episode in range(10):
	done = False
	state = env.reset()
	while not done:
		action = model.predict(state)[0]
		state, reward, done, info = env.step(action)