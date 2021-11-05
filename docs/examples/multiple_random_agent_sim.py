import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import grpc
import libtmux
import itertools
import time
import socket
from contextlib import closing
from concurrent import futures
import logging, logging.config
import yaml
import os
import robo_gym_server_modules.server_manager.client as sm_client


target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'
cmd = "source /home/pierre/PycharmProjects/robo_gym_robot_servers_ws/devel/setup.bash; roslaunch wifibot_robot_server sim_wifibot_server_minimal.launch"

sm_client = sm_client.Client(target_machine_ip)
robot_server_ip_1 = sm_client.start_new_server(cmd=cmd, gui=True)
robot_server_ip_2 = sm_client.start_new_server(cmd=cmd, gui=True)
