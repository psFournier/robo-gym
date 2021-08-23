import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from wifibotEnv.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry

from gym.utils import seeding

class GazeboWifibotEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboWifibot_v0.launch")
        self.vel_pub = rospy.Publisher('/taz01/velocity_controller/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(3) # Forward,Left and Right
        self._seed()
    
    def discretize_observation(self,data):
		discretized_ranges = []
		min_range = 450 # Minimum Range value for collision detection
		done = False
		bridge=CvBridge()
		cv_image = bridge.imgmsg_to_cv2(data) # Transform depth/image_raw into depth map
		
		# We discretize the observation (raw depth image as input) to keep 10 depth values 
		for i in xrange(10):
			a = (680/10)*i-1
			discretized_ranges.append(int(cv_image[240,a]))
				
		for i in discretized_ranges: # For each depth values, we test if it is lower than the threshold (min_range)
			if (min_range > i > 0):
				done = True		# If so, robot crashed, episode is done
		return discretized_ranges,done
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.03
            vel_cmd.angular.z = 0.9
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.03
            vel_cmd.angular.z = -0.9
            self.vel_pub.publish(vel_cmd)

		#read depth camera data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/realsense/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
		
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data)
        
        # We reset the env. if crash is detected
        if not done:
            if action == 0:	# Forward actions take five times more reward than turns, this
                reward = 0.9  # will make the robot take more forward actions as they give
						    # more reward. We want to take as many forward actions as
						    # possible so that the robot goes forward in straight tracks,
						    # will lead to a faster and more realistic behaviour.
            
            else:		    # Left and right actions are rewarded with 1, as they are needed		     
                reward = -0.003  # to avoid crashes too. Setting them higher would result in a
						    # zigzagging behaviour.

        else:				# Crashes (distance to an obstacle reached limits) earn
            reward = -99 	# negative rewards for obvious reasons,
							# we want to avoid obstacles.
			
        return state, reward, done, {}
    
    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        
        #read depth camera data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/realsense/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state = self.discretize_observation(data)
        
        return state
