#!/usr/bin/env python3

"""
Template IAS0060 home assignment 4 Project 1 (SCITOS).
Node which handles odometry and laserdata, updates
the occ_map class and publishes the OccupanyGrid message.
And a map class that handles the probabilistic map up-
dates.

@author: Christian Meurer
@date: February 2022

Update: complete of assignment 6
Team: Scitos group 3
Team members: Benoit Auclair; Michael Bryan
Date: March 17, 2022
"""

import numpy as np
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import OccupancyGrid
# from nav_msgs.msg import MapMetaData
# from coordinate_transformations import world_to_grid
# from bresenham import bresenham

class NoiseModel:
    """
    Class called by the main node
    """

    def __init__(self):
        """
        Function that ...
        @param: TBD
        @result: TBD
        """
        #TO DO: pick value for each parameter alpha
        self.alpha1 = 0
        self.alpha2 = 0
        self.alpha3 = 0
        self.alpha4 = 0
        self.error = np.array([self.alpha1, self.alpha2, self.alpha3, self.alpha4])

class MotionModel:
    """
    Class implementing the motion model for the robot
    """

    def __init__(self,dt,initial_pose):
        """
        Function that ...
        @param: TBD
        @result: TBD
        """
        ### class arguments
        # time step
        self.dt = dt
        self.last_pose = initial_pose
        self.next_pose = np.zeros((3, 1))

        self.noise_model = NoiseModel()
        self.error_model = self.noise_model.error
        #NOTE: let's start debugging with an error equals to null
        self.error = 0

    def updatePose(self, control_input):
        """
        This method updates the predicted robot pose.
        @param: control_input - numpy array of dim 1 x 2 containing:
            *linear speed w.r.t. x-axis in robot frame
            *angular speed w.r.t. z-axis in robot frame
        @result: returns: predicted pose, estimated error
        """
        self.last_pose = self.next_pose
        v = control_input[0]
        w = control_input[1]

        increment = np.array([[v * self.dt * np.cos(self.last_pose[2] + w * self.dt / 2)],
                             [v * self.dt * np.sin(self.last_pose[2] + w * self.dt / 2)],
                             [w * self.dt]])
        self.next_pose = self.last_pose + increment

        #NOTE: let's start debugging without any error
        # self.updateErrorModel(v, w)

        return self.next_pose, self.error_model

    def updateErrorModel(self, v, w):
        """
        This method updates the predicted robot pose.
        @param: control_input - numpy array of dim 1 x 2 containing:
            *linear speed w.r.t. x-axis in robot frame
            *angular speed w.r.t. z-axis in robot frame
        @result: update of error model
        """

        # Taylor expansion of non-linear noise model
        self.error = np.array([[2 * self.error_model[0] * v + 2 * self.error_model[1] * w, 0],
                                [0, 2 * self.error_model[2] * v + 2 * self.error_model[3] * w]])


class KalmanFilter:
    """
    Class called by the main node
    """
    def __init__(self, dt):
        """
        Function that ...
        @param: TBD
        @result: TBD
        """
        ### class arguments
        # TO DO: needs to be initialized with the first reading of the pose
        self.pose = None
        self.dt = dt
        self.motion_model = MotionModel(self.dt, self.pose)
        self.odom_error_model = self.motion_model.error_model

        # TO DO: needs to be initialized with a value TBD
        self.last_state = None
        self.last_covariance = None

        self.next_state = None
        self.next_covariance = None

        self.jacobian_G = None
        self.jacobian_V = None

        # ### declaration of Occupancy Grid message
        # self.grid = OccupancyGrid()
        # self.grid.info = self.map_meta_data
        # self.grid.header = Header()
        # self.grid.header.frame_id = "map"
        # self.grid.data = None

    def predict(self, control_input):
        """
        This method predicts what the next system state will be.
        @param: control_input - numpy array of dim 1 x 2 containing:
            *linear speed w.r.t. x-axis in robot frame
            *angular speed w.r.t. z-axis in robot frame
        @result:
            *next_state - numpy array of dim 3 x 1 containing the 3 tracked variables
            *next_covariance - numpy array of dim 3 x 3 containing covariance matrix
        """
        self.next_state, error_model = self.motion_model.updatePose(control_input)

        self.computeJacobian(control_input)
        self.next_covariance = self.jacobian_G @ self.last_covariance @ self.jacobian_G.T + \
                               self.jacobian_V @ error_model @ self.jacobian_V.T


    def computeJacobian(self, control_input):

        pass



class Localization:
    """
    Main node which handles odometry and laserdata, updates
    the Kalman Filter class
    @input: odometry as nav_msgs Odometry message
    @output: pose as geometry_msgs PoseStamped message
    """
    def __init__(self, dt):
        """
        class initialization
        @param: self
        @param: rate - updating frequency for this node in [Hz]
        @result: get static parameters from parameter server
                 to initialize the controller, and to
                 set up publishers and subscribers
        """
        ### timing ###
        self.dt = dt
        self.rate = rospy.Rate(20)

        ### subscribers ###
        self.ground_truth_sub = rospy.Subscriber("/ground_truth", Odometry, self.groundTruthCallback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odometryCallback)
        
        ### publishers ###
        self.pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=1) # queue_size=1 => only the newest map available

        ### get map parameters ###
        # self.width = rospy.get_param("/map/width")
        # self.height = rospy.get_param("/map/height")
        # self.resolution = rospy.get_param("/map/resolution")
        # self.map_origin = rospy.get_param("/map/origin")
        
        ### initialize KF class ###
        self.kalman_filter = KalmanFilter(self.dt)

        ### initialization of class variables ###
        self.robot_pose = None
        self.odom_msg = None
        self.ground_truth_msg = None

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function for the predicton and update step.
        """
        while not rospy.is_shutdown():
            ### step only when odometry are available ###
            if self.odom_msg:
                self.step()
            self.rate.sleep()

    def step(self):
        """
        Perform an iteration of the localiyation loop
        @param: self
        @result: updates 
        """
        pass

    def odometryCallback(self, data):
        """
        Handles incoming Odometry messages and performs a
        partial quaternion to euler angle transformation to get the yaw angle theta
        @param: pose data stored in the odometry message
        @result: global variable pose_2D containing the planar
                 coordinates robot_x, robot_y and the yaw angle theta
        """
        self.odom_msg = data
        # extract yaw angle of robot pose using the transformation on the odometry message
        self.robot_yaw = euler_from_quaternion([data.pose.pose.orientation.x,
                                                data.pose.pose.orientation.y,
                                                data.pose.pose.orientation.z,
                                                data.pose.pose.orientation.w],
                                               axes='szyx')[0]
        # extract robot pose
        self.robot_pose = [data.pose.pose.position.x, data.pose.pose.position.y]

        # # shift the robot pose to the laser frame
        # self.laserscanner_pose = [self.robot_pose[0] + np.cos(self.robot_yaw)*self.laserScaner_to_robotbase[0],
        #                           self.robot_pose[1] + np.sin(self.robot_yaw)*self.laserScaner_to_robotbase[0]]

    def groundTruthCallback(self, data):
        """
        Handles incoming groud truth messages
        @param: information from Gazebo
        @result: internal update of ground truth
        """
        self.ground_truth_msg = data

if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("OGMapping")
    # go to class that provides all the functionality
    # and check for errors
    try:
        localization = Localization(10)
        localization.run()
    except rospy.ROSInterruptException:
        pass
