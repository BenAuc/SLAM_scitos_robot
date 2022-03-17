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

class KalmanFilter:
    """
    Class called by the main node
    """
    def __init__(self):
        """
        class initialization
        @param: self
        @param: height - map size along y-axis [m]
        @param: width - map size along x-axis [m]
        @param: resolution - size of a grid cell [m]
        @param: map_origin - origin in real world [m, m]
        @param: reading probability
        @param: below reading probability
        @param: tau - depth of the reading point
        @result: initializes occupancy grid variable and
                 logg odds variable based on sensor model
        """
      
        # ### declaration of Occupancy Grid message
        # self.grid = OccupancyGrid()
        # self.grid.info = self.map_meta_data
        # self.grid.header = Header()
        # self.grid.header.frame_id = "map"
        # self.grid.data = None

    def update(self):
        """
        Function that updates the occupancy grid based on the laser scan ranges.
        The logodds formulation of the Bayesian belief update is used
        @param: laser_scan - range data from the laser range finder
        @param: angle_min, angle_max - boundaries of the circular arc of the laser ranges
        @param: angle_increment - angular step between consecutive laser rays
        @param: range_min, range_max - min and max distances at which the laser range finder can detect an obstacle
        @param: robot_pose - the planar robot pose in world coordinates
        @result: updates the list of occupancy cells based on occupancy probabilities ranging from [0,100]
        """


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
        self.kalman_filter = KalmanFilter()

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
