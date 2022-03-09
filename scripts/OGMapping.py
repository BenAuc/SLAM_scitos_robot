#!/usr/bin/env python3

"""
Template IAS0060 home assignment 4 Project 1 (SCITOS).
Node which handles odometry and laserdata, updates
the occ_map class and publishes the OccupanyGrid message.
And a map class that handles the probabilistic map up-
dates.

@author: Christian Meurer
@date: February 2022

Update: complete of assignment 4
Team: Scitos group 3
Team members: Benoit Auclair; Michael Bryan
Date: March 9, 2022
"""

import numpy as np
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from coordinate_transformations import world_to_grid
from bresenham import bresenham

class OGMap:
    """
    Map class which translates the laser ranges into grid cell
    occupancies and updates the OccupancyGrid variable
    @input: map metadata (height, width, resolution, origin)
    @input: sensor model (reading probability, below reading probability, tau)
    @input: laser ranges and laser metadata (min / max angles,
            angle increments, min / max ranges)
    @output: updated occupancy grid map as 2D np.array()
    """
    def __init__(self, height, width, resolution, map_origin, tau, r_prob, below_r_prob):
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
        ### get map metadata ###
        self.height = height
        self.width = width
        self.resolution = resolution
        self.map_origin = map_origin

        ### static content of Occupancy Grid message
        ### map metadata for msg ###
        self.map_meta_data = MapMetaData()
        self.map_meta_data.resolution = resolution # size of a cell
        self.map_meta_data.width = int(width/resolution) # [cells]
        self.map_meta_data.height = int(height/resolution) # [cells]
        self.map_meta_data.origin = Pose()
        self.map_meta_data.origin.position = Point()
        self.map_meta_data.origin.position.x, self.map_meta_data.origin.position.y = map_origin

        ### declaration of Occupancy Grid message
        self.grid = OccupancyGrid()
        self.grid.info = self.map_meta_data
        self.grid.header = Header()
        self.grid.header.frame_id = "map"
        self.grid.data = None

        ### define probabilities for Bayesian belief update ###
        ### get sensor model ###
        self.tau = tau
        self.r_prob = r_prob
        self.below_r_prob = below_r_prob

        ### define logood variables ###
        self.odds_r_prob = np.log(self.r_prob / (1 - self.r_prob))
        self.odds_below_r_prob = np.log(self.below_r_prob / (1 - self.below_r_prob))

        ### initialize occupancy and logodd grid variables ###
        self.prob_map = -1 * np.ones([int(self.height / self.resolution), int(self.width / self.resolution)])
        self.logodds_map = np.zeros([int(self.height / self.resolution), int(self.width / self.resolution)])


    def updatemap(self,laser_scan,angle_min,angle_max,angle_increment,range_min,range_max,robot_pose, yaw):
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

        ### transform robot pose into grid coordinates ###
        robot_pos_grid = world_to_grid(robot_pose[0], robot_pose[1],
                                        self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

        ### for all rays in the laser scan do: ###
        for idx_range, measured_range in enumerate(laser_scan):

            # discard the measured range if outside of allowed range
            if measured_range < range_min or measured_range > range_max:
                continue

            ### calculate coordinates of object the laser ray hit in
            # compute yaw angle of laser beam
            theta = yaw + angle_min + idx_range * angle_increment

            # the line joining the robot to the object is resolved in two sections
            # the first section are non-occupied cells
            # the second section are occupied cells

            # section 1
            # calculate how far away from the robot the object is
            # taking into account the uncertainty tau around the reading
            delta_x = (measured_range - self.tau/2) * np.cos(theta)
            delta_y = (measured_range - self.tau/2) * np.sin(theta)

            # convert world coordinates of (target - tau) into grid coordinates
            target_minus_tau = world_to_grid(robot_pose[0] + delta_x, robot_pose[1] + delta_y,
                                                 self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

            # section 2
            # calculate how far away from the robot the object is
            # taking into account the uncertainty tau around the reading
            delta_x = (measured_range + self.tau/2) * np.cos(theta)
            delta_y = (measured_range + self.tau/2) * np.sin(theta)

            # convert world coordinates of (target + tau) into grid coordinates
            target_plus_tau = world_to_grid(robot_pose[0] + delta_x, robot_pose[1] + delta_y,
                                                 self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

            ### define a line from laser ray point to robot pose
            if target_minus_tau and target_plus_tau:

                # resolve location of non-occupied cells in grid coordinates
                non_occupied_cells = bresenham(robot_pos_grid[0], robot_pos_grid[1], target_minus_tau[0], target_minus_tau[1])
                # get rid of the first cell in the list because it is also listed as an occupied cell
                non_occupied_cells.pop(-1)

                # resolve location of occupied cells in grid coordinates
                occupied_cells = bresenham(target_minus_tau[0], target_minus_tau[1], target_plus_tau[0], target_plus_tau[1])

                # process list of non-occupied cells
                for cell in non_occupied_cells:
                    x = cell[0]
                    y = cell[1]
                    self.cellUpdate(x, y, self.odds_below_r_prob)

                # process list of occupied cells
                for cell in occupied_cells:
                    x = cell[0]
                    y = cell[1]
                    self.cellUpdate(x, y, self.odds_r_prob)

    def cellUpdate(self, x, y, logodds_update):
        """updates a specific cell in the occupancy grid following an observation
            @param: x, y - indices of the cell in the occupancy grid
            @param: logodds_update - likelihood of the observation in logodds representation
            @result: updated occupancy grid maps (in logodds and probability representations)
        """
        # if the cell had not been observed so far we set the prior on it to 0.5
        if self.prob_map[y][x] == -1:
            self.prob_map[y][x] = 0.5

        # update the logodds representation
        self.logodds_map[y][x] += logodds_update

        # update the probability representation
        self.prob_map[y][x] = 1 - 1 / (1 + np.exp(self.logodds_map[y][x]))

    def returnMap(self):
        """returns latest map as OccupancyGrid object
        """
        # scale the occupancy between 0 and 100 for every cell that's been seen at least one
        scaled_prob = self.prob_map.copy()
        scaled_prob[scaled_prob > 0] *= 100

        # integrate the updated map to the occupancy message
        self.grid.data = scaled_prob.flatten().astype(np.int8)
        return self.grid


class OGMapping:
    """
    Main node which handles odometry and laserdata, updates
    the occ_map class and publishes the OccupanyGrid message.
    @input: odometry as nav_msgs Odometry message
    @input: laser data as sensor_msgs LaserScan message
    @output: updated occupancy grid map as
             nav_msgs OccupancyGrid message
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
        self.pose_sub = rospy.Subscriber("/ground_truth", Odometry, self.odometryCallback)
        self.laserScan_sub = rospy.Subscriber("/laser_scan", LaserScan, self.laserScanCallback)
        
        ### publishers ###
        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=1) # queue_size=1 => only the newest map available

        ### get map parameters ###
        self.width = rospy.get_param("/map/width")
        self.height = rospy.get_param("/map/height")
        self.resolution = rospy.get_param("/map/resolution")
        self.map_origin = rospy.get_param("/map/origin")
        
        ### fetch laser frame ###
        self.laserScaner_to_robotbase = rospy.get_param("/robot_parameters/laserscanner_pose")

        ### get sensor model ###
        self.tau = np.array(rospy.get_param("sensor_model/tau"))
        self.r_prob = np.array(rospy.get_param("sensor_model/r_prob"))
        self.below_r_prob = np.array(rospy.get_param("sensor_model/below_r_prob"))

        ### initialize occupancy grid map class ###
        self.occ_grid_map = OGMap(self.height, self.width, self.resolution, self.map_origin,
                                  self.tau, self.r_prob, self.below_r_prob)

        ### initialization of class variables ###
        self.robot_pose = None
        self.laserscanner_pose = None
        self.scan_msg = None
        self.odom_msg = None

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function for map update
        """
        while not rospy.is_shutdown():
            ### step only when odometry and laser data are available ###
            if self.scan_msg and self.odom_msg:
                self.step()
            self.rate.sleep()

    def step(self):
        """
        Perform an iteration of the mapping loop
        @param: self
        @result: updates the map information and publishes new map data
        """
        ### step only when odometry and laser data are available ###
        if self.scan_msg and self.odom_msg:
            # publish current occupancy map
            self.map_pub.publish(self.occ_grid_map.returnMap())

            # update map only if odometry data available
            if self.robot_pose:
                self.occ_grid_map.updatemap(self.scan_msg.ranges, self.scan_msg.angle_min,
                                            self.scan_msg.angle_max, self.scan_msg.angle_increment,
                                            self.scan_msg.range_min, self.scan_msg.range_max,
                                            self.laserscanner_pose, self.robot_yaw)

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

        # shift the robot pose to the laser frame
        self.laserscanner_pose = [self.robot_pose[0] + np.cos(self.robot_yaw)*self.laserScaner_to_robotbase[0],
                                  self.robot_pose[1] + np.sin(self.robot_yaw)*self.laserScaner_to_robotbase[0]]

    def laserScanCallback(self, data):
        """
        Handles incoming Laserscan messages and updates the map
        @param: information from the laser scanner stored in the
                LaserScan message
        @result: internal update of the map using the occ_grid_map class
        """
        self.scan_msg = data

if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("OGMapping")
    # go to class that provides all the functionality
    # and check for errors
    try:
        mapping = OGMapping(10)
        mapping.run()
    except rospy.ROSInterruptException:
        pass
