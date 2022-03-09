#!/usr/bin/env python3

"""
Template IAS0060 home assignment 4 Project 1 (SCITOS).
Node which handles odometry and laserdata, updates
the occ_map class and publishes the OccupanyGrid message.
And a map class that handles the probabilistic map up-
dates.

@author: Christian Meurer
@date: February 2022
"""

import math
import numpy as np
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from coordinate_transformations import world_to_grid, grid_to_world
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
        # http://docs.ros.org/en/lunar/api/nav_msgs/html/msg/MapMetaData.html
        self.map_meta_data = MapMetaData()
        self.map_meta_data.resolution = resolution # size of a cell
        self.map_meta_data.width = int(width/resolution) # [cells]
        self.map_meta_data.height = int(height/resolution) # [cells]
        self.map_meta_data.origin = Pose()
        self.map_meta_data.origin.position = Point()
        self.map_meta_data.origin.position.x, self.map_meta_data.origin.position.y = map_origin
        # self.map_meta_data.origin.orientation = Quaternion()
        # self.map_meta_data.origin.orientation.x, self.map_meta_data.orientation.y, self.map_meta_data.orientation.z, self.map_meta_data.orientation.w = 0
        # print("map origin is : {}".format(self.map_meta_data.origin.position.x))

        ### declaration of Occupancy Grid message
        self.grid = OccupancyGrid()
        self.grid.info = self.map_meta_data
        self.grid.header = Header()
        self.grid.header.frame_id = "map"
        # self.grid.data = -1 * np.ones([int(self.height / self.resolution), int(self.width / self.resolution)]).flatten().astype(np.int8)

#############################################
        # debugging
        # self.grid.data = -1 * np.ones([int(self.height / self.resolution), int(self.width / self.resolution)]).flatten().astype(np.int8)
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
        # grid_map is stored as log-odds
        # 50% chances of each cell being occupied amounts to log(0.5) = 0 hence initialization to 0
        self.prob_map = -1 * np.ones([int(self.height / self.resolution), int(self.width / self.resolution)])
        self.logodds_map = np.zeros([int(self.height / self.resolution), int(self.width / self.resolution)])

        ### initialize auxillary variables ###


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
        ### prior is sensor reading & sensor model
        ### likelihood is the current occupancy map
        ### posterior is the multiplication of the 2 i.e. update of the latter based on the former

        ### transform robot pose into grid coordinates ###
        robot_pos_grid = world_to_grid(robot_pose[0], robot_pose[1],
                                        self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

        print("robot_pos_grid = {}".format(robot_pos_grid))

        ### for all rays in the laser scan do: ###
        for idx_range, measured_range in enumerate(laser_scan):

            print("currently processing range D = {}".format(measured_range))
            # discard the measured range if outside of allowed range
            if measured_range < range_min or measured_range > range_max:
                continue

            ### calculate coordinate of object the laser ray hit in
            # grid coordinates ###

            # compute yaw angle of laser beam
            theta = yaw - angle_min - idx_range * angle_increment
            # I think it should be
            # theta = yaw - angle_min - idx_range*angle_increment
            # if theta is angle in world coor. from x axis towards y axis.

            print("current yaw D = {}".format(yaw))
            print("current theta D = {}".format(theta))
            print("angle_min D = {}".format(angle_min))
            print("angle_increment D = {}".format(angle_increment))

            # convert (measured range - uncertainty per sensor model) into world coordinates
            delta_x = (measured_range - self.tau) * np.cos(theta)
            delta_y = (measured_range - self.tau) * np.sin(theta)

            # convert world coordinates of target into grid coordinates
            target_minus_tau = world_to_grid(robot_pose[0] + delta_x, robot_pose[1] + delta_y,
                                                 self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

            # convert (measured range + uncertainty per sensor model) into world coordinates
            delta_x = (measured_range + self.tau) * np.cos(theta)
            delta_y = (measured_range + self.tau) * np.sin(theta)

            # convert world coordinates of target into grid coordinates
            target_plus_tau = world_to_grid(robot_pose[0] + delta_x, robot_pose[1] + delta_y,
                                                 self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

            ### define a line from laser ray point to robot pose
            # in grid coordinates(for example with bresenham) ###
            non_occupied_cells = bresenham(robot_pos_grid[0], robot_pos_grid[1], target_minus_tau[0], target_minus_tau[1])
            # get rid of the first cell in the list because it is also listed as an occupied cell
            non_occupied_cells.pop(-1)
            print("current set of non-occupied cells = {}".format(non_occupied_cells))
            occupied_cells = bresenham(target_minus_tau[0], target_minus_tau[1], target_plus_tau[0], target_plus_tau[1])
            print("current set of occupied cells = {}".format(occupied_cells))

            # process list of non-occupied cells
            for cell in non_occupied_cells:
                print("current cell = {}".format(cell))
                # first column in the array runs along y-axis
                x = cell[0]
                # last row in the array runs along x-axis
                # y = cell[1]
                y = -1 * cell[1]

                self.cellUpdate(x, y, self.odds_below_r_prob)

            # update occupied cells
            for cell in occupied_cells:
                print("current cell = {}".format(cell))
                # first column in the array runs along y-axis
                x = cell[0]
                # last row in the array runs along x-axis
                y = -1 * cell[1]
                self.cellUpdate(x, y, self.odds_r_prob)

    def cellUpdate(self, x, y, logodds_update):

        # print("x in grid = {}".format(x))
        # print("y in grid = {}".format(y))
        # print("shape of map = {}".format(self.prob_map.shape))

        # if the cell had not been observed so far we set the prior on it to 0.5
        print("current probability is: {}".format(self.prob_map[y][x]))
        if self.prob_map[y][x] == -1:
            self.prob_map[y][x] = 0.5

        # perform the update
        prior_belief = self.logodds_map[y][x]
        posterior_belief = prior_belief + logodds_update
        self.logodds_map[y][x] = posterior_belief

        # return to probabilistic representation
        self.prob_map[y][x] = 1 - 1 / (1 + np.exp(posterior_belief))
        print("current prior_belief logodds: {}".format(prior_belief))
        print("current update: {}".format(logodds_update))
        print("current posterior_belief logodds: {}".format(posterior_belief))
        print("updated probability is: {}".format(self.prob_map[y][x]))

    def returnMap(self):
        """returns latest map as OccupancyGrid object
        """
        # transform logodds into probabilities [0,1] (see formula given by the prof on moodle)
        # NOTE: could potentially be made computationally less costly by only updating specific cells
        # probability = lambda x: 1 - 1 /(1 + np.e**x) # may produce Warning: overflow encountered in power, it's OK to ignore
        # self.grid.data = (probability(self.prob_map)*100).flatten().astype(np.int8)

        #############################################
        # debugging
        # scale between 0 and 100 only the cells that have been visited
        print("##########################################################################################")
        scaled_prob = self.prob_map.copy()
        scaled_prob[scaled_prob > 0] *= 100
        print("Here's the map : {}".format(scaled_prob))
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
        self.rate = rospy.Rate(10)

        ### subscribers ###
        self.pose_sub = rospy.Subscriber("/ground_truth", Odometry, self.odometryCallback)
        self.laserScan_sub = rospy.Subscriber("/laser_scan", LaserScan, self.laserScanCallback)
        
        ### publishers ###
        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=1) # queue_size=1 => only the newest map available

        ### define messages to be handled ###
        self.scan_msg = None
        self.odom_msg = None

        ### get map parameters ###
        self.width = rospy.get_param("/map/width")
        self.height = rospy.get_param("/map/height")
        self.resolution = rospy.get_param("/map/resolution")
        self.map_origin = rospy.get_param("/map/origin")

        ### get sensor model ###
        self.tau = np.array(rospy.get_param("sensor_model/tau"))
        self.r_prob = np.array(rospy.get_param("sensor_model/r_prob"))
        self.below_r_prob = np.array(rospy.get_param("sensor_model/below_r_prob"))

        ### initialize occupancy grid map class ###
        self.occ_grid_map = OGMap(self.height, self.width, self.resolution, self.map_origin,
                                  self.tau, self.r_prob, self.below_r_prob)

        print("here's the map width: {}".format(self.occ_grid_map.width))

        # define static components of occupancy grid to be published
        # --> not sure what this^ means?

        ### initialization of class variables ###
        self.robot_pose = None

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function for map update
        """
        while not rospy.is_shutdown():
            ### step only when odometry and laser data are available ###
            if self.scan_msg and self.odom_msg:
                print("publishing map")
                self.step()

            # sleep for
            # cycle = 5000
            # for step in range(cycle):
            self.rate.sleep()

    def step(self):
        """
        Perform an iteration of the mapping loop
        @param: self
        @result: updates the map information and publishes new map data
        """
        print("##########################################################################################")

        # if self.scan_msg and self.odom_msg:
        #     self.occ_grid_map.updatemap(self.scan_msg.ranges, self.scan_msg.angle_min, self.scan_msg.angle_max,
        #                                 self.scan_msg.angle_increment, self.scan_msg.range_min, self.scan_msg.range_max,
        #                                 self.robot_pose, self.robot_yaw)

        self.map_pub.publish(self.occ_grid_map.returnMap()) # uncomment here


    def odometryCallback(self, data):
        """
        Handles incoming Odometry messages and performs a
        partial quaternion to euler angle transformation to get the yaw angle theta
        @param: pose data stored in the odometry message
        @result: global variable pose_2D containing the planar
                 coordinates robot_x, robot_y and the yaw angle theta
        """
        self.odom_msg = data
        self.robot_yaw = euler_from_quaternion([data.pose.pose.orientation.x,
                                                data.pose.pose.orientation.y,
                                                data.pose.pose.orientation.z,
                                                data.pose.pose.orientation.w],
                                               axes='szyx')[0]
        # print(self.robot_yaw)
        # theta = euler_from_quaternion(np.array(data.pose.pose.orientation))
        self.robot_pose = [data.pose.pose.position.x, data.pose.pose.position.y]
        
        pass

    def laserScanCallback(self, data):
        """
        Handles incoming Laserscan messages and updates the map
        @param: information from the laser scanner stored in the
                LaserScan message
        @result: internal update of the map using the occ_grid_map class
        """
        self.scan_msg = data
        # print(type(data.ranges)) # returned tuple
        if self.robot_pose: # update map only if odometry data available
            self.occ_grid_map.updatemap(data.ranges, data.angle_min, data.angle_max, data.angle_increment, data.range_min, data.range_max, self.robot_pose, self.robot_yaw)



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
