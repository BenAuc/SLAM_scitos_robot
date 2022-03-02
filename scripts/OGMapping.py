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
# from Bresenham import *
# from transformations import *
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from coordinate_transformations import world_to_grid
from coordinate_transformations import grid_to_world
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

        ### define probabilities for Bayesian belief update ###
        ### get sensor model ###
        self.tau = tau
        self.r_prob = r_prob
        self.below_r_prob = below_r_prob

        ### define logood variables ###
        self.odds_r_prob = self.r_prob / (1 - self.r_prob)
        self.odds_below_r_prob = self.below_r_prob / (1 - self.below_r_prob)

        ### initialize occupancy and logodd grid variables ###
        # 50% chances of each cell being occupied amounts to log(0.5) = 0 hence initialization to 0
        self.grid_map = np.zeros([width, height])

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

        ### for all rays in the laser scan do: ###
        for idx_range, measured_range in enumerate(laser_scan):

            # discard the measured range if outside of allowed range
            if measured_range < range_min or measured_range > range_max:
                continue

            ### calculate coordinate of object the laser ray hit in
            # grid coordinates ###

            # compute yaw angle of laser beam
            theta = yaw + angle_min + idx_range * angle_increment

            # convert measured range into world coordinates
            delta_x = measured_range * np.cos(theta)
            delta_y = measured_range * np.sin(theta)

            # convert world coordinates of target into grid coordinates
            target_pos_grid = world_to_grid(robot_pose[0] + delta_x, robot_pose[1] + delta_y,
                                                 self.map_origin[0], self.map_origin[1], self.width, self.height, self.resolution)

            ### define a line from laser ray point to robot pose
            # in grid coordinates(for example with bresenham) ###
            cells_to_update = bresenham(robot_pos_grid[0], robot_pos_grid[1], target_pos_grid[0], target_pos_grid[1])

            ### update logoods array for indices of points along the laser line with either
            # free or occupied probabilities. ###
            first = True
            need_sorting_out = True

            # we start with the coordinates of the target itself and process the inverted list
            cells_not_occupied = cells_to_update.copy()

            for cell in reversed(cells_to_update):

                # if the cell is the coordinates of the target itself
                if first:
                    first = False
                    # define a box of size tau in world coordinates where the target might be
                    target_box = []
                    nb_steps = self.tau // (2 * self.resolution)

                    # define coordinates of each cell in this box
                    # moving along x axis
                    for step_x in range(0, nb_steps + 1):
                        new_cell = (cell[0] + step_x, cell[1])
                        target_box.append(new_cell)

                        # moving along y axis
                        for step_y in range(1, nb_steps + 1):
                            new_cell_plus_step = (new_cell[0], new_cell[1] + step_y)
                            target_box.append(new_cell_plus_step)
                            new_cell_minus_step = (new_cell[0], new_cell[1] - step_y)
                            target_box.append(new_cell_minus_step)

                        if step_x != 0:
                            new_cell = (cell[0] - step_x, cell[1])
                            target_box.append(new_cell)

                            # moving along y axis
                            for step_y in range(1, nb_steps + 1):
                                new_cell_plus_step = (new_cell[0], new_cell[1] + step_y)
                                target_box.append(new_cell_plus_step)
                                new_cell_minus_step = (new_cell[0], new_cell[1] - step_y)
                                target_box.append(new_cell_minus_step)

                else:
                    if need_sorting_out:
                        for item in enumerate(target_box):

                            # if the cell is in the box it is deemed occupied
                            # update occupancy grid with the observation for this cell
                            if cell == item:

                                # update consists of adding to the prior info on this cell (i.e., current logodds value)
                                # the logodds for an occupied cell
                                self.grid_map[cell[0]][cell[1]] += self.odds_r_prob

                                # remove updated cell from the box
                                target_box.remove(item)
                                continue

                            else:
                                # update consists of adding the prior information on this cell (current value in the grid)
                                # with the logodds for a non-occupied cell
                                self.grid_map[cell[0]][cell[1]] += self.odds_below_r_prob

                                # all other items in the list of cells to be updated are deemed outside the box
                                need_sorting_out = False

                    # if the cell is not in the box it is deemed not occupied
                    # update occupancy grid with the observation for this cell
                    else:

                        # update consists of adding the prior information on this cell (current value in the grid)
                        # with the logodds for a non-occupied cell
                        self.grid_map[cell[0]][cell[1]] += self.odds_below_r_prob

            # process remaining cells in the box
            for item in enumerate(target_box):

                # any cell in the box is deemed occupied
                # update occupancy grid with the observation for this cell
                if cell == item:
                    # update consists of adding to the prior info on this cell (i.e., current logodds value)
                    # the logodds for an occupied cell
                    self.grid_map[cell[0]][cell[1]] += self.odds_r_prob


    def returnMap(self):
        """returns latest map as OccupancyGrid object?
        """
        # transform logodds into probabilities (see formula given by the prof on moodle)
        pass
    

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
        self.pose_sub = rospy.Subscriber("/ground_truth", Odometry, callback= self.odometryCallback)
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
                self.step()
            self.rate.sleep()

    def step(self):
        """
        Perform an iteration of the mapping loop
        @param: self
        @result: updates the map information and publishes new map data
        """
        # self.occ_grid_map.updatemap(..., self.robot_pose)
        # --> self.map_pub.publish(self.occ_grid_map.returnMap()) # uncomment here
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
        theta = euler_from_quaternion([data.pose.pose.orientation.x,
                                       data.pose.pose.orientation.y,
                                       data.pose.pose.orientation.z,
                                       data.pose.pose.orientation.w])[2]
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

        # needs to extract the data from
        
        self.occ_grid_map.updatemap(data, data.angle_min, data.angle_max, data.angle_increment, data.range_min, data.range_max, self.robot_pose)
        
        pass


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
