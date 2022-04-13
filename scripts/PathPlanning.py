"""
IAS0060 home assignment "Path Planning 2" Project 1 (SCITOS).

@author: Benoit Aulcair
@date: April 2022

Update: complete of task Path Planning 2
Team: Scitos group 3
Team members: Benoit Auclair
Date: April 25, 2022
"""
import numpy
import yaml
from copy import deepcopy
from laser_line_extraction.msg import LineSegmentList
from laser_line_extraction.msg import LineSegment
import numpy as np
from numpy.linalg import norm as norm
from numpy.linalg import det as matrix_det
from numpy.linalg import inv as matrix_inv
from numpy import arctan2 as atan2
import rospy
from rospy import Duration
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from tf import allFramesAsYAML
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, ColorRGBA
from coordinate_transformations import grid_to_world


class RoadMap:
    """
    This class contains all nodes defining the roadmap of a path.
    """
    def __init__(self):
        self.locations = []


class Path:
    """
    This class contains all information on a specific path being explored.
    @input: notes - list of nodes if this path is a duplicate of another path, otherwise
            the only element in the list is the starting node.
    @input: orientation - current orientation in the path search
    """
    def __init__(self, nodes, orientation):

        self.current_orientation = orientation
        self.current_location = None
        self.move_along_feature = False
        self.move_around_feature = False
        self.cost = 0
        self.nodes = RoadMap()
        self.nodes.locations.append(nodes)
        self.current_location = nodes.locations[-1]


class PathPlanner:
    """
    Main node which searches for a feasible path and optimizes the selected path.
    @input: start - robot's starting position in the environment.
    @input: target - robot's final position in the environment.
    @input: map - map of the environment.
    @input: step - length of step size at each iteration.
    @output: an optimized, feasible path.
    """
    def __init__(self, map_features, target, start, step):
        self.map_features = map_features
        self.target = target
        self.start = start
        self.d_safe = 0.2
        self.path_list = []
        self.path_list.append(Path([start]))
        self.dl = step
        self.target_reached = False
        self.best_roadmap = None
        self.rate = rospy.Rate(0.2)

    def pathFinder(self):
        pass

    def cornerHandler(self):
        pass

    def featureBumpHandler(self):
        pass

    def pathOptimizer(self):
        return None

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function until the target has been reached.
        """
        while not rospy.is_shutdown():

            while not self.target_reached:
                self.step()
                self.rate.sleep()

            self.best_roadmap = self.pathOptimizer()

        return self.best_roadmap


    def step(self):
        """
        Perform an iteration of the path planning loop.
        @param: self
        @result: generates one more step in each path.
        """
        for path in self.path_list:
            x = path.current_location[0] + self.dl * np.cos(path.current_orientation)
            y = path.current_location[1] + self.dl * np.sin(path.current_orientation)
            dx = np.min(self.map_features[:, 0] - x)
            dy = np.min(self.map_features[:, 1] - y)

            if dx > self.d_safe and dy > self.d_safe:
                path.current_orientation[0] = np.copy(x)
                path.current_orientation[1] = np.copy(y)

            else:
                if path.current_location[0] > self.d_safe and \
                        path.current_location[1] > self.d_safe:
                    path.nodes.locations.append(path.current_location)
                    path.cost += 1



if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("LocalizationNode")  # should this be "LocalizationNode" right ? I changed it
    # go to class that provides all the functionality
    # and check for errors
    try:
        pathPlanner = PathPlanner()
        pathPlanner.run()
    except rospy.ROSInterruptException:
        pass


