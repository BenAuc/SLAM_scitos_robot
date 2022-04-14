#!/usr/bin/env python3

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
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, ColorRGBA
from coordinate_transformations import grid_to_world


class RoadMap:
    """
    This class contains all nodes defining the roadmap of a path.
    """
    def __init__(self):
        self.locations = list()


class PathCandidate:
    """
    This class contains all information on a specific path being explored.
    @input: notes - list of nodes if this path is a duplicate of another path, otherwise
            the only element in the list is the starting node.
    @input: orientation - current orientation in the path search
    """
    def __init__(self, nodes):

        # self.current_orientation = orientation
        self.current_orientation = None
        self.current_location = None
        self.move_along_feature = False
        self.move_around_feature = False
        self.cost = 0
        self.nodes = RoadMap()
        self.nodes.locations.append(nodes)
        self.current_location = self.nodes.locations[-1]


class PathPlanner:
    """
    Main node which searches for a feasible path and optimizes the selected path.
    @input: start - robot's starting position in the environment.
    @input: target - robot's final position in the environment.
    @input: map - map of the environment.
    @input: step - length of step size at each iteration.
    @output: an optimized, feasible path.
    """
    def __init__(self): #, map_features, start, step):
        # self.map_features = map_features
        self.map_features = None
        self.target = None
        self.start = np.array([0, 0])
        self.d_safe = 0.2
        self.path_list = []
        self.path_list.append(PathCandidate([self.start]))
        # self.dl = step
        self.dl = 0.1
        self.target_reached = False
        self.best_roadmap = None
        self.rate = rospy.Rate(0.2)

        ### acquire user-selected goal ###
        # subscriber
        self.target_selection_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.targetSelectCallback)

        # publisher
        self.target_selected_pub = rospy.Publisher("/roadmap/target", Marker)
        self.target_selected_msg = Marker()
        self.target_selected_msg.ns = "roadmap_target"
        self.target_selected_msg.text = "TARGET"
        self.target_selected_msg.id = 0
        self.target_selected_msg.type = np.int(9)  # display marker as 3D test
        self.target_selected_msg.scale.z = 1.0
        self.target_selected_msg.header = Header()
        self.target_selected_msg.header.frame_id = "map"
        self.target_selected_msg.action = np.int(0)
        self.target_selected_msg.pose.orientation.x = 0.0
        self.target_selected_msg.pose.orientation.y = 0.0
        self.target_selected_msg.pose.orientation.z = 0.0
        self.target_selected_msg.pose.orientation.w = 1.0
        self.target_selected_msg.color.a = 1.0
        self.target_selected_msg.color.r = 1.0
        self.target_selected_msg.color.g = 0.0
        self.target_selected_msg.color.b = 0.0

        ### publish selected roadmap ###
        self.roadmap_pub = rospy.Publisher("/roadmap/path", Path, queue_size=1)

        self.roadmap_msg = Path()
        self.roadmap_msg.header = Header()
        self.roadmap_msg.header.frame_id = "map"

    def pathFinder(self):
        pass

    def cornerHandler(self):
        pass

    def featureBumpHandler(self):
        pass

    def pathOptimizer(self):
        return None

    def pathPublish(self, selectedRoadMap):
        """
        Publishes the roadmap selected by the node
        @param: list of way points that constitute the road map
        @result: publishes the roadmap as a Path message
        """

        header_stamp = rospy.get_rostime()
        self.roadmap_msg.header.stamp = header_stamp
        print(selectedRoadMap)

        for point in range(len(selectedRoadMap)):
            next_point = PoseStamped()
            next_point.header = Header()
            next_point.header.frame_id = "map"
            next_point.header.stamp = header_stamp

            next_point.pose.position = Point()
            next_point.pose.position.x = selectedRoadMap[point][0]
            next_point.pose.position.y = selectedRoadMap[point][1]

            next_point.pose.orientation = Quaternion()
            q = quaternion_from_euler(0, 0, 0, 'rzyx')
            next_point.pose.orientation.x = q[0]
            next_point.pose.orientation.y = q[1]
            next_point.pose.orientation.z = q[2]
            next_point.pose.orientation.w = q[3]

            self.roadmap_msg.poses.append(next_point)

        self.roadmap_pub.publish(self.roadmap_msg)


    def targetSelectCallback(self, data):
        """
        Handles incoming PoseStamped message containing the target selected by the user
        @param: incoming PoseStamped message
        @result: saves the selected target and calls the publisher for visualization in Rviz.
        """
        self.target = np.array([data.pose.position.x, data.pose.position.y])
        print("target acquired :", self.target)

        self.target_selected_msg.pose.position.x = self.target[0]
        self.target_selected_msg.pose.position.y = self.target[1]
        self.target_selected_msg.pose.position.z = 0.6
        self.target_selected_msg.header.stamp = rospy.get_rostime()
        self.target_selected_pub.publish(self.target_selected_msg)


    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function until the target has been reached.
        """
        while not rospy.is_shutdown():

            if self.target is not None:
                # self.step()
                roadmap = RoadMap()
                roadmap.locations.append([0, 0])
                roadmap.locations.append([1.2 * self.target[0], 0.33 * self.target[1]])
                roadmap.locations.append([0.8 * self.target[0], 0.66 * self.target[1]])
                roadmap.locations.append(self.target)

                self.pathPublish(roadmap.locations)

                while not self.target_reached:
                    self.rate.sleep()

                # self.best_roadmap = self.pathOptimizer()

        return self.best_roadmap


    def step(self):
        """
        Perform an iteration of the path planning loop.
        @param: self
        @result: generates one more step in each path.
        """
        for path_item in self.path_list:
            x = path_item.current_location[0] + self.dl * np.cos(path_item.current_orientation)
            y = path_item.current_location[1] + self.dl * np.sin(path_item.current_orientation)
            dx = np.min(self.map_features[:, 0] - x)
            dy = np.min(self.map_features[:, 1] - y)

            if dx > self.d_safe and dy > self.d_safe:
                path_item.current_orientation[0] = np.copy(x)
                path_item.current_orientation[1] = np.copy(y)

            else:
                if path_item.current_location[0] > self.d_safe and \
                        path_item.current_location[1] > self.d_safe:
                    path_item.nodes.locations.append(path_item.current_location)
                    path_item.cost += 1



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


