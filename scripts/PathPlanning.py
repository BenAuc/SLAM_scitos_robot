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
    This class contains all nodes that define a roadmap from a start to an end point.
    """

    def __init__(self):
        self.locations = list()


class PathCandidate:
    """
    This class contains all information on a specific path as it is being explored.
    @input: notes - list of nodes if this path is a duplicate of another path, otherwise
            the only element in the list is the starting node.
    @input: orientation - current orientation in the path search
    """

    def __init__(self, nodes):

        # flags to indicate whether the path runs along or moves around features or towards the target
        self.move_along_feature = False
        self.move_around_feature = False
        self.moving_towards_target = False
        self.moving_in_circles = False

        # to keep track of features that stood in the way of the path as obstacles
        self.move_along_id = None
        self.history_obstacle_id = list()
        self.feature_is_obstacle_again = False

        # to register all turning points along the path
        self.nodes = RoadMap()
        self.nodes.locations.append(nodes)
        self.turning_points = RoadMap()
        self.turning_points.locations.append(nodes)

        # to perform the iteration of the path search on the basis of a pose
        self.current_location = self.nodes.locations[-1]
        self.current_orientation = None


class PathPlanner:
    """
    Main node which searches for a feasible path and optimizes the selected path.
    @input: start - robot's starting position in the environment.
    @input: target - robot's final position in the environment.
    @input: map - map of the environment.
    @input: step - length of step size at each iteration.
    @output: an optimized, feasible path.
    """

    def __init__(self):
        # for debugging purposes
        self.rate = rospy.Rate(26)

        ### path planning parameters
        # step size in searching for a path
        self.dl = 0.15
        # distance to keep away from features
        self.d_safe = 1.2
        # distance to travel to move around a feature
        self.d_around = 2.7
        # starting point
        self.start = np.array([0, 0])
        # list of all paths under investigation
        self.path_list = []
        # creation of first path
        self.path_list.append(PathCandidate(self.start))

        # variables to save the final, selected path
        self.target_reached = False
        self.feasible_path = None
        self.best_roadmap = None

        ### visualize obstacles along the path ###
        # publisher
        self.obstacle_pub = rospy.Publisher("/obstacle", Marker, queue_size=1)

        # marker message
        self.obstacle_msg = Marker()
        self.obstacle_msg.lifetime = rospy.Duration.from_sec(0.5)
        self.obstacle_msg.ns = "obstacles"
        self.obstacle_msg.id = 0
        self.obstacle_msg.type = np.int(5)  # display marker as line list
        self.obstacle_msg.scale.x = 0.07
        self.obstacle_msg.header = Header()
        self.obstacle_msg.header.frame_id = "map"
        self.obstacle_msg.header.stamp = rospy.get_rostime()

        ### acquire user-selected goal ###
        # storage variable
        self.target = None

        # subscriber
        self.target_selection_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.targetSelectCallback)

        # publisher
        self.target_selected_pub = rospy.Publisher("/roadmap/target", Marker) #, queue_size=2)

        # marker message containing target location
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
        self.target_selected_msg.color.r = 0.0
        self.target_selected_msg.color.g = 0.0
        self.target_selected_msg.color.b = 1.0

        ### publish path as it is computed ###
        self.roadmap_pub = rospy.Publisher("/roadmap/path", Path, queue_size=10)

        # path message containing roadmap
        self.roadmap_msg = Path()
        self.roadmap_msg.header = Header()
        self.roadmap_msg.header.frame_id = "map"

        ### publish individual turning points along the path as it is computed ###
        # publisher
        self.turns_pub = rospy.Publisher("/roadmap/turning_points_marker_array", MarkerArray, queue_size=1)

        # marker array message
        self.turns_msg = MarkerArray()

        ### publish final roadmap as marker array ###
        # publisher
        self.roadmap_turns_pub = rospy.Publisher("/roadmap/roadmap_marker_array", MarkerArray, queue_size=1)

        ### fetch feature map ###
        # subscriber
        self.feature_map_sub = rospy.Subscriber("/map_features", Marker, self.featureMapCallback)

        # variable containing feature map
        self.features = None
        self.features_x = list()
        self.features_y = list()
        self.features_orientation = list()

        # Recording time and robot pose for performance tracking
        self.startTime = 0
        while self.startTime == 0:
            self.startTime = rospy.Time.now().to_sec()


    def duplicatePath(self, path):
        """
        Duplicates an existing path when an obstacle is faced and there are two ways to circumvent it
        @param: the id of the existing path to be duplicated
        @result: TBD
        """
        # method not yet implemented
        pass


    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function until the target has been reached.
        """
        ### automatically define target for testing purposes ###
        # comment out next few lines if the user selects the target
        # self.target = np.array([3, 16.4]) # success
        # self.target = np.array([-9, 10]) # success
        # self.target = np.array([6, 14]) # success
        # self.target = np.array([-3, 16]) # success
        self.target = np.array([-7, 12.4])  # success
        # self.target = np.array([-5, 11.4])  # no success
        # self.target = np.array([-3.3, 8.4])  # success
        # self.target = np.array([-8, 3.4])  # no success
        # self.target = np.array([5, 6.4])  # success

        self.target_selected_msg.pose.position.x = self.target[0]
        self.target_selected_msg.pose.position.y = self.target[1]
        self.target_selected_msg.pose.position.z = 0.6
        self.target_selected_msg.header.stamp = rospy.get_rostime()
        self.target_selected_pub.publish(self.target_selected_msg)
        ### comment out previous lines if the user selects the target ###

        print("**********")
        print("path planning started")
        print("**********")

        while not self.target_reached and not rospy.is_shutdown():

            # if the feature map and the target have been fetched
            if self.features is not None and self.target is not None:

                # set initial search orientation directly towards target
                self.path_list[0].current_orientation = atan2(self.target[1], self.target[0])

                while not self.target_reached:
                    # perform one search step for each path
                    self.step()
                    self.target_selected_pub.publish(self.target_selected_msg)
                    # let sleep for debugging purposes to have time to analyze
                    # self.rate.sleep()

                # optimize the selected roadmap
                self.best_roadmap = self.pathOptimizer(self.best_roadmap)
                # print("list of waypoints after optimization: ", self.best_roadmap)

                rospy.loginfo(f"A path has been found.")
                endTime = rospy.Time.now().to_sec()
                rospy.loginfo(f"Started search  [s]: {self.startTime}")
                rospy.loginfo(f"Finished search [s]: {endTime}")
                totalTime = endTime - self.startTime
                rospy.loginfo(f"Elapsed time  [s]: {totalTime}")

                # publish the selected roadmap
                self.pathPublish(self.best_roadmap)
                self.turningPointsPublish(self.best_roadmap, True)

        return


    def step(self):
        """
        Perform an iteration of the path planning loop.
        @param: self
        @result: generates one more step in each path.
        """
        # go through each path in the list
        path_id = 0

        for path_item in self.path_list:

            ### if an obstacle has already been encountered, and we're moving along this obstacle ###
            if path_item.move_along_feature:

                # check if this obstacle is still in the vicinity
                if self.noLongerMovingAlong(path_id, 1.2):

                    # lower and raise flags
                    path_item.move_along_feature = False
                    path_item.moving_towards_target = False
                    path_item.move_around_feature = True

                    # change orientation
                    self.makeTurnAround(path_id)

                    # register a turning point
                    # print("registering turning point")
                    path_item.turning_points.locations.append(path_item.current_location)
                    path_item.nodes.locations.append(path_item.current_location)

            ### if we're moving around an obstacle ###
            if path_item.move_around_feature:

                # check if this obstacle is still in the vicinity
                if self.isStillMovingAround(path_id, path_item.current_location[0], path_item.current_location[1]):

                    # lower flags
                    path_item.move_around_feature = False
                    path_item.move_along_feature = False
                    path_item.moving_towards_target = False

                    # change orientation
                    self.makeTurnTowardsTarget(path_id)

                    # register a turning point
                    # print("registering turning point")
                    path_item.turning_points.locations.append(path_item.current_location)
                    path_item.nodes.locations.append(path_item.current_location)

            ### compute next pose on the path given current orientation ###
            x = path_item.current_location[0] + self.dl * np.cos(path_item.current_orientation)
            y = path_item.current_location[1] + self.dl * np.sin(path_item.current_orientation)

            ### if the next pose collides with a feature ###
            feature_id = self.isThereAnObstacle(path_id, x, y)
            if feature_id is not None:

                # register id of the feature
                path_item.move_along_id = feature_id

                # lower flags
                path_item.feature_is_obstacle_again = False
                path_item.moving_towards_target = False

                # if this feature hasn't previously been bumped
                if not (feature_id in path_item.history_obstacle_id):
                    # add to the list of obstacles that have been seen
                    path_item.history_obstacle_id.append(feature_id)

                else:
                    # add to the list of obstacles that have been seen
                    path_item.history_obstacle_id.append(feature_id)
                    # raise flag if the feature has already been bumped
                    path_item.feature_is_obstacle_again = True

                # if we're already moving along an obstacle, a corner has been detected
                if path_item.move_along_feature:
                    # change orientation accordingly
                    path_item.moving_towards_target = False
                    self.cornerHandler(path_id)

                    # register a turning point
                    # print("registering turning point")
                    path_item.turning_points.locations.append(path_item.current_location)
                    path_item.nodes.locations.append(path_item.current_location)

                else:
                    # raise flag and change orientation accordingly
                    path_item.move_along_feature = True
                    path_item.moving_towards_target = False
                    self.featureBumpHandler(path_id, feature_id)

                    # register a turning point
                    # print("registering turning point")
                    path_item.turning_points.locations.append(path_item.current_location)
                    path_item.nodes.locations.append(path_item.current_location)

            # otherwise no obstacle is on the way
            else:
                # register the next pose as the current location
                path_item.current_location = np.array([x, y])
                path_item.nodes.locations.append(path_item.current_location)

            ### if the target is in sight ###
            # if we're not already moving towards the target
            if not path_item.moving_towards_target:

                # if we're in the vicinity of the target
                if norm((x - self.target[0], y - self.target[1])) < 6 * self.d_safe:

                    # if the target is considered in sight
                    if self.isTargetInSight(path_id):

                        # raise and lower flags and change orientation accordingly
                        path_item.moving_towards_target = True
                        path_item.move_along_feature = False
                        self.makeTurnTowardsTarget(path_id)

                        # register a turning point
                        # print("registering turning point")
                        path_item.turning_points.locations.append(path_item.current_location)
                        path_item.nodes.locations.append(path_item.current_location)

                # or if we are not moving around any obstacle
                else:
                    if not path_item.move_along_feature and not path_item.move_around_feature:

                        # if the target is considered in sight
                        if self.isTargetInSight(path_id):

                            # raise and lower flags and change orientation accordingly
                            path_item.moving_towards_target = True
                            path_item.move_along_feature = False
                            self.makeTurnTowardsTarget(path_id)

                            # register a turning point
                            # print("registering turning point")
                            path_item.turning_points.locations.append(path_item.current_location)
                            path_item.nodes.locations.append(path_item.current_location)

            ### if target is reached ###
            if norm((x - self.target[0], y - self.target[1])) < 1.5 * self.dl:
                print("**********")
                print("path planning completed")
                print("**********")
                # raise flag
                self.target_reached = True

                # register the target coordinates as the last pose along the path
                path_item.turning_points.locations.append(self.target)

                # self.best_roadmap = path_item.nodes.locations
                self.best_roadmap = path_item.turning_points.locations
                print("list of waypoints before optimization: ", self.best_roadmap)

            ### visualization for testing purposes ###
            self.pathPublish(path_item.nodes.locations)
            self.turningPointsPublish(np.copy(path_item.turning_points.locations), False)

            # counter of path id would be useful if the method pathDuplicate() was implemented
            path_id += 1


    def pathOptimizer(self, nodes):
        """
        Removes unnecessary nodes from a path that's been deemed feasible
        @param: set of nodes from a feasible path
        @result: returns optimized roadmap as a set of nodes
        """
        print("**********")
        print("path optimization in progress")

        # initialization
        path_id = 0
        iterate_again = True

        # optimize the path as long as the previous iteration has removed some node
        while iterate_again:

            # initiliaze starting point
            x0 = np.copy(nodes[0][0])
            y0 = np.copy(nodes[0][1])

            # initialize current position along the path
            x = np.copy(x0)
            y = np.copy(y0)

            # count number of nodes removed from the roadmap in this iteration
            counter = 0

            # go through each node in the list
            for node_id in range(1, len(nodes)):

                # initialize the index in case nodes have been removed
                node_id -= counter

                # break the loop is we've gone through each node in the list
                if node_id + 1 > len(nodes) - 1 - counter:
                    break

                # initialize the target node by skipping one
                x1 = nodes[node_id + 1][0]
                y1 = nodes[node_id + 1][1]

                # compute bearing of next node with respect to starting point
                bearing = atan2(y1 - y0, x1 - x0)

                # move by increments from the starting point to the following node
                node_not_reached = True
                while node_not_reached:

                    ### compute next pose on the path given current orientation ###
                    x = x + self.dl * np.cos(bearing)
                    y = y + self.dl * np.sin(bearing)

                    ### if the next pose collides with a feature ###
                    feature_id = self.isThereAnObstacle(path_id, x, y)
                    if feature_id is not None:

                        # the node can't be skipped
                        # reinitialize the starting point
                        x0 = np.copy(nodes[node_id][0])
                        y0 = np.copy(nodes[node_id][1])
                        x = np.copy(x0)
                        y = np.copy(y0)
                        break

                    else:
                        ### if the target node is reached ###
                        if norm((x - x1, y - y1)) < 1.5 * self.dl:

                            # register the target coordinates as the last pose along the path
                            node_not_reached = False
                            x = np.copy(x0)
                            y = np.copy(y0)

                            # pop the node that can be skipped
                            for idx in range(1):
                                if node_id + idx >= len(nodes) - 1 - counter:
                                    break
                                nodes.pop(node_id + idx)
                                counter += 1

            # if some nodes have been removed, reiterate once again
            if counter == 0:
                iterate_again = False

        print("path optimization completed")
        print("**********")
        return nodes


    def noLongerMovingAlong(self, path_id, safety_factor):
        """
        Checks whether an obstacle is still in the way and whether we're still moving along it
        @param: id of the path for the query, safety factor to measure how far we cleared the obstacle
        @result: boolean indicating whether the obstacle is still present
        """
        # initialization
        obstacle_cleared = False

        # extract id of the obstacle
        feature_id = self.path_list[path_id].move_along_id

        # fetch current position
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]

        # compute how far away the current position along the path is from the obstacle
        dx = np.min([np.abs(self.features_x[feature_id] - x),
                     np.abs(self.features_x[feature_id + 1] - x)])

        dy = np.min([np.abs(self.features_y[feature_id] - y),
                     np.abs(self.features_y[feature_id + 1] - y)])

        # if the feature is along the x-axis and it's been cleared
        if self.features_orientation[feature_id] == -1:

            if not (np.max([self.features_x[feature_id], self.features_x[feature_id + 1]]) > x > np.min(
                    [self.features_x[feature_id], self.features_x[feature_id + 1]])):

                if dx > safety_factor * self.d_safe:

                    obstacle_cleared = True

        # if the feature is along the y-axis and it's been cleared
        if self.features_orientation[feature_id] == 1:

            if not (np.max([self.features_y[feature_id], self.features_y[feature_id + 1]]) > y > np.min(
                    [self.features_y[feature_id], self.features_y[feature_id + 1]])):

                if dy > safety_factor * self.d_safe:
                    obstacle_cleared = True

        return obstacle_cleared


    def makeTurnAround(self, path_id):
        """
        Handles turn around an obstacle
        @param: id of the path that called the method
        @result: change of orientation in the search for a path
        """
        # fetch current position
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]

        # fetch feature id and feature orientation we're moving around
        feature_id = self.path_list[path_id].move_along_id
        orientation = self.features_orientation[feature_id]

        # if bearing of the target is within -pi/2 and pi/2
        target_bearing = atan2(self.target[1] - y, self.target[0] - x)
        # print("target bearing : ", target_bearing)
        if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:

            if not self.path_list[path_id].moving_in_circles:
                # set new orientation directly to the target
                self.path_list[path_id].current_orientation = target_bearing
            else:
                # otherwise set new orientation perpendicular to the feature we're moving around
                if orientation == 1:
                    if (x - self.features_x[feature_id]) > 0:
                        self.path_list[path_id].current_orientation = np.pi
                    else:
                        self.path_list[path_id].current_orientation = 0

                else:
                    if orientation == -1:
                        if (y - self.features_y[feature_id]) > 0:
                            self.path_list[path_id].current_orientation = -1 * np.pi / 2
                        else:
                            self.path_list[path_id].current_orientation = np.pi / 2

        else:
            # otherwise set new orientation perpendicular to the feature we're moving around
            if orientation == 1:
                if (x - self.features_x[feature_id]) > 0:
                    self.path_list[path_id].current_orientation = np.pi
                else:
                    self.path_list[path_id].current_orientation = 0

            else:
                if orientation == -1:
                    if (y - self.features_y[feature_id]) > 0:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = np.pi / 2


    def makeTurnTowardsTarget(self, path_id):
        """
        Reorients path towards the target
        @param: id of the path that called the method
        @result: change of orientation in the search for a path
        """
        # extract current pose
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]

        # compute bearing of target and assign to current orientation of path search
        target_bearing = atan2(self.target[1] - y, self.target[0] - x)
        self.path_list[path_id].current_orientation = target_bearing

        # raise flag
        self.path_list[path_id].moving_towards_target = True


    def isTargetInSight(self, path_id):
        """
        Checks whether the target is in line with the current position
        @param: id of the path for the query
        @result: boolean indicating whether the target is in sight
        """
        # initialization
        target_in_sight = False

        # extract current pose
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]

        # compute target bearing from current pose
        target_bearing = atan2(self.target[1] - y, self.target[0] - x)

        # compute how far away the current pose is from the target
        dx = np.abs(self.target[0] - x)
        dy = np.abs(self.target[1] - y)

        # if we are far away from this obstacle
        if dx < 1.1 * self.dl:
            target_in_sight = True

        if dy < 1.1 * self.dl:
            target_in_sight = True

        # if target is indeed in sight
        if target_in_sight:
            x = x + self.dl * np.cos(target_bearing)
            y = y + self.dl * np.sin(target_bearing)

            # make sure there isn't an obstacle on the way
            feature_id = self.isThereAnObstacle(path_id, x, y)
            if feature_id is not None:

                # if there is one the target is not in sight
                target_in_sight = False

        return target_in_sight


    def isStillMovingAround(self, path_id, x, y):
        """
        Checks whether an obstacle is still on the path
        @param: id of the path for the query, (x,y) current pose along the path
        @result: boolean indicating whether the obstacle is still present
        """
        # initialization
        obstacle_cleared = False

        # extract id of the obstacle
        feature_id = self.path_list[path_id].move_along_id

        # compute how far away the pose is from the obstacle
        dx = np.min([np.abs(self.features_x[feature_id] - x),
                     np.abs(self.features_x[feature_id + 1] - x)])

        dy = np.min([np.abs(self.features_y[feature_id] - y),
                     np.abs(self.features_y[feature_id + 1] - y)])

        # if we are far away from this obstacle
        if norm([dy, dx]) > self.d_around:
            obstacle_cleared = True

        return obstacle_cleared


    def isThereAnObstacle(self, path_id, x, y):
        """
        Checks whether a given pose collides with a feature
        @param: (x,y) next pose along the path
        @result: returns id of the feature standing as obstacle
        """

        # compute how far is each feature from the pose
        min_dx = np.abs(self.features_x - x)
        min_dy = np.abs(self.features_y - y)

        # initialization
        feature_id = None

        # for each feature with respect to the y-axis
        for idx in range(len(min_dy)):

            # if pose is too close to the feature
            if min_dy[idx] < self.d_safe and \
                    (np.max([self.features_x[2*(idx//2)], self.features_x[2*(idx//2) + 1]]) > x > np.min(
                        [self.features_x[2*(idx//2)], self.features_x[2*(idx//2) + 1]])) and \
                    self.features_orientation[2*(idx//2)] == -1:

                # register this feature as current obstacle
                feature_id = idx
                if (feature_id + 1) % 2 == 0:
                    feature_id -= 1
                break

        # if feature_id is None:
        # for each feature with respect to the x-axis
        for idx in range(len(min_dx)):

            # if pose is too close to the feature
            if min_dx[idx] < self.d_safe and \
                    (np.max([self.features_y[2*(idx//2)], self.features_y[2*(idx//2) + 1]]) > y > np.min(
                        [self.features_y[2*(idx//2)], self.features_y[2*(idx//2) + 1]])) and \
                    self.features_orientation[2*(idx//2)] == 1:

                # register this feature as current obstacle
                feature_id = idx
                if (feature_id + 1) % 2 == 0:
                    feature_id -= 1
                break

        if feature_id == self.path_list[path_id].move_along_id:
            feature_id = None

        # if an obstacle has been observed
        if feature_id is not None:

            ### the rest is for debugging purposes ###
            ### visualize the obstacle ###

            self.obstacle_msg = Marker()
            self.obstacle_msg.lifetime = rospy.Duration.from_sec(3.9)
            self.obstacle_msg.ns = "obstacles"
            self.obstacle_msg.id = 0
            self.obstacle_msg.type = np.int(5)  # display marker as line list
            self.obstacle_msg.scale.x = 0.25
            self.obstacle_msg.header = Header()
            self.obstacle_msg.header.frame_id = "map"
            self.obstacle_msg.header.stamp = rospy.get_rostime()

            # add start point to list
            p_start = Point()
            p_start.x = self.features_x[feature_id]
            p_start.y = self.features_y[feature_id]
            p_start.z = 0.2
            # print("append : ", p_start)
            self.obstacle_msg.points.append(p_start)

            # add end point to list
            p_end = Point()
            p_end.x = self.features_x[feature_id + 1]
            p_end.y = self.features_y[feature_id + 1]
            p_end.z = 0.1
            # print("append : ", p_end)
            self.obstacle_msg.points.append(p_end)

            color = ColorRGBA(0.22, 0.0, 0.5, 1.0)

            self.obstacle_msg.colors.append(color)
            self.obstacle_msg.colors.append(color)

            self.obstacle_pub.publish(self.obstacle_msg)

        return feature_id


    def cornerHandler(self, path_id):
        """
        Handles moving away from a corner
        @param: id of the path for the query, id of the feature standing as obstacle
        @result: change orientation for the path search
        """
        # fetch current position
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]

        # fetch ids of the features making the corner
        across_feature_id = self.path_list[path_id].history_obstacle_id[-1]
        along_feature_id = self.path_list[path_id].history_obstacle_id[-2]

        # fetch orientations of the features making the corner
        across_feature_orientation = self.features_orientation[across_feature_id]
        along_feature_orientation = self.features_orientation[along_feature_id]

        if along_feature_orientation == 1:

            # make sure we're still moving along a feature using a safety factor
            if not self.noLongerMovingAlong(path_id, 0.3):

                # check on which side of the feature and in which direction we're moving along
                if (x > self.features_x[along_feature_id]):
                    self.path_list[path_id].current_orientation = 0

                # if we're on the other side of the feature
                else:
                    self.path_list[path_id].current_orientation = np.pi

            # if we passed the feature this is not a corner
            else:
                # call routine that handles feature bumps instead
                self.featureBumpHandler(path_id, across_feature_id)

        # do the same if the feature is in the other direction
        if along_feature_orientation == -1:

            if not self.noLongerMovingAlong(path_id, 0.3):

                if (y > self.features_y[along_feature_id]):
                    self.path_list[path_id].current_orientation = np.pi / 2

                else:
                    self.path_list[path_id].current_orientation = -np.pi / 2
            else:
                self.featureBumpHandler(path_id, across_feature_id)

        # set the feature we ran into as the new feature we're moving along
        self.path_list[path_id].move_along_id = across_feature_id


    def featureBumpHandler(self, path_id, feature_id):
        """
        Handles turns when an obstacle has been encountered
        @param: id of the path for the query, id of the feature standing as obstacle
        @result: new value for orientation along a given path
        """

        # fetch orientation of the feature
        orientation = self.features_orientation[feature_id]

        # extract pose and bearing of target
        x = self.path_list[path_id].current_location[0]
        y = self.path_list[path_id].current_location[1]
        target_bearing = atan2(self.target[1] - y, self.target[0] - x)

        # change orientation based on feature orientation
        if orientation == 1 and not (np.abs(self.path_list[path_id].current_orientation) == np.pi / 2):

            # if this feature is bumped into a second time, go the opposite direction as default
            if self.path_list[path_id].feature_is_obstacle_again:

                # lower the flag
                self.path_list[path_id].feature_is_obstacle_again = False
                self.path_list[path_id].moving_in_circles = True

                # is the target lies ahead
                if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:
                    if np.abs(self.path_list[path_id].current_orientation - np.pi / 2) <= np.pi / 2:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = np.pi / 2
                else:
                    if np.abs(self.path_list[path_id].current_orientation - np.pi / 2) <= np.pi / 2:
                        self.path_list[path_id].current_orientation = np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2

            else:

                if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:
                    # if (x - self.features_x[feature_id]) > 0:
                    if np.abs(self.path_list[path_id].current_orientation - np.pi / 2) <= np.pi / 2:
                        self.path_list[path_id].current_orientation = np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                else:
                    if np.abs(self.path_list[path_id].current_orientation - np.pi / 2) <= np.pi / 2:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = np.pi / 2


        # do the same in the feature is in the other direction
        else:

            if orientation == -1 and not ((np.abs(self.path_list[path_id].current_orientation) == np.pi) or
                                          (np.abs(self.path_list[path_id].current_orientation) == 0)):

                # if this feature is bumped into a second time, go the opposite direction as default
                if self.path_list[path_id].feature_is_obstacle_again:

                    # lower the flag
                    self.path_list[path_id].feature_is_obstacle_again = False
                    self.path_list[path_id].moving_in_circles = True

                    # if np.pi >= self.path_list[path_id].current_orientation - target_bearing >= 0:
                    if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:
                        # print("target in sight")
                        # if (y - self.features_y[feature_id]) > 0:
                        if np.abs(self.path_list[path_id].current_orientation - np.pi) <= np.pi / 2:
                            self.path_list[path_id].current_orientation = 0
                        else:
                            self.path_list[path_id].current_orientation = np.pi
                    else:
                        # print("target not in sight")
                        # if (y - self.features_y[feature_id]) > 0:
                        if np.abs(self.path_list[path_id].current_orientation - np.pi) <= np.pi / 2:
                            self.path_list[path_id].current_orientation = 0
                        else:
                            self.path_list[path_id].current_orientation = np.pi
                else:
                    # if np.pi >= self.path_list[path_id].current_orientation - target_bearing >= 0:
                    if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:
                        # print("target in sight")
                        # if (y - self.features_y[feature_id]) > 0:
                        if np.abs(self.path_list[path_id].current_orientation - np.pi) <= np.pi / 2:
                            self.path_list[path_id].current_orientation = np.pi
                        else:
                            self.path_list[path_id].current_orientation = 0
                    else:
                        # print("target not in sight")
                        # if (y - self.features_y[feature_id]) > 0:
                        if np.abs(self.path_list[path_id].current_orientation - np.pi) <= np.pi / 2:
                            self.path_list[path_id].current_orientation = np.pi
                        else:
                            self.path_list[path_id].current_orientation = 0

            # in case we're stuck and match no case make a 90 degrees turn
            else:
                self.path_list[path_id].current_orientation -= np.pi / 2


    def featureMapCallback(self, data):
        """
        Handles incoming Marker message containing the feature map published by the localization node saves the features
        as lists of coordinates and orientations.
        @param: incoming Marker message
        @result: saves the map features
        """

        # fetch feature map only once that is if variable is still None
        if self.features is None:

            # initialization
            self.features = np.copy(data.points)
            counter = 0

            # go through each end point of each feature
            for point in self.features:

                # register as x or y coordinate
                self.features_x.append(point.x)
                self.features_y.append(point.y)

                # each second point (start and end points of each feature are next to each other in the list)
                counter += 1
                if counter % 2 == 0:

                    # register the feature orientation
                    delta_x = np.abs(self.features_x[-1] - self.features_x[-2])
                    delta_y = np.abs(self.features_y[-1] - self.features_y[-2])

                    if delta_y > delta_x:
                        self.features_orientation.append(1)
                        self.features_orientation.append(1)
                    else:
                        self.features_orientation.append(-1)
                        self.features_orientation.append(-1)


    def pathPublish(self, steps):
        """
        Publishes the roadmap selected by the node
        @param: steps - list of points that defines the entire path at each step along the way
        @result: publishes the points a Path message
        """

        # fetch time stamp
        header_stamp = rospy.get_rostime()

        ### Path message for entire roadmap

        self.roadmap_msg = Path()
        self.roadmap_msg.header = Header()
        self.roadmap_msg.header.frame_id = "map"
        self.roadmap_msg.header.stamp = header_stamp

        # go through each node in the roadmap
        for point in range(len(steps)):

            ### Path message
            next_point = PoseStamped()
            next_point.header = Header()
            next_point.header.frame_id = "map"
            next_point.header.stamp = header_stamp

            next_point.pose.position = Point()
            next_point.pose.position.x = steps[point][0]
            next_point.pose.position.y = steps[point][1]
            next_point.pose.position.z = 0.0

            next_point.pose.orientation = Quaternion()
            q = quaternion_from_euler(0, 0, 0, 'rzyx')
            next_point.pose.orientation.x = q[0]
            next_point.pose.orientation.y = q[1]
            next_point.pose.orientation.z = q[2]
            next_point.pose.orientation.w = q[3]

            # add this node to the Path message
            self.roadmap_msg.poses.append(next_point)

        self.roadmap_pub.publish(self.roadmap_msg)

    def turningPointsPublish(self, turningPoints, final):
        """
        Publishes all nodes in a roadmap as markers
        @param: turningPoints - list of turning points that constitute the road map
        @param: final - whether we are published the final roadmap
        @result: publishes the turning points as a Marker message
        """

        # fetch time stamp
        header_stamp = rospy.get_rostime()

        ### Marker message list of turning points
        self.turns_msg.markers = []

        # go through each node in the roadmap
        for point in range(len(turningPoints)):

            # individual markers
            marker = Marker()

            if final:
                # marker.lifetime = rospy.Duration.from_sec(2.9)
                pass
            else:
                marker.lifetime = rospy.Duration.from_sec(0.4)

            marker.ns = "turning_points"
            marker.id = point
            marker.type = np.int(2)  # display marker as spheres
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = header_stamp

            # marker coordinates
            marker.pose.position.x = turningPoints[point][0]
            marker.pose.position.y = turningPoints[point][1]
            marker.pose.position.z = 0.2

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            self.turns_msg.markers.append(marker)

        if final:
            self.turns_pub.publish(self.turns_msg)

        else:
            self.roadmap_turns_pub.publish(self.turns_msg)



    def targetSelectCallback(self, data):
        """
        Handles incoming PoseStamped message containing the target selected by the user
        @param: incoming PoseStamped message
        @result: saves the selected target and calls the publisher for visualization in Rviz.
        """
        # register target coordinates
        self.target = np.array([data.pose.position.x, data.pose.position.y])
        self.target_selected_msg.pose.position.x = self.target[0]
        self.target_selected_msg.pose.position.y = self.target[1]
        self.target_selected_msg.pose.position.z = 0.6
        self.target_selected_msg.header.stamp = rospy.get_rostime()

        # publish the target to Rviz
        self.target_selected_pub.publish(self.target_selected_msg)
        print("target acquired : ", self.target)


if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("PathPlanning")
    try:
        pathPlanner = PathPlanner()
        pathPlanner.run()
    except rospy.ROSInterruptException:
        pass
