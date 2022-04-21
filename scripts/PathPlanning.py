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

        # flags to indicate whether the path runs along or moves around features
        self.move_along_feature = False
        self.move_around_feature = False

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
        self.rate = rospy.Rate(50)

        ### path planning parameters
        # step size in searching for a path
        self.dl = 0.025
        # distance to keep away from features
        self.d_safe = 0.7
        # distance to travel to move around a feature
        self.d_around = 1.5
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
        self.target_selected_pub = rospy.Publisher("/roadmap/target", Marker)

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
        self.target_selected_msg.color.r = 1.0
        self.target_selected_msg.color.g = 0.0
        self.target_selected_msg.color.b = 0.0

        ### publish selected roadmap ###
        self.roadmap_pub = rospy.Publisher("/roadmap/path", Path, queue_size=20)

        # path message containing roadmap
        self.roadmap_msg = Path()
        self.roadmap_msg.header = Header()
        self.roadmap_msg.header.frame_id = "map"

        ### fetch feature map ###
        # subscriber
        self.feature_map_sub = rospy.Subscriber("/map_features", Marker, self.featureMapCallback)

        # variable containing feature map
        self.features = None
        self.features_x = list()
        self.features_y = list()
        self.features_orientation = list()


    def duplicatePath(self, path):
        """
        Duplicates an existing path when an obstacle is faced and there are two ways to circumvent it
        @param: the existing path to be duplicated
        @result: TBD
        """
        pass


    def pathOptimizer(self, nodes):
        """
        Removes unnecessary nodes from a path that's been deemed feasible
        @param: set of nodes from the feasible path
        @result: optimized roadmap
        """
        pass

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function until the target has been reached.
        """
        ### automatically define target for testing purposes ###
        # comment out next few lines if the user selects the target
        # self.target = np.array([4, 6])
        # self.target = np.array([-9, 10])
        # self.target = np.array([8, 12])
        self.target = np.array([6, 14])
        print("target acquired :", self.target)
        self.target_selected_msg.pose.position.x = self.target[0]
        self.target_selected_msg.pose.position.y = self.target[1]
        self.target_selected_msg.pose.position.z = 0.6
        self.target_selected_msg.header.stamp = rospy.get_rostime()
        self.target_selected_pub.publish(self.target_selected_msg)

        while not rospy.is_shutdown() and not self.target_reached:

            # if the feature map and the target have been fetched
            if self.features is not None and self.target is not None:

                # set initial search orientation directly towards target
                self.path_list[0].current_orientation = atan2(self.target[1], self.target[0])

                while not self.target_reached:
                    # perform one search step for each path
                    self.step()
                    self.target_selected_pub.publish(self.target_selected_msg)
                    # let sleep for debugging purposes to have time to analyze
                    self.rate.sleep()

                # optimize the selected roadmap
                # self.best_roadmap = self.pathOptimizer(self.best_roadmap)

                # publish the selected roadmap
                self.pathPublish(self.best_roadmap)

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
                if self.stillMovingAlong(path_id, path_item.current_location[0], path_item.current_location[1]):

                    # lower and raise flags
                    path_item.move_along_feature = False
                    path_item.move_around_feature = True

                    # change orientation
                    self.makeTurnAround(path_id, path_item.current_location[0], path_item.current_location[1])

                    # register a turning point
                    print("register turning point")
                    path_item.turning_points.locations.append(path_item.current_location)
                    path_item.nodes.locations.append(path_item.current_location)

            ### if we're moving around an obstacle ###
            if path_item.move_around_feature:

                # check if this obstacle is still in the vicinity
                if self.stillMovingAround(path_id, path_item.current_location[0], path_item.current_location[1]):

                    # lower flag
                    path_item.move_around_feature = False

                    # change orientation
                    self.makeTurnTowardsTarget(path_id, path_item.current_location[0], path_item.current_location[1])

            ### compute next pose on the path given current orientation ###
            x = path_item.current_location[0] + self.dl * np.cos(path_item.current_orientation)
            y = path_item.current_location[1] + self.dl * np.sin(path_item.current_orientation)

            ### if the next pose collides with a feature ###
            feature_id = self.isThereAnObstacle(path_id, x, y)
            if feature_id is not None:
                print("****** new obstacle *******")
                # register id of the feature
                path_item.move_along_id = feature_id
                path_item.feature_is_obstacle_again = False

                if not (feature_id in path_item.history_obstacle_id):
                    print("feature never seen")
                    path_item.history_obstacle_id.append(feature_id)

                # raise flag if the feature has already been an obstacle
                else:
                    print("feature already seen")
                    path_item.history_obstacle_id.append(feature_id)
                    path_item.feature_is_obstacle_again = True

                # if we're already moving along an obstacle
                if path_item.move_along_feature:
                    # change orientation accordingly
                    self.cornerHandler(path_id, path_item.current_location[0], path_item.current_location[1])

                else:
                    # raise flag change orientation accordingly
                    path_item.move_along_feature = True
                    self.featureBumpHandler(path_id, feature_id)

                # # recompute next pose on the path
                # x = path_item.current_location[0] + self.dl * np.cos(path_item.current_orientation)
                # y = path_item.current_location[1] + self.dl * np.sin(path_item.current_orientation)
                #
                # # if no obstacle is faced at the projected pose
                # if not self.isThereAnObstacle(path_id, x, y):
                #     print("****** no obstacle after reorienting *******")
                #     # register a turning point and set current location
                #     print("register turning point")
                #     path_item.turning_points.locations.append(path_item.current_location)
                #     path_item.nodes.locations.append(path_item.current_location)
                #     path_item.current_location = np.array([x, y])

            # if no obstacle is faced
            else:
                # print("****** no new obstacle *******")
                # register the next pose as the current location
                path_item.current_location = np.array([x, y])

                # register point for testing purposes
                path_item.nodes.locations.append(path_item.current_location)

            ### if target is reached ###
            if norm((x - self.target[0], y - self.target[1])) < self.dl:
                print("target reached")

                # register the target coordinates as the last pose along the path
                self.target_reached = True
                path_item.turning_points.locations.append(self.target)
                self.best_roadmap = path_item.turning_points.locations


            ### visualization for testing purposes ###
            # if (len(path_item.nodes.locations)) % 5 == 0:
            self.pathPublish(path_item.nodes.locations)

            path_id += 1


    def stillMovingAlong(self, path_id, x, y):
        """
        Checks whether an obstacle is still on the path
        @param: id of the path for the query, (x,y) current pose along the path
        @result: boolean indicating whether the obstacle is still present
        """
        print("****** checking if still moving along ******")

        # initialization
        obstacle_cleared = False

        # extract id of the obstacle
        feature_id = self.path_list[path_id].move_along_id

        # compute how far away the current position along the path is from the obstacle
        dx = np.min([np.abs(self.features_x[feature_id] - x),
                     np.abs(self.features_x[feature_id + 1] - x)])

        dy = np.min([np.abs(self.features_y[feature_id] - y),
                     np.abs(self.features_y[feature_id + 1] - y)])

        # for testing purposes
        # print("feature id :", self.features_y[self.path_list[path_id].move_along_id])
        # print("dx, dy : ", dx, dy)
        # print("x, y : ", x, y)
        # print("feature x1, x2 : ", x, y)

        # if the feature is along the x-axis and it's been cleared
        if self.features_orientation[feature_id] == -1 and dx > self.d_safe:

            if not (np.max([self.features_x[feature_id], self.features_x[feature_id + 1]]) > x > np.min(
                    [self.features_x[feature_id], self.features_x[feature_id + 1]])):

                obstacle_cleared = True
                print("feature id :", self.path_list[path_id].move_along_id)
                print("****** not moving along anymore ******")


        # if the feature is along the y-axis and it's been cleared
        if self.features_orientation[feature_id] == 1 and dy > self.d_safe:

            if not (np.max([self.features_y[feature_id], self.features_y[feature_id + 1]]) > y > np.min(
                    [self.features_y[feature_id], self.features_y[feature_id + 1]])):

                obstacle_cleared = True
                print("feature id :", self.features_y[self.path_list[path_id].move_along_id])
                print("****** not moving along anymore ******")

        return obstacle_cleared


    def makeTurnAround(self, path_id, x, y):
        """
        Handles turn around an obstacle
        @param: id of the path that called the method, (x,y) current pose along the path
        @result: change of orientation
        """
        print(("***** moving around target *****"))

        print("old orientation :", self.path_list[path_id].current_orientation)

        ### new code

        # # fetch orientation of the feature
        # feature_id = self.path_list[path_id].move_along_id
        # orientation = self.features_orientation[feature_id]
        #
        # print("feature orientation : ", orientation)
        #
        # if not self.path_list[path_id].feature_is_obstacle_a      # fetch orientation of the feature
        feature_id = self.path_list[path_id].move_along_id
        orientation = self.features_orientation[feature_id]

        print("feature orientation : ", orientation)

        if not self.path_list[path_id].feature_is_obstacle_again:
            # change orientation based on feature orientation
            if orientation == 1:

                if self.path_list[path_id].feature_is_obstacle_again:

                    self.path_list[path_id].feature_is_obstacle_again = False

                    if np.pi >= self.path_list[path_id].current_orientation > 0:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = np.pi / 2

                else:
                    if np.pi >= self.path_list[path_id].current_orientation > 0:
                        self.path_list[path_id].current_orientation = np.pi / 2
                    else:
                        self.path_list[path_id].current_orientation = -1 * np.pi / 2
                print("new orientation : ", self.path_list[path_id].current_orientation)

            else:

                if orientation == -1:
                    if self.path_list[path_id].feature_is_obstacle_again:

                        self.path_list[path_id].feature_is_obstacle_again = False
                        if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
                            self.path_list[path_id].current_orientation = np.pi  # -1 * np.pi
                        else:
                            self.path_list[path_id].current_orientation = 0
                        print("new orientation : ", self.path_list[path_id].current_orientation)

                    else:

                        if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
                            self.path_list[path_id].current_orientation = 0  # -1 * np.pi
                        else:
                            self.path_list[path_id].current_orientation = np.pi
                        print("new orientation : ", self.path_list[path_id].current_orientation)
                else:
                    print("didn't fit any case *****")
                    self.path_list[path_id].current_orientation -= np.pi / 2
                    print("new orientation : ", self.path_list[path_id].current_orientation)
        else:
            # if bearing of the target is within -pi/2 and pi/2
            target_bearing = atan2(self.target[1] - y, self.target[0] - x)
            print("target bearing : ", target_bearing)
            if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:

                # set new orientation directly to the target
                self.path_list[path_id].current_orientation = target_bearing

                # if not
            else:
                # make a 90 degrees turn towards the target
                if self.path_list[path_id].feature_is_obstacle_again:
                    if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
                        self.path_list[path_id].current_orientation -= np.pi / 2

                    if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
                        self.path_list[path_id].current_orientation += np.pi / 2

                else:
                    if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
                        self.path_list[path_id].current_orientation += np.pi / 2

                    if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
                        self.path_list[path_id].current_orientation -= np.pi / 2gain:
        #     # change orientation based on feature orientation
        #     if orientation == 1:
        #
        #         if self.path_list[path_id].feature_is_obstacle_again:
        #
        #             self.path_list[path_id].feature_is_obstacle_again = False
        #
        #             if np.pi >= self.path_list[path_id].current_orientation > 0:
        #                 self.path_list[path_id].current_orientation = -1 * np.pi / 2
        #             else:
        #                 self.path_list[path_id].current_orientation = np.pi / 2
        #
        #         else:
        #             if np.pi >= self.path_list[path_id].current_orientation > 0:
        #                 self.path_list[path_id].current_orientation = np.pi / 2
        #             else:
        #                 self.path_list[path_id].current_orientation = -1 * np.pi / 2
        #         print("new orientation : ", self.path_list[path_id].current_orientation)
        #
        #     else:
        #
        #         if orientation == -1:
        #             if self.path_list[path_id].feature_is_obstacle_again:
        #
        #                 self.path_list[path_id].feature_is_obstacle_again = False
        #                 if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
        #                     self.path_list[path_id].current_orientation = np.pi  # -1 * np.pi
        #                 else:
        #                     self.path_list[path_id].current_orientation = 0
        #                 print("new orientation : ", self.path_list[path_id].current_orientation)
        #
        #             else:
        #
        #                 if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
        #                     self.path_list[path_id].current_orientation = 0  # -1 * np.pi
        #                 else:
        #                     self.path_list[path_id].current_orientation = np.pi
        #                 print("new orientation : ", self.path_list[path_id].current_orientation)
        #         else:
        #             print("didn't fit any case *****")
        #             self.path_list[path_id].current_orientation -= np.pi / 2
        #             print("new orientation : ", self.path_list[path_id].current_orientation)
        # else:
        #     # if bearing of the target is within -pi/2 and pi/2
        #     target_bearing = atan2(self.target[1] - y, self.target[0] - x)
        #     print("target bearing : ", target_bearing)
        #     if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:
        #
        #         # set new orientation directly to the target
        #         self.path_list[path_id].current_orientation = target_bearing
        #
        #         # if not
        #     else:
        #         # make a 90 degrees turn towards the target
        #         if self.path_list[path_id].feature_is_obstacle_again:
        #             if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
        #                 self.path_list[path_id].current_orientation -= np.pi / 2
        #
        #             if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
        #                 self.path_list[path_id].current_orientation += np.pi / 2
        #
        #         else:
        #             if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
        #                 self.path_list[path_id].current_orientation += np.pi / 2
        #
        #             if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
        #                 self.path_list[path_id].current_orientation -= np.pi / 2

        ### backup

        # if bearing of the target is within -pi/2 and pi/2
        target_bearing = atan2(self.target[1] - y, self.target[0] - x)
        print("target bearing : ", target_bearing)
        if np.abs(self.path_list[path_id].current_orientation - target_bearing) <= np.pi / 2:

            # set new orientation directly to the target
            self.path_list[path_id].current_orientation = target_bearing

            # if not
        else:
            # make a 90 degrees turn towards the target
            if self.path_list[path_id].feature_is_obstacle_again:
                if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
                    self.path_list[path_id].current_orientation -= np.pi / 2

                if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
                    self.path_list[path_id].current_orientation += np.pi / 2

            else:
                if (self.path_list[path_id].current_orientation - target_bearing) > np.pi / 2:
                    self.path_list[path_id].current_orientation += np.pi / 2

                if (self.path_list[path_id].current_orientation - target_bearing) < -1 * np.pi / 2:
                    self.path_list[path_id].current_orientation -= np.pi / 2

        ### backup

        print("new orientation :", self.path_list[path_id].current_orientation)


    def makeTurnTowardsTarget(self, path_id, x, y):
        """
        Reorients path towards the target
        @param: id of the path that called the method, (x,y) current pose along the path
        @result: change of orientation
        """
        print(("***** orienting towards target *****"))
        print("old orientation :", self.path_list[path_id].current_orientation)

        target_bearing = atan2(self.target[1] - y, self.target[0] - x)
        self.path_list[path_id].current_orientation = target_bearing

        print("new orientation :", self.path_list[path_id].current_orientation)


    def stillMovingAround(self, path_id, x, y):
        """
        Checks whether an obstacle is still on the path
        @param: id of the path for the query, (x,y) current pose along the path
        @result: boolean indicating whether the obstacle is still present
        """

        print("**** checking if we moved around the obstacle ****")

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
            print("**** we moved around the obstacle ****")
            obstacle_cleared = True
        else:
            print("**** we haven't moved around yet ****")

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
                # print("feature found :", idx)
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
                # print("feature found :", idx)
                feature_id = idx
                if (feature_id + 1) % 2 == 0:
                    feature_id -= 1
                break

        if feature_id == self.path_list[path_id].move_along_id:
            feature_id = None

        # if an obstacle has been observed
        if feature_id is not None:

            ### the rest is for debugging purposes ###
            print("****************** feature bump ******************")
            print("distance in y to feature : ", min_dy[feature_id])
            print("distance in x to feature : ", min_dx[feature_id])
            print("feature id : ", feature_id)
            # print("position x : ", x)
            # print("position y : ", y)
            # print("feature y max : ", np.max([self.features_y[feature_id], self.features_y[feature_id + 1]]))
            # print("feature y min : ", np.min([self.features_y[feature_id], self.features_y[feature_id + 1]]))
            # print("feature x max : ", np.max([self.features_x[feature_id], self.features_x[feature_id + 1]]))
            # print("feature x min : ", np.min([self.features_x[feature_id], self.features_x[feature_id + 1]]))

            ### visualize the obstacle ###

            self.obstacle_msg = Marker()
            self.obstacle_msg.lifetime = rospy.Duration.from_sec(2.9)
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

            color = ColorRGBA(0.0, 0.0, 1.0, 1.0)

            self.obstacle_msg.colors.append(color)
            self.obstacle_msg.colors.append(color)

            self.obstacle_pub.publish(self.obstacle_msg)

        return feature_id


    def cornerHandler(self, path_id, x, y):
        """
        Handles moving away from a corner
        @param: id of the path for the query, id of the feature standing as obstacle
        @result: change orientation for the path search
        """
        print("****** cornerHandler called ******")
        print("old orientation : ", self.path_list[path_id].current_orientation)

        # fetch ids of the features making the corner
        across_feature_id = self.path_list[path_id].history_obstacle_id[-1]
        along_feature_id = self.path_list[path_id].history_obstacle_id[-2]

        # fetch orientations of the features making the corner
        across_feature_orientation = self.features_orientation[across_feature_id]
        along_feature_orientation = self.features_orientation[along_feature_id]

        print("feature along, orientation : ", along_feature_orientation)
        print("feature across, orientation : ", across_feature_orientation)

        if along_feature_orientation == 1:
            if (x - self.features_x[along_feature_id]) > 0:
                # if self.path_list[path_id].feature_is_obstacle_again:
                #     self.path_list[path_id].current_orientation += np.pi / 2
                # else:
                if self.path_list[path_id].current_orientation == np.pi / 2:
                    self.path_list[path_id].current_orientation -= np.pi / 2
                else:
                    self.path_list[path_id].current_orientation += np.pi / 2
            else:
                # if self.path_list[path_id].feature_is_obstacle_again:
                #     self.path_list[path_id].current_orientation -= np.pi / 2
                # else:
                if self.path_list[path_id].current_orientation == np.pi / 2:
                    self.path_list[path_id].current_orientation += np.pi / 2
                else:
                    self.path_list[path_id].current_orientation -= np.pi / 2

        if along_feature_orientation == -1:
            if (y - self.features_y[along_feature_id]) > 0:
                # if self.path_list[path_id].feature_is_obstacle_again:
                #     self.path_list[path_id].current_orientation -= np.pi / 2
                # else:
                if self.path_list[path_id].current_orientation == 0:
                    self.path_list[path_id].current_orientation += np.pi / 2
                else:
                    self.path_list[path_id].current_orientation -= np.pi / 2

            else:
                # if self.path_list[path_id].feature_is_obstacle_again:
                #     self.path_list[path_id].current_orientation -= np.pi / 2
                # else:
                if self.path_list[path_id].current_orientation == 0:
                    self.path_list[path_id].current_orientation -= np.pi / 2
                else:
                    self.path_list[path_id].current_orientation += np.pi / 2

        print("new orientation : ", self.path_list[path_id].current_orientation)


    def featureBumpHandler(self, path_id, feature_id):
        """
        Handles turns when an obstacle has been encountered
        @param: id of the path for the query, id of the feature standing as obstacle
        @result: new value for orientation along a given path
        """

        # fetch orientation of the feature
        orientation = self.features_orientation[feature_id]

        print("****** featureBumpHandler called ******")
        print("old orientation : ", self.path_list[path_id].current_orientation)
        print("feature orientation : ", orientation)

        # change orientation based on feature orientation
        if orientation == 1 and not (np.abs(self.path_list[path_id].current_orientation) == np.pi / 2):

            if self.path_list[path_id].feature_is_obstacle_again:

                self.path_list[path_id].feature_is_obstacle_again = False

                if np.pi >= self.path_list[path_id].current_orientation > 0:
                    self.path_list[path_id].current_orientation = -1 * np.pi / 2
                else:
                    self.path_list[path_id].current_orientation = np.pi / 2

            else:
                if np.pi >= self.path_list[path_id].current_orientation > 0:
                    self.path_list[path_id].current_orientation = np.pi / 2
                else:
                    self.path_list[path_id].current_orientation = -1 * np.pi / 2
            print("new orientation : ", self.path_list[path_id].current_orientation)

        else:

            if orientation == -1 and not ((np.abs(self.path_list[path_id].current_orientation) == np.pi) or
                                          (np.abs(self.path_list[path_id].current_orientation) == 0)):
                if self.path_list[path_id].feature_is_obstacle_again:

                    self.path_list[path_id].feature_is_obstacle_again = False
                    if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
                        self.path_list[path_id].current_orientation = np.pi # -1 * np.pi
                    else:
                        self.path_list[path_id].current_orientation = 0
                    print("new orientation : ", self.path_list[path_id].current_orientation)

                else:

                    if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
                        self.path_list[path_id].current_orientation = 0  # -1 * np.pi
                    else:
                        self.path_list[path_id].current_orientation = np.pi
                    print("new orientation : ", self.path_list[path_id].current_orientation)
            else:
                print("didn't fit any case *****")
                self.path_list[path_id].current_orientation -= np.pi / 2
                print("new orientation : ", self.path_list[path_id].current_orientation)


        ### code backup ####

        # # fetch orientation of the feature
        # orientation = self.features_orientation[feature_id]
        #
        # print("****** featureBumpHandler called ******")
        #
        # # change current orientation along the path based on feature orientation
        # # this will eventually be refined
        # if orientation == 1 and not (np.abs(self.path_list[path_id].current_orientation) == np.pi / 2):
        #     print("old orientation : ", self.path_list[path_id].current_orientation)
        #     if np.pi >= self.path_list[path_id].current_orientation > 0:
        #         self.path_list[path_id].current_orientation = np.pi / 2
        #     else:
        #         self.path_list[path_id].current_orientation = -1 * np.pi / 2
        #     print("new orientation : ", self.path_list[path_id].current_orientation)
        #     print("feature orientation : ", orientation)
        # else:
        #
        #     if orientation == -1:
        #         print("old orientation : ", self.path_list[path_id].current_orientation)
        #         if np.pi / 2 >= self.path_list[path_id].current_orientation > -1 * np.pi / 2:
        #             self.path_list[path_id].current_orientation = 0 #-1 * np.pi
        #         else:
        #             self.path_list[path_id].current_orientation = -1 * np.pi
        #         print("new orientation : ", self.path_list[path_id].current_orientation)
        #         print("feature orientation : ", orientation)
        #     else:
        #         print("orientation left unchanged")



    def featureMapCallback(self, data):
        """
        Handles incoming Marker message containing the feature map published by the localization node saves the features
        as lists of coordinates and orientations.
        @param: incoming Marker message
        @result: self.features_x, self.features_y, self.features_orientation
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


    def pathPublish(self, selectedRoadMap):
        """
        Publishes the roadmap selected by the node
        @param: selectedRoadMap - list of way points that constitute the road map
        @result: publishes the roadmap as a Path message
        """

        # fetch time stamp
        header_stamp = rospy.get_rostime()

        self.roadmap_msg = Path()
        self.roadmap_msg.header = Header()
        self.roadmap_msg.header.frame_id = "map"
        self.roadmap_msg.header.stamp = header_stamp

        # go through each node in the roadmap
        for point in range(len(selectedRoadMap)):

            next_point = PoseStamped()
            next_point.header = Header()
            next_point.header.frame_id = "map"
            next_point.header.stamp = header_stamp

            next_point.pose.position = Point()
            next_point.pose.position.x = selectedRoadMap[point][0]
            next_point.pose.position.y = selectedRoadMap[point][1]
            next_point.pose.position.z = 0.0

            next_point.pose.orientation = Quaternion()
            q = quaternion_from_euler(0, 0, 0, 'rzyx')
            next_point.pose.orientation.x = q[0]
            next_point.pose.orientation.y = q[1]
            next_point.pose.orientation.z = q[2]
            next_point.pose.orientation.w = q[3]

            # print("point in path # ", point, " x,y :", selectedRoadMap[point][0], selectedRoadMap[point][1])

            # add this node to the Path message
            self.roadmap_msg.poses.append(next_point)

        self.roadmap_pub.publish(self.roadmap_msg)


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
    rospy.init_node("LocalizationNode")
    try:
        pathPlanner = PathPlanner()
        pathPlanner.run()
    except rospy.ROSInterruptException:
        pass
