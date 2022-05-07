#!/usr/bin/env python3

"""
Template for IAS0060 home assignment 2 Project 1 (SCITOS) .
Node to take a set of waypoints and to drive a differential
drive robot through those waypoints using a simple PID controller
and provided odometry data.

Students should complete the code. Note the places marked with "# TODO".

@author: Christian Meurer
@date: January 2022
@input: Odometry as nav_msgs Odometry message
@output: body velocity commands as geometry_msgs Twist message;
         list of waypoints as MarkerArray message
"""

import math

import numpy.linalg
import rospy
import numpy as np
from numpy import arctan2 as atan2
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray, Marker


class PIDController:
    """
    class for PID controller to calculate desired controller output
    """
    def __init__(self, dt):
        """
        class initialization
        @param: self
        @param: dt - time differential between control loops, determined
                     by updating frequency of the MotionController node
        @result: get controller gains from parameter server and
                 initialize class variables
        """
        ### timing ###
        self.dt = dt

        ### get controller gains as (2x1) vectors ###
        self.Kp = np.array(rospy.get_param("controller_diffdrive/gains/p"))

        self.Ki = np.array(rospy.get_param("controller_diffdrive/gains/i"))
        self.Kd = np.array(rospy.get_param("controller_diffdrive/gains/d"))

        ### auxilary variables ###
        self.last_error = np.zeros(2)
        self.int_error = np.zeros(2)

    def control(self, error):
        """
        control update of the controller class
        @param: self
        @param: e1 - (2x1) error vector for linear and angular position
        @result: cmd - (2x1) vector of controller commands
        """
        # Todo: Your code here
        # cumulate error
        self.int_error += error

        # print("current error: ", error)

        # avoid a
        if self.last_error.all() == 0:
            self.last_error = error

        cmd = np.multiply(self.Kp, error) + np.multiply(self.Ki, self.int_error) + np.multiply(self.Kd, (
            error - self.last_error) / self.dt)

        ### capping control commands to regulate the behavior ###
        if np.abs(error[1]) > np.pi / 8:
            if cmd[0] > 0.4:
                cmd[0] = 0.4

        # cap on angular velocity
        if np.abs(error[1]) > np.pi / 10:
            if cmd[1] > 1.9:
                cmd[1] = 1.9
            if cmd[1] < -1.9:
                cmd[1] = -1.9

        # cap on linear velocity
        if cmd[0] > 1.5:
            cmd[0] = 1.5

        self.last_error = error
        # print("commands :", cmd)
        return cmd

    def set_int_error_to_zero(self):
        """
        Called after a waypoint has been reached to set the accumulated error to zero
        @param: self
        @result: sets the variable self.int_error to zero
        """
        self.int_error = np.zeros(2)


class MotionController:
    """
    main class of the node that performs controller updates
    """
    def __init__(self, rate):
        """
        class initialization
        @param: self
        @param: rate - updating frequency for this node in [Hz]
        @result: get static parameters from parameter server
                 (which have been loaded onto the server from the yaml
                 file using the launch file), to initialize the
                 controller, and to set up publishers and subscribers
        """
        ### timing ###
        self.dt = 1.0 / rate

        self.rate = rospy.Rate(rate)

        ### define subscribers ###
        # self.odom_sub = rospy.Subscriber('/controller_diffdrive/odom', Odometry, self.onOdom)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.onOdom)
        self.localization_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.localizationCallback)

        ### define publishers ###
        # self.cmd_vel_pub = rospy.Publisher("/controller_diffdrive/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoints_pub = rospy.Publisher(
            "/mission_control/waypoints", MarkerArray, queue_size=10)

        ### messages to be handled ###
        self.odom_msg = None
        self.marker_array_msg = MarkerArray()
        self.twist_msg = Twist()

        ### get parameters ###
        # self.waypoints = rospy.get_param("/mission/waypoints")
        self.waypoints = None
        self.current_waypoint = [0, 0]
        # print("previous waypoint :", self.current_waypoint)
        # self.distance_margin = rospy.get_param("/mission/distance_margin")
        self.distance_margin = 0.15

        ### fetch roadmap ###
        # subscriber
        self.waypoints_sub = rospy.Subscriber("/roadmap/turning_points_marker_array", MarkerArray, self.roadmapCallback)

        ### initialization of class variables ###
        self.wpIndex = 0    # counter for visited waypoints
        self.done_tracking = False

        ### initialize controller class ###
        self.pid = PIDController(self.dt)

        # TODO: initialize additional class variables if necessary
        self.pose_2D = {'robot_x': 0.0, 'robot_y': 0.0}
        self.theta = 0.0

        # Recording time and robot pose for performance tracking
        self.startTime = 0
        while self.startTime == 0:
            self.startTime = rospy.Time.now().to_sec()


        self.currentTime = None
        self.position_history = {'x': list(), 'y': list(), 'x_planned': list(), 'y_planned': list(), 't': list()}


    def run(self):
        """
        Main loop of the class
        @param: self
        @result: runs the step function for motion control update
        """

        print("**********")
        print("path execution started")
        print("**********")

        while not rospy.is_shutdown():

            if self.waypoints is not None:
                # print("waypoints acquired")
                # self.waypoints = None
                ### run only when odometry data is available and we still
                # have waypoints to reach ###
                if self.odom_msg and not self.done_tracking:
                    # print("*** step ***")
                    self.step()

            # regulate motion control update according to desired timing
            self.rate.sleep()



    def step(self):
        """
        Perform an iteration of the motion control update where the
        desired velocity commands are calculated based on the defined
        controller and the calculated errors
        @param: self
        @result: publishes velocity commands as Twist message
        """
        ### check if current waypoint is reached and set new one if
        # necessary, additionally keep track of time required for
        # tracking ###
        if self.isWaypointReached():
            if not self.setNextWaypoint():
                if not self.done_tracking:
                    rospy.loginfo(f"This was the last waypoint in the list.")
                    endTime = rospy.Time.now().to_sec()
                    rospy.loginfo(f"Started node  [s]: {self.startTime}")
                    rospy.loginfo(f"Finished node [s]: {endTime}")
                    totalTime = endTime - self.startTime
                    rospy.loginfo(f"Elapsed time  [s]: {totalTime}")
                    self.done_tracking = True
                    # create plots for performance assessment
                    self.createPlots()

        if not self.done_tracking:

            #TODO: Your code here

            # record planned and current position
            self.logPosition()

            ### calculate error ###
            distance, angle = self.compute_error()
            error_to_pid = np.array([distance, angle])
            if error_to_pid[1] > np.pi:
                error_to_pid[1] -= 2 * np.pi
                # print("corrected theta :", error_to_pid[1])

            if error_to_pid[1] < -np.pi:
                error_to_pid[1] += 2 * np.pi
                # print("corrected theta :", error_to_pid[1])


            # print("Error sent to pid is : {}".format(error_to_pid))

            ### call controller class to get controller commands ###
            cmd = self.pid.control(error_to_pid)
            # print("Command received from PID : {}".format(cmd))

            self.twist_msg.linear.x = cmd[0]
            self.twist_msg.angular.z = cmd[1]

            # print("velocity commands: ", self.twist_msg)
            ### publish cmd_vel (and marker array) ###
            self.publish_vel_cmd()
            self.publish_waypoints()


    def createPlots(self):
        """
        Create plots of robot's planned and actual position for performance assessment.
        @param: self
        @result: generates the plots
        """

        # fig = plt.figure()
        # fig.add_axes()
        font_size = 24
        line_width = 3

        fig, ax = plt.subplots(2, 1)
        #fig.subtitle("Comparison between planned and actual position as a function of time")


        # ax = fig.add_subplot(111)
        ax[0].plot(self.position_history['t'], self.position_history['x'], color='green', linewidth=line_width)
        ax[0].plot(self.position_history['t'], self.position_history['x_planned'], color='blue', linewidth=line_width)
        ax[0].set_title("Planned (blue) and actual (green) x-position as a function of time", fontsize=font_size)
        ax[0].set_ylabel('x position (m)', fontsize=font_size)

        ax[1].plot(self.position_history['t'], np.abs(np.asarray(self.position_history['x_planned']) - np.asarray(self.position_history['x'])), color='red', linewidth=1)
        ax[1].set_title("Error in x-position as a function of time", fontsize=font_size)
        ax[1].set_ylabel('Error (m)', fontsize=font_size)
        ax[1].set_xlabel('Time (s)', fontsize=font_size)
        # ax[1].set(xlabel='time (s)', ylabel='error (m)', fontsize=20)
        # ax[0].set_ylim(0, 6)

        plt.show()

        fig, ax = plt.subplots(2, 1)

        # ax2 = fig.add_subplot(212)
        ax[0].plot(self.position_history['t'], self.position_history['y'], color='green', linewidth=line_width)
        ax[0].plot(self.position_history['t'], self.position_history['y_planned'], color='blue', linewidth=line_width)
        ax[0].set_title("Planned (blue) and actual (green) y-position as a function of time", fontsize=font_size)
        ax[0].set_ylabel('y position (m)', fontsize=font_size)
        # ax[0].set(ylabel='y position (m)', fontsize=20)

        ax[1].plot(self.position_history['t'], np.abs(np.asarray(self.position_history['y_planned']) - np.asarray(self.position_history['y'])), color='red', linewidth=line_width)
        ax[1].set_title("Error in y-position as a function of time", fontsize=font_size)
        ax[1].set_ylabel('Error (m)', fontsize=font_size)
        ax[1].set_xlabel('Time (s)', fontsize=font_size)
        # ax[1].set(xlabel='time (s)', ylabel='error (m)', fontsize=20)

        plt.show()


    def logPosition(self):
        """
        Records robot position for performance assessment.
        @param: self
        @result: stores the waypoints in the variable self.position_history
        """

        if self.currentTime is not None:
            theta = atan2(self.waypoints[0][0] - self.current_waypoint[0],
                          self.waypoints[0][1] - self.current_waypoint[1])

            distance = norm([self.pose_2D['robot_x'] - self.current_waypoint[0],
                             self.pose_2D['robot_y'] - self.current_waypoint[1]])

            planned_x = distance * np.sin(theta) + self.current_waypoint[0]
            planned_y = distance * np.cos(theta) + self.current_waypoint[1]

            self.position_history['x'].append(self.pose_2D['robot_x'])
            self.position_history['x_planned'].append(planned_x)
            self.position_history['y'].append(self.pose_2D['robot_y'])
            self.position_history['y_planned'].append(planned_y)
            self.position_history['t'].append(self.currentTime)


    def roadmapCallback(self, data):
        """
        Handles incoming MarkerArray message containing the waypoints.
        @param: self
        @result: stores the waypoints in the variable self.waypoints
        """

        if self.waypoints is None:
            waypoint_list = list()

            # extract list of markers
            markers_list = np.copy(data.markers)

            # go through each node in the roadmap
            for idx in range(1, len(markers_list)):
                waypoint = list()

                x = markers_list[idx].pose.position.x
                y = markers_list[idx].pose.position.y
                #
                # print("x, y :", x, y)

                waypoint.append(x)
                waypoint.append(y)

                waypoint_list.append(waypoint)
                # print("waypoint :", waypoint)

            # self.waypoints = np.asarray(waypoint_list)
            self.waypoints = waypoint_list

            # print("all waypoints :", self.waypoints)


    def setNextWaypoint(self):
        """
        Removes current waypoint from list and sets next one as current target.
        @param: self
        @result: returns True if the next waypoint exists and has been set,
                 otherwise False
        """
        if not self.waypoints:
            return False

        # save last waypoint for plotting and performance assessment
        self.current_waypoint = self.waypoints[0]
        # print("previous waypoint :", self.current_waypoint)

        self.waypoints.pop(0)

        self.pid.set_int_error_to_zero()

        if not self.waypoints:
            return False
        self.wpIndex += 1

        rospy.loginfo(f"----------------------------------------------")
        rospy.loginfo(f"                Next waypoint                 ")
        rospy.loginfo(f"----------------------------------------------")

        return True

    def isWaypointReached(self):
        """
        Checks if waypoint is reached based on pre-defined threshold.
        @param: self
        @result: returns True if waypoint is reached, otherwise False
        """
        # print("********** isWaypointReached call **********")
        # print("waypoints : ", self.waypoints)
        #
        # if self.waypoints:
        #     return False

        # TODO: calculate Euclidian (2D) distance to current waypoint
        distance, angle = self.compute_error()

        if distance < self.distance_margin:
            return True
        return False

    def compute_error(self):
        """
        Computes the error between the robot's current position and the target waypoint.
        @param: self
        @result: returns the error vector in 2D coordinates as 2x1 vector (euclidian distance, yaw angle)
        """
        # print("********* compute error call *********")
        # print("Current set waypoint is : {}".format(self.waypoints[0]))

        # compute error in 2D coordinates
        position = np.array([self.pose_2D['robot_x'], self.pose_2D['robot_y']])

        error_vector_2D = np.array(self.waypoints[0]) - position
        # print("error x,y:", error_vector_2D)
        # print("error vector in 2D: {}".format(error_vector_2D))

        # compute Euclidian distance on the 2D error vector
        error_distance = np.linalg.norm(error_vector_2D)
        # print("Error distance: {}".format(error_distance))

        # compute yaw angle of the target waypoint and of the error with respect to this target
        target_theta = np.arctan2(error_vector_2D[1], error_vector_2D[0])
        # print("Target theta: {}".format(target_theta))
        # print("Current theta: {}".format(self.theta))

        error_angle = target_theta - self.theta
        # print("Error theta: {}".format(error_angle))

        return error_distance, error_angle

    def publish_vel_cmd(self):
        """
        Publishes command velocities computed by the control algorithm.
        @param: self
        @result: publish message
        """
        # TODO: Your code here
        # print("Twist msg being sent: {}".format(self.twist_msg))
        self.cmd_vel_pub.publish(self.twist_msg)

    def localizationCallback(self, data):
        """
        Callback function that handles incoming Odometry messages published by the localization node
        @param: pose data stored in the odometry message
        @result: global variable pose_2D containing the planar
                 coordinates robot_x, robot_y and the yaw angle theta
        """
        # record time for performance assessment
        self.currentTime = rospy.Time.now().to_sec()

        # generate pose estimate message
        self.pose_2D["robot_x"] = data.pose.position.x
        self.pose_2D["robot_y"] = data.pose.position.y

        euler = euler_from_quaternion([data.pose.orientation.x,
                                       data.pose.orientation.y,
                                       data.pose.orientation.z,
                                       data.pose.orientation.w])

        self.theta = euler[2]


    def onOdom(self, data):
        """
        Callback function that handles incoming Odometry messages and
        performs a partial quaternion to euler angle transformation to
        get the yaw angle theta
        @param: pose data stored in the odometry message
        @result: global variable pose_2D containing the planar
                 coordinates robot_x, robot_y and the yaw angle theta
        """
        # record time for performance assessment
        # self.currentTime = rospy.Time.now().to_sec()

        # make odometry message globally available for run() condition
        self.odom_msg = data

        # TODO: Your code here
        # make 2D pose globally available as np.array

        # self.pose_2D["robot_x"] = self.odom_msg.pose.pose.position.x
        # self.pose_2D["robot_y"] = self.odom_msg.pose.pose.position.y
        # euler = euler_from_quaternion([self.odom_msg.pose.pose.orientation.x,
        #                                self.odom_msg.pose.pose.orientation.y,
        #                                self.odom_msg.pose.pose.orientation.z,
        #                                self.odom_msg.pose.pose.orientation.w])
        # self.theta = euler[2]

    def publish_waypoints(self):
        """
        Helper function to publish the list of waypoints, so that they
        can be visualized in RViz
        @param: self
        @result: publish message
        """
        self.marker_array = MarkerArray()
        marker_id = 0
        for waypoint in self.waypoints:
            # print("Publishing waypoint")
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = 0.05
            marker.id = marker_id
            marker_id += 1
            self.marker_array.markers.append(marker)
        self.waypoints_pub.publish(self.marker_array)


# entry point of the executable calling the main node function of the
# node only if this .py file is executed directly, not imported
if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("MotionController")
    # go to class that provides all the functionality
    # and check for errors
    try:
        controller = MotionController(10)
        controller.run()
    except rospy.ROSInterruptException:
        pass
