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
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
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
        # self.Kp = np.array(rospy.get_param(...))
        # self.Ki = np.array(rospy.get_param(...))
        # self.Kd = np.array(rospy.get_param(...))

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
        # ...
        # cmd = ...

        # return cmd


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
        # self.odom_sub = rospy.Subscriber(...)

        ### define publishers ###
        # self.cmd_vel_pub = rospy.Publisher(...)
        self.waypoints_pub = rospy.Publisher(
            "/mission_control/waypoints", MarkerArray, queue_size=10)

        ### messages to be handled ###
        self.odom_msg = None
        self.marker_array_msg = MarkerArray()
        self.twist_msg = Twist()

        ### get parameters ###
        # self.waypoints = rospy.get_param(...)
        # self.distance_margin = rospy.get_param(...)

        ### initialization of class variables ###
        self.wpIndex = 0    # counter for visited waypoints
        self.done_tracking = False

        ### initialize controller class ###
        self.pid = PIDController(self.dt)

        # TODO: initialize additional class variables if necessary

        # Registering start time of this node for performance tracking
        self.startTime = 0
        while self.startTime == 0:
            self.startTime = rospy.Time.now().to_sec()

    def run(self):
        """
        Main loop of the class
        @param: self
        @result: runs the step function for motion control update
        """
        while not rospy.is_shutdown():
            ### run only when odometry data is available and we still
            # have waypoints to reach ###
            if self.odom_msg and not self.done_tracking:
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

        if not self.done_tracking:
            pass
            #TODO: Your code here

            ### calculate error ###

            ### call controller class to get controller commands ###

            ### publish cmd_vel (and marker array) ###

    def setNextWaypoint(self):
        """
        Removes current waypoint from list and sets next one as current target.
        @param: self
        @result: returns True if the next waypoint exists and has been set,
                 otherwise False
        """
        if not self.waypoints:
            return False

        self.waypoints.pop(0)

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
        if not self.waypoints:
            return False

        # TODO: calculate Euclidian (2D) distance to current waypoint
        if distance < self.distance_margin:
            return True
        return False

    def publish_vel_cmd(self):
        """
        Publishes command velocities computed by the control algorithm.
        @param: self
        @result: publish message
        """
        # TODO: Your code here

    def onOdom(self, data):
        """
        Callback function that handles incoming Odometry messages and
        performs a partial quaternion to euler angle transformation to
        get the yaw angle theta
        @param: pose data stored in the odometry message
        @result: global variable pose_2D containing the planar
                 coordinates robot_x, robot_y and the yaw angle theta
        """
        # make odometry message globally available for run() condition
        self.odom_msg = data

        # TODO: Your code here
        # make 2D pose globally available as np.array

    def publish_waypoints(self):
        """
        Helper function to publishe the list of waypoints, so that they
        can be visualized in RViz
        @param: self
        @result: publish message
        """
        self.marker_array = MarkerArray()
        marker_id = 0
        for waypoint in self.waypoints:
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
        self.publisher_waypoints.publish(self.marker_array)


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