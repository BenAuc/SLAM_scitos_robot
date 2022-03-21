#!/usr/bin/env python3

"""
Template IAS0060 home assignment 4 Project 1 (SCITOS).
Reused by team 3 towards implementation of home assignment 6 Project 1 (SCITOS).

@author: Christian Meurer
@date: February 2022

Update: complete of assignment 6
Team: Scitos group 3
Team members: Benoit Auclair; Michael Bryan
Date: March 23, 2022
"""

import numpy as np
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose, PoseStamped, Point, Twist
from nav_msgs.msg import Odometry

class NoiseModel:
    """
    Class implementing the estimation of the error on the control inputs to the robot
    """

    def __init__(self):
        """
        Initializes the noise model
        @param: alpha - 4 x 1 array of parameters to estimate the error on v, w
        @result: class initialization
        """
        # TO DO: pick value for each parameter alpha
        # at the beginning let's debug with an error that's null
        # the variable alpha is temporary just to debug the class
        alpha = rospy.get_param("/noise_model/alpha")
        print("parameters sent to noise model :", alpha)
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        print("parameter alpha4 received by noise model :", self.alpha4)

    def estimateError(self, v, w):
        """
        This method updates the estimated error given the control inputs.
        @param: 2 control inputs for which there is a level of uncertainty
            v: linear speed w.r.t. x-axis in robot frame
            w: angular speed w.r.t. z-axis in robot frame
        @result: estimated error in a 2 x 2 numpy array
        """
        next_error = np.zeros((2, 2))
        next_error[0, 0] = self.alpha1 * np.power(v, 2) + self.alpha2 * np.power(w, 2)
        next_error[1, 1] = self.alpha3 * np.power(v, 2) + self.alpha4 * np.power(w, 2)

        return next_error


class MotionModel:
    """
    Class implementing the motion model to estimate the robot's state
    """

    def __init__(self, dt):
        """
        Function that ...
        @param: dt - time step (in seconds) to estimate the system's next state
        @param: alpha - parameters to feed in to the noise model
        @result: class initialization
        """
        ### class arguments
        # time step
        self.dt = dt
        self.noise_model = NoiseModel()

        # initialization of the Jacobians
        self.jacobian_G = np.zeros((3, 3))
        self.jacobian_G[0, 0] = 1
        self.jacobian_G[1, 1] = 1
        self.jacobian_G[2, 2] = 1

        self.jacobian_V = np.zeros((3, 2))
        # self.threshold_div_zero = 1e-2

    def predictPose(self, control_input, last_pose):
        """
        This method updates the predicted robot pose.
        @param: control_input - numpy array of dim 2 x 1 containing:
            *linear speed w.r.t. x-axis in robot frame
            *angular speed w.r.t. z-axis in robot frame
        @result: returns:
            *predicted pose in a 3 x 1 numpy array containing x, y, psi
            *estimated error on the control inputs in a 2 x 2 numpy array containing the covariance matrix
        """
        v = control_input[0]
        print("v :", v)
        w = control_input[1]
        print("w :", w)
        # M_t:
        increment = np.array([v * self.dt * np.cos(last_pose[2] + w * self.dt / 2),
                              v * self.dt * np.sin(last_pose[2] + w * self.dt / 2),
                              w * self.dt], float).reshape(3, 1)
        print("increment :", increment)
        next_pose = last_pose + increment.reshape(3, 1)  # was shape (3,1,1) which lead to (3,3,1)
        print("next pose :", next_pose)

        next_error = self.noise_model.estimateError(v, w)

        self.computeJacobian(v, w, last_pose)

        return next_pose, next_error, self.jacobian_G, self.jacobian_V

    def computeJacobian(self, v, w, last_pose):
        """
        This method computes the Jacobians of the kinematic model.
        @param:
            *v: linear speed w.r.t. x-axis in robot frame
            *w: angular speed w.r.t. z-axis in robot frame
        @result:
            *self.jacobian_G: Jacobian with respect to the state estimate
            *self.jacobian_V: Jacobian with respect to the control inputs
        """

        self.jacobian_G[0, 2] = -1 * v * self.dt * np.sin(last_pose[2] + w * self.dt / 2)
        self.jacobian_G[1, 2] = v * self.dt * np.cos(last_pose[2] + w * self.dt / 2)

        self.jacobian_V[0, 0] = self.dt * np.cos(last_pose[2] + w * self.dt / 2)
        self.jacobian_V[0, 1] = -1 * v * np.power(self.dt, 2) * 0.5 * np.sin(last_pose[2] + w * self.dt / 2)

        self.jacobian_V[1, 0] = self.dt * np.sin(last_pose[2] + w * self.dt / 2)
        self.jacobian_V[1, 1] = v * np.power(self.dt, 2) * 0.5 * np.cos(last_pose[2] + w * self.dt / 2)
        self.jacobian_V[2, 1] = self.dt

class KalmanFilter:
    """
    Class called by the main node and which implements the Kalman Filter
    """

    def __init__(self, dt, initial_pose):
        """
        Method that initializes the class
        @param: dt - time step (in seconds) to feed to the motion model
        @param: initial_pose - robot's initial pose when the environment is launched
        @param: alpha - parameters to feed in to the noise model
        @result: class initialization
        """
        ### class arguments
        self.dt = dt
        self.motion_model = MotionModel(self.dt)
        # self.odom_error_model = self.motion_model.error_model

        # TO DO: needs to be initialized with a value
        # coming from ground truth
        self.last_state_mu = initial_pose

        # covariance on initial position is null because pose comes from ground truth
        self.last_covariance = np.zeros((3, 3))
        # robot doesn't move at t = 0
        self.last_control_input = np.zeros((2, 1))

    def predict(self, control_input):
        """
        This method predicts what the next system state will be.
        @param: control_input - numpy array of dim 2 x 1 containing:
            *linear speed w.r.t. x-axis in robot frame, v
            *angular speed w.r.t. z-axis in robot frame, w
        @result: the method returns:
            *next_state - numpy array of dim 3 x 1 containing the 3 tracked variables (x,y,psi)
        """

        # compute the next state i.e. next robot pose knowing current control inputs
        next_state_mu, next_error, jacobian_G, jacobian_V = self.motion_model.predictPose(control_input, self.last_state_mu)
        print("jac V:", jacobian_V)
        print("jac G:", jacobian_G)
        print("next error:", next_error)
        print("next state:", next_state_mu)

        # compute covariance on the state transition probability
        covariance_R = jacobian_V @ next_error @ jacobian_V.T
        next_covariance = jacobian_G @ self.last_covariance @ jacobian_G.T + covariance_R
        print("next_covariance :", next_covariance)

        # store current state estimate, current covariance on prior belief, current control inputs
        # for use in the next iteration
        self.last_state_mu = next_state_mu
        self.last_covariance = next_covariance
        self.last_control_input = control_input

        return next_state_mu


        #######################
        #Compute the Jacobians v1
        #######################

    # def computeJacobian(self, control_input):
    #     """
    #     This method computes the Jacobians of the kinematic model.
    #     @param: control_input - numpy array of dim 2 x 1 containing:
    #         *linear speed w.r.t. x-axis in robot frame, v
    #         *angular speed w.r.t. z-axis in robot frame, w
    #     @result:
    #         *self.jacobian_G: Jacobian with respect to the state estimate
    #         *self.jacobian_V: Jacobian with respect to the control inputs
    #     """

        # delta_g = self.next_state_mu - self.last_state_mu
        # delta_x = np.array(self.next_state_mu[0, 0] - self.last_state_mu[0, 0])
        # delta_y = np.array(self.next_state_mu[1, 0] - self.last_state_mu[1, 0])
        # delta_psi = np.array(self.next_state_mu[2, 0] - self.last_state_mu[2, 0])
        #
        # print("delta_g:", delta_g)
        # print("delta_x:", delta_x)
        # print("delta_y:", delta_y)
        # print("delta_psi:", delta_psi)

        # # we should make sure we don't divide by zero
        # if delta_x.all() > self.threshold_div_zero:
        #     self.jacobian_G[:, 0] = np.array(delta_g / delta_x).reshape(3, )
        # else:
        #     #             self.jacobian_G[:, 0] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
        #     self.jacobian_G[:, 0] = 0
        #
        # if delta_y.all() > self.threshold_div_zero:
        #     self.jacobian_G[:, 1] = np.array(delta_g / delta_y).reshape(3, )
        # else:
        #     #             self.jacobian_G[:, 1] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
        #     self.jacobian_G[:, 1] = 0
        #
        # if delta_psi.all() > self.threshold_div_zero:
        #     self.jacobian_G[:, 2] = np.array(delta_g / delta_psi).reshape(3, )
        # else:
        #     #             self.jacobian_G[:, 2] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
        #     self.jacobian_G[:, 2] = 0
        #
        # delta_v = control_input[0] - self.last_control_input[0]
        # delta_w = control_input[1] - self.last_control_input[1]
        #
        # print("delta_v:", delta_v)
        # print("delta_w:", delta_w)
        #
        # if delta_v.all() > self.threshold_div_zero:
        #     self.jacobian_V[:, 0] = np.array(delta_g / delta_v).reshape(3, )
        # else:
        #     #             self.jacobian_V[:, 0] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
        #     self.jacobian_V[:, 0] = 0
        #
        # if delta_w.all() > self.threshold_div_zero:
        #     self.jacobian_V[:, 1] = np.array(delta_g / delta_w).reshape(3, )
        # else:
        #     #             self.jacobian_V[:, 1] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
        #     self.jacobian_V[:, 1] = 0


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
        self.control_input_sub = rospy.Subscriber("/cmd_vel", Twist, self.controlInputCallback)
                                                  
        ### publishers ###
        self.pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=1) # queue_size=1 => only the newest map available
        
        ### initialize KF class ###
        # could be initialized in first run of ground_truth callback
        # now it should be  -x 0 -y 0 -z 0, see line 31 in scitos.launch
        initial_pose = np.zeros((3, 1))
        self.kalman_filter = KalmanFilter(self.dt, initial_pose)

        ### initialization of class variables ###
        self.robot_pose = None
        self.odom_msg = None
        self.ground_truth_msg = None
        self.control_input = np.zeros((2, 1)) # [v, w]'

    def run(self):
        """
        Main loop of class.
        @param: self
        @result: runs the step function for the predicton and update steps.
        """
        while not rospy.is_shutdown():
            ### step only when odometry are available ###
            if self.odom_msg:
                self.step()
            self.rate.sleep()

    def step(self):
        """
        Perform an iteration of the localization loop.
        @param: self
        @result: performs the predicton and update steps.
        """
        pose = self.kalman_filter.predict(self.control_input)
        pass

    def odometryCallback(self, data):
        """
        Handles incoming Odometry messages and performs a
        partial quaternion to euler angle transformation to get the yaw angle theta
        @param: pose data stored in the odometry message
        @result: global variables self.robot_pose containing the planar
                 coordinates (x,y)) and self.robot_yaw containing the yaw angle theta
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

    def groundTruthCallback(self, data):
        """
        Handles incoming groud truth messages
        @param: information from Gazebo
        @result: internal update of ground truth
        """
        self.ground_truth_msg = data
        
    def controlInputCallback(self, data):
        """
        gets twist message from teleop_key.py
        @param: Twist message
        @result: control input ndarray 2 x 1
        """
        self.control_input = np.array([[data.linear.x],
                                       [data.angular.z]]) # [v,w]'

if __name__ == '__main__':
    # initialize node and name it
    rospy.init_node("LocalizationNode") # should this be "LocalizationNode" right ? I changed it
    # go to class that provides all the functionality
    # and check for errors
    try:
        localization = Localization(10)
        localization.run()
    except rospy.ROSInterruptException:
        pass
