# def computeJacobian(control_input):
#     """
#     This method computes the Jacobians of the kinematic model.
#     @param: control_input - numpy array of dim 2 x 1 containing:
#         *linear speed w.r.t. x-axis in robot frame, v
#         *angular speed w.r.t. z-axis in robot frame, w
#     @result:
#         *self.jacobian_G: Jacobian with respect to the state estimate
#         *self.jacobian_V: Jacobian with respect to the control inputs
#     """
#
#     delta_g = next_state_mu - last_state_mu
#     delta_x = next_state_mu[0, 0] - last_state_mu[0, 0]
#     delta_y = next_state_mu[1, 0] - last_state_mu[1, 0]
#     delta_psi = next_state_mu[2, 0] - last_state_mu[2, 0]
#
#     print("delta_g:", delta_g)
#     print("delta_x:", delta_x)
#     print("delta_y:", delta_y)
#     print("delta_psi:", delta_psi)
#     # we should make sure we don't divide by zero
#     if delta_x.all() > threshold_div_zero:
#         jacobian_G[:, 0] = np.array(delta_g / delta_x).reshape(3, )
#     else:
#         jacobian_G[:, 0] = np.array(delta_g / threshold_div_zero).reshape(3, )
#
#     if delta_y.all() > threshold_div_zero:
#         jacobian_G[:, 1] = np.array(delta_g / delta_y).reshape(3, )
#     else:
#         jacobian_G[:, 1] = np.array(delta_g / threshold_div_zero).reshape(3, )
#
#     if delta_psi.all() > threshold_div_zero:
#         jacobian_G[:, 2] = np.array(delta_g / delta_psi).reshape(3, )
#     else:
#         jacobian_G[:, 2] = np.array(delta_g / threshold_div_zero).reshape(3, )
#
#     delta_v = control_input[0] - last_control_input[0]
#     delta_w = control_input[1] - last_control_input[1]
#
#     if delta_v.all() > threshold_div_zero:
#         jacobian_V[:, 0] = np.array(delta_g / delta_v).reshape(3, )
#     else:
#         jacobian_V[:, 0] = np.array(delta_g / threshold_div_zero).reshape(3, )
#
#     if delta_w.all() > threshold_div_zero:
#         jacobian_V[:, 1] = np.array(delta_g / delta_w).reshape(3, )
#     else:
#         jacobian_V[:, 1] = np.array(delta_g / threshold_div_zero).reshape(3, )

import numpy as np

class KalmanFilter:
    """
    Class called by the main node and which implements the Kalman Filter
    """

    def __init__(self, dt, initial_pose):
        """
        Function that ...
        @param: TBD
        @result: TBD
        """
        ### class arguments
        self.dt = dt
        self.motion_model = MotionModel(self.dt)
        # self.odom_error_model = self.motion_model.error_model

        # TO DO: needs to be initialized with a value
        # coming from ground truth
        # self.last_state_mu = np.zeros((3, 1))
        self.last_state_mu = initial_pose

        # covariance on initial position is null beccause pose comes from ground truth
        self.prior_last_covariance = np.zeros((3, 3))
        # robot doesn't move at t = 0
        self.last_control_input = np.zeros((2, 1))

        # initialization
        self.jacobian_G = np.zeros((3, 3))
        self.jacobian_V = np.zeros((3, 2))
        self.threshold_div_zero = 1e-6
        self.next_state_mu = np.zeros((3, 1))

    def predict(self, control_input):
        """
        This method predicts what the next system state will be.
        @param: control_input - numpy array of dim 2 x 1 containing:
            *linear speed w.r.t. x-axis in robot frame, v
            *angular speed w.r.t. z-axis in robot frame, w
        @result: the method returns:
            *next_state - numpy array of dim 3 x 1 containing the 3 tracked variables (x,y,psi)
            *next_covariance - numpy array of dim 3 x 3 containing the covariance matrix
        """

        # compute the next state i.e. next robot pose knowing current control inputs
        self.next_state_mu, next_error = self.motion_model.predictPose(control_input, self.last_state_mu)
        print("last_state_mu:", self.last_state_mu)
        print("next_state_mu:", self.next_state_mu)
        # compute the jacobians necessary for the EKF prediction
        self.computeJacobian(control_input)

        # compute covariance on the state transition probability
        self.next_state_covariance_R = self.jacobian_V @ next_error @ self.jacobian_V.T

        ### to be continued

        # store current state estimate, current covariance on prior belief, current control inputs
        # for use in the next iteration
        self.last_state_mu = self.next_state_mu

        self.prior_last_covariance = \
            self.jacobian_G @ self.last_covariance @ self.jacobian_G.T + self.next_state_covariance_R

        self.last_control_input = control_input

        return self.next_state_mu, self.next_covariance

    def computeJacobian(self, control_input):
        """
        This method computes the Jacobians of the kinematic model.
        @param: control_input - numpy array of dim 2 x 1 containing:
            *linear speed w.r.t. x-axis in robot frame, v
            *angular speed w.r.t. z-axis in robot frame, w
        @result:
            *self.jacobian_G: Jacobian with respect to the state estimate
            *self.jacobian_V: Jacobian with respect to the control inputs
        """

        delta_g = self.next_state_mu - self.last_state_mu
        delta_x = self.next_state_mu[0, 0] - self.last_state_mu[0, 0]
        delta_y = self.next_state_mu[1, 0] - self.last_state_mu[1, 0]
        delta_psi = self.next_state_mu[2, 0] - self.last_state_mu[2, 0]

        print("delta_g:", delta_g)
        print("delta_x:", delta_x)
        print("delta_y:", delta_y)
        print("delta_psi:", delta_psi)
        # we should make sure we don't divide by zero
        if delta_x.all() > self.threshold_div_zero:
            self.jacobian_G[:, 0] = np.array(delta_g / delta_x).reshape(3, )
        else:
            self.jacobian_G[:, 0] = np.array(delta_g / self.threshold_div_zero).reshape(3, )

        if delta_y.all() > self.threshold_div_zero:
            self.jacobian_G[:, 1] = np.array(delta_g / delta_y).reshape(3, )
        else:
            self.jacobian_G[:, 1] = np.array(delta_g / self.threshold_div_zero).reshape(3, )

        if delta_psi.all() > self.threshold_div_zero:
            self.jacobian_G[:, 2] = np.array(delta_g / delta_psi).reshape(3, )
        else:
            self.jacobian_G[:, 2] = np.array(delta_g / self.threshold_div_zero).reshape(3, )

        delta_v = control_input[0] - self.last_control_input[0]
        delta_w = control_input[1] - self.last_control_input[1]

        if delta_v.all() > self.threshold_div_zero:
            self.jacobian_V[:, 0] = np.array(delta_g / delta_v).reshape(3, )
        else:
            self.jacobian_V[:, 0] = np.array(delta_g / self.threshold_div_zero).reshape(3, )

        if delta_w.all() > self.threshold_div_zero:
            self.jacobian_V[:, 1] = np.array(delta_g / delta_w).reshape(3, )
        else:
            self.jacobian_V[:, 1] = np.array(delta_g / self.threshold_div_zero).reshape(3, )
