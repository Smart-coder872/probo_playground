"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
from numpy import ndarray
import sympy
from sympy.abc import x, y, v, w, R, theta
from sympy import Matrix, Symbol, Identity, Inverse, matrix2numpy, cos, sin
import random

from utils import wrap_angle


class IteratedEKF:
    """
    This class implements the Extended Kalman Filter algorithm.
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an Extended Kalman Filter.

        A state vector includes the following:
            robot a x position
            robot a y position
            robot a heading
            robot a_b bearing
            robot a_b range


        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable-
        """
        # TODO: (done) set the timestep size to the given parameter
        self.DT:float = dt

        # TODO: (done) set the state vector to the given prior
        self.x = prior

        # TODO: (done) set the process model to an identity matrix
        self.P = np.eye(3)

        # TODO: (done) define the nonlinear state transition model
        
        self.f_xu: Matrix = Matrix(
            [
                [x + v*cos(theta)*self.DT],  # calculation of x
                [y + v*sin(theta)*self.DT],  # calculation of y
                [theta + w*self.DT],  # calculation of theta
            ]
        )

        # TODO: (done) define the Jacobian of the motion model symbolically
        self.F: Matrix = self.f_xu.jacobian([x, y, theta])

        self.B = self.DT * np.eye(3)

        # dictionary that maps Sympy symbols to numerical values. we will use these to substitute values into our symbolic matrices!
        self.subs: dict[Symbol, float] = {
            x: self.x[0],
            y: self.x[1],
            theta: self.x[2],
            v: 0,
            w: 0,
        } 

    def predict(self, u: np.array):
        """
        Predicts the next state vector and its covariance matrix
        using the state transition matrix and an input control vector.
        The Kalman Filter uses the following predict equations:

        x_t+1 = f(x,u)
        P_t+1 = F * P * F.T + Q

        where F is the Jacobian of f(x,u)

        
        Args:
            u: the input control vector
        """
        # TODO: (done) set the value of each symbolic substitution to the actual numerical value being tracked by the EKF
        self.subs[x] = self.x[0].item()
        self.subs[y] = self.x[1].item()
        self.subs[theta] = self.x[2].item()
        self.subs[v] = u[0].item()
        self.subs[w] = u[1].item()

        # TODO: (done) evaluate the nonlinear motion model f(x,u) at the subsitution values
        fxu_eval = matrix2numpy(self.f_xu.subs(self.subs))

        # TODO: (done) evaluate the Jacobian matrix F at the substitution values
        F_eval = matrix2numpy(self.F.subs(self.subs))

        # TODO: (done) calculate the next state prediction
        self.x = fxu_eval

        # TODO: (done) calculate the next covariance prediction
        self.P = F_eval@self.P@F_eval.T + self.get_Q()

        # return state vector and state covariance
        return self.x, self.P
    

    def relative_update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray | None,
        y: np.ndarray | None,
    ):
        """
        Updates the current state prediction using observations
        from the environment. The Extended Kalman Filter uses the
        following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        
        y = z - h(x)
        (residual: error between observation
        and expected observation given estimated state vector)
        
        K = P * H.T * inv(S)
        (Kalman Gain: portion of total uncertainty
        that is from the prediction)
        
        S = H * P * H.T + R
        (total uncertainty in the system)

        where H is the Jacobian of h(x)

        Args:
            H: the Jacobian of the nonlinear measurement model,
            which relates the state space to the measurement space
            R: the measurement noise model (covariance)
            y: the residual, which is the error between the measured observation
            and the observation expected by the predicted state
        """
        # TODO: (done) calculate the total uncertainty in the system
        for i in range(5):
            S = H @ self.P @ H.T + R

            # TODO: (done) calculate the Kalman Gain
            K = self.P @ H.T @ matrix2numpy(np.linalg.inv(S))

            if y is None:
                y = z - H @ self.x

            # TODO: (done) update state vector
            self.x += K @ y

            # TODO: (done) update process model
            self.P -= K @ H.T @ self.P

            # return state vector and process model
            return self.x, self.P
    

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        # TODO: explore different standard deviation values for this function!
        stdev = 0.1
        return np.array(
            [
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
            ]
        )
