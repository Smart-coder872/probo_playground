"""
An abstract base class that all sensor classes must inherit from.
This structure guarantees that all sensors have certain traits,
including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants.
Different sensors may use different distributions to model noise,
and may take in different parameters to shape that noise.
For example, one sensor might have a constant noise mean,
while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world.
This includes GPS, cameras, LiDAR, and anything else that takes a measurement
that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states.
This includes IMUs, wheel encoders, and anything else that measures
how the robot's state is relatively changing, without relating the robot to the world.
"""

from abc import ABC, abstractmethod
from math import pi

import numpy as np
import pandas as pd
import sympy as sp

from sympy.abc import x, y, k, j, theta
from sympy import symbols, Matrix, Symbol, pprint, matrix2numpy
from numpy import sqrt, arctan2, ndarray
from random import gauss


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class WheelEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        linear_noise_ratio=0.01,
        angular_noise_ratio=0.01,
        name="wheel_encoder",
        interval=0.1,
        lin_noise=0.05,
        ang_noise=0.03
        
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        # TODO: (done) save all noise constants as properties
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s
        self.LIN_NOISE_RATIO = linear_noise_ratio
        self.ANG_NOISE_RATIO = angular_noise_ratio


    def sample(self):
        """
        Sample the robot's linear and angular velocity.
        """
        # TODO: (done) fill in the function
        actual_lin_vel = self.robot.LIN_VEL
        actual_ang_vel = self.robot.ANG_VEL
        
        lin_sample = gauss(actual_lin_vel, self.LIN_NOISE + abs(actual_lin_vel) * self.LIN_NOISE_RATIO)
        ang_sample = gauss(actual_ang_vel, self.ANG_NOISE + abs(actual_ang_vel) * self.ANG_NOISE_RATIO)

        return pd.DataFrame(
            {
                f"{self.name}_LinearVelocity": [lin_sample],
                f"{self.name}_AngularVelocity": [ang_sample],
            }
        )
        
class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing
    between the robot and the floating-point landmarks on the map.
    In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        MAX_RANGE (int): maximum distance from a beacon for it to be visible
        RANGE_NOISE (float): absolute noise for range stdev
        RANGE_NOISE_RATIO (float): porportional noise for range stdev
        BEARING_NOISE (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=pi / 6,
        max_range=10.0,
        range = 0.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        # TODO: (done) save max range and all noise constants as properties
        self.MAX_RANGE = max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians
        
        # TODO: (done) define the nonlinear measurement model symbolically
        x_range = j - x
        y_range = k - y
        range = sp.sqrt(x_range**2 + y_range**2)
        self.RANGE = range

        total_angle = sp.atan2(y_range, x_range)
        bearing = total_angle - theta

        self.h_x: Matrix = Matrix(
            [
                [range],  # calculation of r (range)
                [bearing]  # calculation of phi (bearing)
            ]
        )

        # TODO: (done) define the Jacobian of h(x) symbolically
        self.H: Matrix = self.h_x.jacobian(Matrix([x, y, theta]))

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }


    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + z[0] * self.RANGE_PROP_NOISE
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, x, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector
        to be in the observation space. This matrix is used to turn a state prediction
        into an observation prediction for a specific landmark.

        Args:
            x: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        # TODO: find the x and y position of the given landmark
        for lm in self.robot.env.LANDMARKS:
            if lm_id == self.robot.env.LANDMARKS.id:
                lm_x = self.robot.env.LANDMARKS.pos[0]
                lm_y = self.robot.env.LANDMARKS.pos[1]

        # TODO: (done) set the value of each symbolic substitution to the actual numerical value that was passed in
                self.subs[x] = x[0]
                self.subs[y] = x[1]
                self.subs[theta] = x[2]
                self.subs[j] = lm_x  # note: we use j for landmark x position
                self.subs[k] = lm_y  # note: we use k for landmark y position

        # TODO: evaluate the Jacobian at the subs values and convert it to a numpy array
        H_eval = matrix2numpy(self.H)

        # return
        return H_eval

    def y(self, z, x, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation
        derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        # TODO: find the x and y position of the given landmark
        for lm in self.robot.env.LANDMARKS:
            if lm.id == lm_id:
                lm_x = self.robot.env.LANDMARKS.pos[0]
                lm_y = self.robot.env.LANDMARKS.pos[1]

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = x[0]
        self.subs[y] = x[1]
        self.subs[theta] = x[2]
        self.subs[j] = lm_x  # note: we use j for landmark x position
        self.subs[k] = lm_y  # note: we use k for landmark y position

        # TODO: evaluate the measurement model at the subs values and convert it to a numpy array
        hx_eval = matrix2numpy(self.H)

        # TODO: calculate the residual
        y = z - hx_eval * x

        # return
        return y
    
    def sample(self):
        """
        Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
        """
        # TODO: (done) fill in the function

        if self.RANGE <= self.MAX_RANGE:
            noisy_range = gauss(self.RANGE, self.RANGE_NOISE + self.RANGE * self.RANGE_PROP_NOISE)
            noisy_bearing = gauss(self.BEARING, self.BEARING_NOISE + self.BEARING)

        
        return pd.DataFrame(
            {
                f"{self.name}_Range w/Noise": [noisy_range],
                f"{self.name}_Bearing w/Noise": [noisy_bearing],
            }
        )

class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot in 2D space.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        X_NOISE (float): absolute noise for x stdev
        Y_NOISE (float): absolute noise for y stdev
    """

    def __init__(
        self,
        robot,
        name = "GPS",
        interval = 2,
        x_noise= 0.2,
        y_noise = 0.2,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            x_noise (float): absolute noise for x stdev
            y_noise (float): absolute noise for y stdev
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # TODO: fill in the measurement model
        self.H = np.eye(2)

        # TODO: (done) fill in the noise model
        self.R = np.diag([self.X_NOISE, self.Y_NOISE]) ** 2

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        """
        # TODO: fill in the function
        noisy_x = gauss(self.robot.env.robot_pose.pos.x, self.X_NOISE)
        noisy_y = gauss(self.robot.env.robot_pose.pos.y, self.Y_NOISE)
        
        return pd.DataFrame(
            {
                f"{self.name}_X Pos w/Noise": [noisy_x],
                f"{self.name}_Bearing w/Noise": [noisy_y],
            }
        )
