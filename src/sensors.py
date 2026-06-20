"""
Sensor classes for robot perception.

An abstract base class that all sensor classes must inherit from.
This structure guarantees that all sensors have certain traits,
including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants.
Different sensors may use different distributions to model noise,
and may take in different parameters to shape that noise.

Exteroceptive sensors measure the robot's relationship to the world.
This includes GPS, cameras, LiDAR, and anything else that takes a measurement
that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states.
This includes IMUs, wheel encoders, and anything else that measures
how the robot's state is relatively changing, without relating the robot to the world.
"""

from abc import ABC, abstractmethod
from math import pi, inf
from utils import BearingRange, Position

import numpy as np
import pandas as pd
import sympy as sp
import math

from sympy.abc import x, y, k, j, theta
from sympy import symbols, Matrix, Symbol
from numpy import sqrt, arctan2, ndarray
from random import gauss


def matrix2numpy(sympy_matrix):
    """Convert Sympy matrix to numpy array"""
    return np.array(sympy_matrix.tolist(), dtype=float)


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float, title: str = None):
        """
        Initialize a sensor class instance.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.title = title or name
        self.robot = robot
        self.ROBOT_ID = robot.ROBOT_ID
        self.OTHER_ID = robot.OTHER_ID
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """Getter for the name property."""
        return self._name

    @property
    def interval(self) -> float:
        """Getter for the interval property."""
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """Getter for the time of last measurement property."""
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """Setter for the time of last measurement property."""
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """Sample the environment and return the noisy measurement(s)."""
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
        LIN_NOISE_RATIO: proportional noise for linear velocity
        ANG_NOISE_RATIO: proportional noise for angular velocity
    """

    def __init__(
        self,
        robot,
        linear_noise_ratio=0.01,
        angular_noise_ratio=0.01,
        name="Wheel_Encoder",
        interval=0.1,
        lin_noise=0.05,
        ang_noise=0.03,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular velocity
            name: reference identifier
            interval: period between measurements
            lin_noise: absolute noise for linear velocity (m/s)
            ang_noise: absolute noise for angular velocity (rad/s)
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s
        self.LIN_NOISE_RATIO = linear_noise_ratio
        self.ANG_NOISE_RATIO = angular_noise_ratio

    def sample(self):
        """
        Sample the robot's linear and angular velocity.
        
        Returns:
            pd.DataFrame with noisy velocity measurements
        """
        actual_lin_vel = self.robot.LIN_VEL
        actual_ang_vel = self.robot.ANG_VEL

        lin_sample = gauss(
            actual_lin_vel,
            self.LIN_NOISE + abs(actual_lin_vel) * self.LIN_NOISE_RATIO,
        )
        ang_sample = gauss(
            actual_ang_vel,
            self.ANG_NOISE + abs(actual_ang_vel) * self.ANG_NOISE_RATIO,
        )

        return pd.DataFrame({
            "sensor": [self._name],
            "title": [self.title],
            "robot_id": [self.ROBOT_ID],
            "linear_velocity": [lin_sample],
            "angular_velocity": [ang_sample],
        })


class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing
    between the robot and floating-point landmarks on the map.
    In practice, this could be a ToF sensor, a beacon network node, or a camera.

    Attributes:
        name: reference identifier
        robot: reference robot
        interval: period between measurements
        MAX_RANGE: maximum distance for landmark visibility (meters)
        RANGE_NOISE: absolute noise for range stdev (meters)
        RANGE_PROP_NOISE: proportional noise for range stdev
        BEARING_NOISE: absolute noise for bearing stdev (radians)
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=np.pi / 6,
        max_range=10.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            range_noise: absolute noise for range (meters)
            range_prop_noise: proportional noise for range
            bearing_noise: absolute noise for bearing (radians)
            max_range: maximum sensing range (meters)
        """
        super().__init__(name, robot, interval)
        self.MAX_RANGE = max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians

        # Define the nonlinear measurement model symbolically
        x_range = j - x
        y_range = k - y
        range_sym = sp.sqrt(x_range**2 + y_range**2)
        self.RANGE = range_sym

        total_angle = sp.atan2(y_range, x_range)
        bearing = total_angle - theta
        self.BEARING = bearing

        self.h_x = Matrix(
            [
                [range_sym],  # range (r)
                [bearing],  # bearing (phi)
            ]
        )

        # Define the Jacobian of h(x) symbolically
        self.H = self.h_x.jacobian(Matrix([x, y, theta]))

        self.subs = {
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
            z: measurement vector [[range], [bearing]]

        Returns:
            Tuple of (range_variance, bearing_variance)
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + self.RANGE_PROP_NOISE * z[0][0]
        return range_stdev**2, bearing_stdev**2

    def H_eval(self, state_vector, lm_id):
        """
        Evaluate the Jacobian of h(x) at the given state.

        Args:
            state_vector: current state [x, y, theta]
            lm_id: ID of the landmark

        Returns:
            Evaluated Jacobian matrix as numpy array
        """
        # Find the landmark position
        landmark_x = None
        landmark_y = None
        
        for lm in self.robot.env.LANDMARKS:
            if lm.id == lm_id:
                landmark_x = lm.pos.x
                landmark_y = lm.pos.y
                break

        if landmark_x is None:
            raise ValueError(f"Landmark {lm_id} not found in environment")

        # Update substitution dictionary with current values
        self.subs[x] = state_vector[0]
        self.subs[y] = state_vector[1]
        self.subs[theta] = state_vector[2]
        self.subs[j] = landmark_x  # landmark x position
        self.subs[k] = landmark_y  # landmark y position

        # Substitute values into Jacobian
        H_eval = matrix2numpy(self.H.subs(self.subs))

        return H_eval

    def y(self, z, state_vector, lm_id):
        """
        Calculate the residual between observation and prediction.

        Args:
            z: measured observation [[range], [bearing]]
            state_vector: current state [x, y, theta]
            lm_id: landmark ID

        Returns:
            Residual vector
        """
        # Find the landmark position
        landmark_x = None
        landmark_y = None
        
        for lm in self.robot.env.LANDMARKS:
            if lm.id == lm_id:
                landmark_x = lm.pos.x
                landmark_y = lm.pos.y
                break

        if landmark_x is None:
            raise ValueError(f"Landmark {lm_id} not found in environment")

        # Update substitution dictionary
        self.subs[x] = state_vector[0]
        self.subs[y] = state_vector[1]
        self.subs[theta] = state_vector[2]
        self.subs[j] = landmark_x
        self.subs[k] = landmark_y

        # Use h_x (measurement model)
        hx_eval = matrix2numpy(self.h_x.subs(self.subs)).flatten()

        # Calculate residual
        residual = z.flatten() - hx_eval

        return residual

    def sample(self):
        """
        Reports noisy measurements of range and bearing to nearby landmarks.
        
        Returns:
            pd.DataFrame with landmark pinger measurements
        """
        noisy_measurements = []
        
        # Get proximity data for this robot
        gt_data = self.robot.env.get_proximity_to_robot(self.robot.ROBOT_ID, self.robot.OTHER_ID)
        
        if gt_data is None:
            return pd.DataFrame({"LM_Pinger": []})

        target_robot = gt_data[0]
        bearing = gt_data[1]
        range_val = gt_data[2]

        if range_val <= self.MAX_RANGE:
            noisy_range = gauss(
                range_val,
                self.RANGE_NOISE + range_val * self.RANGE_PROP_NOISE,
            )
            noisy_bearing = gauss(bearing, self.BEARING_NOISE)
            noisy_measurements.append({
                "sensor": [self._name],
                "title": [self.title],
                "robot_id": [self.robot.ROBOT_ID],
                "other_robot_id": [self.robot.OTHER_ID],
                "bearing": [noisy_bearing],
                "range": [noisy_range],
            })
        else:
            # Out of range
            noisy_measurements.append({
                "sensor": [self._name],
                "title": [self.title],
                "robot_id": [self.robot.ROBOT_ID],
                "other_robot_id": [self.robot.OTHER_ID],
                "bearing": [float("inf")],
                "range": [float("inf")],
            })

        return pd.concat([pd.DataFrame(row) for row in noisy_measurements], ignore_index=True)


"""
CORRECTED RobotPinger - Matching LandmarkPinger structure
"""

class RobotPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing
    to another robot in the swarm. Enables decentralized awareness of
    relative positions between robots.

    Attributes:
        name: reference identifier
        robot: reference robot (this robot)
        interval: period between measurements
        MAX_RANGE: maximum detection range (meters)
        RANGE_NOISE: absolute noise for range stdev (meters)
        RANGE_PROP_NOISE: proportional noise for range stdev
        BEARING_NOISE: absolute noise for bearing stdev (radians)
    """

    def __init__(
        self,
        robot,
        name="robot_pinger",
        interval=0.5,
        range_noise=0.3,
        range_prop_noise=0.03,
        bearing_noise=np.pi / 12,
        max_range=20.0,
    ):
        """
        Initialize an instance of the RobotPinger class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            range_noise: absolute noise for range (meters)
            range_prop_noise: proportional noise for range
            bearing_noise: absolute noise for bearing (radians)
            max_range: maximum sensing range (meters)
        """
        super().__init__(name, robot, interval)
        self.MAX_RANGE = max_range
        self.RANGE_NOISE = range_noise
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise

        # Define the nonlinear measurement model symbolically
        # x_range and y_range are relative positions to other robot
        x_range = j - x  # other robot x - my x
        y_range = k - y  # other robot y - my y
        range_sym = sp.sqrt(x_range**2 + y_range**2)
        self.RANGE = range_sym

        total_angle = sp.atan2(y_range, x_range)
        bearing = total_angle - theta
        self.BEARING = bearing

        self.h_x = Matrix(
            [
                [range_sym],  # range (r)
                [bearing],  # bearing (phi)
            ]
        )

        # Define the Jacobian of h(x) symbolically
        self.H = self.h_x.jacobian(Matrix([x, y, theta]))

        self.subs = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def R(self, z):
        """
        Estimate variance of a given measurement.

        Args:
            z: measurement vector [[range], [bearing]]

        Returns:
            Tuple of (range_variance, bearing_variance)
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + self.RANGE_PROP_NOISE * z[0][0]
        return range_stdev**2, bearing_stdev**2

    def H_eval(self, state_vector, other_robot_id):
        """
        Evaluate the Jacobian of h(x) at the given state.

        Args:
            state_vector: current state [x, y, theta] of this robot
            other_robot_id: ID of the other robot

        Returns:
            Evaluated Jacobian matrix as numpy array
        """
        # Find the other robot's position
        if other_robot_id >= len(self.robot.env.robot_poses):
            raise ValueError(f"Robot {other_robot_id} not found in environment")
        
        other_pose = self.robot.env.robot_poses[other_robot_id]
        other_x = other_pose.pos.x
        other_y = other_pose.pos.y

        # Update substitution dictionary with current values
        self.subs[x] = state_vector[0].item()
        self.subs[y] = state_vector[1].item()
        self.subs[theta] = state_vector[2].item()
        self.subs[j] = other_x  # other robot x position
        self.subs[k] = other_y  # other robot y position

        # Substitute values into Jacobian
        H_eval = matrix2numpy(self.H.subs(self.subs))

        return H_eval

    def y(self, z, state_vector, other_robot_id):
        """
        Calculate the residual between observation and prediction.

        Args:
            z: measured observation [[range], [bearing]]
            state_vector: current state [x, y, theta] of this robot
            other_robot_id: ID of the other robot

        Returns:
            Residual vector
        """
        # Find the other robot's position
        if other_robot_id >= len(self.robot.env.robot_poses):
            raise ValueError(f"Robot {other_robot_id} not found in environment")
        
        other_pose = self.robot.env.robot_poses[other_robot_id]
        other_x = other_pose.pos.x
        other_y = other_pose.pos.y

        # Update substitution dictionary
        self.subs[x] = float(state_vector[0])
        self.subs[y] = float(state_vector[1])
        self.subs[theta] = float(state_vector[2])
        self.subs[j] = other_x
        self.subs[k] = other_y

        # Use h_x (measurement model)
        hx_eval = matrix2numpy(self.h_x.subs(self.subs)).flatten()

        # Calculate residual
        residual = z.flatten() - hx_eval

        return residual

    def sample(self):
        """
        Measure range and bearing to other robots in the environment.
        
        Returns:
            pd.DataFrame with measurements to each other robot (BearingRange objects)
        """
        noisy_measurements = []

        # Identify pinging robot
        my_robot_id = self.robot.ROBOT_ID
        my_pose = self.robot.env.robot_poses[my_robot_id]

        # Identify other robot

        other_robot_id = self.robot.OTHER_ID
        other_pose = self.robot.env.robot_poses[other_robot_id]

        # Calculate ground truth range and bearing
        dx = other_pose.pos.x - my_pose.pos.x
        dy = other_pose.pos.y - my_pose.pos.y

        range_gt = sqrt(dx**2 + dy**2)
        bearing_gt = arctan2(dy, dx) - my_pose.theta

            # Only measure if within range
        if range_gt <= self.MAX_RANGE:
            noisy_range = gauss(
                range_gt,
                self.RANGE_NOISE + range_gt * self.RANGE_PROP_NOISE,
            )
            noisy_bearing = gauss(bearing_gt, self.BEARING_NOISE)
            
            measurement = BearingRange(other_robot_id, noisy_bearing, noisy_range)
            noisy_measurements.append(measurement)
        else:
            # Out of range
            measurement = BearingRange(other_robot_id, inf, inf)
            noisy_measurements.append(measurement)

        # Return DataFrame with BearingRange objects
        return pd.DataFrame({"Robot_Pinger": noisy_measurements})


class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot
    in 2D space.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        X_NOISE: absolute noise for x stdev (meters)
        Y_NOISE: absolute noise for y stdev (meters)
    """

    def __init__(
        self,
        robot,
        name="GPS",
        interval=2,
        x_noise=0.2,
        y_noise=0.2,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            x_noise: absolute noise for x stdev (meters)
            y_noise: absolute noise for y stdev (meters)
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # Measurement model (GPS observes full position)
        self.H = np.eye(2)

        # Noise model (covariance matrix)
        self.R = np.diag([self.X_NOISE, self.Y_NOISE]) ** 2

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        
        Returns:
            pd.DataFrame with noisy position measurements
        """

        pose = self.robot.env.get_robot_pose(self.ROBOT_ID)
        noisy_x = gauss(pose.pos.x, self.X_NOISE)
        noisy_y = gauss(pose.pos.y, self.Y_NOISE)

        return pd.DataFrame({
            "sensor": [self._name],
            "title": [self.title],
            "robot_id": [self.ROBOT_ID],
            "x": [noisy_x],
            "y": [noisy_y],
        })