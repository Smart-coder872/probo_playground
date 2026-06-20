"""
Robot class for the simulation environment.
"""

from sensors import WheelEncoder, LandmarkPinger, GPS, RobotPinger
from math import cos, sin


class Robot:
    """
    A class that represents a mobile robot operating in 2D.

    Attributes:
        env: reference to the simulation environment
        robot_id: identifier for this robot (0 for robot_a, 1 for robot_b)
        LIN_VEL: current linear velocity (m/s)
        ANG_VEL: current angular velocity (rad/s)
        sensors: dictionary of sensor instances attached to this robot
    """

    def __init__(self, env, robot_id: int, other_id: int, use_robot_pinger: bool = True):
        """
        Initialize a Robot instance.

        Args:
            env: reference to the Environment
            robot_id: identifier for this robot (0 or 1)
            use_robot_pinger: whether to include the RobotPinger sensor
        """
        self.env = env
        self.ROBOT_ID = robot_id  # Track which robot this is
        self.OTHER_ID = other_id
        self.LIN_VEL = 0.0  # m/s
        self.ANG_VEL = 0.0  # rad/s

        # Initialize sensors for this robot
        self.sensors = {
            "WheelEncoder": WheelEncoder(self),
            "LandmarkPinger": LandmarkPinger(self),
            "GPS": GPS(self),
        }

        # Add RobotPinger if multiple robots in environment
        if use_robot_pinger and len(env.robot_poses) > 1:
            self.sensors["RobotPinger"] = RobotPinger(self)

    def take_sensor_measurements(self):
        """
        Take measurements from all sensors on this robot.
        
        Returns:
            Dictionary of sensor measurements
        """
        measurements = {}
        for sensor_name, sensor in self.sensors.items():
            sample = sensor.sample()
            sample["sensor"] = sensor_name
            sample["robot_id"] = self.ROBOT_ID
            sample["timestamp"] = self.env.time
            measurements[sensor_name] = sample
        return measurements

    def robot_step_differential(self, linear_vel: float, angular_vel: float):
        """
        Execute a single timestep with differential drive kinematics.

        Args:
            linear_vel: desired linear velocity (m/s)
            angular_vel: desired angular velocity (rad/s)
        """
        self.LIN_VEL = linear_vel
        self.ANG_VEL = angular_vel

        # Differential drive kinematics
        dx = linear_vel * cos(self.env.robot_poses[self.ROBOT_ID].theta) * self.env.DT
        dy = linear_vel * sin(self.env.robot_poses[self.ROBOT_ID].theta) * self.env.DT
        dtheta = angular_vel * self.env.DT

        # Update robot position in environment
        self.env.is_valid_motion(dx, dy, dtheta, self.ROBOT_ID)

    def robot_step_unicycle(self, linear_vel: float, angular_vel: float):
        """
        Execute a single timestep with unicycle kinematics.
        Alternative to differential drive.

        Args:
            linear_vel: desired linear velocity (m/s)
            angular_vel: desired angular velocity (rad/s)
        """
        self.LIN_VEL = linear_vel
        self.ANG_VEL = angular_vel

        # Unicycle kinematics
        L = 0.1  # wheelbase (meters) - adjust as needed
        dx = linear_vel * cos(self.env.robot_poses[self.ROBOT_ID].theta) * self.env.DT
        dy = linear_vel * sin(self.env.robot_poses[self.ROBOT_ID].theta) * self.env.DT
        dtheta = (linear_vel / L) * angular_vel * self.env.DT

        # Update robot position in environment
        self.env.is_valid_motion(dx, dy, dtheta, self.ROBOT_ID)

    def get_estimated_state(self, filter):
        """
        Get state estimate from a Kalman filter.

        Args:
            filter: KalmanFilter or IteratedEKF instance

        Returns:
            State vector [x, y, theta]
        """
        return filter.x_state_ef

    def get_other_robots(self):
        """
        Get information about other robots in the environment.
        Useful for decentralized swarm coordination.

        Returns:
            List of other robot IDs and their poses
        """
        other_robots = []
        for other_id, pose in enumerate(self.env.robot_poses):
            if other_id != self.ROBOT_ID:
                other_robots.append((other_id, pose))
        return other_robots

    def sense_other_robots(self):
        """
        Get sensor measurements of other robots via RobotPinger.
        
        Returns:
            Sensor measurements of other robots, or None if no RobotPinger
        """
        if "RobotPinger" in self.sensors:
            return self.sensors["RobotPinger"].sample()
        return None

    def sense_landmarks(self):
        """
        Get sensor measurements of landmarks via LandmarkPinger.
        
        Returns:
            Sensor measurements of landmarks
        """
        if "LandmarkPinger" in self.sensors:
            return self.sensors["LandmarkPinger"].sample()
        return None

    def sense_velocity(self):
        """
        Get sensor measurements of velocities via WheelEncoder.
        
        Returns:
            Sensor measurements of linear and angular velocities
        """
        if "WheelEncoder" in self.sensors:
            return self.sensors["WheelEncoder"].sample()
        return None

    def sense_position(self):
        """
        Get sensor measurements of position via GPS.
        
        Returns:
            Sensor measurements of position
        """
        if "GPS" in self.sensors:
            return self.sensors["GPS"].sample()
        return None