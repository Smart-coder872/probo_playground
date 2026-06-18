"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world.
The robot is remote-controlled by angular and linear velocity commands
read from an external file. The robot can execute motor commands to move,
and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

from environment import Environment
from sensors import SensorInterface, WheelEncoder, LandmarkPinger, GPS
from numpy import cos, sin
import pandas as pd
from pandas import DataFrame


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment, sensor_info=dict):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        # TODO: (done) set the environment property to the parameter value
        self.env = env
        self.LIN_VEL = 0.0
        self.ANG_VEL = 0.0

        # TODO: (done)initialize the sensors property as an empty list
        gps_info = sensor_info["GPS"]

        self.sensors: dict[str, SensorInterface]= {
            "GPS": GPS(
                #initialize GPS class
                robot=self
            ),
            "LandmarkPinger": LandmarkPinger(
                #initialize Landmark Pinger class
                robot=self
            ),
            "WheelEncoder": WheelEncoder(
                #initialize WheelEncoder class
                robot = self
            )
        }

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities,
        determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        # TODO: (done) fill in the function
        self.LIN_VEL = lin_vel
        self.ANG_VEL = ang_vel
       
        dx = lin_vel * cos(self.env.robot_pose.theta) * self.env.DT #converts linear velocity input to dy
        dy = lin_vel * sin(self.env.robot_pose.theta) * self.env.DT #converts linear velocity input to dx
        dtheta = ang_vel * self.env.DT                              #converts angular velocity input to dtheta


        return self.env.is_valid_motion(dx, dy, dtheta)

    def robot_step_translational(self, x_vel: float, y_vel: float):
        """
        Swerve-drive mode. Given x and y velocities,
        determine the robot's change in x, y, and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command

        Returns:
            dx: change in x position
            dy: change in y position
        """
        # TODO: (done) fill in the function
        self.X_VEL = x_vel
        self.Y_VEL = y_vel
        
        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT

        return self.env.is_valid_motion(dx, dy, 0.0)

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        # TODO: fill in the function
        wheelencoder_measurements = self.sensors["WheelEncoder"].sample()
        gps_measurements =self.sensors["GPS"].sample()
        lmpinger_measurements = self.sensors["LandmarkPinger"].sample()
        
        sensor_measurements = pd.concat(
            [wheelencoder_measurements,
             gps_measurements,
             lmpinger_measurements
            ], axis=1
        )
        return sensor_measurements
