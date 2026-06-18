"""
Ideation for main to implement local and relative ekf
"""

from environment import Environment
from robot import Robot
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from utils import Position, Pose, Landmark, Bounds
import pandas as pd
import numpy as np
import csv

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = Bounds(x_min=0, x_max=10, y_min=0, y_max=10)    #vertical length or horizontal length in meters
    dt = 0.1                    #0.1 seconds per step
    obstacles = [
    Bounds(x_min=5, x_max=7, y_min=5, y_max=7),
    Bounds(x_min=0, x_max=4, y_min=6, y_max=8)
    ]
    landmarks = None

    robot_a_starting_pose = Pose(Position(0.0, 0.0), 0.6)  #x, y and theta
    robot_b_starting_pose = Pose(Position(3.0, 3.0), 0.6)
    
    bearing_a = robot_a_starting_pose.theta
    bearing_b = robot_b_starting_pose.theta

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        robot_a_starting_pose,
        robot_b_starting_pose,
        bearing_a,
        bearing_b
    )

    # set up the robot
    robot_a = Robot(env)
    robot_b = Robot(env)

    a_starting_states = np.array([robot_a_starting_pose.pos.x, robot_a_starting_pose.pos.y, robot_a_starting_pose.theta]).reshape(3,1)
    b_starting_states = np.array([robot_b_starting_pose.pos.x, robot_b_starting_pose.pos.y, robot_b_starting_pose.theta]).reshape(3,1)


    # set up the (Extended) Kalman Filter if bool LINEAR is True
    if robot_a:      #[DOUBLE CHECK]
        kf = ExtendedKalmanFilter(
            float(dt),
            a_starting_states,
        )

    elif robot_b:      #[DOUBLE CHECK]
        kf = ExtendedKalmanFilter(
            float(dt),
            b_starting_states,
        )

    else:
        f"Robot not recognized as a or b"
    # set up timekeeping
    # TODO: (done) set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = 30                          #total run time
    total_timesteps = total_seconds / env.DT    #calculate time step

    # set up logging
    a_ground_truth_history = []
    a_sensor_data_history = []
    a_kalman_filter_history = []
    a_relative_history = []

    b_ground_truth_history = []
    b_sensor_data_history = []
    b_kalman_filter_history = []
    b_relative_history = []

    # set up input filepath and output filepaths
    a_input_commands_filepath = "./input/a_motor_commands.csv" #[NEED TO CREATE FILE]
    a_output_ground_truth_filepath = "./output/ground_truth_a.csv"
    a_output_sensor_data_filepath = "./output/sensor_data_a.csv"
    a_output_kalman_filter_filepath = "./output/kalman_filter_a.csv"
    a_output_relative_filepath = "./output/relative_filter_a.csv"

    b_input_commands_filepath = "./input/b_motor_commands.csv" #[NEED TO CREATE FILE]
    b_output_ground_truth_filepath = "./output/ground_truth_b.csv"
    b_output_sensor_data_filepath = "./output/sensor_data_b.csv"
    b_output_kalman_filter_filepath = "./output/kalman_filter_b.csv"
    a_output_relative_filepath = "./output/relative_filter_b.csv"


    # open up the instructions for robot a, pop the first
    with open(a_input_commands_filepath, "r") as cmd:
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # TODO: (done) take a ground truth snapshot and add it to the history
            a_ground_truth_history.append(env.take_state_snapshot(0))
            # TODO: (done) take sensor measurements and add it to the history
            a_sensor_data_history.append(robot_a.take_sensor_measurements())
            # TODO: (done) call the Kalman Filter prediction step            
            kf.predict(robot_a.take_sensor_measurements())
                # TODO: (done) call the Kalman Filter update step if new sensor data is available
            z = np.array((env.get_proximity_to_landmarks()[1], env.get_proximity_to_landmarks()[2]))
            id = env.get_proximity_to_landmarks()[0]
            R = robot.sensors["LandmarkPinger"].R(z)
            H =  robot.sensors["LandmarkPinger"].H_eval(starting_states, id)
            
            local_data = kf.local_update(z, H, R)
            a_kalman_filter_history.append(local_data)
            
            rel_data = kf.relative_update()
            a_relative_history = a_relative_history.append(rel_data)

            # TODO: retrieve the next motor command from the input file
            for data_point in a_input_commands_filepath:
                if env.time < 5:
                    linear_input = data_point[1]
                    angular_input = data_point[2]

                
                elif env.time >= 6 and env.time < 8:
                    linear_input = data_point[4]
                    angular_input = data_point[5]

                elif env.time >= 8 and env.time < 15:
                    linear_input = data_point[7]
                    angular_input = data_point[8]
                else:
                    linear_input = data_point[10]
                    angular_input = data_point[11]


            # TODO: execute the motor command
            robot_a.robot_step_differential(linear_input, angular_input)
    
    # at the end, write the histories into output files
    with open(a_output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file
        gt_data.write(a_ground_truth_history)
    with open(a_output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
        sensor_data.write(a_sensor_data_history)
    with open(a_output_kalman_filter_filepath, "w") as kalman_data:
        kalman_data.write(a_kalman_filter_history)
    with open(a_output_relative_filepath, "w") as relative_data:
        relative_data.write(a_relative_history)

    