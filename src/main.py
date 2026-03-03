"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from utils import Position, Pose, Landmark, Bounds
import pandas as pd

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = 10             #vertical length or horizontal length in meters
    dt = 0.1                    #0.1 seconds per step
    obstacles = [
        [5, 7, 5, 7],            # x_min, x_max, y_min, y_max
        [0, 4, 6, 8]]           # additional obstacle
    
    landmarks = [
          [2.0, 2.0],            # x, y
          [5.0, 5.0],
          [8.0, 8.0]
    ]              
    robot_starting_pose = Pose(pos = (0.0, 0.0), theta = 0.6)  #x, y and theta
    bearing = robot_starting_pose.theta
    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        robot_starting_pose,
        bearing
    )

    # set up the robot
    robot = Robot(env)

    # set up the (Extended) Kalman Filter
    LINEAR = True
    if LINEAR:
        kf = KalmanFilter(
            float(dt),
            robot_starting_pose,
        )
    else:
        # set up the Extended Kalman Filter
        kf = ExtendedKalmanFilter(
            dt,
            robot_starting_pose,
        )

    # set up timekeeping
    # TODO: (done) set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = 30                          #total run time
    total_timesteps = total_seconds / env.DT    #calculate time step

    # set up logging
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "./input/motor_commands.csv"
    output_ground_truth_filepath = "./output/ground_truth.csv"
    output_sensor_data_filepath = "./output/sensor_data.csv"
    output_kalman_filter_filepath = "./output/kalman_filter.csv"

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # TODO: (done) take a ground truth snapshot and add it to the history
            ground_truth_history.append(env.take_state_snapshot())
            # TODO: (done) take sensor measurements and add it to the history
            sensor_data_history.append(robot.take_sensor_measurements())
            if LINEAR:
                # TODO: (done) call the Kalman Filter prediction step
                kf.predict(robot.take_sensor_measurements())
                # TODO: (done) call the Kalman Filter update step if new sensor data is available
                kf.update()
            else:
                # TODO: (done) call the Extended Kalman Filter prediction step
                kf.predict(robot.take_sensor_measurements())
                # TODO: (done) call the Extended Kalman Filter update step if new sensor data is available, for each GPS reading and for each landmark ping
                kf.update()

            # TODO: retrieve the next motor command from the input file
            for data_point in input_commands_filepath:
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
            robot.robot_step_differential(linear_input, angular_input)
    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file
        gt_data.write(ground_truth_history)
    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
        sensor_data.write(sensor_data_history)
    with open(output_kalman_filter_filepath, "w") as kalman_data:
        kalman_data.write(kalman_filter_history)