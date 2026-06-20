"""
Main simulation loop for a decentralized two-robot swarm system.
"""

from environment import Environment
from robot import Robot
from extended_kalman_filter import IteratedEKF
from utils import Position, Pose, Landmark, Bounds
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # _______________ENVIRONMENT SETUP____________________________
    
    dimensions = Bounds(x_min=0, x_max=10, y_min=0, y_max=10)
    dt = 0.1  # seconds per step
    
    obstacles = [
        Bounds(x_min=5, x_max=7, y_min=5, y_max=7),
        Bounds(x_min=0, x_max=4, y_min=6, y_max=8),
    ]
    
    # Define landmarks
    landmarks = [
        Landmark(id=0, pos=Position(1.0, 1.0)),
        Landmark(id=1, pos=Position(9.0, 9.0)),
        Landmark(id=2, pos=Position(5.0, 1.0)),
    ]
    
    # Robot starting positions
    robot_a_starting_pose = Pose(Position(0.0, 0.0), 0.6)
    robot_b_starting_pose = Pose(Position(3.0, 3.0), 0.6)
    
    robot_a_bearing = 0.0
    robot_b_bearing = 0.0
    
    # Create environment
    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        robot_a_starting_pose,
        robot_b_starting_pose,
        robot_a_bearing,
        robot_b_bearing,
    )
    
    # _______________ROBOT SETUP ____________________________________
    
    #  Pass robot_id to each robot
    robot_a = Robot(env, robot_id=0, other_id=1, use_robot_pinger=True)
    robot_b = Robot(env, robot_id=1, other_id=0, use_robot_pinger=True)
    
    # Initial state vectors
    a_starting_states = np.array(
        [
            robot_a_starting_pose.pos.x,
            robot_a_starting_pose.pos.y,
            robot_a_starting_pose.theta,
        ]
    ).reshape(3, 1)
    
    b_starting_states = np.array(
        [
            robot_b_starting_pose.pos.x,
            robot_b_starting_pose.pos.y,
            robot_b_starting_pose.theta,
        ]
    ).reshape(3, 1)
    
    # _______________ Iterated I-EKF SETUP________________
    
    # Create separate filters for each robot
    kf_a = IteratedEKF(float(dt), a_starting_states)
    kf_b = IteratedEKF(float(dt), b_starting_states)
    
    # ________________COMMAND INPUT FILE SETUP__________________
    
    # Read motor commands (timestamp-based format)
    a_motor_commands_df = pd.read_csv("./input/a_motor_commands.csv")
    b_motor_commands_df = pd.read_csv("./input/b_motor_commands.csv")
    
    # ________________HISTORY FILE SETUP _______________________
    
    a_ground_truth_history = []
    a_sensor_data_history = []
    a_kalman_filter_history = []
    a_robot_pinger_history = []
    
    b_ground_truth_history = []
    b_sensor_data_history = []
    b_kalman_filter_history = []
    b_robot_pinger_history = []
    
    total_timesteps = 30
    
    # ______________ MAIN SIMULATION LOOP ________________
    
    print("Starting simulation...")
    
    for step in range(int(total_timesteps) + 1): #for each step from inital time until final time...
        # ________ ROBOT A ___________
        
        # Collect ground truth
        a_ground_truth = env.take_state_snapshot(0, 1)
        a_ground_truth_history.append(a_ground_truth)
        
        # Take sensor measurements
        a_measurements = robot_a.take_sensor_measurements()
        a_sensor_data_history.append(a_measurements)
        
        # Extract wheel encoder data for prediction
        a_wheel_data = a_measurements.get("WheelEncoder")
        if a_wheel_data is not None:
            a_lin_vel = a_wheel_data["linear_velocity"].values[0]
            a_ang_vel = a_wheel_data["angular_velocity"].values[0]
            
            # Prediction step
            a_control = np.array([[a_lin_vel], [a_ang_vel]])
            kf_a.predict(a_control)
        
        # Update with robot observations
        # Get RobotPinger measurements
        a_robot_data = a_measurements.get("RobotPinger")
        if a_robot_data is not None and not a_robot_data.empty:
        # Access "Robot_Pinger" column with BearingRange objects
            a_robot_measurements = a_robot_data["Robot_Pinger"].values
    
        for measurement in a_robot_measurements:
            if measurement.range != np.inf:  # Valid (in-range)
                other_robot_id = robot_a.OTHER_ID  # ← Which robot we're measuring
                z = np.array([[measurement.range], [measurement.bearing]])
                
                # Get Jacobian for this specific robot measurement
                H = robot_a.sensors["RobotPinger"].H_eval(a_starting_states, other_robot_id)
                
                # Get noise covariance
                R_range, R_bearing = robot_a.sensors["RobotPinger"].R(z)
                R = np.array([[R_range, 0], [0, R_bearing]])
                
                # Update filter with measurement
                kf_a.relative_update(H, R, z, y=None)
        
        # Store Kalman filter state
        b_kalman_filter_history.append(kf_b.x_state_ef.copy())
        
        # Sense other robots (decentralized awareness)
        b_robot_data = robot_b.sense_other_robots()
        if b_robot_data is not None:
            b_robot_pinger_history.append(b_robot_data)
        
        # ______________ MOTION CONTROL __________________
        
        # Get motor commands for robot A
        current_time = env.time
        a_matching_commands = a_motor_commands_df[
            a_motor_commands_df["timestamp"] <= current_time
        ]
        if not a_matching_commands.empty:
            a_command = a_matching_commands.iloc[-1]
            a_linear = float(a_command["linear_vel"])
            a_angular = float(a_command["angular_vel"])
        else:
            a_linear = 0.0
            a_angular = 0.0
        
        # Execute motion for robot A
        robot_a.robot_step_differential(a_linear, a_angular)
        
        # Get motor commands for robot B
        b_matching_commands = b_motor_commands_df[
            b_motor_commands_df["timestamp"] <= current_time
        ]
        if not b_matching_commands.empty:
            b_command = b_matching_commands.iloc[-1]
            b_linear = float(b_command["linear_vel"])
            b_angular = float(b_command["angular_vel"])
        else:
            b_linear = 0.0
            b_angular = 0.0
        
        # Execute motion for robot B
        robot_b.robot_step_differential(b_linear, b_angular)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(
                f"Step {step}/{total_timesteps} | "
                f"Time: {env.time:.2f}s | "
                f"Robot A pos: ({robot_a.env.robot_poses[0].pos.x:.2f}, "
                f"{robot_a.env.robot_poses[0].pos.y:.2f}) | "
                f"Robot B pos: ({robot_b.env.robot_poses[1].pos.x:.2f}, "
                f"{robot_b.env.robot_poses[1].pos.y:.2f})"
            )
    
    # ________________ WRITE OUTPUT FILES ________________
    
    def write_history_to_csv(filepath, history, data_type="measurement"):
        """Convert history list to CSV"""
        if not history:
            return
        
        # Handle different data types
        if data_type == "dataframe":
            # List of DataFrames
            combined_df = pd.concat(history, ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        elif data_type == "array":
            # List of numpy arrays
            combined_array = np.vstack(history)
            np.savetxt(filepath, combined_array, delimiter=",")
        else:
            # Direct write
            with open(filepath, "w") as f:
                f.write(str(history))
    
    print("\nWriting output files...")
    
    # Robot A outputs
    write_history_to_csv(
        "./output/ground_truth_a.csv", a_ground_truth_history, "dataframe"
    )
    write_history_to_csv(
        "./output/kalman_filter_a.csv", a_kalman_filter_history, "array"
    )
    
    # Robot B outputs
    write_history_to_csv(
        "./output/ground_truth_b.csv", b_ground_truth_history, "dataframe"
    )
    write_history_to_csv(
        "./output/kalman_filter_b.csv", b_kalman_filter_history, "array"
    )
    
    print("Simulation complete! Output files written to ./output/")
    print(
        f"\nFinal states:"
        f"\nRobot A: {kf_a.x_state_ef.flatten()}"
        f"\nRobot B: {kf_b.x_state_ef.flatten()}"
    )