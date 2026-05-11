"""
Main file for running the simulator.
"""

from environment import Environment, Field
from robot import Robot
from actions import Actions
from reward import Reward
from planner import Planner
from utils import Pose, Position, Bounds, Landmark
from viz import Visualizer
import pandas as pd
from pathlib import Path
import csv, argparse, pickle, yaml
import numpy as np

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # set up pathing and argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--config",
        type=Path,
        default=Path("./input/config_example.yaml"),
        help="Path to config.yaml file.",
    )
    parser.add_argument(
        "-c",
        "--cmds",
        type=Path,
        default=Path("./input/vel_cmd_example.csv"),
        help="Path to motor commands file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./output/"),
        help="Indicate the desired output folder.",
    )
    parser.add_argument(
        "-v",
        "--viz_only",
        help="Indicate this to plot data that is already present in the output/ directory.",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--animate",
        help="Indicate this to create an animated GIF of the trajectory.",
        action="store_true",
    )
    args = parser.parse_args()
    CONFIG_PATH: Path = args.config
    assert CONFIG_PATH.exists()
    CMD_PATH: Path = args.cmds
    assert CMD_PATH.exists()
    OUTPUT_PATH: Path = args.output
    assert OUTPUT_PATH.exists()
    VIZ_ONLY: bool = args.viz_only
    ANIMATE: bool = args.animate

    # if not viz only, run the simulator and log the results
    if not VIZ_ONLY:
        # pull config info
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            env_info = config["environment"]
            robot_info = config["robot"]
            sensor_info = config["sensors"]
            action_info = config["action"]
            reward_info = config["reward"]
        # set up the sim
        obstacles = []
        for dims in env_info["obstacles"]:
            obstacles.append(
                Bounds(
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                )
            )
        landmarks = []
        for lm in env_info["landmarks"]:
            landmarks.append(
                Landmark(
                    Position(
                        lm[0],
                        lm[1],
                    ),
                    len(landmarks),
                )
            )
        field = Field(dimensions=Bounds(0,env_info["width"],0,env_info["height"]), 
                      variance=env_info["field"]["variance"],
                      lengthscale=env_info["field"]["lengthscale"],
                      random_seed=env_info["field"]["random_seed"])
        env = Environment(
            dimensions=Bounds(
                0,
                env_info["width"],
                0,
                env_info["height"],
            ),
            agent_pose=Pose(
                Position(
                    env_info["robot_start"][0],
                    env_info["robot_start"][1],
                ),
                env_info["robot_start"][2],
            ),
            obstacles=obstacles,
            landmarks=landmarks,
            field=field,
            timestep=env_info["timestep"],
            lm_range=env_info["pinger_range"],
        )
        robot = Robot(
            env,
            robot_info,
            sensor_info,
        )
        actions = Actions(
            action_info["action_step"],
            action_info["num_actions"],
            action_info["vel1_range"],
            action_info["vel2_range"]
        )
        reward = Reward(reward_info)
        reward_function = reward.reward
        planner = Planner(actions, reward_function, robot)

        # set up timekeeping
        total_seconds = env_info["runtime"]
        total_timesteps = total_seconds / env.DT
        steps_since_last_action = 0
        elapsed_action_time = 0.0
        terminal = False

        # set up logging
        ground_truth_history = pd.DataFrame()
        sensor_data_history = pd.DataFrame()

        # set up start velocity
        current_lin_vel, current_ang_vel = 0.0, 0.0

        # take a series of random actions
        for step in range(int(total_timesteps) + 1):
            # first, sample the environment
            # print(f"\n***TIMESTEP T{env.time}***")
            ground_truth_history = pd.concat(
                [
                    ground_truth_history,
                    robot.take_gt_snapshot(),
                ],
                ignore_index=True,
            )
                
            measurements = robot.take_sensor_measurements()
            sensor_data_history = pd.concat(
                [
                    sensor_data_history,
                    measurements,
                ],
                ignore_index=True,
            )

            # Update the robot's belief
            try:
                robot.update_belief(measurements["InsituInstrument"].values,
                                    robot.env.agent_pose)
            except:
                pass  # no measurement available to use

            # Select a random action
            elapsed_time = env.DT * step
            elapsed_action_time = env.DT * steps_since_last_action
            if round(elapsed_action_time) >= actions.action_step:
                # select a random valid action
                waypoint_targets = actions.get_actions_as_waypoints(robot, "differential")
                action = np.random.choice(list(waypoint_targets.keys()))

                waypoint = waypoint_targets[action]
                waypoint_mean, waypoint_cov = robot.belief.predict(np.asarray([waypoint[0], waypoint[1]]).reshape(1, -1), return_cov=True)
                action_reward = reward_function(waypoint_mean[0], waypoint_cov[0])
                print(f"Expected Action Reward: {action_reward}")
        
                current_lin_vel, current_ang_vel = actions.command_actions[action]
                steps_since_last_action = 0
                elapsed_action_time = 0.0

            # move the robot with current commands
            robot.agent_step_differential(current_lin_vel, current_ang_vel)
            steps_since_last_action += 1
            
        # log the results
        pickle.dump(
            ground_truth_history, open(OUTPUT_PATH / "groundtruth_log.pkl", "wb")
        )
        pickle.dump(sensor_data_history, open(OUTPUT_PATH / "sensor_log.pkl", "wb"))
        pickle.dump(env.info(), open(OUTPUT_PATH / "env_info.pkl", "wb"))
        pickle.dump(robot.info(), open(OUTPUT_PATH / "sensor_info.pkl", "wb"))
        pickle.dump(actions.info(), open(OUTPUT_PATH / "action_info.pkl", "wb"))
        pickle.dump(reward.info(), open(OUTPUT_PATH / "reward_info.pkl", "wb"))
        print("Done running simulator")

    # plot the results
    viz = Visualizer(
        OUTPUT_PATH,
    )
    viz.draw_all()

    if ANIMATE:
        viz.animate_trajectories()
