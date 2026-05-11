# Day 20
## Problem 1
(Attach Photo)

## Problem 2
### Excercise 1
#### Explain in your own words the functionality of each class, what their parameters are, and what just-in-time inputs are needed to compute useful actions/rewards
##### Action Class
**Parameters**: 

- *num_actions*: # of possible actions

- *vel1_range*: min and max for one type of velocity (ie. x velocity)

- *vel2_range*: min and max for a second type of velocity (ie y velocity)

- *action_step*: time step length per acction

**Input to Output Functions**:

- def *_get_actions_as_commands*: Attaches each action to velocity values

- *Outputs*: list of actions, each action paired with a vel1 and vel2

- def *get_actions_as_waypoints*: Update robot pose based on specified action

- *Parameters*: robot(robot of reference), drive_type(differential or swerve)

- *Inputs*: bounds = robot.env.DIMS

- *Time Inputs*: robot_pose = robot.env.agent_pose

- *Outputs*: new robot pose, remove invalid actions

**Getter Functions**

- def *convert_action_to_velocity*: returns velocity for specified action

- *Time Inputs*: specified action

- *Output*: Velocity for specfied action

- def *info*: Stores action info

- Output: action step, num actions, vel1 range, vel2 range, and command actions

##### Reward Class
**Parameters**

- *reward_params*: type of reward function

**Input to Output Functions**

- def *_create_reward_function*: Attach equation to reward function name

- Output: reward as mean and variance

**Getter Functions**

- def *info*: Stores reward info

- *Outputs*: reward params

#### How do you think you would use these classes within a simulation?
In autonomy_simulator.py, I would use the action class to associate action with state and the reward class to select a reward function.

### Excercise 2
Ensure that you understand the new simulation loop. When are new actions updated and sent to the robot? When is reward evaluated? 

- New actions are updated at line 192 which is:

"current_lin_vel, current_ang_vel = actions.command_actions[action]"

- New actions are sent to the robot at line 197 which is:

"robot.agent_step_differential(current_lin_vel, current_ang_vel)"

- Reward is evaluated at line 189 which is:

"action_reward = reward_function(waypoint_mean[0], waypoint_cov[0])"

### Excercise 3
#### As you change the parameter of the UCB function, what do you notice about the reward field and its relationship to the mean and uncertainty of the belief state? As you modify the action space, what do you notice about the expressiveness of the robot’s path?

- Decreasing the UCB parameter makes the mean value stronger and the robot is more exploitive. Increasing the UCB parameter makes the uncertainity stronger which encourages more exploration. 