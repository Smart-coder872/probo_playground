import numpy as np

class Planner:
    '''Implements a  planner class; greedy by default.'''
    def __init__(self, actions, reward, robot):
        self.actions = actions  # action class
        self.reward = reward  # reward class
        self.robot = robot  # robot class to operate with
    
    def select_action(self):
        # picks the best action according to the reward function
        waypoint_targets = self.actions.get_actions_as_waypoints(self.robot, "differential")
        best_action_value = -1000
        best_action = None
        for action_num in waypoint_targets.keys():
            waypoint = waypoint_targets[action_num]
            waypoint_mean, waypoint_cov = self.robot.belief.predict(np.asarray([waypoint[0], waypoint[1]]).reshape(1, -1), return_cov=True)
            action_reward = self.reward(waypoint_mean[0], waypoint_cov[0])
            if action_reward > best_action_value:
                best_action_value = action_reward
                best_action = action_num
        return best_action
