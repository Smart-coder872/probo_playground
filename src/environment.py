"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in.
The world is continuous and two-dimensional. The world possesses
an outer border, internal obstacles, and identifiable landmarks.
The world also manages the passage of time and the motion of robotic agents
within the world over time.

Critically, the environment tracks the robot's state. In this case,
the robot's state is a vector that includes three state variables:
x position, y position, and heading.
"""

from utils import Position, Pose, Bounds, Landmark, BearingRange, wrap_angle
from numpy import sqrt, arctan2
from pandas import DataFrame

class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical borders of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles using Bounds class
        landmarks: a list of landmarks using Landmark class
        robot_a_starting_pose: the position and heading of one robot in the world
        robot_b_starting_pose: the position and heading of another in the world
        robot_a_bearing: the initial bearing of one robot
        robot_b_bearing: the initial bearing of another robot
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_a_starting_pose: Pose,
        robot_b_starting_pose: Pose,
        robot_a_bearing: BearingRange,
        robot_b_bearing: BearingRange
    ):
        """
        Initialize an instance of the Environment class.

        Store each attritute as a referenceable variable using self

        """
        # TODO (done): set the dimensions property to the parameter value
        self.DIMENSIONS = dimensions

        # TODO (done): set the timestep size property to the parameter value
        self.DT = dt

        # TODO (done): set the current time to zero
        self.time = 0.0

        # TODO (done): set the obstacles and landmarks properties to the parameter lists
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks
        self.robot_bearings = [robot_a_bearing, robot_b_bearing]

        # TODO (done): set the robot pose property to the parameter value
        self.robot_poses = [robot_a_starting_pose, robot_b_starting_pose]

    def robot_step(self, dx: float, dy: float, dtheta: float, which_robot: bool):
        """
        Update the robot's position and heading in the world.
        The robot should not be able to pass through obstacles
        or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading
            which_robot: specify robot a or b where 0 is a and 1 is b

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        # TODO: (done) fill in the function
        if which_robot == 0:                            ##If robot a...
            self.robot_poses[0].pos.x += dx             ##Update x
            self.robot_poses[0].pos.y += dy             ##Update y
        
            self.robot_poses[0].theta = wrap_angle(
            self.robot_poses[0].theta + dtheta)     ##Update heading
            
            self.time += self.DT                    ##Update time step

        
        elif which_robot == 1:                          #Of robot b...
            self.robot_poses[1].pos.x += dx             ##Update x
            self.robot_poses[1].pos.y += dy             ##Update y
        
            self.robot_poses[1].theta = wrap_angle(
            self.robot_poses[1].theta + dtheta)     ##Update heading
            
            self.time += self.DT                    ##Update time step
        
        else:
            f"Robot is not recognized as a or b"
        

    def is_valid_motion(self, dx: float, dy: float, dtheta: float, which_robot: bool):
        """
        Given attempted x and y motion by the robot, determine
        what motion is physically possible (i.e. doesn't go through
        any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """

        
        # TODO: (done) fill in the function
        if which_robot == 0:                               #If robot a...
            attempted_x = self.robot_poses[0].pos.x + dx   #Calculate new x without storing in instance
            attempted_y = self.robot_poses[0].pos.y + dy   #Calculate new y without storing in instance
        
            if self.DIMENSIONS.within_bounds(
                Position(attempted_x, attempted_y)
                ):                                              ##If dx and dy are valid...
                self.robot_step(dx, dy, dtheta, 0)              ##...Update robot pose
                
                return self.robot_poses[0]                      ##...Output the new heading and position
            elif not self.DIMENSIONS.within_x(attempted_x):                                   ##If dx is not valid...
                f"X motion change for robot_a by {dx} causes a collision or is out of bounds" ##...Print dx error message
            
            elif not self.DIMENSIONS.within_y(attempted_y):                                   ##If dy is not valid...
                f"Y motion change for robot_b by {dy} causes a collision or is out of bounds" ##...Print dy error message

        
        elif which_robot == 1:                             #If robot b...
            attempted_x = self.robot_poses[1].pos.x + dx   #Calculate new x without storing in instance
            attempted_y = self.robot_poses[1].pos.y + dy   #Calculate new y without storing in instance
        
            if self.DIMENSIONS.within_bounds(
                Position(attempted_x, attempted_y)
                ):                                              ##If dx and dy are valid...
                self.robot_step(dx, dy, dtheta, 1)              ##...Update robot pose
                
                return self.robot_poses[1]                      ##...Output the new heading and position
            elif not self.DIMENSIONS.within_x(attempted_x):                                   ##If dx is not valid...
                f"X motion change for robot_a by {dx} causes a collision or is out of bounds" ##...Print dx error message
            
            elif not self.DIMENSIONS.within_y(attempted_y):                                   ##If dy is not valid...
                f"Y motion change for robot_b by {dy} causes a collision or is out of bounds" ##...Print dy error message
        else:
            f"Robot is not recognized as a or b"

    def is_valid_position(self, position: Position, which_robot: bool):
        """
        Check if a given robot position is valid;
        i.e. not out-of-bounds or within an obstacle.
        Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """
        # TODO: (done) fill in the function
        if which_robot == 0:                            #If robot a
            if self.DIMENSIONS.within_bounds(position): #If the robot position is within bounds...
                return True                             #...This condition is true
            else:                                       #Otherwise...        
                return False                            #This condition is false
        elif which_robot == 1:
            if self.DIMENSIONS.within_bounds(position): #If the robot position is within bounds...
                return True                             #...This condition is true
            else:                                       #Otherwise...        
                return False                            #This condition is false 
        else:
            f"Robot is not recognized as a or b"

    def get_robot_pose(self, which_robot: bool):
        """
        Return the true robot pose.
        """
        # TODO: (done) fill in the function
        if which_robot == 0:
            return self.robot_poses[0]          #Output current robot position and heading
        elif which_robot == 1:
            return self.robot_poses[1]
        else:
            f"Robot is not recognized as a or b"

    def get_proximity_to_robot(self, which_robot: bool, which_other:bool):
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        # TODO: (done) fill in the function
        robot_a_x = self.robot_poses[which_robot].pos.x
        robot_a_y = self.robot_poses[which_robot].pos.y
            
        robot_b_x = self.robot_poses[which_other].pos.x
        robot_b_y = self.robot_poses[which_other].pos.y

        x_range = robot_a_x - robot_b_x
        y_range = robot_a_y - robot_b_y
        range = sqrt(x_range**2 + y_range**2)

        total_angle = arctan2(y_range, x_range)
       
            
        bearing = total_angle - self.robot_poses[which_robot].theta     
                       
        result = [which_other, range, bearing] 
                
 
        return result   

    def take_state_snapshot(self, which_robot: bool, which_other:bool):
        """
        Return true state information about this timestep,
        including time, robot position, and the robot's bearing/range
        to landmarks, in a table format.
        """
        # TODO: (done) fill in the function       
        snapshot = DataFrame(
            {"Time": [self.time],
            "Robot pos": [self.robot_poses[which_robot]],
            "Range:": [self.get_proximity_to_robot(which_robot, which_other)[0]],
            "Robot bearing": [self.get_proximity_to_robot(which_robot, which_other)[1]]
            })
           
        

        return snapshot
    
    def get_environment_info(self):
        """
        Return static information about the environment,
        including dimensions, timestep size, locations and dimensions
        of obstacles, and locations of landmarks.
        """
        # TODO: (done) fill in the function
        print("Environment dimensions: " + self.DIMENSIONS +
              "Timestep size: " + self.DT +
              "Obstacle Locations/Dimensions: " + self.OBSTACLES +
              "Landmark Locations: " + self.LANDMARKS) 