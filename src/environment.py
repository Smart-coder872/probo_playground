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
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_starting_pose: Pose,
        bearing: BearingRange
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
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
        self.BEARING = bearing

        # TODO (done): set the robot pose property to the parameter value
        self.robot_pose = robot_starting_pose

    def robot_step(self, dx: float, dy: float, dtheta: float):
        """
        Update the robot's position and heading in the world.
        The robot should not be able to pass through obstacles
        or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        # TODO: (done) fill in the function
        self.robot_pose.pos.x += dx             ##Update x
        self.robot_pose.pos.y += dy             ##Update y
        
        self.robot_pose.theta = wrap_angle(
            self.robot_pose.theta + dtheta)     ##Update heading
        
        self.time += self.DT                    ##Update time step
        

    def is_valid_motion(self, dx: float, dy: float, dtheta: float):
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
        attempted_x = self.robot_pose.pos.x + dx   #Calculate new x without storing in instance
        attempted_y = self.robot_pose.pos.y + dy   #Calculate new y without storing in instance
        
        if self.DIMENSIONS.within_bounds(
            Position(attempted_x, attempted_y)): ##If dx and dy are valid...
            self.robot_step(dx, dy, dtheta)      ##...Update robot pose
            return self.robot_pose               ##...Output the new heading and position
        elif not self.DIMENSIONS.within_x(
            attempted_x):                                    ##If dx is not valid...
            print("Changing x motion by " + 
                  dx + 
                  "causes a collision or is out of bounds" ) ##...Print dx error message
        
        elif not self.DIMENSIONS.within_y(attempted_y):      ##If dy is not valid...
            print("Changing y motion by " + 
                  dy + 
                  "causes a collision or is out of bounds")  ##...Print dy error message

    def is_valid_position(self, position: Position):
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
        if self.DIMENSIONS.within_bounds(position): #If the robot position is within bounds...
            return True                             #...This condition is true
        else:                                       #Otherwise...        
            return False                            #This condition is false

    def get_robot_pose(self):
        """
        Return the true robot pose.
        """
        # TODO: (done) fill in the function
        return self.robot_pose          #Output current robot position and heading

    def get_proximity_to_landmarks(self):
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        # TODO: (done) fill in the function
        landmark_proximities = []

        for landmark in self.LANDMARKS:
            l_x = landmark.pos.x
            l_y = landmark.pos.y
            x_range = l_x - self.robot_pose.pos.x
            y_range = l_y - self.robot_pose.pos.y
            range = sqrt(x_range**2 + y_range**2)
            
            total_angle = arctan2(y_range, x_range)
            bearing = total_angle - self.robot_pose.theta
            
            result = range, bearing 
            landmark_proximities.append(result)
 
        return landmark_proximities       

    def take_state_snapshot(self):
        """
        Return true state information about this timestep,
        including time, robot position, and the robot's bearing/range
        to landmarks, in a table format.
        """
        # TODO: (done) fill in the function
        
        snapshot = DataFrame(
            {"Time": [self.time],
             "Robot pos": [self.robot_pose],
             "Range:": [self.get_proximity_to_landmarks()[0]],
             "Bearing": [self.get_proximity_to_landmarks()[1]]}
        )
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
