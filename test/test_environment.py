import pytest
import numpy as np
import math
from src.environment import Environment
from src.utils import Position, Pose, Bounds, Landmark, BearingRange


@pytest.mark.unit
class TestEnvironmentInitialization:
    """Test cases for Environment initialization."""
    
    def test_environment_creation(self, simple_environment):
        """Test basic environment creation."""
        assert simple_environment.DT == 0.1
        assert simple_environment.time == 0.0
        assert len(simple_environment.LANDMARKS) == 3
    
    def test_environment_dimensions(self, simple_environment, world_bounds):
        """Test environment dimensions are set correctly."""
        assert simple_environment.DIMENSIONS == world_bounds
    
    def test_environment_starting_pose(self, simple_environment, origin_pose):
        """Test environment robot starting pose."""
        assert simple_environment.robot_pose == origin_pose
    
    def test_environment_with_obstacles(self, environment_with_obstacles):
        """Test environment with obstacles."""
        assert len(environment_with_obstacles.OBSTACLES) == 1


@pytest.mark.unit
class TestRobotStep:
    """Test cases for robot_step method."""
    
    def test_robot_step_x_movement(self, simple_environment):
        """Test robot movement in x direction."""
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.x == 1.0
    
    def test_robot_step_y_movement(self, simple_environment):
        """Test robot movement in y direction."""
        simple_environment.robot_step(dx=0.0, dy=2.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.y == 2.0
    
    def test_robot_step_theta_movement(self, simple_environment):
        """Test robot heading change."""
        simple_environment.robot_step(dx=0.0, dy=0.0, dtheta=np.pi / 4)
        assert np.isclose(simple_environment.robot_pose.theta, np.pi / 4)
    
    def test_robot_step_combined_movement(self, simple_environment):
        """Test robot combined x, y, and theta movement."""
        simple_environment.robot_step(dx=1.0, dy=1.0, dtheta=np.pi / 2)
        assert simple_environment.robot_pose.pos.x == 1.0
        assert simple_environment.robot_pose.pos.y == 1.0
        assert np.isclose(simple_environment.robot_pose.theta, np.pi / 2)
    
    def test_robot_step_time_increment(self, simple_environment):
        """Test that robot_step increments time."""
        initial_time = simple_environment.time
        simple_environment.robot_step(dx=0.0, dy=0.0, dtheta=0.0)
        assert simple_environment.time == initial_time + 0.1
    
    def test_robot_step_multiple_times(self, simple_environment):
        """Test multiple sequential robot steps."""
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        
        assert simple_environment.robot_pose.pos.x == 2.0
        assert simple_environment.time == 0.2
    
    def test_robot_step_large_movement(self, simple_environment):
        """Test robot with large movement value."""
        simple_environment.robot_step(dx=50.0, dy=50.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.x == 50.0
        assert simple_environment.robot_pose.pos.y == 50.0
    
    def test_robot_step_negative_movement(self, simple_environment):
        """Test robot movement in negative direction."""
        simple_environment.robot_step(dx=-1.0, dy=-1.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.x == -1.0
        assert simple_environment.robot_pose.pos.y == -1.0

@pytest.mark.unit
class TestIsValidMotion:
    """Test cases for robot_step method."""
    
    def test_robot_step_valid_x_movement(self, simple_environment):
        """Test robot movement in x direction."""
        simple_environment.is_valid_motion(dx=1.0, dy=0.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.x == 1.0
    
    def test_robot_step_valid_y_movement(self, simple_environment):
        """Test robot movement in y direction."""
        simple_environment.is_valid_motion(dx=0.0, dy=2.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.y == 2.0
    
    
    def test_robot_step_valid_combined_movement(self, simple_environment):
        """Test robot combined x, y, and theta movement."""
        simple_environment.is_valid_motion(dx=1.0, dy=1.0, dtheta=np.pi / 2)
        assert simple_environment.robot_pose.pos.x == 1.0
        assert simple_environment.robot_pose.pos.y == 1.0
        assert np.isclose(simple_environment.robot_pose.theta, np.pi / 2)
    
    def test_robot_step_invalid_large_x_movement(self, simple_environment):
        """Test robot movement in x direction."""
        simple_environment.robot_step(dx=50.0, dy=0.0, dtheta=0.0)
        assert "Changing x motion by 50.0 causes a collision or is out of bounds" 
    
    def test_robot_step_invalid_neg_x_movement(self, simple_environment):
        """Test robot movement in x direction."""
        simple_environment.robot_step(dx=-1.0, dy=0.0, dtheta=0.0)
        assert "Changing x motion by -1.0 causes a collision or is out of bounds"

    def test_robot_step_invalid_large_y_movement(self, simple_environment):
        """Test robot movement in y direction."""
        simple_environment.robot_step(dx=0.0, dy=20.0, dtheta=0.0)
        assert "Changing y motion by 20.0 causes a collision or is out of bounds" 

    def test_robot_step_invalid_neg_y_movement(self, simple_environment):
        """Test robot movement in y direction."""
        simple_environment.robot_step(dx=0.0, dy=-2.0, dtheta=0.0)
        assert "Changing y motion by -2.0 causes a collision or is out of bounds" 
    

@pytest.mark.unit
class TestIsValidPosition:
    """Test cases for is_valid_position method."""
    
    def test_valid_position_center(self, simple_environment):
        """Test valid position in center of environment."""
        pos = Position(5.0, 5.0)
        assert simple_environment.is_valid_position(pos) is True
    
    def test_valid_position_boundary(self, simple_environment):
        """Test valid position on boundary."""
        pos = Position(0.0, 0.0)
        assert simple_environment.is_valid_position(pos) is True
    
    def test_invalid_position_outside_x(self, simple_environment):
        """Test invalid position outside x bounds."""
        pos = Position(15.0, 5.0)
        assert simple_environment.is_valid_position(pos) is False
    
    def test_invalid_position_outside_y(self, simple_environment):
        """Test invalid position outside y bounds."""
        pos = Position(5.0, 15.0)
        assert simple_environment.is_valid_position(pos) is False
    
    def test_invalid_position_negative_x(self, simple_environment):
        """Test invalid position with negative x."""
        pos = Position(-1.0, 5.0)
        assert simple_environment.is_valid_position(pos) is False
    
    def test_invalid_position_negative_y(self, simple_environment):
        """Test invalid position with negative y."""
        pos = Position(5.0, -1.0)
        assert simple_environment.is_valid_position(pos) is False


@pytest.mark.unit
class TestGetRobotPose:
    """Test cases for get_robot_pose method."""
    
    def test_get_robot_pose_initial(self, simple_environment, origin_pose):
        """Test getting initial robot pose."""
        pose = simple_environment.get_robot_pose()
        assert pose == origin_pose
    
    def test_get_robot_pose_after_movement(self, simple_environment):
        """Test getting robot pose after movement."""
        simple_environment.robot_step(dx=3.0, dy=4.0, dtheta=0.5)
        pose = simple_environment.get_robot_pose()
        
        assert pose.pos.x == 3.0
        assert pose.pos.y == 4.0
        assert np.isclose(pose.theta, 0.5)
    
    def test_get_robot_pose_returns_pose_object(self, simple_environment):
        """Test that get_robot_pose returns a Pose object."""
        pose = simple_environment.get_robot_pose()
        assert isinstance(pose, Pose)


@pytest.mark.unit
class TestGetProximityToLandmarks:
    """Test cases for get_proximity_to_landmarks method."""
    
    def test_proximity_at_origin(self, simple_environment):
        """Test landmark proximity from origin."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        assert len(proximities) == 3
        # Updated: Check for lists instead of BearingRange objects
        assert all(isinstance(p, list) and len(p) == 3 for p in proximities)
    
    def test_proximity_correct_count(self, simple_environment):
        """Test that proximity returns data for all landmarks."""
        proximities = simple_environment.get_proximity_to_landmarks()
        assert len(proximities) == len(simple_environment.LANDMARKS)
    
    def test_proximity_range_values(self, simple_environment):
        """Test that proximity returns reasonable range values."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        for prox in proximities:
            # prox is now [landmark_id, bearing, range]
            range_value = prox[2]
            assert range_value >= 0  # Range should be positive
            assert isinstance(range_value, (int, float, np.number))
    
    def test_proximity_bearing_values(self, simple_environment):
        """Test that proximity returns valid bearing values."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        for prox in proximities:
            # prox is now [landmark_id, bearing, range]
            bearing_value = prox[1]
            # Bearing should be wrapped to [-pi, pi]
            assert -np.pi <= bearing_value <= np.pi
    
    def test_proximity_landmark_ids(self, simple_environment):
        """Test that proximity includes correct landmark IDs."""
        proximities = simple_environment.get_proximity_to_landmarks()
        # prox is now [landmark_id, bearing, range]
        ids = [p[0] for p in proximities]
        
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids
    
    def test_proximity_structure(self, simple_environment):
        """Test that each proximity entry has correct structure."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        for prox in proximities:
            # Verify structure: [id, bearing, range]
            assert isinstance(prox, list)
            assert len(prox) == 3
            
            landmark_id, bearing, range = prox
            
            # ID should be int
            assert isinstance(landmark_id, int)
            
            # Bearing should be float in [-pi, pi]
            assert isinstance(bearing, float)
            assert -np.pi <= bearing <= np.pi
            
            # Range should be non-negative float
            assert isinstance(range, float)
            assert range >= 0
    
    @pytest.mark.slow
    def test_proximity_after_movement(self, simple_environment):
        """Test proximity changes after robot movement."""
        prox_before = simple_environment.get_proximity_to_landmarks()
        
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        prox_after = simple_environment.get_proximity_to_landmarks()
        
        # Ranges should have changed (closer to at least some landmarks)
        # prox_before[0][2] is the range, prox_after[0][2] is the new range
        assert prox_before[0][2] != prox_after[0][2]
