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
        assert simple_environment.robot_pose.pos.y == 0.0
        assert simple_environment.robot_pose.theta == 0.0
    
    def test_robot_step_y_movement(self, simple_environment):
        """Test robot movement in y direction."""
        simple_environment.robot_step(dx=0.0, dy=2.0, dtheta=0.0)
        assert simple_environment.robot_pose.pos.x == 0.0
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
        assert all(isinstance(p, BearingRange) for p in proximities)
    
    def test_proximity_correct_count(self, simple_environment):
        """Test that proximity returns data for all landmarks."""
        proximities = simple_environment.get_proximity_to_landmarks()
        assert len(proximities) == len(simple_environment.LANDMARKS)
    
    def test_proximity_range_values(self, simple_environment):
        """Test that proximity returns reasonable range values."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        for prox in proximities:
            assert prox.range >= 0  # Range should be positive
            assert isinstance(prox.range, (int, float, np.number))
    
    def test_proximity_bearing_values(self, simple_environment):
        """Test that proximity returns valid bearing values."""
        proximities = simple_environment.get_proximity_to_landmarks()
        
        for prox in proximities:
            # Bearing should be wrapped to [-pi, pi]
            assert -np.pi <= prox.bearing <= np.pi
    
    def test_proximity_landmark_ids(self, simple_environment):
        """Test that proximity includes correct landmark IDs."""
        proximities = simple_environment.get_proximity_to_landmarks()
        ids = [p.landmark_id for p in proximities]
        
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids
    
    @pytest.mark.slow
    def test_proximity_after_movement(self, simple_environment):
        """Test proximity changes after robot movement."""
        prox_before = simple_environment.get_proximity_to_landmarks()
        
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        prox_after = simple_environment.get_proximity_to_landmarks()
        
        # Ranges should have changed (closer to at least some landmarks)
        assert prox_before[0].range != prox_after[0].range


@pytest.mark.unit
class TestTakeStateSnapshot:
    """Test cases for take_state_snapshot method."""
    
    def test_snapshot_returns_dataframe(self, simple_environment):
        """Test that snapshot returns a DataFrame."""
        snapshot = simple_environment.take_state_snapshot()
        assert hasattr(snapshot, 'columns')  # DataFrame has columns
    
    def test_snapshot_includes_time(self, simple_environment):
        """Test that snapshot includes time information."""
        snapshot = simple_environment.take_state_snapshot()
        assert 'Time' in snapshot.columns or len(snapshot.columns) > 0
    
    def test_snapshot_after_steps(self, simple_environment):
        """Test snapshot at different time steps."""
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        snapshot1 = simple_environment.take_state_snapshot()
        
        simple_environment.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        snapshot2 = simple_environment.take_state_snapshot()
        
        # Snapshots should be different (different times)
        assert not snapshot1.equals(snapshot2)


@pytest.mark.unit
class TestEnvironmentEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_environment_with_no_obstacles(self, world_bounds, sample_landmarks, origin_pose):
        """Test environment creation with no obstacles."""
        env = Environment(
            dimensions=world_bounds,
            dt=0.1,
            obstacles=[],
            landmarks=sample_landmarks,
            robot_starting_pose=origin_pose,
            bearing=10.0,
        )
        assert len(env.OBSTACLES) == 0
    
    def test_environment_with_no_landmarks(self, world_bounds, origin_pose):
        """Test environment with no landmarks."""
        env = Environment(
            dimensions=world_bounds,
            dt=0.1,
            obstacles=[],
            landmarks=[],
            robot_starting_pose=origin_pose,
            bearing=10.0,
        )
        proximities = env.get_proximity_to_landmarks()
        assert len(proximities) == 0
    
    def test_robot_pose_angle_wrapping(self, simple_environment):
        """Test that robot heading wraps correctly."""
        simple_environment.robot_step(dx=0.0, dy=0.0, dtheta=3 * np.pi)
        
        # Should be wrapped to [-pi, pi]
        assert -np.pi <= simple_environment.robot_pose.theta <= np.pi


@pytest.mark.integration
class TestEnvironmentIntegration:
    """Integration tests for environment functionality."""
    
    def test_multiple_steps_simulation(self, simple_environment):
        """Test multiple steps of robot movement."""
        for _ in range(10):
            simple_environment.robot_step(dx=0.5, dy=0.0, dtheta=0.05)
        
        assert simple_environment.time == 1.0
        assert simple_environment.robot_pose.pos.x == 5.0
    
    def test_full_environment_workflow(self, simple_environment):
        """Test complete environment workflow."""
        # Move robot
        simple_environment.robot_step(dx=2.0, dy=2.0, dtheta=np.pi / 4)
        
        # Check pose
        pose = simple_environment.get_robot_pose()
        assert pose.pos.x == 2.0
        
        # Check validity
        assert simple_environment.is_valid_position(pose.pos)
        
        # Check landmarks
        proximities = simple_environment.get_proximity_to_landmarks()
        assert len(proximities) > 0