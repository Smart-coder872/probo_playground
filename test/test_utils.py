"""
Unit tests for the utils module.

Tests the custom datatypes and utility functions like Position, Pose,
Bounds, Landmark, BearingRange, and wrap_angle.
"""

import pytest
import numpy as np
from src.utils import Position, Pose, Bounds, Landmark, BearingRange, wrap_angle


class TestPosition:
    """Test cases for the Position dataclass."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(x=5.0, y=3.0)
        assert pos.x == 5.0
        assert pos.y == 3.0
    
    def test_position_default_values(self):
        """Test position creation with default values."""
        pos = Position()
        assert pos.x == 0.0
        assert pos.y == 0.0
    
    def test_position_to_dict(self):
        """Test converting position to dictionary."""
        pos = Position(x=2.5, y=3.5)
        result = pos.to_dict()
        assert result == {"x": 2.5, "y": 3.5}
    
    def test_position_to_string(self):
        """Test converting position to string."""
        pos = Position(x=1.0, y=2.0)
        result = pos.to_string()
        assert result == "X1.0Y2.0"
    


class TestPose:
    """Test cases for the Pose dataclass."""
    
    def test_pose_creation(self):
        """Test basic pose creation."""
        pos = Position(x=1.0, y=2.0)
        pose = Pose(pos=pos, theta=0.5)
        assert pose.pos == pos
        assert pose.theta == 0.5
    
    def test_pose_default_values(self):
        """Test pose with default values."""
        pose = Pose()
        assert pose.pos == Position(0.0, 0.0)
        assert pose.theta == 0.0
    
    def test_pose_to_dict(self):
        """Test converting pose to dictionary."""
        pose = Pose(pos=Position(1.0, 2.0), theta=3.14)
        result = pose.to_dict()
        assert result["pos"] == {"x": 1.0, "y": 2.0}
        assert result["theta"] == 3.14
    
    def test_pose_to_string(self):
        """Test converting pose to string."""
        pose = Pose(pos=Position(1.0, 2.0), theta=0.5)
        result = pose.to_string()
        assert "X1.0" in result
        assert "Y2.0" in result
        assert "T0.5" in result


class TestBounds:
    """Test cases for the Bounds dataclass."""
    
    def test_bounds_creation(self):
        """Test basic bounds creation."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.x_min == 0.0
        assert bounds.x_max == 10.0
        assert bounds.y_min == 0.0
        assert bounds.y_max == 10.0
    
    def test_within_x_inside(self):
        """Test x boundary checking - inside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.within_x(5.0) is True
    
    def test_within_x_outside(self):
        """Test x boundary checking - outside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.within_x(15.0) is False
    
    def test_within_x_boundary(self):
        """Test x boundary checking - on boundary."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.within_x(0.0) is True
        assert bounds.within_x(10.0) is True
    
    def test_within_y_inside(self):
        """Test y boundary checking - inside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.within_y(5.0) is True
    
    def test_within_y_outside(self):
        """Test y boundary checking - outside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        assert bounds.within_y(-1.0) is False
    
    def test_within_bounds_inside(self):
        """Test position within bounds - inside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        pos = Position(5.0, 5.0)
        assert bounds.within_bounds(pos) is True
    
    def test_within_bounds_outside(self):
        """Test position within bounds - outside."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        pos = Position(15.0, 15.0)
        assert bounds.within_bounds(pos) is False
    
    def test_within_bounds_corner(self):
        """Test position within bounds - corner."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        pos = Position(0.0, 0.0)
        assert bounds.within_bounds(pos) is True
    
    def test_bounds_to_dict(self):
        """Test converting bounds to dictionary."""
        bounds = Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
        result = bounds.to_dict()
        assert result == {
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": 0.0,
            "y_max": 10.0,
        }


class TestLandmark:
    """Test cases for the Landmark dataclass."""
    
    def test_landmark_creation(self):
        """Test basic landmark creation."""
        pos = Position(5.0, 5.0)
        landmark = Landmark(pos=pos, id=1)
        assert landmark.pos == pos
        assert landmark.id == 1
    
    def test_landmark_to_dict(self):
        """Test converting landmark to dictionary."""
        landmark = Landmark(Position(3.0, 4.0), id=2)
        result = landmark.to_dict()
        assert result["id"] == 2
        assert result["pos"] == {"x": 3.0, "y": 4.0}
    
    def test_landmark_to_string(self):
        """Test converting landmark to string."""
        landmark = Landmark(Position(1.0, 2.0), id=5)
        result = landmark.to_string()
        assert "L5" in result

    def test_landmark_to_list(self):
        """Test converting landmark to list"""
        landmark = Landmark(Position(3.0, 1.0), id = 3)
        result = landmark.to_list()
        assert result == [3.0, 1.0, 3]

class TestBearingRange:
    """Test cases for the BearingRange dataclass."""
    
    def test_bearing_range_creation(self):
        """Test basic bearing/range creation."""
        br = BearingRange(landmark_id=1, bearing=0.5, range=3.0)
        assert br.landmark_id == 1
        assert br.bearing == 0.5
        assert br.range == 3.0
    
    def test_bearing_range_to_dict(self):
        """Test converting bearing/range to dictionary."""
        br = BearingRange(landmark_id=2, bearing=1.0, range=5.0)
        result = br.to_dict()
        assert result == {
            "landmark_id": 2,
            "bearing": 1.0,
            "range": 5.0,
        }
    
    def test_bearing_range_to_string(self):
        """Test converting bearing/range to string."""
        br = BearingRange(landmark_id=1, bearing=0.5, range=3.0)
        result = br.to_string()
        assert "LM1" in result


class TestWrapAngle:
    """Test cases for the wrap_angle utility function."""
    
    def test_wrap_angle_zero(self):
        """Test wrap_angle with zero."""
        result = wrap_angle(0.0)
        assert result == 0.0
    
    def test_wrap_angle_positive_in_range(self):
        """Test wrap_angle with positive angle already in range."""
        result = wrap_angle(np.pi / 4)
        assert np.isclose(result, np.pi / 4)
    
    def test_wrap_angle_negative_in_range(self):
        """Test wrap_angle with negative angle already in range."""
        result = wrap_angle(-np.pi / 4)
        assert np.isclose(result, -np.pi / 4)
    
    def test_wrap_angle_positive_overflow(self):
        """Test wrap_angle with angle > pi."""
        result = wrap_angle(1.5 * np.pi)
        assert np.isclose(result, -0.5 * np.pi)
    
    def test_wrap_angle_two_pi(self):
        """Test wrap_angle with 2*pi (should wrap to 0)."""
        result = wrap_angle(2 * np.pi)
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_wrap_angle_large_positive(self):
        """Test wrap_angle with large positive angle."""
        result = wrap_angle(10 * np.pi)
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_wrap_angle_large_negative(self):
        """Test wrap_angle with large negative angle."""
        result = wrap_angle(-10 * np.pi)
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_wrap_angle_boundary_pi(self):
        """Test wrap_angle at boundary pi."""
        result = wrap_angle(np.pi)
        # pi should be at the positive boundary
        assert np.isclose(result, np.pi) or np.isclose(result, -np.pi)
    
    def test_wrap_angle_just_over_pi(self):
        """Test wrap_angle just over pi."""
        result = wrap_angle(np.pi + 0.1)
        assert -np.pi <= result <= np.pi
        assert result < 0


class TestDataIntegration:
    """Integration tests for multiple data types working together."""
    
    def test_pose_with_position_modification(self):
        """Test modifying a pose's position."""
        pose = Pose(pos=Position(1.0, 2.0), theta=0.5)
        new_pos = Position(pose.pos.x + 1.0, pose.pos.y + 1.0)
        new_pose = Pose(pos=new_pos, theta=pose.theta)
        
        assert new_pose.pos.x == 2.0
        assert new_pose.pos.y == 3.0
    
    def test_landmark_in_bounds(self):
        """Test checking if a landmark is within bounds."""
        landmark = Landmark(Position(5.0, 5.0), id=1)
        bounds = Bounds(0.0, 10.0, 0.0, 10.0)
        
        assert bounds.within_bounds(landmark.pos) is True
    
    def test_bearing_range_with_invalid_values(self):
        """Test bearing/range with extreme values."""
        import math
        br = BearingRange(landmark_id=0, bearing=math.inf, range=math.inf)
        
        assert math.isinf(br.bearing)
        assert math.isinf(br.range)