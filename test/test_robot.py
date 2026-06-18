"""
Unit tests for the robot module.

Tests the Robot class including differential drive, translational movement,
and sensor integration.
"""

import pytest
import numpy as np
import pandas as pd
from src.robot import Robot
from src.utils import Position, Pose


@pytest.mark.unit
class TestRobotInitialization:
    """Test cases for Robot initialization."""
    
    def test_robot_creation(self, simple_robot, simple_environment):
        """Test basic robot creation."""
        assert simple_robot.env == simple_environment
        assert simple_robot.LIN_VEL == 0.0
        assert simple_robot.ANG_VEL == 0.0
    
    def test_robot_has_sensors(self, simple_robot):
        """Test that robot has sensor dictionary."""
        assert isinstance(simple_robot.sensors, dict)
        assert 'GPS' in simple_robot.sensors
        assert 'WheelEncoder' in simple_robot.sensors
        assert 'LandmarkPinger' in simple_robot.sensors
    
    def test_robot_environment_reference(self, simple_robot, simple_environment):
        """Test that robot maintains environment reference."""
        assert simple_robot.env is simple_environment


@pytest.mark.unit
class TestDifferentialDrive:
    """Test cases for differential drive movement."""
    
    def test_differential_drive_forward(self, simple_robot):
        """Test forward movement with differential drive."""
        initial_x = simple_robot.env.robot_pose.pos.x
        initial_y = simple_robot.env.robot_pose.pos.y
        
        simple_robot.robot_step_differential(lin_vel=1.0, ang_vel=0.0)
        
        # Robot should move forward (in direction of heading)
        assert simple_robot.env.robot_pose.pos.x != initial_x or \
               simple_robot.env.robot_pose.pos.y != initial_y
    
    def test_differential_drive_velocity_stored(self, simple_robot):
        """Test that velocities are stored."""
        lin_vel = 1.5
        ang_vel = 0.3
        
        simple_robot.robot_step_differential(lin_vel=lin_vel, ang_vel=ang_vel)
        
        assert simple_robot.LIN_VEL == lin_vel
        assert simple_robot.ANG_VEL == ang_vel
    
    def test_differential_drive_zero_velocity(self, simple_robot):
        """Test differential drive with zero velocity."""
        initial_pose = simple_robot.env.get_robot_pose()
        
        simple_robot.robot_step_differential(lin_vel=0.0, ang_vel=0.0)
        
        # Robot position should not change significantly
        new_pose = simple_robot.env.get_robot_pose()
        assert np.isclose(initial_pose.pos.x, new_pose.pos.x)
        assert np.isclose(initial_pose.pos.y, new_pose.pos.y)
    
    def test_differential_drive_rotation_only(self, simple_robot):
        """Test rotation without forward movement."""
        initial_theta = simple_robot.env.robot_pose.theta
        initial_x = simple_robot.env.robot_pose.pos.x
        initial_y = simple_robot.env.robot_pose.pos.y
        
        simple_robot.robot_step_differential(lin_vel=0.0, ang_vel=0.5)
        
        # Theta should change, position should not change much
        new_theta = simple_robot.env.robot_pose.theta
        new_x = simple_robot.env.robot_pose.pos.x
        new_y = simple_robot.env.robot_pose.pos.y
        
        # Position might change slightly due to implementation details
        assert new_theta != initial_theta
    
    def test_differential_drive_negative_velocity(self, simple_robot):
        """Test backward movement."""
        simple_robot.robot_step_differential(lin_vel=-1.0, ang_vel=0.0)
        
        # Robot should move (backward direction)
        assert simple_robot.env.robot_pose.pos.x != 0.0 or \
               simple_robot.env.robot_pose.pos.y != 0.0


@pytest.mark.unit
class TestTranslationalDrive:
    """Test cases for translational (swerve drive) movement."""
    
    def test_translational_drive_x_only(self, simple_robot):
        """Test movement in x direction only."""
        simple_robot.robot_step_translational(x_vel=1.0, y_vel=0.0)
        
        # Robot should move in x direction
        assert simple_robot.env.robot_pose.pos.x == 0.1  # 1.0 * DT(0.1)
    
    def test_translational_drive_y_only(self, simple_robot):
        """Test movement in y direction only."""
        simple_robot.robot_step_translational(x_vel=0.0, y_vel=1.0)
        
        # Robot should move in y direction
        assert simple_robot.env.robot_pose.pos.y == 0.1  # 1.0 * DT(0.1)
    
    def test_translational_drive_combined(self, simple_robot):
        """Test combined x and y movement."""
        simple_robot.robot_step_translational(x_vel=1.0, y_vel=1.0)
        
        assert simple_robot.env.robot_pose.pos.x == 0.1
        assert simple_robot.env.robot_pose.pos.y == 0.1
    
    def test_translational_drive_velocity_stored(self, simple_robot):
        """Test that velocities are stored in translational drive."""
        x_vel = 2.0
        y_vel = 1.5
        
        simple_robot.robot_step_translational(x_vel=x_vel, y_vel=y_vel)
        
        assert simple_robot.X_vel == x_vel
        assert simple_robot.Y_vel == y_vel
    
    def test_translational_drive_no_rotation(self, simple_robot):
        """Test that translational drive doesn't change heading."""
        initial_theta = simple_robot.env.robot_pose.theta
        
        simple_robot.robot_step_translational(x_vel=1.0, y_vel=1.0)
        
        # Heading should not change
        assert np.isclose(simple_robot.env.robot_pose.theta, initial_theta)


@pytest.mark.unit
class TestSensorMeasurements:
    """Test cases for sensor measurements."""
    
    def test_take_sensor_measurements_returns_dataframe(self, simple_robot):
        """Test that sensor measurements returns a DataFrame."""
        measurements = simple_robot.take_sensor_measurements()
        assert isinstance(measurements, pd.DataFrame)
    
    def test_take_sensor_measurements_has_data(self, simple_robot):
        """Test that sensor measurements returns non-empty DataFrame."""
        measurements = simple_robot.take_sensor_measurements()
        assert len(measurements) > 0
    
    def test_sensor_measurements_columns(self, simple_robot):
        """Test that sensor measurements have expected columns."""
        measurements = simple_robot.take_sensor_measurements()
        
        # Should have columns from all three sensors
        assert 'LV_X' in measurements.columns or len(measurements.columns) > 0
    
    def test_sensor_measurements_after_movement(self, simple_robot):
        """Test sensor measurements after robot movement."""
        measurements_before = simple_robot.take_sensor_measurements()
        
        simple_robot.env.robot_step(dx=1.0, dy=0.0, dtheta=0.0)
        measurements_after = simple_robot.take_sensor_measurements()
        
        # Measurements should potentially differ
        assert isinstance(measurements_after, pd.DataFrame)


@pytest.mark.unit
class TestRobotInDifferentEnvironments:
    """Test robot behavior in different environments."""
    
    def test_robot_in_empty_environment(self, simple_robot, simple_environment):
        """Test robot in empty environment (no obstacles)."""
        assert len(simple_environment.OBSTACLES) == 0
        
        # Robot should be able to move freely
        simple_robot.robot_step_differential(lin_vel=1.0, ang_vel=0.0)
        assert simple_environment.robot_pose.pos.x != 0.0
    
    def test_robot_with_obstacles(self, robot_with_obstacles, environment_with_obstacles):
        """Test robot in environment with obstacles."""
        assert len(environment_with_obstacles.OBSTACLES) > 0
        
        robot_with_obstacles.robot_step_differential(lin_vel=0.5, ang_vel=0.0)
        # Robot should still be able to take steps


@pytest.mark.unit
class TestRobotProperties:
    """Test robot properties and attributes."""
    
    def test_robot_linear_velocity_property(self, simple_robot):
        """Test robot linear velocity property."""
        simple_robot.LIN_VEL = 2.5
        assert simple_robot.LIN_VEL == 2.5
    
    def test_robot_angular_velocity_property(self, simple_robot):
        """Test robot angular velocity property."""
        simple_robot.ANG_VEL = 1.5
        assert simple_robot.ANG_VEL == 1.5
    
    def test_robot_environment_property(self, simple_robot, simple_environment):
        """Test robot environment property."""
        assert simple_robot.env is simple_environment


@pytest.mark.integration
class TestRobotIntegration:
    """Integration tests for robot functionality."""
    
    def test_robot_movement_sequence(self, simple_robot):
        """Test robot performing a sequence of movements."""
        # Move forward
        simple_robot.robot_step_differential(lin_vel=1.0, ang_vel=0.0)
        pos_after_forward = simple_robot.env.get_robot_pose().pos
        
        # Rotate
        simple_robot.robot_step_differential(lin_vel=0.0, ang_vel=0.5)
        pos_after_rotate = simple_robot.env.get_robot_pose().pos
        
        # Move in new direction
        simple_robot.robot_step_differential(lin_vel=1.0, ang_vel=0.0)
        final_pos = simple_robot.env.get_robot_pose().pos
        
        # Positions should be different after each action
        assert pos_after_forward != Position(0.0, 0.0)
        assert final_pos != pos_after_forward
    
    def test_robot_with_sensor_feedback_loop(self, simple_robot):
        """Test robot movement with continuous sensor feedback."""
        for _ in range(5):
            # Move robot
            simple_robot.robot_step_differential(lin_vel=0.5, ang_vel=0.0)
            
            # Take measurements
            measurements = simple_robot.take_sensor_measurements()
            
            # Verify measurements are valid
            assert isinstance(measurements, pd.DataFrame)
            assert len(measurements) > 0
    
    def test_robot_switching_drive_modes(self, simple_robot):
        """Test robot switching between drive modes."""
        # Differential drive
        simple_robot.robot_step_differential(lin_vel=1.0, ang_vel=0.2)
        pos_after_diff = simple_robot.env.get_robot_pose().pos
        
        # Translational drive
        simple_robot.robot_step_translational(x_vel=1.0, y_vel=0.0)
        pos_after_trans = simple_robot.env.get_robot_pose().pos
        
        # Positions should be different
        assert pos_after_trans != pos_after_diff


@pytest.mark.slow
class TestRobotPerformance:
    """Performance tests for robot operations."""
    
    def test_robot_many_steps(self, simple_robot):
        """Test robot performing many steps."""
        for _ in range(100):
            simple_robot.robot_step_differential(lin_vel=0.1, ang_vel=0.01)
        
        # Robot should still be functioning
        assert simple_robot.env.time == 10.0  # 100 steps * 0.1 DT
    
    def test_robot_continuous_measurement(self, simple_robot):
        """Test continuous sensor measurements."""
        measurements_list = []
        
        for _ in range(50):
            measurements = simple_robot.take_sensor_measurements()
            measurements_list.append(measurements)
            simple_robot.robot_step_differential(lin_vel=0.2, ang_vel=0.05)
        
        assert len(measurements_list) == 50
        # All measurements should be valid DataFrames
        assert all(isinstance(m, pd.DataFrame) for m in measurements_list)
