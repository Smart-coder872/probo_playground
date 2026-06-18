"""
Unit tests for the Extended Kalman Filter module.

Tests the ExtendedKalmanFilter class including prediction and update steps
with nonlinear motion models, state propagation, and covariance operations.
"""

import pytest
import numpy as np
from src.utils import Pose, Position
from src.extended_kalman_filter import ExtendedKalmanFilter


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterInitialization:
    """Test cases for Extended Kalman Filter initialization."""
    
    def test_ekf_creation(self, simple_extended_kalman_filter):
        """Test basic Extended Kalman filter creation."""
        assert simple_extended_kalman_filter.DT == 0.1
        assert simple_extended_kalman_filter.P is not None
    
    def test_ekf_state_initialization(self, simple_extended_kalman_filter):
        """Test that EKF is initialized with correct state."""
        assert simple_extended_kalman_filter.x is not None
    
    def test_ekf_process_model_identity(self, simple_extended_kalman_filter):
        """Test that process model starts as identity."""
        expected_P = np.eye(3)
        np.testing.assert_array_almost_equal(simple_extended_kalman_filter.P, expected_P)
    
    def test_ekf_symbolic_models_exist(self, simple_extended_kalman_filter):
        """Test that symbolic motion and Jacobian models are defined."""
        assert simple_extended_kalman_filter.f_xu is not None
        assert simple_extended_kalman_filter.F is not None
    
    def test_ekf_substitution_dict(self, simple_extended_kalman_filter):
        """Test that substitution dictionary exists."""
        assert simple_extended_kalman_filter.subs is not None
        assert isinstance(simple_extended_kalman_filter.subs, dict)


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterPredict:
    """Test cases for Extended Kalman Filter prediction step."""
    
    def test_predict_zero_input(self, simple_extended_kalman_filter):
        """Test prediction with zero control input."""
        u = np.array([0.0, 0.0])  # [v, w]
        
        initial_state = simple_extended_kalman_filter.x_state_ef.copy() if hasattr(simple_extended_kalman_filter, 'x_state_ef') else None
        new_state, new_P = simple_extended_kalman_filter.predict(u)
        
        assert new_state is not None
        assert new_P is not None
    
    def test_predict_with_linear_velocity(self, simple_extended_kalman_filter):
        """Test prediction with linear velocity input."""
        u = np.array([1.0, 0.0])  # v=1.0, w=0
        
        new_state, new_P = simple_extended_kalman_filter.predict(u)
        
        assert new_state is not None
        assert new_P is not None
    
    def test_predict_with_angular_velocity(self, simple_extended_kalman_filter):
        """Test prediction with angular velocity input."""
        u = np.array([0.0, 0.5])  # v=0, w=0.5
        
        new_state, new_P = simple_extended_kalman_filter.predict(u)
        
        assert new_state is not None
        assert new_P is not None
    
    def test_predict_with_both_velocities(self, simple_extended_kalman_filter):
        """Test prediction with both linear and angular velocities."""
        u = np.array([1.0, 0.5])  # v=1.0, w=0.5
        
        new_state, new_P = simple_extended_kalman_filter.predict(u)
        
        assert new_state is not None
        assert new_P is not None
        assert new_state.shape == (3, 1)
        assert new_P.shape == (3, 3)
    
    def test_predict_multiple_steps(self, simple_extended_kalman_filter):
        """Test multiple prediction steps."""
        u = np.array([0.5, 0.1])
        
        for _ in range(5):
            state, P = simple_extended_kalman_filter.predict(u)
            assert state is not None
            assert P is not None
    
    def test_predict_returns_correct_shapes(self, simple_extended_kalman_filter):
        """Test that predict returns correctly shaped matrices."""
        u = np.array([1.0, 0.5])
        
        new_state, new_P = simple_extended_kalman_filter.predict(u)
        
        assert new_state.shape == (3, 1)
        assert new_P.shape == (3, 3)


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterUpdate:
    """Test cases for Extended Kalman Filter update step."""
    
    def test_update_with_observation(self, simple_extended_kalman_filter):
        """Test update step with observation."""
        # Need H and R matrices for landmark observations
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])  # Measurement model
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])  # Measurement noise
        z = np.array([[5.0], [5.0]])  # Observation
        
        updated_state, updated_P = simple_extended_kalman_filter.update(H, R, z, None)
        
        assert updated_state is not None
        assert updated_P is not None
    
    def test_update_with_residual(self, simple_extended_kalman_filter):
        """Test update with pre-computed residual."""
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])
        z = np.array([[0.0], [0.0]])
        y = np.array([[0.5], [0.5]])  # Pre-computed residual
        
        updated_state, updated_P = simple_extended_kalman_filter.update(H, R, z, y)
        
        assert updated_state is not None
        assert updated_P is not None


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterIntegration:
    """Integration tests for Extended Kalman Filter."""
    
    def test_predict_update_cycle(self, simple_extended_kalman_filter):
        """Test a complete predict-update cycle."""
        # Predict
        u = np.array([0.5, 0.1])
        predicted_state, predicted_P = simple_extended_kalman_filter.predict(u)
        
        # Update
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])
        z = np.array([[0.5], [0.5]])
        
        updated_state, updated_P = simple_extended_kalman_filter.update(H, R, z, None)
        
        assert updated_state is not None
        assert updated_P is not None
    
    def test_multiple_predict_update_cycles(self, simple_extended_kalman_filter):
        """Test multiple predict-update cycles."""
        u = np.array([0.1, 0.01])
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])
        
        for i in range(10):
            # Predict
            simple_extended_kalman_filter.predict(u)
            
            # Update with observation
            z = np.array([[float(i) * 0.1], [float(i) * 0.1]])
            simple_extended_kalman_filter.update(H, R, z, None)
        
        # Filter should still be operational
        assert simple_extended_kalman_filter.x_state_ef is not None
    
    @pytest.mark.slow
    def test_track_nonlinear_motion(self, simple_extended_kalman_filter):
        """Test tracking with nonlinear motion model."""
        u = np.array([1.0, 0.3])  # Forward with turning
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])
        
        states = []
        
        for i in range(20):
            # Predict motion
            simple_extended_kalman_filter.predict(u)
            
            # Measurement
            z = np.array([[float(i) * 0.1], [float(i) * 0.05]])
            state, P = simple_extended_kalman_filter.update(H, R, z, None)
            states.append(state.copy())
        
        assert len(states) == 20


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterMatrices:
    """Test cases for Extended Kalman Filter matrices."""
    
    def test_F_jacobian_matrix_exists(self, simple_extended_kalman_filter):
        """Test that Jacobian matrix is defined."""
        assert simple_extended_kalman_filter.F is not None
    
    def test_P_matrix_symmetric(self, simple_extended_kalman_filter):
        """Test that covariance matrix is symmetric."""
        P = simple_extended_kalman_filter.P
        
        # P should be symmetric
        np.testing.assert_array_almost_equal(P, P.T)
    
    def test_P_matrix_positive_definite(self, simple_extended_kalman_filter):
        """Test that covariance matrix is positive definite."""
        P = simple_extended_kalman_filter.P
        
        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvals(P)
        assert np.all(eigenvalues > 0)
    
    def test_Q_matrix_generation(self, simple_extended_kalman_filter):
        """Test process noise matrix generation."""
        Q = simple_extended_kalman_filter.get_Q()
        
        assert Q.shape == (3, 3)


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterEdgeCases:
    """Test edge cases for Extended Kalman Filter."""
    
    def test_ekf_with_very_small_dt(self):
        """Test EKF with very small timestep."""
        pose = Pose(Position(0.0, 0.0), 0.0)
        ekf = ExtendedKalmanFilter(dt=0.001, prior=pose)
        
        u = np.array([1.0, 0.5])
        state, P = ekf.predict(u)
        
        assert state is not None
    
    def test_ekf_with_large_dt(self):
        """Test EKF with large timestep."""
        pose = Pose(Position(0.0, 0.0), 0.0)
        ekf = ExtendedKalmanFilter(dt=1.0, prior=pose)
        
        u = np.array([1.0, 0.5])
        state, P = ekf.predict(u)
        
        assert state is not None
    
    def test_ekf_with_high_angular_velocity(self, simple_extended_kalman_filter):
        """Test EKF with high angular velocity."""
        u = np.array([0.5, 2.0])  # High angular velocity
        
        state, P = simple_extended_kalman_filter.predict(u)
        
        assert state is not None
        assert not np.any(np.isnan(state))
    
    def test_ekf_state_bounds(self, simple_extended_kalman_filter):
        """Test that EKF state remains bounded."""
        u = np.array([1.0, 0.1])
        
        for _ in range(100):
            simple_extended_kalman_filter.predict(u)
        
        # State should not have NaN or Inf values
        assert not np.any(np.isnan(simple_extended_kalman_filter.x_state_ef))
        assert not np.any(np.isinf(simple_extended_kalman_filter.x_state_ef))


@pytest.mark.unit
@pytest.mark.filters
class TestExtendedKalmanFilterNonlinearity:
    """Test cases for nonlinear aspects of Extended Kalman Filter."""
    
    def test_ekf_handles_angle_wrapping(self, simple_extended_kalman_filter):
        """Test that EKF properly handles angle wrapping."""
        # Large angular velocities
        u = np.array([0.0, 2.0])  # Large rotation
        
        for _ in range(5):
            state, P = simple_extended_kalman_filter.predict(u)
        
        # Theta should be wrapped to [-pi, pi]
        theta = state[2, 0]
        assert -np.pi - 0.1 <= theta <= np.pi + 0.1
    
    def test_ekf_nonlinear_motion_integration(self, simple_extended_kalman_filter):
        """Test that EKF integrates nonlinear motion correctly."""
        # Forward and turn motion
        u = np.array([1.0, 0.5])
        
        state1, _ = simple_extended_kalman_filter.predict(u)
        state2, _ = simple_extended_kalman_filter.predict(u)
        
        # Different predictions indicate nonlinear effects are captured
        assert not np.allclose(state1, state2)