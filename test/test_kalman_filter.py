import numpy as np
import pytest
from numpy import ndarray
from src.kalman_filter import KalmanFilter  # replace with actual module name

class TestKalmanFilterInit:
    """Tests for KalmanFilter initialization."""
    
    def test_init_sets_dt(self):
        """Test that dt is correctly set."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        assert kf.DT == 0.5
    
    def test_init_sets_state_vector(self):
        """Test that state vector is correctly set."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        np.testing.assert_array_equal(kf.x_state_kf, prior)
    
    def test_init_P_is_identity(self):
        """Test that process model P is identity matrix."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        np.testing.assert_array_equal(kf.P, np.eye(3))
    
    def test_init_F_is_identity(self):
        """Test that state transition matrix F is identity matrix."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        np.testing.assert_array_equal(kf.F, np.eye(3))
    
    def test_init_B_shape(self):
        """Test that control model B has correct shape."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        assert kf.B.shape == (3, 3)
    
    def test_init_B_values_depend_on_dt(self):
        """Test that B matrix diagonal values equal dt."""
        prior = np.array([1.0, 2.0, 3.0])
        dt = 0.5
        kf = KalmanFilter(dt=dt, prior=prior)
        expected_B = np.eye(3) * dt
        np.testing.assert_array_almost_equal(kf.B, expected_B)
    
    def test_init_Q_shape(self):
        """Test that process noise Q has correct shape."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        assert kf.Q.shape == (3, 3)


class TestKalmanFilterPredict:
    """Tests for KalmanFilter.predict method."""
    
    def test_predict_returns_tuple(self):
        """Test that predict returns (state, covariance) tuple."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        u = np.array([0.1, 0.2, 0.3])
        result = kf.predict(u)
        assert len(result) == 2
    
    def test_predict_updates_state_with_control(self):
        """Test that state is updated using F*x + B*u."""
        prior = np.array([1.0, 2.0, 3.0])
        dt = 0.5
        kf = KalmanFilter(dt=dt, prior=prior)
        u = np.array([1.0, 0.0, 0.0])
        
        # Since F is identity, new state = x + B*u
        expected_state = prior + kf.B @ u.reshape(3, 1)
        new_state, _ = kf.predict(u)
        
        np.testing.assert_array_almost_equal(new_state.flatten(), expected_state.flatten())
    
    def test_predict_updates_covariance(self):
        """Test that covariance is updated using F*P*F.T + Q."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        u = np.array([0.1, 0.2, 0.3])
        
        initial_P = kf.P.copy()
        _, new_P = kf.predict(u)
        
        # P should change (even with identity F, Q is added)
        assert not np.testing.assert_array_equal(new_P, initial_P)
    
    def test_predict_with_zero_control(self):
        """Test predict with zero control input."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        u = np.array([0.0, 0.0, 0.0])
        
        new_state, new_P = kf.predict(u)
        
        # With zero control and identity F, state should remain similar (only Q affects P)
        np.testing.assert_array_almost_equal(new_state.flatten(), prior.flatten())


class TestKalmanFilterUpdate:
    """Tests for KalmanFilter.update method."""
    
    def test_update_returns_tuple(self):
        """Test that update returns (state, covariance) tuple."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        z = np.array([1.1, 2.1, 3.1])
        H = np.eye(3)
        R = np.eye(3) * 0.1
        
        result = kf.update(z, H, R)
        assert len(result) == 2
    
    def test_update_with_perfect_measurement(self):
        """Test update when measurement matches state exactly."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        z = prior.copy()  # perfect measurement
        H = np.eye(3)
        R = np.eye(3) * 0.01  # low measurement noise
        
        new_state, _ = kf.update(z, H, R)
        
        # State should stay close to measurement with low R
        np.testing.assert_array_almost_equal(new_state.flatten(), z.flatten(), decimal=1)
    
    def test_update_with_H_not_identity(self):
        """Test update with non-identity measurement model."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        
        # H only measures first two states
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])
        z = np.array([1.0, 2.0])
        R = np.eye(2) * 0.1
        
        new_state, _ = kf.update(z, H, R)
        
        # State vector should still be 3D
        assert new_state.shape == (3, 1) or new_state.shape == (3,)
    
    def test_update_reduces_uncertainty(self):
        """Test that update reduces covariance with good measurement."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        z = prior.copy()
        H = np.eye(3)
        R = np.eye(3) * 0.01
        
        initial_P = kf.P.copy()
        _, new_P = kf.update(z, H, R)
        
        # Covariance should generally decrease after update
        assert np.trace(new_P) <= np.trace(initial_P)


class TestKalmanFilterGetQ:
    """Tests for KalmanFilter.get_Q method."""
    
    def test_get_Q_returns_3x3(self):
        """Test that get_Q returns 3x3 array."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        Q = kf.get_Q()
        assert Q.shape == (3, 3)
    
    def test_get_Q_has_zero_mean_noise(self):
        """Test that get_Q generates noise around zero (statistical test)."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        
        # Generate multiple Q matrices and check mean
        n_samples = 1000
        Q_sum = np.zeros((3, 3))
        for _ in range(n_samples):
            Q_sum += kf.get_Q()
        
        Q_mean = Q_sum / n_samples
        # Mean should be close to zero (statistical test)
        np.testing.assert_array_almost_equal(Q_mean, np.zeros((3, 3)), decimal=1)
    
    def test_get_Q_standard_deviation(self):
        """Test that get_Q noise has approximately correct standard deviation."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        
        n_samples = 1000
        all_values = []
        for _ in range(n_samples):
            Q = kf.get_Q()
            all_values.extend(Q.flatten())
        
        std = np.std(all_values)
        # Should be close to 0.1 (the stdev in get_Q)
        assert 0.05 < std < 0.15


class TestKalmanFilterFullCycle:
    """Tests for complete predict-update cycles."""
    
    def test_predict_update_cycle(self):
        """Test a complete predict then update cycle."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        
        # Predict
        u = np.array([0.1, 0.1, 0.1])
        pred_state, pred_P = kf.predict(u)
        
        # Update with measurement close to predicted state
        z = pred_state.flatten() + np.array([0.01, 0.01, 0.01])
        H = np.eye(3)
        R = np.eye(3) * 0.1
        
        updated_state, updated_P = kf.update(z, H, R)
        
        # Should have valid state and covariance
        assert updated_state.shape == (3, 1) or updated_state.shape == (3,)
        assert updated_P.shape == (3, 3)
    
    def test_multiple_cycles(self):
        """Test multiple predict-update cycles."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.1, prior=prior)
        
        for i in range(5):
            u = np.array([0.01, 0.01, 0.01])
            pred_state, _ = kf.predict(u)
            
            z = pred_state.flatten() + np.random.randn(3) * 0.01
            H = np.eye(3)
            R = np.eye(3) * 0.01
            
            updated_state, _ = kf.update(z, H, R)
        
        # Should have converged to some valid state
        assert np.all(np.isfinite(updated_state))


class TestKalmanFilterEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_init_with_different_prior_shapes(self):
        """Test initialization with different prior array shapes."""
        prior_1d = np.array([1.0, 2.0, 3.0])
        prior_2d = np.array([[1.0], [2.0], [3.0]])
        
        kf1 = KalmanFilter(dt=0.5, prior=prior_1d)
        kf2 = KalmanFilter(dt=0.5, prior=prior_2d)
        
        # Both should work and have 3D state
        assert kf1.x_state_kf.shape[0] == 3
        assert kf2.x_state_kf.shape[0] == 3
    
    def test_update_with_high_measurement_noise(self):
        """Test update with very high measurement noise."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        z = np.array([5.0, 6.0, 7.0])  # very different from prior
        H = np.eye(3)
        R = np.eye(3) * 100.0  # very high noise
        
        new_state, _ = kf.update(z, H, R)
        
        # With high R, state should stay closer to prior (measurement trusted less)
        assert np.all(np.isfinite(new_state))
    
    def test_predict_with_large_control(self):
        """Test predict with large control input."""
        prior = np.array([1.0, 2.0, 3.0])
        kf = KalmanFilter(dt=0.5, prior=prior)
        u = np.array([100.0, 100.0, 100.0])
        
        new_state, new_P = kf.predict(u)
        
        # State should change significantly
        assert not np.testing.assert_array_almost_equal(new_state.flatten(), prior.flatten())
        assert np.all(np.isfinite(new_state))
        assert np.all(np.isfinite(new_P))


# Helper to make tests deterministic for get_Q
@pytest.fixture
def deterministic_kf():
    """Create KalmanFilter with seeded random for deterministic Q tests."""
    np.random.seed(42)
    prior = np.array([1.0, 2.0, 3.0])
    return KalmanFilter(dt=0.5, prior=prior)