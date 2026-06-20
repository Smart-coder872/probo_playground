"""
Shared test fixtures for the robot simulator.
conftest.py is automatically discovered by pytest and provides
fixtures that can be used across multiple test files.
"""

import pytest
import numpy as np
from src.utils import Position, Pose, Bounds, Landmark, BearingRange
from src.environment import Environment
from src.robot import Robot
from src.sensors import GPS, WheelEncoder, LandmarkPinger
from src.kalman_filter import KalmanFilter
from src.extended_kalman_filter import ExtendedKalmanFilter


# ____________________________________________________________________________
# POSITION AND POSE FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def origin_position():
    """Create a position at the origin."""
    return Position(x=0.0, y=0.0)


@pytest.fixture
def test_position():
    """Create a test position."""
    return Position(x=5.0, y=5.0)


@pytest.fixture
def origin_pose():
    """Create a pose at the origin facing right."""
    return Pose(pos=Position(0.0, 0.0), theta=0.0)


@pytest.fixture
def test_pose():
    """Create a test pose."""
    return Pose(pos=Position(3.0, 4.0), theta=np.pi / 4)


# ____________________________________________________________________________
# BOUNDS AND LANDMARK FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def world_bounds():
    """Create bounds for a 10x10 world."""
    return Bounds(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)


@pytest.fixture
def obstacle_bounds():
    """Create bounds for a simple obstacle."""
    return Bounds(x_min=4.0, x_max=6.0, y_min=4.0, y_max=6.0)


@pytest.fixture
def sample_landmarks():
    """Create a list of sample landmarks."""
    return [
        Landmark(Position(2.0, 2.0), id=0),
        Landmark(Position(5.0, 5.0), id=1),
        Landmark(Position(8.0, 8.0), id=2),
    ]


@pytest.fixture
def bearing_range():
    """Create a sample bearing/range measurement."""
    return BearingRange(landmark_id=0, bearing=0.5, range=3.0)


# ____________________________________________________________________________
# ENVIRONMENT FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def simple_environment(world_bounds, sample_landmarks, origin_pose):
    """Create a simple environment for testing."""
    env = Environment(
        dimensions=world_bounds,
        dt=0.1,
        obstacles=[],
        landmarks=sample_landmarks,
        robot_starting_pose=origin_pose,
        bearing=10.0,  # max range for landmarks
    )
    return env


@pytest.fixture
def environment_with_obstacles(world_bounds, sample_landmarks, origin_pose, obstacle_bounds):
    """Create an environment with obstacles."""
    env = Environment(
        dimensions=world_bounds,
        dt=0.1,
        obstacles=[obstacle_bounds],
        landmarks=sample_landmarks,
        robot_starting_pose=origin_pose,
        bearing=10.0,
    )
    return env


# ____________________________________________________________________________
# ROBOT FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def simple_robot(simple_environment):
    """Create a simple robot in a test environment."""
    return Robot(simple_environment)


@pytest.fixture
def robot_with_obstacles(environment_with_obstacles):
    """Create a robot in an environment with obstacles."""
    return Robot(environment_with_obstacles)


# ____________________________________________________________________________
# KALMAN FILTER FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def kalman_filter_state():
    """Create an initial state vector for Kalman filtering."""
    return np.array([[0.0], [0.0], [0.0]])  # [x, y, theta]


@pytest.fixture
def simple_kalman_filter(kalman_filter_state):
    """Create a simple Kalman filter for testing."""
    return KalmanFilter(dt=0.1, prior=kalman_filter_state)


@pytest.fixture
def simple_extended_kalman_filter(kalman_filter_state):
    """Create an Extended Kalman filter for testing."""
    pose = Pose(Position(0.0, 0.0), 0.0)
    return ExtendedKalmanFilter(dt=0.1, prior=pose)


# ____________________________________________________________________________
# SENSOR FIXTURES
# ____________________________________________________________________________

@pytest.fixture
def gps_sensor(simple_robot):
    """Create a GPS sensor."""
    return GPS(robot=simple_robot, x_noise=0.2, y_noise=0.2)


@pytest.fixture
def wheel_encoder_sensor(simple_robot):
    """Create a wheel encoder sensor."""
    return WheelEncoder(robot=simple_robot, lin_noise=0.05, ang_noise=0.03)


@pytest.fixture
def landmark_pinger_sensor(simple_robot):
    """Create a landmark pinger sensor."""
    return LandmarkPinger(
        robot=simple_robot,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=np.pi / 6,
        max_range=10.0,
    )


# ____________________________________________________________________________
# PARAMETRIZED FIXTURES (For testing multiple scenarios)
# ____________________________________________________________________________

@pytest.fixture(params=[
    Position(0.0, 0.0),
    Position(5.0, 5.0),
    Position(9.99, 9.99),
])
def various_positions(request):
    """Parametrized fixture providing multiple test positions."""
    return request.param


@pytest.fixture(params=[0.0, np.pi/4, np.pi/2, np.pi, 2*np.pi])
def various_angles(request):
    """Parametrized fixture providing multiple test angles."""
    return request.param


# ____________________________________________________________________________
# CLEANUP AND UTILITIES
# ____________________________________________________________________________

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup (if needed)


@pytest.fixture
def numpy_tolerance():
    """Provide tolerance for numpy float comparisons."""
    return 1e-6