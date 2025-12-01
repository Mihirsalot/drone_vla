"""Configuration constants for LeRobot dataset creation."""

from pathlib import Path

# Directory paths
DATA_DIR = Path("../AirSim 5M Hover Data")
OUTPUT_DIR = Path("lerobot_dataset")
SYNC_THRESHOLD = 0.1  # seconds

# Observation features in order (15 values total)
OBSERVATION_FEATURES = [ 'imu_linear_acceleration_z',
    'altitude',
]

# Action features
ACTION_FEATURES = ['action_throttle','dummy']

# Task configuration
TASK_NAME = 'takeoff and hover'
TASK_INDEX = 0

