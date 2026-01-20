"""Constants for DualPiper robot."""

import numpy as np
from dataclasses import dataclass


@dataclass
class DualPiperConstants:
    """Constants for the DualPiper robot."""

    # Observation space
    STATE_DIM: int = 14  # [left_arm_6, left_gripper, right_arm_6, right_gripper]

    # Action space (same as state)
    ACTION_DIM: int = 14

    # Camera settings
    IMAGE_HEIGHT: int = 224
    IMAGE_WIDTH: int = 224
    IMAGE_CHANNELS: int = 3

    # Camera names
    CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    # Control frequency
    HZ: int = 50  # 50 Hz control loop
    DT: float = 1.0 / 50.0

    # Joint limits (in radians, for reference)
    # These are typical limits for a 6-DOF arm
    JOINT_LIMITS_LOW = np.array([-np.pi] * 6)
    JOINT_LIMITS_HIGH = np.array([np.pi] * 6)

    # Gripper limits
    GRIPPER_MIN = 0.0  # Closed
    GRIPPER_MAX = 0.1  # Open (10 cm)

    # Reset positions (in radians)
    # Based on stack_blocks_three dataset analysis
    # Computed from mean of all episode start positions
    LEFT_ARM_RESET = np.array([0.0566, 0.0001, 0.2001, 0.0307, 0.0853, 0.0495])
    RIGHT_ARM_RESET = np.array([0.0564, 0.0009, 0.2078, 0.0118, 0.0853, -0.0056])

    # Combined reset position [left_arm_6, left_gripper, right_arm_6, right_gripper]
    # Note: Training data shows grippers start in CLOSED position (~0.99)
    RESET_POSITION = np.concatenate([
        LEFT_ARM_RESET,
        [0.9965],  # Left gripper closed (matches training data)
        RIGHT_ARM_RESET,
        [0.9931],  # Right gripper closed (matches training data)
    ])


# Default instance
CONSTANTS = DualPiperConstants()
