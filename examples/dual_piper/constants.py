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
    # Based on ALOHA reset pose: [0, -0.96, 1.16, 0, -0.3, 0]
    LEFT_ARM_RESET = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    RIGHT_ARM_RESET = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])

    # Combined reset position [left_arm_6, left_gripper, right_arm_6, right_gripper]
    RESET_POSITION = np.concatenate([
        LEFT_ARM_RESET,
        [GRIPPER_MAX],  # Left gripper open
        RIGHT_ARM_RESET,
        [GRIPPER_MAX],  # Right gripper open
    ])


# Default instance
CONSTANTS = DualPiperConstants()
