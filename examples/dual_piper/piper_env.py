"""Piper dual-arm robot environment using Piper SDK.

This module provides hardware control for the Agilex Piper dual-arm robot.
It handles CAN communication, joint state reading, and actuator control.
"""

import time
import logging
from typing import List, Optional

import numpy as np

try:
    from piper_sdk import C_PiperInterface_V2, LogLevel
except ImportError:
    raise ImportError(
        "piper_sdk is not installed. Install it with: pip install piper_sdk"
    )

logger = logging.getLogger(__name__)

# Piper SDK unit conversions
JOINT_UNIT = 0.001  # Degrees (SDK returns joint angles in 0.001 degree units)
GRIPPER_UNIT = 1e-6  # Meters (SDK returns gripper position in µm)


class PiperSingleArm:
    """Wrapper for a single Piper arm using the SDK."""

    def __init__(
        self,
        can_name: str = "can0",
        judge_flag: bool = False,
        can_auto_init: bool = True,
    ):
        """Initialize a single Piper arm.

        Args:
            can_name: CAN interface name
            judge_flag: Whether to use official CAN module detection
            can_auto_init: Whether to auto-initialize CAN bus
        """
        self.can_name = can_name
        self.piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=judge_flag,
            can_auto_init=can_auto_init,
            logger_level=LogLevel.WARNING,
        )
        self.piper.ConnectPort()
        logger.info(f"Connected to {can_name}")

    def enable(self) -> bool:
        """Enable the robot arm.

        Returns:
            True if successfully enabled
        """
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        logger.info(f"{self.can_name} enabled")
        return True

    def setup_joint_mode(self, velocity: int = 50):
        """Set up joint control mode.

        Args:
            velocity: Movement speed percentage (0-100)
        """
        # CAN command control mode + Joint control mode
        self.piper.MotionCtrl_2(0x01, 0x01, velocity)

    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles.

        Returns:
            Array of 6 joint angles in radians
        """
        arm_joint = self.piper.GetArmJointMsgs()  # Returns ArmJoint object
        # Extract joint angles from ArmMsgFeedBackJointStates object
        joint_state = arm_joint.joint_state
        joints = [
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        ]
        # Convert from 0.001 degrees to radians
        return np.array(joints, dtype=np.float64) * JOINT_UNIT * np.pi / 180

    def get_gripper_position(self) -> float:
        """Get gripper position.

        Returns:
            Gripper position in normalized [0, 1] range (0=open, 1=closed)
        """
        arm_gripper = self.piper.GetArmGripperMsgs()  # Returns ArmGripper object
        gripper_state = arm_gripper.gripper_state
        # Convert from µm to meters, then normalize to [0, 1]
        # 0.05 meters = fully closed (1.0), 0.0 meters = fully open (0.0)
        gripper_meters = gripper_state.grippers_angle * GRIPPER_UNIT
        return gripper_meters / 0.05  # Normalize to [0, 1]

    def set_joint_angles(self, angles: np.ndarray):
        """Set joint angles.

        Args:
            angles: Array of 6 joint angles in radians
        """
        if len(angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(angles)}")

        # Convert from radians to 0.001 degrees
        angles_deg = angles * 180 / np.pi / JOINT_UNIT

        # DEBUG: Log what we're sending (only on first call)
        if not hasattr(self, '_set_joint_debug_done'):
            logger.info(f"[{self.can_name}] set_joint_angles DEBUG:")
            logger.info(f"  Input (rad):  {angles}")
            logger.info(f"  Input (deg):  {angles * 180 / np.pi}")
            logger.info(f"  Sending (0.001deg units): {[int(x) for x in angles_deg]}")
            self._set_joint_debug_done = True

        self.piper.JointCtrl(*[int(x) for x in angles_deg])

    def set_gripper(self, position: float, torque: int = 1000):
        """Set gripper position.

        Args:
            position: Gripper position in meters (0-1 range)
            torque: Gripper torque in mN/m
        """
        # Convert from meters to µm
        position_um = int(position / GRIPPER_UNIT)
        self.piper.GripperCtrl(abs(position_um), torque, 0x01, 0)

    def reset_to_zero(self):
        """Reset arm to zero position."""
        zero_joints = [0] * 6
        self.set_joint_angles(np.array(zero_joints))
        time.sleep(1)


class PiperDualArmEnv:
    """Environment for Piper dual-arm robot.

    Action space:  [left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
    Observation space: Same as action space (current joint state)

    Units:
    - Joint angles: radians (internal SDK uses 0.001 degrees)
    - Gripper: meters (internal SDK uses µm)
    """

    # Default reset position for both arms
    RESET_POSITION: List[float] = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

    def __init__(
        self,
        can_left: str = "can_left",
        can_right: str = "can_right",
        velocity: int = 50,
    ):
        """Initialize dual-arm Piper environment.

        Args:
            can_left: CAN interface name for left arm
            can_right: CAN interface name for right arm
            velocity: Movement speed percentage (0-100)
        """
        logger.info(f"Initializing Piper dual-arm: left={can_left}, right={can_right}")

        # Initialize left arm
        self.left_arm = PiperSingleArm(can_left)
        self.left_arm.enable()
        self.left_arm.setup_joint_mode(velocity)

        # Initialize right arm
        self.right_arm = PiperSingleArm(can_right)
        self.right_arm.enable()
        self.right_arm.setup_joint_mode(velocity)

        logger.info("Piper dual-arm initialized")

    def get_joint_positions(self) -> np.ndarray:
        """Get joint positions for both arms.

        Returns:
            Array of shape (14,) containing:
            [left_arm_6_joints, left_gripper, right_arm_6_joints, right_gripper]
            All in radians (or meters for gripper)
        """
        left_joints = self.left_arm.get_joint_angles()  # (6,)
        left_gripper = self.left_arm.get_gripper_position()  # scalar

        right_joints = self.right_arm.get_joint_angles()  # (6,)
        right_gripper = self.right_arm.get_gripper_position()  # scalar

        return np.concatenate([
            left_joints,
            [left_gripper],
            right_joints,
            [right_gripper],
        ])  # (14,)

    def set_joint_positions(self, action: np.ndarray):
        """Set joint positions for both arms.

        Args:
            action: Array of shape (14,) containing:
                [left_arm_6_joints, left_gripper, right_arm_6_joints, right_gripper]
                All in radians (or meters for gripper)
        """
        if action.shape != (14,):
            raise ValueError(f"Expected action shape (14,), got {action.shape}")

        # Split into left and right arms
        left_joints = action[:6]      # Left arm: 6 joints
        left_gripper = action[6]      # Left gripper
        right_joints = action[7:13]   # Right arm: 6 joints
        right_gripper = action[13]    # Right gripper

        # Send commands
        self.left_arm.set_joint_angles(left_joints)
        self.left_arm.set_gripper(left_gripper)

        self.right_arm.set_joint_angles(right_joints)
        self.right_arm.set_gripper(right_gripper)

    def reset(self, reset_position: Optional[List[float]] = None):
        """Reset both arms to specified or default position.

        Args:
            reset_position: Optional list of 14 positions [left_arm_6, left_gripper, right_arm_6, right_gripper].
                - Joint angles: in radians
                - Gripper positions: normalized [0, 1] (0=open, 1=closed)
                If None, resets to zero position (joints only, grippers not changed).
        """
        if reset_position is not None:
            if len(reset_position) != 14:
                raise ValueError(f"reset_position must have 14 values, got {len(reset_position)}")

            # Extract positions
            left_joints = reset_position[:6]
            left_gripper = reset_position[6]
            right_joints = reset_position[7:13]
            right_gripper = reset_position[13]

            # Convert gripper from normalized [0, 1] to meters [0, 0.05]
            # 0.0 = open, 1.0 = closed
            left_gripper_m = left_gripper * 0.05
            right_gripper_m = right_gripper * 0.05

            logger.info(f"Resetting arms to custom position:")
            logger.info(f"  Left joints: {left_joints}")
            logger.info(f"  Left gripper: {left_gripper} (normalized) = {left_gripper_m:.4f}m")
            logger.info(f"  Right joints: {right_joints}")
            logger.info(f"  Right gripper: {right_gripper} (normalized) = {right_gripper_m:.4f}m")
        else:
            # Reset joints to zero, leave grippers unchanged
            left_joints = [0.0] * 6
            right_joints = [0.0] * 6
            left_gripper_m = None
            right_gripper_m = None
            logger.info("Resetting arms to zero position (grippers unchanged)")

        # Reset joint angles
        self.left_arm.set_joint_angles(np.array(left_joints))
        self.right_arm.set_joint_angles(np.array(right_joints))

        # Reset grippers if specified
        if left_gripper_m is not None:
            self.left_arm.set_gripper(left_gripper_m)
        if right_gripper_m is not None:
            self.right_arm.set_gripper(right_gripper_m)

        time.sleep(0.5)


def create_piper_dual_arm(
    can_left: str = "can_left",
    can_right: str = "can_right",
    velocity: int = 50,
) -> PiperDualArmEnv:
    """Factory function to create a Piper dual-arm environment.

    Args:
        can_left: CAN interface name for left arm
        can_right: CAN interface name for right arm
        velocity: Movement speed percentage (0-100)

    Returns:
        PiperDualArmEnv instance
    """
    return PiperDualArmEnv(
        can_left=can_left,
        can_right=can_right,
        velocity=velocity,
    )
