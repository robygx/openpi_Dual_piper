"""Environment adapter for DualPiper robot with openpi runtime.

This module adapts the Piper hardware to the openpi runtime interface,
handling observation formatting and action execution.
"""

import einops
import logging
from typing import Dict, List, Optional

import numpy as np

from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.dual_piper import piper_env as _piper_env
from examples.dual_piper import realsense_camera as _realsense
from examples.dual_piper import constants as _constants

logger = logging.getLogger(__name__)


class DualPiperRealEnvironment(_environment.Environment):
    """DualPiper real robot environment compatible with openpi runtime.

    This environment:
    - Reads joint states from Piper arms
    - Captures images from cameras
    - Formats observations for the policy server
    - Executes actions received from the policy server

    Observation format:
        state: (14,) array [left_arm_6, left_gripper, right_arm_6, right_gripper]
        images: dict with keys "cam_high", "cam_left_wrist", "cam_right_wrist"

    Action format:
        actions: (14,) array [left_arm_6, left_gripper, right_arm_6, right_gripper]
    """

    def __init__(
        self,
        can_left: str = "can_left",
        can_right: str = "can_right",
        render_height: int = 224,
        render_width: int = 224,
        velocity: int = 50,
        reset_position: Optional[List[float]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        prompt: str = "stack the blocks",
    ):
        """Initialize the DualPiper environment.

        Args:
            can_left: CAN interface name for left arm
            can_right: CAN interface name for right arm
            render_height: Image height for policy input
            render_width: Image width for policy input
            velocity: Movement speed percentage (0-100)
            reset_position: Optional reset position (14-dim: [left_arm_6, left_gripper, right_arm_6, right_gripper])
                If None, uses constants.RESET_POSITION (dataset start position).
            camera_serials: Optional dict mapping camera names to RealSense serial numbers
                {"cam_high": "xxx", "cam_left_wrist": "yyy", "cam_right_wrist": "zzz"}
                If None, will auto-discover cameras.
            prompt: Task description prompt for the policy
        """
        self._render_height = render_height
        self._render_width = render_width
        self._prompt = prompt

        # Create Piper hardware interface
        self._env = _piper_env.create_piper_dual_arm(
            can_left=can_left,
            can_right=can_right,
            velocity=velocity,
        )

        # Set reset position (default to dataset start position from constants)
        if reset_position is None:
            self._reset_position = _constants.CONSTANTS.RESET_POSITION.tolist()
        else:
            self._reset_position = reset_position

        # Initialize RealSense cameras
        self._init_cameras(camera_serials)

        logger.info(f"DualPiperRealEnvironment initialized, reset_position={self._reset_position}")

    def _init_cameras(self, camera_serials: Optional[Dict[str, str]]):
        """Initialize RealSense cameras.

        Args:
            camera_serials: Dict mapping camera names to serial numbers.
                If None, will auto-discover connected cameras.
        """
        # Define expected cameras
        camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

        if camera_serials is None:
            # Use default camera serial numbers
            camera_serials = {
                "cam_high": "152122072933",
                "cam_left_wrist": "213722070453",
                "cam_right_wrist": "152122076290",
            }
            logger.info("Using default camera serial numbers...")

        camera_configs = {}
        for name in camera_names:
            serial = camera_serials.get(name)
            if serial:
                camera_configs[name] = {"serial": serial}
            else:
                raise RuntimeError(f"Camera serial not provided for {name}")
        else:
            # Use provided serial numbers
            camera_configs = {
                name: {"serial": camera_serials.get(name)}
                for name in camera_names
            }

        # Log camera configuration
        for name, config in camera_configs.items():
            logger.info(f"{name}: serial={config.get('serial', 'auto')}")

        # Create multi-camera wrapper
        self._cameras = _realsense.RealSenseMultiCamera(camera_configs)
        logger.info("RealSense cameras initialized")

    @override
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self._env.reset(reset_position=self._reset_position)
        # TODO: Reset cameras if needed

    @override
    def is_episode_complete(self) -> bool:
        """Check if the current episode is complete.

        For real robots, episodes typically run until max_steps is reached.
        """
        return False

    @override
    def get_observation(self) -> dict:
        """Get the current observation from the robot.

        Returns:
            Dictionary with "state", "images", and "prompt" keys formatted for the policy.
        """
        # Get joint positions from Piper arms
        state = self._env.get_joint_positions()  # (14,)

        # DEBUG: Print observation details on first call
        if not hasattr(self, '_obs_debug_done'):
            expected_start = _constants.CONSTANTS.RESET_POSITION
            logger.info(f"\n{'='*60}")
            logger.info(f"=== OBSERVATION DEBUG ===")
            logger.info(f"{'='*60}")
            logger.info(f"State shape: {state.shape}, dtype: {state.dtype}")
            logger.info(f"State: {state}")
            logger.info(f"")
            logger.info(f"Expected (dataset start): {expected_start[:7]}...")
            logger.info(f"")
            logger.info(f"Comparison:")
            logger.info(f"  Left arm joints (0-5):")
            for i in range(6):
                match = "✓" if abs(state[i] - expected_start[i]) < 0.01 else "✗"
                logger.info(f"    Joint {i}: {state[i]:.4f} (expected {expected_start[i]:.4f}) {match}")
            logger.info(f"  Left gripper (6): {state[6]:.4f} (expected {expected_start[6]:.4f})")
            logger.info(f"")
            logger.info(f"State matches dataset: {np.allclose(state[:7], expected_start[:7], atol=0.01)}")
            logger.info(f"{'='*60}\n")
            self._obs_debug_done = True

        # Get images from RealSense cameras
        images = self._cameras.get_images()

        # DEBUG: Print image shapes on first call
        if not hasattr(self, '_img_debug_done'):
            logger.info(f"=== IMAGE DEBUG ===")
            for cam_name, img in images.items():
                logger.info(f"{cam_name}: original shape={img.shape}, dtype={img.dtype}")
            logger.info(f"")
            self._img_debug_done = True

        # Preprocess images: resize to 224x224, convert to uint8, and CHW format
        for cam_name in images:
            img = images[cam_name]
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, self._render_height, self._render_width)
            )
            # Convert HWC to CHW format (as expected by the policy)
            images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        # DEBUG: Verify final image format
        if not hasattr(self, '_img_final_debug_done'):
            logger.info(f"=== IMAGE FINAL FORMAT ===")
            for cam_name, img in images.items():
                logger.info(f"{cam_name}: final shape={img.shape}, dtype={img.dtype}")
            logger.info(f"{'='*60}\n")
            self._img_final_debug_done = True

        return {
            "state": state,
            "images": images,
            "prompt": self._prompt,
        }

    def _get_camera_image(self, camera_name: str) -> np.ndarray:
        """Get image from a camera.

        This is now handled by self._cameras.get_images() in get_observation().

        Args:
            camera_name: Name of the camera

        Returns:
            Image array in HWC format, uint8 dtype
        """
        # Images are now captured in batch by get_images()
        # This method is kept for backward compatibility
        all_images = self._cameras.get_images()
        return all_images.get(camera_name)

    @override
    def apply_action(self, action: dict) -> None:
        """Apply an action to the robot.

        Args:
            action: Dictionary with "actions" key containing (14,) array
                Format: [left_arm_6_joints, left_gripper, right_arm_6_joints, right_gripper]
                - Joint angles: radians
                - Gripper positions: normalized [0.0, 1.0] from Pi0 model
        """
        actions = action["actions"].copy()  # (14,)

        # DEBUG: Print first 5 steps with detailed info
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0

        if self._debug_count < 5:
            # Get current position
            current = self._env.get_joint_positions()

            logger.info(f"\n{'='*60}")
            logger.info(f"=== ACTION DEBUG (Step {self._debug_count + 1}) ===")
            logger.info(f"{'='*60}")
            logger.info(f"Current state:  {current}")
            logger.info(f"")
            logger.info(f"Received action: {actions}")
            logger.info(f"Action dtype: {actions.dtype}, shape: {actions.shape}")
            logger.info(f"")

            # Check joint changes (should be small deltas)
            left_arm_delta = actions[:6] - current[:6]
            right_arm_delta = actions[7:13] - current[7:13]
            logger.info(f"Joint delta (degrees):")
            logger.info(f"  Left arm:  {np.degrees(left_arm_delta)}")
            logger.info(f"  Right arm: {np.degrees(right_arm_delta)}")
            logger.info(f"")

            # Check gripper values before transformation
            logger.info(f"Gripper BEFORE transform:")
            logger.info(f"  Left gripper:  {actions[6]:.6f} (normalized 0-1)")
            logger.info(f"  Right gripper: {actions[13]:.6f} (normalized 0-1)")
            logger.info(f"")

            # Transform gripper values
            actions[6] = actions[6] * 0.05  # Left gripper
            actions[13] = actions[13] * 0.05  # Right gripper

            logger.info(f"Gripper AFTER transform:")
            logger.info(f"  Left gripper:  {actions[6]:.6f} meters")
            logger.info(f"  Right gripper: {actions[13]:.6f} meters")
            logger.info(f"")

            # Check if values are reasonable
            logger.info(f"Sanity checks:")
            logger.info(f"  Joint delta reasonable (< 10 deg): {np.all(np.abs(np.degrees(left_arm_delta)) < 10)}")
            logger.info(f"  Gripper in range [0, 0.05]: {0 <= actions[6] <= 0.05 and 0 <= actions[13] <= 0.05}")
            logger.info(f"{'='*60}\n")

            self._debug_count += 1

        # Convert gripper positions from normalized [0, 1] to meters [0, 0.05]
        # Pi0 model outputs: 0.0 = open, 1.0 = closed
        # Piper hardware expects: meters (0.0 = open, 0.05 = closed)
        actions[6] = actions[6] * 0.05  # Left gripper
        actions[13] = actions[13] * 0.05  # Right gripper

        self._env.set_joint_positions(actions)
