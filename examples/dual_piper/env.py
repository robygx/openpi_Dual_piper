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
    ):
        """Initialize the DualPiper environment.

        Args:
            can_left: CAN interface name for left arm
            can_right: CAN interface name for right arm
            render_height: Image height for policy input
            render_width: Image width for policy input
            velocity: Movement speed percentage (0-100)
            reset_position: Optional reset position [left_arm_6, right_arm_6]
            camera_serials: Optional dict mapping camera names to RealSense serial numbers
                {"cam_high": "xxx", "cam_left_wrist": "yyy", "cam_right_wrist": "zzz"}
                If None, will auto-discover cameras.
        """
        self._render_height = render_height
        self._render_width = render_width

        # Create Piper hardware interface
        self._env = _piper_env.create_piper_dual_arm(
            can_left=can_left,
            can_right=can_right,
            velocity=velocity,
        )

        # Set reset position (default is zero position)
        self._reset_position = reset_position

        # Initialize RealSense cameras
        self._init_cameras(camera_serials)

        logger.info("DualPiperRealEnvironment initialized")

    def _init_cameras(self, camera_serials: Optional[Dict[str, str]]):
        """Initialize RealSense cameras.

        Args:
            camera_serials: Dict mapping camera names to serial numbers.
                If None, will auto-discover connected cameras.
        """
        # Define expected cameras
        camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

        if camera_serials is None:
            # Auto-discover cameras
            logger.info("Auto-discovering RealSense cameras...")
            discovered = _realsense.discover_cameras()
            if len(discovered) < len(camera_names):
                raise RuntimeError(
                    f"Found {len(discovered)} camera(s), need {len(camera_names)}. "
                    "Please specify camera_serials or ensure all cameras are connected."
                )

            # Map discovered cameras to names
            camera_configs = {}
            for i, name in enumerate(camera_names):
                serial = discovered.get(f"cam_{i}")
                if serial:
                    camera_configs[name] = {"serial": serial}
                else:
                    raise RuntimeError(f"Not enough cameras discovered for {name}")
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
            Dictionary with "state" and "images" keys formatted for the policy.
        """
        # Get joint positions from Piper arms
        state = self._env.get_joint_positions()  # (14,)

        # Get images from RealSense cameras
        images = self._cameras.get_images()

        # Preprocess images: resize to 224x224, convert to uint8, and CHW format
        for cam_name in images:
            img = images[cam_name]
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, self._render_height, self._render_width)
            )
            # Convert HWC to CHW format (as expected by the policy)
            images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": state,
            "images": images,
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
        """
        actions = action["actions"]
        self._env.set_joint_positions(actions)
