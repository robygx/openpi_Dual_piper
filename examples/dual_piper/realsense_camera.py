"""RealSense D435 camera capture for DualPiper robot.

This module provides image capture from Intel RealSense D435 cameras
using the pyrealsense2 SDK.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError(
        "pyrealsense2 is not installed. Install it with: "
        "pip install pyrealsense2"
    )

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """Wrapper for a single RealSense D435 camera."""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        """Initialize a RealSense D435 camera.

        Args:
            serial_number: Camera serial number (None for any camera)
            width: Image width
            height: Image height
            fps: Frames per second
            enable_depth: Whether to also capture depth
        """
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        # Configure the pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device with specific serial number if provided
        if serial_number:
            self.config.enable_device(serial_number)

        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)

        # Get the profile to align depth to color if needed
        if enable_depth:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        # Skip first few frames to let auto-exposure stabilize
        for _ in range(30):
            self.pipeline.wait_for_frames()

        logger.info(f"RealSense camera initialized: {serial_number or 'any'}")

    def get_color_frame(self) -> np.ndarray:
        """Get a single color frame.

        Returns:
            Image array in HWC format, uint8 dtype
        """
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get color frame")

        # Convert to numpy array (HWC, BGR)
        color_image = np.asanyarray(color_frame.get_data())

        # Convert BGR to RGB
        color_image = color_image[..., ::-1]  # BGR -> RGB

        return color_image

    def get_depth_frame(self) -> np.ndarray:
        """Get a single depth frame.

        Returns:
            Depth image array in HW format
        """
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("Failed to get depth frame")

        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def release(self):
        """Release the camera resources."""
        self.pipeline.stop()


class RealSenseMultiCamera:
    """Wrapper for multiple RealSense cameras.

    This manages multiple RealSense D435 cameras and provides
    synchronized image capture.
    """

    def __init__(
        self,
        camera_configs: Dict[str, dict],
        enable_depth: bool = False,
    ):
        """Initialize multiple RealSense cameras.

        Args:
            camera_configs: Dictionary mapping camera names to their configs
                {
                    "cam_high": {"serial": "...", "width": 640, "height": 480},
                    "cam_left_wrist": {"serial": "...", "width": 640, "height": 480},
                    "cam_right_wrist": {"serial": "...", "width": 640, "height": 480},
                }
                Serial numbers are optional; if not provided, the first available
                camera will be used for each config.
            enable_depth: Whether to also capture depth frames
        """
        self.cameras = {}
        self.enable_depth = enable_depth

        # Initialize each camera
        for cam_name, config in camera_configs.items():
            serial = config.get("serial")
            width = config.get("width", 640)
            height = config.get("height", 480)
            fps = config.get("fps", 30)

            logger.info(f"Initializing {cam_name}...")
            self.cameras[cam_name] = RealSenseCamera(
                serial_number=serial,
                width=width,
                height=height,
                fps=fps,
                enable_depth=enable_depth,
            )

        self.camera_names = list(camera_configs.keys())
        logger.info(f"Initialized {len(self.cameras)} RealSense cameras")

    def get_images(self) -> Dict[str, np.ndarray]:
        """Get current images from all cameras.

        Returns:
            Dictionary mapping camera names to RGB images (HWC, uint8)
        """
        images = {}
        for cam_name in self.camera_names:
            images[cam_name] = self.cameras[cam_name].get_color_frame()
        return images

    def release(self):
        """Release all camera resources."""
        for cam in self.cameras.values():
            cam.release()
        logger.info("All RealSense cameras released")


def list_connected_cameras() -> Tuple[list, list]:
    """List all connected RealSense cameras.

    Returns:
        Tuple of (serial_numbers, device_names)
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    names = [d.get_info(rs.camera_info.name) for d in devices]
    return serials, names


def discover_cameras() -> Dict[str, str]:
    """Discover connected RealSense cameras and auto-assign names.

    Returns:
        Dictionary mapping camera names to serial numbers
        {
            "cam_0": "<serial1>",
            "cam_1": "<serial2>",
            "cam_2": "<serial3>",
        }
    """
    serials, names = list_connected_cameras()

    if len(serials) == 0:
        logger.warning("No RealSense cameras found!")
        return {}

    result = {}
    for i, (serial, name) in enumerate(zip(serials, names)):
        result[f"cam_{i}"] = serial
        logger.info(f"Found camera {i}: {name} (serial: {serial})")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # List connected cameras
    print("Scanning for RealSense cameras...")
    discovered = discover_cameras()

    # Example usage
    if discovered:
        camera_configs = {
            "cam_high": {"serial": discovered.get("cam_0")},
            "cam_left_wrist": {"serial": discovered.get("cam_1")},
            "cam_right_wrist": {"serial": discovered.get("cam_2")},
        }

        multi_cam = RealSenseMultiCamera(camera_configs)

        # Test capture
        print("Capturing test frames...")
        images = multi_cam.get_images()
        for name, img in images.items():
            print(f"{name}: shape={img.shape}, dtype={img.dtype}")

        multi_cam.release()
