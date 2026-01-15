"""DualPiper robot policy - Inputs/Outputs transformations for training and inference.

This policy is designed for dual-arm robots similar to ALOHA but with 3 cameras instead of 4.
Key differences from ALOHA:
- No cam_low camera (only cam_high, cam_left_wrist, cam_right_wrist)
- Simplified version without ALOHA-specific joint/gripper transformations
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_dual_piper_example() -> dict:
    """Creates a random input example for the DualPiper policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualPiperInputs(transforms.DataTransformFn):
    """Inputs for the DualPiper policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width] or [height, width, channel].
             Must contain cam_high, cam_left_wrist, cam_right_wrist.
    - state: [14] - dual arm state (left arm 7 dims + right arm 7 dims)
    - actions: [action_horizon, 14] - only available during training
    - prompt: str - language instruction
    """

    # The expected cameras. All input cameras must be in this set.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Parse all 3 camera images
        base_image = _parse_image(in_images["cam_high"])
        left_wrist = _parse_image(in_images["cam_left_wrist"])
        right_wrist = _parse_image(in_images["cam_right_wrist"])

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DualPiperOutputs(transforms.DataTransformFn):
    """Outputs for the DualPiper policy.

    This transforms model outputs back to the DualPiper format.
    Only used during inference.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims (dual arm actions).
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
