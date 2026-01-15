"""
Script to convert DualPiper data to the LeRobot dataset v2.0 format.

This is a template script for converting DualPiper dual-arm robot data to LeRobot format.
DualPiper specifications:
- 14 dimensional state/action (left arm 7 dims + right arm 7 dims)
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist (no cam_low)
- Absolute actions that will be converted to delta during training

Example usage:
    uv run examples/dual_piper/convert_dual_piper_data_to_lerobot.py \\
        --raw-dir /path/to/raw/data \\
        --repo-id your_username/dual_piper
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str = "dual_piper",
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """
    Create an empty LeRobot dataset for DualPiper.

    Args:
        repo_id: HuggingFace repo ID (e.g., "your_username/dual_piper")
        robot_type: Type of robot (metadata only, doesn't affect training)
        mode: Storage mode for images
        dataset_config: Dataset configuration
    """
    # DualPiper has 14 motors: left arm 6 joints + 1 gripper, right arm 6 joints + 1 gripper
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]

    # DualPiper has 3 cameras (no cam_low unlike ALOHA)
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),  # TODO: Adjust to your camera resolution
            "names": ["channels", "height", "width"],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,  # TODO: Adjust to your data frequency
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def main(
    raw_dir: Path,
    repo_id: str,
    robot_type: str = "dual_piper",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Convert raw DualPiper data to LeRobot format.

    Args:
        raw_dir: Path to raw data directory
        repo_id: HuggingFace repo ID (e.g., "your_username/dual_piper")
        robot_type: Type of robot (metadata only)
        dataset_config: Dataset configuration
    """
    print(f"Converting DualPiper data from {raw_dir} to {repo_id}")

    # Create empty dataset
    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        dataset_config=dataset_config,
    )

    # TODO: Load your raw data and add episodes to the dataset
    # This is highly dependent on your data format.
    # Refer to examples/aloha_real/convert_aloha_data_to_lerobot.py for a complete example.

    print(f"Dataset created successfully at {LEROBOT_HOME / repo_id}")
    print(f"Total episodes: {len(dataset)}")
    print(f"Total frames: {dataset.total_frames}")
    print("\nNext steps:")
    print(f"1. Compute normalization stats: uv run scripts/compute_norm_stats.py --config-name pi05_dual_piper")
    print(f"2. Start training: uv run scripts/train.py pi05_dual_piper --exp-name=dual_piper_finetune --overwrite")


if __name__ == "__main__":
    tyro.cli(main)
