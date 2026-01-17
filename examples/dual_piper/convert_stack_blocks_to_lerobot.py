"""
Convert Stack Blocks HDF5 data to LeRobot format.

Stack Blocks dataset:
- 383 episodes from 4 subdirectories merged into one task
- 3 cameras: cam_head (→ cam_high), cam_left_wrist, cam_right_wrist
- 14 DOF: left arm (6 joints + gripper) + right arm (6 joints + gripper)
- Images are raw RGB uint8 (not JPEG compressed like X-VLA)
- Action data: computed as next frame's qpos (absolute position)

Data subdirectories:
- stack_blocks_three (97 episodes)
- stack_blocks_three_1 (96 episodes)
- stack_blocks_three_2 (92 episodes)
- 4 subdirectories → 1 dataset

Example usage:
    uv run examples/dual_piper/convert_stack_blocks_to_lerobot.py
"""
import shutil
from pathlib import Path
from typing import List

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

REPO_ID = "ygx/stack_blocks_three"
RAW_DIR = Path("/data0/ygx_data/raw_data/stack_blocks_three_hdf5")

# 4 subdirectories to merge
SUBDIRS = [
    "stack_blocks_three",
    "stack_blocks_three_1",
    "stack_blocks_three_2",
    "stack_blocks_three_3",
]


def main(
    raw_dir: Path = RAW_DIR,
    repo_id: str = REPO_ID,
):
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset (same features as stack_bowls_three!)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="aloha",
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
            "observation.images.cam_high": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
            "observation.images.cam_left_wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
            "observation.images.cam_right_wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        },
        use_videos=True,
        image_writer_threads=5,
        image_writer_processes=10,
    )

    # Collect all HDF5 files from all subdirectories, sorted by subdirectory then filename
    all_files = []
    for subdir in SUBDIRS:
        subdir_path = raw_dir / subdir
        files = sorted(subdir_path.glob("*.hdf5"))
        all_files.extend([(subdir, f) for f in files])

    # Sort by subdirectory order then filename to maintain episode sequence
    all_files.sort(key=lambda x: (SUBDIRS.index(x[0]), int(x[1].stem)))

    print(f"Found {len(all_files)} episodes from {len(SUBDIRS)} subdirectories")

    ep_idx = 0
    for subdir, ep_path in all_files:
        with h5py.File(ep_path, "r") as f:
            n_frames = f['left_arm/qpos'].shape[0]

            # Load state: concatenate left (6+1) + right (6+1)
            left_qpos = f['left_arm/qpos'][:]  # (N, 6)
            left_gripper = f['left_arm/gripper'][:]  # (N,)
            right_qpos = f['right_arm/qpos'][:]  # (N, 6)
            right_gripper = f['right_arm/gripper'][:]  # (N,)

            # Stack to get (N, 14) state
            # Order: left_arm_joints (6) + left_gripper (1) + right_arm_joints (6) + right_gripper (1)
            state = np.concatenate([
                left_qpos,
                left_gripper[:, None],
                right_qpos,
                right_gripper[:, None]
            ], axis=1)  # (N, 14)

            # Action: next frame's qpos (absolute position)
            # For last frame, repeat the last state
            action = np.concatenate([
                state[1:],  # frames 1 to end
                state[-1:]  # repeat last frame
            ], axis=0)

            # Load images (already RGB uint8, just need HWC→CHW)
            cam_high = f['cam_head/color'][:]  # (N, H, W, C)
            cam_left = f['cam_left_wrist/color'][:]
            cam_right = f['cam_right_wrist/color'][:]

            # Add frames
            for i in range(n_frames):
                frame = {
                    "observation.state": state[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                    "task": "stack the blocks",  # Updated task description
                    "observation.images.cam_high": cam_high[i].transpose(2, 0, 1),  # HWC→CHW
                    "observation.images.cam_left_wrist": cam_left[i].transpose(2, 0, 1),
                    "observation.images.cam_right_wrist": cam_right[i].transpose(2, 0, 1),
                }
                dataset.add_frame(frame)

        dataset.save_episode()
        print(f"Processed {subdir}/{ep_path.name}: {n_frames} frames (episode {ep_idx})")
        ep_idx += 1

    print(f"\nDataset created: {output_path}")
    print(f"Total episodes: {len(dataset)}, Frames: {dataset.num_frames}")
    print(f"Merged from {len(SUBDIRS)} subdirectories")
    print("\nNext steps:")
    print("1. Update repo_id in config.py to: ygx/stack_blocks_three")
    print("2. Compute normalization stats: uv run scripts/compute_norm_stats.py --config-name pi05_dual_piper")
    print("3. Start training: uv run scripts/train.py pi05_dual_piper --exp-name=stack_blocks_three --overwrite")


if __name__ == "__main__":
    tyro.cli(main)
