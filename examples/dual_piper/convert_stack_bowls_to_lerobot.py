"""
Convert Stack Bowls HDF5 data to LeRobot format.

Stack Bowls dataset:
- 394 episodes of DualPiper robot stacking bowls
- 3 cameras: cam_head (→ cam_high), cam_left_wrist, cam_right_wrist
- 14 DOF: left arm (6 joints + gripper) + right arm (6 joints + gripper)
- Images are raw RGB uint8 (not JPEG compressed like X-VLA)
- Action data: computed as next frame's qpos (absolute position)

Example usage:
    uv run examples/dual_piper/convert_stack_bowls_to_lerobot.py
"""
import shutil
from pathlib import Path

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

REPO_ID = "ygx/stack_bowls_three"
RAW_DIR = Path("/data0/ygx_data/raw_data/stack_bowls_three_hdf5")


def main(
    raw_dir: Path = RAW_DIR,
    repo_id: str = REPO_ID,
):
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset (same features as X-VLA!)
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
        image_writer_threads=16,
        image_writer_processes=32,
    )

    # Get all HDF5 files, sorted numerically
    hdf5_files = sorted(raw_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    print(f"Found {len(hdf5_files)} episodes")

    for ep_path in hdf5_files:
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
                    "task": "stack the bowls",
                    "observation.images.cam_high": cam_high[i].transpose(2, 0, 1),  # HWC→CHW
                    "observation.images.cam_left_wrist": cam_left[i].transpose(2, 0, 1),
                    "observation.images.cam_right_wrist": cam_right[i].transpose(2, 0, 1),
                }
                dataset.add_frame(frame)

        dataset.save_episode()
        print(f"Processed {ep_path.name}: {n_frames} frames")

    print(f"\nDataset created: {output_path}")
    print(f"Episodes: {len(dataset)}, Frames: {dataset.num_frames}")
    print("\nNext steps:")
    print("1. Update repo_id in config.py to: ygx/stack_bowls_three")
    print("2. Compute normalization stats: uv run scripts/compute_norm_stats.py --config-name pi05_dual_piper")
    print("3. Start training: uv run scripts/train.py pi05_dual_piper --exp-name=stack_bowls_three --overwrite")


if __name__ == "__main__":
    tyro.cli(main)
