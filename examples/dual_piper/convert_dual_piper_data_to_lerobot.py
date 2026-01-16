"""
Minimal script to convert X-VLA HDF5 data to LeRobot format.
Based on examples/libero/convert_libero_data_to_lerobot.py

Data format:
- X-VLA cloth folding dataset with 108 episodes
- 14-dimensional bimanual robot state/actions (6+1 joints+gripper per arm)
- 3 cameras: cam_high, cam_left_wrist, cam_right_wrist
- Compressed JPEG images stored as bytes (requires cv2.imdecode + BGR→RGB + HWC→CHW)
- Task: "fold the cloth"

Usage:
    uv run examples/dual_piper/convert_dual_piper_data_to_lerobot.py --raw_dir /data0/ygx_data/X-VLA/0930_10am_new --repo_id ygx/xvla_cloth_folding
"""

import shutil
from pathlib import Path

import cv2
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def main(
    raw_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
):
    # Clean up existing dataset
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset (similar to libero, but with video mode and CHW format)
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

    # Get HDF5 files
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if episodes is None:
        episodes = range(len(hdf5_files))

    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    # Convert each episode (minimal code, inline processing)
    for ep_idx in episodes:
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            state = ep["/observations/qpos"][:]
            action = ep["/action"][:]
            task = ep["/language_instruction"][()].decode("utf-8")

            # Decode images for each camera (X-VLA specific: JPEG + BGR→RGB + HWC→CHW)
            imgs_per_cam = {}
            for cam in cameras:
                imgs = []
                for data in ep[f"/observations/images/{cam}"]:
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1)  # HWC → CHW
                    imgs.append(img)
                imgs_per_cam[cam] = imgs

            # Add frames
            for i in range(len(state)):
                frame = {
                    "observation.state": state[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                    "task": task,
                }
                for cam in cameras:
                    frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]
                dataset.add_frame(frame)

        dataset.save_episode()

    print(f"Dataset created: {output_path}")
    print(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")


if __name__ == "__main__":
    tyro.cli(main)
