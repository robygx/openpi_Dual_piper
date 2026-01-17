"""
DualPiper 远程推理客户端示例

这个脚本展示了如何从 DualPiper 机器人连接到远程策略服务器，
发送观察值并接收动作预测。

运行方式：
    # 使用随机数据测试连接
    uv run python examples/dual_piper/inference_client.py --host <服务器IP>

    # 或者从机器人获取真实数据
    python your_robot_script.py --server-ip <服务器IP>
"""

import dataclasses
import logging
import pathlib
import time

import numpy as np
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DualPiperObservation:
    """DualPiper 机器人观察值

    相机:
        - cam_high: 顶视角相机 (H, W, 3) HWC 格式 (numpy array 或 uint8)
        - cam_left_wrist: 左腕相机 (H, W, 3)
        - cam_right_wrist: 右腕相机 (H, W, 3)

    状态:
        - state: 14维关节状态 (左臂7维 + 右臂7维)
    """

    cam_high: np.ndarray  # 顶视角图像，HWC 格式
    cam_left_wrist: np.ndarray  # 左腕图像
    cam_right_wrist: np.ndarray  # 右腕图像
    state: np.ndarray  # (14,) 关节位置

    def to_policy_input(self, resize_size: int = 224) -> dict:
        """转换为策略服务器期望的输入格式

        Args:
            resize_size: 图像将被 resize 到这个尺寸

        Returns:
            dict: 策略服务器期望的观察值格式
        """
        # 确保 HWC 格式
        cam_high = self.cam_high if self.cam_high.ndim == 3 else self.cam_high.transpose(1, 2, 0)
        cam_left = self.cam_left_wrist if self.cam_left_wrist.ndim == 3 else self.cam_left_wrist.transpose(1, 2, 0)
        cam_right = self.cam_right_wrist if self.cam_right_wrist.ndim == 3 else self.cam_right_wrist.transpose(1, 2, 0)

        return {
            "images": {
                "cam_high": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(cam_high, resize_size, resize_size)
                ),
                "cam_left_wrist": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(cam_left, resize_size, resize_size)
                ),
                "cam_right_wrist": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(cam_right, resize_size, resize_size)
                ),
            },
            "state": self.state.astype(np.float32),  # 服务器会处理归一化
        }


class DualPiperPolicyClient:
    """DualPiper 策略客户端

    用法:
        client = DualPiperPolicyClient(host="192.168.1.100", port=8000)

        # 初始化连接
        client.initialize()

        # 在控制循环中
        for step in range(num_steps):
            action_chunk = client.get_action(observation, task="fold the cloth")
            # 执行 action_chunk[0]  # 第一个动作
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
    ):
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )
        self._action_chunk = None
        self._chunk_index = 0

    def initialize(self) -> dict:
        """初始化连接并获取服务器元数据

        发送几个虚拟观察值以确保模型已加载。

        Returns:
            dict: 服务器元数据
        """
        metadata = self.client.get_server_metadata()
        logger.info(f"Connected to server: {metadata}")

        # 发送几个虚拟观察值预热
        for _ in range(2):
            dummy_obs = self._get_dummy_observation()
            dummy_task = "fold the cloth"
            self.client.infer({"observation": dummy_obs, "prompt": dummy_task})

        return metadata

    def get_action(self, observation: DualPiperObservation, task: str) -> np.ndarray:
        """获取动作预测

        如果当前 action_chunk 还有剩余动作，直接返回；
        否则调用策略服务器获取新的 action chunk。

        Args:
            observation: 当前观察值
            task: 任务描述

        Returns:
            np.ndarray: 动作，形状 (14,)
        """
        # 检查是否需要获取新的 action chunk
        if self._action_chunk is None or self._chunk_index >= len(self._action_chunk):
            inference_start = time.time()
            response = self.client.infer({
                "observation": observation.to_policy_input(),
                "prompt": task,
            })
            logger.info(f"Inference time: {1000 * (time.time() - inference_start):.1f}ms")

            self._action_chunk = response["actions"]  # (action_horizon, 14)
            self._chunk_index = 0

        action = self._action_chunk[self._chunk_index]
        self._chunk_index += 1

        return action

    def _get_dummy_observation(self) -> dict:
        """获取用于预热的虚拟观察值"""
        return {
            "images": {
                "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            },
            "state": np.zeros(14, dtype=np.float32),
        }


def example_with_random_data(host: str = "localhost", port: int = 8000, task: str = "fold the cloth"):
    """使用随机数据测试客户端连接

    Args:
        host: 策略服务器 IP 地址
        port: 策略服务器端口
        task: 任务描述
    """

    # 创建客户端
    client = DualPiperPolicyClient(host=host, port=port)

    # 初始化
    logger.info(f"Connecting to {host}:{port}...")
    metadata = client.initialize()
    logger.info(f"Server metadata: {metadata}")

    # 模拟控制循环
    num_steps = 10

    for step in range(num_steps):
        # 在实际使用中，这里从机器人获取真实观察值
        observation = DualPiperObservation(
            cam_high=np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # HWC
            cam_left_wrist=np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            cam_right_wrist=np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            state=np.random.randn(14).astype(np.float32),
        )

        # 获取动作
        action = client.get_action(observation, task)

        logger.info(f"Step {step}: action shape = {action.shape}, mean = {action.mean():.4f}")

        # 在实际使用中，这里将 action 发送给机器人执行
        # execute_action_on_robot(action)


@dataclasses.dataclass
class Args:
    """命令行参数"""
    host: str = "localhost"
    port: int = 8000
    task: str = "fold the cloth"


if __name__ == "__main__":
    tyro.cli(example_with_random_data)
