# DualPiper Dataset Configuration & X-VLA Conversion Guide

本目录包含 DualPiper 双臂机器人数据集的配置，以及 X-VLA Cloth Folding 数据集的转换脚本。

## X-VLA Cloth Folding 数据集 (当前使用)

### 数据集信息

- **任务**: 折叠布料 (fold the cloth)
- **机器人**: Agilex Aloha (双臂 14 DOF)
- **数据量**: 108 episodes, 约 157,000 帧
- **摄像头**: 3个 (cam_high, cam_left_wrist, cam_right_wrist)
- **分辨率**: 480x640x3
- **FPS**: 30

### 原始数据格式 (X-VLA HDF5)

```
/data0/ygx_data/X-VLA/0930_10am_new/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
└── episode_106.hdf5
```

每个 HDF5 文件结构：
```python
{
    "/observations/qpos": (N, 14)      # 关节位置
    "/action": (N, 14)                  # 动作
    "/language_instruction": "fold the cloth"
    "/observations/images/cam_high": (N,)     # 压缩的 JPEG 字节
    "/observations/images/cam_left_wrist": (N,)
    "/observations/images/cam_right_wrist": (N,)
}
```

### 转换命令

```bash
cd /home/ygx/VLA/openpi

# 测试转换单个 episode
uv run examples/dual_piper/convert_dual_piper_data_to_lerobot.py \
    --raw_dir /data0/ygx_data/X-VLA/0930_10am_new \
    --repo_id ygx/xvla_cloth_folding \
    --episodes 0

# 转换全部 108 个 episodes (约 3 小时)
uv run examples/dual_piper/convert_dual_piper_data_to_lerobot.py \
    --raw_dir /data0/ygx_data/X-VLA/0930_10am_new \
    --repo_id ygx/xvla_cloth_folding
```

---

### 数据集验证结果 ✅

转换后的数据集位于 `~/.cache/huggingface/lerobot/ygx/xvla_cloth_folding_test/`

#### 验证结果总结

| 项目 | 期望值 | 实际值 | 状态 |
|------|--------|--------|------|
| Episodes | 1 | 1 | ✅ |
| Total frames | 1460 | 1460 | ✅ |
| FPS | 30 | 30 | ✅ |
| 图像形状 | [3, 480, 640] | [3, 480, 640] | ✅ |
| State 形状 | [14] | [14] | ✅ |
| Action 形状 | [14] | [14] | ✅ |
| Task | "fold the cloth" | "fold the cloth" | ✅ |

#### 数据键名 (与配置文件匹配)

| 数据集键名 | 形状 | 用途 |
|------------|------|------|
| `observation.images.cam_high` | [3, 480, 640] | 顶视角图像 |
| `observation.images.cam_left_wrist` | [3, 480, 640] | 左腕图像 |
| `observation.images.cam_right_wrist` | [3, 480, 640] | 右腕图像 |
| `observation.state` | [14] | 关节状态 |
| `action` | [14] | 动作 |
| `task` | - | 任务描述 |

#### 文件结构

```
ygx/xvla_cloth_folding_test/
├── meta/
│   ├── info.json              # 元数据 ✅
│   ├── episodes.jsonl         # episode 信息 ✅
│   ├── episodes_stats.jsonl   # 统计信息 ✅
│   └── tasks.jsonl            # 任务信息 ✅
├── data/chunk-000/            # parquet 数据文件 ✅
└── videos/chunk-000/          # 3个摄像头的 MP4 视频 ✅
    ├── observation.images.cam_high/episode_000000.mp4
    ├── observation.images.cam_left_wrist/episode_000000.mp4
    └── observation.images.cam_right_wrist/episode_000000.mp4
```

#### 单 episode 大小

- 约 **34 MB** (包含 3 个摄像头的视频 + parquet 数据)

---

## LeRobot API 变更说明 (重要!)

相对于旧版转换脚本（如 `aloha_real/convert_aloha_data_to_lerobot.py`），需要做以下更改：

### 1. 导入路径变更

```python
# ❌ 旧版 (不支持)
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

# ✅ 新版 (lerobot 0.1.0)
from lerobot.common.constants import HF_LEROBOT_HOME
```

### 2. save_episode() 方法签名变更

```python
# ❌ 旧版 (不再支持)
dataset.save_episode(task=task)

# ✅ 新版 - task 需要在每帧的 frame 字典中
for i in range(num_frames):
    frame = {
        "observation.state": state[i],
        "action": action[i],
        "task": task,  # ← task 必须在每个 frame 中
    }
    dataset.add_frame(frame)

dataset.save_episode()  # 不需要参数
```

### 3. 移除的方法

```python
# ❌ 旧版
dataset.consolidate()  # 不再存在

# ✅ 新版 - 不需要调用 consolidate，数据在 save_episode 时已保存
```

### 4. 属性名变更

| 旧版 | 新版 |
|------|------|
| `dataset.total_episodes` | `dataset.num_episodes` |
| `dataset.total_frames` | `dataset.num_frames` |

### 5. 图像格式要求

LeRobot 要求图像为 **Channels-First (CHW)** 格式：

```python
# X-VLA 图像处理流程
img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)  # 解码 JPEG → BGR HWC
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # BGR → RGB
img = img.transpose(2, 0, 1)                       # HWC → CHW
```

---

## 通用 DualPiper 配置

### DualPiper Specifications

- **Robot Type**: Dual-arm robot (similar to ALOHA)
- **State Dimension**: 14 (left arm 6 joints + 1 gripper, right arm 6 joints + 1 gripper)
- **Action Dimension**: 14 (same as state)
- **Cameras**: 3 cameras
  - `cam_high`: High-angle third-person view
  - `cam_left_wrist`: Left wrist camera
  - `cam_right_wrist`: Right wrist camera
  - Note: No `cam_low` (unlike ALOHA which has 4 cameras)
- **Action Type**: Absolute positions (converted to delta during training)

### Key Differences from ALOHA

| Feature | ALOHA | DualPiper |
|---------|-------|-----------|
| Cameras | 4 (high, low, left_wrist, right_wrist) | 3 (high, left_wrist, right_wrist) |
| State/Action Dim | 14 | 14 |
| Delta Actions | Yes | Yes |

---

## 下一步 (训练流程)

### 1. 更新配置文件

编辑 `/home/ygx/VLA/openpi/src/openpi/training/config.py`:

```python
repo_id="ygx/xvla_cloth_folding",  # 更新为你的 repo_id
```

### 2. 计算归一化统计量

```bash
cd /home/ygx/VLA/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_dual_piper
```

这会创建 `assets/ygx/xvla_cloth_folding/norm_stats.json`。

### 3. 开始训练

```bash
cd /home/ygx/VLA/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_dual_piper \
    --exp-name=xvla_finetune \
    --overwrite
```

训练检查点会保存到 `checkpoints/pi05_dual_piper/xvla_finetune/`。

### 4. 推理

启动策略服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dual_piper \
    --policy.dir=checkpoints/pi05_dual_piper/xvla_finetune/30000
```

---

## Files

- `convert_dual_piper_data_to_lerobot.py`: X-VLA 数据转换脚本
- Configuration in `/home/ygx/VLA/openpi/src/openpi/training/config.py`:
  - `LeRobotDualPiperDataConfig`: Data configuration class
  - `pi05_dual_piper`: Training configuration
- Policy in `/home/ygx/VLA/openpi/src/openpi/policies/dual_piper_policy.py`:
  - `DualPiperInputs`: Input transformation
  - `DualPiperOutputs`: Output transformation

---

## Action Space

14维动作空间表示：
- **维度 0-5**: 左臂关节角度 (绝对位置)
- **维度 6**: 左夹爪 (0=闭合, 1=打开)
- **维度 7-12**: 右臂关节角度 (绝对位置)
- **维度 13**: 右夹爪 (0=闭合, 1=打开)

训练时，关节角度自动转换为 delta 动作，夹爪保持绝对位置。

---

## 参考资源

- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
- [openpi Issue #353](https://github.com/Physical-Intelligence/openpi/issues/353)
- [openpi Issue #354](https://github.com/Physical-Intelligence/openpi/issues/354)
- [openpi PR #508](https://github.com/Physical-Intelligence/openpi/pull/508)
