# DualPiper 机器人客户端部署指南

本文档说明如何在机器人控制器上部署 DualPiper 客户端，用于连接远程推理服务器。

---

## 架构概述

```
┌────────────────────────────────────────────────────────────────────────┐
│                           部署架构                                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  【GPU 服务器】                     【机器人控制器】                   │
│                                                                        │
│  openpi/                          ~/dual_piper_client/                   │
│  ├── src/openpi/                   ├── realsense_camera.py                 │
│  ├── packages/openpi-client/         ├── piper_env.py                      │
│  └── scripts/serve_policy.py        ├── env.py                            │
│                                     ├── main.py                           │
│  运行: serve_policy.py             ├── constants.py                      │
│       ↓                              ├── requirements.txt                   │
│  监听: 0.0.0.0:8000                  └── main.py (运行这个)                  │
│       ↓                                     ↓                                │
│  WebSocket ────────────────────────────────► 连接获取 action                     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 机器人端硬件要求

| 组件 | 规格 |
|------|------|
| 计算机 | Ubuntu 20.04+ (推荐), 有 USB 接口 |
| 机械臂 | Agilex Piper 双臂机器人 |
| 摄像头 | 3x Intel RealSense D435 |
| CAN 模块 | 2x USB 转 CAN (连接左右臂) |
| 网络 | 能 ping 通 GPU 服务器 |

---

## 机器人端安装步骤

### 1. 安装 Python 依赖

```bash
# 更新系统包
sudo apt update
sudo apt install -y python3 python3-pip

# 安装 pyrealsense2 (Intel RealSense SDK)
# 先安装依赖
sudo apt-get -y install software-properties-common
sudo apt-get -y install ubuntu-restricted-extras

# 添加 Intel 服务器
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F6E65AC044F8F761085A77BBBBBBBBBBBBBBBBBBB
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -sc) main"
sudo apt-get update
sudo apt-get install -y librealsense2-utils python3-librealsense2

# 安装 Python 包
pip3 install piper_sdk python-can numpy opencv-python einops tyro
```

### 2. 安装 openpi-client

有两种方式：

**方式 A: 从源码安装** (如果可以访问源码)
```bash
cd /path/to/openpi
pip3 install -e packages/openpi-client
```

**方式 B: 从 GitHub 安装** (如果无法访问源码)
```bash
# 需要将以下文件复制到机器人:
# - packages/openpi-client/ 源码
# - 或者创建一个 wheel 包后传输

# 推荐方式: 在 GPU 服务器上打包
cd /home/ygx/VLA/openpi/packages/openpi-client
pip3 wheel --no-deps .
# 生成的 .whl 文件复制到机器人，然后:
pip3 install openpi_client-*.whl
```

### 3. 配置 CAN 模块

```bash
# 检查 CAN 模块连接
bash /path/to/openpi/third_party/piper_sdk/find_all_can_port.sh

# 编辑 can 配置
nano /path/to/openpi/third_party/piper_sdk/can_muti_activate.sh

# 设置 USB 端口映射 (根据你的硬件)
USB_PORTS["你的USB端口1"]="can_left:1000000"
USB_PORTS["你的USB端口2"]="can_right:1000000"

# 激活 CAN 模块
bash /path/to/openpi/third_party/piper_sdk/can_muti_activate.sh
```

### 4. 复制客户端文件

将以下文件复制到机器人 (例如 `~/dual_piper_client/`):

```bash
# 在机器人上创建目录
mkdir -p ~/dual_piper_client

# 从 GPU 服务器复制 (使用 scp)
# 在 GPU 服务器上执行:
scp examples/dual_piper/realsense_camera.py user@robot:~/dual_piper_client/
scp examples/dual_piper/piper_env.py user@robot:~/dual_piper_client/
scp examples/dual_piper/env.py user@robot:~/dual_piper_client/
scp examples/dual_piper/main.py user@robot:~/dual_piper_client/
scp examples/dual_piper/constants.py user@robot:~/dual_piper_client/
scp examples/dual_piper/requirements.txt user@robot:~/dual_piper_client/
```

---

## 运行客户端

### 1. 测试 RealSense 摄像头

```bash
cd ~/dual_piper_client
python3 realsense_camera.py
```

应该输出类似：
```
Scanning for RealSense cameras...
Found camera 0: Intel RealSense D435 (serial: xxx)
Found camera 1: Intel RealSense D435 (serial: yyy)
Found camera 2: Intel RealSense D435 (serial: zzz)
```

### 2. 运行客户端 (连接到服务器)

```bash
cd ~/dual_piper_client

# 基础运行 (自动发现摄像头)
python3 main.py --host <GPU服务器IP> --port 8000

# 指定摄像头序列号 (推荐)
python3 main.py \
    --host <GPU服务器IP> \
    --port 8000 \
    --camera-serials '{"cam_high": "xxx", "cam_left_wrist": "yyy", "cam_right_wrist": "zzz"}'

# 指定 CAN 接口
python3 main.py \
    --host <GPU服务器IP> \
    --can-left can_left \
    --can-right can_right

# 完整参数
python3 main.py \
    --host 192.168.1.100 \
    --port 8000 \
    --can-left can_left \
    --can-right can_right \
    --camera-serials '{"cam_high": "xxx", "cam_left_wrist": "yyy", "cam_right_wrist": "zzz"}' \
    --action-horizon 10 \
    --num-episodes 1 \
    --max-episode-steps 1000 \
    --velocity 50
```

---

## 文件清单

机器人端需要以下文件：

| 文件 | 作用 | 必需 |
|------|------|------|
| `main.py` | 入口程序 | ✅ |
| `env.py` | Environment 适配层 | ✅ |
| `piper_env.py` | Piper SDK 硬件控制 | ✅ |
| `realsense_camera.py` | RealSense 摄像头 | ✅ |
| `constants.py` | 常量定义 | ✅ |
| `openpi_client/` | openpi-client 包 | ✅ |

---

## 故障排查

### 摄像头连接问题

```bash
# 检查连接的摄像头
rs-enumerate-devices | grep RealSense

# 查看摄像头信息
rs-enumerate-devices | grep -A 5 RealSense
```

### CAN 通信问题

```bash
# 检查 CAN 接口
ifconfig

# 查看 CAN 总线流量
candump can_left -c
candump can_right -c
```

### 网络连接问题

```bash
# 测试与 GPU 服务器的网络
ping <GPU服务器IP>

# 测试端口 (如果有 nc)
nc -zv <GPU服务器IP> 8000

# 或用 curl 测试
curl http://<GPU服务器IP>:8000/healthz
```

### 依赖问题

```bash
# 检查安装的包
pip3 list | grep -E "piper|realsense|openpi"

# 检查 Python 版本
python3 --version  # 需要 >= 3.8
```

---

## 网络配置建议

### 方法 1: 有线网络 (推荐)

直接用网线连接机器人和 GPU 服务器到同一路由器/交换机。

### 方法 2: WiFi

确保机器人和服务器在同一局域网，配置好固定 IP。

### 方法 3: 跨网络 (需要配置)

如果服务器在云端：
1. 服务器需要开放 8000 端口
2. 配置防火墙允许机器人 IP 访问
3. 使用服务器公网 IP 或域名

---

## 与服务器端通信格式

客户端发送到服务器的 observation 格式：

```python
{
    "state": np.ndarray,  # (14,) [左臂6关节, 左夹爪, 右臂6关节, 右夹爪] (弧度/米)
    "images": {
        "cam_high": np.ndarray,      # (3, 224, 224) uint8, CHW 格式
        "cam_left_wrist": np.ndarray, # (3, 224, 224)
        "cam_right_wrist": np.ndarray, # (3, 224, 224)
    },
    "prompt": str,  # 任务描述
}
```

服务器返回的 action 格式：

```python
{
    "actions": np.ndarray,  # (action_horizon, 14) 动作序列
    "server_timing": {
        "infer_ms": float,  # 推理耗时 (毫秒)
    }
}
```

---

## 开发/调试技巧

### 1. 本地测试 (不连接真实机器人)

```python
# 测试 Piper SDK 连接
python3 -c "
from piper_env import create_piper_dual_arm
env = create_piper_dual_arm()
print('Joint positions:', env.get_joint_positions())
"

# 测试摄像头
python3 realsense_camera.py
```

### 2. 测试远程服务器连接

```bash
# 使用 simple_client 测试 (在 GPU 服务器上训练好的模型)
cd /path/to/openpi
uv run examples/simple_client/main.py --env DROID
```

### 3. 查看日志

```bash
# 启用详细日志
python3 main.py --host <IP> 2>&1 | tee client.log
```

---

## 常见问题

**Q: 能不能不用 openpi-client，直接用 pyrealsense2 + piper_sdk?**

A: 可以，但你需要自己实现 WebSocket 客户端和 action chunking 逻辑。建议使用 openpi-client，它已经封装好了。

**Q: 客户端需要 GPU 吗?**

A: 不需要。客户端只做数据采集和动作执行，所有的模型推理都在服务器端完成。

**Q: 网络延迟有什么影响?**

A: 理想情况延迟 < 10ms。如果延迟太高，可能需要调整 action_horizon 或增加本地缓存。

**Q: 如何调试 camera 序列号?**

A:
```bash
# 插一个摄像头，运行:
python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
for d in devices:
    print('Serial:', d.get_info(rs.camera_info.serial_number))
    print('Name:', d.get_info(rs.camera_info.name))
"
```
