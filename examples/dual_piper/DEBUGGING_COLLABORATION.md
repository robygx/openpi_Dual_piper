# DualPiper 调试协作说明

## 目标

调试 DualPiper 机械臂控制问题，通过 git 分支协作，确保两端代码同步。

## 工作流程

```
┌─────────────────────┐                    ┌─────────────────────┐
│   服务器端 (我)       │                    │   客户端 (你)       │
│  (代码修改 + 分析)     │◄────────────────────► │  (运行测试 + 反馈)   │
└─────────────────────┘                    └─────────────────────┘
         │                                          │
         │ Git Push                                │ Git Pull
         │ (origin/debug/dual-piper-action)        │ (origin/debug/dual-piper-action)
         ▼                                          ▼
    GitHub/远程仓库 (共享)
    branch: debug/dual-piper-action
```

## Git 分支操作

### 客户端 - 同步代码

每次我开始修改代码后，我会执行：

```bash
git add examples/dual_piper/
git commit -m "debug: <描述>: <原因>"
git push origin debug/dual-piper-action
```

**你需要执行的同步命令**：

```bash
cd /home/shiyanyan/ygx/openpi_Dual_piper

# 1. 确保在正确的分支
git checkout debug/dual-piper-action

# 2. 拉取最新代码
git fetch origin debug/dual-piper-action
git rebase origin/debug/dual-piper-action

# 或者简单方式
git pull origin debug/dual-piper-action
```

### 客户端 - 运行测试并反馈

```bash
# 1. 同步代码后，运行客户端
cd examples/dual_piper
python main.py --host 127.0.0.1 --port 9000 \
    --can-left can0 \
    --can-right can1 \
    --camera-serials '{"cam_high": "152122072933", "cam_left_wrist": "213722070453", "cam_right_wrist": "152122076290"}'
```

### 客户端 - 收集日志反馈

运行后，**完整复制**以下日志内容发给我：

1. **初始化日志**（包含 "OBSERVATION DEBUG" 的部分）
2. **ACTION DEBUG 日志**（前 5 步）
3. **任何错误信息**

## Commit 信息规范

我会在每次 commit 中清楚说明：

```
debug: <简短描述>

<详细说明>
- 修改了什么文件/函数
- 为什么这样修改
- 预期效果是什么
</详细说明>
```

示例：
```
debug: add observation and action logging

修改:
- env.py::get_observation(): 添加 state 和 images 格式检查
- env.py::apply_action(): 添加 actions 转换过程检查

目的:
- 验证 state 是否匹配数据集起始位置
- 验证 actions 是否合理
- 验证夹爪转换是否正确

预期输出:
- OBSERVATION DEBUG: 显示 state vs dataset start 对比
- ACTION DEBUG: 显示 current state, received action, gripper transform
```

## 当前分支状态

| 文件 | 状态 | 说明 |
|------|------|------|
| constants.py | 已修改 | RESET_POSITION 更新为数据集起始位置 |
| env.py | 已修改 | 添加详细的调试日志 |
| piper_env.py | 已修改 | reset() 支持 14 维并重置夹爪 |
| main.py | 已修改 | 添加 camera_serials 参数 |

## 调试检查清单

运行客户端时，确认以下检查点的输出：

### 检查点 1: State 是否正确
- 日志: `State matches dataset: True`
- 如果 `False`，说明 reset() 没有生效

### 检查点 2: 图像格式是否正确
- 日志: `final shape=(3, 224, 224), dtype=uint8`
- 3 个摄像头都应该是这个格式

### 检查点 3: Actions 是否合理
- 日志: `Joint delta reasonable (< 10 deg): True`
- 如果 `False`，说明模型输出可能有问题

### 检查点 4: 夹爪转换是否正确
- 日志: `Gripper in range [0, 0.05]: True`
- BEFORE: 0-1 归一化值
- AFTER: 0-0.05 米

## 常见问题

**Q: git pull 时冲突怎么办？**
```bash
# 本地修改先暂存
git stash
git pull origin debug/dual-piper-action
git stash pop
```

**Q: 如何查看 commit 历史？**
```bash
git log --oneline -10
```

**Q: 如何回滚到上一个版本？**
```bash
git reset --hard HEAD~1
```

## 联系方式

- 服务器端修改完成通知: 查看 git commit 历史
- 客户端测试完成后反馈: 发送完整日志输出
- 问题分析: 我根据日志分析并给出下一步修改建议
