# Wuji Robot Deployment with OpenPI / 基于 OpenPI 的 Wuji 机器人部署

> [English](#english) | [中文](#中文)

---

<a id="english"></a>

# English

Wuji robot OpenPI deployment solution — a complete training-to-deployment workflow.

## Project Overview

This project provides a deployment solution for the Wuji robot based on the OpenPI framework. It connects to a policy server over WebSocket to enable coordinated bimanual control with integrated visual perception.

### Main Features

- **OpenPI control pipeline**: connects to a remote policy server over WebSocket for real-time inference
- **Bimanual support**: single-arm (left/right) or coordinated bimanual operation
- **54-dimensional control**: dual arms (14) + dual dexterous hands (40) for precise control
- **Multi-camera input**: head camera + left/right wrist cameras (3 visual streams)
- **ROS2 integration**: complete topic subscription and publishing system
- **Flexible configuration**: YAML-based configuration system

### System Architecture

```
OpenPI Runtime (orchestrator)
  ├─> Environment (Wuji ROS2 environment)
  │     ├─> ROS2 topic subscriptions (joint states, camera images)
  │     └─> ROS2 topic publishers (joint commands)
  ├─> Agent (policy agent)
  │     └─> Broker (serial / RTG)
  │           └─> WebsocketClientPolicy (policy server client)
  └─> Subscribers (observers)
```

## Requirements

- **ROS2**: Humble (Python 3.10)
- **Python**: 3.10+ (for the ROS2 client) / 3.11+ (for the policy server)
- **OS**: Ubuntu 22.04 / Linux
- **GPU**: NVIDIA GPU recommended (for model inference)

## Installation

### 1. Clone the project

```bash
git clone https://github.com/wuji-technology/wuji-openpi.git
cd wuji-openpi
```

### 2. Install project dependencies (with uv)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Install ROS2 Humble

Follow the official ROS2 documentation: https://docs.ros.org/en/humble/Installation.html

After installation, source the ROS2 setup script provided by your installation.

## Usage

Deployment has two steps: **server side** (policy inference) and **client side** (robot control). All commands below assume your current working directory is the project root.

### Step 1: start the policy server

In the **first terminal**, start the policy server (using a trained model):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=<your-checkpoint-dir>
```

**Parameter description**:
- `--policy.config`: model configuration name (defined in `src/openpi/training/config.py`)
- `--policy.dir`: checkpoint directory path (containing the `params` file)

**Optional parameters**:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=<your-checkpoint-dir> \
  --host=0.0.0.0 \
  --port=8000
```

**Example of successful output**:
```
INFO:root:Loading model...
INFO:root:Loaded norm stats from <checkpoint-dir>/assets/placeholder
INFO:root:Creating server (host: your-hostname, ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### Step 2: start the robot control client

Run this step in a ROS2 environment that has `openpi-client` installed. In the **second terminal** (from the project root):

```bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

**Example of successful output**:
```
[INFO] [timestamp] [wuji_openpi_node]: Node wuji_openpi_node created
[INFO] Connecting to policy server: localhost:8000
[INFO] Server metadata: {}
[INFO] Data ready (0.0s)
[INFO] Wuji robot control system configuration (OpenPI):
[INFO]   Policy server: localhost:8000
[INFO]   Control mode: dual
[INFO]   Control frequency: 30.0Hz
[INFO]   Action horizon: 50 steps
[INFO]   Inference frequency: 0.60Hz
[INFO] Starting...
```

### Command-line arguments

Command-line arguments can override the configuration file:

```bash
# Specify the server address
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --host 192.168.1.100 --port 12346

# Specify the control mode
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --side left  # left arm only
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --side both  # both arms

# Other arguments
python3 -m wuji.deploy.main \
  --host localhost \
  --port 8000 \
  --side both \
  --broker-mode rtg \
  --control-hz 30.0 \
  --action-horizon 50 \
  --prompt "tidy up the table"
```

## Configuration

Configuration files live under `examples/wuji/config/`; the main configuration file is `examples/wuji/config/deploy.yaml`.

### Core configuration options

```yaml
# Basic configuration
id: "wuji_robot"
arm_mode: "dual"  # single_left, single_right, dual
use_ee_pose: false

# Arm type configuration
arm_type: "wuji"  # wuji, agilex, custom
arm_dof: 7
hand_dof: 20
control_hz: 30.0

# OpenPI server configuration
server_host: "localhost"
server_port: 8000

# Action chunk configuration
action_horizon: 50        # inference frequency = control_hz / action_horizon

# Broker mode
broker_mode: "rtg"        # serial / rtg

# RTG smoothing configuration (broker_mode=rtg)
rtg_trigger_fraction: 0.5
rtg_guidance_steps: 3

# Episode configuration
num_episodes: 1
max_episode_steps: 1000000

# Task prompt
prompt: "tidy up the table"
```

### Two action-execution modes

- `serial`: the original serial mode; the full chunk is inferred and then executed sequentially.
- `rtg`: the RTG mode; the client asynchronously receives a new chunk and, following the paper `arXiv:2507.17141`, applies cubic smoothing to the prefix of the new chunk before stitching and executing.

### ROS2 topic configuration

**Subscribed topics (inputs)**:
```yaml
joint_state_topics:
  left_arm:    { topic: "/tianji_arm/left/joint_state",  enabled: true }
  right_arm:   { topic: "/tianji_arm/right/joint_state", enabled: true }
  left_hand:   { topic: "/wuji_hand/left/joint_state",   enabled: true }
  right_hand:  { topic: "/wuji_hand/right/joint_state",  enabled: true }

camera_topics:
  cam_high:        { topic: "/stereo/right/compressed",                       enabled: true }
  cam_left_wrist:  { topic: "/cam_left_wrist/color/image_rect_raw/compressed",  enabled: true }
  cam_right_wrist: { topic: "/cam_right_wrist/color/image_rect_raw/compressed", enabled: true }
```

**Published topics (outputs)**:
```yaml
action_cmd_topics:
  left_arm:    { topic: "/tianji_arm/left/joint_command",  enabled: true }
  right_arm:   { topic: "/tianji_arm/right/joint_command", enabled: true }
  left_hand:   { topic: "/wuji_hand/left/joint_command",   enabled: true }
  right_hand:  { topic: "/wuji_hand/right/joint_command",  enabled: true }
```

### Initial position configuration (optional)

```yaml
initial_arm_positions:
  right_arm: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  left_arm:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## Model training configuration

Training configurations live in `src/openpi/training/config.py`; the main one is `pi05_wuji_multi_54d`.

**Key parameters**:
- **Observation dimension**: 54 (14 from dual arms + 40 from dual hands)
- **Action dimension**: 54
- **Image inputs**: 3 cameras (cam_high, cam_left_wrist, cam_right_wrist)
- **Action chunk**: 50 steps
- **Batch size**: adjust based on GPU memory

**Data format requirements**:
- **Observations**:
  - `state`: (54,) — joint state
  - `cam_high`: (H, W, 3) — head camera
  - `cam_left_wrist`: (H, W, 3) — left wrist camera
  - `cam_right_wrist`: (H, W, 3) — right wrist camera
- **Actions**: (50, 54) — action sequence

## Troubleshooting

### 1. ModuleNotFoundError: No module named 'wuji' / 'openpi_client'

Make sure you launch the client from the project root and inside a ROS2 environment with `openpi-client` installed. The Python used to run the module must be the one that has `openpi-client` and `rclpy` available.

### 2. ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'

The active Python interpreter does not match the one used by your ROS2 installation (ROS2 Humble ships with Python 3.10). Run the client with the Python interpreter from your ROS2 environment, not a separate virtualenv.

### 3. websockets.exceptions.ConnectionClosedOK

The WebSocket connection drops during inference. Check:
1. Server-side logs
2. Model configuration (`pi05_wuji_multi_54d`) matches the data format
3. Observation dimensionality (should be 54)

### 4. Missing ROS2 data

```bash
ros2 topic list
ros2 topic echo /tianji_arm/left/joint_state
ros2 topic hz /stereo/right/compressed
```

### 5. Timestamp synchronization failure

Adjust the synchronization parameters in the configuration file:
```yaml
max_time_diff: 0.05
sync_slop: 0.1
```

## Project structure

```
examples/wuji/
├── config/                    # configuration files
│   └── deploy.yaml           # deployment configuration
├── core/                      # core modules
│   ├── __init__.py
│   ├── config.py             # configuration management (DeployConfig)
│   ├── ros2_interface.py     # ROS2 interface wrapper
│   ├── timestamp_sync.py     # multi-sensor timestamp synchronization
│   └── utils.py              # utility functions (unit conversion, etc.)
├── openpi/                    # OpenPI control pipeline
│   ├── __init__.py
│   ├── main.py               # main entry point
│   └── ros_env.py            # ROS environment wrapper (OpenPIRosEnvironment)
├── resource/                  # resource files
├── package.xml               # ROS2 package manifest
└── README.md                 # this file
```

### Code architecture

```
Runtime (runtime orchestrator)
  │
  ├─> Environment (OpenPIRosEnvironment)
  │     ├─> ROS2Interface
  │     │     ├─> subscribes to joint state topics
  │     │     ├─> subscribes to camera topics
  │     │     └─> publishes action command topics
  │     └─> TimestampSynchronizer
  │           └─> synchronizes multi-sensor data
  │
  ├─> Agent (PolicyAgent)
  │     └─> Policy (ActionChunkBroker)
  │           └─> WebsocketClientPolicy
  │                 └─> connects to the remote policy server
  │
  └─> Subscribers (optional observers)
```

## Development guide

**Adding a new arm type**:
1. Add a new arm type configuration in `config.py`
2. Update the unit conversion logic in `ros2_interface.py`
3. Configure the corresponding topics in `deploy.yaml`

**Adding a new sensor**:
1. Add a new topic under `camera_topics` in `deploy.yaml`
2. `ros_env.py` will automatically handle the new camera input

**Customizing the control frequency**:
```
inference frequency = control_hz / action_horizon
```
- `control_hz=30.0`, `action_horizon=50` → inference frequency = 0.6Hz
- `control_hz=50.0`, `action_horizon=25` → inference frequency = 2.0Hz

## Performance tuning

**Control-frequency tuning**:
- Lower inference frequency: increase `action_horizon` (e.g. 50 → 100)
- Higher responsiveness: decrease `action_horizon` (e.g. 50 → 25)

**Network tuning**:
- Local deployment: run server and client on the same machine
- LAN deployment: use gigabit Ethernet
- Remote deployment: use a tunnel/NAT-traversal solution or VPN

**GPU tuning**:
```bash
python -c "import jax; print(jax.devices())"
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py ...
```

## Frequently Asked Questions (FAQ)

**Q1: How do I verify the system is working?**

Check the following:
1. The policy server starts and listens on the port
2. The client connects to the server (log shows "Connecting to policy server")
3. ROS2 data is being received correctly (log shows "Data ready")
4. The arms begin executing actions

**Q2: How do I switch the trained model?**

Change the arguments in the launch command:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<new config name> \
  --policy.dir=<new checkpoint path>
```

**Q3: Does it support multi-robot control simultaneously?**

Yes. You need to:
1. Configure a distinct topic namespace per robot
2. Update topic names in `deploy.yaml`
3. Launch multiple client instances

**Q4: How do I record runtime data?**
```bash
ros2 bag record -a
# Or record specific topics
ros2 bag record /tianji_arm/left/joint_state /tianji_arm/right/joint_state
```

**Q5: How should the control frequency be chosen?**
- High-precision tasks: 30–50Hz
- Everyday tasks: 10–30Hz
- Demo tasks: 5–10Hz

## Related resources

- [OpenPI official documentation](https://github.com/Physical-Intelligence/openpi)
- [ROS2 Humble documentation](https://docs.ros.org/en/humble/)
- [Training configuration reference](../../src/openpi/training/config.py)

## Contributors

- **HanDuo223**
  - Deployment code development
  - Training configuration authoring
  - 54-dimensional control scheme design

## Changelog

**v1.0.0 (2026-02-06)**
- Initial release with OpenPI integration
- 54-dimensional dual-arm + dual-hand control
- Multi-camera visual input
- Timestamp synchronization system
- Supports Wuji arm bimanual operation

## License

Apache-2.0 License

---

<a id="中文"></a>

# 中文

Wuji 机器人 OpenPI 部署方案——完整的训练到部署工作流。

## 项目概述

本项目提供基于 OpenPI 框架的 Wuji 机器人部署方案。通过 WebSocket 连接策略服务器，结合视觉感知实现双臂协同控制。

### 主要特性

- **OpenPI 控制管线**：通过 WebSocket 连接远程策略服务器，进行实时推理
- **双臂支持**：支持单臂（左/右）或双臂协同操作
- **54 维控制**：双臂（14）+ 双灵巧手（40），实现精确控制
- **多相机输入**：头部相机 + 左右腕部相机（共 3 路视觉流）
- **ROS2 集成**：完整的话题订阅与发布系统
- **灵活配置**：基于 YAML 的配置系统

### 系统架构

```
OpenPI Runtime (调度器)
  ├─> Environment (Wuji ROS2 环境)
  │     ├─> ROS2 话题订阅 (关节状态、相机图像)
  │     └─> ROS2 话题发布 (关节指令)
  ├─> Agent (策略代理)
  │     └─> Broker (serial / RTG)
  │           └─> WebsocketClientPolicy (策略服务器客户端)
  └─> Subscribers (观察者)
```

## 环境要求

- **ROS2**：Humble (Python 3.10)
- **Python**：3.10+ (ROS2 客户端) / 3.11+ (策略服务器)
- **操作系统**：Ubuntu 22.04 / Linux
- **GPU**：推荐使用 NVIDIA GPU (用于模型推理)

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/wuji-technology/wuji-openpi.git
cd wuji-openpi
```

### 2. 安装项目依赖（使用 uv）

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

### 3. 安装 ROS2 Humble

请参考 ROS2 官方文档：https://docs.ros.org/en/humble/Installation.html

安装完成后，请根据自身安装方式 source 对应的 ROS2 环境脚本。

## 使用方法

部署分为两步：**服务端**（策略推理）和**客户端**（机器人控制）。以下命令均假设当前工作目录为项目根目录。

### 第 1 步：启动策略服务器

在**第一个终端**中，启动策略服务器（使用已训练的模型）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=<你的-checkpoint-目录>
```

**参数说明**：
- `--policy.config`：模型配置名称（定义于 `src/openpi/training/config.py`）
- `--policy.dir`：checkpoint 目录路径（包含 `params` 文件）

**可选参数**：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=<你的-checkpoint-目录> \
  --host=0.0.0.0 \
  --port=8000
```

**启动成功示例**：
```
INFO:root:Loading model...
INFO:root:Loaded norm stats from <checkpoint-dir>/assets/placeholder
INFO:root:Creating server (host: your-hostname, ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### 第 2 步：启动机器人控制客户端

此步骤需在已安装 `openpi-client` 的 ROS2 环境中运行。在**第二个终端**（项目根目录下）执行：

```bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

**启动成功示例**：
```
[INFO] [timestamp] [wuji_openpi_node]: Node wuji_openpi_node created
[INFO] Connecting to policy server: localhost:8000
[INFO] Server metadata: {}
[INFO] Data ready (0.0s)
[INFO] Wuji 机器人控制系统配置 (OpenPI):
[INFO]   策略服务器: localhost:8000
[INFO]   控制模式: dual
[INFO]   控制频率: 30.0Hz
[INFO]   动作步长: 50 步
[INFO]   推理频率: 0.60Hz
[INFO] 启动中...
```

### 命令行参数

命令行参数可覆盖配置文件中的设置：

```bash
# 指定服务器地址
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --host 192.168.1.100 --port 12346

# 指定控制模式
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --side left  # 仅左臂
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml --side both  # 双臂

# 其它参数
python3 -m wuji.deploy.main \
  --host localhost \
  --port 8000 \
  --side both \
  --broker-mode rtg \
  --control-hz 30.0 \
  --action-horizon 50 \
  --prompt "tidy up the table"
```

## 配置说明

配置文件位于 `examples/wuji/config/` 目录下，主配置文件为 `examples/wuji/config/deploy.yaml`。

### 核心配置项

```yaml
# 基础配置
id: "wuji_robot"
arm_mode: "dual"  # single_left, single_right, dual
use_ee_pose: false

# 机械臂类型配置
arm_type: "wuji"  # wuji, agilex, custom
arm_dof: 7
hand_dof: 20
control_hz: 30.0

# OpenPI 服务器配置
server_host: "localhost"
server_port: 8000

# 动作块配置
action_horizon: 50        # 推理频率 = control_hz / action_horizon

# Broker 模式
broker_mode: "rtg"        # serial / rtg

# RTG 平滑配置 (broker_mode=rtg)
rtg_trigger_fraction: 0.5
rtg_guidance_steps: 3

# Episode 配置
num_episodes: 1
max_episode_steps: 1000000

# 任务提示
prompt: "tidy up the table"
```

### 两种动作执行模式

- `serial`：原始串行模式，推理完整 chunk 后顺序执行。
- `rtg`：RTG 模式，客户端异步接收新 chunk，按论文 `arXiv:2507.17141` 对新 chunk 前缀做三次样条平滑后拼接执行。

### ROS2 话题配置

**订阅话题（输入）**：
```yaml
joint_state_topics:
  left_arm:    { topic: "/tianji_arm/left/joint_state",  enabled: true }
  right_arm:   { topic: "/tianji_arm/right/joint_state", enabled: true }
  left_hand:   { topic: "/wuji_hand/left/joint_state",   enabled: true }
  right_hand:  { topic: "/wuji_hand/right/joint_state",  enabled: true }

camera_topics:
  cam_high:        { topic: "/stereo/right/compressed",                       enabled: true }
  cam_left_wrist:  { topic: "/cam_left_wrist/color/image_rect_raw/compressed",  enabled: true }
  cam_right_wrist: { topic: "/cam_right_wrist/color/image_rect_raw/compressed", enabled: true }
```

**发布话题（输出）**：
```yaml
action_cmd_topics:
  left_arm:    { topic: "/tianji_arm/left/joint_command",  enabled: true }
  right_arm:   { topic: "/tianji_arm/right/joint_command", enabled: true }
  left_hand:   { topic: "/wuji_hand/left/joint_command",   enabled: true }
  right_hand:  { topic: "/wuji_hand/right/joint_command",  enabled: true }
```

### 初始位置配置（可选）

```yaml
initial_arm_positions:
  right_arm: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  left_arm:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## 模型训练配置

训练配置位于 `src/openpi/training/config.py`，主配置为 `pi05_wuji_multi_54d`。

**关键参数**：
- **观测维度**：54（双臂 14 + 双手 40）
- **动作维度**：54
- **图像输入**：3 个相机（cam_high, cam_left_wrist, cam_right_wrist）
- **动作块**：50 步
- **Batch size**：根据 GPU 显存调整

**数据格式要求**：
- **观测**：
  - `state`：(54,) — 关节状态
  - `cam_high`：(H, W, 3) — 头部相机
  - `cam_left_wrist`：(H, W, 3) — 左腕相机
  - `cam_right_wrist`：(H, W, 3) — 右腕相机
- **动作**：(50, 54) — 动作序列

## 常见问题排查

### 1. ModuleNotFoundError: No module named 'wuji' / 'openpi_client'

请确认在项目根目录下、并且已 source 含有 `openpi-client` 的 ROS2 环境中启动客户端。运行模块所用的 Python 解释器需同时具备 `openpi-client` 与 `rclpy`。

### 2. ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'

当前 Python 解释器与 ROS2 安装所用的 Python 不一致（ROS2 Humble 自带 Python 3.10）。请使用 ROS2 环境对应的 Python 解释器运行客户端，而不是独立的虚拟环境。

### 3. websockets.exceptions.ConnectionClosedOK

推理过程中 WebSocket 连接中断。检查：
1. 服务端日志
2. 模型配置 (`pi05_wuji_multi_54d`) 与数据格式一致
3. 观测维度（应为 54）

### 4. ROS2 数据缺失

```bash
ros2 topic list
ros2 topic echo /tianji_arm/left/joint_state
ros2 topic hz /stereo/right/compressed
```

### 5. 时间戳同步失败

调整配置文件中的同步参数：
```yaml
max_time_diff: 0.05
sync_slop: 0.1
```

## 项目结构

```
examples/wuji/
├── config/                    # 配置文件
│   └── deploy.yaml           # 部署配置
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── config.py             # 配置管理 (DeployConfig)
│   ├── ros2_interface.py     # ROS2 接口封装
│   ├── timestamp_sync.py     # 多传感器时间戳同步
│   └── utils.py              # 工具函数（单位转换等）
├── openpi/                    # OpenPI 控制管线
│   ├── __init__.py
│   ├── main.py               # 主入口
│   └── ros_env.py            # ROS 环境封装 (OpenPIRosEnvironment)
├── resource/                  # 资源文件
├── package.xml               # ROS2 包清单
└── README.md                 # 本文件
```

### 代码架构

```
Runtime (运行时调度器)
  │
  ├─> Environment (OpenPIRosEnvironment)
  │     ├─> ROS2Interface
  │     │     ├─> 订阅关节状态话题
  │     │     ├─> 订阅相机话题
  │     │     └─> 发布动作指令话题
  │     └─> TimestampSynchronizer
  │           └─> 多传感器数据同步
  │
  ├─> Agent (PolicyAgent)
  │     └─> Policy (ActionChunkBroker)
  │           └─> WebsocketClientPolicy
  │                 └─> 连接远程策略服务器
  │
  └─> Subscribers (可选观察者)
```

## 开发指南

**添加新机械臂类型**：
1. 在 `config.py` 中添加新的机械臂类型配置
2. 更新 `ros2_interface.py` 中的单位转换逻辑
3. 在 `deploy.yaml` 中配置对应话题

**添加新传感器**：
1. 在 `deploy.yaml` 的 `camera_topics` 下添加新话题
2. `ros_env.py` 会自动处理新的相机输入

**自定义控制频率**：
```
推理频率 = control_hz / action_horizon
```
- `control_hz=30.0`, `action_horizon=50` → 推理频率 = 0.6Hz
- `control_hz=50.0`, `action_horizon=25` → 推理频率 = 2.0Hz

## 性能调优

**控制频率调优**：
- 降低推理频率：增大 `action_horizon`（如 50 → 100）
- 提高响应性：减小 `action_horizon`（如 50 → 25）

**网络调优**：
- 本地部署：服务端与客户端运行在同一机器上
- 局域网部署：使用千兆以太网
- 远程部署：使用隧道/NAT 穿透或 VPN

**GPU 调优**：
```bash
python -c "import jax; print(jax.devices())"
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py ...
```

## 常见问题（FAQ）

**Q1：如何验证系统正常工作？**

检查以下几点：
1. 策略服务器成功启动并监听端口
2. 客户端成功连接服务器（日志显示 "Connecting to policy server"）
3. ROS2 数据正常接收（日志显示 "Data ready"）
4. 机械臂开始执行动作

**Q2：如何切换训练好的模型？**

修改启动命令的参数：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<新配置名> \
  --policy.dir=<新 checkpoint 路径>
```

**Q3：是否支持同时控制多台机器人？**

支持。需要：
1. 为每台机器人配置独立的话题命名空间
2. 在 `deploy.yaml` 中更新话题名
3. 启动多个客户端实例

**Q4：如何记录运行时数据？**
```bash
ros2 bag record -a
# 或录制指定话题
ros2 bag record /tianji_arm/left/joint_state /tianji_arm/right/joint_state
```

**Q5：控制频率如何选择？**
- 高精度任务：30–50Hz
- 常规任务：10–30Hz
- 演示任务：5–10Hz

## 相关资源

- [OpenPI 官方文档](https://github.com/Physical-Intelligence/openpi)
- [ROS2 Humble 文档](https://docs.ros.org/en/humble/)
- [训练配置参考](../../src/openpi/training/config.py)

## 贡献者

- **HanDuo223**
  - 部署代码开发
  - 训练配置编写
  - 54 维控制方案设计

## 更新日志

**v1.0.0 (2026-02-06)**
- 初始版本发布，完成 OpenPI 集成
- 54 维双臂 + 双手控制
- 多相机视觉输入
- 时间戳同步系统
- 支持 Wuji 机械臂双臂操作

## 许可证

Apache-2.0 License

---

**Maintainer / 维护者**：wuji
**Last updated / 最后更新**：2026-05-18
