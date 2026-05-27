# wuji-openpi

[English](README.md) | **中文**

本仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi) 修改，专门用于针对 **wuji 双臂机器人** 进行 SFT（监督微调）训练与部署。在原版 openpi 的基础上，新增了 wuji 双臂机器人数据处理、训练配置以及 ROS2 部署流程。

> 上游模型说明请参阅 [`docs/`](docs/) 以及 openpi 官方仓库；本 README 仅聚焦于 wuji 机器人相关的使用方法。

## 1. 仓库定位

- **目的**：在 wuji 双臂机器人上完成 pi0 / pi0.5 等 VLA 模型的 SFT 与部署。
- **数据来源**：通过 [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) ���库进行遥操作采集，得到 ROS2 mcap 格式的原始数据。
- **训练框架**：基于 openpi 的 JAX 训练管线，新增了 wuji 专用的数据策略、配置和 norm stats 计算脚本。
- **部署形态**：通过 ROS2 与 wuji-hand-teleop 仓库通信，交换 observation 与 action 消息，实现实机推理。

## 2. 端到端工作流

```text
┌────────────────────┐    ROS2 mcap     ┌──────────────────┐    LeRobot v2.1    ┌──────────────┐
│  wuji-hand-teleop  │ ───────────────▶ │  数据转换脚本     │ ─────────────────▶ │   SFT 训练   │
│  （遥操作采集）     │                  │  (mcap → lerobot)│                    │  (本仓库)    │
└────────────────────┘                  └──────────────────┘                    └──────┬───────┘
                                                                                       │
                                                                                       ▼
                                                            ┌──────────────────────────────┐
                                                            │  Policy Server (本仓库)       │
                                                            │  WebSocket / ROS2 推理服务    │
                                                            └──────────────┬───────────────┘
                                                                           │ ROS2 obs/action
                                                                           ▼
                                                            ┌──────────────────────────────┐
                                                            │  wuji 双臂机器人 (实机执行)   │
                                                            │  wuji-hand-teleop 提供 ROS2 桥│
                                                            └──────────────────────────────┘
```

### Step 1：采集遥操作数据

使用 [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) 仓库执行遥操作，采集双臂 + 双灵巧手的同步轨迹，以及头部相机、左右腕部相机视频，保存为 **ROS2 mcap** bag。

### Step 2：转换为 LeRobot v2.1 数据集

将采集到的 ROS2 mcap 数据集转换为 LeRobot v2.1 格式，输出到训练数据目录，供后续训练使用。LeRobot v2.1 数据集字段需与训练配置中的 `WujiInputs` 对齐（详见 [`src/openpi/policies/wuji_policy.py`](src/openpi/policies/wuji_policy.py)）。

转换后期望的字段：

- `observation.state`：54 维（双臂 14 + 双灵巧手 40）
- `observation.images.cam_high`：头部相机
- `observation.images.cam_left_wrist`：左腕相机
- `observation.images.cam_right_wrist`：右腕相机
- `action`：54 维动作序列

### Step 3：修改训练配置

训练配置位于 [`src/openpi/training/config.py`](src/openpi/training/config.py)，本仓库提供了 wuji 双臂机器人示例配置 **`pi05_wuji_multi_54d`**。在开始训练前，需根据自己的数据修改以下字段：

- `data.repo_id`：指向 Step 2 生成的 LeRobot 数据集路径（可在 `LeRobotDataConfig` 中配置多个数据集进行混合训练）。
- `checkpoint_base_dir`：训练 checkpoint 输出目录。
- `weight_loader` 中的 base checkpoint 路径（如 `pi05_base/params`）。
- batch size、训练步数、学习率等超参数。

参考片段（节选自 `pi05_wuji_multi_54d`）：

```python
TrainConfig(
    name="pi05_wuji_multi_54d",
    checkpoint_base_dir="/path/to/your/checkpoints",
    data=LeRobotWujiDataConfig(
        repo_ids=[
            "/path/to/lerobot_dataset_1",
            "/path/to/lerobot_dataset_2",
            # ...
        ],
        ...
    ),
    weight_loader=CheckpointWeightLoader(
        "/path/to/pi05_base/params"
    ),
)
```

### Step 4：计算 norm stats 并训练

```bash
# 计算归一化统计量（仅首次或数据更新时需要）
uv run scripts/compute_norm_stats.py --config-name pi05_wuji_multi_54d

# 启动训练（如需覆盖已有实验，加上 --overwrite）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_wuji_multi_54d \
  --exp-name=my_wuji_run
```

训练日志会写入控制台并同步到 Weights & Biases，checkpoint 默认保存到 `checkpoint_base_dir` 下。

### Step 5：实机推理与部署

本仓库提供完整的 wuji 双臂机器人部署流程：通过 **policy server + ROS2 客户端** 的架构与 wuji-hand-teleop 仓库交换 observation/action。

启动 policy server（机器学习端）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=/path/to/checkpoint
```

启动机器人控制客户端（ROS2 端，与 wuji-hand-teleop 通信）。请在项目根目录下、并且已 source 含有 `openpi-client` 与 `rclpy` 的 ROS2 环境中执行：

```bash
source /opt/ros/humble/setup.bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

客户端会订阅 wuji-hand-teleop 发布的 ROS2 关节状态和图像 topic，将拼接后的 observation 通过 WebSocket 发送给 policy server，获取 action chunk 后再以 ROS2 命令的形式发布回机器人执行。

> 更完整的部署细节（YAML 配置、broker 模式、topic 列表、故障排查）请参阅 [`examples/wuji/README.md`](examples/wuji/README.md)。

## 3. 目录结构（wuji 相关部分）

```
wuji-openpi/
├── examples/wuji/                    # wuji 双臂机器人部署示例
│   ├── config/deploy.yaml           # 部署配置（ROS2 topic、broker、控制频率）
│   ├── core/                        # ROS2 接口、时间戳同步、工具函数
│   ├── deploy/                      # 部署入口（main.py / ros_env.py）
│   └── README.md                    # 部署详细说明
├── src/openpi/
│   ├── policies/wuji_policy.py      # WujiInputs / WujiOutputs 数据映射
│   └── training/config.py           # pi05_wuji_multi_54d 等训练配置
├── scripts/
│   ├── compute_norm_stats.py        # norm stats 计算
│   ├── train.py                     # 训练入口
│   └── serve_policy.py              # Policy server 入口
└── docs/                            # openpi 上游文档
```

## 4. 环境准备

依赖与原版 openpi 保持一致，使用 [uv](https://docs.astral.sh/uv/) 管理：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

部署端额外需要：

- ROS2 Humble（Python 3.10）
- 与 wuji-hand-teleop 一致的 ROS2 topic 命名空间

## 5. 与上游 openpi 的差异

本仓库保留了 openpi 原有的训练/推理能力（pi0、pi0-FAST、pi0.5），并在此基础上新增/修改了以下五大模块以适配 wuji 双臂机器人。

### 5.1 Wuji 双臂双灵巧手训练 + 部署

- **54 维 action 空间**：双臂 7+7 + 双手 20+20，覆盖手臂关节与灵巧手所有自由度。
- **数据配置**：`LeRobotWujiDataConfig` —— 原生 54 维数据管线。
- **训练注册项**（在 [`src/openpi/training/config.py`](src/openpi/training/config.py)）：`pi05_wuji_multi_54d` 直出 54 维 action。
- **完整 ROS2 部署包** [`examples/wuji/`](examples/wuji/)：topic 订阅/发布、时间戳同步、broker 接入，开箱即用对接 wuji-hand-teleop。

### 5.2 多数据集训练

- 新增 `ConcatLeRobotDataset` 与 `MultiLeRobotDataset`（带加权采样），便于将多次采集的数据合并训练。
- `DataConfig` 新增字段：
  - `lerobot_datasets`：多数据集列表。
  - `multi_dataset_mode`：拼接 / 加权采样切换。
- 支持 **per-dataset prompt transform**：每个子数据集可绑定独立的语言指令。

### 5.3 RTG（Real-Time Trajectory Generation）

针对 chunk 切换时的边界跳变，提供轨迹拼接平滑能力：

- `RTGActionBroker` 及其 QP 变体。
- 平滑工具：
  - `qp_smooth_prefix`
  - `cubic_smooth_prefix`
  - `build_time_window_old_reference`

参考论文 `arXiv:2507.17141`，在客户端异步收到新 chunk 后对其前缀做平滑再拼接执行。

### 5.4 通用工具改进

- **`PartialCheckpointWeightLoader`**：加载 base checkpoint 时，遇到形状不匹配的层（典型如 `action_proj`）自动跳过，方便不同 action 维度之间的迁移。
- **[`examples/open_loop_eval.py`](examples/open_loop_eval.py)**：通用 open-loop 评估器，内置 RTG 对比，方便对训练好的 checkpoint 做轨迹复现 & 平滑算法对照。

> 其余通用功能（如 LIBERO/ALOHA/DROID 示例、PyTorch 后端等）保持与上游一致，请参考原 openpi 文档与 `examples/` 下对应子目录。

## 6. 相关链接

- 数据采集仓库：<https://github.com/wuji-technology/wuji-hand-teleop.git>
- 上游 openpi：<https://github.com/Physical-Intelligence/openpi>
- wuji 部署说明：[`examples/wuji/README.md`](examples/wuji/README.md)
- 训练配置入口：[`src/openpi/training/config.py`](src/openpi/training/config.py)

## License

Apache-2.0（继承自上游 openpi）。
